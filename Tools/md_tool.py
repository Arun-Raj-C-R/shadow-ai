# md_tool.py
"""
Molecular Dynamics Tool for SHADOW/Shadow AI
=============================================
Performs MD simulations and analysis using:
  - ASE (Atomic Simulation Environment) for MD engine
  - Materials Project API for real crystal structures
  - Pymatgen for structure manipulation
  - Wolfram Alpha for thermodynamic depth calculations
  - Web search for supplementary data

Runs as a BACKGROUND tool â€” results print to terminal when done.
"""

import os
import json
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
config_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", ".env")
if os.path.exists(config_env):
    load_dotenv(config_env, override=True)
else:
    load_dotenv()

MP_API_KEY = os.environ.get("MP_API_KEY", "")
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")

MD_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "md_runs")
os.makedirs(MD_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("md_tool")


# ======================================================================
# STRUCTURE FETCHER (from Materials Project)
# ======================================================================

# Common aliases for complex/organic formulas that aren't in MP directly
FORMULA_ALIASES = {
    "FAPbI3": "CsPbI3",    # FA perovskite â†’ closest inorganic analog
    "MAPbI3": "CsPbI3",    # MA perovskite â†’ closest inorganic analog
    "FAPbBr3": "CsPbBr3",
    "MAPbBr3": "CsPbBr3",
    "FAPbCl3": "CsPbCl3",
    "MAPbCl3": "CsPbCl3",
    "FAPI": "CsPbI3",
    "MAPI": "CsPbI3",
}


def _resolve_formula(formula: str) -> tuple:
    """Resolve complex/organic formulas to searchable MP formulas.
    Returns (resolved_formula, note) tuple."""
    if not formula:
        return formula, None

    upper = formula.strip()

    # Check direct aliases
    for alias, resolved in FORMULA_ALIASES.items():
        if upper.upper() == alias.upper():
            return resolved, f"'{formula}' is an organic-inorganic compound. Using inorganic analog '{resolved}' from Materials Project."

    # ABX3-type generic perovskite notation
    if upper.upper() == "ABX3":
        return "CsPbI3", f"'ABX3' is a perovskite family notation. Using CsPbI3 as representative perovskite."

    return formula, None


def _ensure_structure(obj):
    """Convert dict/whatever to a real pymatgen Structure if needed."""
    from pymatgen.core import Structure
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        return Structure.from_dict(obj)
    if hasattr(obj, 'as_dict'):
        return Structure.from_dict(obj.as_dict())
    raise TypeError(f"Cannot convert {type(obj)} to Structure")


def _get_ase_atoms(formula: str = None, material_id: str = None, cif_file: str = None):
    """Fetch structure from MP or load from CIF, and convert to ASE Atoms."""
    try:
        from mp_api.client import MPRester
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        
        if cif_file:
            if not os.path.exists(cif_file): return None, f"CIF file not found: {cif_file}"
            struct = Structure.from_file(cif_file)
            atoms = AseAtomsAdaptor.get_atoms(struct)
            return atoms, os.path.basename(cif_file)
            
        from mp_api.client import MPRester
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        return None, "Install mp-api and pymatgen: pip install mp-api pymatgen"

    if not MP_API_KEY:
        return None, "MP_API_KEY not set in .env"

    note = None
    if formula and not material_id:
        formula, note = _resolve_formula(formula)

    try:
        with MPRester(MP_API_KEY) as mpr:
            if formula and not material_id:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "energy_above_hull"],
                    num_chunks=1
                )
                if not docs:
                    return None, f"No materials found for '{formula}'" + (f" ({note})" if note else "")
                docs_sorted = sorted(docs, key=lambda d: d.energy_above_hull or 999)
                material_id = str(docs_sorted[0].material_id)

            raw_struct = mpr.get_structure_by_material_id(material_id)
            struct = _ensure_structure(raw_struct)

            # Handle disordered structures (partial occupancy)
            if not struct.is_ordered:
                from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
                try:
                    trans = OrderDisorderedStructureTransformation()
                    struct = trans.apply_transformation(struct)
                except Exception:
                    # Fallback: just use as-is, ASE might still handle it
                    pass

            atoms = AseAtomsAdaptor.get_atoms(struct)
            result_id = material_id
            if note:
                result_id = f"{material_id} ({note})"
            return atoms, result_id
    except Exception as e:
        return None, str(e)


# ======================================================================
# WOLFRAM + WEB SEARCH (reuse from dft_tool pattern)
# ======================================================================

def _wolfram_calc(query: str) -> str:
    if not WOLFRAM_APP_ID:
        return "Wolfram Alpha not available"
    try:
        from wolfram_orchestrator_tool import simple_wolfram_query
        return simple_wolfram_query(query)
    except Exception as e:
        return f"Wolfram error: {e}"


def _web_search(query: str) -> str:
    try:
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "No API key for web search"
        client = genai.Client(api_key=api_key)
        tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[tool])
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Molecular dynamics simulation: {query}",
            config=config
        )
        return resp.text if resp.text else "No results"
    except Exception as e:
        return f"Search error: {e}"


# ======================================================================
# MD SIMULATION ENGINE (ASE-based)
# ======================================================================

def _run_md_simulation(atoms, ensemble: str = "nvt", temperature_K: float = 300,
                       timestep_fs: float = 1.0, steps: int = 500,
                       job_dir: str = None) -> Dict:
    """Run MD simulation using ASE with a universal force field."""
    try:
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    except ImportError:
        return {"error": "ASE not installed. Run: pip install ase"}

    results = {"ensemble": ensemble, "temperature_K": temperature_K,
               "timestep_fs": timestep_fs, "steps": steps}

    try:
        # Try ML potential first (MACE), fall back to LJ
        calculator = None
        calc_name = "none"

        # Try MACE (state-of-the-art ML potential)
        try:
            from mace.calculators import mace_mp
            calculator = mace_mp(model="medium", default_dtype="float32")
            calc_name = "MACE-MP-0 (medium)"
        except Exception:
            pass

        # Try EMT (good for metals: Cu, Ag, Au, Ni, Pd, Pt, Al)
        if calculator is None:
            try:
                from ase.calculators.emt import EMT
                supported_emt = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"}
                if set(atoms.get_chemical_symbols()).issubset(supported_emt):
                    calculator = EMT()
                    calc_name = "EMT (Effective Medium Theory)"
            except Exception:
                pass

        # Fallback: Lennard-Jones
        if calculator is None:
            try:
                from ase.calculators.lj import LennardJones
                calculator = LennardJones()
                calc_name = "Lennard-Jones"
            except Exception:
                return {"error": "No ASE calculator available"}

        atoms.calc = calculator
        results["calculator"] = calc_name

        # Initial energy
        try:
            e_init = atoms.get_potential_energy()
            results["initial_energy_eV"] = round(float(e_init), 6)
        except Exception as e:
            results["initial_energy_error"] = str(e)

        # Set initial velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

        # Setup MD integrator
        ensemble_lower = ensemble.lower()
        if ensemble_lower == "nve":
            from ase.md.verlet import VelocityVerlet
            dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
        elif ensemble_lower == "nvt":
            from ase.md.langevin import Langevin
            dyn = Langevin(atoms, timestep=timestep_fs * units.fs,
                           temperature_K=temperature_K, friction=0.01)
        elif ensemble_lower == "npt":
            try:
                from ase.md.npt import NPT as NPTIntegrator
                dyn = NPTIntegrator(atoms, timestep=timestep_fs * units.fs,
                                    temperature_K=temperature_K,
                                    externalstress=0.0, ttime=25 * units.fs,
                                    pfactor=None)
            except Exception:
                from ase.md.langevin import Langevin
                dyn = Langevin(atoms, timestep=timestep_fs * units.fs,
                               temperature_K=temperature_K, friction=0.01)
                results["note"] = "NPT unavailable, fell back to NVT (Langevin)"
        else:
            return {"error": f"Unknown ensemble: {ensemble}. Use nve, nvt, or npt"}

        # Trajectory recording
        trajectory = []
        energies = []
        temperatures = []
        trajectory_frames = []

        def _record():
            ke = atoms.get_kinetic_energy()
            try:
                pe = atoms.get_potential_energy()
            except Exception:
                pe = 0.0
            t = 2 * ke / (3 * len(atoms) * units.kB)
            energies.append(float(pe + ke))
            temperatures.append(float(t))
            trajectory.append({
                "step": len(energies),
                "PE_eV": round(float(pe), 6),
                "KE_eV": round(float(ke), 6),
                "T_K": round(float(t), 2),
            })
            trajectory_frames.append(atoms.copy())

        dyn.attach(_record, interval=max(1, steps // 50))

        # Run simulation
        dyn.run(steps)

        # Final state
        try:
            e_final = atoms.get_potential_energy()
            results["final_energy_eV"] = round(float(e_final), 6)
        except Exception:
            pass

        results["trajectory_samples"] = trajectory[:25]  # First 25 samples
        results["avg_temperature_K"] = round(float(np.mean(temperatures)), 2) if temperatures else None
        results["avg_total_energy_eV"] = round(float(np.mean(energies)), 6) if energies else None
        results["energy_drift_eV"] = round(float(energies[-1] - energies[0]), 6) if len(energies) > 1 else None
        results["total_steps_completed"] = steps
        results["status"] = "success"

        # Save trajectory data
        if job_dir:
            from ase.io import write
            import webbrowser
            
            traj_path = os.path.join(job_dir, "trajectory.json")
            with open(traj_path, "w") as f:
                json.dump(trajectory, f, indent=2)
                
            xyz_path = os.path.join(job_dir, "trajectory.xyz")
            write(xyz_path, trajectory_frames, format="extxyz")
            
            html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MD Animation Viewer</title>
<style>body {{ margin:0; overflow:hidden; background:#0a0a1a; color:#fff; font-family:sans-serif; }}
#header {{ position:fixed; top:10px; left:20px; z-index:100; }}</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="header"><h2>Molecular Dynamics Animation</h2><p>Ensemble: {ensemble.upper()} | Temp: {temperature_K}K</p><p>Scroll to zoom, drag to rotate.</p></div>
<div id="viewer" style="width: 100vw; height: 100vh;"></div>
<script>
  let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x0a0a1a"}});
  fetch("file:///{xyz_path.replace(os.sep, '/')}").then(r => r.text()).catch(e => `{open(xyz_path).read()}`).then(data => {{
      viewer.addModelsAsFrames(data, "xyz");
      viewer.setStyle({{}}, {{sphere: {{scale: 0.3}}, stick: {{radius: 0.15}}}});
      viewer.animate({{loop: "forward", step: 1, reps: 0}});
      viewer.zoomTo();
      viewer.render();
      
      // Dynamic Element Legend
      setTimeout(function() {{
          var legend = document.createElement("div");
          legend.style.position = "fixed";
          legend.style.right = "20px";
          legend.style.bottom = "20px";
          legend.style.background = "rgba(20,20,40,0.85)";
          legend.style.padding = "10px 15px";
          legend.style.borderRadius = "10px";
          legend.style.border = "1px solid rgba(100,100,255,0.2)";
          legend.style.zIndex = "100";
          
          var atoms = viewer.getModel().selectedAtoms({{}});
          var uniqueElements = [];
          atoms.forEach(function(a) {{ if(uniqueElements.indexOf(a.elem) === -1) uniqueElements.push(a.elem); }});
          uniqueElements.sort();
          
          var html = "<h4 style='margin:0 0 10px 0; color:#fff; border-bottom:1px solid #555; padding-bottom:5px; font-weight:normal;'>Elements</h4>";
          uniqueElements.forEach(function(elem) {{
              var color = $3Dmol.ElementColors[elem];
              var hex = color ? "#" + color.toString(16).padStart(6, '0') : "#ffffff";
              html += `<div style="display:flex; align-items:center; margin-bottom:6px;">
                         <div style="width:14px; height:14px; border-radius:50%; background:${{hex}}; margin-right:10px; border:1px solid #777;"></div>
                         <span style="font-size:14px; color:#e0e0e0; font-family:sans-serif;">${{elem}}</span>
                       </div>`;
          }});
          legend.innerHTML = html;
          document.body.appendChild(legend);
      }}, 500);
  }});
</script>
</body></html>'''
            
            html_path = os.path.join(job_dir, "animation.html")
            with open(html_path, "w") as f:
                f.write(html)
            
            webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
            
            results["trajectory_file"] = traj_path
            results["animation_file"] = html_path

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def _energy_minimize(atoms, fmax: float = 0.05, max_steps: int = 200) -> Dict:
    """Geometry optimization using ASE."""
    try:
        from ase.optimize import BFGS
    except ImportError:
        return {"error": "ASE not installed"}

    results = {}
    try:
        # Try calculators in order
        calculator = None
        calc_name = "none"
        try:
            from mace.calculators import mace_mp
            calculator = mace_mp(model="medium", default_dtype="float32")
            calc_name = "MACE-MP-0"
        except Exception:
            pass
        if calculator is None:
            try:
                from ase.calculators.emt import EMT
                calculator = EMT()
                calc_name = "EMT"
            except Exception:
                from ase.calculators.lj import LennardJones
                calculator = LennardJones()
                calc_name = "Lennard-Jones"

        atoms.calc = calculator
        results["calculator"] = calc_name

        e_before = float(atoms.get_potential_energy())
        results["energy_before_eV"] = round(e_before, 6)

        opt = BFGS(atoms)
        opt.run(fmax=fmax, steps=max_steps)

        e_after = float(atoms.get_potential_energy())
        results["energy_after_eV"] = round(e_after, 6)
        results["energy_change_eV"] = round(e_after - e_before, 6)
        results["converged"] = opt.converged()
        results["n_steps"] = opt.nsteps
        results["final_forces_max_eV_A"] = round(float(atoms.get_forces().max()), 6)

        results["relaxed_cell"] = {
            "a": round(float(atoms.cell.lengths()[0]), 4),
            "b": round(float(atoms.cell.lengths()[1]), 4),
            "c": round(float(atoms.cell.lengths()[2]), 4),
        }
        results["status"] = "success"
    except Exception as e:
        results["error"] = str(e)

    return results


def _compute_rdf(atoms, rmax: float = 8.0, nbins: int = 100) -> Dict:
    """Compute radial distribution function."""
    results = {}
    try:
        from ase.geometry.analysis import Analysis
        ana = Analysis(atoms)

        # Get all unique element pairs
        symbols = list(set(atoms.get_chemical_symbols()))
        rdf_data = {}

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i:]:
                try:
                    rdf_result = ana.get_rdf(rmax=rmax, nbins=nbins,
                                             elements=(s1, s2))
                    if rdf_result is not None:
                        r_vals = np.linspace(0, rmax, nbins)
                        rdf_vals = rdf_result[0] if isinstance(rdf_result, list) else rdf_result
                        # Find first peak
                        peak_idx = np.argmax(rdf_vals[5:]) + 5 if len(rdf_vals) > 5 else 0
                        rdf_data[f"{s1}-{s2}"] = {
                            "first_peak_distance_A": round(float(r_vals[peak_idx]), 3),
                            "first_peak_height": round(float(rdf_vals[peak_idx]), 3),
                        }
                except Exception:
                    continue

        results["rdf"] = rdf_data if rdf_data else {"note": "RDF computation not available for this structure"}
        results["status"] = "success"
    except Exception as e:
        results["error"] = str(e)
        # Fallback: manual neighbor analysis
        try:
            from ase.neighborlist import neighbor_list
            i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=rmax)
            if len(d_list) > 0:
                results["neighbor_stats"] = {
                    "num_pairs": len(d_list),
                    "min_distance_A": round(float(d_list.min()), 4),
                    "max_distance_A": round(float(d_list.max()), 4),
                    "mean_distance_A": round(float(d_list.mean()), 4),
                }
                results["status"] = "partial"
        except Exception:
            pass

    return results


def _thermal_expansion(atoms, temp_range: list = None) -> Dict:
    """Estimate thermal properties at different temperatures."""
    if temp_range is None:
        temp_range = [100, 200, 300, 400, 500, 600]

    results = {"temperatures_K": temp_range, "volumes": [], "energies": []}
    try:
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        # Try to get a calculator
        try:
            from ase.calculators.emt import EMT
            supported_emt = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"}
            if set(atoms.get_chemical_symbols()).issubset(supported_emt):
                calc = EMT()
            else:
                raise ValueError("Unsupported EMT elements")
        except Exception:
            from ase.calculators.lj import LennardJones
            calc = LennardJones()

        for T in temp_range:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            MaxwellBoltzmannDistribution(atoms_copy, temperature_K=T)
            dyn = Langevin(atoms_copy, timestep=1.0 * units.fs,
                           temperature_K=T, friction=0.01)
            dyn.run(100)  # Short equilibration

            try:
                vol = float(atoms_copy.get_volume())
                eng = float(atoms_copy.get_potential_energy())
                results["volumes"].append(round(vol, 4))
                results["energies"].append(round(eng, 6))
            except Exception:
                results["volumes"].append(None)
                results["energies"].append(None)

        # Estimate thermal expansion coefficient
        valid = [(T, V) for T, V in zip(temp_range, results["volumes"]) if V is not None]
        if len(valid) >= 2:
            temps = [v[0] for v in valid]
            vols = [v[1] for v in valid]
            dV_dT = (vols[-1] - vols[0]) / (temps[-1] - temps[0])
            alpha = dV_dT / vols[0]
            results["thermal_expansion_coeff_per_K"] = f"{alpha:.2e}"

        results["status"] = "success"
    except Exception as e:
        results["error"] = str(e)

    return results


# ======================================================================
# MAIN MD ORCHESTRATOR
# ======================================================================

def run_md_simulation(
    task: str,
    formula: str = None,
    material_id: str = None,
    cif_file: str = None,
    ensemble: str = "nvt",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    steps: int = 500,
    fmax: float = 0.05,
    temp_range: list = None,
    wolfram_query: str = None,
    search_query: str = None,
) -> str:
    """
    Main MD tool entry point. Called by Shadow AI.

    Tasks:
        - "run_md"           : Run MD simulation (NVE/NVT/NPT)
        - "energy_minimize"  : Geometry optimization
        - "rdf"              : Radial distribution function
        - "thermal"          : Thermal expansion analysis
        - "full_analysis"    : MD + RDF + thermal + Wolfram + web
        - "compare"          : Compare MD for multiple materials
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"âš›ï¸  MOLECULAR DYNAMICS ENGINE")
    print(f"   Task: {task}")
    print(f"   Material: {cif_file or formula or material_id}")
    print(f"   Time: {start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    output = {
        "task": task, "formula": formula, "material_id": material_id, "cif_file": cif_file,
        "timestamp": start_time.isoformat(),
    }

    try:
        # Create job directory
        job_id = f"md_{int(start_time.timestamp())}"
        job_dir = os.path.join(MD_OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        output["job_dir"] = job_dir

        # Get atoms (except for compare)
        atoms = None
        if task != "compare":
            print("ðŸ” Fetching/Loading structure...")
            atoms, mid_or_err = _get_ase_atoms(formula=formula, material_id=material_id, cif_file=cif_file)
            if atoms is None:
                output["error"] = f"Could not get structure: {mid_or_err}"
                return _finalize(output, start_time)
            output["material_id"] = mid_or_err
            output["structure"] = {
                "formula": atoms.get_chemical_formula(),
                "n_atoms": len(atoms),
                "cell_A": [round(float(x), 4) for x in atoms.cell.lengths()],
            }

        # â”€â”€ TASK ROUTING â”€â”€
        if task == "run_md":
            print(f"ðŸƒ Running {ensemble.upper()} MD @ {temperature_K}K for {steps} steps...")
            output["md_results"] = _run_md_simulation(
                atoms, ensemble=ensemble, temperature_K=temperature_K,
                timestep_fs=timestep_fs, steps=steps, job_dir=job_dir
            )

        elif task == "energy_minimize":
            print("âš¡ Running energy minimization...")
            output["optimization"] = _energy_minimize(atoms, fmax=fmax)

        elif task == "rdf":
            print("ðŸ“Š Computing radial distribution function...")
            output["rdf_analysis"] = _compute_rdf(atoms)

        elif task == "thermal":
            print("ðŸŒ¡ï¸ Running thermal expansion analysis...")
            output["thermal"] = _thermal_expansion(atoms, temp_range=temp_range)

        elif task == "full_analysis":
            print("ðŸ“Š Step 1/5: Energy minimization...")
            output["optimization"] = _energy_minimize(atoms, fmax=fmax)

            print(f"ðŸƒ Step 2/5: MD simulation @ {temperature_K}K...")
            output["md_results"] = _run_md_simulation(
                atoms.copy(), ensemble=ensemble, temperature_K=temperature_K,
                timestep_fs=timestep_fs, steps=steps, job_dir=job_dir
            )

            print("ðŸ“Š Step 3/5: RDF analysis...")
            output["rdf_analysis"] = _compute_rdf(atoms)

            print("ðŸ§® Step 4/5: Wolfram Alpha calculations...")
            mat_name = formula or material_id
            wolfram_data = {}
            for prop in ["melting point", "thermal conductivity", "specific heat"]:
                wolfram_data[prop] = _wolfram_calc(f"{prop} of {mat_name}")
            output["wolfram_data"] = wolfram_data

            print("ðŸŒ Step 5/5: Web search...")
            output["web_data"] = _web_search(
                f"{mat_name} molecular dynamics simulation thermal properties"
            )

        elif task == "compare":
            print("ðŸ”„ Comparing materials...")
            formulas = [f.strip() for f in (formula or "").split(",")]
            comparisons = []
            for f in formulas:
                a, mid = _get_ase_atoms(formula=f)
                if a is not None:
                    result = _energy_minimize(a, fmax=fmax)
                    result["formula"] = f
                    result["material_id"] = mid
                    comparisons.append(result)
                else:
                    comparisons.append({"formula": f, "error": mid})
            output["comparison"] = comparisons

        else:
            output["error"] = (f"Unknown task: {task}. Use: run_md, "
                               "energy_minimize, rdf, thermal, full_analysis, compare")

        # Optional extras
        if wolfram_query:
            print(f"ðŸ§® Wolfram: {wolfram_query}")
            output["wolfram_extra"] = _wolfram_calc(wolfram_query)
        if search_query:
            print(f"ðŸŒ Search: {search_query}")
            output["web_extra"] = _web_search(search_query)

    except Exception as e:
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"âŒ MD Error: {e}")

    return _finalize(output, start_time)


def _finalize(output: Dict, start_time) -> str:
    elapsed = (datetime.now() - start_time).total_seconds()
    output["elapsed_seconds"] = round(elapsed, 2)
    output["status"] = "error" if "error" in output else "success"

    result_json = json.dumps(output, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"âœ… MD SIMULATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(result_json[:2000])
    if len(result_json) > 2000:
        print(f"... ({len(result_json)} total chars)")

    try:
        save_path = os.path.join(MD_OUTPUT_DIR,
                                 f"result_{int(start_time.timestamp())}.json")
        with open(save_path, "w") as f:
            f.write(result_json)
        print(f"ðŸ’¾ Saved to: {save_path}")
    except Exception:
        pass

    return result_json
