# dft_tool.py
"""
DFT Calculation Tool for SHADOW/Shadow AI
==========================================
Performs DFT-level materials calculations using:
  - Materials Project API (real DFT data: energies, band structures, DOS)
  - Pymatgen (structure manipulation, analysis, input generation)
  - Wolfram Alpha (depth calculations, physical constants)
  - Web search (supplementary data, literature values)

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

# â”€â”€ API Keys â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MP_API_KEY = os.environ.get("MP_API_KEY", "")
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")

# â”€â”€ Output directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DFT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dft_runs")
os.makedirs(DFT_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dft_tool")

# â”€â”€ Complex formula aliases (organic/hybrid â†’ closest MP analog) â”€â”€
FORMULA_ALIASES = {
    "FAPbI3": "CsPbI3", "MAPbI3": "CsPbI3",
    "FAPbBr3": "CsPbBr3", "MAPbBr3": "CsPbBr3",
    "FAPbCl3": "CsPbCl3", "MAPbCl3": "CsPbCl3",
    "FAPI": "CsPbI3", "MAPI": "CsPbI3",
}

def _resolve_formula(formula: str) -> tuple:
    """Resolve complex/organic formulas. Returns (resolved, note)."""
    if not formula:
        return formula, None
    stripped = formula.strip()
    for alias, resolved in FORMULA_ALIASES.items():
        if stripped.upper() == alias.upper():
            return resolved, f"'{formula}' â†’ using inorganic analog '{resolved}' from MP"
    if stripped.upper() == "ABX3":
        return "CsPbI3", "'ABX3' is a perovskite family. Using CsPbI3 as representative."
    return formula, None

def _ensure_structure(obj):
    """Convert dict/model to real pymatgen Structure."""
    from pymatgen.core import Structure
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        return Structure.from_dict(obj)
    if hasattr(obj, 'as_dict'):
        return Structure.from_dict(obj.as_dict())
    raise TypeError(f"Cannot convert {type(obj)} to Structure")

# ======================================================================
# MATERIALS PROJECT DATA FETCHER
# ======================================================================

def _fetch_mp_data(formula: str = None, material_id: str = None) -> Dict:
    """Fetch DFT-computed data from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        return {"error": "mp-api not installed. Run: pip install mp-api"}

    if not MP_API_KEY:
        return {"error": "MP_API_KEY not set in .env"}

    results = {}
    note = None
    try:
        with MPRester(MP_API_KEY) as mpr:
            # Resolve complex formulas
            if formula and not material_id:
                formula, note = _resolve_formula(formula)
                if note:
                    results["formula_note"] = note

                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "energy_above_hull"],
                    num_chunks=1
                )
                if not docs:
                    return {"error": f"No materials found for formula: {formula}", "formula_note": note}
                # Pick the most stable (lowest energy above hull)
                docs_sorted = sorted(docs, key=lambda d: d.energy_above_hull or 999)
                material_id = str(docs_sorted[0].material_id)
                results["material_id"] = material_id
                results["all_ids"] = [str(d.material_id) for d in docs_sorted[:5]]

            if not material_id:
                return {"error": "No formula or material_id provided"}

            results["material_id"] = material_id

            # â”€â”€ Summary properties â”€â”€
            try:
                summary_docs = mpr.materials.summary.search(material_ids=[material_id])
                if summary_docs:
                    summary = summary_docs[0]
                    results["properties"] = {
                    "formula": str(summary.formula_pretty) if summary.formula_pretty else None,
                    "space_group": str(summary.symmetry.symbol) if summary.symmetry else None,
                    "crystal_system": str(summary.symmetry.crystal_system) if summary.symmetry else None,
                    "band_gap_eV": summary.band_gap,
                    "is_metal": summary.is_metal,
                    "is_magnetic": summary.is_magnetic,
                    "energy_per_atom_eV": summary.energy_per_atom,
                    "formation_energy_per_atom_eV": summary.formation_energy_per_atom,
                    "energy_above_hull_eV": summary.energy_above_hull,
                    "density_g_cm3": summary.density,
                    "volume_A3": summary.volume,
                    "nsites": summary.nsites,
                }
            except Exception as e:
                results["properties_error"] = str(e)

            # â”€â”€ Structure â”€â”€
            try:
                raw_struct = mpr.get_structure_by_material_id(material_id)
                struct = _ensure_structure(raw_struct)
                results["structure"] = {
                    "lattice_a": round(struct.lattice.a, 4),
                    "lattice_b": round(struct.lattice.b, 4),
                    "lattice_c": round(struct.lattice.c, 4),
                    "alpha": round(struct.lattice.alpha, 2),
                    "beta": round(struct.lattice.beta, 2),
                    "gamma": round(struct.lattice.gamma, 2),
                    "num_sites": len(struct),
                    "species": list(set(str(s) for s in struct.species)),
                }
            except Exception as e:
                results["structure_error"] = str(e)

            # â”€â”€ Electronic structure â”€â”€
            try:
                es_docs = mpr.materials.electronic_structure.search(material_ids=[material_id])
                if es_docs:
                    es = es_docs[0]
                    results["electronic"] = {
                        "band_gap_eV": es.band_gap if hasattr(es, 'band_gap') else None,
                        "cbm_eV": es.cbm if hasattr(es, 'cbm') else None,
                        "vbm_eV": es.vbm if hasattr(es, 'vbm') else None,
                        "is_gap_direct": es.is_gap_direct if hasattr(es, 'is_gap_direct') else None,
                    }
            except Exception as e:
                results["electronic_error"] = str(e)

            # â”€â”€ Elasticity â”€â”€
            try:
                elast_docs = mpr.materials.elasticity.search(material_ids=[material_id])
                if elast_docs:
                    elast_doc = elast_docs[0]
                    if elast_doc.bulk_modulus:
                        results["elasticity"] = {
                        "bulk_modulus_GPa": elast_doc.bulk_modulus.vrh,
                        "shear_modulus_GPa": elast_doc.shear_modulus.vrh if elast_doc.shear_modulus else None,
                        "youngs_modulus_GPa": elast_doc.homogeneous_poisson if hasattr(elast_doc, 'homogeneous_poisson') else None,
                    }
            except Exception:
                pass  # Not all materials have elasticity data

            # â”€â”€ Dielectric â”€â”€
            try:
                diel_docs = mpr.materials.dielectric.search(material_ids=[material_id])
                if diel_docs:
                    diel_doc = diel_docs[0]
                    results["dielectric"] = {
                        "e_total": diel_doc.e_total if hasattr(diel_doc, 'e_total') else None,
                        "e_ionic": diel_doc.e_ionic if hasattr(diel_doc, 'e_ionic') else None,
                        "e_electronic": diel_doc.e_electronic if hasattr(diel_doc, 'e_electronic') else None,
                    }
            except Exception:
                pass

    except Exception as e:
        results["error"] = f"MP API error: {str(e)}"

    return results


# ======================================================================
# WOLFRAM ALPHA DEPTH CALCULATOR
# ======================================================================

def _wolfram_calculation(query: str) -> str:
    """Use Wolfram Alpha for precise physics/math calculations."""
    if not WOLFRAM_APP_ID:
        return "Wolfram Alpha not available (no WOLFRAM_APP_ID)"

    try:
        from wolfram_orchestrator_tool import simple_wolfram_query
        return simple_wolfram_query(query)
    except Exception as e:
        return f"Wolfram error: {str(e)}"


# ======================================================================
# WEB SEARCH FOR SUPPLEMENTARY DATA
# ======================================================================

def _web_search(query: str) -> str:
    """Search the web for supplementary DFT/materials data."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "No Google API key for web search"

        client = genai.Client(api_key=api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"DFT calculation data: {query}",
            config=config
        )
        return response.text if response.text else "No results"
    except Exception as e:
        return f"Search error: {str(e)}"


# ======================================================================
# PYMATGEN-BASED DFT ANALYSIS
# ======================================================================

def _analyze_structure(formula: str = None, material_id: str = None) -> Dict:
    """Perform structural analysis using pymatgen."""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from mp_api.client import MPRester
    except ImportError:
        return {"error": "pymatgen or mp-api not installed"}

    if not MP_API_KEY:
        return {"error": "MP_API_KEY not set"}

    results = {}
    try:
        with MPRester(MP_API_KEY) as mpr:
            if formula and not material_id:
                formula, note = _resolve_formula(formula)
                if note:
                    results["formula_note"] = note
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "energy_above_hull"],
                    num_chunks=1
                )
                if not docs:
                    return {"error": f"No structure found for {formula}"}
                docs_sorted = sorted(docs, key=lambda d: d.energy_above_hull or 999)
                material_id = str(docs_sorted[0].material_id)

            raw_struct = mpr.get_structure_by_material_id(material_id)
            struct = _ensure_structure(raw_struct)

            # Handle disordered structures
            if not struct.is_ordered:
                try:
                    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
                    trans = OrderDisorderedStructureTransformation()
                    struct = trans.apply_transformation(struct)
                    results["note"] = "Structure was disordered; auto-ordered for analysis."
                except Exception:
                    results["warning"] = "Disordered structure â€” some analyses may be approximate."

            analyzer = SpacegroupAnalyzer(struct)

            results["symmetry"] = {
                "space_group_symbol": analyzer.get_space_group_symbol(),
                "space_group_number": analyzer.get_space_group_number(),
                "crystal_system": analyzer.get_crystal_system(),
                "point_group": analyzer.get_point_group_symbol(),
                "hall_symbol": analyzer.get_hall(),
            }

            conv = analyzer.get_conventional_standard_structure()
            results["conventional_cell"] = {
                "a": round(conv.lattice.a, 4),
                "b": round(conv.lattice.b, 4),
                "c": round(conv.lattice.c, 4),
                "alpha": round(conv.lattice.alpha, 2),
                "beta": round(conv.lattice.beta, 2),
                "gamma": round(conv.lattice.gamma, 2),
                "volume_A3": round(conv.lattice.volume, 4),
                "num_atoms": len(conv),
            }

            prim = analyzer.get_primitive_standard_structure()
            results["primitive_cell"] = {
                "a": round(prim.lattice.a, 4),
                "b": round(prim.lattice.b, 4),
                "c": round(prim.lattice.c, 4),
                "volume_A3": round(prim.lattice.volume, 4),
                "num_atoms": len(prim),
            }

            # Bond lengths
            try:
                from pymatgen.analysis.local_env import CrystalNN
                cnn = CrystalNN()
                bond_info = []
                for i in range(min(len(struct), 4)):  # First 4 sites
                    try:
                        nn = cnn.get_nn_info(struct, i)
                        for n in nn[:3]:  # Top 3 neighbors
                            dist = struct.get_distance(i, n["site_index"])
                            bond_info.append({
                                "site": str(struct[i].species_string),
                                "neighbor": str(n["site"].species_string),
                                "distance_A": round(dist, 4),
                                "weight": round(n.get("weight", 0), 4),
                            })
                    except Exception:
                        continue
                if bond_info:
                    results["bonds"] = bond_info
            except Exception:
                pass

    except Exception as e:
        results["error"] = str(e)

    return results


def _generate_input_files(formula: str = None, material_id: str = None,
                          code: str = "vasp", calculation: str = "scf") -> Dict:
    """Generate DFT input files for VASP, QE, or ABINIT."""
    try:
        from mp_api.client import MPRester
        from pymatgen.core import Structure
    except ImportError:
        return {"error": "pymatgen/mp-api not installed"}

    if not MP_API_KEY:
        return {"error": "MP_API_KEY not set"}

    results = {}
    try:
        with MPRester(MP_API_KEY) as mpr:
            if formula and not material_id:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "energy_above_hull"],
                    num_chunks=1
                )
                if not docs:
                    return {"error": f"No structure for {formula}"}
                docs_sorted = sorted(docs, key=lambda d: d.energy_above_hull or 999)
                material_id = str(docs_sorted[0].material_id)

            struct = mpr.get_structure_by_material_id(material_id)
            results["material_id"] = material_id

            # Create output directory
            job_id = f"dft_{int(datetime.now().timestamp())}"
            job_dir = os.path.join(DFT_OUTPUT_DIR, job_id)
            os.makedirs(job_dir, exist_ok=True)

            code_lower = code.lower()

            if code_lower == "vasp":
                try:
                    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
                    if calculation == "relax":
                        vis = MPRelaxSet(struct)
                    else:
                        vis = MPStaticSet(struct)
                    vis.write_input(job_dir)
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "VASP"
                except Exception as e:
                    results["vasp_error"] = str(e)

            elif code_lower in ("qe", "quantum_espresso"):
                try:
                    from pymatgen.io.pwscf import PWInput
                    pseudos = {str(el): f"{el}.UPF" for el in struct.composition.elements}
                    pw_input = PWInput(
                        struct,
                        pseudo=pseudos,
                        control={"calculation": "'scf'", "prefix": "'material'",
                                 "outdir": "'./tmp'", "tprnfor": ".true."},
                        system={"ecutwfc": 50, "ecutrho": 400,
                                "occupations": "'smearing'", "smearing": "'gaussian'",
                                "degauss": 0.01},
                        electrons={"conv_thr": "1.0d-8", "mixing_beta": 0.7},
                        kpoints_grid=(6, 6, 6),
                    )
                    qe_path = os.path.join(job_dir, "scf.in")
                    pw_input.write_file(qe_path)
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "Quantum ESPRESSO"
                except Exception as e:
                    results["qe_error"] = str(e)

            elif code_lower == "abinit":
                try:
                    from pymatgen.io.abinit.abiobjects import structure_to_abivars
                    abivars = structure_to_abivars(struct)
                    abi_path = os.path.join(job_dir, "input.abi")
                    with open(abi_path, "w") as f:
                        f.write("# ABINIT input generated by Shadow AI\n")
                        f.write(f"# Material: {material_id}\n\n")
                        f.write("ecut 40.0  # Hartree\n")
                        f.write("nstep 100\n")
                        f.write("tolvrs 1.0d-10\n\n")
                        for key, val in abivars.items():
                            f.write(f"{key}  {val}\n")
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "ABINIT"
                except Exception as e:
                    results["abinit_error"] = str(e)

            elif code_lower == "cp2k":
                try:
                    cp2k_path = os.path.join(job_dir, "cp2k.inp")
                    with open(cp2k_path, "w") as f:
                        f.write("&FORCE_EVAL\n  METHOD Quickstep\n  &SUBSYS\n")
                        f.write(f"    &CELL\n      A {struct.lattice.a} 0 0\n      B 0 {struct.lattice.b} 0\n      C 0 0 {struct.lattice.c}\n    &END CELL\n")
                        f.write("    &COORD\n")
                        for site in struct:
                            f.write(f"      {site.species_string}  {site.x} {site.y} {site.z}\n")
                        f.write("    &END COORD\n  &END SUBSYS\n&END FORCE_EVAL\n")
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "CP2K"
                except Exception as e:
                    results["cp2k_error"] = str(e)

            elif code_lower == "orca":
                try:
                    orca_path = os.path.join(job_dir, "orca.inp")
                    with open(orca_path, "w") as f:
                        f.write(f"! B3LYP def2-SVP Opt\n%pal nprocs 4 end\n* xyz 0 1\n")
                        for site in struct:
                            f.write(f"{site.species_string}  {site.x} {site.y} {site.z}\n")
                        f.write("*\n")
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "ORCA"
                except Exception as e:
                    results["orca_error"] = str(e)
                    
            elif code_lower == "pyscf":
                try:
                    pyscf_path = os.path.join(job_dir, "run_pyscf.py")
                    with open(pyscf_path, "w") as f:
                        f.write("from pyscf import gto, scf\n")
                        f.write("mol = gto.M(\n")
                        f.write("    atom='''\n")
                        for site in struct:
                            f.write(f"        {site.species_string}  {site.x} {site.y} {site.z}\n")
                        f.write("    ''',\n")
                        f.write("    basis='def2-svp',\n")
                        f.write("    charge=0,\n")
                        f.write("    spin=0\n")
                        f.write(")\n")
                        f.write("mf = scf.RHF(mol)\n")
                        f.write("mf.kernel()\n")
                    results["files"] = os.listdir(job_dir)
                    results["code"] = "PySCF"
                except Exception as e:
                    results["pyscf_error"] = str(e)

            results["output_dir"] = job_dir
            results["status"] = "input_files_generated"

    except Exception as e:
        results["error"] = str(e)

    return results


# ======================================================================
# MAIN DFT CALCULATION ORCHESTRATOR
# ======================================================================

def run_dft_calculation(
    task: str,
    formula: str = None,
    material_id: str = None,
    code: str = "vasp",
    calculation_type: str = "scf",
    properties: List[str] = None,
    wolfram_query: str = None,
    search_query: str = None,
) -> str:
    """
    Main DFT tool entry point. Called by Shadow AI.

    Args:
        task: What to do. One of:
            - "full_analysis"     : Complete DFT data fetch + analysis
            - "get_properties"    : Fetch computed properties from MP
            - "analyze_structure" : Symmetry, bonds, cell analysis
            - "generate_inputs"   : Generate VASP/QE/ABINIT/CP2K/ORCA/PySCF input files
            - "band_structure"    : Get band structure data
            - "dos"               : Get density of states
            - "compare"           : Compare multiple materials
        formula:           Chemical formula (e.g. "Si", "GaAs", "CsPbI3")
        material_id:       Materials Project ID (e.g. "mp-149")
        code:              DFT code for input generation ("vasp", "qe", "abinit", "cp2k", "orca", "pyscf")
        calculation_type:  "scf", "relax", "bands", "dos"
        properties:        List of specific properties to fetch
        wolfram_query:     Optional Wolfram Alpha query for depth calc
        search_query:      Optional web search for supplementary data

    Returns:
        JSON string with all results
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"ðŸ”¬ DFT CALCULATION ENGINE")
    print(f"   Task: {task}")
    print(f"   Material: {formula or material_id}")
    print(f"   Time: {start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    output = {
        "task": task,
        "formula": formula,
        "material_id": material_id,
        "timestamp": start_time.isoformat(),
    }

    try:
        # â”€â”€ TASK ROUTING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        if task == "full_analysis":
            print("ðŸ“Š Step 1/4: Fetching DFT data from Materials Project...")
            mp_data = _fetch_mp_data(formula=formula, material_id=material_id)
            output["mp_data"] = mp_data
            material_id = mp_data.get("material_id", material_id)

            print("ðŸ” Step 2/4: Analyzing crystal structure...")
            struct_data = _analyze_structure(formula=formula, material_id=material_id)
            output["structure_analysis"] = struct_data

            print("ðŸ§® Step 3/4: Wolfram Alpha depth calculations...")
            mat_name = formula or material_id
            wolfram_results = {}
            for prop in ["band gap", "density", "lattice constant", "bulk modulus"]:
                q = f"{prop} of {mat_name}"
                wolfram_results[prop] = _wolfram_calculation(q)
            output["wolfram_data"] = wolfram_results

            print("ðŸŒ Step 4/4: Web search for supplementary data...")
            search_result = _web_search(
                f"{mat_name} DFT calculated properties band gap crystal structure"
            )
            output["web_data"] = search_result

        elif task == "get_properties":
            print("ðŸ“Š Fetching DFT-computed properties...")
            output["mp_data"] = _fetch_mp_data(formula=formula, material_id=material_id)

        elif task == "analyze_structure":
            print("ðŸ” Running structural analysis...")
            output["structure_analysis"] = _analyze_structure(
                formula=formula, material_id=material_id
            )

        elif task == "generate_inputs":
            print(f"ðŸ“ Generating {code.upper()} input files...")
            output["input_generation"] = _generate_input_files(
                formula=formula, material_id=material_id,
                code=code, calculation=calculation_type
            )

        elif task == "band_structure":
            print("ðŸ“ˆ Fetching band structure data...")
            mp_data = _fetch_mp_data(formula=formula, material_id=material_id)
            output["mp_data"] = mp_data
            mid = mp_data.get("material_id", material_id)
            if mid:
                try:
                    from mp_api.client import MPRester
                    with MPRester(MP_API_KEY) as mpr:
                        bs = mpr.get_bandstructure_by_material_id(mid)
                        if bs:
                            output["band_structure"] = {
                                "band_gap_eV": round(bs.get_band_gap()["energy"], 4),
                                "is_direct": bs.get_band_gap()["direct"],
                                "vbm": round(bs.get_vbm()["energy"], 4),
                                "cbm": round(bs.get_cbm()["energy"], 4),
                                "num_bands": bs.nb_bands,
                            }
                except Exception as e:
                    output["band_structure_error"] = str(e)

        elif task == "dos":
            print("ðŸ“Š Fetching density of states...")
            mp_data = _fetch_mp_data(formula=formula, material_id=material_id)
            output["mp_data"] = mp_data
            mid = mp_data.get("material_id", material_id)
            if mid:
                try:
                    from mp_api.client import MPRester
                    with MPRester(MP_API_KEY) as mpr:
                        dos = mpr.get_dos_by_material_id(mid)
                        if dos:
                            output["dos"] = {
                                "efermi_eV": round(dos.efermi, 4),
                                "band_gap_eV": round(dos.get_gap(), 4),
                                "num_energies": len(dos.energies),
                            }
                except Exception as e:
                    output["dos_error"] = str(e)

        elif task == "compare":
            print("ðŸ”„ Comparing materials...")
            formulas = [f.strip() for f in (formula or "").split(",")]
            comparisons = []
            for f in formulas:
                data = _fetch_mp_data(formula=f)
                comparisons.append({"formula": f, "data": data})
            output["comparison"] = comparisons

        else:
            output["error"] = f"Unknown task: {task}. Use: full_analysis, get_properties, analyze_structure, generate_inputs, band_structure, dos, compare"

        # â”€â”€ Optional Wolfram query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if wolfram_query:
            print(f"ðŸ§® Wolfram Alpha: {wolfram_query}")
            output["wolfram_extra"] = _wolfram_calculation(wolfram_query)

        # â”€â”€ Optional web search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if search_query:
            print(f"ðŸŒ Web search: {search_query}")
            output["web_extra"] = _web_search(search_query)

    except Exception as e:
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"âŒ DFT Error: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    output["elapsed_seconds"] = round(elapsed, 2)
    output["status"] = "error" if "error" in output else "success"

    result_json = json.dumps(output, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"âœ… DFT CALCULATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(result_json[:2000])
    if len(result_json) > 2000:
        print(f"... ({len(result_json)} total chars)")

    # Save to file
    try:
        save_path = os.path.join(DFT_OUTPUT_DIR, f"result_{int(start_time.timestamp())}.json")
        with open(save_path, "w") as f:
            f.write(result_json)
        print(f"ðŸ’¾ Saved to: {save_path}")
    except Exception:
        pass

    return result_json