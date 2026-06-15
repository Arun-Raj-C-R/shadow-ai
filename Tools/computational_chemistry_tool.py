import os
import json
import random
import traceback
import webbrowser
from datetime import datetime

try:
    from pymatgen.core import Structure
except ImportError:
    Structure = None

# We can import the viewer logic we already built!
try:
    from crystal_viewer_tool import _get_structure, _build_html, OUTPUT_DIR
except ImportError:
    _get_structure = None

def run_computational_chemistry(
    base_formula: str,
    action: str,
    supercell: list = [1, 1, 1],
    dopant_original: str = None,
    dopant_new: str = None,
    num_dopants: int = 1,
    guest_molecule: str = None,
    relax_structure: bool = True
) -> str:
    """
    Advanced computational chemistry simulation.
    Builds a supercell from a base material, performs atomic substitutions (doping),
    and optionally relaxes the structure to its quantum mechanical ground state 
    using the CHGNet deep-learning interatomic potential.
    """
    if not Structure or not _get_structure:
        return "Error: Missing pymatgen or crystal_viewer_tool. Ensure pymatgen is installed."

    try:
        # 1. Fetch the base structure
        print(f"🔬 Fetching base structure for {base_formula}...")
        struct, mid = _get_structure(formula=base_formula)
        
        # 2. Make Supercell
        if supercell != [1, 1, 1]:
            struct.make_supercell(supercell)
            print(f"📦 Created {supercell} supercell. Total atoms: {struct.num_sites}")
            
        # 3. Apply Doping / Defects
        if action == "dope":
            if not dopant_original or not dopant_new:
                return "Error: 'dopant_original' and 'dopant_new' are required for doping."
                
            # Find all sites matching the original element
            target_indices = [i for i, site in enumerate(struct) if site.species_string == dopant_original]
            
            if not target_indices:
                return f"Error: No {dopant_original} atoms found in the structure to replace!"
                
            num_to_replace = min(num_dopants, len(target_indices))
            sites_to_replace = random.sample(target_indices, num_to_replace)
            
            for i in sites_to_replace:
                struct.replace(i, dopant_new)
                
            print(f"🧪 Doped structure: Replaced {num_to_replace} {dopant_original} atom(s) with {dopant_new}.")
            
        elif action == "intercalate":
            if not guest_molecule:
                return "Error: 'guest_molecule' required for intercalation."
            
            import requests
            print(f"🧬 Fetching guest molecule '{guest_molecule}' from PubChem...")
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{guest_molecule}/cids/JSON"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                cid = r.json()['IdentifierList']['CID'][0]
                sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
                r_sdf = requests.get(sdf_url, timeout=10)
                if r_sdf.status_code != 200:
                    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=2d"
                    r_sdf = requests.get(sdf_url, timeout=10)
                
                try:
                    from ase.io import read
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tf:
                        tf.write(r_sdf.text)
                        temp_path = tf.name
                    
                    mol_ase = read(temp_path, format="sdf")
                    os.remove(temp_path)
                    
                    # Center molecule and insert into the middle of the crystal lattice
                    center = struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
                    mol_center = mol_ase.get_center_of_mass()
                    for atom in mol_ase:
                        pos = atom.position - mol_center + center
                        struct.append(atom.symbol, pos, coords_are_cartesian=True)
                        
                    print(f"🧩 Intercalated {guest_molecule} into the crystal lattice!")
                except Exception as e:
                    return json.dumps({"error": f"Failed to insert guest molecule: {e}"})
            else:
                return json.dumps({"error": f"Could not find {guest_molecule} in PubChem."})
            
        formula_pretty = struct.composition.reduced_formula
        energy_str = "Not calculated"
        
        # 4. Geometry Optimization (Relaxation) using CHGNet
        if relax_structure:
            try:
                print("⚛️  Booting CHGNet Universal Machine-Learned Potential...")
                from chgnet.model import StructOptimizer
                import warnings
                warnings.filterwarnings("ignore") # suppress verbose ASE warnings
                
                relaxer = StructOptimizer()
                print("🔄 Relaxing atomic positions to find minimum energy state. This may take a few seconds...")
                result = relaxer.relax(struct, verbose=False)
                
                struct = result["final_structure"]
                energy = float(result["trajectory"].energies[-1])
                energy_str = f"{energy:.3f} eV"
                print(f"✅ Relaxation complete! Final Energy: {energy_str}")
                
            except ImportError:
                return json.dumps({
                    "error": "CHGNet or ASE not installed.",
                    "instruction": "Sir, to perform quantum-level geometry relaxation, you MUST install CHGNet and ASE. Please open your terminal and run: `pip install chgnet ase`. Until then, I can only build the unrelaxed rigid structures."
                })
                
        # 5. Generate 3D Hologram Viewer
        action_name = "Relaxed Supercell"
        if action == "dope": action_name = "Doped Supercell"
        elif action == "intercalate": action_name = f"Hybrid ({guest_molecule})"
        
        info = {
            "System": action_name,
            "Base Material": f"{base_formula} ({mid})",
            "Supercell": str(supercell),
            "Total Atoms": struct.num_sites,
            "Formula": formula_pretty,
            "Final Energy": energy_str,
            "Lattice a": f"{struct.lattice.a:.3f} Å",
            "Lattice b": f"{struct.lattice.b:.3f} Å",
            "Lattice c": f"{struct.lattice.c:.3f} Å"
        }
        
        html = _build_html(struct, "COMP-CHEM", formula_pretty, info)
        
        # Save and open
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%H%M%S')
        fname = f"sim_{formula_pretty}_{timestamp}.html"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html)
            
        # Also save the raw CIF file for external analysis (Vesta, Ovito)
        cif_path = os.path.join(OUTPUT_DIR, f"sim_{formula_pretty}_{timestamp}.cif")
        struct.to(filename=cif_path)
            
        webbrowser.open(f"file:///{fpath.replace(os.sep, '/')}")
        
        return json.dumps({
            "status": "success",
            "formula": formula_pretty,
            "atoms": struct.num_sites,
            "energy": energy_str,
            "cif_file": cif_path,
            "viewer_file": fpath,
            "message": "Structure successfully built, doped, relaxed, and loaded into 3D viewer!"
        })
        
    except Exception as e:
        return f"Computational Chemistry Simulation Failed: {str(e)}\n{traceback.format_exc()}"
