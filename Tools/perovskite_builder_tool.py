import os
import json
import random
import numpy as np
import requests
from datetime import datetime
import webbrowser
import traceback

from crystal_viewer_tool import _get_structure, _build_html, OUTPUT_DIR

# Temporary cache for downloaded molecules
_mol_cache = {}

def _fetch_molecule(name: str):
    if name in _mol_cache: return _mol_cache[name]
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
    r = requests.get(url, timeout=10)
    if r.status_code != 200: raise ValueError(f"Molecule {name} not found")
    cid = r.json()['IdentifierList']['CID'][0]
    
    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
    r_sdf = requests.get(sdf_url, timeout=10)
    if r_sdf.status_code != 200:
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=2d"
        r_sdf = requests.get(sdf_url, timeout=10)
        
    from ase.io import read
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tf:
        tf.write(r_sdf.text)
        temp_path = tf.name
    mol = read(temp_path, format="sdf")
    os.remove(temp_path)
    
    _mol_cache[name] = mol
    return mol

def build_complex_perovskite(
    cs_frac: float = 0.05,
    fa_frac: float = 0.80,
    ma_frac: float = 0.15,
    i_frac: float = 0.85,
    br_frac: float = 0.15,
    supercell_size: int = 3
):
    """
    Builds a massive Triple Cation Mixed Halide Perovskite structure.
    Generates fractional substitutions by inserting entire FA and MA molecules.
    """
    try:
        from pymatgen.core import Structure
        from scipy.spatial.transform import Rotation
    except ImportError:
        return json.dumps({"error": "pymatgen and scipy required. Run `pip install pymatgen scipy`"})

    print(f"🏗️ Building Complex Perovskite: Cs({cs_frac}) FA({fa_frac}) MA({ma_frac}) Pb I({i_frac}) Br({br_frac})")
    
    # 1. Fetch Cubic CsPbI3 (Base)
    struct, _ = _get_structure(formula="CsPbI3")
    
    # 2. Make Supercell
    struct.make_supercell([supercell_size, supercell_size, supercell_size])
    total_cells = supercell_size ** 3
    print(f"📦 Supercell created. Volume: {struct.volume:.1f} A^3. Total unit cells: {total_cells}")
    
    # Normalize fractions just in case
    a_total = cs_frac + fa_frac + ma_frac
    c_f, f_f, m_f = cs_frac/a_total, fa_frac/a_total, ma_frac/a_total
    x_total = i_frac + br_frac
    i_f, b_f = i_frac/x_total, br_frac/x_total
    
    n_fa = int(round(total_cells * f_f))
    n_ma = int(round(total_cells * m_f))
    n_cs = total_cells - n_fa - n_ma  # remainder
    
    total_x = total_cells * 3
    n_br = int(round(total_x * b_f))
    
    # 3. Handle Halide (X-site) substitution
    i_sites = [i for i, s in enumerate(struct) if s.species_string == "I"]
    random.shuffle(i_sites)
    for i in i_sites[:n_br]:
        struct.replace(i, "Br")
    print(f"⚛️  Halide Mixing: Replaced {n_br} Iodine with Bromine.")
    
    # 4. Handle Cation (A-site) Organic substitution
    cs_sites = [i for i, s in enumerate(struct) if s.species_string == "Cs"]
    random.shuffle(cs_sites)
    
    fa_indices = cs_sites[:n_fa]
    ma_indices = cs_sites[n_fa : n_fa+n_ma]
    remove_indices = fa_indices + ma_indices
    
    # Pre-fetch molecules
    print("🧬 Fetching Organic Cations (FA and MA)...")
    mol_fa = _fetch_molecule("Formamidinium") if n_fa > 0 else None
    mol_ma = _fetch_molecule("Methylammonium") if n_ma > 0 else None
    
    # Extract coords of sites to replace
    fa_coords = [struct[i].coords for i in fa_indices]
    ma_coords = [struct[i].coords for i in ma_indices]
    
    # Delete the Cs atoms being replaced
    struct.remove_sites(remove_indices)
    
    # Insert FA
    for center_coord in fa_coords:
        mol = mol_fa.copy()
        # Random 3D rotation
        rot = Rotation.random().as_matrix()
        mol.positions = np.dot(mol.positions - mol.get_center_of_mass(), rot) + center_coord
        for atom in mol:
            struct.append(atom.symbol, atom.position, coords_are_cartesian=True)
            
    # Insert MA
    for center_coord in ma_coords:
        mol = mol_ma.copy()
        rot = Rotation.random().as_matrix()
        mol.positions = np.dot(mol.positions - mol.get_center_of_mass(), rot) + center_coord
        for atom in mol:
            struct.append(atom.symbol, atom.position, coords_are_cartesian=True)

    print(f"🧩 Organic Insertion: Added {n_fa} FA and {n_ma} MA molecules into the lattice.")
    
    # 5. Save and Visualize
    formula_pretty = "TripleCationPerovskite"
    timestamp = datetime.now().strftime('%H%M%S')
    cif_path = os.path.join(OUTPUT_DIR, f"{formula_pretty}_{timestamp}.cif")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    struct.to(filename=cif_path)
    
    info = {
        "System": "Hybrid Organic-Inorganic Perovskite",
        "Requested": f"Cs({cs_frac}) FA({fa_frac}) MA({ma_frac}) Pb(I{i_frac} Br{br_frac})3",
        "Actual Count": f"Cs:{n_cs}, FA:{n_fa}, MA:{n_ma}, Br:{n_br}",
        "Total Atoms": struct.num_sites,
        "Supercell": f"{supercell_size}x{supercell_size}x{supercell_size}"
    }
    
    html = _build_html(struct, "HOIP", formula_pretty, info)
    html_path = os.path.join(OUTPUT_DIR, f"{formula_pretty}_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
    
    return json.dumps({
        "status": "success",
        "total_atoms": struct.num_sites,
        "cif_file": cif_path,
        "viewer_file": html_path,
        "message": f"Successfully constructed 3D complex perovskite model! CIF saved to {cif_path}. Use this CIF file for MD simulation."
    })
