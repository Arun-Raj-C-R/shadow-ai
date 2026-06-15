# pymatgen_orchestrator_tool.py
"""
The "Bridge" tool that Shadow calls.
This tool takes JSON strings (from materials_orchestrator_tool) and
deserializes them into Pymatgen objects, then passes them to the
PymatgenAnalysisTool for "peak-capacity" calculations.
"""
import json
import logging
import pathlib
from monty.serialization import MontyDecoder, MontyEncoder
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core.composition import Composition

# Import the "engine"
from pymatgen_analysis_tool import PymatgenAnalysisTool

# --- Configuration ---
PLOTS_DIR = pathlib.Path(r"D:\Project File\Shadow\Shadow\Brain\Shadow2\plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def _decode_pymatgen_json(json_string: str):
    """Deserializes a JSON string back into a Pymatgen object."""
    try:
        data = json.loads(json_string)
        # MontyDecoder will turn any Pymatgen objects *within* the data
        # from dicts into real objects.
        return MontyDecoder().decode(data)
    except Exception as e:
        logging.error(f"Failed to decode pymatgen JSON: {e}", exc_info=True)
        return None

def _encode_pymatgen_json(obj) -> str:
    """Serializes a Pymatgen object into a JSON string."""
    try:
        return json.dumps(obj, cls=MontyEncoder)
    except Exception as e:
        logging.error(f"Failed to encode pymatgen JSON: {e}", exc_info=True)
        return json.dumps({"error": str(e)})

def _create_entries_from_json(json_string: str) -> list:
    """
    Constructs a list of ComputedEntry objects from the JSON
    returned by the 'discover_mp_materials' tool.
    """
    try:
        materials_list = json.loads(json_string)
        entries = []
        for mat in materials_list:
            # We need composition, energy, and ID
            # Use formula_pretty for Composition
            comp = Composition(mat['formula_pretty'])
            # We need total formation energy, not per-atom
            energy = mat['formation_energy_per_atom'] * comp.num_atoms
            entry_id = mat.get('material_id')
            # Create the entry
            entries.append(ComputedEntry(composition=comp, energy=energy, entry_id=entry_id))
        return entries
    except Exception as e:
        logging.error(f"Failed to create ComputedEntries: {e}", exc_info=True)
        return []

# ==============================================================================
# --- 1. TOOL WRAPPER FUNCTIONS (What the AI will call) ---
# ==============================================================================

def get_slme_efficiency(band_gap: float) -> str:
    try:
        result = PymatgenAnalysisTool.get_theoretical_efficiency(band_gap)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def analyze_effective_mass(band_structure_json: str) -> str:
    bs = _decode_pymatgen_json(band_structure_json)
    if not bs:
        return json.dumps({"error": "Invalid BandStructure JSON provided."})
    result = PymatgenAnalysisTool.calculate_effective_mass(bs)
    return _encode_pymatgen_json(result) 

def plot_band_structure(band_structure_json: str, filename: str) -> str:
    bs = _decode_pymatgen_json(band_structure_json)
    if not bs:
        return "Error: Invalid BandStructure JSON provided."
    
    save_path = PLOTS_DIR / (filename + ".png")
    result_path = PymatgenAnalysisTool.plot_band_structure(bs, str(save_path))
    return f"SUCCESS: Plot saved to {result_path}"

def plot_dos(dos_json: str, filename: str) -> str:
    dos = _decode_pymatgen_json(dos_json)
    if not dos:
        return "Error: Invalid DOS JSON provided."
        
    save_path = PLOTS_DIR / (filename + ".png")
    result_path = PymatgenAnalysisTool.plot_dos(dos, str(save_path))
    return f"SUCCESS: Plot saved to {result_path}"

def analyze_phase_stability(materials_list_json: str) -> str:
    entries = _create_entries_from_json(materials_list_json)
    if not entries:
        return json.dumps({"error": "Could not construct ComputedEntry list from JSON. Ensure JSON is from 'discover_mp_materials' and contains 'formula_pretty' and 'formation_energy_per_atom'."})
    result = PymatgenAnalysisTool.get_phase_stability(entries)
    return json.dumps(result)

# --- FULL CAPACITY FUNCTIONS (NOW FIXED) ---

def analyze_dielectric_constant(dielectric_json: str) -> str:
    try:
        data = json.loads(dielectric_json)
        result = PymatgenAnalysisTool.get_dielectric_constant(data)
        return json.dumps({"average_static_dielectric_constant": result})
    except Exception as e:
        return json.dumps({"error": f"Invalid dielectric JSON: {e}"})

def analyze_reaction_energy(materials_list_json: str, reaction_str: str) -> str:
    entries = _create_entries_from_json(materials_list_json)
    if not entries:
        return json.dumps({"error": "Could not construct ComputedEntry list from JSON."})
    
    result = PymatgenAnalysisTool.calculate_reaction_energy(entries, reaction_str)
    return json.dumps({"reaction": reaction_str, "reaction_energy_ev": result})

def generate_surface_slab(properties_json: str, miller_index: list[int]) -> str:
    """
    Takes a properties JSON (from get_mp_properties) and generates a 
    surface slab. Returns the new slab structure as JSON.
    """
    # Fixed bug
    # We decode the *entire* properties dictionary
    properties_dict = _decode_pymatgen_json(properties_json)
    if not properties_dict:
        return json.dumps({"error": "Invalid properties JSON provided."})
    
    # We *extract* the structure object from the dictionary
    structure = properties_dict.get('structure')
    if not structure:
        return json.dumps({"error": "No 'structure' object found in the provided JSON."})
    # --- END FIX ---
    
    slab = PymatgenAnalysisTool.create_surface_slab(structure, miller_index)
    return _encode_pymatgen_json(slab) # Return the NEW structure

def generate_doped_structure(properties_json: str, species_to_replace: str, dopant: str, fraction: float) -> str:
    """
    Takes a properties JSON and creates a new doped structure.
    Returns the new doped structure as JSON.
    """
    # Fixed bug
    properties_dict = _decode_pymatgen_json(properties_json)
    if not properties_dict:
        return json.dumps({"error": "Invalid properties JSON provided."})
    
    structure = properties_dict.get('structure')
    if not structure:
        return json.dumps({"error": "No 'structure' object found in the provided JSON."})
    # --- END FIX ---
        
    doped_structure = PymatgenAnalysisTool.create_doped_structure(structure, species_to_replace, dopant, fraction)
    return _encode_pymatgen_json(doped_structure)

def generate_vacancy(properties_json: str, site_index: int) -> str:
    """
    Takes a properties JSON and creates a vacancy at a specific site.
    Returns the new structure with the vacancy as JSON.
    """
    # Fixed bug: <insert bug description>
    properties_dict = _decode_pymatgen_json(properties_json)
    if not properties_dict:
        return json.dumps({"error": "Invalid properties JSON provided."})
    
    structure = properties_dict.get('structure')
    if not structure:
        return json.dumps({"error": "No 'structure' object found in the provided JSON."})
    # --- END FIX ---
    
    if site_index >= len(structure):
        return json.dumps({"error": f"Invalid site_index. Structure only has {len(structure)} sites."})
        
    vacant_structure = PymatgenAnalysisTool.create_vacancy(structure, site_index)
    return _encode_pymatgen_json(vacant_structure)


# ==============================================================================
# --- 2. TOOL DEFINITIONS FOR THE AGENT ---
# ==============================================================================

PYMATGEN_TOOL_DEFINITIONS = [
    {
        "name": "get_slme_efficiency",
        "description": "Calculates the Shockley-Queisser (SLME) theoretical maximum solar cell efficiency. Use this when asked for 'theoretical efficiency'.",
        "parameters": {
            "type": "OBJECT",
            "properties": { "band_gap": { "type": "NUMBER", "description": "The band gap in eV (e.g., 1.55)." } },
            "required": ["band_gap"]
        }
    },
    {
        "name": "analyze_effective_mass",
        "description": "Calculates effective mass. Requires the *full JSON* from the 'get_mp_band_structure' tool as input.",
        "parameters": {
            "type": "OBJECT",
            "properties": { "band_structure_json": { "type": "STRING", "description": "The JSON string output from a previous 'get_mp_band_structure' call." } },
            "required": ["band_structure_json"]
        }
    },
    {
        "name": "plot_band_structure",
        "description": "Saves a plot of a band structure. Requires the *full JSON* from 'get_mp_band_structure' as input.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "band_structure_json": { "type": "STRING", "description": "The JSON string output from a previous 'get_mp_band_structure' call." },
                "filename": { "type": "STRING", "description": "The name for the plot file (e.g., 'mp-123_bs'). Do not include .png." }
            },
            "required": ["band_structure_json", "filename"]
        }
    },
    {
        "name": "plot_dos",
        "description": "Saves a plot of a Density of States. Requires the *full JSON* from 'get_mp_dos' as input.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dos_json": { "type": "STRING", "description": "The JSON string output from a previous 'get_mp_dos' call." },
                "filename": { "type": "STRING", "description": "The name for the plot file (e.g., 'mp-123_dos'). Do not include .png." }
            },
            "required": ["dos_json", "filename"]
        }
    },
    {
        "name": "analyze_phase_stability",
        "description": "Calculates phase stability (energy above hull) for a chemical system. Requires the *full JSON list* from 'discover_mp_materials' as input.",
        "parameters": {
            "type": "OBJECT",
            "properties": { "materials_list_json": { "type": "STRING", "description": "The JSON list output from a previous 'discover_mp_materials' call." } },
            "required": ["materials_list_json"]
        }
    },
    {
        "name": "analyze_reaction_energy",
        "description": "Calculates the energy of a reaction. Requires *both* a JSON list of materials (from 'discover_mp_materials') *and* a reaction string.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "materials_list_json": { "type": "STRING", "description": "The JSON list output from a 'discover_mp_materials' call for that chemical system." },
                "reaction_str": { "type": "STRING", "description": "The reaction string (e.g., 'CsPbI3 -> CsI + PbI2')." }
            },
            "required": ["materials_list_json", "reaction_str"]
        }
    },
    {
        "name": "generate_surface_slab",
        "description": "Generates a surface slab from a bulk structure. Requires the *full properties JSON* from a 'get_mp_properties' call.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                # --- BUG FIX ---
                "properties_json": { "type": "STRING", "description": "The *full* JSON string output from a 'get_mp_properties' call, which contains the 'structure' object." },
                # --- END FIX ---
                "miller_index": { "type": "ARRAY", "description": "The Miller index for the surface (e.g., [1, 0, 0]).", "items": {"type": "INTEGER"} }
            },
            "required": ["properties_json", "miller_index"]
        }
    },
    {
        "name": "generate_doped_structure",
        "description": "Generates a new doped structure. Requires the *full properties JSON* from a 'get_mp_properties' call.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                # --- BUG FIX ---
                "properties_json": { "type": "STRING", "description": "The *full* JSON string output from a 'get_mp_properties' call, which contains the 'structure' object." },
                # --- END FIX ---
                "species_to_replace": { "type": "STRING", "description": "Element to remove (e.g., 'Pb')." },
                "dopant": { "type": "STRING", "description": "Element to add (e.g., 'Sn')." },
                "fraction": { "type": "NUMBER", "description": "Fractional amount to substitute (e.g., 0.05 for 5%)." }
            },
            "required": ["properties_json", "species_to_replace", "dopant", "fraction"]
        }
    },
    {
        "name": "generate_vacancy",
        "description": "Generates a new structure with a vacancy. Requires the *full properties JSON* from a 'get_mp_properties' call.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                # --- BUG FIX ---
                "properties_json": { "type": "STRING", "description": "The *full* JSON string output from a 'get_mp_properties' call, which contains the 'structure' object." },
                # --- END FIX ---
                "site_index": { "type": "INTEGER", "description": "The index of the atom to remove (starts at 0)." }
            },
            "required": ["properties_json", "site_index"]
        }
    }
]