# materials_orchestrator_tool.py
"""
Orchestrator for LLM function calling.
Cleans output, handles errors, returns JSON strings.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Union
from materials_project_tool import MaterialsProjectTool

# Load API key
from dotenv import load_dotenv
import os
config_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", ".env")
if os.path.exists(config_env):
    load_dotenv(config_env, override=True)
else:
    load_dotenv()
MATERIALS_PROJECT_API_KEY = os.environ.get("MP_API_KEY")

# Initialize client
try:
    if not MATERIALS_PROJECT_API_KEY:
        raise ValueError("MP_API_KEY missing")
    mp_client = MaterialsProjectTool(api_key=MATERIALS_PROJECT_API_KEY)
    logging.info("MP Orchestrator initialized.")
except Exception as e:
    logging.error(f"MP tool failed: {e}")
    mp_client = None


def _handle_tool_call(func, *args, **kwargs) -> str:
    if not mp_client:
        return "Error: Materials Project not available. Check MP_API_KEY."
    try:
        result = func(*args, **kwargs)
        if not result:
            return "No data found."
        output = json.dumps(result, indent=2, default=str)
        # Truncate very large outputs to avoid context bloat
        if len(output) > 8000:
            output = output[:8000] + f"\n... (truncated, {len(output)} total chars)"
        return output
    except TypeError as e:
        # Fallback for non-serializable objects
        return json.dumps({"result": str(result), "warning": "Simplified output"}, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# ==============================================================================
# CORE DATA ACCESS TOOLS
# ==============================================================================

def search_mp_by_formula(formula: str) -> str:
    return _handle_tool_call(mp_client.search_mp_by_formula, formula)

def get_mp_properties(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_mp_properties, material_id)

def get_mp_data(criteria: Dict, properties: List[str] = None) -> str:
    return _handle_tool_call(mp_client.get_data, criteria, properties)

def get_materials_ids(formula: str) -> str:
    return _handle_tool_call(mp_client.get_materials_ids, formula)

def get_structure_by_material_id(material_id: str, final: bool = True, conventional_unit_cell: bool = False) -> str:
    return _handle_tool_call(mp_client.get_structure_by_material_id, material_id, final, conventional_unit_cell)

def get_structures_by_material_ids(material_ids: List[str], final: bool = True) -> str:
    return _handle_tool_call(mp_client.get_structures_by_material_ids, material_ids, final)

def get_entry_by_material_id(material_id: str, compatible_only: bool = True) -> str:
    return _handle_tool_call(mp_client.get_entry_by_material_id, material_id, compatible_only)

def get_entries(criteria: Dict, compatible_only: bool = True, inc_structure: str = None) -> str:
    return _handle_tool_call(mp_client.get_entries, criteria, compatible_only, inc_structure)

def get_entries_in_system(elements: List[str], compatible_only: bool = True) -> str:
    return _handle_tool_call(mp_client.get_entries_in_system, elements, compatible_only)

# ==============================================================================
# ELECTRONIC STRUCTURE TOOLS
# ==============================================================================

def get_mp_band_structure(material_id: str, line_mode: bool = True) -> str:
    return _handle_tool_call(mp_client.get_bandstructure_by_material_id, material_id, line_mode)

def get_mp_dos(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_dos_by_material_id, material_id)

def get_electronic_structure_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_electronic_structure_data, material_id)

def get_phonon_bandstructure_by_material_id(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_phonon_bandstructure_by_material_id, material_id)

def get_phonon_dos_by_material_id(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_phonon_dos_by_material_id, material_id)

# ==============================================================================
# THERMODYNAMIC & PHASE DIAGRAM TOOLS
# ==============================================================================

def get_thermo_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_thermo_data, material_id)

def get_pourbaix_entries(elements: List[str]) -> str:
    return _handle_tool_call(mp_client.get_pourbaix_entries, elements)

def get_reaction(entries: List[str]) -> str:
    return _handle_tool_call(mp_client.get_reaction, entries)

def get_phase_diagram_from_entries(entries: List[str]) -> str:
    return _handle_tool_call(mp_client.get_phase_diagram_from_entries, entries)

def get_exp_thermo_data(formula: str) -> str:
    return _handle_tool_call(mp_client.get_exp_thermo_data, formula)

# ==============================================================================
# SURFACE & INTERFACE TOOLS
# ==============================================================================

def get_wulff_shape(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_wulff_shape, material_id)

def get_surface_data(material_id: str, miller_index: List[int] = None) -> str:
    return _handle_tool_call(mp_client.get_surface_data, material_id, miller_index)

def get_substrate_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_substrate_data, material_id)

def get_interface_reactions(reactants: List[str], products: List[str]) -> str:
    return _handle_tool_call(mp_client.get_interface_reactions, reactants, products)

def get_gb_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_gb_data, material_id)

# ==============================================================================
# MECHANICAL & ELECTRONIC PROPERTY TOOLS
# ==============================================================================

def get_elasticity_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_elasticity_data, material_id)

def get_piezoelectric_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_piezoelectric_data, material_id)

def get_dielectric_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_dielectric_data, material_id)

def get_magnetism_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_magnetism_data, material_id)

def get_cohesive_energy(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_cohesive_energy, material_id)

# ==============================================================================
# SPECTROSCOPY TOOLS
# ==============================================================================

def get_xas_data(material_id: str, spectrum_type: str = "XANES") -> str:
    return _handle_tool_call(mp_client.get_xas_data, material_id, spectrum_type)

def get_ir_spectra(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_ir_spectra, material_id)

# ==============================================================================
# BATTERY & ELECTROCHEMISTRY TOOLS
# ==============================================================================

def get_battery_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_battery_data, material_id)

def get_electrode_data(elements: List[str]) -> str:
    return _handle_tool_call(mp_client.get_electrode_data, elements)

# ==============================================================================
# CHEMICAL ANALYSIS TOOLS
# ==============================================================================

def get_oxidation_states(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_oxidation_states, material_id)

def get_bond_valence_data(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_bond_valence_data, material_id)

# ==============================================================================
# DISCOVERY & SEARCH TOOLS
# ==============================================================================

def discover_mp_materials(
    elements: List[str],
    nelements: int = None,
    crystal_system: str = None,
    band_gap_min: float = None,
    band_gap_max: float = None,
    max_energy_above_hull: float = None
) -> str:
    if not elements:
        return "Error: 'elements' required."
    bg = (band_gap_min or 0, band_gap_max or 10) if band_gap_min or band_gap_max else None
    return _handle_tool_call(
        mp_client.search_mp_by_criteria,
        elements=elements,
        nelements=nelements,
        crystal_system=crystal_system,
        band_gap=bg,
        energy_above_hull=max_energy_above_hull
    )

def query_mp(criteria: Dict, properties: List[str] = None) -> str:
    return _handle_tool_call(mp_client.query, criteria, properties)

# ==============================================================================
# ADVANCED DATA ACCESS
# ==============================================================================

def get_materials_id_doc(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_materials_id_doc, material_id)

def get_task_data(task_id: str) -> str:
    return _handle_tool_call(mp_client.get_task_data, task_id)

def get_task_ids_associated_with_material_id(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_task_ids_associated_with_material_id, material_id)

def get_materials_id_from_task_id(task_id: str) -> str:
    return _handle_tool_call(mp_client.get_materials_id_from_task_id, task_id)

def get_materials_id_references(material_id: str) -> str:
    return _handle_tool_call(mp_client.get_materials_id_references, material_id)

def get_charge_density_data(task_id: str) -> str:
    return _handle_tool_call(mp_client.get_charge_density_data, task_id)

def get_wavefunction_data(task_id: str) -> str:
    return _handle_tool_call(mp_client.get_wavefunction_data, task_id)

# ==============================================================================
# TOOL DEFINITIONS
# ==============================================================================

MATERIALS_TOOL_DEFINITIONS = [
    # Core Data Access
    {
        "name": "search_mp_by_formula",
        "description": "Search materials by chemical formula (e.g., SiC, Fe2O3)",
        "parameters": {
            "type": "object",
            "properties": {"formula": {"type": "string"}},
            "required": ["formula"]
        }
    },
    {
        "name": "get_mp_properties",
        "description": "Get basic material properties (band gap, stability, density, etc.)",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_mp_data",
        "description": "General-purpose query with MongoDB-style criteria and property selection",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {"type": "object"},
                "properties": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["criteria"]
        }
    },
    {
        "name": "get_materials_ids",
        "description": "Get all material IDs for a given formula",
        "parameters": {
            "type": "object",
            "properties": {"formula": {"type": "string"}},
            "required": ["formula"]
        }
    },
    {
        "name": "get_structure_by_material_id",
        "description": "Get crystal structure data",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "final": {"type": "boolean", "default": True},
                "conventional_unit_cell": {"type": "boolean", "default": False}
            },
            "required": ["material_id"]
        }
    },
    
    # Electronic Structure
    {
        "name": "get_mp_band_structure",
        "description": "Get electronic band structure data",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "line_mode": {"type": "boolean", "default": True}
            },
            "required": ["material_id"]
        }
    },
    {
        "name": "get_mp_dos",
        "description": "Get density of states data",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_electronic_structure_data",
        "description": "Get unified electronic structure data (band gap, DOS summary)",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_phonon_bandstructure_by_material_id",
        "description": "Get phonon band structure data",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    
    # Thermodynamic & Phase Diagrams
    {
        "name": "get_thermo_data",
        "description": "Get thermodynamic properties (heat capacity, entropy, etc.)",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_pourbaix_entries",
        "description": "Get Pourbaix diagram entries for aqueous stability analysis",
        "parameters": {
            "type": "object",
            "properties": {"elements": {"type": "array", "items": {"type": "string"}}},
            "required": ["elements"]
        }
    },
    {
        "name": "get_phase_diagram_from_entries",
        "description": "Build phase diagram from list of material entries",
        "parameters": {
            "type": "object",
            "properties": {"entries": {"type": "array", "items": {"type": "string"}}},
            "required": ["entries"]
        }
    },
    
    # Surface & Interface
    {
        "name": "get_wulff_shape",
        "description": "Get Wulff shape from surface energies",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_surface_data",
        "description": "Get surface energy and slab data for specific Miller indices",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "miller_index": {"type": "array", "items": {"type": "integer"}}
            },
            "required": ["material_id"]
        }
    },
    
    # Mechanical & Electronic Properties
    {
        "name": "get_elasticity_data",
        "description": "Get elastic tensor and mechanical properties",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_piezoelectric_data",
        "description": "Get piezoelectric tensor data",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    {
        "name": "get_dielectric_data",
        "description": "Get dielectric constants (electronic + ionic)",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    
    # Spectroscopy
    {
        "name": "get_xas_data",
        "description": "Get X-ray absorption spectra (XANES/K-edge)",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "spectrum_type": {"type": "string", "default": "XANES"}
            },
            "required": ["material_id"]
        }
    },
    
    # Battery & Electrochemistry
    {
        "name": "get_battery_data",
        "description": "Get battery-specific properties (voltage, capacity, etc.)",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    },
    
    # Discovery & Search
    {
        "name": "discover_mp_materials",
        "description": "Find materials by elements, band gap, crystal system, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "elements": {"type": "array", "items": {"type": "string"}},
                "nelements": {"type": "integer"},
                "crystal_system": {"type": "string"},
                "band_gap_min": {"type": "number"},
                "band_gap_max": {"type": "number"},
                "max_energy_above_hull": {"type": "number"}
            },
            "required": ["elements"]
        }
    },
    {
        "name": "query_mp",
        "description": "Advanced query with custom criteria and property selection",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {"type": "object"},
                "properties": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["criteria"]
        }
    },
    
    # Chemical Analysis
    {
        "name": "get_oxidation_states",
        "description": "Get computed oxidation states for elements in material",
        "parameters": {
            "type": "object",
            "properties": {"material_id": {"type": "string"}},
            "required": ["material_id"]
        }
    }
]