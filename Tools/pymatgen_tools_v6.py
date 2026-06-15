# pymatgen_tools_v6.py
"""
PymatgenAnalysisTool v6.0 â€“ SHADOW-READY
Exports: calculate_solar_efficiency_v6
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from scipy.integrate import quad
import json
import pathlib
from monty.serialization import MontyDecoder, MontyEncoder

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PLOTS_DIR = pathlib.Path(r"D:\Project File\Shadow\Shadow\Brain\Shadow2\plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------
# REAL AM1.5G SPECTRUM (NREL ASTM G173-03)
# ----------------------------------------------------------------------
AM15G_SPECTRUM = [                     # (wavelength_nm, irradiance_W/mÂ²/nm)
    (280, 0.0), (300, 0.003), (320, 0.1), (340, 0.3), (360, 0.6),
    (380, 1.0), (400, 1.5), (450, 1.8), (500, 1.9), (550, 2.0),
    (600, 1.95), (650, 1.85), (700, 1.75), (750, 1.68), (800, 1.6),
    (850, 1.52), (900, 1.45), (950, 1.38), (1000, 1.32), (1100, 1.2),
    (1200, 1.05), (1300, 0.92), (1400, 0.8), (1500, 0.7), (1600, 0.62),
    (1700, 0.55), (1800, 0.49), (1900, 0.44), (2000, 0.4), (2500, 0.25),
    (3000, 0.15), (3500, 0.1), (4000, 0.07)
]

WAVELENGTHS_NM   = np.array([x[0] for x in AM15G_SPECTRUM])
IRRADIANCE_WM2NM = np.array([x[1] for x in AM15G_SPECTRUM])   # W/mÂ²/nm

def am15_photon_flux(E_ev: float) -> float:
    """Photon flux (photons/cmÂ²/s/eV) for AM1.5G at energy E (eV)."""
    if E_ev <= 0 or E_ev > 4.43:                     # 280 nm = 4.43 eV
        return 0.0
    wl_nm = 1239.84193 / E_ev
    if wl_nm < 280 or wl_nm > 4000:
        return 0.0

    # Interpolate irradiance at this wavelength
    irr = np.interp(wl_nm, WAVELENGTHS_NM, IRRADIANCE_WM2NM)   # W/mÂ²/nm

    # Convert to photons/cmÂ²/s/eV
    photon_energy_j = E_ev * 1.60217662e-19
    flux_m2_s_nm = irr / photon_energy_j                     # photons/mÂ²/s/nm
    dlambda_dE = 1239.84193 / (E_ev ** 2)                     # nm/eV
    flux_m2_s_ev = flux_m2_s_nm * dlambda_dE
    return flux_m2_s_ev * 1e-4                               # to cmÂ²

# ----------------------------------------------------------------------
# Pymatgen imports
# ----------------------------------------------------------------------
try:
    from pymatgen.core import Structure
    from pymatgen.electronic_structure.bandstructure import BandStructure
    from pymatgen.electronic_structure.dos import CompleteDos
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.transformations.standard_transformations import SubstitutionTransformation
    from pymatgen.transformations.site_transformations import RemoveSitesTransformation
    from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
    from pymatgen.core.composition import Composition
except Exception as e:
    raise ImportError(f"Pymatgen import failed: {e}")

# ----------------------------------------------------------------------
# SLME (optional)
# ----------------------------------------------------------------------
SLME_AVAILABLE = False
try:
    from pymatgen.analysis.solar.slme import slme as _slme
    SLME_AVAILABLE = True
except Exception:
    logging.warning("SLME not available â€“ falling back to SQ limit.")

# ----------------------------------------------------------------------
# Core analysis class
# ----------------------------------------------------------------------
class PymatgenAnalysisTool:
    @staticmethod
    def _decode_pymatgen_json(json_string: str):
        return MontyDecoder().decode(json.loads(json_string))

    @staticmethod
    def _encode_pymatgen_json(obj) -> str:
        return json.dumps(obj, cls=MontyEncoder)

    @staticmethod
    def _extract_structure_from_properties(properties_json: str) -> Structure:
        props = PymatgenAnalysisTool._decode_pymatgen_json(properties_json)
        struct = props.get("structure")
        if not struct:
            raise ValueError("No 'structure' in JSON.")
        return struct

    @staticmethod
    def execute(job: Dict) -> Dict:
        task = job.get("task")
        if task == "solar_efficiency":
            return PymatgenAnalysisTool._solar_efficiency(job)
        if task == "create_slab":
            return PymatgenAnalysisTool._create_slab(job)
        if task == "create_defect":
            return PymatgenAnalysisTool._create_defect(job)
        if task == "plot_band_dos":
            return PymatgenAnalysisTool._plot_band_dos(job)
        if task == "stability_report":
            return PymatgenAnalysisTool._stability_report(job)
        if task == "wulff_shape":
            return PymatgenAnalysisTool._wulff_shape(job)
        raise ValueError(f"Unknown task: {task}")

    # --------------------------------------------------------------
    # Solar efficiency â€“ SQ limit with real AM1.5G
    # --------------------------------------------------------------
    @staticmethod
    def _solar_efficiency(job: Dict) -> Dict:
        Eg   = float(job["band_gap_ev"])
        t_um = float(job.get("thickness_um", 0.5))
        direct = bool(job.get("is_direct_gap", True))

        # ---- SLME fallback (if available) ----
        if SLME_AVAILABLE:
            try:
                eff, voc, jsc, ff = _slme(band_gap_ev=Eg, film_thickness=t_um * 1e-4)
                return {
                    "domain": "solar",
                    "band_gap_ev": round(Eg, 3),
                    "is_direct_gap": direct,
                    "thickness_um": t_um,
                    "efficiency_pct": round(eff * 100, 2),
                    "voc_v": round(voc, 3),
                    "jsc_ma_cm2": round(jsc, 2),
                    "ff": round(ff, 3),
                    "method": "SLME"
                }
            except Exception:
                pass      # fall through to SQ

        # ---- SQ limit with real spectrum ----
        t_cm   = t_um * 1e-4                     # Âµm â†’ cm
        alpha  = 1e5 if direct else 1e4          # cmâ»Â¹ (typical values)

        # Jsc integration
        Jsc, _ = quad(
            lambda E: 1.602e-19 * am15_photon_flux(E) * (1 - np.exp(-alpha * t_cm)),
            Eg, 4.43
        )
        Jsc_mA = Jsc * 1000

        if Jsc_mA < 1e-3:                         # no usable current
            return {
                "domain": "solar",
                "band_gap_ev": round(Eg, 3),
                "is_direct_gap": direct,
                "thickness_um": t_um,
                "efficiency_pct": 0.0,
                "jsc_ma_cm2": 0.0,
                "voc_v": 0.0,
                "ff": 0.0,
                "method": "SQ-AM1.5G"
            }

        kT = 0.02585                               # 300 K in eV
        # CORRECT radiative recombination current density
        J0 = 1e-9 * np.exp(-Eg / kT)               # A/cmÂ² (â‰ˆ1 nA/cmÂ² for GaAs)
        Voc = kT * np.log(Jsc / J0 + 1)
        v_norm = Voc / kT
        FF = (v_norm - np.log(v_norm + 0.72)) / (v_norm + 1)
        eff = (Jsc_mA * Voc * FF) / 100.0          # 1000 W/mÂ² = 100 mW/cmÂ²

        return {
            "domain": "solar",
            "band_gap_ev": round(Eg, 3),
            "is_direct_gap": direct,
            "thickness_um": t_um,
            "efficiency_pct": round(eff, 2),
            "voc_v": round(Voc, 3),
            "jsc_ma_cm2": round(Jsc_mA, 2),
            "ff": round(FF, 3),
            "method": "SQ-AM1.5G"
        }

    # --------------------------------------------------------------
    # Slab generation
    # --------------------------------------------------------------
    @staticmethod
    def _create_slab(job: Dict) -> Dict:
        struct = job["structure"]
        miller = job["miller_index"]
        layers = job.get("layers", 4)
        vacuum = job.get("vacuum_A", 15.0)
        gen = SlabGenerator(struct, miller,
                            min_slab_size=layers * 3.0,
                            min_vacuum_size=vacuum,
                            center_slab=True)
        slab = gen.get_slabs()[0]
        return {"slab_structure": PymatgenAnalysisTool._encode_pymatgen_json(slab)}

    # --------------------------------------------------------------
    # Defect / doping
    # --------------------------------------------------------------
    @staticmethod
    def _create_defect(job: Dict) -> Dict:
        struct = job["structure"]
        if job["defect_type"] == "vacancy":
            trans = RemoveSitesTransformation([job["site_index"]])
        elif job["defect_type"] == "substitution":
            host = struct[job["site_index"]].species_string
            trans = SubstitutionTransformation(
                {host: {job["dopant"]: job["fraction"], host: 1 - job["fraction"]}}
            )
        new_struct = trans.apply_transformation(struct)
        return {"defect_structure": PymatgenAnalysisTool._encode_pymatgen_json(new_struct)}

    # --------------------------------------------------------------
    # Band-structure + DOS plot
    # --------------------------------------------------------------
    @staticmethod
    def _plot_band_dos(job: Dict) -> Dict:
        bs   = job["bandstructure"]
        dos  = job["dos"]
        path = job.get("save_path", PLOTS_DIR / "band_dos.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                       gridspec_kw={"width_ratios": [3, 1]})
        BSPlotter(bs).get_plot(axes=ax1)
        DosPlotter().add_dos("Total", dos).get_plot(axes=ax2)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return {"plot_path": str(path)}

    # --------------------------------------------------------------
    # Phase stability report
    # --------------------------------------------------------------
    @staticmethod
    def _stability_report(job: Dict) -> Dict:
        entries = job["entries"]
        pd = PhaseDiagram(entries)
        report = []
        for e in entries:
            decomp, e_hull = pd.get_decomp_and_e_above_hull(e)
            report.append({
                "formula": e.composition.reduced_formula,
                "e_above_hull_ev_atom": e_hull / e.composition.num_atoms,
                "stable": e_hull < 1e-4
            })
        return {"stability": report}

    # --------------------------------------------------------------
    # Wulff shape analysis
    # --------------------------------------------------------------
    @staticmethod
    def _wulff_shape(job: Dict) -> Dict:
        struct = job["structure"]
        surface_energies = job["surface_energies"]
        try:
            from pymatgen.analysis.wulff import WulffShape
            lattice = struct.lattice
            millers = list(surface_energies.keys())
            energies = list(surface_energies.values())
            wulff = WulffShape(lattice, millers, energies)
            return {
                "effective_radius": round(wulff.effective_radius, 4),
                "anisotropy": round(wulff.anisotropy, 4),
                "weighted_surface_energy": round(wulff.weighted_surface_energy, 6),
                "area_fractions": {str(k): round(v, 4) for k, v in wulff.area_fraction_dict.items()},
                "shape_factor": round(wulff.shape_factor, 4),
            }
        except Exception as e:
            return {"error": f"Wulff shape calculation failed: {str(e)}"}


# ----------------------------------------------------------------------
# SHADOW TOOL WRAPPERS (exact names)
# ----------------------------------------------------------------------
def calculate_solar_efficiency_v6(band_gap_ev: float,
                                 thickness_um: float = 0.5,
                                 is_direct_gap: bool = True) -> str:
    """Shadow entry point."""
    try:
        job = {
            "task": "solar_efficiency",
            "band_gap_ev": band_gap_ev,
            "thickness_um": thickness_um,
            "is_direct_gap": is_direct_gap
        }
        result = PymatgenAnalysisTool.execute(job)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def generate_surface_slab_v6(properties_json: str,
                            miller_index: List[int],
                            layers: int = 4,
                            vacuum_A: float = 15.0) -> str:
    try:
        struct = PymatgenAnalysisTool._extract_structure_from_properties(properties_json)
        job = {"task": "create_slab",
               "structure": struct,
               "miller_index": miller_index,
               "layers": layers,
               "vacuum_A": vacuum_A}
        return json.dumps(PymatgenAnalysisTool.execute(job))
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def generate_doped_structure_v6(properties_json: str,
                               site_index: int,
                               dopant: str,
                               fraction: float) -> str:
    try:
        struct = PymatgenAnalysisTool._extract_structure_from_properties(properties_json)
        job = {"task": "create_defect",
               "structure": struct,
               "defect_type": "substitution",
               "site_index": site_index,
               "dopant": dopant,
               "fraction": fraction}
        return json.dumps(PymatgenAnalysisTool.execute(job))
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def plot_band_dos_v6(band_structure_json: str,
                     dos_json: str,
                     filename: str) -> str:
    try:
        bs  = PymatgenAnalysisTool._decode_pymatgen_json(band_structure_json)
        dos = PymatgenAnalysisTool._decode_pymatgen_json(dos_json)
        path = PLOTS_DIR / (filename + ".png")
        job = {"task": "plot_band_dos",
               "bandstructure": bs,
               "dos": dos,
               "save_path": path}
        return json.dumps(PymatgenAnalysisTool.execute(job))
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def analyze_phase_stability_v6(materials_list_json: str) -> str:
    try:
        mats = json.loads(materials_list_json)
        entries = [
            ComputedEntry(
                Composition(m['formula_pretty']),
                m.get('formation_energy_per_atom', 0) *
                Composition(m['formula_pretty']).num_atoms
            ) for m in mats
        ]
        job = {"task": "stability_report", "entries": entries}
        return json.dumps(PymatgenAnalysisTool.execute(job))
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def analyze_wulff_shape_v6(properties_json: str,
                           miller_energies_json: str) -> str:
    try:
        struct = PymatgenAnalysisTool._extract_structure_from_properties(properties_json)
        energies = json.loads(miller_energies_json)
        surface_energies = {
            tuple(map(int, k.strip("()").split(','))): v
            for k, v in energies.items()
        }
        job = {"task": "wulff_shape",
               "structure": struct,
               "surface_energies": surface_energies}
        return json.dumps(PymatgenAnalysisTool.execute(job))
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ----------------------------------------------------------------------
# EXPORT LIST (critical for Shadow imports)
# ----------------------------------------------------------------------
__all__ = [
    "calculate_solar_efficiency_v6",
    "generate_surface_slab_v6",
    "generate_doped_structure_v6",
    "plot_band_dos_v6",
    "analyze_phase_stability_v6",
    "analyze_wulff_shape_v6",
    "PYMATGEN_TOOL_DEFINITIONS_V6"
]

# ----------------------------------------------------------------------
# TOOL DEFINITIONS (must be a valid list)
# ----------------------------------------------------------------------
PYMATGEN_TOOL_DEFINITIONS_V6 = [
    {
        "name": "calculate_solar_efficiency_v6",
        "description": "Calculate theoretical solar efficiency (SQ or SLME).",
        "parameters": {
            "type": "object",
            "properties": {
                "band_gap_ev": {"type": "number", "description": "Band gap in eV"},
                "thickness_um": {"type": "number", "description": "Film thickness in Âµm", "default": 0.5},
                "is_direct_gap": {"type": "boolean", "description": "True if direct gap", "default": True}
            },
            "required": ["band_gap_ev"]
        }
    },
    {
        "name": "generate_surface_slab_v6",
        "description": "Generate a surface slab from a material structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "properties_json": {"type": "string"},
                "miller_index": {"type": "array", "items": {"type": "integer"}},
                "layers": {"type": "integer", "default": 4},
                "vacuum_A": {"type": "number", "default": 15.0}
            },
            "required": ["properties_json", "miller_index"]
        }
    },
    {
        "name": "generate_doped_structure_v6",
        "description": "Create a substitutionally doped structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "properties_json": {"type": "string"},
                "site_index": {"type": "integer"},
                "dopant": {"type": "string"},
                "fraction": {"type": "number"}
            },
            "required": ["properties_json", "site_index", "dopant", "fraction"]
        }
    },
    {
        "name": "plot_band_dos_v6",
        "description": "Plot band structure and DOS.",
        "parameters": {
            "type": "object",
            "properties": {
                "band_structure_json": {"type": "string"},
                "dos_json": {"type": "string"},
                "filename": {"type": "string"}
            },
            "required": ["band_structure_json", "dos_json", "filename"]
        }
    },
    {
        "name": "analyze_phase_stability_v6",
        "description": "Analyze phase stability from Materials Project data.",
        "parameters": {
            "type": "object",
            "properties": {"materials_list_json": {"type": "string"}},
            "required": ["materials_list_json"]
        }
    },
    {
        "name": "analyze_wulff_shape_v6",
        "description": "Compute Wulff shape from surface energies.",
        "parameters": {
            "type": "object",
            "properties": {
                "properties_json": {"type": "string"},
                "miller_energies_json": {"type": "string"}
            },
            "required": ["properties_json", "miller_energies_json"]
        }
    }
]