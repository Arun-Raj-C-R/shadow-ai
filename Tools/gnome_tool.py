"""
GNoME Materials Discovery Tool â€” Google DeepMind (Nature 2023)
Wraps the materials_discovery repo for use in the SHADOW/Shadow AI ecosystem.

Capabilities:
  - Search 520,000+ novel stable materials from the GNoME database
  - Lookup by formula, composition, elements, crystal system, bandgap
  - Compute decomposition energies against the updated convex hull
  - Explore chemical systems for stable candidates
  - Download and visualize GNoME crystal structures (CIF)
  - Hybrid workflows with Materials Project + DFT + MD tools
"""

import os
import sys
import json
import traceback
import subprocess
from typing import Optional

# â”€â”€ Paths â”€â”€
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = os.path.dirname(_THIS_DIR)
_GNOME_DIR = os.path.join(_CLI_DIR, "materials_discovery")
_DATA_DIR = os.path.join(_GNOME_DIR, "data", "gnome_data")
_SUMMARY_CSV = os.path.join(_DATA_DIR, "stable_materials_summary.csv")
_R2SCAN_CSV = os.path.join(_DATA_DIR, "stable_materials_r2scan.csv")
_EXTERNAL_CSV = os.path.join(_DATA_DIR, "..", "external_data", "external_materials_summary.csv")

# Lazy-loaded DataFrame cache
_df_cache = {}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# INTERNAL HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _ensure_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise RuntimeError("pandas is required. Install: pip install pandas")


def _load_summary_df():
    """Load the main GNoME summary CSV (lazy, cached)."""
    if "summary" in _df_cache:
        return _df_cache["summary"]
    pd = _ensure_pandas()
    if not os.path.exists(_SUMMARY_CSV):
        raise FileNotFoundError(
            f"GNoME dataset not found at {_SUMMARY_CSV}. "
            "Run the data download first: python materials_discovery/scripts/download_data_wget.py"
        )
    df = pd.read_csv(_SUMMARY_CSV)
    _df_cache["summary"] = df
    return df


def _safe_float(val):
    try:
        import math
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return None


def _row_to_dict(row):
    """Convert a DataFrame row to a clean JSON-serializable dict."""
    d = {}
    for col in row.index:
        val = row[col]
        if hasattr(val, 'item'):
            val = val.item()
        if isinstance(val, float):
            val = _safe_float(val)
        d[col] = val
    return d


def _dataset_available():
    return os.path.exists(_SUMMARY_CSV)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CORE FUNCTIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def gnome_status() -> str:
    """Check if GNoME dataset is downloaded and ready."""
    info = {
        "gnome_repo": os.path.exists(_GNOME_DIR),
        "dataset_downloaded": _dataset_available(),
        "summary_csv": _SUMMARY_CSV if _dataset_available() else "NOT FOUND",
    }
    if _dataset_available():
        pd = _ensure_pandas()
        df = _load_summary_df()
        info["total_materials"] = len(df)
        info["columns"] = list(df.columns)
    return json.dumps(info, indent=2)


def gnome_download_data() -> str:
    """Download the GNoME dataset from Google Cloud Storage."""
    script = os.path.join(_GNOME_DIR, "scripts", "download_data_wget.py")
    if not os.path.exists(script):
        return "ERROR: download script not found at " + script
    data_dir = os.path.join(_GNOME_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, script, f"--data_dir={data_dir}"],
            capture_output=True, text=True, timeout=600, cwd=_GNOME_DIR
        )
        if result.returncode == 0:
            return f"Dataset downloaded successfully to {data_dir}\n{result.stdout[-500:]}"
        return f"Download failed:\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return "Download timed out (10 min). Try manually: python scripts/download_data_wget.py"
    except Exception as e:
        return f"Download error: {e}"


def _exact_element_match(elements_col, element: str):
    """Match exact element symbol in the Elements column.
    The column contains strings like "['Fe', 'O', 'Li']".
    Using word-boundary regex to avoid 'S' matching 'Sb', 'Si', 'Sr', etc.
    """
    import re
    # Match the exact element surrounded by non-alpha chars (quotes, brackets, commas)
    pattern = r"(?<![A-Za-z])" + re.escape(element) + r"(?![a-z])"
    return elements_col.astype(str).str.contains(pattern, regex=True, na=False)


def gnome_search(
    formula: Optional[str] = None,
    elements: Optional[list] = None,
    exact_system: bool = False,
    crystal_system: Optional[str] = None,
    space_group: Optional[str] = None,
    bandgap_min: Optional[float] = None,
    bandgap_max: Optional[float] = None,
    formation_energy_max: Optional[float] = None,
    decomposition_energy_max: Optional[float] = None,
    max_results: int = 20,
) -> str:
    """Search the GNoME database of 520K+ novel stable materials."""
    if not _dataset_available():
        return json.dumps({"error": "Dataset not downloaded. Call gnome_download_data first."})

    pd = _ensure_pandas()
    df = _load_summary_df()
    mask = pd.Series([True] * len(df), index=df.index)
    filters_applied = []

    # Filter by formula (exact or partial match)
    if formula:
        formula_lower = formula.lower().strip()
        if "Reduced Formula" in df.columns:
            mask &= df["Reduced Formula"].astype(str).str.lower().str.contains(formula_lower, na=False)
        elif "Composition" in df.columns:
            mask &= df["Composition"].astype(str).str.lower().str.contains(formula_lower, na=False)
        filters_applied.append(f"formula_contains={formula}")

    # Filter by elements â€” EXACT element symbol matching (fixes Issue #7)
    # 'S' will NOT match 'Sb', 'Si', 'Sr', etc.
    if elements:
        if "Elements" in df.columns:
            for el in elements:
                mask &= _exact_element_match(df["Elements"], el)
            filters_applied.append(f"contains_elements={elements}")

            # If exact_system=True, exclude materials with extra elements
            if exact_system:
                def _count_elements(s):
                    try:
                        return len(eval(s)) if isinstance(s, str) and s.startswith('[') else 99
                    except:
                        return 99
                el_counts = df["Elements"].apply(_count_elements)
                mask &= el_counts == len(elements)
                filters_applied.append(f"exact_system={'-'.join(elements)}")

    # Filter by crystal system
    if crystal_system and "Crystal System" in df.columns:
        mask &= df["Crystal System"].astype(str).str.lower() == crystal_system.lower()
        filters_applied.append(f"crystal_system={crystal_system}")

    # Filter by space group
    if space_group and "Space Group" in df.columns:
        mask &= df["Space Group"].astype(str).str.lower() == space_group.lower()
        filters_applied.append(f"space_group={space_group}")

    # Filter by bandgap range
    if bandgap_min is not None and "Bandgap" in df.columns:
        mask &= pd.to_numeric(df["Bandgap"], errors="coerce") >= bandgap_min
        filters_applied.append(f"bandgap>={bandgap_min}")
    if bandgap_max is not None and "Bandgap" in df.columns:
        mask &= pd.to_numeric(df["Bandgap"], errors="coerce") <= bandgap_max
        filters_applied.append(f"bandgap<={bandgap_max}")

    # Filter by formation energy
    if formation_energy_max is not None and "Formation Energy Per Atom" in df.columns:
        mask &= pd.to_numeric(df["Formation Energy Per Atom"], errors="coerce") <= formation_energy_max
        filters_applied.append(f"formation_energy<={formation_energy_max}")

    # Filter by decomposition energy (stability proxy)
    if decomposition_energy_max is not None and "Decomposition Energy Per Atom" in df.columns:
        mask &= pd.to_numeric(df["Decomposition Energy Per Atom"], errors="coerce") <= decomposition_energy_max
        filters_applied.append(f"decomposition_energy<={decomposition_energy_max}")

    results = df[mask].head(max_results)
    if results.empty:
        return json.dumps({
            "matches": 0, "results": [], "filters_applied": filters_applied,
            "hint": "Try broader filters or fewer element constraints.",
            "provenance": "GNoME database (Google DeepMind, Nature 2023) â€” DFT-computed, NOT experimentally verified",
        })

    out = []
    for _, row in results.iterrows():
        out.append(_row_to_dict(row))

    return json.dumps({
        "matches": int(mask.sum()),
        "showing": len(out),
        "filters_applied": filters_applied,
        "confidence": "DATA_RETRIEVED â€” values are DFT-computed predictions, not experimental measurements",
        "provenance": "GNoME database (Google DeepMind, Nature 2023)",
        "results": out,
    }, indent=2, default=str)


def gnome_get_material(material_id: str) -> str:
    """Get full details of a specific GNoME material by its MaterialId."""
    if not _dataset_available():
        return json.dumps({"error": "Dataset not downloaded."})

    pd = _ensure_pandas()
    df = _load_summary_df()

    if "MaterialId" not in df.columns:
        return json.dumps({"error": "MaterialId column not found in dataset"})

    match = df[df["MaterialId"].astype(str) == str(material_id)]
    if match.empty:
        return json.dumps({"error": f"Material '{material_id}' not found"})

    return json.dumps(_row_to_dict(match.iloc[0]), indent=2, default=str)


def gnome_statistics(elements: Optional[list] = None) -> str:
    """Get statistical overview of the GNoME database (or a filtered subset)."""
    if not _dataset_available():
        return json.dumps({"error": "Dataset not downloaded."})

    pd = _ensure_pandas()
    df = _load_summary_df()

    if elements:
        mask = pd.Series([True] * len(df), index=df.index)
        if "Elements" in df.columns:
            for el in elements:
                mask &= df["Elements"].astype(str).str.contains(el, case=False, na=False)
        df = df[mask]

    stats = {"total_materials": len(df)}

    # Crystal system distribution
    if "Crystal System" in df.columns:
        stats["crystal_systems"] = df["Crystal System"].value_counts().head(10).to_dict()

    # Bandgap stats
    if "Bandgap" in df.columns:
        import numpy as np
        bg = pd.to_numeric(df["Bandgap"], errors="coerce").dropna()
        bg = bg[np.isfinite(bg)]  # Remove inf values
        if len(bg) > 0:
            stats["bandgap"] = {
                "min": round(float(bg.min()), 4),
                "max": round(float(bg.max()), 4),
                "mean": round(float(bg.mean()), 4),
                "median": round(float(bg.median()), 4),
                "count_with_data": int(len(bg)),
            }

    # Formation energy stats
    if "Formation Energy Per Atom" in df.columns:
        fe = pd.to_numeric(df["Formation Energy Per Atom"], errors="coerce").dropna()
        if len(fe) > 0:
            stats["formation_energy_per_atom"] = {
                "min": round(float(fe.min()), 4),
                "max": round(float(fe.max()), 4),
                "mean": round(float(fe.mean()), 4),
            }

    # NSites distribution
    if "NSites" in df.columns:
        ns = pd.to_numeric(df["NSites"], errors="coerce").dropna()
        if len(ns) > 0:
            stats["nsites"] = {
                "min": int(ns.min()),
                "max": int(ns.max()),
                "mean": round(float(ns.mean()), 1),
            }

    return json.dumps(stats, indent=2, default=str)


def gnome_explore_system(elements: list, top_n: int = 15, exact: bool = True) -> str:
    """
    Explore a chemical system (e.g., ['Li', 'Fe', 'O']) to find
    the most stable novel materials discovered by GNoME.
    Sorted by decomposition energy (lowest = most stable).
    
    exact=True  â†’ ONLY materials containing exactly these elements (no extras)
    exact=False â†’ materials containing at least these elements (may have more)
    """
    if not _dataset_available():
        return json.dumps({"error": "Dataset not downloaded."})

    pd = _ensure_pandas()
    df = _load_summary_df()

    # Use exact element matching (fixes Issue #7)
    mask = pd.Series([True] * len(df), index=df.index)
    if "Elements" in df.columns:
        for el in elements:
            mask &= _exact_element_match(df["Elements"], el)
        
        # For exact system exploration, exclude materials with extra elements
        if exact:
            def _count_elements(s):
                try:
                    return len(eval(s)) if isinstance(s, str) and s.startswith('[') else 99
                except:
                    return 99
            el_counts = df["Elements"].apply(_count_elements)
            mask &= el_counts == len(elements)

    filtered = df[mask].copy()

    if filtered.empty:
        # If exact match found nothing, retry with contains-all
        if exact:
            return gnome_explore_system(elements, top_n, exact=False)
        return json.dumps({
            "system": "-".join(elements), "matches": 0, "results": [],
            "search_mode": "exact" if exact else "contains_all",
            "provenance": "GNoME database (Google DeepMind, Nature 2023)",
        })

    # Sort by stability
    if "Decomposition Energy Per Atom" in filtered.columns:
        filtered["_decomp"] = pd.to_numeric(filtered["Decomposition Energy Per Atom"], errors="coerce")
        filtered = filtered.sort_values("_decomp", ascending=True)

    results = []
    for _, row in filtered.head(top_n).iterrows():
        results.append(_row_to_dict(row))

    return json.dumps({
        "system": "-".join(elements),
        "search_mode": "exact_system" if exact else "contains_all_elements",
        "total_matches": int(mask.sum()),
        "showing_top": len(results),
        "sorted_by": "decomposition_energy_ascending (most stable first)",
        "ranking_method": "Decomposition Energy Per Atom â€” lower value = closer to convex hull = more thermodynamically stable",
        "confidence": "DATA_RETRIEVED â€” stability from DFT-computed convex hull, not experimental synthesis",
        "provenance": "GNoME database (Google DeepMind, Nature 2023)",
        "results": results,
    }, indent=2, default=str)


def gnome_find_for_application(application: str, max_results: int = 15) -> str:
    """
    Find GNoME materials suited for a specific application using
    heuristic bandgap/property filters.
    
    Applications: solar_cell, led, thermoelectric, battery_cathode,
                  battery_anode, topological, superconductor, catalyst,
                  wide_bandgap, semiconductor, transparent_conductor
    """
    if not _dataset_available():
        return json.dumps({"error": "Dataset not downloaded."})

    pd = _ensure_pandas()
    df = _load_summary_df()

    # Application-specific heuristic filters (convenience presets)
    # The raw 'search' task has ZERO restrictions â€” these are just shortcuts
    presets = {
        # â”€â”€ Energy & Solar â”€â”€
        "solar_cell": {"bandgap_min": 1.0, "bandgap_max": 1.8},
        "tandem_solar": {"bandgap_min": 1.6, "bandgap_max": 2.3},
        "perovskite_solar": {"bandgap_min": 1.2, "bandgap_max": 1.8, "elements": ["Pb"]},
        # â”€â”€ Lighting & Display â”€â”€
        "led": {"bandgap_min": 1.8, "bandgap_max": 3.5},
        "phosphor": {"bandgap_min": 2.5, "bandgap_max": 6.0},
        "laser": {"bandgap_min": 1.5, "bandgap_max": 4.0},
        # â”€â”€ Electronics â”€â”€
        "semiconductor": {"bandgap_min": 0.5, "bandgap_max": 3.0},
        "wide_bandgap": {"bandgap_min": 3.0, "bandgap_max": 7.0},
        "ultrawide_bandgap": {"bandgap_min": 4.5, "bandgap_max": 9.0},
        "transparent_conductor": {"bandgap_min": 3.0, "bandgap_max": 5.0},
        "topological": {"bandgap_min": 0.0, "bandgap_max": 0.3},
        "superconductor": {"bandgap_max": 0.01},
        # â”€â”€ Energy Storage â”€â”€
        "battery_cathode": {"elements": ["Li", "O"], "bandgap_max": 4.0},
        "battery_anode": {"elements": ["Li"], "bandgap_max": 1.0},
        "sodium_battery": {"elements": ["Na", "O"]},
        "solid_electrolyte": {"elements": ["Li", "O"], "bandgap_min": 3.0},
        # â”€â”€ Thermal & Energy Conversion â”€â”€
        "thermoelectric": {"bandgap_min": 0.1, "bandgap_max": 1.0},
        "pcm": {"bandgap_min": 0.1, "bandgap_max": 1.5},  # Phase-change memory
        "thermal_barrier": {"bandgap_min": 3.0, "elements": ["O", "Zr"]},
        # â”€â”€ Magnetic & Spintronic â”€â”€
        "magnetic": {"elements": ["Fe"]},
        "spintronic": {"bandgap_min": 0.0, "bandgap_max": 0.5, "elements": ["Fe"]},
        "permanent_magnet": {"elements": ["Nd", "Fe"]},
        # â”€â”€ Piezo / Ferro / Dielectric â”€â”€
        "piezoelectric": {"bandgap_min": 2.0, "bandgap_max": 6.0},
        "ferroelectric": {"bandgap_min": 2.0, "bandgap_max": 5.0},
        "dielectric": {"bandgap_min": 4.0},
        # â”€â”€ Optical & Photonic â”€â”€
        "optical": {"bandgap_min": 1.5, "bandgap_max": 5.0},
        "nonlinear_optical": {"bandgap_min": 3.0, "bandgap_max": 7.0},
        "scintillator": {"bandgap_min": 3.0, "bandgap_max": 6.0},
        "metamaterial": {"bandgap_max": 0.5},
        "photocatalyst": {"bandgap_min": 1.5, "bandgap_max": 3.2},
        # â”€â”€ Bio & Medical â”€â”€
        "biomaterial": {"elements": ["Ca", "P", "O"]},  # Hydroxyapatite-like
        "biocompatible": {"elements": ["Ti", "O"]},
        "bioceramic": {"elements": ["Ca", "Si", "O"]},
        # â”€â”€ Structural & Hard â”€â”€
        "superhard": {"formation_energy_max": -1.5},
        "catalyst": {"formation_energy_max": -0.5},
        "refractory": {"elements": ["W"], "formation_energy_max": -1.0},
        "ceramic": {"elements": ["O"], "bandgap_min": 3.0},
        # â”€â”€ Quantum â”€â”€
        "quantum_computing": {"bandgap_min": 0.0, "bandgap_max": 0.1},
        "qubit_host": {"bandgap_min": 3.5, "bandgap_max": 6.0},
    }

    app_key = application.lower().replace(" ", "_").replace("-", "_")
    if app_key not in presets:
        return json.dumps({
            "error": f"Unknown application '{application}'",
            "available": list(presets.keys()),
        })

    params = presets[app_key]
    return gnome_search(
        elements=params.get("elements"),
        bandgap_min=params.get("bandgap_min"),
        bandgap_max=params.get("bandgap_max"),
        formation_energy_max=params.get("formation_energy_max"),
        max_results=max_results,
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MASTER DISPATCHER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_gnome_tool(
    task: str = "search",
    formula: Optional[str] = None,
    elements: Optional[list] = None,
    crystal_system: Optional[str] = None,
    space_group: Optional[str] = None,
    bandgap_min: Optional[float] = None,
    bandgap_max: Optional[float] = None,
    formation_energy_max: Optional[float] = None,
    decomposition_energy_max: Optional[float] = None,
    material_id: Optional[str] = None,
    application: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """
    Master entry point for GNoME materials discovery tool.
    
    Tasks:
      status          â€” Check if dataset is ready
      download        â€” Download GNoME dataset (~2GB)
      search          â€” Search materials by formula/elements/bandgap/etc
      get_material    â€” Get full details of a material by ID
      statistics      â€” Get database statistics
      explore_system  â€” Explore a chemical system for stable materials
      find_for_app    â€” Find materials for a specific application
    """
    try:
        if task == "status":
            return gnome_status()
        elif task == "download":
            return gnome_download_data()
        elif task == "search":
            return gnome_search(
                formula=formula, elements=elements,
                crystal_system=crystal_system, space_group=space_group,
                bandgap_min=bandgap_min, bandgap_max=bandgap_max,
                formation_energy_max=formation_energy_max,
                decomposition_energy_max=decomposition_energy_max,
                max_results=max_results,
            )
        elif task == "get_material":
            return gnome_get_material(material_id or "")
        elif task == "statistics":
            return gnome_statistics(elements=elements)
        elif task == "explore_system":
            return gnome_explore_system(elements or [], top_n=max_results)
        elif task == "find_for_app":
            return gnome_find_for_application(application or "", max_results=max_results)
        else:
            return json.dumps({"error": f"Unknown task '{task}'", "available_tasks": [
                "status", "download", "search", "get_material",
                "statistics", "explore_system", "find_for_app",
            ]})
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
