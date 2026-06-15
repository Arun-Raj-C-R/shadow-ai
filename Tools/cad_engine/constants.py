"""
CAD Engine Constants & Material Properties
============================================

Tolerances, unit systems, material data, and physical constants
used throughout the CAD kernel.

All internal geometry uses millimeters (mm) as the base length unit,
consistent with ISO standards and mechanical CAD conventions.
"""

import numpy as np

# =====================================================================
# NUMERICAL TOLERANCES
# =====================================================================
# These are the backbone of robust CAD — without them, geometry explodes.

EPSILON = 1e-10              # General floating-point comparison tolerance
LINEAR_TOLERANCE = 1e-6      # mm — positional coincidence check
ANGULAR_TOLERANCE = 1e-8     # radians — angular coincidence check
PARAMETRIC_TOLERANCE = 1e-9  # NURBS parameter space tolerance
AREA_TOLERANCE = 1e-12       # mm² — degenerate face detection
VOLUME_TOLERANCE = 1e-15     # mm³ — degenerate solid detection
CURVATURE_TOLERANCE = 1e-10  # 1/mm — curvature comparison

# Boolean operation tolerances
BOOLEAN_TOLERANCE = 1e-6     # mm — intersection tolerance for CSG
MERGE_TOLERANCE = 1e-5       # mm — vertex merging after booleans

# Constraint solver
CONSTRAINT_TOLERANCE = 1e-8  # Constraint satisfaction threshold
MAX_SOLVER_ITERATIONS = 200  # Newton-Raphson iteration cap
DAMPING_FACTOR = 0.5         # Under-relaxation for stability

# =====================================================================
# UNIT CONVERSIONS
# =====================================================================

MM_TO_M = 1e-3
M_TO_MM = 1e3
MM_TO_INCH = 1.0 / 25.4
INCH_TO_MM = 25.4
MM_TO_MIL = 1000.0 / 25.4
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# =====================================================================
# MATHEMATICAL CONSTANTS
# =====================================================================

PI = np.pi
TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi
SQRT_2 = np.sqrt(2.0)
SQRT_3 = np.sqrt(3.0)
GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

# =====================================================================
# STANDARD AXES & PLANES
# =====================================================================

ORIGIN = np.array([0.0, 0.0, 0.0])
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

# Standard reference planes (point + normal)
XY_PLANE = {'point': ORIGIN.copy(), 'normal': Z_AXIS.copy()}
XZ_PLANE = {'point': ORIGIN.copy(), 'normal': Y_AXIS.copy()}
YZ_PLANE = {'point': ORIGIN.copy(), 'normal': X_AXIS.copy()}

# =====================================================================
# MATERIAL PROPERTIES DATABASE
# =====================================================================
# Young's modulus (GPa), Poisson's ratio, density (kg/m³),
# yield strength (MPa), thermal conductivity (W/m·K)

MATERIALS = {
    'steel_1018': {
        'name': 'AISI 1018 Steel',
        'E_GPa': 205.0,
        'poisson': 0.29,
        'density_kg_m3': 7870.0,
        'yield_MPa': 370.0,
        'thermal_conductivity': 51.9,
        'color': [0.7, 0.7, 0.75],
    },
    'steel_304': {
        'name': '304 Stainless Steel',
        'E_GPa': 193.0,
        'poisson': 0.29,
        'density_kg_m3': 8000.0,
        'yield_MPa': 215.0,
        'thermal_conductivity': 16.2,
        'color': [0.75, 0.75, 0.8],
    },
    'aluminum_6061': {
        'name': 'Aluminum 6061-T6',
        'E_GPa': 68.9,
        'poisson': 0.33,
        'density_kg_m3': 2710.0,
        'yield_MPa': 276.0,
        'thermal_conductivity': 167.0,
        'color': [0.85, 0.85, 0.88],
    },
    'titanium_6al4v': {
        'name': 'Ti-6Al-4V',
        'E_GPa': 113.8,
        'poisson': 0.342,
        'density_kg_m3': 4430.0,
        'yield_MPa': 880.0,
        'thermal_conductivity': 6.7,
        'color': [0.6, 0.6, 0.65],
    },
    'copper': {
        'name': 'Copper C11000',
        'E_GPa': 117.0,
        'poisson': 0.34,
        'density_kg_m3': 8960.0,
        'yield_MPa': 69.0,
        'thermal_conductivity': 385.0,
        'color': [0.85, 0.55, 0.3],
    },
    'abs_plastic': {
        'name': 'ABS Plastic',
        'E_GPa': 2.3,
        'poisson': 0.35,
        'density_kg_m3': 1070.0,
        'yield_MPa': 43.0,
        'thermal_conductivity': 0.17,
        'color': [0.95, 0.95, 0.9],
    },
    'nylon_66': {
        'name': 'Nylon 6/6',
        'E_GPa': 3.3,
        'poisson': 0.39,
        'density_kg_m3': 1140.0,
        'yield_MPa': 83.0,
        'thermal_conductivity': 0.25,
        'color': [0.9, 0.88, 0.8],
    },
    'carbon_fiber': {
        'name': 'Carbon Fiber (Epoxy)',
        'E_GPa': 181.0,
        'poisson': 0.28,
        'density_kg_m3': 1600.0,
        'yield_MPa': 1500.0,
        'thermal_conductivity': 7.0,
        'color': [0.15, 0.15, 0.18],
    },
}


def get_material(name: str) -> dict:
    """Look up a material by key. Returns material dict or raises KeyError."""
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key in MATERIALS:
        return MATERIALS[key]
    # Fuzzy match
    for k, v in MATERIALS.items():
        if key in k or key in v['name'].lower():
            return v
    raise KeyError(f"Unknown material: '{name}'. Available: {list(MATERIALS.keys())}")
