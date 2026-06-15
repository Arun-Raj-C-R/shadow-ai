"""
Physical Constants and Unit Conversions for DFT
================================================

All internal calculations use ATOMIC UNITS (Hartree atomic units):
    - Length: bohr (a₀ = 0.529177 Å)
    - Energy: hartree (E_h = 27.2114 eV)
    - Mass: electron mass (mₑ)
    - Charge: elementary charge (e)
    - ℏ = mₑ = e = 4πε₀ = 1

In atomic units, the Schrödinger equation simplifies:
    [-½∇² + V(r)] ψ(r) = E ψ(r)

where the factor ℏ²/2mₑ becomes simply ½.

NIST 2018 CODATA values used throughout.
"""

import numpy as np

# =====================================================================
# FUNDAMENTAL CONSTANTS (SI)
# =====================================================================

PLANCK_CONSTANT = 6.62607015e-34       # J·s (exact, SI redefinition 2019)
HBAR = 1.054571817e-34                 # J·s  (ℏ = h/2π)
ELECTRON_MASS = 9.1093837015e-31       # kg
ELEMENTARY_CHARGE = 1.602176634e-19    # C (exact)
BOHR_RADIUS = 5.29177210903e-11        # m
HARTREE_ENERGY = 4.3597447222071e-18   # J
SPEED_OF_LIGHT = 299792458.0           # m/s (exact)
AVOGADRO = 6.02214076e23              # mol⁻¹ (exact)
BOLTZMANN = 1.380649e-23              # J/K (exact)
VACUUM_PERMITTIVITY = 8.8541878128e-12 # F/m
FINE_STRUCTURE = 7.2973525693e-3       # dimensionless (α)

# =====================================================================
# UNIT CONVERSIONS
# =====================================================================

# Length conversions
BOHR_TO_ANGSTROM = 0.529177210903     # 1 bohr = 0.529177 Å
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM  # 1 Å = 1.8897 bohr

# Energy conversions
HARTREE_TO_EV = 27.211386245988       # 1 Hartree = 27.211 eV
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
HARTREE_TO_KCAL = 627.5094740631      # 1 Hartree = 627.5 kcal/mol
HARTREE_TO_KJ = 2625.4996394799      # 1 Hartree = 2625.5 kJ/mol
HARTREE_TO_CM = 219474.63136320      # 1 Hartree = 219474.6 cm⁻¹
HARTREE_TO_K = 315775.02480407       # 1 Hartree = 315775 K
RYDBERG_TO_EV = HARTREE_TO_EV / 2.0  # 1 Ry = 13.606 eV

# =====================================================================
# MATHEMATICAL CONSTANTS
# =====================================================================

PI = np.pi
TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
SQRT_PI = np.sqrt(np.pi)
PI_POWER_3_2 = np.pi ** 1.5           # π^(3/2) — appears in Gaussian integrals

# =====================================================================
# ATOMIC DATA
# =====================================================================
# Nuclear charges (atomic numbers) -- Z=1 through Z=54
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38,
    'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
}

# Atomic masses in amu
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.066, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62,
    'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07,
    'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411,
    'In': 114.818, 'Sn': 118.710, 'Sb': 121.760, 'Te': 127.60, 'I': 126.904, 'Xe': 131.293,
}

# Covalent radii in Angstroms
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
}


def convert_energy(value, from_unit, to_unit):
    """
    Convert energy between units.
    
    Supported units: 'hartree', 'eV', 'kcal/mol', 'kJ/mol', 'cm-1', 'K', 'Ry'
    
    Implementation:
        First convert to Hartree (internal unit), then to target unit.
    """
    # Convert to Hartree
    to_hartree = {
        'hartree': 1.0,
        'eV': EV_TO_HARTREE,
        'kcal/mol': 1.0 / HARTREE_TO_KCAL,
        'kJ/mol': 1.0 / HARTREE_TO_KJ,
        'cm-1': 1.0 / HARTREE_TO_CM,
        'K': 1.0 / HARTREE_TO_K,
        'Ry': 0.5,  # 1 Ry = 0.5 Hartree
    }
    from_hartree = {
        'hartree': 1.0,
        'eV': HARTREE_TO_EV,
        'kcal/mol': HARTREE_TO_KCAL,
        'kJ/mol': HARTREE_TO_KJ,
        'cm-1': HARTREE_TO_CM,
        'K': HARTREE_TO_K,
        'Ry': 2.0,
    }
    
    if from_unit not in to_hartree or to_unit not in from_hartree:
        raise ValueError(f"Unknown unit. Supported: {list(to_hartree.keys())}")
    
    hartree_val = value * to_hartree[from_unit]
    return hartree_val * from_hartree[to_unit]


def convert_length(value, from_unit, to_unit):
    """Convert length between bohr and Angstrom."""
    if from_unit == to_unit:
        return value
    if from_unit == 'angstrom' and to_unit == 'bohr':
        return value * ANGSTROM_TO_BOHR
    if from_unit == 'bohr' and to_unit == 'angstrom':
        return value * BOHR_TO_ANGSTROM
    raise ValueError(f"Unknown units: {from_unit}, {to_unit}")
