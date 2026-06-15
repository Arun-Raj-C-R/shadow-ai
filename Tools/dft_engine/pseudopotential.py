"""
Pseudopotential Framework
==========================

In all-electron calculations, core electrons require very tight basis functions
(large exponents) and contribute enormous kinetic/potential energy that nearly
cancels. Pseudopotentials replace core electrons with an effective potential.

Types:
  1. Norm-Conserving (NC): Hamann, Schlüter, Chiang (1979)
     - Pseudo-wavefunction matches all-electron outside r_c
     - Same norm inside r_c: ∫₀^r_c |φ_ps|² = ∫₀^r_c |φ_ae|²
     - Same scattering properties (logarithmic derivative)

  2. Ultrasoft (US): Vanderbilt (1990)
     - Relaxes norm-conservation → smoother, lower cutoff
     - Requires augmentation charges

  3. Projector Augmented Wave (PAW): Blöchl (1994)
     - All-electron accuracy with pseudopotential efficiency
     - ψ_ae = T ψ_ps where T is a linear transformation
     - Most accurate and widely used (VASP, GPAW)

This module provides the framework for pseudopotential-based calculations.
The actual pseudopotential parameters would come from published tables
(e.g., ONCV, HGH, or PAW datasets).
"""

import numpy as np


class PseudoPotential:
    """
    Abstract pseudopotential for an element.
    
    V_ps(r) = V_local(r) + Σ_lm |p_lm> V_l <p_lm|
    
    V_local: long-range part (Coulomb-like for large r)
    V_l: angular-momentum-dependent nonlocal part
    p_lm: projector functions
    """
    
    def __init__(self, element, Z_val, r_cutoffs, local_potential=None):
        """
        Args:
            element: element symbol
            Z_val: valence electron count (Z - Z_core)
            r_cutoffs: dict of cutoff radii per angular momentum channel
            local_potential: callable V_local(r) or None
        """
        self.element = element
        self.Z_val = Z_val
        self.r_cutoffs = r_cutoffs
        self.local_potential = local_potential
        self.projectors = {}
    
    def add_projector(self, l, projector_func, V_l):
        """Add a nonlocal projector for angular momentum l."""
        self.projectors[l] = {'projector': projector_func, 'V_l': V_l}
    
    def V_local_eval(self, r):
        """Evaluate local potential at distance r."""
        if self.local_potential is not None:
            return self.local_potential(r)
        # Default: screened Coulomb
        return -self.Z_val / (r + 1e-10)


# ─── Simple GTH-type pseudopotential (Goedecker-Teter-Hutter) ───

class GTHPseudoPotential(PseudoPotential):
    """
    GTH (Goedecker-Teter-Hutter) pseudopotential.
    
    V_local(r) = -Z_ion/r · erf(r/(√2·r_loc)) + exp(-½(r/r_loc)²) ·
                 (C₁ + C₂(r/r_loc)² + C₃(r/r_loc)⁴ + C₄(r/r_loc)⁶)
    
    Widely used in CP2K. Analytical in both real and reciprocal space.
    """
    
    def __init__(self, element, Z_val, r_loc, C_coeffs):
        super().__init__(element, Z_val, {})
        self.r_loc = r_loc
        self.C_coeffs = C_coeffs  # [C1, C2, C3, C4]
    
    def V_local_eval(self, r):
        """GTH local potential."""
        from scipy.special import erf
        r_safe = np.maximum(r, 1e-10)
        x = r_safe / (np.sqrt(2.0) * self.r_loc)
        V = -self.Z_val / r_safe * erf(x)
        
        rr = r_safe / self.r_loc
        gauss = np.exp(-0.5 * rr**2)
        poly = 0.0
        for i, C in enumerate(self.C_coeffs):
            poly += C * rr**(2*i)
        V += gauss * poly
        return V


# Example GTH parameters for common elements
GTH_PARAMETERS = {
    'H': {'Z_val': 1, 'r_loc': 0.2, 'C': [-4.180237, 0.725075]},
    'C': {'Z_val': 4, 'r_loc': 0.3385, 'C': [-8.803674, 1.339210]},
    'N': {'Z_val': 5, 'r_loc': 0.2891, 'C': [-12.415107, 1.868060]},
    'O': {'Z_val': 6, 'r_loc': 0.2477, 'C': [-16.580318, 2.395701]},
    'Si': {'Z_val': 4, 'r_loc': 0.44, 'C': [-7.336103, 0.0]},
}


def get_gth_pseudopotential(element):
    """Get GTH pseudopotential for an element."""
    if element not in GTH_PARAMETERS:
        return None
    params = GTH_PARAMETERS[element]
    return GTHPseudoPotential(element, params['Z_val'], params['r_loc'], params['C'])
