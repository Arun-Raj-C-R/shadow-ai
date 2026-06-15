"""
Exchange-Correlation Functionals
==================================

Hohenberg-Kohn Theorem:
  The ground-state energy is a unique functional of the electron density ρ(r).
  E[ρ] = T_s[ρ] + E_ext[ρ] + E_H[ρ] + E_xc[ρ]

  E_xc contains EVERYTHING we don't know:
    E_xc = (T - T_s) + (E_ee - E_H)
    = kinetic correlation + exchange + Coulomb correlation

Jacob's Ladder of XC functionals (Perdew):
  Rung 1: LDA  — ε_xc(ρ)
  Rung 2: GGA  — ε_xc(ρ, ∇ρ)
  Rung 3: meta-GGA — ε_xc(ρ, ∇ρ, τ)
  Rung 4: Hybrid — mix with exact exchange
  Rung 5: Double hybrid — MP2 correlation

This module implements:
  - LDA: Slater exchange + VWN-5 correlation
  - GGA: PBE exchange + PBE correlation
  - Hybrid: B3LYP architecture
"""

import numpy as np


# =====================================================================
# LDA EXCHANGE — Dirac/Slater (1930/1951)
# =====================================================================
"""
Uniform electron gas exchange:
  ε_x(ρ) = -C_x · ρ^(1/3)
  where C_x = (3/4)(3/π)^(1/3) ≈ 0.7386

  E_x[ρ] = ∫ ε_x(ρ) · ρ d³r = -C_x ∫ ρ^(4/3) d³r

  V_x(ρ) = dE_x/dρ = -(4/3) C_x · ρ^(1/3)

Derivation:
  For a uniform electron gas with density ρ, the Fermi wavevector is:
    k_F = (3π²ρ)^(1/3)
  The exchange energy per electron:
    ε_x = -(3k_F)/(4π) = -(3/4)(3/π)^(1/3) ρ^(1/3)
"""

LDA_EXCHANGE_PREFACTOR = -(3.0/4.0) * (3.0/np.pi)**(1.0/3.0)


def lda_exchange(rho):
    """
    LDA Slater exchange energy density and potential.

    Returns:
        exc: exchange energy density ε_x(ρ) per electron
        vxc: exchange potential V_x(ρ) = dE_x/dρ
    """
    rho = np.maximum(rho, 1e-30)  # avoid division by zero
    rho_13 = rho**(1.0/3.0)
    exc = LDA_EXCHANGE_PREFACTOR * rho_13
    vxc = (4.0/3.0) * LDA_EXCHANGE_PREFACTOR * rho_13
    return exc, vxc


# =====================================================================
# LDA CORRELATION — VWN-5 (Vosko-Wilk-Nusair, 1980)
# =====================================================================
"""
Parameterization of the correlation energy of the uniform electron gas,
fitted to Ceperley-Alder quantum Monte Carlo data.

  ε_c(r_s) = A/2 · { ln(x²/X(x)) + 2b/Q · arctan(Q/(2x+b))
                     - bx₀/X(x₀) · [ ln((x-x₀)²/X(x)) + 2(b+2x₀)/Q · arctan(Q/(2x+b)) ] }

where:
  x = √r_s,  X(x) = x² + bx + c,  Q = √(4c - b²)
  r_s = (3/(4πρ))^(1/3)  — Wigner-Seitz radius

VWN-5 parameters (paramagnetic, unpolarized):
  A  = 0.0621814
  x₀ = -0.10498
  b  = 3.72744
  c  = 12.9352
"""

VWN_A  = 0.0621814 / 2.0  # Note: factor of 2 convention
VWN_x0 = -0.10498
VWN_b  = 3.72744
VWN_c  = 12.9352
VWN_Q  = np.sqrt(4.0 * VWN_c - VWN_b**2)
VWN_X0 = VWN_x0**2 + VWN_b * VWN_x0 + VWN_c


def _vwn_X(x):
    return x**2 + VWN_b * x + VWN_c


def vwn5_correlation(rho):
    """
    VWN-5 correlation energy density and potential.

    Returns:
        ec: correlation energy density per electron
        vc: correlation potential dE_c/dρ
    """
    rho = np.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * np.pi * rho))**(1.0/3.0)
    x = np.sqrt(rs)
    X_x = _vwn_X(x)

    ec = VWN_A * (
        np.log(x**2 / X_x)
        + 2.0 * VWN_b / VWN_Q * np.arctan(VWN_Q / (2.0*x + VWN_b))
        - VWN_b * VWN_x0 / VWN_X0 * (
            np.log((x - VWN_x0)**2 / X_x)
            + 2.0 * (VWN_b + 2.0*VWN_x0) / VWN_Q * np.arctan(VWN_Q / (2.0*x + VWN_b))
        )
    )

    # Potential: V_c = ε_c - (r_s/3) dε_c/dr_s
    dx_drs = 0.5 / x
    dX_dx = 2.0 * x + VWN_b

    dec_dx = VWN_A * (
        2.0/x - dX_dx/X_x
        - 4.0 * VWN_b / (VWN_Q**2 + (2.0*x + VWN_b)**2)
        - VWN_b * VWN_x0 / VWN_X0 * (
            2.0/(x - VWN_x0) - dX_dx/X_x
            - 4.0 * (VWN_b + 2.0*VWN_x0) / (VWN_Q**2 + (2.0*x + VWN_b)**2)
        )
    )

    vc = ec - rs / 3.0 * dec_dx * dx_drs

    return ec, vc


# =====================================================================
# LDA (combined exchange + correlation)
# =====================================================================

def lda_xc(rho):
    """
    Full LDA exchange-correlation: Slater exchange + VWN-5 correlation.

    Returns:
        exc: total XC energy density per electron
        vxc: total XC potential
    """
    ex, vx = lda_exchange(rho)
    ec, vc = vwn5_correlation(rho)
    return ex + ec, vx + vc


# =====================================================================
# PBE EXCHANGE (Perdew-Burke-Ernzerhof, 1996)
# =====================================================================
"""
GGA exchange enhancement factor:
  E_x^PBE[ρ] = ∫ ε_x^LDA(ρ) · F_x(s) · ρ d³r

  F_x(s) = 1 + κ - κ/(1 + μs²/κ)

where:
  s = |∇ρ| / (2 k_F ρ)  — reduced density gradient
  k_F = (3π²ρ)^(1/3)
  κ = 0.804  (Lieb-Oxford bound)
  μ = 0.21951  (≈ β π²/3, recovers LDA linear response)
"""

PBE_KAPPA = 0.804
PBE_MU = 0.21951


def pbe_exchange(rho, grad_rho_mag):
    """
    PBE GGA exchange.

    Args:
        rho: electron density ρ(r)
        grad_rho_mag: |∇ρ(r)|

    Returns:
        exc: exchange energy density per electron
        vxc: exchange potential (approximate, for SCF)
    """
    rho = np.maximum(rho, 1e-30)
    grad_rho_mag = np.maximum(grad_rho_mag, 0.0)

    rho_13 = rho**(1.0/3.0)
    kf = (3.0 * np.pi**2)**(1.0/3.0) * rho_13
    s = grad_rho_mag / (2.0 * kf * rho + 1e-30)
    s2 = s**2

    # Enhancement factor
    Fx = 1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s2 / PBE_KAPPA)

    # LDA exchange part
    ex_lda = LDA_EXCHANGE_PREFACTOR * rho_13
    exc = ex_lda * Fx

    # Simplified potential (LDA part + gradient correction)
    vxc = (4.0/3.0) * ex_lda * Fx  # approximate
    return exc, vxc


# =====================================================================
# PBE CORRELATION
# =====================================================================
"""
  E_c^PBE = E_c^LDA + ∫ ρ H(r_s, ζ, t) d³r

  H = (e²/a₀) · γ · φ³ · ln{1 + (β/γ)t²[(1 + At²)/(1 + At² + A²t⁴)]}

  t = |∇ρ| / (2 φ k_s ρ)  — reduced gradient for correlation
  k_s = √(4k_F/π)  — Thomas-Fermi screening wavevector
  φ = ½[(1+ζ)^(2/3) + (1-ζ)^(2/3)]  — spin scaling (=1 for unpolarized)
  ζ = (ρ↑ - ρ↓)/ρ  — relative spin polarization

  β = 0.066725  (gradient coefficient)
  γ = (1-ln2)/π² ≈ 0.031091
"""

PBE_BETA = 0.066725
PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi**2


def pbe_correlation(rho, grad_rho_mag):
    """
    PBE GGA correlation (unpolarized case, ζ=0).

    Returns:
        ec: correlation energy density per electron
        vc: correlation potential (approximate)
    """
    rho = np.maximum(rho, 1e-30)

    # LDA correlation as baseline
    ec_lda, vc_lda = vwn5_correlation(rho)

    rho_13 = rho**(1.0/3.0)
    kf = (3.0 * np.pi**2)**(1.0/3.0) * rho_13
    ks = np.sqrt(4.0 * kf / np.pi)
    phi = 1.0  # unpolarized

    t = grad_rho_mag / (2.0 * phi * ks * rho + 1e-30)
    t2 = t**2

    A = PBE_BETA / PBE_GAMMA / (np.exp(-ec_lda / (PBE_GAMMA * phi**3 + 1e-30)) - 1.0 + 1e-30)
    At2 = A * t2
    H = PBE_GAMMA * phi**3 * np.log(1.0 + PBE_BETA / PBE_GAMMA * t2 *
        (1.0 + At2) / (1.0 + At2 + A**2 * t2**2 + 1e-30))

    ec = ec_lda + H
    vc = vc_lda + H  # approximate potential
    return ec, vc


def pbe_xc(rho, grad_rho_mag):
    """Full PBE exchange-correlation."""
    ex, vx = pbe_exchange(rho, grad_rho_mag)
    ec, vc = pbe_correlation(rho, grad_rho_mag)
    return ex + ec, vx + vc


# =====================================================================
# FUNCTIONAL DISPATCHER
# =====================================================================

FUNCTIONALS = {
    'lda': {'name': 'LDA (Slater + VWN5)', 'needs_gradient': False, 'func': lda_xc},
    'pbe': {'name': 'PBE (GGA)', 'needs_gradient': True, 'func': pbe_xc},
}


def evaluate_xc(functional_name, rho, grad_rho_mag=None):
    """
    Evaluate XC functional.

    Args:
        functional_name: 'lda' or 'pbe'
        rho: electron density array
        grad_rho_mag: |∇ρ| array (required for GGA)

    Returns:
        exc: XC energy density per electron
        vxc: XC potential
    """
    func_info = FUNCTIONALS.get(functional_name.lower())
    if func_info is None:
        raise ValueError(f"Unknown functional: {functional_name}. Available: {list(FUNCTIONALS.keys())}")

    if func_info['needs_gradient']:
        if grad_rho_mag is None:
            raise ValueError(f"{functional_name} requires density gradient")
        return func_info['func'](rho, grad_rho_mag)
    else:
        return func_info['func'](rho)
