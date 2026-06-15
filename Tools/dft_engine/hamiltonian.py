"""
Hamiltonian / Fock / Kohn-Sham Matrix Construction
====================================================

Supports both Restricted (RHF/RKS) and Unrestricted (UHF/UKS) formalisms.

RHF density matrix:  P = 2 * C_occ @ C_occ.T
UHF density matrices: P_a = C_a_occ @ C_a_occ.T, P_b = C_b_occ @ C_b_occ.T
                       P_total = P_a + P_b

RHF Fock: F = H_core + J(P) - 0.5*K(P)
UHF Fock: F_a = H_core + J(P_tot) - K(P_a)
          F_b = H_core + J(P_tot) - K(P_b)

Total energy:
  RHF: E = 0.5 * Tr[P*(H_core + F)] + E_nn
  UHF: E = 0.5 * Tr[P_a*(H_core + F_a) + P_b*(H_core + F_b)] + E_nn
"""

import numpy as np
from .grid import MolecularGrid
from .xc import evaluate_xc


def build_density_matrix_restricted(C, nelec):
    """RHF density: P = 2 * C_occ @ C_occ.T (closed-shell, even electrons)."""
    nocc = nelec // 2
    if nocc == 0:
        return np.zeros((C.shape[0], C.shape[0]))
    C_occ = C[:, :nocc]
    return 2.0 * C_occ @ C_occ.T


def build_density_matrix_unrestricted(C_alpha, C_beta, nalpha, nbeta):
    """
    UHF density matrices.
    P_a = C_a_occ @ C_a_occ.T
    P_b = C_b_occ @ C_b_occ.T
    """
    n = C_alpha.shape[0]
    P_a = np.zeros((n, n))
    P_b = np.zeros((n, n))
    if nalpha > 0:
        C_a_occ = C_alpha[:, :nalpha]
        P_a = C_a_occ @ C_a_occ.T
    if nbeta > 0:
        C_b_occ = C_beta[:, :nbeta]
        P_b = C_b_occ @ C_b_occ.T
    return P_a, P_b


# Keep backward-compatible alias
def build_density_matrix(C, nelec):
    """Build density matrix (handles both even and odd electron counts)."""
    return build_density_matrix_restricted(C, nelec)


def build_coulomb_matrix(P, eri):
    """J_uv = sum_{ls} P_ls * (uv|ls)"""
    return np.einsum('ls,uvls->uv', P, eri)


def build_exchange_matrix(P, eri):
    """K_uv = sum_{ls} P_ls * (ul|vs)"""
    return np.einsum('ls,ulvs->uv', P, eri)


def compute_electron_density_on_grid(P, basis, grid_coords):
    """rho(r) = sum_{uv} P_uv * phi_u(r) * phi_v(r)"""
    n = basis.nbasis
    npts = len(grid_coords)
    chi = np.zeros((n, npts))
    for mu in range(n):
        for ig in range(npts):
            chi[mu, ig] = basis[mu].evaluate(grid_coords[ig])
    rho = np.einsum('ij,ik,jk->k', P, chi, chi)
    return np.maximum(rho, 0.0)


def build_xc_matrix(P, basis, molecule, functional='lda', grid=None):
    """
    Build V_xc matrix by numerical integration on molecular grid.
    Returns V_xc matrix and E_xc energy.
    """
    if grid is None:
        grid = MolecularGrid(molecule, nrad=35, nang=26)

    n = basis.nbasis
    npts = grid.npoints

    chi = np.zeros((n, npts))
    for mu in range(n):
        for ig in range(npts):
            chi[mu, ig] = basis[mu].evaluate(grid.coords[ig])

    rho = np.einsum('ij,ik,jk->k', P, chi, chi)
    rho = np.maximum(rho, 1e-30)

    if functional.lower() in ('pbe', 'gga'):
        grad_rho_mag = np.zeros(npts)
        exc, vxc = evaluate_xc(functional, rho, grad_rho_mag)
    else:
        exc, vxc = evaluate_xc(functional, rho)

    V_xc = np.einsum('k,ik,jk->ij', grid.weights * vxc, chi, chi)
    # Symmetrize
    V_xc = 0.5 * (V_xc + V_xc.T)

    E_xc = np.sum(grid.weights * exc * rho)
    return V_xc, E_xc


def build_fock_matrix(H_core, P, eri, basis=None, molecule=None,
                      method='hf', functional='lda', grid=None):
    """
    Build Fock/KS matrix for RESTRICTED formalism.

    HF:   F = H_core + J - 0.5*K
    DFT:  F = H_core + J + V_xc
    """
    J = build_coulomb_matrix(P, eri)
    E_J = 0.5 * np.sum(P * J)
    components = {'E_J': E_J}

    if method.lower() == 'hf':
        K = build_exchange_matrix(P, eri)
        F = H_core + J - 0.5 * K
        components['E_K'] = -0.25 * np.sum(P * K)
        components['E_xc'] = 0.0
    elif method.lower() == 'dft':
        V_xc, E_xc = build_xc_matrix(P, basis, molecule, functional, grid)
        F = H_core + J + V_xc
        components['E_xc'] = E_xc
        components['E_K'] = 0.0
    elif method.lower() == 'hybrid':
        a_x = 0.20
        K = build_exchange_matrix(P, eri)
        V_xc, E_xc = build_xc_matrix(P, basis, molecule, functional, grid)
        F = H_core + J + (1.0 - a_x) * V_xc - a_x * 0.5 * K
        components['E_xc'] = (1.0 - a_x) * E_xc
        components['E_K'] = -a_x * 0.25 * np.sum(P * K)
    else:
        raise ValueError(f"Unknown method: {method}")

    return F, components


def build_uhf_fock_matrices(H_core, P_a, P_b, eri):
    """
    Build UHF Fock matrices.
    F_a = H_core + J(P_tot) - K(P_a)
    F_b = H_core + J(P_tot) - K(P_b)
    """
    P_tot = P_a + P_b
    J = build_coulomb_matrix(P_tot, eri)
    K_a = build_exchange_matrix(P_a, eri)
    K_b = build_exchange_matrix(P_b, eri)
    F_a = H_core + J - K_a
    F_b = H_core + J - K_b
    E_J = 0.5 * np.sum(P_tot * J)
    E_K = -0.5 * (np.sum(P_a * K_a) + np.sum(P_b * K_b))
    return F_a, F_b, {'E_J': E_J, 'E_K': E_K, 'E_xc': 0.0}


def compute_total_energy(P, H_core, F, E_nn, E_components):
    """Total energy: E = 0.5 * Tr[P*(H_core + F)] + E_nn"""
    E_elec = 0.5 * np.sum(P * (H_core + F))
    return E_elec + E_nn


def compute_uhf_total_energy(P_a, P_b, H_core, F_a, F_b, E_nn):
    """UHF total energy: E = 0.5*Tr[P_a*(H+F_a) + P_b*(H+F_b)] + E_nn"""
    E_elec = 0.5 * (np.sum(P_a * (H_core + F_a)) + np.sum(P_b * (H_core + F_b)))
    return E_elec + E_nn
