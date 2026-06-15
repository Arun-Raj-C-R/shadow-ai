"""
Electronic Properties: Band Structure, DOS, Charge Density
============================================================
"""

import numpy as np
from .constants import HARTREE_TO_EV


def compute_dos(orbital_energies, sigma=0.05, npoints=500, emin=None, emax=None):
    """
    Density of States using Gaussian broadening.

    DOS(E) = Σ_i (1/√(2πσ²)) exp(-(E - ε_i)² / (2σ²))

    Args:
        orbital_energies: array of eigenvalues (Hartree)
        sigma: Gaussian broadening width (Hartree)
        npoints: number of energy grid points

    Returns:
        energies_eV: energy grid in eV
        dos: DOS values
    """
    e_min = (emin if emin else np.min(orbital_energies) - 5*sigma)
    e_max = (emax if emax else np.max(orbital_energies) + 5*sigma)
    energies = np.linspace(e_min, e_max, npoints)

    dos = np.zeros(npoints)
    for ei in orbital_energies:
        dos += np.exp(-0.5 * ((energies - ei) / sigma)**2) / (sigma * np.sqrt(2*np.pi))

    return energies * HARTREE_TO_EV, dos


def compute_homo_lumo(orbital_energies, nelec):
    """
    HOMO-LUMO gap.
    
    HOMO = Highest Occupied Molecular Orbital
    LUMO = Lowest Unoccupied Molecular Orbital
    Gap = LUMO - HOMO (approximation to fundamental gap)
    
    Note: KS eigenvalue gap underestimates the true gap.
    """
    nocc = nelec // 2
    if nocc <= 0 or nocc >= len(orbital_energies):
        return {'homo': None, 'lumo': None, 'gap_eV': None}

    homo = orbital_energies[nocc - 1]
    lumo = orbital_energies[nocc]
    gap = lumo - homo
    return {
        'homo_Ha': homo, 'homo_eV': homo * HARTREE_TO_EV,
        'lumo_Ha': lumo, 'lumo_eV': lumo * HARTREE_TO_EV,
        'gap_Ha': gap, 'gap_eV': gap * HARTREE_TO_EV,
    }


def compute_charge_density(P, basis, grid_coords):
    """
    Electron density on a 3D grid.
    ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
    """
    n = basis.nbasis
    npts = len(grid_coords)
    chi = np.zeros((n, npts))
    for mu in range(n):
        for ig in range(npts):
            chi[mu, ig] = basis[mu].evaluate(grid_coords[ig])
    rho = np.einsum('ij,ik,jk->k', P, chi, chi)
    return np.maximum(rho, 0.0)


def potential_energy_surface(molecule_builder, distances, basis_name='sto-3g',
                              method='hf', functional='lda'):
    """
    Compute potential energy surface along a coordinate.

    Args:
        molecule_builder: function(distance) -> Molecule
        distances: array of distances to scan

    Returns:
        distances, energies arrays
    """
    from .dft_engine import DFTCalculation
    energies = []
    for d in distances:
        mol = molecule_builder(d)
        calc = DFTCalculation(mol, basis_name, method, functional, verbose=False)
        result = calc.run()
        energies.append(result['energy'])
    return np.array(distances), np.array(energies)
