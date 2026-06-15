"""
Numerical Integration Grids for DFT
=====================================
Implements Becke partitioning + Lebedev angular quadrature + Euler-Maclaurin radial.

DFT requires numerical integration for exchange-correlation:
  E_xc[ρ] = ∫ ε_xc(ρ(r), ∇ρ(r)) ρ(r) d³r

The integral is evaluated on a molecular grid:
  ∫ f(r) d³r ≈ Σ_g w_g f(r_g)

Grid construction (Becke scheme):
  1. Atom-centered grids: r = (r, θ, φ) for each atom
  2. Radial quadrature: Euler-Maclaurin or Gauss-Chebyshev
  3. Angular quadrature: Lebedev grids (exact for spherical harmonics)
  4. Becke partitioning: smooth weight function to avoid double-counting
"""

import numpy as np
from .constants import ANGSTROM_TO_BOHR, COVALENT_RADII


# ─── Lebedev Angular Grids ───────────────────────────────────────────
# Exact for spherical harmonics up to order L.
# We store a few standard grid sizes.
# Format: (x, y, z, weight) on unit sphere, Σw = 4π

def _lebedev_6():
    """6-point Lebedev grid (exact for L=3). Octahedral vertices."""
    pts = []
    w = 4.0 * np.pi / 6.0
    for axis in range(3):
        for sign in [1.0, -1.0]:
            p = np.zeros(3)
            p[axis] = sign
            pts.append((*p, w))
    return np.array(pts)


def _lebedev_26():
    """26-point Lebedev grid (exact for L=5)."""
    pts = []
    # 6 octahedral vertices
    w1 = 4.0 * np.pi * 1.0 / 21.0
    for axis in range(3):
        for sign in [1.0, -1.0]:
            p = np.zeros(3)
            p[axis] = sign
            pts.append((*p, w1))
    # 12 edge centers
    w2 = 4.0 * np.pi * 4.0 / 105.0
    s = 1.0 / np.sqrt(2.0)
    for i in range(3):
        for j in range(i+1, 3):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    p = np.zeros(3)
                    p[i] = si * s
                    p[j] = sj * s
                    pts.append((*p, w2))
    # 8 cube vertices
    w3 = 4.0 * np.pi * 27.0 / 840.0
    s = 1.0 / np.sqrt(3.0)
    for sx in [1.0, -1.0]:
        for sy in [1.0, -1.0]:
            for sz in [1.0, -1.0]:
                pts.append((sx*s, sy*s, sz*s, w3))
    return np.array(pts)


def lebedev_grid(npoints=26):
    """Get Lebedev angular grid. Returns (N, 4): x, y, z, weight."""
    if npoints <= 6:
        return _lebedev_6()
    return _lebedev_26()


# ─── Radial Grids ────────────────────────────────────────────────────

def euler_maclaurin_radial(nrad, atom_Z):
    """
    Euler-Maclaurin radial grid (Murray-Handy-Laming).

    Mapping: r_i = R_m · i² / (nrad - i)²
    where R_m is a scaling factor (Bragg-Slater radius).

    Returns (radii, weights) arrays.
    """
    # Bragg-Slater radii by row (approximate, in bohr)
    if atom_Z <= 2:
        R_m = 1.0
    elif atom_Z <= 10:
        R_m = 1.5
    elif atom_Z <= 18:
        R_m = 2.0
    else:
        R_m = 2.5

    radii = np.zeros(nrad)
    weights = np.zeros(nrad)
    for i in range(1, nrad + 1):
        x = i / (nrad + 1.0)
        r = R_m * x**2 / (1.0 - x)**2
        dr = R_m * 2.0 * x * (nrad + 1.0) / (1.0 - x)**3 / (nrad + 1.0)
        radii[i-1] = r
        weights[i-1] = r**2 * dr / (nrad + 1.0)
    # Normalize: ∫₀^∞ r² dr → Σ w_i
    # Weight includes r² and Jacobian
    return radii, weights * (4.0 * np.pi)  # angular factor


def gauss_chebyshev_radial(nrad, atom_Z):
    """
    Gauss-Chebyshev radial grid of the second kind with Becke mapping.

    Mapping: r = R_m · (1+x)/(1-x), x = cos(πi/(n+1))
    """
    if atom_Z <= 2:
        R_m = 1.0
    elif atom_Z <= 10:
        R_m = 1.5
    else:
        R_m = 2.0

    radii = np.zeros(nrad)
    weights = np.zeros(nrad)
    for i in range(1, nrad + 1):
        x = np.cos(np.pi * i / (nrad + 1))
        r = R_m * (1.0 + x) / (1.0 - x)
        dr = 2.0 * R_m / (1.0 - x)**2
        w_cheb = np.pi / (nrad + 1) * np.sin(np.pi * i / (nrad + 1))**2
        radii[i-1] = r
        weights[i-1] = r**2 * dr * w_cheb
    return radii, weights * (4.0 * np.pi)


# ─── Becke Partitioning ──────────────────────────────────────────────

def becke_partition_weights(grid_coords, molecule, atom_idx):
    """
    Becke partitioning function for multi-center integration.

    The molecular integral is decomposed:
      ∫ f(r) d³r = Σ_A ∫ w_A(r) f(r) d³r

    where w_A(r) is a smooth partition of unity: Σ_A w_A(r) = 1

    Uses the Becke switching function:
      s(μ) = ½(1 - p₃(μ))  where p₃ is iterated 3 times:
      p(μ) = 3μ/2 - μ³/2

    μ_AB = (|r-A| - |r-B|) / |A-B|  (confocal elliptic coordinate)
    """
    natoms = molecule.natoms
    if natoms == 1:
        return np.ones(len(grid_coords))

    coords = molecule.nuclear_coords
    npts = len(grid_coords)
    P = np.ones((natoms, npts))

    for i in range(natoms):
        for j in range(natoms):
            if i == j:
                continue
            R_ij = np.linalg.norm(coords[i] - coords[j])
            if R_ij < 1e-10:
                continue
            r_i = np.linalg.norm(grid_coords - coords[i], axis=1)
            r_j = np.linalg.norm(grid_coords - coords[j], axis=1)
            mu = (r_i - r_j) / R_ij
            # Becke's iterated switching function (3 iterations)
            for _ in range(3):
                mu = 1.5 * mu - 0.5 * mu**3
            s = 0.5 * (1.0 - mu)
            P[i] *= s

    # Normalize: w_A = P_A / Σ_B P_B
    P_sum = np.sum(P, axis=0)
    P_sum = np.where(P_sum < 1e-15, 1.0, P_sum)
    return P[atom_idx] / P_sum


# ─── Molecular Grid Construction ─────────────────────────────────────

class MolecularGrid:
    """
    Full molecular integration grid.

    Combines radial (Euler-Maclaurin/Chebyshev) and angular (Lebedev)
    quadratures with Becke partitioning.
    """

    def __init__(self, molecule, nrad=20, nang=26):
        """
        Build molecular grid.

        Args:
            molecule: Molecule object
            nrad: number of radial points per atom (20-100 typical)
            nang: number of angular points per atom (6, 26, etc.)
        """
        self.molecule = molecule
        self.nrad = nrad
        self.nang = nang
        self._build_grid()

    def _build_grid(self):
        """Construct atom-centered grids with Becke partitioning."""
        all_coords = []
        all_weights = []
        ang_grid = lebedev_grid(self.nang)
        ang_xyz = ang_grid[:, :3]
        ang_w = ang_grid[:, 3]

        for iatom, atom in enumerate(self.molecule.atoms):
            radii, rad_w = gauss_chebyshev_radial(self.nrad, atom.Z)
            for ir in range(self.nrad):
                r = radii[ir]
                if r < 1e-12:
                    continue
                # Combine radial and angular
                for ia in range(len(ang_w)):
                    xyz = atom.coords + r * ang_xyz[ia]
                    w = rad_w[ir] * ang_w[ia] / (4.0 * np.pi)
                    all_coords.append(xyz)
                    all_weights.append(w)

        self.coords = np.array(all_coords)
        self.weights_raw = np.array(all_weights)
        self.npoints = len(self.coords)

        # Apply Becke partitioning
        self.weights = np.zeros(self.npoints)
        idx = 0
        for iatom in range(self.molecule.natoms):
            npts_atom = self.nrad * len(ang_grid)
            atom_coords = self.coords[idx:idx+npts_atom]
            bw = becke_partition_weights(atom_coords, self.molecule, iatom)
            self.weights[idx:idx+npts_atom] = self.weights_raw[idx:idx+npts_atom] * bw
            idx += npts_atom

    def integrate(self, f_values):
        """∫ f(r) d³r ≈ Σ_g w_g f(r_g)"""
        return np.sum(self.weights * f_values)
