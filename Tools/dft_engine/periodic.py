"""
Periodic Boundary Conditions & k-point Sampling
=================================================

Bloch's Theorem:
  For a periodic potential V(r+R) = V(r), the eigenstates have the form:
    ψ_k(r) = e^{ik·r} u_k(r)
  where u_k(r+R) = u_k(r) is lattice-periodic.

  This reduces the infinite crystal problem to solving for each k-point
  independently in the unit cell.

Reciprocal Lattice:
  Given real-space lattice vectors a₁, a₂, a₃:
    b₁ = 2π (a₂ × a₃) / (a₁ · (a₂ × a₃))
    b₂ = 2π (a₃ × a₁) / (a₁ · (a₂ × a₃))
    b₃ = 2π (a₁ × a₂) / (a₁ · (a₂ × a₃))

  Any reciprocal lattice vector: G = n₁b₁ + n₂b₂ + n₃b₃

Brillouin Zone:
  The first BZ is the Wigner-Seitz cell of the reciprocal lattice.
  Integration over k is replaced by discrete sampling:
    ∫_{BZ} f(k) dk/(2π)³ ≈ (1/N_k) Σ_k w_k f(k)

Monkhorst-Pack Grid:
  k_p = (2p - N - 1)/(2N) · bᵢ,  p = 1, ..., N
  Generates uniformly spaced k-points in the BZ.
"""

import numpy as np
from .constants import PI, TWO_PI


class Lattice:
    """Crystal lattice definition."""

    def __init__(self, vectors):
        """
        Args:
            vectors: (3, 3) array where rows are lattice vectors a₁, a₂, a₃ in bohr
        """
        self.vectors = np.array(vectors, dtype=np.float64)
        self.volume = abs(np.dot(self.vectors[0], np.cross(self.vectors[1], self.vectors[2])))

    @property
    def reciprocal_vectors(self):
        """Compute reciprocal lattice vectors b_i = 2π (a_j × a_k) / V"""
        a = self.vectors
        V = self.volume
        b = np.zeros((3, 3))
        b[0] = TWO_PI * np.cross(a[1], a[2]) / V
        b[1] = TWO_PI * np.cross(a[2], a[0]) / V
        b[2] = TWO_PI * np.cross(a[0], a[1]) / V
        return b

    @classmethod
    def cubic(cls, a):
        """Simple cubic lattice with parameter a (in bohr)."""
        return cls(a * np.eye(3))

    @classmethod
    def fcc(cls, a):
        """Face-centered cubic."""
        return cls(a/2.0 * np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=np.float64))

    @classmethod
    def bcc(cls, a):
        """Body-centered cubic."""
        return cls(a/2.0 * np.array([[-1,1,1],[1,-1,1],[1,1,-1]], dtype=np.float64))


def monkhorst_pack(nk):
    """
    Generate Monkhorst-Pack k-point grid.

    Args:
        nk: (n1, n2, n3) grid dimensions

    Returns:
        kpoints: (N, 3) fractional coordinates
        weights: (N,) integration weights (1/N_k each)
    """
    n1, n2, n3 = nk
    kpoints = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                kx = (2*i - n1 + 1) / (2.0 * n1)
                ky = (2*j - n2 + 1) / (2.0 * n2)
                kz = (2*k - n3 + 1) / (2.0 * n3)
                kpoints.append([kx, ky, kz])

    kpoints = np.array(kpoints)
    weights = np.ones(len(kpoints)) / len(kpoints)
    return kpoints, weights


def high_symmetry_path(lattice_type='fcc', npoints=50):
    """
    Generate k-point path along high-symmetry directions for band structure.

    For FCC: Γ → X → W → K → Γ → L
    """
    if lattice_type == 'fcc':
        special_points = {
            'Γ': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
            'W': np.array([0.5, 0.25, 0.75]),
            'K': np.array([0.375, 0.375, 0.75]),
            'L': np.array([0.5, 0.5, 0.5]),
        }
        path = ['Γ', 'X', 'W', 'K', 'Γ', 'L']
    elif lattice_type == 'bcc':
        special_points = {
            'Γ': np.array([0.0, 0.0, 0.0]),
            'H': np.array([0.5, -0.5, 0.5]),
            'N': np.array([0.0, 0.0, 0.5]),
            'P': np.array([0.25, 0.25, 0.25]),
        }
        path = ['Γ', 'H', 'N', 'Γ', 'P', 'H']
    else:  # simple cubic
        special_points = {
            'Γ': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.0]),
            'M': np.array([0.5, 0.5, 0.0]),
            'R': np.array([0.5, 0.5, 0.5]),
        }
        path = ['Γ', 'X', 'M', 'Γ', 'R', 'X']

    kpoints = []
    labels = []
    label_positions = []
    pos = 0

    for i in range(len(path) - 1):
        k1 = special_points[path[i]]
        k2 = special_points[path[i+1]]
        segment = np.linspace(k1, k2, npoints, endpoint=(i == len(path)-2))
        label_positions.append(pos)
        labels.append(path[i])
        kpoints.extend(segment)
        pos += len(segment)

    label_positions.append(pos - 1)
    labels.append(path[-1])

    return np.array(kpoints), labels, label_positions
