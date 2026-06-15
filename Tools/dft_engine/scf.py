"""
Self-Consistent Field (SCF) Solver
====================================
Supports RHF, UHF, RKS, UKS with DIIS acceleration.

SCF Algorithm:
  1. Compute 1e integrals: S, T, V -> H_core = T + V
  2. Compute 2e integrals: ERI tensor
  3. Orthogonalization matrix X = S^{-1/2} (Lowdin)
  4. Initial guess from core Hamiltonian
  5. LOOP:
     a. Build F(P)
     b. DIIS extrapolation
     c. Solve F'C' = C'e in orthogonal basis
     d. Back-transform C = X @ C'
     e. Build P from occupied orbitals
     f. Check convergence: |dE| < tol AND rms(dP) < tol
  6. Final energy and analysis

Convergence criteria:
  - Energy change: |E_new - E_old| < E_threshold
  - Density change: rms(P_new - P_old) < P_threshold
  - Both must be satisfied simultaneously
"""

import numpy as np
import time
from .integrals import (compute_overlap_matrix, compute_kinetic_matrix,
                        compute_nuclear_matrix, compute_eri_tensor,
                        compute_core_hamiltonian, validate_overlap_matrix)
from .hamiltonian import (build_density_matrix_restricted,
                          build_density_matrix_unrestricted,
                          build_fock_matrix, build_uhf_fock_matrices,
                          compute_total_energy, compute_uhf_total_energy)
from .grid import MolecularGrid


class DIISAccelerator:
    """
    DIIS (Direct Inversion of Iterative Subspace) convergence accelerator.
    Pulay, Chem. Phys. Lett. 73, 393 (1980).

    Error vector: e = FPS - SPF (commutator, zero at convergence).
    Minimizes |sum_i c_i e_i|^2 subject to sum_i c_i = 1.
    """

    def __init__(self, max_vectors=8):
        self.max_vectors = max_vectors
        self.fock_list = []
        self.error_list = []

    def add(self, F, P, S):
        err = F @ P @ S - S @ P @ F
        if len(self.fock_list) >= self.max_vectors:
            self.fock_list.pop(0)
            self.error_list.pop(0)
        self.fock_list.append(F.copy())
        self.error_list.append(err.copy())

    def extrapolate(self):
        n = len(self.fock_list)
        if n < 2:
            return None

        # Build B matrix with regularization
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                bij = np.sum(self.error_list[i] * self.error_list[j])
                B[i, j] = bij
                B[j, i] = bij
        # Regularize diagonal to avoid singular matrix
        for i in range(n):
            B[i, i] *= 1.0 + 1e-10
        B[:n, n] = -1.0
        B[n, :n] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            # Fall back to least-squares if singular
            try:
                coeffs, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
            except Exception:
                return None

        F_diis = np.zeros_like(self.fock_list[0])
        for i in range(n):
            F_diis += coeffs[i] * self.fock_list[i]
        return F_diis

    def reset(self):
        self.fock_list.clear()
        self.error_list.clear()

    @property
    def max_error(self):
        if not self.error_list:
            return float('inf')
        return max(np.max(np.abs(e)) for e in self.error_list)


def _lowdin_orthogonalize(S, threshold=1e-6):
    """
    Lowdin symmetric orthogonalization: X = S^{-1/2}.

    S = U @ diag(s) @ U.T
    X = U @ diag(s^{-1/2}) @ U.T

    Eigenvalues below threshold are removed (linear dependency).
    Returns X and the number of removed functions.
    """
    eigvals, U = np.linalg.eigh(S)
    mask = eigvals > threshold
    n_removed = np.sum(~mask)
    s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals[mask]))
    X = U[:, mask] @ s_inv_sqrt
    return X, n_removed


class SCFSolver:
    """
    Self-Consistent Field solver supporting RHF, UHF, RKS, UKS.
    """

    def __init__(self, molecule, basis, method='hf', functional='lda',
                 max_iter=200, e_threshold=1e-8, p_threshold=1e-6,
                 diis=True, damping=0.0, verbose=True, unrestricted=None):
        self.molecule = molecule
        self.basis = basis
        self.method = method
        self.functional = functional
        self.max_iter = max_iter
        self.e_threshold = e_threshold
        self.p_threshold = p_threshold
        self.use_diis = diis
        self.damping = damping
        self.verbose = verbose

        self.nbasis = basis.nbasis
        self.nelec = molecule.nelec
        self.nalpha = molecule.nalpha
        self.nbeta = molecule.nbeta
        self.E_nn = molecule.nuclear_repulsion_energy()

        # Auto-detect need for unrestricted
        if unrestricted is None:
            self.unrestricted = (self.nalpha != self.nbeta)
        else:
            self.unrestricted = unrestricted

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def solve(self):
        if self.unrestricted:
            return self._solve_uhf()
        else:
            return self._solve_rhf()

    def _solve_rhf(self):
        """Restricted Hartree-Fock / Kohn-Sham SCF."""
        t_start = time.time()
        prefix = "RKS" if self.method == 'dft' else "RHF"

        self._print(f"\n{'='*60}")
        self._print(f"  SCF Calculation: {prefix}")
        if self.method == 'dft':
            self._print(f"  Functional: {self.functional.upper()}")
        self._print(f"  Basis functions: {self.nbasis}")
        self._print(f"  Electrons: {self.nelec}")
        self._print(f"  Nuclear repulsion: {self.E_nn:.10f} Ha")
        self._print(f"{'='*60}")

        # 1. One-electron integrals
        self._print("  Computing one-electron integrals...")
        S = compute_overlap_matrix(self.basis)
        H_core, T, V = compute_core_hamiltonian(self.basis, self.molecule)

        # Validate overlap
        issues = validate_overlap_matrix(S)
        for issue in issues:
            self._print(f"  WARNING: {issue}")

        # 2. Two-electron integrals
        self._print("  Computing two-electron integrals...")
        eri = compute_eri_tensor(self.basis)

        # 3. Orthogonalization
        X, n_removed = _lowdin_orthogonalize(S)
        if n_removed > 0:
            self._print(f"  WARNING: {n_removed} linear dependencies removed")

        # 4. Initial guess
        F_prime = X.T @ H_core @ X
        eigvals, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime
        P = build_density_matrix_restricted(C, self.nelec)

        # 5. Grid for DFT
        grid = None
        if self.method in ('dft', 'hybrid'):
            self._print("  Building integration grid...")
            grid = MolecularGrid(self.molecule, nrad=35, nang=26)
            self._print(f"  Grid points: {grid.npoints}")

        # 6. SCF Loop
        self._print(f"\n  {'Iter':>4s}  {'Energy':>18s}  {'dE':>14s}  {'dP_rms':>12s}  {'Status'}")
        self._print(f"  {'-'*68}")

        E_old = 0.0
        converged = False
        diis = DIISAccelerator(max_vectors=8) if self.use_diis else None

        for iteration in range(1, self.max_iter + 1):
            F, E_comp = build_fock_matrix(
                H_core, P, eri, self.basis, self.molecule,
                self.method, self.functional, grid)

            if diis is not None:
                diis.add(F, P, S)
                F_diis = diis.extrapolate()
                if F_diis is not None:
                    F = F_diis

            F_prime = X.T @ F @ X
            eigvals, C_prime = np.linalg.eigh(F_prime)
            C = X @ C_prime
            P_new = build_density_matrix_restricted(C, self.nelec)

            E_total = compute_total_energy(P_new, H_core, F, self.E_nn, E_comp)
            dE = E_total - E_old
            dP_rms = np.sqrt(np.mean((P_new - P) ** 2))

            # Check BOTH convergence criteria
            e_conv = abs(dE) < self.e_threshold
            p_conv = dP_rms < self.p_threshold
            status = ""
            if e_conv and p_conv and iteration > 1:
                status = "CONVERGED"
                converged = True

            self._print(f"  {iteration:4d}  {E_total:18.10f}  {dE:14.2e}  {dP_rms:12.2e}  {status}")

            if converged:
                break

            if self.damping > 0 and iteration < 5:
                P = (1.0 - self.damping) * P_new + self.damping * P
            else:
                P = P_new.copy()
            E_old = E_total

        # Density normalization check
        nelec_check = np.trace(P @ S)

        t_elapsed = time.time() - t_start
        E_1e = np.sum(P * H_core)

        self._print(f"\n  {'='*60}")
        if converged:
            self._print(f"  SCF CONVERGED in {iteration} iterations ({t_elapsed:.2f}s)")
        else:
            self._print(f"  SCF DID NOT CONVERGE after {self.max_iter} iterations")
        self._print(f"  Total Energy:     {E_total:20.12f} Ha")
        self._print(f"  Electron count:   {nelec_check:.6f} (expected {self.nelec})")
        self._print(f"  {'='*60}\n")

        return {
            'converged': converged,
            'energy': E_total,
            'orbital_energies': eigvals,
            'C': C, 'P': P, 'S': S,
            'H_core': H_core, 'T': T, 'V': V, 'F': F, 'eri': eri,
            'iterations': iteration,
            'electron_count': nelec_check,
            'energy_components': {
                'E_total': E_total, 'E_1e': E_1e,
                'E_J': E_comp['E_J'], 'E_K': E_comp.get('E_K', 0.0),
                'E_xc': E_comp.get('E_xc', 0.0), 'E_nn': self.E_nn,
            },
            'time_seconds': t_elapsed,
            'method': self.method,
            'functional': self.functional if self.method == 'dft' else None,
            'unrestricted': False,
        }

    def _solve_uhf(self):
        """Unrestricted Hartree-Fock SCF (for open-shell systems)."""
        t_start = time.time()

        self._print(f"\n{'='*60}")
        self._print(f"  SCF Calculation: UHF")
        self._print(f"  Basis functions: {self.nbasis}")
        self._print(f"  Electrons: {self.nelec} (alpha={self.nalpha}, beta={self.nbeta})")
        self._print(f"  Nuclear repulsion: {self.E_nn:.10f} Ha")
        self._print(f"{'='*60}")

        # 1. Integrals
        self._print("  Computing integrals...")
        S = compute_overlap_matrix(self.basis)
        H_core, T, V = compute_core_hamiltonian(self.basis, self.molecule)
        eri = compute_eri_tensor(self.basis)

        # 2. Orthogonalization
        X, n_removed = _lowdin_orthogonalize(S)

        # 3. Initial guess (same for both spins)
        F_prime = X.T @ H_core @ X
        eigvals_a, C_prime = np.linalg.eigh(F_prime)
        C_a = X @ C_prime
        C_b = C_a.copy()
        eigvals_b = eigvals_a.copy()

        P_a, P_b = build_density_matrix_unrestricted(C_a, C_b, self.nalpha, self.nbeta)

        # 4. SCF Loop
        self._print(f"\n  {'Iter':>4s}  {'Energy':>18s}  {'dE':>14s}  {'dP_rms':>12s}  {'Status'}")
        self._print(f"  {'-'*68}")

        E_old = 0.0
        converged = False
        diis_a = DIISAccelerator(max_vectors=8) if self.use_diis else None
        diis_b = DIISAccelerator(max_vectors=8) if self.use_diis else None

        for iteration in range(1, self.max_iter + 1):
            F_a, F_b, E_comp = build_uhf_fock_matrices(H_core, P_a, P_b, eri)

            # DIIS for each spin channel
            if diis_a is not None:
                diis_a.add(F_a, P_a, S)
                F_a_diis = diis_a.extrapolate()
                if F_a_diis is not None:
                    F_a = F_a_diis
            if diis_b is not None and self.nbeta > 0:
                diis_b.add(F_b, P_b, S)
                F_b_diis = diis_b.extrapolate()
                if F_b_diis is not None:
                    F_b = F_b_diis

            # Solve for alpha
            F_a_prime = X.T @ F_a @ X
            eigvals_a, C_a_prime = np.linalg.eigh(F_a_prime)
            C_a = X @ C_a_prime

            # Solve for beta
            F_b_prime = X.T @ F_b @ X
            eigvals_b, C_b_prime = np.linalg.eigh(F_b_prime)
            C_b = X @ C_b_prime

            P_a_new, P_b_new = build_density_matrix_unrestricted(
                C_a, C_b, self.nalpha, self.nbeta)

            E_total = compute_uhf_total_energy(P_a_new, P_b_new, H_core, F_a, F_b, self.E_nn)
            dE = E_total - E_old
            dP_rms = np.sqrt(0.5 * (np.mean((P_a_new - P_a)**2) + np.mean((P_b_new - P_b)**2)))

            e_conv = abs(dE) < self.e_threshold
            p_conv = dP_rms < self.p_threshold
            status = ""
            if e_conv and p_conv and iteration > 1:
                status = "CONVERGED"
                converged = True

            self._print(f"  {iteration:4d}  {E_total:18.10f}  {dE:14.2e}  {dP_rms:12.2e}  {status}")

            if converged:
                break

            P_a = P_a_new.copy()
            P_b = P_b_new.copy()
            E_old = E_total

        P_tot = P_a + P_b
        nelec_check = np.trace(P_tot @ S)
        t_elapsed = time.time() - t_start

        # Spin contamination: <S^2> = S_exact + N_beta - sum_{ij} |<alpha_i|beta_j>|^2
        S2_exact = 0.5 * (self.nalpha - self.nbeta) * (0.5 * (self.nalpha - self.nbeta) + 1)

        self._print(f"\n  {'='*60}")
        if converged:
            self._print(f"  UHF CONVERGED in {iteration} iterations ({t_elapsed:.2f}s)")
        else:
            self._print(f"  UHF DID NOT CONVERGE after {self.max_iter} iterations")
        self._print(f"  Total Energy:     {E_total:20.12f} Ha")
        self._print(f"  Electron count:   {nelec_check:.6f} (expected {self.nelec})")
        self._print(f"  <S^2> exact:      {S2_exact:.4f}")
        self._print(f"  Alpha orbitals:   {eigvals_a[:min(5, len(eigvals_a))]}")
        self._print(f"  Beta orbitals:    {eigvals_b[:min(5, len(eigvals_b))]}")
        self._print(f"  {'='*60}\n")

        return {
            'converged': converged,
            'energy': E_total,
            'orbital_energies': eigvals_a,
            'orbital_energies_beta': eigvals_b,
            'C': C_a, 'C_beta': C_b,
            'P': P_tot, 'P_alpha': P_a, 'P_beta': P_b,
            'S': S, 'H_core': H_core, 'T': T, 'V': V,
            'F': F_a, 'F_beta': F_b, 'eri': eri,
            'iterations': iteration,
            'electron_count': nelec_check,
            'energy_components': {
                'E_total': E_total,
                'E_1e': np.sum(P_tot * H_core),
                'E_J': E_comp['E_J'], 'E_K': E_comp['E_K'],
                'E_xc': 0.0, 'E_nn': self.E_nn,
            },
            'time_seconds': t_elapsed,
            'method': 'uhf',
            'functional': None,
            'unrestricted': True,
        }


class MullikenAnalysis:
    """Mulliken population analysis: q_A = Z_A - sum_{mu in A} (PS)_{mu,mu}"""

    @staticmethod
    def compute(molecule, basis, P, S):
        PS = P @ S
        populations = np.diag(PS)

        atom_pops = np.zeros(molecule.natoms)
        bf_idx = 0
        for iatom, atom in enumerate(molecule.atoms):
            n_bf = 0
            for bf in basis.basis_functions:
                if np.allclose(bf.center, atom.coords, atol=1e-10):
                    n_bf += 1
            atom_pops[iatom] = np.sum(populations[bf_idx:bf_idx + n_bf])
            bf_idx += n_bf

        charges = molecule.nuclear_charges - atom_pops

        return {
            'atomic_populations': atom_pops,
            'atomic_charges': charges,
            'total_electrons': np.sum(populations),
            'atoms': [a.symbol for a in molecule.atoms],
        }
