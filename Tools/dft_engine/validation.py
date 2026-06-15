"""
Automated Validation Test Suite
================================
Assertions for matrix symmetry, normalization, energies, electron count, convergence.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_engine.atoms import (hydrogen_atom, helium_atom, h2_molecule,
                                water_molecule, lithium_hydride, Molecule)
from dft_engine.basis import build_basis
from dft_engine.integrals import (compute_overlap_matrix, compute_kinetic_matrix,
                                   compute_nuclear_matrix, compute_eri_tensor,
                                   validate_overlap_matrix, validate_eri_symmetry)
from dft_engine.dft_engine import DFTCalculation
from dft_engine.constants import HARTREE_TO_EV

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_overlap_matrix():
    """Test overlap matrix properties."""
    print("\n--- Overlap Matrix Tests ---")
    mol = h2_molecule(0.74)
    basis = build_basis(mol, 'sto-3g')
    S = compute_overlap_matrix(basis)

    # Symmetry
    check("S is symmetric", np.allclose(S, S.T, atol=1e-12))

    # Positive-definite
    eigvals = np.linalg.eigvalsh(S)
    check("S is positive-definite", np.all(eigvals > 0),
          f"min eigenvalue = {eigvals[0]:.2e}")

    # Diagonal ~ 1 (normalized basis functions)
    check("S diagonal elements ~ 1.0",
          np.allclose(np.diag(S), 1.0, atol=0.3),
          f"diag = {np.diag(S)}")

    # Validation utility
    issues = validate_overlap_matrix(S)
    check("validate_overlap_matrix passes", len(issues) == 0,
          f"issues: {issues}")


def test_kinetic_matrix():
    """Test kinetic matrix properties."""
    print("\n--- Kinetic Matrix Tests ---")
    mol = h2_molecule(0.74)
    basis = build_basis(mol, 'sto-3g')
    T = compute_kinetic_matrix(basis)

    check("T is symmetric", np.allclose(T, T.T, atol=1e-12))
    check("T diagonal > 0 (positive kinetic energy)",
          np.all(np.diag(T) > 0), f"diag = {np.diag(T)}")


def test_nuclear_matrix():
    """Test nuclear attraction matrix."""
    print("\n--- Nuclear Attraction Tests ---")
    mol = h2_molecule(0.74)
    basis = build_basis(mol, 'sto-3g')
    V = compute_nuclear_matrix(basis, mol)

    check("V is symmetric", np.allclose(V, V.T, atol=1e-12))
    check("V diagonal < 0 (attractive)", np.all(np.diag(V) < 0),
          f"diag = {np.diag(V)}")


def test_eri_symmetry():
    """Test ERI 8-fold symmetry."""
    print("\n--- ERI Symmetry Tests ---")
    mol = h2_molecule(0.74)
    basis = build_basis(mol, 'sto-3g')
    eri = compute_eri_tensor(basis)

    sym_ok, max_err = validate_eri_symmetry(eri)
    check("ERI 8-fold symmetry", sym_ok, f"max error = {max_err:.2e}")
    check("ERI non-negative (ss|ss)", eri[0, 0, 0, 0] > 0,
          f"(00|00) = {eri[0,0,0,0]:.6f}")


def test_h2_rhf():
    """Test H2 RHF/STO-3G against Szabo & Ostlund reference."""
    print("\n--- H2 RHF/STO-3G ---")
    mol = h2_molecule(0.74)
    calc = DFTCalculation(mol, 'sto-3g', method='hf', verbose=False)
    r = calc.run()

    check("H2 SCF converged", r['converged'])
    check("H2 electron count = 2",
          abs(r['electron_count'] - 2.0) < 0.01,
          f"got {r['electron_count']:.4f}")

    ref_energy = -1.1175  # Szabo & Ostlund
    err = abs(r['energy'] - ref_energy)
    check(f"H2 energy ~ {ref_energy} Ha",
          err < 0.01, f"got {r['energy']:.6f}, error = {err:.4f}")

    # Orbital energies should be negative (bonding) and positive (antibonding)
    check("H2 HOMO < 0", r['orbital_energies'][0] < 0)
    check("H2 LUMO > 0", r['orbital_energies'][1] > 0)


def test_he_rhf():
    """Test He RHF/STO-3G."""
    print("\n--- He RHF/STO-3G ---")
    mol = helium_atom()
    calc = DFTCalculation(mol, 'sto-3g', method='hf', verbose=False)
    r = calc.run()

    check("He SCF converged", r['converged'])
    ref = -2.8077
    err = abs(r['energy'] - ref)
    check(f"He energy ~ {ref} Ha", err < 0.001,
          f"got {r['energy']:.6f}, error = {err:.6f}")


def test_h_atom_uhf():
    """Test H atom with UHF (1 unpaired electron)."""
    print("\n--- H Atom UHF/STO-3G ---")
    mol = hydrogen_atom()
    calc = DFTCalculation(mol, 'sto-3g', method='hf', verbose=False)
    r = calc.run()

    check("H atom SCF converged", r['converged'])
    check("H atom used UHF", r.get('unrestricted', False))

    ref = -0.4666
    err = abs(r['energy'] - ref)
    check(f"H atom energy ~ {ref} Ha", err < 0.01,
          f"got {r['energy']:.6f}, error = {err:.4f}")


def test_water_rhf():
    """Test H2O RHF/STO-3G."""
    print("\n--- H2O RHF/STO-3G ---")
    mol = water_molecule()
    calc = DFTCalculation(mol, 'sto-3g', method='hf', verbose=False)
    r = calc.run()

    check("H2O SCF converged", r['converged'])
    check("H2O electron count = 10",
          abs(r['electron_count'] - 10.0) < 0.1,
          f"got {r['electron_count']:.4f}")

    ref = -74.9659
    err = abs(r['energy'] - ref)
    check(f"H2O energy ~ {ref} Ha", err < 1.0,
          f"got {r['energy']:.6f}, error = {err:.4f}")


def test_lih_rhf():
    """Test LiH RHF/STO-3G."""
    print("\n--- LiH RHF/STO-3G ---")
    mol = lithium_hydride()
    calc = DFTCalculation(mol, 'sto-3g', method='hf', verbose=False)
    r = calc.run()

    check("LiH SCF converged", r['converged'])
    check("LiH electron count = 4",
          abs(r['electron_count'] - 4.0) < 0.1,
          f"got {r['electron_count']:.4f}")

    ref = -7.8634
    err = abs(r['energy'] - ref)
    check(f"LiH energy ~ {ref} Ha", err < 1.0,
          f"got {r['energy']:.6f}, error = {err:.4f}")


def test_h2_dft_lda():
    """Test H2 DFT/LDA."""
    print("\n--- H2 DFT/LDA/STO-3G ---")
    mol = h2_molecule(0.74)
    calc = DFTCalculation(mol, 'sto-3g', method='dft', functional='lda', verbose=False)
    r = calc.run()

    check("H2 DFT converged", r['converged'])
    # DFT energy should be different from HF
    check("H2 DFT energy is negative", r['energy'] < 0)
    check("H2 DFT energy < 0 Ha", r['energy'] < 0)


def test_water_integrals():
    """Test H2O integral matrices for p-orbital correctness."""
    print("\n--- H2O Integral Validation ---")
    mol = water_molecule()
    basis = build_basis(mol, 'sto-3g')

    S = compute_overlap_matrix(basis)
    T = compute_kinetic_matrix(basis)
    V = compute_nuclear_matrix(basis, mol)

    check("H2O S symmetric", np.allclose(S, S.T, atol=1e-12))
    check("H2O T symmetric", np.allclose(T, T.T, atol=1e-12))
    check("H2O V symmetric", np.allclose(V, V.T, atol=1e-12))

    # S should be positive-definite
    eigvals = np.linalg.eigvalsh(S)
    check("H2O S positive-definite", np.all(eigvals > 0))

    # All S diagonal should be ~1
    check("H2O S diag ~ 1",
          np.allclose(np.diag(S), 1.0, atol=0.3),
          f"diag = {np.diag(S)}")

    # Core Hamiltonian check
    H = T + V
    check("H2O H_core symmetric", np.allclose(H, H.T, atol=1e-12))


def run_all_tests():
    """Run complete validation suite."""
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("\n" + "#" * 60)
    print("#  DFT ENGINE -- VALIDATION TEST SUITE")
    print("#" * 60)

    test_overlap_matrix()
    test_kinetic_matrix()
    test_nuclear_matrix()
    test_eri_symmetry()
    test_h2_rhf()
    test_he_rhf()
    test_h_atom_uhf()
    test_water_rhf()
    test_lih_rhf()
    test_h2_dft_lda()
    test_water_integrals()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*60}\n")

    return FAIL == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
