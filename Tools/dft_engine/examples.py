"""
Benchmark Calculations
=======================
Test the DFT engine against known reference values.

Reference energies (Hartree-Fock / STO-3G):
  H atom:   -0.4666 Ha  (exact: -0.5 Ha, STO-3G is limited)
  He atom:  -2.8077 Ha
  H2:       -1.1175 Ha  (exp bond length 0.74 A)
  H2O:      -74.9659 Ha (Szabo & Ostlund reference)
  LiH:      -7.8634 Ha

Note: STO-3G is a minimal basis -- results are qualitatively correct
but quantitatively limited. Larger basis sets (6-31G*, cc-pVDZ)
would give much better energies.
"""

import sys
import os
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_engine.atoms import (hydrogen_atom, helium_atom, h2_molecule,
                                water_molecule, lithium_hydride, Molecule)
from dft_engine.dft_engine import DFTCalculation, run_hf, run_dft
from dft_engine.properties import compute_dos, compute_homo_lumo
from dft_engine.constants import HARTREE_TO_EV


def benchmark_hydrogen_atom():
    """Hydrogen atom -- simplest quantum system."""
    print("\n" + "="*60)
    print("  BENCHMARK: Hydrogen Atom")
    print("="*60)

    mol = hydrogen_atom()
    calc = DFTCalculation(mol, 'sto-3g', method='hf')
    result = calc.run()
    calc.print_summary()

    ref = -0.4666
    error = abs(result['energy'] - ref)
    print(f"  Reference (HF/STO-3G): {ref:.4f} Ha")
    print(f"  Computed:              {result['energy']:.4f} Ha")
    print(f"  Error:                 {error:.4f} Ha ({error*HARTREE_TO_EV:.4f} eV)")
    return result


def benchmark_helium_atom():
    """He atom -- 2-electron system."""
    print("\n" + "="*60)
    print("  BENCHMARK: Helium Atom")
    print("="*60)

    mol = helium_atom()
    calc = DFTCalculation(mol, 'sto-3g', method='hf')
    result = calc.run()
    calc.print_summary()

    ref = -2.8077
    print(f"  Reference (HF/STO-3G): {ref:.4f} Ha")
    print(f"  Computed:              {result['energy']:.4f} Ha")
    return result


def benchmark_h2():
    """H2 molecule -- simplest chemical bond."""
    print("\n" + "="*60)
    print("  BENCHMARK: H2 Molecule")
    print("="*60)

    mol = h2_molecule(0.74)
    calc = DFTCalculation(mol, 'sto-3g', method='hf')
    result = calc.run()
    calc.print_summary()

    ref = -1.1175
    print(f"  Reference (HF/STO-3G): {ref:.4f} Ha")
    print(f"  Computed:              {result['energy']:.4f} Ha")

    # HOMO-LUMO
    hl = compute_homo_lumo(result['orbital_energies'], mol.nelec)
    print(f"  HOMO-LUMO gap: {hl['gap_eV']:.4f} eV")
    return result


def benchmark_water():
    """H2O -- standard benchmark for molecular DFT."""
    print("\n" + "="*60)
    print("  BENCHMARK: Water Molecule (H2O)")
    print("="*60)

    mol = water_molecule()
    calc = DFTCalculation(mol, 'sto-3g', method='hf')
    result = calc.run()
    calc.print_summary()

    ref = -74.9659
    print(f"  Reference (HF/STO-3G): {ref:.4f} Ha")
    print(f"  Computed:              {result['energy']:.4f} Ha")
    return result


def benchmark_h2_dft():
    """H2 with DFT/LDA for comparison."""
    print("\n" + "="*60)
    print("  BENCHMARK: H2 with LDA DFT")
    print("="*60)

    mol = h2_molecule(0.74)
    calc = DFTCalculation(mol, 'sto-3g', method='dft', functional='lda')
    result = calc.run()
    calc.print_summary()
    return result


def run_all_benchmarks():
    """Run all benchmark calculations."""
    print("\n" + "#"*60)
    print("#  DFT ENGINE -- BENCHMARK SUITE")
    print("#"*60)

    results = {}

    try:
        results['H'] = benchmark_hydrogen_atom()
    except Exception as e:
        print(f"  H atom FAILED: {e}")

    try:
        results['He'] = benchmark_helium_atom()
    except Exception as e:
        print(f"  He atom FAILED: {e}")

    try:
        results['H2'] = benchmark_h2()
    except Exception as e:
        print(f"  H2 FAILED: {e}")

    try:
        results['H2O'] = benchmark_water()
    except Exception as e:
        print(f"  H2O FAILED: {e}")

    try:
        results['H2_DFT'] = benchmark_h2_dft()
    except Exception as e:
        print(f"  H2 DFT FAILED: {e}")

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in results.items():
        status = "OK" if r.get('converged', False) else "FAIL"
        print(f"  [{status}] {name:10s}: E = {r['energy']:16.10f} Ha "
              f"({r['energy']*HARTREE_TO_EV:12.6f} eV)  "
              f"[{r['iterations']} iter, {r['time_seconds']:.1f}s]")

    return results


if __name__ == '__main__':
    run_all_benchmarks()
