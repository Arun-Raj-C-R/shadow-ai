"""
Main DFT Engine -- Orchestrator
================================
High-level interface for HF/DFT calculations (restricted and unrestricted).
"""

import numpy as np
import time
from .atoms import Molecule
from .basis import build_basis
from .scf import SCFSolver, MullikenAnalysis
from .constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM


class DFTCalculation:
    """
    Main DFT/HF calculation class.

    Usage:
        from dft_engine.atoms import h2_molecule
        from dft_engine.dft_engine import DFTCalculation

        calc = DFTCalculation(molecule=h2_molecule(), basis_name='sto-3g', method='hf')
        result = calc.run()
        calc.print_summary()
    """

    def __init__(self, molecule, basis_name='sto-3g', method='hf',
                 functional='lda', max_iter=200, conv_threshold=1e-8,
                 diis=True, verbose=True, unrestricted=None):
        self.molecule = molecule
        self.basis_name = basis_name
        self.method = method
        self.functional = functional
        self.max_iter = max_iter
        self.conv_threshold = conv_threshold
        self.diis = diis
        self.verbose = verbose
        self.unrestricted = unrestricted
        self.result = None

    def run(self):
        if self.verbose:
            print(self.molecule.info())
        basis = build_basis(self.molecule, self.basis_name)
        if self.verbose:
            print(f"\nBasis set: {self.basis_name.upper()}, {basis.nbasis} functions")

        solver = SCFSolver(
            self.molecule, basis,
            method=self.method, functional=self.functional,
            max_iter=self.max_iter, e_threshold=self.conv_threshold,
            diis=self.diis, verbose=self.verbose,
            unrestricted=self.unrestricted,
        )
        self.result = solver.solve()
        self.result['basis'] = basis
        self.result['molecule'] = self.molecule

        mulliken = MullikenAnalysis.compute(
            self.molecule, basis, self.result['P'], self.result['S'])
        self.result['mulliken'] = mulliken
        return self.result

    def print_summary(self):
        if self.result is None:
            print("No calculation run yet.")
            return
        r = self.result
        ec = r['energy_components']
        meth = "UHF" if r.get('unrestricted') else r['method'].upper()
        if r.get('functional'):
            meth += f"/{r['functional'].upper()}"

        print(f"\n{'='*60}")
        print(f"  CALCULATION SUMMARY")
        print(f"{'='*60}")
        print(f"  System:       {self.molecule}")
        print(f"  Method:       {meth}  /  {self.basis_name.upper()}")
        print(f"  Converged:    {r['converged']}")
        print(f"  Iterations:   {r['iterations']}")
        print(f"  Time:         {r['time_seconds']:.2f} s")
        print(f"  Electrons:    {r.get('electron_count', 'N/A'):.4f} (expected {self.molecule.nelec})")
        print(f"\n  Energy Decomposition:")
        print(f"    One-electron:   {ec['E_1e']:16.10f} Ha  ({ec['E_1e']*HARTREE_TO_EV:12.6f} eV)")
        print(f"    Coulomb (J):    {ec['E_J']:16.10f} Ha  ({ec['E_J']*HARTREE_TO_EV:12.6f} eV)")
        if ec.get('E_K', 0) != 0:
            print(f"    Exchange (K):   {ec['E_K']:16.10f} Ha  ({ec['E_K']*HARTREE_TO_EV:12.6f} eV)")
        if ec.get('E_xc', 0) != 0:
            print(f"    XC Energy:      {ec['E_xc']:16.10f} Ha  ({ec['E_xc']*HARTREE_TO_EV:12.6f} eV)")
        print(f"    Nuclear rep.:   {ec['E_nn']:16.10f} Ha  ({ec['E_nn']*HARTREE_TO_EV:12.6f} eV)")
        print(f"    -------------------------------------")
        print(f"    TOTAL:          {ec['E_total']:16.10f} Ha  ({ec['E_total']*HARTREE_TO_EV:12.6f} eV)")

        print(f"\n  Orbital Energies (Ha):")
        nocc_a = self.molecule.nalpha
        nocc_b = self.molecule.nbeta
        if r.get('unrestricted'):
            print(f"    Alpha:")
            for i, e in enumerate(r['orbital_energies'][:min(10, len(r['orbital_energies']))]):
                occ = "occ" if i < nocc_a else "vir"
                print(f"      {i+1:3d}  {e:12.6f}  ({occ})")
            if 'orbital_energies_beta' in r:
                print(f"    Beta:")
                for i, e in enumerate(r['orbital_energies_beta'][:min(10, len(r['orbital_energies_beta']))]):
                    occ = "occ" if i < nocc_b else "vir"
                    print(f"      {i+1:3d}  {e:12.6f}  ({occ})")
        else:
            nocc = self.molecule.nelec // 2
            for i, e in enumerate(r['orbital_energies'][:min(10, len(r['orbital_energies']))]):
                occ = "occ" if i < nocc else "vir"
                print(f"    {i+1:3d}  {e:12.6f}  ({occ})")

        if 'mulliken' in r:
            m = r['mulliken']
            print(f"\n  Mulliken Charges:")
            for sym, q in zip(m['atoms'], m['atomic_charges']):
                print(f"    {sym:2s}  {q:8.4f}")
            print(f"    Total electrons: {m['total_electrons']:.4f}")
        print(f"{'='*60}\n")


def run_hf(molecule, basis_name='sto-3g', **kwargs):
    calc = DFTCalculation(molecule, basis_name, method='hf', **kwargs)
    return calc.run(), calc


def run_dft(molecule, basis_name='sto-3g', functional='lda', **kwargs):
    calc = DFTCalculation(molecule, basis_name, method='dft', functional=functional, **kwargs)
    return calc.run(), calc
