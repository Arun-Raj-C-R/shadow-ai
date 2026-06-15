"""
DFT Computational Engine
========================
A first-principles Density Functional Theory implementation in Python.

This package implements the Kohn-Sham DFT formalism from scratch,
including analytical Gaussian integrals, SCF solver with DIIS,
LDA/GGA exchange-correlation functionals, and numerical integration.

Architecture inspired by PySCF, Quantum ESPRESSO, and GAUSSIAN.

Modules:
    constants    â€” Physical constants and unit conversions
    atoms        â€” Atomic data structures and molecular geometry
    basis        â€” Gaussian basis functions (STO-3G, 6-31G, etc.)
    integrals    â€” One-electron and two-electron integral evaluation
    grid         â€” Numerical integration grids (Becke + Lebedev)
    xc           â€” Exchange-correlation functionals (LDA, PBE)
    hamiltonian  â€” Hamiltonian / Fock / Kohn-Sham matrix construction
    scf          â€” Self-Consistent Field solver with DIIS
    dft_engine   â€” Main DFT calculation orchestrator
    optimization â€” Geometry optimization (BFGS, conjugate gradient)
    periodic     â€” Periodic boundary conditions and k-point sampling
    properties   â€” Band structure, DOS, charge analysis
    pseudopotential â€” Pseudopotential framework (NC, US, PAW)
    examples     â€” Benchmark calculations (H, H2, H2O, Si)
"""

__version__ = "1.0.0"
__author__ = "Shadow AI / SHADOW DFT Engine"
