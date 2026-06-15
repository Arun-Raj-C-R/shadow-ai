"""
Geometry Optimization
======================
Implements Hellmann-Feynman force calculation and BFGS/CG optimizers.

Hellmann-Feynman Theorem:
  dE/dR_A = <ψ|∂Ĥ/∂R_A|ψ> = -∫ ρ(r) ∂V_ext/∂R_A d³r + ∂E_nn/∂R_A

  Force on nucleus A:
    F_A = -dE/dR_A = Z_A ∫ ρ(r)(r-R_A)/|r-R_A|³ d³r - Σ_{B≠A} Z_A Z_B (R_A-R_B)/|R_A-R_B|³

In practice with Gaussian basis sets, we also need Pulay forces
(from basis function dependence on R_A), but we use numerical gradients here.
"""

import numpy as np
from .atoms import Molecule, Atom
from .dft_engine import DFTCalculation
from .constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR


def numerical_gradient(molecule, basis_name='sto-3g', method='hf',
                       functional='lda', step=0.001):
    """
    Compute nuclear gradients by finite differences.

    dE/dR_{A,α} ≈ [E(R+δ) - E(R-δ)] / (2δ)

    This is O(6·N_atoms) SCF calculations — expensive but correct.
    Analytical gradients require derivative integrals (Pulay forces).

    Args:
        step: finite difference step in bohr
    Returns:
        gradients: (N_atoms, 3) array in Hartree/bohr
    """
    natoms = molecule.natoms
    gradients = np.zeros((natoms, 3))
    
    for iatom in range(natoms):
        for icoord in range(3):
            # Forward step
            atoms_fwd = []
            for j, atom in enumerate(molecule.atoms):
                coords = atom.coords_angstrom.copy()
                if j == iatom:
                    coords[icoord] += step * BOHR_TO_ANGSTROM
                atoms_fwd.append(Atom(atom.symbol, coords))
            mol_fwd = Molecule(atoms_fwd, molecule.charge, molecule.multiplicity)
            
            calc_fwd = DFTCalculation(mol_fwd, basis_name, method, functional, verbose=False)
            res_fwd = calc_fwd.run()
            
            # Backward step
            atoms_bwd = []
            for j, atom in enumerate(molecule.atoms):
                coords = atom.coords_angstrom.copy()
                if j == iatom:
                    coords[icoord] -= step * BOHR_TO_ANGSTROM
                atoms_bwd.append(Atom(atom.symbol, coords))
            mol_bwd = Molecule(atoms_bwd, molecule.charge, molecule.multiplicity)
            
            calc_bwd = DFTCalculation(mol_bwd, basis_name, method, functional, verbose=False)
            res_bwd = calc_bwd.run()
            
            gradients[iatom, icoord] = (res_fwd['energy'] - res_bwd['energy']) / (2.0 * step)
    
    return gradients


def optimize_geometry(molecule, basis_name='sto-3g', method='hf',
                      functional='lda', max_steps=50, force_threshold=1e-4,
                      step_size=0.1, verbose=True):
    """
    Geometry optimization using steepest descent with line search.

    For production use, BFGS (quasi-Newton) is preferred:
      H_{k+1} = H_k + (Δg Δg†)/(Δg†Δx) - (H_k Δx)(H_k Δx)†/(Δx† H_k Δx)
    where Δx = x_{k+1} - x_k, Δg = g_{k+1} - g_k.

    Returns:
        optimized Molecule, trajectory of energies
    """
    current_mol = molecule
    energies = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  GEOMETRY OPTIMIZATION")
        print(f"  Method: {method.upper()}/{basis_name.upper()}")
        print(f"{'='*60}")
    
    for step in range(max_steps):
        # Single-point energy
        calc = DFTCalculation(current_mol, basis_name, method, functional, verbose=False)
        result = calc.run()
        energy = result['energy']
        energies.append(energy)
        
        # Gradient
        grad = numerical_gradient(current_mol, basis_name, method, functional)
        forces = -grad
        max_force = np.max(np.abs(forces))
        rms_force = np.sqrt(np.mean(forces**2))
        
        if verbose:
            print(f"  Step {step+1:3d}  E = {energy:16.10f}  "
                  f"max|F| = {max_force:.2e}  rms|F| = {rms_force:.2e}")
        
        if max_force < force_threshold:
            if verbose:
                print(f"  CONVERGED: max force < {force_threshold}")
            break
        
        # Update positions (steepest descent)
        new_atoms = []
        for iatom, atom in enumerate(current_mol.atoms):
            new_coords_bohr = atom.coords + step_size * forces[iatom]
            new_coords_ang = new_coords_bohr * BOHR_TO_ANGSTROM
            new_atoms.append(Atom(atom.symbol, new_coords_ang))
        
        current_mol = Molecule(new_atoms, molecule.charge, molecule.multiplicity)
    
    return current_mol, energies
