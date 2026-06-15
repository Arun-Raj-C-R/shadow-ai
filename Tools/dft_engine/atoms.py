"""
Atomic Data Structures and Molecular Geometry
==============================================

Defines Atom and Molecule classes for DFT calculations.

Born-Oppenheimer Approximation:
    Since M_nucleus >> m_electron (by factor ~1836 for hydrogen),
    nuclear and electronic motion can be separated:
    
    Ψ_total(R, r) ≈ Φ_nuclear(R) · ψ_electronic(r; R)
    
    Electrons see nuclei as fixed point charges → parametric dependence.
    This allows us to solve the electronic Schrödinger equation at 
    fixed nuclear positions R:
    
    Ĥ_elec ψ(r; R) = E_elec(R) ψ(r; R)
    
    where E_elec(R) forms the potential energy surface (PES).

The many-electron Hamiltonian in atomic units:
    Ĥ = -½ Σᵢ ∇ᵢ² - Σᵢ Σ_A Z_A/|rᵢ - R_A| + Σᵢ<ⱼ 1/|rᵢ - rⱼ| + Σ_A<_B Z_A·Z_B/|R_A - R_B|
    
    Term 1: Electron kinetic energy
    Term 2: Electron-nucleus attraction (external potential)
    Term 3: Electron-electron repulsion (Coulomb)
    Term 4: Nuclear-nuclear repulsion (classical, constant for fixed nuclei)
"""

import numpy as np
from typing import List, Optional, Tuple
from .constants import (ATOMIC_NUMBERS, ATOMIC_MASSES, ANGSTROM_TO_BOHR, 
                         BOHR_TO_ANGSTROM, COVALENT_RADII)


class Atom:
    """
    Represents a single atom with nuclear charge and position.
    
    Attributes:
        symbol:  Element symbol (e.g., 'H', 'O')
        Z:       Atomic number (nuclear charge)
        mass:    Atomic mass in amu
        coords:  Nuclear coordinates in BOHR (internal units)
        coords_angstrom: Nuclear coordinates in Angstrom
    """
    
    def __init__(self, symbol: str, coords_angstrom: np.ndarray):
        """
        Initialize atom.
        
        Args:
            symbol: Element symbol
            coords_angstrom: (3,) array of [x, y, z] in Angstroms
        """
        self.symbol = symbol.strip().capitalize()
        
        if self.symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element: {self.symbol}")
        
        self.Z = ATOMIC_NUMBERS[self.symbol]
        self.mass = ATOMIC_MASSES.get(self.symbol, 2.0 * self.Z)
        
        # Store in both units; internal = bohr
        self.coords_angstrom = np.array(coords_angstrom, dtype=np.float64)
        self.coords = self.coords_angstrom * ANGSTROM_TO_BOHR  # Convert to bohr
    
    def __repr__(self):
        x, y, z = self.coords_angstrom
        return f"Atom({self.symbol}, [{x:.6f}, {y:.6f}, {z:.6f}] A)"
    
    @property
    def R(self) -> np.ndarray:
        """Nuclear position vector in bohr (alias for self.coords)."""
        return self.coords


class Molecule:
    """
    Collection of atoms forming a molecular system.
    
    Handles:
        - Geometry specification
        - Nuclear repulsion energy
        - Center of mass
        - Number of electrons
        - Charge and spin multiplicity
    """
    
    def __init__(self, atoms: List[Atom] = None, charge: int = 0, 
                 multiplicity: int = 1, name: str = ""):
        """
        Initialize molecule.
        
        Args:
            atoms: List of Atom objects
            charge: Net molecular charge (0 = neutral)
            multiplicity: Spin multiplicity (2S+1). 1 = singlet, 2 = doublet, etc.
            name: Optional label
        """
        self.atoms = atoms or []
        self.charge = charge
        self.multiplicity = multiplicity
        self.name = name
    
    @classmethod
    def from_xyz(cls, xyz_string: str, charge: int = 0, multiplicity: int = 1,
                 name: str = "") -> 'Molecule':
        """
        Build molecule from XYZ-format string.
        
        Format:
            H  0.0  0.0  0.0
            H  0.0  0.0  0.74
        
        Coordinates in Angstroms.
        """
        atoms = []
        for line in xyz_string.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                atoms.append(Atom(symbol, coords))
            elif len(parts) == 1:
                # Skip atom count line or blank
                continue
        
        return cls(atoms=atoms, charge=charge, multiplicity=multiplicity, name=name)
    
    @classmethod
    def from_atoms_list(cls, atoms_list: List[Tuple[str, List[float]]], 
                        charge: int = 0, multiplicity: int = 1,
                        name: str = "") -> 'Molecule':
        """
        Build from list of (symbol, [x, y, z]) tuples. Coords in Angstrom.
        
        Example:
            mol = Molecule.from_atoms_list([
                ('H', [0.0, 0.0, 0.0]),
                ('H', [0.0, 0.0, 0.74]),
            ])
        """
        atoms = [Atom(sym, np.array(c)) for sym, c in atoms_list]
        return cls(atoms=atoms, charge=charge, multiplicity=multiplicity, name=name)
    
    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)
    
    @property
    def nelec(self) -> int:
        """
        Total number of electrons.
        
        N_elec = Σ_A Z_A - charge
        
        A neutral molecule has N_elec = sum of atomic numbers.
        """
        return sum(atom.Z for atom in self.atoms) - self.charge
    
    @property
    def nalpha(self) -> int:
        """Number of alpha (spin-up) electrons."""
        # N_alpha = (N_elec + n_unpaired) / 2
        # n_unpaired = multiplicity - 1
        n_unpaired = self.multiplicity - 1
        return (self.nelec + n_unpaired) // 2
    
    @property
    def nbeta(self) -> int:
        """Number of beta (spin-down) electrons."""
        return self.nelec - self.nalpha
    
    @property
    def nuclear_charges(self) -> np.ndarray:
        """Array of nuclear charges Z_A."""
        return np.array([atom.Z for atom in self.atoms])
    
    @property
    def nuclear_coords(self) -> np.ndarray:
        """(N_atoms, 3) array of nuclear positions in bohr."""
        return np.array([atom.coords for atom in self.atoms])
    
    def nuclear_repulsion_energy(self) -> float:
        """
        Nuclear-nuclear repulsion energy (in Hartree).
        
        E_nn = Σ_{A<B} Z_A · Z_B / |R_A - R_B|
        
        This is a classical Coulomb energy between point charges.
        In atomic units, the Coulomb constant k = 1/(4πε₀) = 1.
        
        Computational complexity: O(N_atoms²)
        For large systems, Ewald summation would be needed.
        """
        E_nn = 0.0
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                R_ij = np.linalg.norm(self.atoms[i].coords - self.atoms[j].coords)
                if R_ij < 1e-10:
                    raise ValueError(
                        f"Atoms {i} ({self.atoms[i].symbol}) and {j} ({self.atoms[j].symbol}) "
                        f"are at the same position!"
                    )
                E_nn += self.atoms[i].Z * self.atoms[j].Z / R_ij
        return E_nn
    
    def center_of_mass(self) -> np.ndarray:
        """Center of mass in bohr."""
        total_mass = sum(a.mass for a in self.atoms)
        com = np.zeros(3)
        for atom in self.atoms:
            com += atom.mass * atom.coords
        return com / total_mass
    
    def center_of_charge(self) -> np.ndarray:
        """Center of nuclear charge in bohr."""
        total_Z = sum(a.Z for a in self.atoms)
        coc = np.zeros(3)
        for atom in self.atoms:
            coc += atom.Z * atom.coords
        return coc / total_Z
    
    def max_distance(self) -> float:
        """Maximum interatomic distance in bohr."""
        max_d = 0.0
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                d = np.linalg.norm(self.atoms[i].coords - self.atoms[j].coords)
                max_d = max(max_d, d)
        return max_d
    
    def info(self) -> str:
        """Human-readable molecule summary."""
        lines = [
            f"Molecule: {self.name or 'unnamed'}",
            f"  Atoms: {self.natoms}",
            f"  Electrons: {self.nelec} (alpha={self.nalpha}, beta={self.nbeta})",
            f"  Charge: {self.charge}",
            f"  Multiplicity: {self.multiplicity}",
            f"  E_nn: {self.nuclear_repulsion_energy():.10f} Hartree",
            f"  Geometry (Angstrom):"
        ]
        for atom in self.atoms:
            x, y, z = atom.coords_angstrom
            lines.append(f"    {atom.symbol:2s}  {x:12.8f}  {y:12.8f}  {z:12.8f}")
        return '\n'.join(lines)
    
    def __repr__(self):
        formula = {}
        for atom in self.atoms:
            formula[atom.symbol] = formula.get(atom.symbol, 0) + 1
        formula_str = ''.join(f"{k}{v if v > 1 else ''}" for k, v in sorted(formula.items()))
        return f"Molecule({formula_str}, charge={self.charge}, mult={self.multiplicity})"


# =====================================================================
# PREDEFINED MOLECULES FOR BENCHMARKING
# =====================================================================

def hydrogen_atom() -> Molecule:
    """H atom — simplest quantum system, exact solution known."""
    return Molecule.from_atoms_list(
        [('H', [0.0, 0.0, 0.0])],
        charge=0, multiplicity=2, name="Hydrogen atom"
    )

def h2_molecule(bond_length_angstrom: float = 0.74) -> Molecule:
    """
    H₂ molecule — simplest molecular system.
    
    Experimental bond length: 0.74 Å
    Experimental dissociation energy: 4.75 eV
    Exact non-relativistic energy: -1.1745 Hartree (Kolos & Wolniewicz)
    HF limit: -1.1336 Hartree
    """
    return Molecule.from_atoms_list([
        ('H', [0.0, 0.0, 0.0]),
        ('H', [0.0, 0.0, bond_length_angstrom]),
    ], charge=0, multiplicity=1, name="H2")

def helium_atom() -> Molecule:
    """He atom — 2-electron system, no exact solution."""
    return Molecule.from_atoms_list(
        [('He', [0.0, 0.0, 0.0])],
        charge=0, multiplicity=1, name="Helium atom"
    )

def water_molecule() -> Molecule:
    """
    H₂O — standard benchmark molecule.
    
    Experimental geometry:
        O-H bond: 0.9572 Å
        H-O-H angle: 104.52°
    
    HF/STO-3G energy: ~-74.965 Hartree
    Experimental atomization energy: 9.51 eV
    """
    # Place O at origin, H atoms at experimental geometry
    angle_rad = np.radians(104.52)
    r_OH = 0.9572  # Angstroms
    
    H1_x = r_OH * np.sin(angle_rad / 2)
    H1_z = r_OH * np.cos(angle_rad / 2)
    
    return Molecule.from_atoms_list([
        ('O', [0.0, 0.0, 0.0]),
        ('H', [H1_x, 0.0, H1_z]),
        ('H', [-H1_x, 0.0, H1_z]),
    ], charge=0, multiplicity=1, name="Water")

def lithium_hydride() -> Molecule:
    """LiH — polar diatomic, good test for charge distribution."""
    return Molecule.from_atoms_list([
        ('Li', [0.0, 0.0, 0.0]),
        ('H', [0.0, 0.0, 1.5949]),
    ], charge=0, multiplicity=1, name="LiH")

def methane_molecule() -> Molecule:
    """CH₄ — tetrahedral molecule."""
    # Tetrahedral geometry, C-H bond = 1.089 Å
    d = 1.089 / np.sqrt(3)
    return Molecule.from_atoms_list([
        ('C', [0.0, 0.0, 0.0]),
        ('H', [d, d, d]),
        ('H', [d, -d, -d]),
        ('H', [-d, d, -d]),
        ('H', [-d, -d, d]),
    ], charge=0, multiplicity=1, name="Methane")
