# pymatgen_analysis_tool.py
"""
Pymatgen Analysis Tool (Shadow-Ready) - PEAK CAPACITY (v3.0)

This is the "physics and chemistry brain" module for Shadow. It is built 
from the core pymatgen namespace to provide "peak capacity" analysis.

It is designed to be imported by an AI orchestrator (Shadow) to perform
calculations on Pymatgen objects (Structure, BandStructure, DOS) that
are fetched by the `MaterialsProjectTool`.
"""

import logging
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple

# --- Import only the "peak capacity" 1% of Pymatgen ---

# For core data structures
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import Dos, CompleteDos
from pymatgen.entries.computed_entries import ComputedEntry

# For plotting band structures and DOS
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter

# For calculating theoretical solar cell efficiency (Shockley-Queisser)
from pymatgen.analysis.solar.slme import slme

# For phase diagram / stability analysis (ESSENTIAL for perovskites)
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, ComputedEntry
from pymatgen.analysis.reaction_calculator import BalancedReaction

# For creating new structures (e.g., surfaces, substitutions, defects)
from pymatgen.core.surface import SlabGenerator
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.transformations.site_transformations import RemoveSitesTransformation

# For "peak" electronic/crystal property analysis
from pymatgen.analysis.effective_mass import EffectiveMassCalculator
from pymatgen.analysis.wulff import WulffShape


class PymatgenAnalysisTool:
    """
    This class is a "tool" for your Shadow. It performs calculations
    on Pymatgen objects. It is stateless and all methods are static.
    """

    # --- 1. SOLAR CELL ANALYSIS ---

    @staticmethod
    def get_theoretical_efficiency(band_gap: float) -> Dict[str, Any]:
        logging.info(f"Calculating SLME for band_gap: {band_gap} eV")
        slme_results = slme(band_gap_ev=band_gap)
        return {
            "band_gap_ev": band_gap,
            "max_efficiency": slme_results[0],
            "max_voc_v": slme_results[1],
            "max_jsc_ma_cm2": slme_results[2],
            "max_ff": slme_results[3]
        }

    # --- 2. ELECTRONIC STRUCTURE ANALYSIS ---

    @staticmethod
    def calculate_effective_mass(bs: BandStructure) -> Dict[str, Any]:
        logging.info("Calculating effective mass...")
        try:
            emc = EffectiveMassCalculator(bs)
            eff_mass_data = emc.get_effective_masses()
            return eff_mass_data # Return the full dict
        except Exception as e:
            logging.error(f"Could not calculate effective mass: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_dielectric_constant(dielectric_data: Dict[str, Any]) -> Optional[float]:
        try:
            e_static = dielectric_data['e_static']
            e_ionic = dielectric_data['e_ionic']
            e_total_tensor = [[e_static[i][j] + e_ionic[i][j] for j in range(3)] for i in range(3)]
            avg_dielectric = (e_total_tensor[0][0] + e_total_tensor[1][1] + e_total_tensor[2][2]) / 3.0
            logging.info(f"Calculated average static dielectric constant: {avg_dielectric}")
            return avg_dielectric
        except Exception as e:
            logging.error(f"Could not parse dielectric data: {e}")
            return None

    # --- 3. THERMODYNAMIC & STABILITY ANALYSIS ---

    @staticmethod
    def get_phase_stability(entries: List[ComputedEntry]) -> Dict[str, Any]:
        logging.info(f"Building phase diagram for {len(entries)} entries.")
        if not entries:
            return {"error": "No entries provided."}
            
        pd = PhaseDiagram(entries)
        results = {}
        for entry in pd.all_entries:
            decomp, e_hull = pd.get_decomp_and_e_above_hull(entry)
            results[entry.composition.reduced_formula] = {
                "material_id": entry.entry_id,
                "is_stable": e_hull < 1e-6,
                "energy_above_hull_ev_atom": e_hull / entry.composition.num_atoms,
                "decomposition": {k.composition.reduced_formula: f"{v:.3f}" for k, v in decomp.items()}
            }
        return results

    @staticmethod
    def calculate_reaction_energy(entries: List[ComputedEntry], reaction_str: str) -> Optional[float]:
        logging.info(f"Calculating energy for reaction: {reaction_str}")
        try:
            reactants_str, products_str = reaction_str.split("->")
            # Create dummy entries from compositions just for reaction balancing
            reactants = [ComputedEntry(c, 0) for c in reactants_str.split("+")]
            products = [ComputedEntry(c, 0) for c in products_str.split("+")]
            
            reaction = BalancedReaction(reactants, products)
            
            # Create a PhaseDiagram to get energies for all compounds
            pd = PhaseDiagram(entries)
            
            # This calculates the energy *using the convex hull*
            energy = pd.get_reaction_energy(reaction)
            return energy / reaction.get_coeff(reactants[0].composition)
        
        except Exception as e:
            logging.error(f"Could not calculate reaction energy: {e}")
            return None

    # --- 4. PLOTTING & VISUALIZATION ---

    @staticmethod
    def plot_band_structure(bs: BandStructure, save_path: str = "band_structure.png") -> str:
        logging.info(f"Plotting band structure, saving to {save_path}")
        plotter = BSPlotter(bs)
        plt = plotter.get_plot(vbm_cbm_marker=True)
        formula = bs.structure.composition.reduced_formula
        plt.title(f"Band Structure ({formula})")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    @staticmethod
    def plot_dos(dos: CompleteDos, save_path: str = "dos.png") -> str:
        logging.info(f"Plotting DOS, saving to {save_path}")
        plotter = DosPlotter()
        plotter.add_dos("Total DOS", dos)
        plotter.add_dos_dict(dos.get_element_dos()) # Element-projected
        plt = plotter.get_plot()
        formula = dos.structure.composition.reduced_formula
        plt.title(f"Density of States ({formula})")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    @staticmethod
    def get_wulff_shape(structure: Structure, surface_energies: Dict[Tuple[int, int, int], float]) -> str:
        logging.info("Calculating Wulff shape...")
        lattice = structure.lattice
        miller_indices = list(surface_energies.keys())
        energies = list(surface_energies.values())
        
        wulff = WulffShape(lattice, miller_indices, energies)
        
        report = "Wulff Shape Analysis:\n"
        for area_frac in wulff.area_fraction_dict:
            report += f"  - Facet {area_frac['miller']}: {area_frac['area_fraction']*100:.2f}% of total area\n"
        return report

    # --- 5. STRUCTURE MANIPULATION ---

    @staticmethod
    def create_surface_slab(structure: Structure, miller_index: List[int]) -> Structure:
        logging.info(f"Generating {miller_index} slab for {structure.formula}")
        slab_gen = SlabGenerator(
            initial_structure=structure,
            miller_index=miller_index,
            min_slab_size=10.0,  # 10 Angstroms thick
            min_vacuum_size=15.0 # 15 Angstroms of vacuum
        )
        # Get the first (most common) termination
        slab = slab_gen.get_slabs()[0]
        return slab

    @staticmethod
    def create_doped_structure(structure: Structure, 
                               species_to_replace: str,
                               dopant: str, 
                               fraction: float) -> Structure:
        logging.info(f"Creating {fraction*100}% {dopant}-doped {structure.formula}")
        trans = SubstitutionTransformation({
            species_to_replace: {dopant: fraction}
        })
        doped_structure = trans.apply_transformation(structure)
        return doped_structure.get_sorted_structure()
    
    @staticmethod
    def create_vacancy(structure: Structure, site_index: int) -> Structure:
        logging.info(f"Creating vacancy at site {site_index} (Species: {structure[site_index].specie})")
        trans = RemoveSitesTransformation([site_index])
        return trans.apply_transformation(structure)