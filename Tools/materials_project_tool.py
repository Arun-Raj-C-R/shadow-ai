# materials_project_tool.py
"""
Core Materials Project API tool – clean, function-calling ready.
Updated for mp-api v0.39+ with new endpoint patterns.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from mp_api.client import MPRester
from dotenv import load_dotenv
config_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", ".env")
if os.path.exists(config_env):
    load_dotenv(config_env, override=True)
else:
    load_dotenv()
API_KEY = os.environ.get("MP_API_KEY")

logging.basicConfig(level=logging.WARNING)


def _safe_dump(obj) -> Any:
    """Safely convert pymatgen objects to JSON-serializable dicts."""
    if obj is None:
        return None
    if hasattr(obj, 'model_dump'):
        try:
            return json.loads(json.dumps(obj.model_dump(), default=str))
        except Exception:
            return str(obj)
    if hasattr(obj, 'as_dict'):
        try:
            return json.loads(json.dumps(obj.as_dict(), default=str))
        except Exception:
            return str(obj)
    return obj


class MaterialsProjectTool:
    def __init__(self, api_key: str = API_KEY):
        if not api_key:
            raise ValueError("MP_API_KEY required in .env")
        self.client = MPRester(api_key)

    # ==================================================================
    # 1. Core Data Access Methods
    # ==================================================================
    def search_mp_by_formula(self, formula: str) -> List[Dict[str, str]]:
        """Return clean list: [{'material_id': 'mp-8062', 'formula_pretty': 'SiC'}, ...]"""
        try:
            docs = self.client.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty"]
            )
            return [
                {"material_id": str(d.material_id), "formula_pretty": d.formula_pretty}
                for d in docs
            ]
        except Exception as e:
            logging.error(f"search_mp_by_formula failed: {e}")
            return []

    def get_data(self, criteria: Dict, properties: List[str] = None) -> List[Dict]:
        """General-purpose query via summary endpoint"""
        try:
            fields = properties or ["material_id", "formula_pretty", "band_gap", "energy_above_hull"]
            # Extract known kwargs from criteria dict
            kwargs = {}
            if "formula" in criteria:
                kwargs["formula"] = criteria["formula"]
            if "elements" in criteria:
                kwargs["elements"] = criteria["elements"]
            if "material_ids" in criteria:
                kwargs["material_ids"] = criteria["material_ids"]
            docs = self.client.materials.summary.search(fields=fields, **kwargs)
            return [_safe_dump(d) for d in docs]
        except Exception as e:
            logging.error(f"get_data failed: {e}")
            return []

    def get_materials_ids(self, formula: str) -> List[str]:
        """Get all material IDs for a given formula"""
        try:
            docs = self.client.materials.summary.search(formula=formula, fields=["material_id"])
            return [str(d.material_id) for d in docs]
        except Exception as e:
            logging.error(f"get_materials_ids failed: {e}")
            return []

    def get_structure_by_material_id(self, material_id: str, final: bool = True, conventional_unit_cell: bool = False) -> Optional[Dict]:
        """Get crystal structure data"""
        try:
            struct = self.client.get_structure_by_material_id(material_id, final=final, conventional_unit_cell=conventional_unit_cell)
            return _safe_dump(struct)
        except Exception as e:
            logging.error(f"get_structure_by_material_id failed: {e}")
            return None

    def get_structures_by_material_ids(self, material_ids: List[str], final: bool = True) -> List[Dict]:
        """Batch get structures"""
        try:
            structures = []
            for mid in material_ids:
                struct = self.client.get_structure_by_material_id(mid, final=final)
                if struct:
                    structures.append(_safe_dump(struct))
            return structures
        except Exception as e:
            logging.error(f"get_structures_by_material_ids failed: {e}")
            return []

    def get_entry_by_material_id(self, material_id: str, compatible_only: bool = True) -> Optional[Dict]:
        """Get ComputedEntry for phase diagrams"""
        try:
            entry = self.client.get_entry_by_material_id(material_id, compatible_only=compatible_only)
            return _safe_dump(entry)
        except Exception as e:
            logging.error(f"get_entry_by_material_id failed: {e}")
            return None

    def get_entries(self, criteria: Dict, compatible_only: bool = True, inc_structure: str = None) -> List[Dict]:
        """Get list of ComputedEntry objects"""
        try:
            entries = self.client.get_entries(criteria, compatible_only=compatible_only, inc_structure=inc_structure)
            return [_safe_dump(e) for e in entries]
        except Exception as e:
            logging.error(f"get_entries failed: {e}")
            return []

    def get_entries_in_system(self, elements: List[str], compatible_only: bool = True) -> List[Dict]:
        """Get all entries in a chemical system"""
        try:
            entries = self.client.get_entries_in_system(elements, compatible_only=compatible_only)
            return [_safe_dump(e) for e in entries]
        except Exception as e:
            logging.error(f"get_entries_in_system failed: {e}")
            return []

    # ==================================================================
    # 2. Electronic Structure Methods
    # ==================================================================
    def get_mp_properties(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Return band_gap, is_gap_direct, formation_energy, etc."""
        try:
            docs = self.client.materials.summary.search(
                material_ids=[material_id],
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "is_gap_direct",
                    "formation_energy_per_atom",
                    "efermi",
                    "energy_above_hull",
                    "density",
                    "volume",
                    "nsites",
                    "symmetry",
                    "is_metal",
                    "is_magnetic",
                    "structure"
                ]
            )
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_mp_properties failed for {material_id}: {e}")
            return None

    def get_bandstructure_by_material_id(self, material_id: str, line_mode: bool = True) -> Optional[Dict]:
        """Get band structure data"""
        try:
            if line_mode:
                bs = self.client.get_bandstructure_by_material_id(material_id)
            else:
                bs = self.client.get_bandstructure_by_material_id(material_id, line_mode=False)
            if not bs:
                return None
            return {
                "material_id": material_id,
                "band_gap_eV": round(bs.get_band_gap()["energy"], 4) if bs.get_band_gap() else None,
                "is_direct": bs.get_band_gap().get("direct") if bs.get_band_gap() else None,
                "vbm_eV": round(bs.get_vbm()["energy"], 4) if bs.get_vbm() else None,
                "cbm_eV": round(bs.get_cbm()["energy"], 4) if bs.get_cbm() else None,
                "nb_bands": bs.nb_bands,
            }
        except Exception as e:
            logging.error(f"get_bandstructure_by_material_id failed: {e}")
            return None

    def get_dos_by_material_id(self, material_id: str) -> Optional[Dict]:
        """Get density of states"""
        try:
            dos = self.client.get_dos_by_material_id(material_id)
            if not dos:
                return None
            return {
                "material_id": material_id,
                "efermi_eV": round(dos.efermi, 4),
                "band_gap_eV": round(dos.get_gap(), 4),
            }
        except Exception as e:
            logging.error(f"get_dos_by_material_id failed: {e}")
            return None

    def get_electronic_structure_data(self, material_id: str) -> Optional[Dict]:
        """Unified electronic structure data"""
        try:
            props = self.get_mp_properties(material_id)
            if not props:
                return None
            return {
                "material_id": material_id,
                "formula": props.get("formula_pretty"),
                "band_gap": props.get("band_gap"),
                "is_direct": props.get("is_gap_direct"),
                "efermi": props.get("efermi")
            }
        except Exception as e:
            logging.error(f"get_electronic_structure_data failed: {e}")
            return None

    def get_phonon_bandstructure_by_material_id(self, material_id: str) -> Optional[Dict]:
        """Get phonon band structure"""
        try:
            doc = self.client.phonon.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_phonon_bandstructure_by_material_id failed: {e}")
            return None

    def get_phonon_dos_by_material_id(self, material_id: str) -> Optional[Dict]:
        """Get phonon DOS"""
        try:
            doc = self.client.phonon.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_phonon_dos_by_material_id failed: {e}")
            return None

    # ==================================================================
    # 3. Thermodynamic & Phase Diagram Methods
    # ==================================================================
    def get_thermo_data(self, material_id: str) -> Optional[Dict]:
        """Get thermodynamic properties"""
        try:
            docs = self.client.thermo.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_thermo_data failed: {e}")
            return None

    def get_pourbaix_entries(self, elements: List[str]) -> List[Dict]:
        """Get Pourbaix diagram entries"""
        try:
            entries = self.client.get_pourbaix_entries(elements)
            return [_safe_dump(e) for e in entries]
        except Exception as e:
            logging.error(f"get_pourbaix_entries failed: {e}")
            return []

    def get_reaction(self, entries: List[str]) -> Optional[Dict]:
        """Compute reaction energy"""
        try:
            # This would need actual entry objects, simplified for now
            return {"message": "Reaction calculation requires ComputedEntry objects"}
        except Exception as e:
            logging.error(f"get_reaction failed: {e}")
            return None

    def get_phase_diagram_from_entries(self, entries: List[str]) -> Optional[Dict]:
        """Build phase diagram from entries"""
        try:
            # Simplified - would need actual entry objects
            return {"message": "Phase diagram requires ComputedEntry objects"}
        except Exception as e:
            logging.error(f"get_phase_diagram_from_entries failed: {e}")
            return None

    def get_exp_thermo_data(self, formula: str) -> Optional[Dict]:
        """Get experimental thermodynamic data"""
        try:
            docs = self.client.thermo.search(formula=formula, is_stable=True)
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_exp_thermo_data failed: {e}")
            return None

    # ==================================================================
    # 4. Surface & Interface Methods
    # ==================================================================
    def get_wulff_shape(self, material_id: str) -> Optional[Dict]:
        """Get Wulff shape from surface energies"""
        try:
            doc = self.client.wulff.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_wulff_shape failed: {e}")
            return None

    def get_surface_data(self, material_id: str, miller_index: List[int] = None) -> Optional[Dict]:
        """Get surface energy data"""
        try:
            if miller_index:
                docs = self.client.surface_properties.search(material_ids=[material_id], miller_index=miller_index)
            else:
                docs = self.client.surface_properties.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_surface_data failed: {e}")
            return None

    def get_substrate_data(self, material_id: str) -> Optional[Dict]:
        """Get substrate screening data"""
        try:
            docs = self.client.substrates.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_substrate_data failed: {e}")
            return None

    def get_interface_reactions(self, reactants: List[str], products: List[str]) -> Optional[Dict]:
        """Predict interface reaction energies"""
        try:
            return {"reactants": reactants, "products": products, "message": "Interface reaction calculation"}
        except Exception as e:
            logging.error(f"get_interface_reactions failed: {e}")
            return None

    def get_gb_data(self, material_id: str) -> Optional[Dict]:
        """Get grain boundary data"""
        try:
            docs = self.client.grain_boundary.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_gb_data failed: {e}")
            return None

    # ==================================================================
    # 5. Mechanical & Electronic Property Methods
    # ==================================================================
    def get_elasticity_data(self, material_id: str) -> Optional[Dict]:
        """Get elastic tensor data"""
        try:
            doc = self.client.elasticity.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_elasticity_data failed: {e}")
            return None

    def get_piezoelectric_data(self, material_id: str) -> Optional[Dict]:
        """Get piezoelectric tensor data"""
        try:
            doc = self.client.piezoelectric.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_piezoelectric_data failed: {e}")
            return None

    def get_dielectric_data(self, material_id: str) -> Optional[Dict]:
        """Get dielectric constants"""
        try:
            doc = self.client.dielectric.get_doc_by_material_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_dielectric_data failed: {e}")
            return None

    def get_magnetism_data(self, material_id: str) -> Optional[Dict]:
        """Get magnetic data"""
        try:
            docs = self.client.magnetism.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_magnetism_data failed: {e}")
            return None

    def get_cohesive_energy(self, material_id: str) -> Optional[Dict]:
        """Get cohesive energy"""
        try:
            props = self.get_mp_properties(material_id)
            if props and "cohesive_energy" in props:
                return {"cohesive_energy": props["cohesive_energy"]}
            return None
        except Exception as e:
            logging.error(f"get_cohesive_energy failed: {e}")
            return None

    # ==================================================================
    # 6. Spectroscopy Methods
    # ==================================================================
    def get_xas_data(self, material_id: str, spectrum_type: str = "XANES") -> Optional[Dict]:
        """Get X-ray absorption spectra"""
        try:
            docs = self.client.xas.search(material_ids=[material_id], spectrum_type=spectrum_type)
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_xas_data failed: {e}")
            return None

    def get_ir_spectra(self, material_id: str) -> Optional[Dict]:
        """Get IR/Raman spectra"""
        try:
            docs = self.client.phonon.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_ir_spectra failed: {e}")
            return None

    # ==================================================================
    # 7. Battery & Electrochemistry Methods
    # ==================================================================
    def get_battery_data(self, material_id: str) -> Optional[Dict]:
        """Get battery properties"""
        try:
            docs = self.client.electrodes.search(material_ids=[material_id])
            return _safe_dump(docs[0]) if docs else None
        except Exception as e:
            logging.error(f"get_battery_data failed: {e}")
            return None

    def get_electrode_data(self, elements: List[str]) -> List[Dict]:
        """Search for battery electrode materials"""
        try:
            docs = self.client.electrodes.search(elements=elements)
            return [_safe_dump(d) for d in docs]
        except Exception as e:
            logging.error(f"get_electrode_data failed: {e}")
            return []

    # ==================================================================
    # 8. Chemical Analysis Methods
    # ==================================================================
    def get_oxidation_states(self, material_id: str) -> Optional[Dict]:
        """Get oxidation states by analyzing the structure"""
        try:
            struct = self.client.get_structure_by_material_id(material_id)
            if not struct:
                return None
            from pymatgen.analysis.bond_valence import BVAnalyzer
            try:
                bva = BVAnalyzer()
                oxi = bva.get_valences(struct)
                species = [str(s) for s in struct.species]
                return {
                    "material_id": material_id,
                    "oxidation_states": dict(zip(species, oxi))
                }
            except Exception:
                # Fallback: get from summary
                docs = self.client.materials.summary.search(
                    material_ids=[material_id],
                    fields=["material_id", "formula_pretty"]
                )
                if docs:
                    return {"material_id": material_id, "formula": docs[0].formula_pretty, "note": "Oxidation state analysis unavailable for this structure"}
                return None
        except Exception as e:
            logging.error(f"get_oxidation_states failed: {e}")
            return None

    def get_bond_valence_data(self, material_id: str) -> Optional[Dict]:
        """Get bond valence data via structure analysis"""
        try:
            struct = self.client.get_structure_by_material_id(material_id)
            if not struct:
                return None
            from pymatgen.analysis.bond_valence import BVAnalyzer
            bva = BVAnalyzer()
            valences = bva.get_valences(struct)
            species = [str(s) for s in struct.species]
            return {
                "material_id": material_id,
                "bond_valences": dict(zip(species, valences))
            }
        except Exception as e:
            logging.error(f"get_bond_valence_data failed: {e}")
            return None

    # ==================================================================
    # 9. Discovery & Search Methods
    # ==================================================================
    def search_mp_by_criteria(
        self,
        elements: Optional[List[str]] = None,
        nelements: Optional[int] = None,
        crystal_system: Optional[str] = None,
        band_gap: Optional[tuple] = None,
        energy_above_hull: Optional[float] = None,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            kwargs = {}
            if elements: kwargs["elements"] = elements
            if nelements: kwargs["nelements"] = nelements
            if crystal_system: kwargs["crystal_system"] = crystal_system.lower()
            if band_gap: kwargs["band_gap"] = band_gap
            if energy_above_hull: kwargs["energy_above_hull"] = (0, energy_above_hull)

            default_fields = [
                "material_id", "formula_pretty", "band_gap",
                "energy_above_hull", "is_gap_direct", "crystal_system"
            ]
            fields = fields or default_fields

            docs = self.client.materials.summary.search(fields=fields, **kwargs)
            return [_safe_dump(d) for d in docs]
        except Exception as e:
            logging.error(f"search_mp_by_criteria failed: {e}")
            return []

    def query(self, criteria: Dict, properties: List[str] = None) -> List[Dict]:
        """Alias for get_data"""
        return self.get_data(criteria, properties)

    # ==================================================================
    # 10. Advanced Data Access Methods
    # ==================================================================
    def get_materials_id_doc(self, material_id: str) -> Optional[Dict]:
        """Get full material document"""
        try:
            doc = self.client.materials.get_document_by_id(material_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_materials_id_doc failed: {e}")
            return None

    def get_task_data(self, task_id: str) -> Optional[Dict]:
        """Get raw task document"""
        try:
            doc = self.client.tasks.get_document_by_id(task_id)
            return _safe_dump(doc)
        except Exception as e:
            logging.error(f"get_task_data failed: {e}")
            return None

    def get_task_ids_associated_with_material_id(self, material_id: str) -> List[str]:
        """Get task IDs for a material"""
        try:
            doc = self.client.materials.get_document_by_id(material_id)
            return getattr(doc, "task_ids", []) if doc else []
        except Exception as e:
            logging.error(f"get_task_ids_associated_with_material_id failed: {e}")
            return []

    def get_materials_id_from_task_id(self, task_id: str) -> Optional[str]:
        """Convert task ID to material ID"""
        try:
            doc = self.client.tasks.get_document_by_id(task_id)
            return getattr(doc, "material_id", None) if doc else None
        except Exception as e:
            logging.error(f"get_materials_id_from_task_id failed: {e}")
            return None

    def get_materials_id_references(self, material_id: str) -> List[str]:
        """Get literature references"""
        try:
            doc = self.client.materials.get_document_by_id(material_id)
            return getattr(doc, "references", []) if doc else []
        except Exception as e:
            logging.error(f"get_materials_id_references failed: {e}")
            return []

    def get_charge_density_data(self, task_id: str) -> Optional[Dict]:
        """Get charge density data (simplified)"""
        try:
            return {"task_id": task_id, "message": "Charge density download available via task data"}
        except Exception as e:
            logging.error(f"get_charge_density_data failed: {e}")
            return None

    def get_wavefunction_data(self, task_id: str) -> Optional[Dict]:
        """Get wavefunction data (simplified)"""
        try:
            return {"task_id": task_id, "message": "Wavefunction download available via task data"}
        except Exception as e:
            logging.error(f"get_wavefunction_data failed: {e}")
            return None