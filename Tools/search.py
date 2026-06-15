# perovskite_optimizer.py

from typing import List, Dict
from materials_project_tool import MaterialsProjectTool


class PerovskiteOptimizer:

    def __init__(self):
        self.mp = MaterialsProjectTool()

    # ---------------------------------------------------------
    # 1. Fetch candidate materials
    # ---------------------------------------------------------
    def fetch_candidates(
        self,
        elements: List[str],
        band_gap_range: tuple = (1.0, 2.0),
        max_energy_above_hull: float = 0.1
    ) -> List[Dict]:

        print("Fetching candidate materials from Materials Project...")

        materials = self.mp.search_mp_by_criteria(
            elements=elements,
            band_gap=band_gap_range,
            energy_above_hull=max_energy_above_hull
        )

        return materials

    # ---------------------------------------------------------
    # 2. Score materials (simple ranking logic)
    # ---------------------------------------------------------
    def score_material(self, material: Dict) -> float:
        """
        Higher score = better candidate
        Criteria:
        - Band gap close to ideal (1.34 eV for solar)
        - Low energy above hull (stable)
        """

        ideal_bandgap = 1.34

        band_gap = material.get("band_gap", 0)
        stability = material.get("energy_above_hull", 1)

        # Penalize deviation from ideal band gap
        bandgap_score = max(0, 1 - abs(band_gap - ideal_bandgap))

        # Stability score (lower is better)
        stability_score = max(0, 1 - stability)

        total_score = (0.6 * bandgap_score) + (0.4 * stability_score)

        return round(total_score, 3)

    # ---------------------------------------------------------
    # 3. Rank materials
    # ---------------------------------------------------------
    def rank_materials(self, materials: List[Dict]) -> List[Dict]:

        print("Scoring and ranking materials...")

        for m in materials:
            m["score"] = self.score_material(m)

        ranked = sorted(materials, key=lambda x: x["score"], reverse=True)

        return ranked

    # ---------------------------------------------------------
    # 4. Generate recommendations
    # ---------------------------------------------------------
    def generate_recommendations(self, material: Dict) -> str:
        """
        Basic reasoning layer (you can later replace with AI model)
        """

        band_gap = material.get("band_gap", 0)
        stability = material.get("energy_above_hull", 1)

        suggestions = []

        if band_gap < 1.2:
            suggestions.append("Band gap too low → consider halide tuning (Br substitution)")

        elif band_gap > 1.8:
            suggestions.append("Band gap too high → consider I-rich composition")

        if stability > 0.05:
            suggestions.append("Low stability → try passivation or doping")

        else:
            suggestions.append("Good stability")

        return " | ".join(suggestions)

    # ---------------------------------------------------------
    # 5. Full pipeline
    # ---------------------------------------------------------
    def optimize(
        self,
        elements: List[str],
        band_gap_range=(1.0, 2.0),
        max_energy_above_hull=0.1,
        top_n=5
    ):

        candidates = self.fetch_candidates(
            elements,
            band_gap_range,
            max_energy_above_hull
        )

        if not candidates:
            print("No materials found.")
            return []

        ranked = self.rank_materials(candidates)

        results = []

        print("\nTop Materials:\n")

        for m in ranked[:top_n]:
            recommendation = self.generate_recommendations(m)

            result = {
                "material_id": m["material_id"],
                "formula": m["formula_pretty"],
                "band_gap": m["band_gap"],
                "stability": m["energy_above_hull"],
                "score": m["score"],
                "recommendation": recommendation
            }

            results.append(result)

            print(result)

        return results


# ---------------------------------------------------------
# RUN SYSTEM
# ---------------------------------------------------------
if __name__ == "__main__":

    optimizer = PerovskiteOptimizer()

    # Example: Cs-Pb-I system (perovskite)
    optimizer.optimize(
        elements=["Cs", "Pb", "I"],
        band_gap_range=(1.0, 2.0),
        max_energy_above_hull=0.1,
        top_n=5
    )