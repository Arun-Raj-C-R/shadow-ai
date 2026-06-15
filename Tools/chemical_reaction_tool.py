import json
import traceback

def analyze_chemical_reaction(reaction_query: str) -> str:
    """
    Advanced Chemical Reaction Thermodynamics and Stoichiometry Analyzer.
    Uses Wolfram Alpha to compute exact physics, balanced equations, 
    reaction enthalpies, and entropy for general chemistry, organics, and inorganics.
    """
    try:
        from wolfram_orchestrator_tool import simple_wolfram_query
        
        print(f"⚗️ Analyzing Chemical Reaction: {reaction_query} ...")
        
        # 1. Ask Wolfram for balanced equation and thermodynamics
        wa_query = f"chemical reaction {reaction_query}"
        wa_result = simple_wolfram_query(wa_query)
        
        # 2. Extract specific thermodynamic values using focused queries
        print("📊 Calculating Enthalpy (ΔH) and Thermodynamics...")
        wa_thermo = simple_wolfram_query(f"thermodynamics of reaction {reaction_query}")
        
        # Output synthesis
        output = {
            "status": "success",
            "reaction_query": reaction_query,
            "reaction_details": wa_result,
            "thermodynamics": wa_thermo,
            "note": "Thermodynamics calculated at Standard State (298.15 K, 1 atm). For complex solid-state phase diagrams, use Pymatgen PhaseDiagram integration."
        }
        
        return json.dumps(output, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze reaction: {e}\n{traceback.format_exc()}"})
