# wolfram_orchestrator_tool.py
"""
Complete Wolfram Alpha tools — agent-ready, generic, and exhaustive.
Use for physics, materials, nano, solar, math, chemistry, etc.
"""
import os
import json
import logging
import uuid
import requests
import webbrowser
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()
# Import Wolfram Alpha API
try:
    from wolfram_tool import WolframAlphaAPI, WolframAPIError, WolframQueryError
except ImportError:
    # Fallback for when wolfram_tool is not available
    class WolframAlphaAPI:
        def __init__(self, app_id):
            self.app_id = app_id
        
        def query(self, query, **kwargs):
            raise NotImplementedError("WolframAlphaAPI not properly installed")
        
        def result(self, query):
            raise NotImplementedError("WolframAlphaAPI not properly installed")
        
        def spoken(self, query):
            raise NotImplementedError("WolframAlphaAPI not properly installed")
        
        def simple(self, query):
            raise NotImplementedError("WolframAlphaAPI not properly installed")

# --- Load AppID ---
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID")
if not WOLFRAM_APP_ID:
    logging.error("Set WOLFRAM_APP_ID in environment.")
    api_client = None
else:
    try:
        api_client = WolframAlphaAPI(WOLFRAM_APP_ID)
    except Exception as e:
        logging.error(f"API init failed: {e}")
        api_client = None

# --- Enhanced Helper Functions ---
def _get_pod(pods: List[Dict], title: str) -> Optional[Dict]:
    """Safely get pod by title with fuzzy matching."""
    if not pods: 
        return None
    
    # Exact match first
    exact_match = next((p for p in pods if p.get("title", "").strip().lower() == title.strip().lower()), None)
    if exact_match:
        return exact_match
    
    # Partial match
    partial_match = next((p for p in pods if title.strip().lower() in p.get("title", "").strip().lower()), None)
    return partial_match

def _get_pod_text(pod: Dict) -> str:
    """Safely extract plaintext from pod."""
    if not pod: 
        return ""
    
    subpods = pod.get("subpods", [])
    if not subpods: 
        return ""
    
    # Combine all subpod texts
    texts = []
    for sp in subpods:
        txt = sp.get("plaintext", "").strip()
        if txt:
            texts.append(txt)
    
    return "\n".join(texts) if texts else ""

def _collect_pods(pods: List[Dict], include_titles: bool = True) -> str:
    """Collect all pod texts with optional titles."""
    lines = []
    for pod in pods:
        title = pod.get("title", "")
        for sp in pod.get("subpods", []):
            txt = sp.get("plaintext", "")
            if txt:
                if include_titles and title:
                    lines.append(f"【{title}】\n{txt}\n")
                else:
                    lines.append(f"{txt}\n")
    return "\n".join(lines) if lines else "No results found."

def _get_image_url(pods: List[Dict]) -> Optional[str]:
    """Extract first available image URL from pods."""
    for pod in pods:
        for sp in pod.get("subpods", []):
            img = sp.get("img", {})
            src = img.get("src", "")
            if src and src.startswith("http"):
                return src
    return None

def _get_all_image_urls(pods: List[Dict]) -> List[str]:
    """Extract all image URLs from pods."""
    urls = []
    for pod in pods:
        for sp in pod.get("subpods", []):
            img = sp.get("img", {})
            src = img.get("src", "")
            if src and src.startswith("http"):
                urls.append(src)
    return urls

# ==============================================================================
# CORE QUERY TOOLS
# ==============================================================================

def simple_wolfram_query(query: str, include_images: bool = False) -> str:
    """Perform a general Wolfram|Alpha query with optional image URLs."""
    if not api_client: 
        return "Error: Wolfram API not initialized. Set WOLFRAM_APP_ID environment variable."
    
    try:
        result = api_client.query(query, format="plaintext")
        pods = result.get("pods", [])
        
        text_result = _collect_pods(pods)
        
        if include_images:
            image_urls = _get_all_image_urls(pods)
            if image_urls:
                text_result += f"\n\n📷 Images: {', '.join(image_urls[:3])}"  # Limit to 3 URLs
        
        return text_result if text_result else "No results found for query."
    
    except Exception as e:
        return f"Query error: {str(e)}"

def get_short_answer(query: str) -> str:
    """Get concise answer for quick facts."""
    if not api_client: 
        return "Error: API not initialized."
    
    try:
        answer = api_client.result(query)
        return answer if answer else "No short answer available."
    except Exception as e:
        # Fallback to full query
        return simple_wolfram_query(query)

def get_spoken_answer(query: str) -> str:
    """Get natural language answer optimized for speech."""
    if not api_client: 
        return "Error: API not initialized."
    
    try:
        spoken = api_client.spoken(query)
        return spoken if spoken else "No spoken answer available."
    except Exception as e:
        return f"Spoken answer error: {str(e)}"

# ==============================================================================
# INTERACTIVE QUERY TOOLS
# ==============================================================================

def query_with_podstate(query: str, pod_title: str, state_name: str) -> str:
    """Two-step query: click a button in a pod (e.g., 'Show steps')."""
    if not api_client: 
        return "Error: API not initialized."
    
    try:
        # Step 1: Initial query
        res1 = api_client.query(query)
        pods = res1.get("pods", [])
        
        # Find target pod
        pod = _get_pod(pods, pod_title)
        if not pod:
            available_pods = [p.get("title", "Untitled") for p in pods]
            return f"Pod '{pod_title}' not found. Available pods: {', '.join(available_pods)}"
        
        # Find target state/button
        states = pod.get("states", [])
        state = next((s for s in states if s.get("name") == state_name), None)
        if not state:
            available_states = [s.get("name", "Unnamed") for s in states]
            return f"State '{state_name}' not found. Available states: {', '.join(available_states)}"
        
        # Step 2: Query with podstate
        res2 = api_client.query(
            query, 
            podstate=state["input"], 
            includepodid=pod.get("id", ""),
            format="plaintext"
        )
        
        # Get updated pod content
        updated_pods = res2.get("pods", [])
        updated_pod = _get_pod(updated_pods, pod_title)
        
        return _get_pod_text(updated_pod) if updated_pod else "No content after state change."
    
    except Exception as e:
        return f"Podstate query error: {str(e)}"

def query_with_assumption(query: str, assumption_type: str, assumption_desc: str) -> str:
    """Two-step query: resolve ambiguity using assumptions."""
    if not api_client: 
        return "Error: API not initialized."
    
    try:
        # Step 1: Get assumptions
        res1 = api_client.query(query)
        assumptions = res1.get("assumptions", {}).get("assumption", [])
        
        # Find matching assumption
        target_assumption = None
        target_value = None
        
        for assump in assumptions:
            if assump.get("type") == assumption_type:
                for val in assump.get("values", []):
                    desc = val.get("desc", "")
                    input_val = val.get("input", "")
                    if assumption_desc.lower() in desc.lower():
                        target_assumption = assump
                        target_value = val
                        break
                if target_value:
                    break
        
        if not target_value:
            available_types = list(set(a.get("type", "Unknown") for a in assumptions))
            return f"Assumption '{assumption_desc}' of type '{assumption_type}' not found. Available types: {', '.join(available_types)}"
        
        # Step 2: Query with assumption
        res2 = api_client.query(
            query, 
            assumption=target_value["input"],
            format="plaintext"
        )
        
        return _collect_pods(res2.get("pods", []))
    
    except Exception as e:
        return f"Assumption query error: {str(e)}"

def get_step_by_step(query: str) -> str:
    """Get step-by-step solution for math problems."""
    return query_with_podstate(query, "Result", "Step-by-step solution")

# ==============================================================================
# VISUALIZATION TOOLS
# ==============================================================================

def get_plot_url(query: str) -> str:
    """Get direct URL to plot image."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        res = api_client.query(query)
        url = _get_image_url(res.get("pods", []))
        return url if url else "No plot image found."
    except Exception as e:
        return f"Plot URL error: {str(e)}"

def get_simple_image(query: str, save_path: Optional[str] = None, open_browser: bool = False) -> str:
    """Generate high-res image and save locally."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        img_data = api_client.simple(query)
        os.makedirs("plots", exist_ok=True)
        
        if save_path and not save_path.endswith(('.png', '.jpg', '.jpeg')):
            save_path += '.png'
        
        path = save_path or f"plots/wolfram_{uuid.uuid4().hex[:8]}.png"
        
        with open(path, "wb") as f:
            f.write(img_data)
        
        result = f"SUCCESS: Image saved to {path}"
        
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(path)}")
            result += " (opened in browser)"
        
        return result
    
    except Exception as e:
        return f"Image generation failed: {str(e)}"

def plot_and_show(query: str, title: str = "Wolfram Plot", figsize: tuple = (10, 6)) -> str:
    """Download and display plot using matplotlib."""
    url = get_plot_url(query)
    
    if not url.startswith("http"):
        return f"Cannot display plot: {url}"
    
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        return f"Plot displayed: {query}"
    
    except Exception as e:
        return f"Plot display error: {str(e)}"

def get_all_visualizations(query: str) -> Dict[str, Any]:
    """Get all available visualizations for a query."""
    if not api_client: 
        return {"error": "API not initialized"}
    
    try:
        res = api_client.query(query)
        pods = res.get("pods", [])
        
        visualizations = {
            "images": _get_all_image_urls(pods),
            "plots": [],
            "diagrams": [],
            "charts": [],
            "other": []
        }
        
        # Categorize images by pod title
        for pod in pods:
            title = pod.get("title", "").lower()
            pod_images = _get_all_image_urls([pod])
            
            if "plot" in title:
                visualizations["plots"].extend(pod_images)
            elif "diagram" in title:
                visualizations["diagrams"].extend(pod_images)
            elif "chart" in title or "graph" in title:
                visualizations["charts"].extend(pod_images)
            else:
                visualizations["other"].extend(pod_images)
        
        return visualizations
    
    except Exception as e:
        return {"error": str(e)}

# ==============================================================================
# SCIENCE & ENGINEERING TOOLS
# ==============================================================================

def get_material_property(material: str, property: str) -> str:
    """Get material property with intelligent query formulation."""
    if not api_client: 
        return "Error: API not initialized"
    
    # Map common property names to Wolfram-friendly queries
    property_map = {
        "bandgap": "band gap",
        "lattice_constant": "lattice constant",
        "density": "density",
        "melting_point": "melting point",
        "thermal_conductivity": "thermal conductivity",
        "youngs_modulus": "Young's modulus",
        "refractive_index": "refractive index",
        "dielectric_constant": "dielectric constant",
        "mobility": "charge carrier mobility",
        "work_function": "work function"
    }
    
    wolfram_property = property_map.get(property.lower(), property)
    query = f"{wolfram_property} of {material}"
    
    try:
        result = get_short_answer(query)
        if "did not understand" in result.lower() or len(result.strip()) < 5:
            # Fallback to full query
            result = simple_wolfram_query(query)
        
        return result if result and "no results" not in result.lower() else f"No {property} data found for {material}."
    
    except Exception as e:
        return f"Material property error: {str(e)}"

def get_unit_conversion(value: Union[str, float, int], from_unit: str, to_unit: str) -> str:
    """Convert between units with validation."""
    query = f"{value} {from_unit} to {to_unit}"
    
    try:
        result = get_short_answer(query)
        if not result or "did not understand" in result.lower():
            result = simple_wolfram_query(query)
        
        return result if result else f"Conversion failed: {value} {from_unit} → {to_unit}"
    
    except Exception as e:
        return f"Unit conversion error: {str(e)}"

def solve_equation(equation: str, variable: str = "x", step_by_step: bool = False) -> str:
    """Solve equations with optional step-by-step solution."""
    query = f"solve {equation} for {variable}"
    
    if step_by_step:
        return get_step_by_step(query)
    else:
        return get_short_answer(query)

def get_spectrum(material: str, spectrum_type: str) -> str:
    """Get spectrum data or plot URL."""
    valid_types = ["absorption", "emission", "raman", "uv-vis", "ir", "nmr", "xps"]
    
    if spectrum_type.lower() not in valid_types:
        return f"Invalid spectrum type. Choose from: {', '.join(valid_types)}"
    
    query = f"{spectrum_type} spectrum of {material}"
    plot_url = get_plot_url(query)
    
    if plot_url.startswith("http"):
        return f"[STATS] {spectrum_type.title()} Spectrum Plot: {plot_url}"
    else:
        # Fallback to data
        return simple_wolfram_query(query)

def get_phase_diagram(compound: str) -> str:
    """Get phase diagram URL or data."""
    query = f"phase diagram of {compound}"
    plot_url = get_plot_url(query)
    
    if plot_url.startswith("http"):
        return f"📈 Phase Diagram: {plot_url}"
    else:
        return simple_wolfram_query(query)

def get_quantum_property(particle: str, size: str, property: str) -> str:
    """Get quantum dot properties."""
    query = f"{property} of {size} {particle} quantum dot"
    return get_short_answer(query)

def get_shockley_queisser(bandgap: Union[float, str]) -> str:
    """Calculate Shockley-Queisser limit for solar cells."""
    try:
        bg = float(bandgap)
        query = f"Shockley-Queisser limit for {bg} eV bandgap"
        
        result = get_short_answer(query)
        if "did not understand" in result.lower():
            # Theoretical calculation fallback
            import math
            max_eff = 33.7 * (1 - 1/bg) if bg > 1 else 33.7 * bg * math.exp(-bg)
            return f"Estimated SQ limit: ~{max_eff:.1f}% (theoretical approximation)"
        
        return result
    
    except ValueError:
        return "Invalid bandgap value. Please provide a number."
    except Exception as e:
        return f"SQ calculation error: {str(e)}"

# ==============================================================================
# ADVANCED & UTILITY TOOLS
# ==============================================================================

def is_valid_query(query: str) -> Dict[str, Any]:
    """Validate if query is understandable by Wolfram."""
    if not api_client: 
        return {"valid": False, "error": "API not initialized"}
    
    try:
        # Simple validation by attempting a short query
        result = get_short_answer(query)
        valid = not any(phrase in result.lower() for phrase in [
            "did not understand", "no results", "invalid query"
        ])
        
        return {
            "valid": valid,
            "confidence": "high" if valid else "low",
            "suggestion": "Try rephrasing" if not valid else "Query looks good"
        }
    
    except Exception as e:
        return {"valid": False, "error": str(e)}

def get_summary_box(entity: str) -> str:
    """Get Wikipedia-style summary box."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        # Wolfram typically provides summary pods for entities
        query = f"summary {entity}"
        result = simple_wolfram_query(query)
        
        if "no results" in result.lower():
            # Try alternative query
            result = simple_wolfram_query(entity)
        
        return result
    
    except Exception as e:
        return f"Summary error: {str(e)}"

def start_async_query(query: str) -> str:
    """Start long-running query (for complex calculations)."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        # Implementation depends on Wolfram client capabilities
        result = api_client.query(query, params={"async": "true"})
        async_url = getattr(result, "async", None) or result.get("async", "")
        
        if async_url:
            return f"Async query started. Poll URL: {async_url}"
        else:
            return "Async query started (check client for polling details)"
    
    except Exception as e:
        return f"Async query error: {str(e)}"

def get_sound(query: str, save_path: Optional[str] = None) -> str:
    """Get sound data (pronunciations, audio signals)."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        # This requires Wolfram client support for sound format
        result = api_client.query(query, format="sound")
        
        for pod in result.get("pods", []):
            for sp in pod.get("subpods", []):
                sound = sp.get("sound")
                if sound and isinstance(sound, list) and len(sound) > 0:
                    audio_url = sound[0].get("url")
                    if audio_url:
                        response = requests.get(audio_url)
                        if save_path:
                            with open(save_path, "wb") as f:
                                f.write(response.content)
                            return f"Sound saved to: {save_path}"
                        else:
                            return f"Audio URL: {audio_url}"
        
        return "No sound data available for this query."
    
    except Exception as e:
        return f"Sound query error: {str(e)}"

def get_mathml(query: str) -> str:
    """Get MathML representation for mathematical expressions."""
    if not api_client: 
        return "Error: API not initialized"
    
    try:
        result = api_client.query(query, format="mathml")
        mathml_pod = _get_pod(result.get("pods", []), "Result")
        
        if mathml_pod:
            # Extract MathML from subpods
            for sp in mathml_pod.get("subpods", []):
                mathml = sp.get("mathml")
                if mathml:
                    return mathml
        
        return "No MathML representation available."
    
    except Exception as e:
        return f"MathML error: {str(e)}"

def batch_query(queries: List[str]) -> List[Dict[str, Any]]:
    """Execute multiple queries in sequence."""
    results = []
    
    for i, query in enumerate(queries):
        try:
            result = simple_wolfram_query(query)
            results.append({
                "query": query,
                "result": result,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "status": "failed"
            })
    
    return results

def chemical_analysis(compound: str, analysis_type: str = "properties") -> str:
    """Comprehensive chemical analysis."""
    analysis_map = {
        "properties": f"chemical properties of {compound}",
        "structure": f"molecular structure of {compound}",
        "reactions": f"chemical reactions of {compound}",
        "spectra": f"spectra of {compound}",
        "safety": f"safety data for {compound}",
        "synthesis": f"synthesis of {compound}"
    }
    
    query = analysis_map.get(analysis_type.lower(), f"{compound} {analysis_type}")
    return simple_wolfram_query(query)

def physical_constant(constant: str, units: str = "SI") -> str:
    """Get physical constant value."""
    query = f"{constant} in {units}" if units != "SI" else constant
    return get_short_answer(query)

# ==============================================================================
# TOOL DEFINITIONS FOR AGENT FRAMEWORK
# ==============================================================================

WOLFRAM_TOOL_DEFINITIONS = [
    # Core Query Tools
    {
        "name": "simple_wolfram_query",
        "description": "General Wolfram|Alpha query for math, facts, data, definitions, or calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Plain-text query"},
                "include_images": {"type": "boolean", "description": "Include image URLs", "default": False}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_short_answer", 
        "description": "Quick factual answer as short text",
        "parameters": {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "Query for short answer"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_spoken_answer",
        "description": "Natural language answer optimized for speech",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query for spoken answer"}
            },
            "required": ["query"]
        }
    },
    
    # Interactive Tools
    {
        "name": "query_with_podstate",
        "description": "Click buttons in Wolfram pods (Show steps, More digits, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Initial query"},
                "pod_title": {"type": "string", "description": "Pod title to interact with"},
                "state_name": {"type": "string", "description": "Button/state name to activate"}
            },
            "required": ["query", "pod_title", "state_name"]
        }
    },
    {
        "name": "query_with_assumption",
        "description": "Resolve ambiguous queries using assumptions",
        "parameters": {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "Ambiguous query"},
                "assumption_type": {"type": "string", "description": "Type of assumption"},
                "assumption_desc": {"type": "string", "description": "Specific assumption description"}
            },
            "required": ["query", "assumption_type", "assumption_desc"]
        }
    },
    {
        "name": "get_step_by_step",
        "description": "Get detailed step-by-step solutions for math problems",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Math problem to solve step-by-step"}
            },
            "required": ["query"]
        }
    },
    
    # Visualization Tools
    {
        "name": "get_plot_url",
        "description": "Get direct URL to plot images",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Plot query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_simple_image",
        "description": "Generate high-resolution Wolfram images (Mandelbrot, fractals, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Image generation query"},
                "save_path": {"type": "string", "description": "Optional custom save path"},
                "open_browser": {"type": "boolean", "description": "Open image in browser", "default": False}
            },
            "required": ["query"]
        }
    },
    {
        "name": "plot_and_show",
        "description": "Display Wolfram plots using matplotlib",
        "parameters": {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "Plot query"},
                "title": {"type": "string", "description": "Plot title", "default": "Wolfram Plot"},
                "figsize": {"type": "array", "items": {"type": "number"}, "description": "Figure size", "default": [10, 6]}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_all_visualizations",
        "description": "Get all available visualizations for a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to visualize"}
            },
            "required": ["query"]
        }
    },
    
    # Science & Engineering Tools
    {
        "name": "get_material_property",
        "description": "Get material properties (bandgap, density, conductivity, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "Material name or formula"},
                "property": {"type": "string", "description": "Property to retrieve"}
            },
            "required": ["material", "property"]
        }
    },
    {
        "name": "get_unit_conversion",
        "description": "Convert between units with validation",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"], "description": "Value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"}
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
    {
        "name": "solve_equation",
        "description": "Solve equations with optional step-by-step solution",
        "parameters": {
            "type": "object",
            "properties": {
                "equation": {"type": "string", "description": "Equation to solve"},
                "variable": {"type": "string", "description": "Variable to solve for", "default": "x"},
                "step_by_step": {"type": "boolean", "description": "Show detailed steps", "default": False}
            },
            "required": ["equation"]
        }
    },
    {
        "name": "get_spectrum",
        "description": "Get spectrum data or plots (absorption, emission, Raman, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "Material to analyze"},
                "spectrum_type": {"type": "string", "description": "Type of spectrum", "enum": ["absorption", "emission", "raman", "uv-vis", "ir", "nmr", "xps"]}
            },
            "required": ["material", "spectrum_type"]
        }
    },
    {
        "name": "get_phase_diagram", 
        "description": "Get phase diagrams for compounds",
        "parameters": {
            "type": "object",
            "properties": {
                "compound": {"type": "string", "description": "Chemical compound"}
            },
            "required": ["compound"]
        }
    },
    {
        "name": "get_quantum_property",
        "description": "Get quantum dot properties (confinement energy, exciton radius, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "particle": {"type": "string", "description": "Quantum dot material"},
                "size": {"type": "string", "description": "Dot size (e.g., '5 nm')"},
                "property": {"type": "string", "description": "Quantum property to calculate"}
            },
            "required": ["particle", "size", "property"]
        }
    },
    {
        "name": "get_shockley_queisser",
        "description": "Calculate Shockley-Queisser limit for solar cell efficiency",
        "parameters": {
            "type": "object", 
            "properties": {
                "bandgap": {"type": ["string", "number"], "description": "Bandgap in eV"}
            },
            "required": ["bandgap"]
        }
    },
    {
        "name": "chemical_analysis",
        "description": "Comprehensive chemical analysis (properties, structure, reactions, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "compound": {"type": "string", "description": "Chemical compound"},
                "analysis_type": {"type": "string", "description": "Type of analysis", "enum": ["properties", "structure", "reactions", "spectra", "safety", "synthesis"], "default": "properties"}
            },
            "required": ["compound"]
        }
    },
    {
        "name": "physical_constant",
        "description": "Get physical constant values",
        "parameters": {
            "type": "object",
            "properties": {
                "constant": {"type": "string", "description": "Physical constant name"},
                "units": {"type": "string", "description": "Unit system", "default": "SI"}
            },
            "required": ["constant"]
        }
    },
    
    # Advanced & Utility Tools
    {
        "name": "is_valid_query",
        "description": "Validate if Wolfram can understand a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to validate"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_summary_box",
        "description": "Get Wikipedia-style summary for entities",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to summarize"}
            },
            "required": ["entity"]
        }
    },
    {
        "name": "start_async_query",
        "description": "Start long-running complex queries",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Complex query for async processing"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_sound",
        "description": "Get sound data (pronunciations, audio signals)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query for sound data"},
                "save_path": {"type": "string", "description": "Optional path to save audio file"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_mathml",
        "description": "Get MathML representation for mathematical expressions",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Mathematical expression"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "batch_query",
        "description": "Execute multiple queries in sequence",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "description": "List of queries to execute"}
            },
            "required": ["queries"]
        }
    }
]
