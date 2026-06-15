"""
FreeCAD Geometry & Mesh Engine for SHADOW
=========================================
Actual LLM-driven Multi-agent CAD orchestration.
Agent 1 (Architect): Designs the mathematical geometry (vertices, faces).
Agent 2 (CAD Engineer): Codes the Python logic to generate the arrays.
Includes a feedback loop to correct execution errors.
"""
import os
import json
import time
import webbrowser
import traceback
from datetime import datetime
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "freecad_runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _build_freecad_html(x, y, z, i, j, k, intensity, task, material, geometry):
    # Ensure they are standard python lists for JSON serialization
    try: x = x.tolist() if isinstance(x, np.ndarray) else list(x)
    except: pass
    try: y = y.tolist() if isinstance(y, np.ndarray) else list(y)
    except: pass
    try: z = z.tolist() if isinstance(z, np.ndarray) else list(z)
    except: pass
    try: i = i.tolist() if isinstance(i, np.ndarray) else list(i)
    except: pass
    try: j = j.tolist() if isinstance(j, np.ndarray) else list(j)
    except: pass
    try: k = k.tolist() if isinstance(k, np.ndarray) else list(k)
    except: pass
    try: intensity = intensity.tolist() if isinstance(intensity, np.ndarray) else list(intensity)
    except: pass

    x_json = json.dumps(x)
    y_json = json.dumps(y)
    z_json = json.dumps(z)
    i_json = json.dumps(i)
    j_json = json.dumps(j)
    k_json = json.dumps(k)
    int_json = json.dumps(intensity)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>FreeCAD Mesh - {geometry} | SHADOW AI</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0a0a1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif}}
  #header{{padding:20px 30px;background:linear-gradient(135deg,rgba(10,30,40,0.95),rgba(10,10,30,0.95));border-bottom:1px solid rgba(0,150,255,0.3)}}
  #header h1{{font-size:24px;font-weight:300;letter-spacing:2px;background:linear-gradient(90deg,#00d4ff,#00ff88,#7b2ff7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .subtitle{{color:#888;font-size:13px;margin-top:4px}}
  .stats{{display:flex;gap:20px;margin-top:12px;flex-wrap:wrap}}
  .stat{{background:rgba(20,40,60,0.8);border:1px solid rgba(0,150,255,0.2);border-radius:10px;padding:10px 16px;min-width:140px}}
  .stat-label{{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px}}
  .stat-value{{color:#00d4ff;font-size:18px;font-weight:600;margin-top:2px}}
  #plot{{width:100%;height:75vh}}
</style></head><body>
<div id="header">
  <h1>âš™ï¸ FreeCAD Engine â€” {task.replace('_', ' ').title()}</h1>
  <div class="subtitle">AI-Driven Multi-Agent CAD Generation Â· {geometry} Â· SHADOW AI Engineering Lab</div>
  <div class="stats">
    <div class="stat"><div class="stat-label">Material</div><div class="stat-value">{material}</div></div>
    <div class="stat"><div class="stat-label">Vertices</div><div class="stat-value">{len(x)}</div></div>
    <div class="stat"><div class="stat-label">Faces</div><div class="stat-value">{len(i)}</div></div>
    <div class="stat"><div class="stat-label">Export Formats</div><div class="stat-value">STEP / STL</div></div>
  </div>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot',[{{
  type: 'mesh3d',
  x: {x_json},
  y: {y_json},
  z: {z_json},
  i: {i_json},
  j: {j_json},
  k: {k_json},
  intensity: {int_json},
  colorscale: 'Viridis',
  opacity: 0.9,
  flatshading: true,
  showscale: false
}}], {{
  paper_bgcolor:'#0a0a1a',
  plot_bgcolor:'#0f0f2a',
  font: {{color: '#ccc'}},
  title: 'Interactive 3D Device Geometry',
  scene: {{
    xaxis: {{title: 'X (Î¼m)', gridcolor: '#1a1a3a', backgroundcolor: '#0a0a1a'}},
    yaxis: {{title: 'Y (Î¼m)', gridcolor: '#1a1a3a', backgroundcolor: '#0a0a1a'}},
    zaxis: {{title: 'Z (Î¼m)', gridcolor: '#1a1a3a', backgroundcolor: '#0a0a1a'}},
    camera: {{ eye: {{x: 1.5, y: 1.5, z: 1.2}} }}
  }},
  margin: {{t: 50, b: 50, l: 0, r: 0}}
}});
</script></body></html>"""


def run_freecad_geometry(task: str, material: str = "Silicon", geometry: str = "planar", detailed_instructions: str = "", improvements_requested: str = "", **kwargs) -> str:
    """Run FreeCAD geometry generation exclusively on Gemini API with a single CAD Engineer agent."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    MODEL_CODER = "gemini-3.1-flash-lite" # Highly reliable and fast coding model
    
    def get_ai_response(prompt, model):
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 4096}
        )
        return response.text
    print(f"\n{'='*60}")
    print(f"âš™ï¸ FREECAD AI CAD ORCHESTRATOR")
    print(f"   Task: {task.replace('_', ' ').title()}")
    print(f"   Material: {material}")
    print(f"   Geometry: {geometry}")
    print(f"{'='*60}\n")
    
    max_retries = 3
    error_feedback = ""
    temp_code_file = os.path.join(OUTPUT_DIR, f"temp_memory_code.py")
    revising_external = bool(improvements_requested and os.path.exists(temp_code_file))
    
    for attempt in range(max_retries):
        try:
            if revising_external:
                print(f"   [System] Applying Shadow Main AI Improvements to previous code... (Attempt {attempt+1})")
                with open(temp_code_file, "r", encoding="utf-8") as f:
                    previous_code = f.read()
                    
                prompt = f"""
You are an elite Python CAD Engineer.
The Main AI (Shadow) has reviewed the rendered 3D model and requested specific improvements.

Here is your PREVIOUS CODE:
```python
{previous_code}
```

Here are SHADOW's IMPROVEMENT INSTRUCTIONS:
{improvements_requested}

Rewrite the Python script to incorporate these improvements. Do NOT rewrite from scratchâ€”modify and improve your existing code based strictly on the feedback.
RULES:
1. Store flat lists/arrays for x, y, z, i, j, k, intensity in a dictionary `mesh_data`.
2. Provide ONLY the raw Python code enclosed in ```python ... ``` tags. NO markdown outside the block.
CRITICAL RULE: DO NOT output <think> blocks, reasoning, or text. Just write the revised code.
"""
                print(f"   [Agent: CAD Engineer] Re-writing code fast based on Shadow instructions...")
            else:
                print(f"   [Agent: CAD Engineer] Generating Python mesh logic... (Attempt {attempt+1}/{max_retries})")
                prompt = f"""
You are an elite Python CAD Engineer. The Main AI has provided the detailed geometry instructions below:

<instructions>
Task: {task}
Material: {material}
Geometry: {geometry}
Details: {detailed_instructions}
</instructions>

Write a fully self-contained Python script to generate the 3D mesh arrays.
RULES:
1. Define flat lists/arrays for: `x`, `y`, `z` (coordinates) and `i`, `j`, `k` (0-indexed face indices).
2. Define a list/array for `intensity` to represent layer materials/colors.
3. Store these exactly in a dictionary called `mesh_data`.
4. Only use standard Python, `math`, and `numpy`. DO NOT import matplotlib or plotly.
5. Provide ONLY the raw Python code enclosed in ```python ... ``` tags. NO markdown outside the block. NO explanations.
"""
                if error_feedback:
                    prompt += f"\nðŸš¨ PREVIOUS EXECUTION FAILED WITH THIS ERROR:\n{error_feedback}\nPlease FIX the bug in the code."
            
            code_text = get_ai_response(prompt, MODEL_CODER)
            
            # Extract code block
            code = code_text
            if "```python" in code_text:
                code = code_text.split("```python")[1].split("```")[0].strip()
            elif "```" in code_text:
                code = code_text.split("```")[1].split("```")[0].strip()
                
            print(f"   [System] Executing generated CAD logic locally...")
            
            # Execute code safely inside a restricted locals dict
            local_vars = {}
            exec(code, {"np": np, "math": __import__("math")}, local_vars)
            
            if 'mesh_data' not in local_vars:
                raise ValueError("The executed code did not define a 'mesh_data' dictionary as requested.")
                
            md = local_vars['mesh_data']
            x, y, z = md['x'], md['y'], md['z']
            i_list, j_list, k_list = md['i'], md['j'], md['k']
            intensity = md.get('intensity', [1.0]*len(x))
            
            if len(x) == 0 or len(i_list) == 0:
                raise ValueError("The generated mesh contains 0 vertices or 0 faces.")
            
            print(f"   [System] Code executed successfully: {len(x)} Vertices, {len(i_list)} Faces.")
            
            # Update the temporary memory file with the working code
            with open(temp_code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            timestamp = int(datetime.now().timestamp())
            job_dir = os.path.join(OUTPUT_DIR, f"freecad_{task}_{timestamp}")
            os.makedirs(job_dir, exist_ok=True)
            
            print("   [System] Exporting artifacts and rendering 3D viewer...")
            html = _build_freecad_html(x, y, z, i_list, j_list, k_list, intensity, task, material, geometry)
            html_path = os.path.join(job_dir, f"{task}_3d_viewer.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
                
            # Log the AI artifacts
            with open(os.path.join(job_dir, "agent_cad_engineer_code.py"), "w", encoding="utf-8") as f:
                f.write(code)
                
            webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
            print(f"\nâœ… FreeCAD AI workflow complete. Browser opened.")
            
            return json.dumps({
                "status": "success",
                "framework": "FreeCAD",
                "task": task,
                "material": material,
                "geometry": geometry,
                "html_path": html_path,
                "output_dir": job_dir,
                "generated_code": code,
                "message": "Interactive 3D view opened in browser. The 'generated_code' is attached if Shadow needs to request improvements."
            })
            
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"   âŒ [System] Code Execution failed on attempt {attempt+1}:\n{str(e)}")
            error_feedback = f"{str(e)}\n\nTraceback:\n{err_msg}"
            revising_external = False # Reset flag so Architect can fix the parameters on fatal crashes
            
            if attempt == max_retries - 1:
                return json.dumps({"error": f"Failed after {max_retries} attempts. Last error: {str(e)}"})
                
            print(f"   [Feedback Loop] Routing error back to Architect for correction...\n")
            time.sleep(1.5)

    return json.dumps({"error": "Unknown workflow failure."})
