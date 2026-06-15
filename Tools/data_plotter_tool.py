import os
import json
import traceback
import webbrowser
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Placeholder for Gemini 3.1 Flash Lite model name
MODEL_NAME = "gemini-3.1-flash-lite"

def run_origin_plot_agent(filename: str, prompt: str) -> str:
    """
    Reads a dataset from the Read&Write folder, uses Gemini 3.1 Flash Lite to write
    a data filtering and plotting script, and executes it to produce an Origin-style plot.
    """
    print(f"\n{'='*60}")
    print(f"📊 ORIGIN-STYLE DATA PLOTTER (Gemini 3.1 Flash Lite)")
    print(f"   Target File: {filename}")
    print(f"   User Request: {prompt}")
    print(f"{'='*60}\n")

    base_dir = os.path.dirname(__file__)
    readwrite_dir = os.path.join(base_dir, "Read&Write")
    
    # Clean up filename if it contains absolute path
    if os.path.isabs(filename):
        file_path = filename
    else:
        file_path = os.path.join(readwrite_dir, filename)

    if not os.path.exists(file_path):
        error_msg = f"File '{filename}' not found. Ensure it exists in the Read&Write folder."
        print(f"   [-] Error: {error_msg}")
        return json.dumps({"error": error_msg})

    print("   [1/4] Reading dataset structure...")
    try:
        # Try reading as CSV first for preview
        if file_path.endswith('.csv'):
            df_preview = pd.read_csv(file_path, nrows=10)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df_preview = pd.read_excel(file_path, nrows=10)
        else:
            # Fallback for plain text or space separated
            df_preview = pd.read_csv(file_path, sep=None, engine='python', nrows=10)
            
        schema_info = f"Columns: {list(df_preview.columns)}\n\nFirst 5 rows of data:\n{df_preview.head().to_string()}"
    except Exception as e:
        schema_info = f"Could not read dataset preview automatically. The AI will need to guess the structure. Error: {e}"

    print(f"   [2/4] Initializing AI Model ({MODEL_NAME})...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("   [-] Warning: GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
         return json.dumps({"error": f"Failed to initialize model '{MODEL_NAME}'. Error: {e}"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_html = os.path.join(base_dir, f"origin_plot_{timestamp}.html")

    system_instruction = f'''
    You are an expert Data Scientist and Python Developer. 
    Your task is to write a Python script that reads a dataset, filters or cleans it if necessary based on the user's request, and generates a highly professional, publication-quality graph that mimics "Origin Software" aesthetics using Plotly.
    
    Origin Software Aesthetics in Plotly:
    - White background (`plot_bgcolor='white'`, `paper_bgcolor='white'`)
    - Black, thick axes lines (`showline=True`, `linecolor='black'`, `linewidth=2`, `mirror=True` for both x and y axes to create a full box boundary)
    - Ticks pointing inward (`ticks='inside'`, `ticklen=8`, `tickwidth=1.5`, `tickcolor='black'`, `showticklabels=True`)
    - Grid lines are usually OFF in Origin by default, unless requested (`showgrid=False`)
    - Professional fonts (Arial or Times New Roman, size 14), bold axes titles (size 16).
    - Legend with a black border (`legend=dict(bordercolor="black", borderwidth=1)`).
    - Line traces should be solid and clearly visible, scatter points should have distinct markers.
    
    Data filtering instructions:
    - If the user asks to filter the data (e.g., remove noise, drop NaN, apply moving average), you MUST apply those operations using pandas before plotting.
    
    The file path is stored in the string variable `file_path`. Use `pandas` (available as `pd`) to read it.
    The final plot MUST be saved as an HTML file to the path specified in the string variable `output_html` using `fig.write_html(output_html)`.
    
    Write ONLY valid Python code. Do not wrap it in markdown blocks like ```python. Do not add explanations. Just the raw Python code.
    The variables `file_path`, `output_html`, `pd`, `np`, `go` (plotly.graph_objects), and `px` (plotly.express) are already available in the execution environment.
    '''
    
    full_prompt = f"User Request: {prompt}\n\nDataset Info:\n{schema_info}\n\nData file is at: {file_path}\nOutput HTML must be saved to: {output_html}"
    
    print("   [3/4] Generating data processing & plotting script...")
    try:
        response = model.generate_content(system_instruction + "\n\n" + full_prompt)
        code = response.text.strip()
        
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
    except Exception as e:
        return json.dumps({"error": f"Gemini API request failed: {str(e)}"})

    print("   [4/4] Executing dynamic plotting script in background...")
    
    env_globals = {
        "file_path": file_path,
        "output_html": output_html,
        "pd": pd,
        "np": np,
        "go": go,
        "px": px,
        "os": os
    }
    
    try:
        exec(code, env_globals)
        if os.path.exists(output_html):
            webbrowser.open(f"file:///{output_html.replace(os.sep, '/')}")
            print(f"   [+] Success! Plot generated and opened: {output_html}")
            return json.dumps({
                "status": "success",
                "message": "Origin-style plot generated successfully.",
                "html_path": output_html,
                "dataset_preview": schema_info
            })
        else:
            return json.dumps({"error": "Script executed, but HTML plot was not created."})
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"   [-] Execution failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": error_msg,
            "generated_code": code
        })

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        prompt = " ".join(sys.argv[2:])
        run_origin_plot_agent(filename, prompt)
    else:
        print("Usage: python data_plotter_tool.py <filename> <your prompt>")
        print("Example: python data_plotter_tool.py dataset.csv Plot column A vs column B and filter out values below 0")
