import os
import json
import traceback
import google.generativeai as genai
from pyautocad import Autocad, APoint

# Set up Gemini 3.1 Flash Lite
# Replace with actual model string for Gemini 3.1 Flash Lite if different.
MODEL_NAME = "gemini-3.1-flash-lite" # Placeholder for Gemini 3.1 Flash Lite

def run_autocad_agent(prompt: str) -> str:
    """
    Connects to an active AutoCAD instance and uses Gemini 3.1 Flash Lite 
    to generate and execute real-time pyautocad code based on the user's prompt.
    """
    print(f"\n{'='*60}")
    print(f"📐 AUTOCAD REAL-TIME CODING AGENT (Gemini 3.1 Flash Lite)")
    print(f"   Task: {prompt}")
    print(f"{'='*60}\n")

    # 1. Connect to AutoCAD using pyautocad
    try:
        print("   [1/4] Connecting to AutoCAD instance...")
        acad = Autocad(create_if_not_exists=True)
        # acad.prompt prints a message to the AutoCAD command line
        acad.prompt(f"Hello from Shadow AI! Generating geometry for: {prompt}\n")
        print(f"   [+] Connected to AutoCAD drawing: {acad.doc.Name}")
    except Exception as e:
        error_msg = f"Could not connect to AutoCAD. Ensure AutoCAD is open and running on Windows. Details: {str(e)}"
        print(f"   [-] Error: {error_msg}")
        return json.dumps({"error": error_msg})

    # 2. Prepare the Gemini Model
    print(f"   [2/4] Initializing AI Model ({MODEL_NAME})...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("   [-] Warning: GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
         return json.dumps({"error": f"Failed to initialize Gemini model '{MODEL_NAME}'. Error: {e}"})

    # System instruction enforces strict pyautocad code output
    system_instruction = '''
    You are an expert AutoCAD Python programmer using the `pyautocad` library.
    Write Python code to execute the user's request in AutoCAD.
    The `acad` object (instance of `Autocad`) and `APoint` are already imported and available in the global scope.
    Write ONLY valid Python code. Do not wrap it in markdown code blocks like ```python. Do not add explanations.
    Just the raw Python code that can be passed directly to `exec()`.
    Example:
    p1 = APoint(0, 0)
    p2 = APoint(10, 10)
    acad.model.AddLine(p1, p2)
    acad.model.AddCircle(APoint(5, 5), 2.5)
    '''
    
    print("   [3/4] Generating AutoCAD commands via Gemini 3.1 Flash Lite...")
    try:
        response = model.generate_content(
            system_instruction + "\n\nUser Request: " + prompt
        )
        code = response.text.strip()
        
        # Strip markdown if the model hallucinates it anyway
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
    except Exception as e:
        return json.dumps({"error": f"Gemini API request failed: {str(e)}"})

    print("   [+] Generated PyAutoCAD Code:\n")
    for line in code.split('\n'):
        print(f"       {line}")
    
    print("\n   [4/4] Executing code in real-time in AutoCAD...")
    try:
        # Execute the generated code dynamically within this scope
        exec(code, globals(), {"acad": acad, "APoint": APoint})
        print("   [+] Execution completed successfully. Check AutoCAD window!")
        return json.dumps({
            "status": "success",
            "prompt": prompt,
            "executed_code": code,
            "message": "AutoCAD real-time code executed successfully."
        })
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"   [-] Execution failed: {e}")
        return json.dumps({
            "status": "error",
            "prompt": prompt,
            "generated_code": code,
            "error": str(e),
            "traceback": error_msg
        })

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_autocad_agent(" ".join(sys.argv[1:]))
    else:
        print("Usage: python autocad_tool.py <your prompt>")
        print("Example: python autocad_tool.py Draw a circle with radius 15 at the origin")
