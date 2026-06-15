import sys
import io
import contextlib
import traceback
import subprocess
import tempfile
import os

def run_python_code(code: str, timeout: int = 15) -> str:
    """
    Executes arbitrary Python code in an isolated subprocess and returns the output.
    Useful for doing math, dynamic web requests, complex logic, or data processing on the fly.
    """
    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # Run the code in a subprocess to prevent it from crashing the main SHADOW thread
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = ""
        if result.stdout:
            output += f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"--- STDERR ---\n{result.stderr}\n"
            
        if not output:
            output = "Code executed successfully with no output."
            
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Execution failed: {str(e)}"
    finally:
        # Cleanup
        try:
            os.remove(temp_file_path)
        except:
            pass
