import os
import subprocess
import sys
import tempfile
import re
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_groq_client = None
def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to use the physics agent.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

# ============================================================================
# VISUALIZATION TEMPLATES
# ============================================================================

VISUALIZATION_TEMPLATES = {
    "3d_surface": """
import plotly.graph_objects as go
import numpy as np

def create_3d_surface(x_range, y_range, z_func, title="3D Surface Plot"):
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = z_func(X, Y)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=title,
        autosize=False,
        width=900,
        height=700,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    fig.show()
    return fig
""",
    
    "animated_plot": """
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_animation(data_func, frames=100, interval=50, title="Animation"):
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        return line,
    
    def animate(frame):
        x, y = data_func(frame)
        line.set_data(x, y)
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval, blit=True)
    plt.show()
    return anim
""",

    "interactive_dashboard": """
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

def create_interactive_plot(plot_func, params):
    output = widgets.Output()
    
    # Create sliders for each parameter
    sliders = {}
    for param_name, (min_val, max_val, default, step) in params.items():
        sliders[param_name] = widgets.FloatSlider(
            value=default,
            min=min_val,
            max=max_val,
            step=step,
            description=f'{param_name}:',
            continuous_update=False
        )
    
    def update_plot(**kwargs):
        with output:
            output.clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plot_func(**kwargs)
            plt.show()
    
    # Link sliders to update function
    interactive_plot = widgets.interactive_output(update_plot, sliders)
    
    # Layout
    controls = widgets.VBox(list(sliders.values()))
    ui = widgets.HBox([controls, interactive_plot])
    display(ui)
""",

    "multi_panel": """
import matplotlib.pyplot as plt
import numpy as np

def create_multi_panel_plot(data_dict, title="Multi-Panel Analysis"):
    n_plots = len(data_dict)
    cols = min(2, n_plots)
    rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (subplot_title, (x, y, plot_type)) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        if plot_type == 'line':
            ax.plot(x, y, linewidth=2)
        elif plot_type == 'scatter':
            ax.scatter(x, y, alpha=0.6)
        elif plot_type == 'bar':
            ax.bar(x, y)
        
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()
    return fig
""",

    "3d_vector_field": """
import plotly.graph_objects as go
import numpy as np

def create_vector_field(field_func, bounds, title="Vector Field"):
    x = np.linspace(bounds[0], bounds[1], 10)
    y = np.linspace(bounds[0], bounds[1], 10)
    z = np.linspace(bounds[0], bounds[1], 10)
    X, Y, Z = np.meshgrid(x, y, z)
    
    U, V, W = field_func(X, Y, Z)
    
    fig = go.Figure(data=go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=W.flatten(),
        colorscale='Blues',
        sizemode="absolute",
        sizeref=2
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )
    fig.show()
    return fig
"""
}

# ============================================================================
# PHYSICS TEMPLATES
# ============================================================================

PHYSICS_TEMPLATES = {
    "quantum_mechanics": """
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Quantum harmonic oscillator
def quantum_harmonic_oscillator(n, x_range=(-5, 5), points=1000):
    x = np.linspace(x_range[0], x_range[1], points)
    hbar = 1.0545718e-34
    m = 9.10938e-31
    omega = 1.0
    
    # Hermite polynomials and wavefunctions
    from scipy.special import hermite, factorial
    
    H_n = hermite(n)
    alpha = np.sqrt(m * omega / hbar)
    
    normalization = (alpha / np.pi)**0.25 / np.sqrt(2**n * factorial(n))
    psi = normalization * np.exp(-alpha**2 * x**2 / 2) * H_n(alpha * x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, psi, linewidth=2, label=f'n={n}')
    plt.plot(x, psi**2, '--', linewidth=2, label=f'|ψ|² (n={n})')
    plt.xlabel('Position (x)')
    plt.ylabel('Wave function')
    plt.title(f'Quantum Harmonic Oscillator (n={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return x, psi
""",

    "relativity": """
import numpy as np
import matplotlib.pyplot as plt

def time_dilation(v, t0=1.0):
    c = 299792458  # Speed of light in m/s
    gamma = 1 / np.sqrt(1 - (v/c)**2)
    t = gamma * t0
    return t

def plot_relativistic_effects():
    c = 299792458
    v = np.linspace(0, 0.99*c, 1000)
    
    # Time dilation
    gamma = 1 / np.sqrt(1 - (v/c)**2)
    
    # Length contraction
    length = 1 / gamma
    
    # Relativistic momentum
    mass = 1.0  # kg
    momentum = gamma * mass * v
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].plot(v/c, gamma, linewidth=2)
    axes[0].set_xlabel('v/c')
    axes[0].set_ylabel('γ (Lorentz factor)')
    axes[0].set_title('Time Dilation')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(v/c, length, linewidth=2)
    axes[1].set_xlabel('v/c')
    axes[1].set_ylabel('L/L₀')
    axes[1].set_title('Length Contraction')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(v/c, momentum, linewidth=2)
    axes[2].set_xlabel('v/c')
    axes[2].set_ylabel('Momentum (kg⋅m/s)')
    axes[2].set_title('Relativistic Momentum')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
""",

    "electromagnetism": """
import numpy as np
import matplotlib.pyplot as plt

def electric_field_dipole():
    # Create grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Dipole positions
    q = 1.0
    d = 0.5
    pos1 = np.array([0, d])
    pos2 = np.array([0, -d])
    
    # Calculate electric field
    r1 = np.sqrt((X - pos1[0])**2 + (Y - pos1[1])**2)
    r2 = np.sqrt((X - pos2[0])**2 + (Y - pos2[1])**2)
    
    Ex = q * (X - pos1[0]) / r1**3 - q * (X - pos2[0]) / r2**3
    Ey = q * (Y - pos1[1]) / r1**3 - q * (Y - pos2[1]) / r2**3
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, Ex, Ey, density=1.5, linewidth=1, arrowsize=1.5, color=np.sqrt(Ex**2 + Ey**2), cmap='plasma')
    plt.plot([pos1[0]], [pos1[1]], 'ro', markersize=10, label='+q')
    plt.plot([pos2[0]], [pos2[1]], 'bo', markersize=10, label='-q')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field of a Dipole')
    plt.colorbar(label='Field Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
""",

    "thermodynamics": """
import numpy as np
import matplotlib.pyplot as plt

def carnot_cycle(T_hot=500, T_cold=300, p_max=10e5, v_min=0.001, v_max=0.01):
    # Carnot cycle: isothermal expansion, adiabatic expansion, isothermal compression, adiabatic compression
    gamma = 1.4  # For air
    n_points = 100
    
    # Volume arrays for each process
    v1 = np.linspace(v_min, v_max, n_points)
    v2 = np.linspace(v_max, v_max * (T_hot/T_cold)**(1/(gamma-1)), n_points)
    v3 = np.linspace(v2[-1], v1[-1] * (T_hot/T_cold)**(1/(gamma-1)), n_points)
    v4 = np.linspace(v3[-1], v_min, n_points)
    
    # Pressure calculations
    p1 = p_max * v_min / v1  # Isothermal at T_hot
    p2 = p1[-1] * (v2[0]/v2)**gamma  # Adiabatic
    p3 = p2[-1] * v3[-1] / v3  # Isothermal at T_cold
    p4 = p3[-1] * (v4[0]/v4)**gamma  # Adiabatic
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(v1*1000, p1/1e5, 'r-', linewidth=2, label='Isothermal Expansion (Hot)')
    plt.plot(v2*1000, p2/1e5, 'b-', linewidth=2, label='Adiabatic Expansion')
    plt.plot(v3*1000, p3/1e5, 'g-', linewidth=2, label='Isothermal Compression (Cold)')
    plt.plot(v4*1000, p4/1e5, 'm-', linewidth=2, label='Adiabatic Compression')
    
    plt.xlabel('Volume (L)')
    plt.ylabel('Pressure (bar)')
    plt.title(f'Carnot Cycle: Efficiency = {(1 - T_cold/T_hot)*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    efficiency = 1 - T_cold/T_hot
    print(f"Carnot Efficiency: {efficiency*100:.2f}%")
    return efficiency
"""
}

# ============================================================================
# MANAGER AGENT
# ============================================================================

class ManagerAgent:
    @property
    def client(self):
        return get_groq_client()
        
    def create_execution_plan(self, user_request):
        """Manager AI creates detailed execution plan with visualization hints"""
        prompt = f"""
        USER REQUEST: {user_request}
        
        As the Manager AI, analyze this request and create a detailed execution plan for the Coder AI.
        
        Your output should be structured as:
        
        ANALYSIS: [Brief analysis of what needs to be done]
        
        PROBLEM TYPE: [Physics/Math domain - e.g., mechanics, thermodynamics, quantum, etc.]
        
        EXECUTION PLAN:
        1. [Step 1 - Specific instruction]
        2. [Step 2 - Specific instruction] 
        3. [Step 3 - Specific instruction]
        
        VISUALIZATION RECOMMENDATION: [Suggest best visualization type: plot, 3D surface, animation, vector field, etc.]
        
        EXPECTED OUTPUT: [What the final output should look like]
        
        VALIDATION: [How to verify the code worked correctly]
        """
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def analyze_result(self, execution_result):
        """Analyze execution result and provide insights"""
        if execution_result["status"] != "success":
            return None
            
        prompt = f"""
        Analyze this physics problem execution result:
        
        OUTPUT: {execution_result['output']}
        
        Provide:
        1. Summary of what was calculated
        2. Physical interpretation of results
        3. Key insights or observations
        4. Potential follow-up questions
        
        Keep it concise and educational.
        """
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# ============================================================================
# CODER AGENT
# ============================================================================

class CoderAgent:
    def __init__(self):
        self.execution_attempts = 0
        self.max_attempts = 3
        self.last_error = ""
        self.visualization_templates = VISUALIZATION_TEMPLATES
        self.physics_templates = PHYSICS_TEMPLATES

    @property
    def client(self):
        return get_groq_client()
        
    def generate_and_execute_code(self, manager_plan, user_request, use_visualization=True):
        """Coder AI generates code and executes it with retry logic"""
        
        print("[RETRY] Coder Agent starting execution...")
        
        for attempt in range(self.max_attempts):
            self.execution_attempts = attempt + 1
            print(f"\n[RETRY] Attempt {self.execution_attempts}/{self.max_attempts}")
            
            # Generate code based on manager's plan
            generated_code = self._generate_code(manager_plan, user_request, attempt, use_visualization)
            
            if not generated_code:
                print("[ERR] Failed to generate code")
                continue
                
            print("[NOTE] Generated code successfully")
            
            # Execute the code
            execution_result = self._execute_code(generated_code)
            
            if execution_result["success"]:
                print("[OK] Code executed successfully!")
                return {
                    "status": "success",
                    "attempts": self.execution_attempts,
                    "generated_code": generated_code,
                    "output": execution_result["output"],
                    "manager_plan": manager_plan
                }
            else:
                print(f"[ERR] Execution failed: {execution_result['error']}")
                self.last_error = execution_result['error']
                
                # If last attempt, return failure
                if attempt == self.max_attempts - 1:
                    return {
                        "status": "failed",
                        "attempts": self.execution_attempts,
                        "error": execution_result["error"],
                        "last_code": generated_code,
                        "manager_plan": manager_plan
                    }
                
                print("[RETRY] Learning from error and retrying...")
                
    def _generate_code(self, manager_plan, user_request, attempt, use_visualization):
        """Generate Python code based on manager's plan"""
        
        error_context = ""
        if attempt > 0:
            error_context = f"\n\nPREVIOUS ATTEMPT ERROR: {self.last_error}\nLearn from this error and fix the code."
        
        viz_hint = ""
        if use_visualization:
            viz_hint = "\n\nVISUALIZATION HINT: Include matplotlib or plotly visualizations if appropriate for the problem."
        
        prompt = f"""
        MANAGER'S PLAN:
        {manager_plan}
        
        USER REQUEST: {user_request}
        {error_context}
        {viz_hint}
        
        Your task: Generate Python code that accomplishes the manager's plan.
        
        REQUIREMENTS:
        - Output ONLY Python code, no explanations
        - Use only ASCII characters (no special Unicode characters)
        - Use '*' for multiplication, not '·' or other symbols
        - Use standard Python operators only
        - Include proper error handling
        - Make sure the code is executable
        - Include print statements to show progress/results
        - If doing calculations, print the final result clearly
        - Use matplotlib, numpy, scipy as needed
        - Add visualizations where appropriate
        
        AVAILABLE LIBRARIES: numpy, scipy, matplotlib, plotly (if installed)
        
        IMPORTANT: Your response should contain ONLY the Python code, no markdown, no additional text.
        Use only basic ASCII characters in the code.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="groq/compound-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2
            )
            
            code = response.choices[0].message.content
            code = self._clean_code(code)
            return code
            
        except Exception as e:
            print(f"[ERR] Code generation failed: {e}")
            return None
    
    def _clean_code(self, code):
        """Remove markdown and extract pure Python code with proper encoding"""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Replace common problematic Unicode characters
        replacements = {
            '\xb7': '*',
            '\u00b7': '*',
            '\u2212': '-',
            '\u2013': '-',
            '\u2014': '-',
            '\u2018': "'",
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2026': '...',
        }
        
        for old, new in replacements.items():
            code = code.replace(old, new)
        
        # Add UTF-8 encoding declaration at the top
        if not code.startswith('# -*- coding: utf-8 -*-'):
            code = '# -*- coding: utf-8 -*-\n' + code
        
        return code
    
    def _execute_code(self, code):
        """Execute Python code locally"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "error": "Execution timeout (30 seconds)"
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class AICodeOrchestrator:
    def __init__(self):
        self.manager = ManagerAgent()
        self.coder = CoderAgent()
        
    def process_request(self, user_request, use_visualization=True):
        """Main orchestration - Manager plans, Coder executes"""
        
        print("[GO] Starting AI Manager-Agent System")
        print(f"[NOTE] User Request: {user_request}")
        
        # Step 1: Manager creates execution plan
        print("[MGR] Manager Agent is creating execution plan...")
        manager_plan = self.manager.create_execution_plan(user_request)
        print("[OK] Manager Plan Created")
        
        # Step 2: Coder executes with retry logic
        print("[CODE] Coder Agent is generating and executing code...")
        result = self.coder.generate_and_execute_code(manager_plan, user_request, use_visualization)
        
        if result["status"] == "success":
            print(f"[OK] SUCCESS after {result['attempts']} attempt(s)")
            return f"""
Physics Problem Solved Successfully!

Problem: {user_request}

Results:
{result['output']}

Generated code executed successfully.
"""
        else:
            print(f"[ERR] FAILED after {result['attempts']} attempts")
            return f"""
Physics Solver Failed

Problem: {user_request}

Error: {result['error']}

Please try rephrasing the problem or breaking it into simpler parts.
"""

# ... (all your existing physics agent code)

# Global instance
physics_orchestrator = AICodeOrchestrator()

def run_physics_calculation(problem_description: str, use_visualization: bool = True) -> str:
    """
    Solve physics problems requiring code execution and visualization
    
    Args:
        problem_description: The physics problem to solve
        use_visualization: Whether to include plots/visualizations
        
    Returns:
        String result with calculations and outputs
    """
    try:
        return physics_orchestrator.process_request(problem_description, use_visualization)
    except Exception as e:
        return f"Physics solver error: {str(e)}"