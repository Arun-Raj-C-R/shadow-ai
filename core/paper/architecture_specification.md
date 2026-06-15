# Aetheris: Core Software Architecture and Tool Integration Specification

This document provides a comprehensive technical specification of the **Aetheris** autonomous scientific agentic framework. It details the runtime execution loops, real-time WebSocket communication channels, structured tool-calling handlers, numerical solvers, and multi-agent self-evolution pipelines.

---

## 1. Real-Time Multimodal Communication Core

Aetheris operates on an asynchronous, bidirectional streaming execution loop. Unlike traditional block-request LLM integrations, Aetheris maintains a continuous state connection to a multimodal LLM (Gemini Live API) over a persistent WebSocket session.

### 1.1 WebSockets & Frame Streaming Loop
The client interface continuously records raw audio and visual frame buffers from the user's workspace environment:
* **Audio Input:** Captures monaural PCM audio at a 16kHz sampling rate with a 16-bit depth (256kbps stream).
* **Video Input:** Captures screen coordinates and active windows using OpenCV (`cv2.VideoCapture`), compressing frame buffers into JPEG format at a rate of 1 frame per second (fps).
* **Communication Protocol:** JSON-wrapped binary websocket streams transport audio/video payloads, structured as:

```json
{
  "client_content_upload": {
    "parts": [
      {
        "mime_type": "audio/pcm;rate=16000",
        "data": "<base64_encoded_pcm_chunk>"
      },
      {
        "mime_type": "image/jpeg",
        "data": "<base64_encoded_jpeg_frame>"
      }
    ]
  }
}
```

---

## 2. Structured Tool-Calling and Dispatch Bus

Aetheris handles complex scientific queries by translating multimodal user intents into structured tool execution commands. When the model requests computational support, it halts streaming output and issues a structured `toolCall` request.

```
+------------------+                   +----------------------+                   +---------------------+
|                  |   Websocket Stream|                      |   Tool Schema List|                     |
|   USER / CLIENT  +------------------>+ AETHERIS ORCHESTRATOR+------------------>+ MULTIMODAL LLM      |
|                  |<------------------+    (code5.py)        |<------------------+ (Gemini Live API)   |
+------------------+   Audio Response  +-------+---^----------+   Tool Call JSON  +---------------------+
                                               |   |
                                       Execute |   | Return Payload
                                               v   |
                                       +-------+---+----------+
                                       | WORKSPACE SOLVER BUS |
                                       | (DFT/CAD/FDTD/TCAD)  |
                                       +----------------------+
```

### 2.1 The Dispatch Loop Sequence
1. **Interception:** The `Orchestrator` receives a tool call frame over the active session.
2. **Schema Validation:** The incoming JSON call structure (method name and parameter arguments) is checked against the registered python tool definitions in `Tools/`.
3. **Execution Routing:** The call is routed to the corresponding Python module (e.g., calling `run_dft_solver` in `Tools/dft_tool.py`).
4. **Sandboxed Evaluation:** Python executes the solver within the local workspace environment. If a execution error occurs, it is caught, wrapped in an error structure, and returned without halting the session.
5. **Session Update:** The return payload is wrapped in a `toolResponse` frame and piped back to the Gemini Live session, allowing the LLM to resume reasoning:

```json
{
  "tool_response": {
    "function_responses": [
      {
        "response": {
          "output": {
            "status": "success",
            "converged_energy_ha": -74.965821,
            "bandgap_ev": 3.42
          }
        },
        "id": "call_dft_001"
      }
    ]
  }
}
```

---

## 3. Workspace Tool Directories and API Mappings

The primary tools located in `Tools/` provide specialized physics and chemistry routines.

```
d:\Project File\Shadow\Shadow\Brain\Shadow2\cli\Tools\
â”œâ”€â”€ dft_tool.py                      # Quantum chemistry molecular DFT engine
â”œâ”€â”€ cad_tool.py                      # Parametric boundary representation (B-Rep) modeler
â”œâ”€â”€ optics_tool.py                   # 2D Finite-Difference Time-Domain (FDTD) wave solver
â”œâ”€â”€ device_simulator_tool.py         # 1D Drift-Diffusion TCAD diode solver
â”œâ”€â”€ multiphysics_tool.py             # 2D Heat/Electrostatics finite-difference solver
â”œâ”€â”€ gnome_tool.py                    # GNoME stable materials database explorer
â”œâ”€â”€ materials_orchestrator_tool.py   # Materials Project API client wrapper
â”œâ”€â”€ memory_tools.py                  # ChromaDB vector retrieval & homeostasis loop
â””â”€â”€ self_evolve_tool.py              # Self-evolution Researcher-Surgeon loop
```

### 3.1 Quantum Chemistry Engine (`Tools/dft_tool.py`)
This module calculates electronic structure energies without relying on external quantum chemistry packages.
* **Basis Sets:** Scaled STO-3G expansions from Hydrogen ($Z=1$) through Xenon ($Z=54$).
* **Core Integrals:**
  - Overlap ($S_{\mu\nu}$), kinetic energy ($T_{\mu\nu}$), and nuclear attraction ($V_{\mu\nu}$) are calculated analytically using the **Obara-Saika** (OS) recursion scheme.
  - Electron Repulsion Integrals (ERI) are computed using the **McMurchie-Davidson** algorithm, utilizing 8-fold index symmetry to optimize memory usage.
* **Exchange-Correlation ($V_{xc}$):** Integrates LDA (VWN5) and GGA (PBE) potentials over a multicenter grid combining Gauss-Chebyshev radial grids and angular Lebedev points, partitioned with Becke weights.
* **SCF Acceleration:** Implements Pulay's **Direct Inversion in the Iterative Subspace (DIIS)** to damp charge sloshing by resolving the error matrix constraint:
  $$e_i = F_i P_i S - S P_i F_i$$

### 3.2 Parametric CAD Modeler (`Tools/cad_tool.py`)
Provides geometric modeling features using a custom Boundary Representation (B-Rep) kernel.
* **Data Abstractions:** Represents shapes using connected topological classes: `Solid`, `Shell`, `Face`, `Wire`, `HalfEdge`, `Edge`, and `Vertex`.
* **Euler Operators:** Ensures topological validity by applying Euler-PoincarÃ© shell equations:
  $$V - E + F - (L - F) - 2(S - G) = 0$$
* **CSG Boolean Engine:** Performs Union, Intersection, and Difference operations using Binary Space Partitioning (BSP) trees to split and classify polygons.
* **Sketch Solver:** Resolves geometric constraints (distance, parallelism, perpendicularity) using a Levenberg-Marquardt optimizer.

### 3.3 Electromagnetics 2D FDTD Wave Solver (`Tools/optics_tool.py`)
Solves Maxwell's equations on a staggered grid.
* **Algorithm:** Implements a 2D Yee grid updating transverse electric (TE) wave equations:
  $$H_x^{n+1/2}\left(i, j+1/2\right) = H_x^{n-1/2}\left(i, j+1/2\right) - \frac{\Delta t}{\mu \Delta y} \left[ E_z^n\left(i, j+1\right) - E_z^n\left(i, j\right) \right]$$
* **Boundaries:** Uses uniaxial perfectly matched layers (UPML) to damp outgoing waves at grid edges.

### 3.4 Semiconductor TCAD Simulator (`Tools/device_simulator_tool.py`)
Simulates charge transport in semiconductor devices.
* **Physics Model:** Solves 1D coupled Poisson and drift-diffusion equations for electron ($n$) and hole ($p$) concentrations.
* **Discretization:** Applies the Scharfetter-Gummel scheme to ensure numeric stability during current calculation.
* **AutoCAD Interface:** Integrates ActiveX automation (`win32com.client`) to draw computed device junctions and depletion widths directly in CAD models:
  $$W_{dep} = \sqrt{\frac{2\varepsilon_s (V_{bi} - V_a)(N_A + N_D)}{q N_A N_D}}$$

### 3.5 Materials Discovery Databases (`Tools/gnome_tool.py`)
Queries Google DeepMind's stable compounds dataset.
* **Search Mechanics:** Elements are parsed using exact element regular expressions to avoid false positives (e.g. matching `S` without returning compound matches for `Sb`, `Si`, or `Sr`):
  ```python
  pattern = rf"'(?:{element})'"  # Matches exact list items in the element string
  ```

### 3.6 Long-Term Memory & Homeostasis Loop (`Tools/memory_tools.py`)
Coordinates contextual memory retrieval and monitors agent state.
* **Vector Search:** Manages a local ChromaDB collection using L2 distance metrics.
* **Fallback Search:** Falls back to keyword-based regex searches if ChromaDB is offline.
* **Compression Loop:** Uses background threads to compress interactions into Semantic Prose Representations (SPR), minimizing token footprint.
* **Homeostatic Variables:**
  - **Dopamine:** Tracks task success, increasing search depth and step permissions when high.
  - **Cortisol:** Tracks tool execution errors. If Cortisol passes a threshold of 7, the system switches to defensive programming mode, enforcing stricter prompt assertions and safety checks.

---

## 4. Multi-Agent Self-Evolution Pipeline

Aetheris can modify and repair its own tools using a closed-loop multi-agent self-evolution system (`Tools/self_evolve_tool.py`).

```
[Diagnostics Phase]                  [Implementation Phase]               [Verification Phase]
  (Researcher Agent)                   (Surgeon Agent)                      (Reviewer Agent)
         +                                    +                                    +
         |                                    |                                    |
         | Scans workspace logs               |                                    |
         | and detects failures               |                                    |
         +----------------------------------->+                                    |
         | Writes surgeon_instructions.txt    | Reads target code file             |
         |                                    | and applies line-range edits       |
         |                                    +----------------------------------->+
         |                                    | Writes modified python files       | compiles and executes
         |                                    |                                    | automated unit tests
         |                                    |                                    +-----------------------+
         |                                    |<-----------------------------------+                       |
         |                                    | Feedback Loop: If tests fail,      |                       | PASS
         |                                    | Surgeon applies corrective patches |                       v
         |                                    +                                    |               [Deploy Fix]
```

### 4.1 Evolution Execution Flow
1. **Researcher Agent:** Scans the codebase, analyzes dependency errors, and writes step-by-step instructions to `surgeon_instructions.txt`.
2. **Surgeon Agent:** Reads the instructions and target files. It uses line-range matching and replacements rather than full file overwrites, preserving existing code and comments.
3. **Reviewer Agent:** Performs static syntax analysis (`ast.parse`) and runs unit tests. If errors occur, it logs the traceback and instructs the Surgeon to patch the issue. The cycle repeats until the tests pass.
