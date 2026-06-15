<h1 align="center">
  <br>
  🤖 Shadow AI Assistant
  <br>
</h1>

<p align="center">
  <b>A real-time, multimodal AI assistant powered by Google Gemini with voice, vision, long-term memory, a knowledge graph, and a deep scientific toolset.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-3.1%20Flash%20Live-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyMatGen-Materials%20Science-009688?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Wolfram-Alpha-DD1100?style=for-the-badge&logo=wolfram&logoColor=white" />
</p>

---

## ✨ What is Shadow?

**Shadow** is a personal AI assistant built around Google's **Gemini 3.1 Flash Live Preview** model for real-time interaction, and **Gemini 3.1 Flash Lite** for memory processing. It runs as a real-time CLI agent that can hear you, see your screen or camera, think, remember you between sessions, and execute a rich suite of scientific and computer-control tools — all in a single conversation loop.

Think of it as a local Jarvis: voice-in, voice-out, with persistent memory, tool use, and the ability to do everything from Wolfram Alpha math to DFT simulations to CAD design.

---

## 🎥 Modes

| Mode | Description |
|---|---|
| `none` (default) | Audio-only — microphone input + spoken responses |
| `screen` | Streams your desktop live to the model |
| `camera` | Streams your webcam live to the model |

```bash
python core/code5.py --mode none      # audio only
python core/code5.py --mode screen    # screen share
python core/code5.py --mode camera    # webcam
```

---

## 🧠 Core Architecture

```
cli/
├── core/
│   ├── code5.py            # Main real-time agent (AudioLoop class)
│   ├── groq2.py            # Groq-backed text agent with visualization templates
│   ├── tool_config.py      # Unified tool registry (ALL_TOOLS)
│   └── hardware_watchdog.py
├── Tools/
│   ├── memory_tools.py     # Long-term memory + vector DB + knowledge graph
│   ├── knowledge_graph.py  # Person/relationship graph (NetworkX)
│   ├── wolfram_orchestrator_tool.py
│   ├── materials_orchestrator_tool.py
│   ├── pymatgen_tools_v6.py
│   ├── dft_tool.py         # DFT simulation engine
│   ├── md_tool.py          # Molecular dynamics
│   ├── cad_tool.py / freecad_tool.py / autocad_tool.py
│   ├── computer_control_tool.py
│   ├── face_recognition_tool.py
│   ├── arxiv_tool.py
│   ├── latex_renderer_tool.py
│   ├── music_tool.py
│   ├── google_maps_tool.py
│   └── ... (50+ tools total)
├── memory/                 # Persistent flat-file + vector memory store
├── prompts/                # System prompt, SPR generator, retrieval prompt
└── logs/                   # User-defined protocol overrides
```

The agent is built around a single `AudioLoop` class that runs five concurrent asyncio tasks:

| Task | Role |
|---|---|
| `send_text` | Reads keyboard input from the CLI |
| `listen_audio` | Streams microphone PCM @ 16 kHz |
| `send_realtime` | Forwards audio/video frames to Gemini Live |
| `receive_audio` | Handles model responses, tool calls, transcriptions |
| `play_audio` | Plays model audio back through speakers @ 24 kHz |

---

## 🗃️ Memory System

Shadow has a **three-layer memory architecture**:

### 1. Short-Term Memory
A rolling `deque(maxlen=20)` that holds the last 20 turns in-process for context continuity within a session.

### 2. Long-Term Memory (Flat-file + Vector DB)
- **Store:** Compresses facts via an SPR (Sparse Priming Representation) LLM pass and writes them to `memory/memory.txt` with a JSON index.
- **Retrieve:** Hybrid search — BM25 keyword retrieval fused with ChromaDB vector similarity search, then synthesized by a retrieval LLM.
- **Memory Types:** `fact`, `preference`, `experience`, `insight`, `procedure`, `protocol`
- **Rate-limited:** Max 10 memory API calls per minute to avoid quota exhaustion.

### 3. Knowledge Graph (People & Relationships)
- Powered by **NetworkX**, persisted to JSON.
- Tracks people (name, description, disambiguation) and directed relationships (friend, sibling, colleague, etc.).
- Supports multi-hop graph traversal up to N degrees of separation.

### User-Defined Protocols
Any instruction like *"from now on, always call me Captain"* is saved to `logs/protocol.txt` and re-injected into the system prompt on every restart.

### Session Resumption
Gemini Live session handles are persisted to `session_handle.txt` so conversations can resume across restarts without losing context.

---

## 🔬 Scientific Tools

Shadow ships with a deep stack of scientific simulation and analysis tools:

### Materials Science
| Tool | Description |
|---|---|
| **Materials Project** | Query `mp-*` entries, band structures, DOS, phase diagrams, Pourbaix diagrams, elasticity, piezoelectric, dielectric, XAS, battery data |
| **PyMatGen v6** | Solar cell efficiency (Shockley-Queisser), surface slab generation, doped structures, Wulff shapes, phase stability |
| **DFT Engine** | Density functional theory simulations |
| **MD Tool** | Molecular dynamics simulations |
| **Crystal Viewer** | 3D crystal structure visualization |
| **Structure Workspace** | CIF file manipulation and structure analysis |
| **Perovskite Builder** | Perovskite structure generation |

### Mathematics & Physics
| Tool | Description |
|---|---|
| **Wolfram Alpha (Orchestrated)** | `simple_query`, `step_by_step`, `plot_url`, `unit_conversion`, `solve_equation`, `spectrum`, `phase_diagram`, `quantum_property`, `Shockley-Queisser`, `chemical_analysis`, `physical_constant`, batch queries |
| **Physics Agent** | Code-based physics simulations with optional visualization |
| **Optics Tool** | Optical simulation and ray tracing |
| **Device Simulator** | Electronic device simulation |
| **Multiphysics** | FEniCS-based multiphysics simulations |

### Engineering & CAD
| Tool | Description |
|---|---|
| **CAD Tool** | Parametric CAD model generation |
| **FreeCAD Tool** | FreeCAD integration |
| **AutoCAD Tool** | AutoCAD integration |

### Research & Information
| Tool | Description |
|---|---|
| **ArXiv** | Search papers, fetch by ID |
| **LaTeX Renderer** | Render LaTeX to PDF/PNG |
| **PDF Reader** | Extract text from PDFs |
| **Web Scraper** | Scrape web pages |
| **File Downloader** | Download files from URLs |
| **Google Maps** | Location queries and mapping |

### System & Computer Control
| Tool | Description |
|---|---|
| **Computer Control** | Mouse, keyboard, window management |
| **Face Recognition** | Identify faces from camera |
| **System Stats** | CPU, RAM, disk, network, battery, temperature |
| **File Watcher** | Monitor filesystem changes |
| **Python REPL** | Execute Python code in a sandbox |
| **Music Tool** | Play/control music |
| **Data Plotter** | Generate charts and plots |
| **GNOME Tool** | Linux desktop automation |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- A microphone and speakers
- API keys (see below)

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd cli

# Create and activate a virtual environment
python -m venv shadow_env
shadow_env\Scripts\activate      # Windows
# source shadow_env/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root (see `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
WOLFRAM_APP_ID=your_wolfram_alpha_app_id
MP_API_KEY=your_materials_project_api_key
```

> **Get your keys:**
> - Gemini: [aistudio.google.com](https://aistudio.google.com)
> - Groq: [console.groq.com](https://console.groq.com)
> - Wolfram Alpha: [developer.wolframalpha.com](https://developer.wolframalpha.com)
> - Materials Project: [materialsproject.org/api](https://materialsproject.org/api)

### 3. Run Shadow

```bash
# From the cli/ directory
python core/code5.py

# With screen share
python core/code5.py --mode screen

# With webcam
python core/code5.py --mode camera
```

Press `q` + Enter to quit.

---

## 🗣️ Example Interactions

```
> What is the band gap of silicon?
> Find me recent arxiv papers on perovskite solar cells
> Store this: my sister's name is Priya, she's a doctor in Bangalore
> Who is my sister?
> Solve the Schrodinger equation for a particle in a box
> What is 500 miles in kilometers?
> Take a screenshot and tell me what's on my screen
> Play some music
> Search for mp-149 on Materials Project and show me the band structure
```

---

## 🔧 Configuration

### System Prompt
Edit `prompts/shadow_system_prompt.txt` to customize Shadow's personality, style, and base instructions.

### User Protocols
Tell Shadow any persistent behavioral rule at runtime:
```
> From now on, always respond in a formal tone.
> Always call me by my first name, Arun.
```
These are saved automatically and reloaded on every restart.

### Tool Selection
Edit `core/tool_config.py` to add, remove, or modify the tools exposed to the agent.

---

## 📁 Key Files

| File | Purpose |
|---|---|
| `core/code5.py` | Main agent entry point |
| `core/groq2.py` | Groq-backed text reasoning agent |
| `core/tool_config.py` | Tool registry (ALL_TOOLS) |
| `Tools/memory_tools.py` | Long-term memory system |
| `Tools/knowledge_graph.py` | People & relationships graph |
| `Tools/wolfram_orchestrator_tool.py` | Wolfram Alpha interface |
| `Tools/materials_orchestrator_tool.py` | Materials Project interface |
| `prompts/shadow_system_prompt.txt` | System personality prompt |
| `logs/protocol.txt` | User-defined behavior rules |
| `memory/memory.txt` | Long-term memory store |

---

## 🛡️ Privacy & Security

- All memory is stored **locally** on your machine.
- API keys are loaded from `.env` and never hardcoded.
- Add `.env` and `memory/` to your `.gitignore` before pushing.
- Face recognition data is stored locally in `Tools/known_faces/`.

---

## 🚫 .gitignore Recommendations

```gitignore
.env
memory/
logs/protocol.txt
session_handle.txt
__pycache__/
*.pyc
shadow_env/
build/
dist/
*.spec
config/.env
config/*.json
```

---

## 📜 License

Copyright © 2026 **Arun Raj**

Free to use, modify, and build upon — for any purpose, personal, academic, or commercial.
**One rule: credit me.** A citation, mention, or acknowledgement in your work is all that is required.

> Arun Raj — Shadow AI (https://github.com/Arun-Raj-C-R/shadow-ai)

See [LICENSE](LICENSE) for full terms.

---

<p align="center">
  Built with ❤️ — <i>"I am Shadow. I am always listening."</i>
</p>
