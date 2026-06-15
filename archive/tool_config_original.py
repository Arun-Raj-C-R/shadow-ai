"""
Contains the static tool definitions for the main agent.
"""

# Import the new tool definitions
from arxiv_tool import ARXIV_TOOL_DEFINITIONS
from file_downloader_tool import DOWNLOAD_TOOL_DEFINITIONS
from wolfram_orchestrator_tool import WOLFRAM_TOOL_DEFINITIONS
from materials_orchestrator_tool import MATERIALS_TOOL_DEFINITIONS
from system_stats_tool import get_system_stats
from pymatgen_tools_v6 import PYMATGEN_TOOL_DEFINITIONS_V6

MEMORY_TOOL_DEFINITIONS = [
    {
        "name": "store_data",
        "description": "Store user-provided facts, memories, or information into a long-term memory file. It is mostly related to users personal life: past experiences, likes, dislikes, future events, plans, preferences, and important personal information.",
        "parameters": {
            "type": "OBJECT",
            "properties": { 
                "data_to_store": { 
                    "type": "STRING", 
                    "description": "The specific fact, memory, or information to store in long-term memory" 
                } 
            },
            "required": ["data_to_store"]
        }
    },
    {
        "name": "retrieve_data",
        "description": "Recall, search, or retrieve information from the long-term memory file based on a user's query. It is mostly related to users personal life: past experiences, likes, dislikes, future events, plans, preferences, and important personal information.",
       
                    "type": "STRING",
                    "description": "Detailed description of the physics problem to solve"
                },
                "use_visualization": {
                    "type": "BOOLEAN", 
                    "description": "Whether to generate plots and visualizations (default: True)",
                    "default": True
                }
            },
            "required": ["problem_description"]
        }
    }
]
WOLFRAM_TOOL_DEFINITIONS=[

# ADD THIS TO WOLFRAM_TOOL_DEFINITIONS
{
    "name": "get_simple_image",
    "description": "Generate high-resolution Wolfram plot (Mandelbrot, fractals, complex visuals). Saves to 'plots/' folder.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Wolfram plot query"},
            "save_path": {"type": "string", "description": "Optional custom path"}
        },
        "required": ["query"]
    }
},
]
# --- MODIFICATION ---
# Combine all tool definitions
ALL_FUNCTION_DEFINITIONS = (
    *MEMORY_TOOL_DEFINITIONS,
    *ARXIV_TOOL_DEFINITIONS, 
    *DOWNLOAD_TOOL_DEFINITIONS,
    *WOLFRAM_TOOL_DEFINITIONS,
    *MATERIALS_TOOL_DEFINITIONS,
    *SYSTEM_STATS_DEFINITION,
    *PHYSICS_TOOL_DEFINITIONS,# ← NEW
    *PYMATGEN_TOOL_DEFINITIONS_V6 # <-- Use the new V6 set
)
# --- END MODIFICATION ---

# 2. Define the complete list of tools for the agent
# We export this variable to be used by the main app
ALL_TOOLS = [
    {"google_search": {}},
    {"code_execution": {}},
    # Use the combined list
    {"function_declarations": ALL_FUNCTION_DEFINITIONS}, 
]
