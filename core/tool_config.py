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
        "parameters": {
            "type": "OBJECT",
            "properties": { 
                "query": { 
                    "type": "STRING", 
                    "description": "The user's question or topic to search for in memory" 
                } 
            },
            "required": ["query"]
        }
    },
    {
        "name": "update_ai_protocol",
        "description": "Updates or adds a new behavior, rule, or protocol for the AI. Use this ONLY when the user explicitly requests an important, permanent change in how you act, respond, or address them (e.g., 'From now on, call me Captain' or 'Never mention the weather').",
        "parameters": {
            "type": "OBJECT",
            "properties": { 
                "protocol_request": { 
                    "type": "STRING", 
                    "description": "The user's full request for the behavior change" 
                } 
            },
            "required": ["protocol_request"]
        }
    },
    {
        "name": "add_memory_feedback",
        "description": "Provide feedback on memory usefulness to help the system learn and improve. Use when memories are particularly helpful or need correction.",
        "parameters": {
            "type": "OBJECT",  # ✅ Fixed: Should be string "OBJECT" not Type.OBJECT
            "properties": {
                "memory_id": {
                    "type": "STRING",  # ✅ Fixed: Should be string "STRING"
                    "description": "The ID of the memory to provide feedback on"
                },
                "feedback": {
                    "type": "STRING",  # ✅ Fixed: Should be string "STRING"
                    "description": "Detailed feedback about why the memory was helpful or needs improvement"
                },
                "was_helpful": {
                    "type": "BOOLEAN",  # ✅ Fixed: Should be string "BOOLEAN"
                    "description": "Whether the memory was helpful (true) or needs correction (false)"
                    # Note: No "default" in Google's tool definitions - remove this line
                }
            },
            "required": ["memory_id", "feedback", "was_helpful"]
        }
    }
]
# Add to the combined list (near the end, before ALL_TOOLS)
SYSTEM_STATS_DEFINITION = [
    {
        "name": "get_system_stats",
        "description": "Collect and summarize current system performance metrics. Reports CPU usage, core/thread count, and frequency; RAM and swap utilization; disk usage and I/O; network data transfer; system uptime; battery status; and temperature sensors (CPU, GPU, or motherboard when available). Use when the user requests information about resource usage, thermal status, or overall system load before running intensive tasks.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    }
]

# Physics Tool Definition
PHYSICS_TOOL_DEFINITIONS = [
    {
        "name": "run_physics_calculation",
        "description": "Execute code-based physics calculations and simulations. Use for problems requiring mathematical modeling, simulations, data visualization, or complex calculations that need programmatic solutions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "problem_description": {
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
# ==============================================================================
# --- KNOWLEDGE GRAPH TOOL DEFINITIONS ---
# ==============================================================================

KNOWLEDGE_GRAPH_TOOL_DEFINITIONS = [
    {
        "name": "graph_add_person",
        "description": "Add a new person to the knowledge graph with their description and context. Use when the user mentions someone new or provides details about a person. The AI should call this proactively when it detects new people in conversation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING", "description": "Full name of the person"},
                "description": {"type": "STRING", "description": "Brief description (college, profession, how they're connected, key dates)"},
                "disambiguation": {"type": "STRING", "description": "Unique identifying context if name might be shared (e.g., 'NIT Calicut, MSc Chemistry, met 2024')"},
                "aliases": {"type": "STRING", "description": "Comma-separated alternative names or nicknames"}
            },
            "required": ["name", "description"]
        }
    },
    {
        "name": "graph_add_relationship",
        "description": "Add or update a relationship between two people in the knowledge graph. Relationships are directional (e.g., 'Arun Raj' -> 'friend' -> 'John'). Use whenever the user mentions how two people are connected.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "person1_name": {"type": "STRING", "description": "Name of the first person (source of relationship)"},
                "person2_name": {"type": "STRING", "description": "Name of the second person (target of relationship)"},
                "relation_type": {"type": "STRING", "description": "Type of relationship: friend, girlfriend, boyfriend, spouse, parent, child, sibling, colleague, classmate, teacher, mentor, mother_in_law, father_in_law, cousin, uncle, aunt, neighbor, boss"}
            },
            "required": ["person1_name", "person2_name", "relation_type"]
        }
    },
    {
        "name": "graph_query_person",
        "description": "Look up a person in the knowledge graph. Returns their profile, description, all relationships, and linked memories. Use when the user asks about someone (e.g., 'tell me about Divya', 'who is Gopika?'). If multiple people share the same name, returns all matches with disambiguation info.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name_query": {"type": "STRING", "description": "Name to search for (partial match supported)"},
                "context": {"type": "STRING", "description": "Optional context clues to disambiguate (e.g., 'from NIT', 'my college friend')"}
            },
            "required": ["name_query"]
        }
    },
    {
        "name": "graph_query_relationship",
        "description": "Find relationships for a person or between people. Use for questions like 'who are my friends?', 'how is X related to Y?', 'who is my mother-in-law?'",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "person_name": {"type": "STRING", "description": "Name of the person to query relationships for"},
                "relation_type": {"type": "STRING", "description": "Optional filter: only show this type of relationship"}
            },
            "required": ["person_name"]
        }
    },
    {
        "name": "graph_get_connections",
        "description": "Get the network of connections around a person up to N degrees of separation. Use for exploring 'who knows who' or understanding social circles.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "person_name": {"type": "STRING", "description": "Center person to explore connections from"},
                "depth": {"type": "STRING", "description": "How many degrees of separation to explore (1-3, default 2)"}
            },
            "required": ["person_name"]
        }
    },
    {
        "name": "graph_get_summary",
        "description": "Get a summary of the entire knowledge graph: total people, relationships, recent additions. Use when the user asks about their social network or 'who do you know about?'",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
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
    *KNOWLEDGE_GRAPH_TOOL_DEFINITIONS,
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