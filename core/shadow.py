import os
import asyncio
import base64
import io
import traceback
import argparse
import pathlib
import time
from collections import deque
import websockets
import cv2
import pyaudio
import PIL.Image
import mss
from google import genai

from google.genai import types
# --- FIX: Added specific type imports ---
from google.genai.types import GoogleSearch, Tool, FunctionDeclaration, Schema, Type

from dotenv import load_dotenv
# --- Import the tool DEFINITIONS from the config file ---
from tool_config import ALL_TOOLS

# --- Import the tool FUNCTIONS from the module ---
# --- Import the tool FUNCTIONS from the module ---
from memory_tools import (
    run_store_logic,
    run_retrieve_logic, 
    run_update_protocol_logic,
    get_memory_health,
    add_memory_feedback,
    load_prompt_from_file,
    PROTOCOL_FILE_PATH,
    SHADOW_PROMPT_PATH,
    # Vector Database Functions
    find_similar_memories,
    get_vector_stats, 
    rebuild_vector_database,
    # Knowledge Graph Functions
    graph_add_person,
    graph_add_relationship,
    graph_query_person,
    graph_query_relationship,
    graph_get_connections,
    graph_get_summary
)


# --- Import new tool functions ---
from arxiv_tool import (
    search_arxiv,
    get_arxiv_papers_by_id
)
from file_downloader_tool import download_file_from_url
from physics_agent_tool import run_physics_calculation
# --- Wolfram Alpha Tools ---
from wolfram_orchestrator_tool import (
    simple_wolfram_query,
    query_with_podstate,
    query_with_assumption,
    get_material_property,
    get_step_by_step,
    get_plot_url,
    get_unit_conversion,
    solve_equation,
    get_spectrum,
    get_phase_diagram,
    get_quantum_property,
    get_shockley_queisser,
    is_valid_query,
    get_summary_box,
    start_async_query,
    get_sound,
    get_mathml,
    batch_query,
    get_spoken_answer,
    get_simple_image,
    plot_and_show,
    get_all_visualizations,
    chemical_analysis,
    physical_constant
)

# --- Materials Project Tools ---
from materials_orchestrator_tool import (
    search_mp_by_formula,
    get_mp_properties,
    get_mp_band_structure,
    get_mp_dos,
    discover_mp_materials,
    get_mp_data,
    get_materials_ids,
    get_structure_by_material_id,
    get_entries_in_system,
    get_electronic_structure_data,
    get_phonon_bandstructure_by_material_id,
    get_thermo_data,
    get_pourbaix_entries,
    get_phase_diagram_from_entries,
    get_wulff_shape,
    get_surface_data,
    get_elasticity_data,
    get_piezoelectric_data,
    get_dielectric_data,
    get_magnetism_data,
    get_xas_data,
    get_battery_data,
    get_oxidation_states,
    query_mp
)

from system_stats_tool import get_system_stats

# --- PyMatGen Tools ---
from pymatgen_tools_v6 import (
    calculate_solar_efficiency_v6,
    generate_surface_slab_v6,
    generate_doped_structure_v6,
    plot_band_dos_v6,
    analyze_phase_stability_v6,
    analyze_wulff_shape_v6,
    PYMATGEN_TOOL_DEFINITIONS_V6
)

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")


# ==============================================================================
# --- REAL-TIME AGENT CONFIGURATION ---
# ==============================================================================

# --- Audio Config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Model and Client Config ---
LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
DEFAULT_MODE = "none"
SHORT_TERM_MEMORY = deque(maxlen=20)

# --- Session Resumption Storage ---
HANDLE_FILE = "session_handle.txt"

# ----------------------------
# Utilities: file helpers
# ----------------------------
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ----------------------------
# Session handle helpers
# ----------------------------
def load_session_handle():
    if os.path.exists(HANDLE_FILE):
        txt = read_file(HANDLE_FILE).strip()
        return txt if txt else None
    return None

def save_session_handle(handle: str):
    if not handle:
        return
    write_file(HANDLE_FILE, handle)

# ----------------------------
# Live audio/video session
# ----------------------------
previous_session_handle = load_session_handle()

# --- System Prompt ---
def _load_combined_system_prompt() -> str:
    """Loads the main Shadow prompt and appends any user protocols."""
    main_prompt = load_prompt_from_file(SHADOW_PROMPT_PATH)
    
    protocol_updates = load_prompt_from_file(PROTOCOL_FILE_PATH, is_optional=True).strip()
    
    if protocol_updates:
        print("âš™ï¸ [System] Loading additional user protocols...")
        return (
            f"{main_prompt}\n\n"
            f"--- ADDITIONAL USER-DEFINED PROTOCOLS ---\n"
            f"These are rules you must follow, defined by the user:\n"
            f"{protocol_updates}\n"
            f"--- END OF USER-DEFINED PROTOCOLS ---"
        )
    else:
        print("âš™ï¸ [System] Loading standard system prompt.")
        return main_prompt

SYSTEM_INSTRUCTION = _load_combined_system_prompt()

# --- Client Initialization ---
live_client = genai.Client(
    http_options={"api_version": "v1alpha"}, 
    api_key=API_KEY,
)

# We'll create the config dynamically in the run method
pya = pyaudio.PyAudio()


# ==============================================================================
# --- REAL-TIME AGENT CLASS ---
# ==============================================================================

import websockets
import traceback
from collections import deque

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.current_session_handle = previous_session_handle
        
        # Memory rate limiting
        self.memory_call_count = 0
        self.last_memory_call_time = 0

    # Color codes for terminal output
    COLOR_RESET = "\033[0m"
    COLOR_USER = "\033[1;32m"  # Bold Green
    COLOR_SHADOW = "\033[1;36m"  # Bold Sky Blue
    COLOR_THINKING = "\033[1;33m"  # Bold Yellow
    COLOR_TOOL = "\033[1;35m"  # Bold Magenta
    COLOR_ERROR = "\033[1;31m"  # Bold Red

    async def _rate_limited_memory_call(self, func, *args, **kwargs):
        """Prevent memory API rate limits"""
        current_time = time.time()
        
        # Limit: max 10 memory calls per minute
        if self.memory_call_count >= 10 and current_time - self.last_memory_call_time < 60:
            wait_time = 60 - (current_time - self.last_memory_call_time)
            print(f"â° [Memory Rate Limit] Waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            self.memory_call_count = 0
        
        self.memory_call_count += 1
        self.last_memory_call_time = current_time
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _provide_memory_feedback(self, user_input: str, shadow_response: str, was_helpful: bool):
        """Automatically provide feedback for memory learning"""
        # Only provide feedback for clearly helpful/unhelpful responses
        helpful_indicators = ['thanks', 'thank you', 'perfect', 'great', 'exactly', 'awesome', 'good', 'nice']
        unhelpful_indicators = ['wrong', 'incorrect', 'not what', "that's not", 'no that', 'bad', 'terrible']
        
        user_lower = user_input.lower()
        
        if was_helpful and any(indicator in user_lower for indicator in helpful_indicators):
            # Store this as a positive learning experience
            feedback_data = f"User appreciated response about: {user_input[:100]}... Response: {shadow_response[:100]}..."
            await self._rate_limited_memory_call(
                run_store_logic, 
                feedback_data, 
                "experience"
            )
            print(f"ðŸ“ [Memory] Stored positive feedback")
        
        elif not was_helpful and any(indicator in user_lower for indicator in unhelpful_indicators):
            # Store this as correction learning
            correction_data = f"User corrected: {user_input[:100]}... Correct response should avoid: {shadow_response[:100]}..."
            await self._rate_limited_memory_call(
                run_store_logic, 
                correction_data, 
                "insight"
            )
            print(f"ðŸ“ [Memory] Stored correction feedback")

    async def send_text(self):
        """Handles user text input from the CLI."""
        while True:
            text = await asyncio.to_thread(
                input,
                f"{self.COLOR_USER}message > {self.COLOR_RESET}",
            )
            SHORT_TERM_MEMORY.append(f"User: {text}")
            if text.lower() == "q":
                break
            
            # Use the correct API method for text input
            await self.session.send(
                input=text or ".",
                end_of_turn=True
            )

    def _get_frame(self, cap):
        """Captures and processes a single camera frame."""
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Continuously captures camera frames and adds them to the out_queue."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

    def _get_screen(self):
        """Captures and processes a single screenshot."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            i = sct.grab(monitor)
            mime_type = "image/jpeg"
            img = PIL.Image.frombytes("RGB", i.size, i.rgb)
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)
            image_bytes = image_io.read()
            return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        """Continuously captures screenshots and adds them to the out_queue."""
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        """Sends microphone audio and video frames to the model."""
        while True:
            msg = await self.out_queue.get()
            # Use the correct API method for real-time input
            await self.session.send(
                input=msg,
                end_of_turn=False
            )

    async def listen_audio(self):
        """Listens to the microphone and adds audio chunks to the out_queue."""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Receives responses, prints transcriptions, and handles tool calls."""
        global previous_session_handle
        
        full_user_transcription = ""
        full_model_transcription = ""

        try:
            while True:
                turn = self.session.receive()
                
                try:
                    async for response in turn:
                        # ==========================================================================
                        # --- SESSION RESUMPTION HANDLING ---
                        # ==========================================================================
                        if getattr(response, "session_resumption_update", None):
                            update = response.session_resumption_update
                            if getattr(update, "resumable", False) and getattr(update, "new_handle", None):
                                self.current_session_handle = update.new_handle
                                save_session_handle(previous_session_handle)
                                print(f"{self.COLOR_SHADOW}[SESSION] New handle saved: {previous_session_handle}{self.COLOR_RESET}")

                        # Handle interruption (Barge-in)
                        if (getattr(response, "server_content", None) and 
                            getattr(response.server_content, "interrupted", False)):
                            print(f"\n{self.COLOR_USER}[INTERRUPT] User spoke, stopping playback.{self.COLOR_RESET}")
                            # Clear the audio playback queue to stop old audio
                            while not self.audio_in_queue.empty():
                                try:
                                    self.audio_in_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    continue

                        # Handle generation completion
                        if (getattr(response, "server_content", None) and 
                            getattr(response.server_content, "generation_complete", False)):
                            print(f"{self.COLOR_SHADOW}[INFO] Model has completed generation for this turn.{self.COLOR_RESET}")

                        # ==========================================================================
                        # --- EXISTING RESPONSE HANDLING ---
                        # ==========================================================================
                        
                        # Handle audio data for playback
                        if hasattr(response, 'data') and response.data:
                            print(" " * 80, end="\r")  # Clear any thinking line
                            self.audio_in_queue.put_nowait(response.data)
                            continue

                        # Handle server content with transcriptions
                        if getattr(response, "server_content", None):
                            server_content = response.server_content
                            
                            # Handle thinking thoughts
                            if hasattr(server_content, 'thinking_thoughts') and server_content.thinking_thoughts:
                                thinking_text = server_content.thinking_thoughts.text
                                if thinking_text and thinking_text.strip():
                                    print(f"{self.COLOR_THINKING}\n[THINKING] {thinking_text.strip()}{self.COLOR_RESET}")
                            
                            # Handle model turn with parts
                            if hasattr(server_content, 'model_turn') and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if hasattr(part, 'text') and part.text and part.text.strip():
                                        print(f"{self.COLOR_THINKING}[THINKING]: {part.text.strip()}{self.COLOR_RESET}", end="\r")
                                    elif hasattr(part, 'executable_code') and part.executable_code:
                                        print(f"{self.COLOR_TOOL}\n[CODE EXECUTION]: {part.executable_code.code}{self.COLOR_RESET}")
                                    elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                                        print(f"{self.COLOR_TOOL}\n[CODE RESULT]: {part.code_execution_result.output}{self.COLOR_RESET}")

                            # Handle user transcription
                            if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                                user_text_chunk = server_content.input_transcription.text
                                if user_text_chunk and user_text_chunk.strip():
                                    full_user_transcription += user_text_chunk + " "

                            # Handle model transcription
                            if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                                model_text_chunk = server_content.output_transcription.text
                                if model_text_chunk and model_text_chunk.strip():
                                    full_model_transcription += model_text_chunk + "   "

                        # Print transcriptions and provide feedback
                        if full_user_transcription.strip():
                            final_user_text_chunk = full_user_transcription.strip()
                            print(f"{self.COLOR_USER}\nðŸŽ¤ User: {final_user_text_chunk}{self.COLOR_RESET}")
                            
                            # Provide automatic feedback for learning when we have both user and model text
                            if full_model_transcription.strip():
                                await self._provide_memory_feedback(
                                    final_user_text_chunk,
                                    full_model_transcription.strip(),
                                    was_helpful=True
                                )
                            
                            full_user_transcription = ""

                        if full_model_transcription.strip():
                            final_model_text_chunk = full_model_transcription.strip()
                            print(f"{self.COLOR_SHADOW}ðŸ¤– Shadow: {final_model_text_chunk}{self.COLOR_RESET}")
                            full_model_transcription = ""

                        # Handle tool calls
                        tool_call = getattr(response, 'tool_call', None)
                        if tool_call and hasattr(tool_call, 'function_calls') and tool_call.function_calls:
                            print(f"{self.COLOR_TOOL}\n[TOOL CALL] Model requested tool execution...{self.COLOR_RESET}")
                            function_responses = []
                            
                            for fc in tool_call.function_calls:
                                print(f"{self.COLOR_TOOL}   Tool: {fc.name}{self.COLOR_RESET}")
                                result = None
                                
                                try:
                                    # --- Route to the IMPORTED functions ---
                                    # === ENHANCED MEMORY TOOLS (RATE LIMITED) ===
                                    if fc.name == "store_data":
                                        data_to_store = fc.args.get("data_to_store", "")
                                        result = await self._rate_limited_memory_call(run_store_logic, data_to_store)
                                    
                                    elif fc.name == "retrieve_data":
                                        query = fc.args.get("query", "")
                                        context = "\n".join(SHORT_TERM_MEMORY)
                                        result = await self._rate_limited_memory_call(run_retrieve_logic, query, context)
                                    
                                    elif fc.name == "update_ai_protocol":
                                        request = fc.args.get("protocol_request", "")
                                        result = await self._rate_limited_memory_call(run_update_protocol_logic, request)
                                        result += " (Note: A restart may be required for changes to take full effect.)"
                                    
                                    elif fc.name == "get_memory_health":
                                        result = await self._rate_limited_memory_call(get_memory_health)
                                        # Add these new tool cases in the tool call section:

                                    elif fc.name == "find_similar_memories":
                                        content = fc.args.get("content", "")
                                        n_results = fc.args.get("n_results", 5)
                                        result = await self._rate_limited_memory_call(find_similar_memories, content, n_results)

                                    elif fc.name == "get_memory_system_health":
    # Combine regular health with vector stats
                                        base_health = await self._rate_limited_memory_call(get_memory_health)
                                        vector_stats = await self._rate_limited_memory_call(get_vector_stats)
                                        result = {
                                            "base_system": base_health,
                                            "vector_database": vector_stats
                                         }

                                    elif fc.name == "rebuild_vector_database":
                                        result = await self._rate_limited_memory_call(rebuild_vector_database)
                                        result = "Vector database rebuild initiated. This may take a few moments."
                                        
                                        # Add this case to your tool call handlers in main1.py (around line 450):

                                    elif fc.name == "add_memory_feedback":
                                        memory_id = fc.args.get("memory_id", "")
                                        feedback = fc.args.get("feedback", "")
                                        was_helpful = fc.args.get("was_helpful", True)
                                        result = await self._rate_limited_memory_call(add_memory_feedback, memory_id, feedback, was_helpful)

                                    # === KNOWLEDGE GRAPH TOOLS ===
                                    elif fc.name == "graph_add_person":
                                        name = fc.args.get("name", "")
                                        description = fc.args.get("description", "")
                                        disambiguation = fc.args.get("disambiguation", "")
                                        aliases = fc.args.get("aliases", "")
                                        result = await self._rate_limited_memory_call(
                                            graph_add_person, name, description, disambiguation, aliases
                                        )

                                    elif fc.name == "graph_add_relationship":
                                        p1 = fc.args.get("person1_name", "")
                                        p2 = fc.args.get("person2_name", "")
                                        rel = fc.args.get("relation_type", "")
                                        result = await self._rate_limited_memory_call(
                                            graph_add_relationship, p1, p2, rel
                                        )

                                    elif fc.name == "graph_query_person":
                                        name_query = fc.args.get("name_query", "")
                                        context = fc.args.get("context", "")
                                        result = await self._rate_limited_memory_call(
                                            graph_query_person, name_query, context
                                        )

                                    elif fc.name == "graph_query_relationship":
                                        person = fc.args.get("person_name", "")
                                        rel_type = fc.args.get("relation_type", "")
                                        result = await self._rate_limited_memory_call(
                                            graph_query_relationship, person, rel_type
                                        )

                                    elif fc.name == "graph_get_connections":
                                        person = fc.args.get("person_name", "")
                                        depth = int(fc.args.get("depth", "2"))
                                        result = await self._rate_limited_memory_call(
                                            graph_get_connections, person, depth
                                        )

                                    elif fc.name == "graph_get_summary":
                                        result = await self._rate_limited_memory_call(graph_get_summary)

                                    # === EXISTING TOOLS (NON-RATE LIMITED) ===
                                    elif fc.name == "search_arxiv":
                                        query = fc.args.get("search_query", "")
                                        max_res = fc.args.get("max_results", 5)
                                        sort_by = fc.args.get("sort_by", "submittedDate")
                                        sort_order = fc.args.get("sort_order", "descending")
                                        result = await asyncio.to_thread(search_arxiv, query, max_res, sort_by, sort_order)

                                    elif fc.name == "get_arxiv_papers_by_id":
                                        ids = fc.args.get("id_list", [])
                                        result = await asyncio.to_thread(get_arxiv_papers_by_id, ids)

                                    elif fc.name == "download_file_from_url":
                                        url = fc.args.get("url", "")
                                        filename = fc.args.get("filename", "")
                                        if not url or not filename:
                                            result = "FAILURE: Both 'url' and 'filename' are required."
                                        else:
                                            result = await asyncio.to_thread(download_file_from_url, url, filename)

                                    # === WOLFRAM ALPHA TOOLS ===
                                    elif fc.name in ["simple_wolfram_query", "get_short_answer"]:
                                        result = await asyncio.to_thread(simple_wolfram_query, fc.args.get("query", ""))
                                    elif fc.name == "get_spoken_answer":
                                        result = await asyncio.to_thread(get_spoken_answer, fc.args.get("query", ""))
                                    elif fc.name == "query_with_podstate":  
                                        result = await asyncio.to_thread(query_with_podstate, **fc.args)
                                    elif fc.name == "query_with_assumption":  
                                        result = await asyncio.to_thread(query_with_assumption, **fc.args)
                                    elif fc.name == "get_material_property":
                                        result = await asyncio.to_thread(get_material_property, **fc.args)
                                    elif fc.name == "get_step_by_step":
                                        result = await asyncio.to_thread(get_step_by_step, fc.args.get("query", ""))
                                    elif fc.name == "get_plot_url":
                                        result = await asyncio.to_thread(get_plot_url, fc.args.get("query", ""))
                                    elif fc.name == "get_simple_image":
                                        query = fc.args.get("query", "")
                                        save_path = fc.args.get("save_path")
                                        open_browser = fc.args.get("open_browser", False)
                                        result = await asyncio.to_thread(get_simple_image, query, save_path, open_browser)
                                    elif fc.name == "plot_and_show":
                                        result = await asyncio.to_thread(plot_and_show, **fc.args)
                                    elif fc.name == "get_all_visualizations":
                                        result = await asyncio.to_thread(get_all_visualizations, fc.args.get("query", ""))
                                    elif fc.name == "get_unit_conversion":
                                        result = await asyncio.to_thread(get_unit_conversion, **fc.args)
                                    elif fc.name == "solve_equation":
                                        result = await asyncio.to_thread(solve_equation, **fc.args)
                                    elif fc.name == "get_spectrum":
                                        result = await asyncio.to_thread(get_spectrum, **fc.args)
                                    elif fc.name == "get_phase_diagram":
                                        result = await asyncio.to_thread(get_phase_diagram, fc.args.get("compound", ""))
                                    elif fc.name == "get_quantum_property":
                                        result = await asyncio.to_thread(get_quantum_property, **fc.args)
                                    elif fc.name == "get_shockley_queisser":
                                        result = await asyncio.to_thread(get_shockley_queisser, fc.args.get("bandgap", 0.0))
                                    elif fc.name == "is_valid_query":
                                        result = await asyncio.to_thread(is_valid_query, fc.args.get("query", ""))
                                    elif fc.name == "get_summary_box":
                                        result = await asyncio.to_thread(get_summary_box, fc.args.get("entity", ""))
                                    elif fc.name == "start_async_query":
                                        result = await asyncio.to_thread(start_async_query, fc.args.get("query", ""))
                                    elif fc.name == "get_sound":
                                        result = await asyncio.to_thread(get_sound, fc.args.get("query", ""), fc.args.get("save_path"))
                                    elif fc.name == "get_mathml":
                                        result = await asyncio.to_thread(get_mathml, fc.args.get("query", ""))
                                    elif fc.name == "batch_query":
                                        result = await asyncio.to_thread(batch_query, fc.args.get("queries", []))
                                    elif fc.name == "chemical_analysis":
                                        result = await asyncio.to_thread(chemical_analysis, **fc.args)
                                    elif fc.name == "physical_constant":
                                        result = await asyncio.to_thread(physical_constant, **fc.args)

                                    # === MATERIALS PROJECT TOOLS ===
                                    elif fc.name == "search_mp_by_formula":
                                        formula = fc.args.get("formula", "")
                                        result = await asyncio.to_thread(search_mp_by_formula, formula)
                                    elif fc.name == "get_mp_properties":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_mp_properties, mp_id)
                                    elif fc.name == "get_mp_data":
                                        criteria = fc.args.get("criteria", {})
                                        properties = fc.args.get("properties")
                                        result = await asyncio.to_thread(get_mp_data, criteria, properties)
                                    elif fc.name == "get_materials_ids":
                                        formula = fc.args.get("formula", "")
                                        result = await asyncio.to_thread(get_materials_ids, formula)
                                    elif fc.name == "get_structure_by_material_id":
                                        material_id = fc.args.get("material_id", "")
                                        final = fc.args.get("final", True)
                                        conventional_unit_cell = fc.args.get("conventional_unit_cell", False)
                                        result = await asyncio.to_thread(get_structure_by_material_id, material_id, final, conventional_unit_cell)
                                    elif fc.name == "get_entries_in_system":
                                        elements = fc.args.get("elements", [])
                                        compatible_only = fc.args.get("compatible_only", True)
                                        result = await asyncio.to_thread(get_entries_in_system, elements, compatible_only)
                                    elif fc.name == "get_mp_band_structure":
                                        mp_id = fc.args.get("material_id", "")
                                        line_mode = fc.args.get("line_mode", True)
                                        result = await asyncio.to_thread(get_mp_band_structure, mp_id, line_mode)
                                    elif fc.name == "get_mp_dos":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_mp_dos, mp_id)
                                    elif fc.name == "get_electronic_structure_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_electronic_structure_data, mp_id)
                                    elif fc.name == "get_phonon_bandstructure_by_material_id":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_phonon_bandstructure_by_material_id, mp_id)
                                    elif fc.name == "get_thermo_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_thermo_data, mp_id)
                                    elif fc.name == "get_pourbaix_entries":
                                        elements = fc.args.get("elements", [])
                                        result = await asyncio.to_thread(get_pourbaix_entries, elements)
                                    elif fc.name == "get_phase_diagram_from_entries":
                                        entries = fc.args.get("entries", [])
                                        result = await asyncio.to_thread(get_phase_diagram_from_entries, entries)
                                    elif fc.name == "get_wulff_shape":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_wulff_shape, mp_id)
                                    elif fc.name == "get_surface_data":
                                        mp_id = fc.args.get("material_id", "")
                                        miller_index = fc.args.get("miller_index")
                                        result = await asyncio.to_thread(get_surface_data, mp_id, miller_index)
                                    elif fc.name == "get_elasticity_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_elasticity_data, mp_id)
                                    elif fc.name == "get_piezoelectric_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_piezoelectric_data, mp_id)
                                    elif fc.name == "get_dielectric_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_dielectric_data, mp_id)
                                    elif fc.name == "get_magnetism_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_magnetism_data, mp_id)
                                    elif fc.name == "get_xas_data":
                                        mp_id = fc.args.get("material_id", "")
                                        spectrum_type = fc.args.get("spectrum_type", "XANES")
                                        result = await asyncio.to_thread(get_xas_data, mp_id, spectrum_type)
                                    elif fc.name == "get_battery_data":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_battery_data, mp_id)
                                    elif fc.name == "get_oxidation_states":
                                        mp_id = fc.args.get("material_id", "")
                                        result = await asyncio.to_thread(get_oxidation_states, mp_id)
                                    elif fc.name == "query_mp":
                                        criteria = fc.args.get("criteria", {})
                                        properties = fc.args.get("properties")
                                        result = await asyncio.to_thread(query_mp, criteria, properties)
                                    elif fc.name == "discover_mp_materials":
                                        result = await asyncio.to_thread(
                                                discover_mp_materials,
                                                elements=fc.args.get("elements", []),
                                                nelements=fc.args.get("nelements"),
                                                crystal_system=fc.args.get("crystal_system"),
                                                band_gap_min=fc.args.get("band_gap_min"),
                                                band_gap_max=fc.args.get("band_gap_max"),
                                                max_energy_above_hull=fc.args.get("max_energy_above_hull")
                                        )

                                    # --- SOLAR EFFICIENCY (FIXED) ---
                                    elif fc.name == "calculate_solar_efficiency_v6":
                                        band_gap_ev   = fc.args.get("band_gap_ev", 0.0)
                                        thickness_um  = fc.args.get("thickness_um", 0.5)
                                        is_direct_gap = fc.args.get("is_direct_gap", True)
                                        try:
                                            result = await asyncio.to_thread(
                                                calculate_solar_efficiency_v6,
                                                band_gap_ev,
                                                thickness_um,
                                                is_direct_gap
                                            )
                                        except Exception as e:
                                            result = f"Error during solarâ€‘efficiency calculation: {e}"

                                    # --- SLAB & DEFECT ---
                                    elif fc.name == "generate_surface_slab_v6":
                                        props_json = fc.args.get("properties_json", "")
                                        miller = fc.args.get("miller_index", [1, 0, 0])
                                        layers = fc.args.get("layers", 4)
                                        vacuum = fc.args.get("vacuum_A", 15.0)
                                        result = await asyncio.to_thread(
                                            generate_surface_slab_v6,
                                            props_json, miller, layers, vacuum
                                        )

                                    elif fc.name == "generate_doped_structure_v6":
                                        props_json = fc.args.get("properties_json", "")
                                        site_idx = fc.args.get("site_index", 0)
                                        dopant = fc.args.get("dopant", "")
                                        fraction = fc.args.get("fraction", 0.0)
                                        result = await asyncio.to_thread(
                                            generate_doped_structure_v6,
                                            props_json, site_idx, dopant, fraction
                                        )

                                    # --- PLOTTING & STABILITY ---
                                    elif fc.name == "plot_band_dos_v6":
                                        bs_json = fc.args.get("band_structure_json", "")
                                        dos_json = fc.args.get("dos_json", "")
                                        filename = fc.args.get("filename", "band_dos_plot")
                                        result = await asyncio.to_thread(plot_band_dos_v6, bs_json, dos_json, filename)
                                    
                                    elif fc.name == "analyze_phase_stability_v6":
                                        materials_json = fc.args.get("materials_list_json", "")
                                        result = await asyncio.to_thread(analyze_phase_stability_v6, materials_json)

                                    elif fc.name == "analyze_wulff_shape_v6":
                                        props_json = fc.args.get("properties_json", "")
                                        energies_json = fc.args.get("miller_energies_json", "{}")
                                        result = await asyncio.to_thread(analyze_wulff_shape_v6, props_json, energies_json)
                                    
                                    elif fc.name == "get_system_stats":
                                        result = await asyncio.to_thread(get_system_stats)

                                    # === PHYSICS AGENT TOOL ===
                                    elif fc.name == "run_physics_calculation":
                                        problem_desc = fc.args.get("problem_description", "")
                                        use_viz = fc.args.get("use_visualization", True)
                                        result = await asyncio.to_thread(run_physics_calculation, problem_desc, use_viz)

                                    else:
                                        print(f"Warning: Unhandled tool call: {fc.name}")
                                        result = {"error": f"Unknown function {fc.name}"}

                                    # Print result
                                    print(f"\n[Tool Agent Result]: {result}\n")
                                    function_response = types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": str(result)}
                                    )
                                    function_responses.append(function_response)

                                except Exception as e:
                                    print(f"Error during tool execution: {e}")
                                    function_responses.append(types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"error": str(e)}
                                    ))

                            print("[Sending tool response(s)...]\n")
                            await self.session.send_tool_response(function_responses=function_responses)

                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"{self.COLOR_ERROR}WebSocket connection closed: {e}{self.COLOR_RESET}")
                    break
                except Exception as e:
                    print(f"{self.COLOR_ERROR}Error in response loop: {e}{self.COLOR_RESET}")
                    break

        except Exception as e:
            print(f"{self.COLOR_ERROR}Receive audio error: {e}{self.COLOR_RESET}")

    async def play_audio(self):
        """Plays audio from the audio_in_queue to the speakers."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            by_stream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, by_stream)

    async def run(self):
        """Starts and manages all asyncio tasks for the chat session."""
        global previous_session_handle

        max_retries = 2
        for retry in range(max_retries):
            try:
                # Create config with proper session resumption using LiveConnectConfig
                connect_config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    tools=ALL_TOOLS,
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    media_resolution="MEDIA_RESOLUTION_LOW",
                    speech_config={
                        "language_code": "en-US",
                        "voice_config": {"prebuilt_voice_config": {"voice_name": "puck"}}
                    },
                    context_window_compression=types.ContextWindowCompressionConfig(
                        trigger_tokens=25600,
                        sliding_window=types.SlidingWindow(target_tokens=12800)
                    ),
                    input_audio_transcription={},
                    output_audio_transcription={},
                    system_instruction={"parts": [{"text": SYSTEM_INSTRUCTION}]},
                    enable_affective_dialog=True,
                )
                
                # Add session resumption if we have a handle
                if self.current_session_handle:
                    connect_config.session_resumption = types.SessionResumptionConfig(
                        handle=self.current_session_handle
                    )
                    print(f"{self.COLOR_SHADOW}ðŸ”„ RESUMING existing session: {self.current_session_handle}{self.COLOR_RESET}")
                else:
                    print(f"{self.COLOR_SHADOW}ðŸš€ STARTING new session (no previous session found){self.COLOR_RESET}")

                print(f"{self.COLOR_SHADOW}Connecting to the service...{self.COLOR_RESET}")
                
                async with (
                    live_client.aio.live.connect(model=LIVE_MODEL, config=connect_config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session

                    if SYSTEM_INSTRUCTION:
                        print(f"System prompt loaded ({'with' if PROTOCOL_FILE_PATH.exists() else 'without'} protocols) and sent via config.")

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    
                    if self.video_mode == "camera":
                        print("Starting camera stream...")
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        print("Starting screen share stream...")
                        tg.create_task(self.get_screen())
                    else:
                        print("Running in audio-only mode.")

                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    await send_text_task
                    raise asyncio.CancelledError("User requested exit")

            except (websockets.exceptions.ConnectionClosedError, ConnectionError) as e:
                print(f"{self.COLOR_ERROR}Connection failed (attempt {retry + 1}/{max_retries}): {e}{self.COLOR_RESET}")
                if retry < max_retries - 1:
                    # Clear session handle and retry
                    previous_session_handle = None
                    self.current_session_handle = None
                    save_session_handle("")
                    print(f"{self.COLOR_SHADOW}Retrying with new session...{self.COLOR_RESET}")
                    await asyncio.sleep(2)
                else:
                    print(f"{self.COLOR_ERROR}All connection attempts failed.{self.COLOR_RESET}")
                    raise
            except asyncio.CancelledError:
                print("\nUser requested exit. Shutting down...")
                break
            except ExceptionGroup as EG:
                print("\n--- An Error Occurred ---")
                # Check if it's a connection error
                connection_error = False
                for exc in EG.exceptions:
                    if "ConnectionClosed" in str(exc) or "websocket" in str(exc).lower():
                        connection_error = True
                        break
                
                if connection_error and retry < max_retries - 1:
                    print(f"{self.COLOR_ERROR}Connection error detected. Retrying...{self.COLOR_RESET}")
                    previous_session_handle = None
                    self.current_session_handle = None
                    save_session_handle("")
                    await asyncio.sleep(2)
                    continue
                else:
                    traceback.print_exception(EG)
                    break
            except Exception as e:
                print(f"{self.COLOR_ERROR}Unexpected error: {e}{self.COLOR_RESET}")
                break
            finally:
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    self.audio_stream.close()
                pya.terminate()
                print("PyAudio terminated. Exiting.")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    
    if not API_KEY:
        print("\n--- ERROR ---")
        print("The 'GEMINI_API_KEY' environment variable is not set.")
        print("Please set it in your terminal or in a .env file.")
        exit(1)
    
    # Initialize the global genai client
    genai.Client(api_key=API_KEY)
    
    print("Session handle file:", os.path.abspath(HANDLE_FILE))
    print(f"Starting live session with mode: {args.mode}")
    
    # Show session status
    if previous_session_handle:
        print(f"ðŸ“ Session status: FOUND existing session - will RESUME")
    else:
        print(f"ðŸ“ Session status: No existing session - will START NEW")
    
    print("Press 'q' and Enter in the CLI to quit.")
    
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())