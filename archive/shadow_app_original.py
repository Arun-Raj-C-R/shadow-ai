import os
import sys
import asyncio
import base64
import io
import time
import pathlib
import threading
import traceback
import pyaudio
import PIL.Image
import cv2
import mss
import numpy as np
import psutil
import webview
import websockets
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Ensure we can load sibling directories (core & Tools) when running from anywhere
CLI_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(CLI_DIR))
sys.path.insert(0, str(CLI_DIR / "core"))
sys.path.insert(0, str(CLI_DIR / "Tools"))

# Load config and definitions
from tool_config import ALL_TOOLS
from memory_tools import (
    run_store_logic,
    run_retrieve_logic,
    run_update_protocol_logic,
    get_memory_health,
    add_memory_feedback,
    load_prompt_from_file,
    PROTOCOL_FILE_PATH,
    SHADOW_PROMPT_PATH,
    find_similar_memories,
    get_vector_stats,
    rebuild_vector_database
)
from arxiv_tool import search_arxiv, get_arxiv_papers_by_id
from file_downloader_tool import download_file_from_url
from physics_agent_tool import run_physics_calculation
from wolfram_orchestrator_tool import (
    simple_wolfram_query, query_with_podstate, query_with_assumption,
    get_material_property, get_step_by_step, get_plot_url,
    get_unit_conversion, solve_equation, get_spectrum,
    get_phase_diagram, get_quantum_property, get_shockley_
            webview.windows[0].evaluate_js(f"window.updateState('{state}');")

    def add_transcript(self, role, text):
        # Escape single quotes and newlines for safe Javascript evaluation
        clean = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", "")
        if webview.windows:
            webview.windows[0].evaluate_js(f"window.addTranscript('{role}', '{clean}');")

    def notify_tool(self, tool_name, is_active):
        active_val = "true" if is_active else "false"
        if webview.windows:
            webview.windows[0].evaluate_js(f"window.notifyToolActive('{tool_name}', {active_val});")

# -------------------------------------------------------------
# Main Application Launcher
# -------------------------------------------------------------
def main():
    bridge = ShadowBridge()
    ui_html_path = CLI_DIR / "core" / "shadow_ui.html"

    if not ui_html_path.exists():
        print(f"Error: UI file not found at {ui_html_path}")
        sys.exit(1)

    print("Launching Shadow Desktop HUD Application...")
    
    # Custom Windows WebView2 config
    window = webview.create_window(
        title="SHADOW - Tactical Assistant Interface",
        url=str(ui_html_path.resolve()),
        js_api=bridge,
        width=1100,
        height=750,
        min_size=(900, 600),
        background_color="#03060f"
    )

    # Start PyWebView loop (blocks main thread until closed)
    webview.start(gui="edgehtml" if sys.platform == "win32" and not hasattr(webview, "OPEN_MSHTML") else "cef", debug=False)
    
    # Clean standby triggers upon window close
    bridge.stop_shadow()
    print("Application closed safely.")

if __name__ == "__main__":
    main()

