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
    rebuild_vector_database
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
