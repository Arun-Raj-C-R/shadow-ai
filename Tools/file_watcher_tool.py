# tools/file_watcher_tool.py
import asyncio
import base64
import os
from pathlib import Path
from typing import Optional
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ----------------------------------------------------------------------
#  CONFIG: Folder to watch
# ----------------------------------------------------------------------
WATCHED_DIR = Path("watched_folder")
WATCHED_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
#  Async queue to push new file events into the main loop
# ----------------------------------------------------------------------
file_event_queue: asyncio.Queue = asyncio.Queue()

# ----------------------------------------------------------------------
#  Watchdog Handler
# ----------------------------------------------------------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        # Optional: filter by extension
        if file_path.suffix.lower() in {".txt", ".pdf", ".jpg", ".png", ".jpeg", ".json", ".csv"}:
            self.loop.call_soon_threadsafe(
                file_event_queue.put_nowait, file_path
            )

# ----------------------------------------------------------------------
#  Start the file watcher (non-blocking)
# ----------------------------------------------------------------------
def start_watcher(loop: asyncio.AbstractEventLoop) -> Observer:
    event_handler = NewFileHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, str(WATCHED_DIR), recursive=False)
    observer.start()
    print(f"File watcher started on: {WATCHED_DIR.resolve()}")
    return observer

# ----------------------------------------------------------------------
#  Encode file to base64 (for sending to Gemini)
# ----------------------------------------------------------------------
def encode_file(file_path: Path) -> Optional[dict]:
    try:
        mime_type = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".json": "application/json",
            ".csv": "text/csv",
        }.get(file_path.suffix.lower(), "application/octet-stream")

        data = file_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return {
            "mime_type": mime_type,
            "data": b64,
            "filename": file_path.name
        }
    except Exception as e:
        print(f"Failed to encode {file_path}: {e}")
        return None