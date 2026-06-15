# music_tool.py
"""
Music Player Tool for SHADOW/Shadow AI
========================================
Plays music based on user request, mood, or emotion.
Uses FREE methods â€” no API keys needed:
  - yt-dlp for YouTube search + audio streaming
  - System default player / VLC / pygame for playback
  - Web search for song recommendations

Runs as an INSTANT tool â€” starts playback immediately.
"""

import os
import json
import subprocess
import threading
import logging
import webbrowser
from typing import Optional, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music_tool")

MUSIC_CACHE_DIR = os.path.join(os.path.dirname(__file__), "music_cache")
os.makedirs(MUSIC_CACHE_DIR, exist_ok=True)

# Track currently playing process so we can stop it
_current_player = None
_current_track = None


# ======================================================================
# YOUTUBE SEARCH + STREAM (via yt-dlp â€” free, no API key)
# ======================================================================

def _search_youtube(query: str, max_results: int = 5) -> list:
    """Search YouTube for songs using yt-dlp. Returns list of results."""
    try:
        cmd = [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return []

        entries = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    data = json.loads(line)
                    entries.append({
                        "title": data.get("title", "Unknown"),
                        "url": data.get("url") or f"https://www.youtube.com/watch?v={data.get('id', '')}",
                        "id": data.get("id", ""),
                        "duration": data.get("duration"),
                        "channel": data.get("channel") or data.get("uploader", "Unknown"),
                    })
                except json.JSONDecodeError:
                    continue
        return entries
    except FileNotFoundError:
        logger.warning("yt-dlp not found â€” install with: pip install yt-dlp")
        return []
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return []


def _get_audio_url(video_url: str) -> Optional[str]:
    """Get direct audio stream URL from a YouTube video."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--get-url",
            "--no-download",
            "--quiet",
            video_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.error(f"Get audio URL error: {e}")
    return None


def _download_audio(video_url: str, filename: str = None) -> Optional[str]:
    """Download audio to local file for playback."""
    try:
        if not filename:
            filename = f"track_{int(datetime.now().timestamp())}"
        output_path = os.path.join(MUSIC_CACHE_DIR, f"{filename}.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "-o", output_path,
            "--no-playlist",
            "--quiet",
            video_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Find the downloaded file
            mp3_path = os.path.join(MUSIC_CACHE_DIR, f"{filename}.mp3")
            if os.path.exists(mp3_path):
                return mp3_path
            # Check for other extensions
            for ext in ["mp3", "m4a", "opus", "webm", "wav"]:
                p = os.path.join(MUSIC_CACHE_DIR, f"{filename}.{ext}")
                if os.path.exists(p):
                    return p
    except Exception as e:
        logger.error(f"Download error: {e}")
    return None


# ======================================================================
# PLAYBACK METHODS (tries multiple approaches)
# ======================================================================

def _play_with_vlc(file_or_url: str) -> bool:
    """Play audio using VLC (if installed)."""
    global _current_player
    try:
        # Try python-vlc first
        import vlc
        player = vlc.MediaPlayer(file_or_url)
        player.play()
        _current_player = player
        return True
    except ImportError:
        pass

    # Try VLC executable
    vlc_paths = [
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
        "vlc",
    ]
    for vlc_path in vlc_paths:
        try:
            proc = subprocess.Popen(
                [vlc_path, "--intf", "dummy", "--play-and-exit", file_or_url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _current_player = proc
            return True
        except (FileNotFoundError, OSError):
            continue
    return False


def _play_with_ffplay(file_or_url: str) -> bool:
    """Play audio using ffplay (comes with ffmpeg)."""
    global _current_player
    try:
        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_or_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _current_player = proc
        return True
    except FileNotFoundError:
        return False


def _play_with_system(filepath: str) -> bool:
    """Play audio using the system default player."""
    global _current_player
    try:
        if os.name == "nt":
            os.startfile(filepath)
            return True
        elif os.name == "posix":
            proc = subprocess.Popen(["xdg-open", filepath])
            _current_player = proc
            return True
    except Exception:
        pass
    return False


def _open_in_browser(url: str) -> bool:
    """Open song in web browser as final fallback."""
    try:
        webbrowser.open(url)
        return True
    except Exception:
        return False


def _play_audio(source: str, title: str = "") -> Dict:
    """Try all playback methods in order."""
    global _current_track
    _current_track = title

    # Try playback methods in priority order
    methods = [
        ("VLC", lambda: _play_with_vlc(source)),
        ("ffplay", lambda: _play_with_ffplay(source)),
        ("System Player", lambda: _play_with_system(source) if os.path.isfile(source) else False),
        ("Browser", lambda: _open_in_browser(source) if source.startswith("http") else False),
    ]

    for method_name, method_fn in methods:
        try:
            if method_fn():
                return {
                    "status": "playing",
                    "title": title,
                    "method": method_name,
                    "source": source[:100],
                }
        except Exception:
            continue

    return {"status": "error", "error": "No playback method available. Install VLC or ffmpeg."}


# ======================================================================
# MOOD-BASED SONG RECOMMENDATIONS
# ======================================================================

MOOD_QUERIES = {
    "happy": "upbeat happy feel good songs",
    "sad": "sad emotional songs",
    "energetic": "high energy workout pump up songs",
    "relaxed": "chill relaxing lo-fi songs",
    "focused": "focus concentration study music instrumental",
    "romantic": "romantic love songs",
    "angry": "intense aggressive rock metal songs",
    "nostalgic": "nostalgic throwback classic hits",
    "sleepy": "sleep music ambient calm",
    "party": "party dance club hits",
    "motivational": "motivational inspirational songs",
    "melancholy": "melancholy bittersweet indie songs",
    "peaceful": "peaceful meditation ambient nature sounds",
    "workout": "workout gym motivation high energy",
}


# ======================================================================
# MAIN MUSIC TOOL ENTRY POINT
# ======================================================================

def play_music(
    query: str = None,
    mood: str = None,
    action: str = "play",
) -> str:
    """
    Main music tool entry point. Called by Shadow AI.

    Args:
        query:  Song name, artist, or description (e.g. "Shape of You Ed Sheeran")
        mood:   Mood-based playback (e.g. "happy", "sad", "relaxed", "focused")
        action: What to do:
                - "play"       : Search and play a song
                - "mood_play"  : Play based on mood/emotion
                - "stop"       : Stop current playback
                - "search"     : Search only, don't play
                - "recommend"  : Get recommendations based on mood

    Returns:
        JSON string with results
    """
    global _current_player, _current_track
    start_time = datetime.now()

    output = {
        "action": action,
        "query": query,
        "mood": mood,
        "timestamp": start_time.isoformat(),
    }

    try:
        # â”€â”€ STOP â”€â”€
        if action == "stop":
            if _current_player:
                try:
                    if hasattr(_current_player, 'stop'):
                        _current_player.stop()
                    elif hasattr(_current_player, 'terminate'):
                        _current_player.terminate()
                except Exception:
                    pass
                _current_player = None
            output["status"] = "stopped"
            output["stopped_track"] = _current_track or "Unknown"
            _current_track = None
            return json.dumps(output, indent=2)

        # â”€â”€ BUILD SEARCH QUERY â”€â”€
        search_query = query
        if action == "mood_play" or (not query and mood):
            mood_lower = (mood or "happy").lower()
            search_query = MOOD_QUERIES.get(mood_lower, f"{mood_lower} music songs")
            if query:
                search_query = f"{query} {search_query}"
            output["mood_search"] = search_query

        if not search_query:
            output["error"] = "No query or mood provided. Tell me a song name or your mood!"
            return json.dumps(output, indent=2)

        # â”€â”€ SEARCH â”€â”€
        print(f"ðŸŽµ Searching: {search_query}")
        results = _search_youtube(search_query)

        if not results:
            # Fallback: open YouTube search in browser
            yt_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
            _open_in_browser(yt_url)
            output["status"] = "opened_browser"
            output["message"] = f"Opened YouTube search for: {search_query}"
            output["url"] = yt_url
            return json.dumps(output, indent=2)

        output["search_results"] = results[:5]

        if action == "search" or action == "recommend":
            output["status"] = "results_found"
            return json.dumps(output, indent=2)

        # â”€â”€ PLAY â”€â”€
        top_result = results[0]
        video_url = top_result["url"]
        if not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={top_result['id']}"

        print(f"ðŸŽ¶ Playing: {top_result['title']}")
        print(f"   Channel: {top_result.get('channel', 'Unknown')}")

        # Stop current playback first
        if _current_player:
            try:
                if hasattr(_current_player, 'stop'):
                    _current_player.stop()
                elif hasattr(_current_player, 'terminate'):
                    _current_player.terminate()
            except Exception:
                pass

        # Try to get direct audio URL and play
        played = False

        # Method 1: Get audio stream URL and play with VLC/ffplay
        audio_url = _get_audio_url(video_url)
        if audio_url:
            play_result = _play_audio(audio_url, title=top_result["title"])
            if play_result.get("status") == "playing":
                output.update(play_result)
                played = True

        # Method 2: Download and play locally
        if not played:
            print("   â¬‡ï¸ Downloading audio...")
            local_file = _download_audio(video_url)
            if local_file:
                play_result = _play_audio(local_file, title=top_result["title"])
                if play_result.get("status") == "playing":
                    output.update(play_result)
                    played = True

        # Method 3: Open in browser
        if not played:
            _open_in_browser(video_url)
            output["status"] = "opened_browser"
            output["title"] = top_result["title"]
            output["url"] = video_url
            output["method"] = "Browser"

    except Exception as e:
        output["error"] = str(e)
        print(f"âŒ Music error: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    output["elapsed_seconds"] = round(elapsed, 2)

    result_json = json.dumps(output, indent=2, default=str)
    print(f"ðŸŽµ Music result: {result_json[:500]}")
    return result_json
