# computer_control_tool.py
"""
Computer Control Tool for SHADOW/Shadow AI
============================================
Provides mouse, keyboard, app, and browser control.
Also implements MASTER CONTROL MODE â€” AI-driven screen automation.

Dependencies: pip install pyautogui pygetwindow keyboard psutil
"""

import os
import sys
import json
import time
import subprocess
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger("computer_control")

# â”€â”€ Imports with graceful fallback â”€â”€
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.1
except ImportError:
    pyautogui = None
    logger.warning("pyautogui not installed: pip install pyautogui")

try:
    import pygetwindow as gw
except ImportError:
    gw = None
    logger.warning("pygetwindow not installed: pip install pygetwindow")

try:
    import psutil
except ImportError:
    psutil = None

# â”€â”€ Master Control State â”€â”€
master_control_active = False
_master_stop_event = threading.Event()


# ======================================================================
# MOUSE CONTROL
# ======================================================================

def mouse_move(x: int, y: int, duration: float = 0.3, expected_window: str = None) -> str:
    """Move mouse to absolute position (x, y)."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if expected_window and gw:
            active = gw.getActiveWindow()
            if not active or expected_window.lower() not in active.title.lower():
                return json.dumps({"error": f"Expected window '{expected_window}' not active. Current: {active.title if active else 'None'}"})

        pyautogui.moveTo(x, y, duration=duration)
        pos = pyautogui.position()
        if abs(pos.x - x) > 10 or abs(pos.y - y) > 10:
            return json.dumps({"error": f"Mouse failed to reach ({x}, {y}). Current pos: {pos.x}, {pos.y}"})
            
        return json.dumps({"status": "success", "action": "mouse_move", "x": pos.x, "y": pos.y})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mouse_move_relative(dx: int, dy: int, duration: float = 0.2) -> str:
    """Move mouse relative to current position. dx>0=right, dy>0=down."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.moveRel(dx, dy, duration=duration)
        pos = pyautogui.position()
        return json.dumps({"status": "success", "action": "mouse_move_relative",
                          "dx": dx, "dy": dy, "new_x": pos.x, "new_y": pos.y})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mouse_click(x: int = None, y: int = None, button: str = "left",
                clicks: int = 1, expected_window: str = None) -> str:
    """Click at position. If x,y omitted, clicks current position."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if expected_window and gw:
            active = gw.getActiveWindow()
            if not active or expected_window.lower() not in active.title.lower():
                return json.dumps({"error": f"Expected window '{expected_window}' not active. Current: {active.title if active else 'None'}"})

        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.2)
            pos = pyautogui.position()
            if abs(pos.x - x) > 10 or abs(pos.y - y) > 10:
                return json.dumps({"error": f"Mouse failed to reach ({x}, {y}). Current pos: {pos.x}, {pos.y}. Not clicking."})
            pyautogui.click(clicks=clicks, button=button)
        else:
            pyautogui.click(clicks=clicks, button=button)
            
        pos = pyautogui.position()
        return json.dumps({"status": "success", "action": "click",
                          "button": button, "clicks": clicks,
                          "x": pos.x, "y": pos.y})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mouse_scroll(amount: int, x: int = None, y: int = None) -> str:
    """Scroll. Positive=up, negative=down."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if x is not None and y is not None:
            pyautogui.scroll(amount, x, y)
        else:
            pyautogui.scroll(amount)
        return json.dumps({"status": "success", "action": "scroll", "amount": amount})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mouse_drag(start_x: int, start_y: int, end_x: int, end_y: int,
               duration: float = 0.5, button: str = "left") -> str:
    """Drag from start to end position."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.moveTo(start_x, start_y, duration=0.2)
        pyautogui.drag(end_x - start_x, end_y - start_y,
                       duration=duration, button=button)
        return json.dumps({"status": "success", "action": "drag",
                          "from": [start_x, start_y], "to": [end_x, end_y]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_mouse_position() -> str:
    """Get current mouse position."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    pos = pyautogui.position()
    screen = pyautogui.size()
    return json.dumps({"x": pos.x, "y": pos.y,
                       "screen_width": screen.width, "screen_height": screen.height})


# ======================================================================
# KEYBOARD CONTROL
# ======================================================================

def type_text(text: str, interval: float = 0.02, expected_window: str = None) -> str:
    """Type text character by character."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if expected_window and gw:
            active = gw.getActiveWindow()
            if not active or expected_window.lower() not in active.title.lower():
                return json.dumps({"error": f"Expected window '{expected_window}' not active. Current: {active.title if active else 'None'}"})

        pyautogui.typewrite(text, interval=interval)
        return json.dumps({"status": "success", "action": "type", "text": text})
    except Exception:
        # Fallback for unicode characters
        try:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            return json.dumps({"status": "success", "action": "type_paste", "text": text})
        except Exception as e:
            return json.dumps({"error": str(e)})


def press_key(key: str) -> str:
    """Press a single key (enter, tab, escape, space, backspace, delete, etc.)."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.press(key)
        return json.dumps({"status": "success", "action": "press", "key": key})
    except Exception as e:
        return json.dumps({"error": str(e)})


def hotkey(*keys) -> str:
    """Press a keyboard shortcut (e.g., ctrl+c, alt+tab, ctrl+shift+t)."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.hotkey(*keys)
        return json.dumps({"status": "success", "action": "hotkey", "keys": list(keys)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# APP CONTROL
# ======================================================================

def open_app(app_name: str) -> str:
    """Open an application by name (uses Windows Start menu search)."""
    try:
        # Method 1: Direct known paths
        known_apps = {
            "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
            "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "vscode": "code",
            "task manager": "taskmgr.exe",
        }

        app_lower = app_name.lower().strip()
        if app_lower in known_apps:
            path = known_apps[app_lower]
            subprocess.Popen(path, shell=True)
            return json.dumps({"status": "success", "action": "open_app",
                              "app": app_name, "method": "direct"})

        # Method 2: Windows Start menu search
        if pyautogui:
            pyautogui.hotkey("win")
            time.sleep(0.5)
            pyautogui.typewrite(app_name, interval=0.03)
            time.sleep(0.8)
            pyautogui.press("enter")
            time.sleep(0.5)
            return json.dumps({"status": "success", "action": "open_app",
                              "app": app_name, "method": "start_menu"})

        # Method 3: subprocess
        subprocess.Popen(app_name, shell=True)
        return json.dumps({"status": "success", "action": "open_app",
                          "app": app_name, "method": "subprocess"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def close_app(app_name: str) -> str:
    """Close an application by name."""
    if not psutil:
        return json.dumps({"error": "psutil not installed"})
    try:
        app_lower = app_name.lower().strip()
        closed = []
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if app_lower in proc.info['name'].lower():
                    proc.terminate()
                    closed.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if closed:
            return json.dumps({"status": "success", "action": "close_app",
                              "closed": closed})
        return json.dumps({"status": "not_found",
                          "message": f"No process matching '{app_name}' found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def list_windows() -> str:
    """List all visible windows."""
    if not gw:
        return json.dumps({"error": "pygetwindow not installed"})
    try:
        windows = []
        for w in gw.getAllWindows():
            if w.title and w.visible:
                windows.append({
                    "title": w.title[:80],
                    "x": w.left, "y": w.top,
                    "width": w.width, "height": w.height,
                })
        return json.dumps({"windows": windows[:20]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def focus_window(title_contains: str) -> str:
    """Bring a window to focus by partial title match."""
    if not gw:
        return json.dumps({"error": "pygetwindow not installed"})
    try:
        for w in gw.getAllWindows():
            if title_contains.lower() in w.title.lower() and w.visible:
                w.activate()
                time.sleep(0.3)
                return json.dumps({"status": "success", "action": "focus_window",
                                  "title": w.title})
        return json.dumps({"status": "not_found",
                          "message": f"No window matching '{title_contains}'"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# BROWSER / TAB CONTROL (via keyboard shortcuts)
# ======================================================================

def browser_new_tab(url: str = None) -> str:
    """Open a new browser tab, optionally navigate to URL."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.hotkey("ctrl", "t")
        time.sleep(0.3)
        if url:
            pyautogui.typewrite(url, interval=0.01)
            time.sleep(0.1)
            pyautogui.press("enter")
        return json.dumps({"status": "success", "action": "new_tab", "url": url})
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_close_tab() -> str:
    """Close the current browser tab."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.hotkey("ctrl", "w")
        return json.dumps({"status": "success", "action": "close_tab"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_switch_tab(direction: str = "next") -> str:
    """Switch to next or previous browser tab."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if direction == "next":
            pyautogui.hotkey("ctrl", "tab")
        else:
            pyautogui.hotkey("ctrl", "shift", "tab")
        return json.dumps({"status": "success", "action": "switch_tab",
                          "direction": direction})
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_navigate(url: str) -> str:
    """Navigate current tab to a URL."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.hotkey("ctrl", "l")  # Focus address bar
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")  # Select all
        pyautogui.typewrite(url, interval=0.01)
        pyautogui.press("enter")
        return json.dumps({"status": "success", "action": "navigate", "url": url})
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_search(query: str) -> str:
    """Search in the browser address bar."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        # Use clipboard for unicode support
        try:
            import pyperclip
            pyperclip.copy(query)
            pyautogui.hotkey("ctrl", "v")
        except ImportError:
            pyautogui.typewrite(query, interval=0.02)
        time.sleep(0.1)
        pyautogui.press("enter")
        return json.dumps({"status": "success", "action": "search", "query": query})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# MEDIA CONTROL (YouTube / any media player)
# ======================================================================

def media_play_pause() -> str:
    """Toggle play/pause on media (YouTube, Spotify, etc.)."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        # Try media key first
        try:
            import keyboard as kb
            kb.send("play/pause media")
            return json.dumps({"status": "success", "action": "play_pause",
                              "method": "media_key"})
        except Exception:
            pass
        # Fallback: spacebar (works for YouTube when focused)
        pyautogui.press("space")
        return json.dumps({"status": "success", "action": "play_pause",
                          "method": "spacebar"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def media_skip() -> str:
    """Skip to next track/video."""
    try:
        import keyboard as kb
        kb.send("next track")
        return json.dumps({"status": "success", "action": "skip"})
    except Exception:
        if pyautogui:
            pyautogui.hotkey("shift", "n")  # YouTube next
            return json.dumps({"status": "success", "action": "skip",
                              "method": "shift+n"})
        return json.dumps({"error": "No control method available"})


def media_previous() -> str:
    """Go to previous track/video."""
    try:
        import keyboard as kb
        kb.send("previous track")
        return json.dumps({"status": "success", "action": "previous"})
    except Exception:
        if pyautogui:
            pyautogui.hotkey("shift", "p")  # YouTube previous
            return json.dumps({"status": "success", "action": "previous",
                              "method": "shift+p"})
        return json.dumps({"error": "No control method available"})


def media_volume(direction: str = "up", amount: int = 5) -> str:
    """Adjust volume up or down."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        key = "volumeup" if direction == "up" else "volumedown"
        for _ in range(amount):
            pyautogui.press(key)
        return json.dumps({"status": "success", "action": "volume",
                          "direction": direction, "steps": amount})
    except Exception as e:
        return json.dumps({"error": str(e)})


def media_mute() -> str:
    """Toggle mute."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        pyautogui.press("volumemute")
        return json.dumps({"status": "success", "action": "mute_toggle"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def screenshot_region(x: int = 0, y: int = 0, width: int = 0,
                      height: int = 0) -> str:
    """Take a screenshot (full screen or region)."""
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if width > 0 and height > 0:
            img = pyautogui.screenshot(region=(x, y, width, height))
        else:
            img = pyautogui.screenshot()
        path = os.path.join(os.path.dirname(__file__),
                           f"screenshot_{int(time.time())}.png")
        img.save(path)
        return json.dumps({"status": "success", "action": "screenshot",
                          "path": path})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# AI CHAT INTERACTION â€” type, scroll, read, follow-up
# ======================================================================

def ai_type_and_send(text: str, press_enter: bool = True, expected_window: str = None) -> str:
    """
    Type a message into an AI chat interface (ChatGPT, Gemini, Claude,
    DeepSeek, Grok, etc.) and optionally press Enter to send.
    Assumes the chat input box is already focused.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        if expected_window and gw:
            active = gw.getActiveWindow()
            if not active or expected_window.lower() not in active.title.lower():
                return json.dumps({"error": f"Expected window '{expected_window}' not active. Current: {active.title if active else 'None'}"})

        # Use clipboard paste for unicode support
        try:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        except ImportError:
            pyautogui.typewrite(text, interval=0.02)
        time.sleep(0.3)
        if press_enter:
            pyautogui.press("enter")
        return json.dumps({"status": "success", "action": "ai_type_and_send",
                          "text": text[:100], "sent": press_enter})
    except Exception as e:
        return json.dumps({"error": str(e)})


def ai_scroll_down(presses: int = 5) -> str:
    """
    Scroll down in the AI chat response area.
    Uses Page Down for larger jumps to read through long responses.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        for _ in range(presses):
            pyautogui.press("pagedown")
            time.sleep(0.3)
        return json.dumps({"status": "success", "action": "ai_scroll_down",
                          "presses": presses})
    except Exception as e:
        return json.dumps({"error": str(e)})


def ai_scroll_up(presses: int = 5) -> str:
    """
    Scroll up in the AI chat to re-read earlier content.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        for _ in range(presses):
            pyautogui.press("pageup")
            time.sleep(0.3)
        return json.dumps({"status": "success", "action": "ai_scroll_up",
                          "presses": presses})
    except Exception as e:
        return json.dumps({"error": str(e)})


def ai_read_and_scroll_to_bottom(max_scrolls: int = 30, scroll_pause: float = 1.5) -> str:
    """
    Scroll down repeatedly through an AI response, reading all content
    until the bottom of the page/response is reached.

    Detection strategy for bottom:
    - Takes screenshot before and after scroll
    - If screenshots are identical â†’ no new content â†’ bottom reached
    - Also checks for common bottom indicators (input box, footer)

    Returns status with scroll count.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        scrolls_done = 0
        reached_bottom = False

        for i in range(max_scrolls):
            # Capture screen before scroll
            img_before = pyautogui.screenshot()
            before_data = np.array(img_before) if 'np' in dir() else None

            # Scroll down
            pyautogui.press("pagedown")
            time.sleep(scroll_pause)

            # Capture screen after scroll
            img_after = pyautogui.screenshot()
            after_data = np.array(img_after) if before_data is not None else None

            scrolls_done += 1

            # Compare: if screen didn't change, we hit bottom
            if before_data is not None and after_data is not None:
                try:
                    import numpy as _np
                    diff = _np.sum(_np.abs(before_data.astype(int) - after_data.astype(int)))
                    # Very small diff = same screen = bottom
                    if diff < 50000:
                        reached_bottom = True
                        break
                except Exception:
                    pass

            # Alt detection: use mouse scroll and check
            # After many scrolls, assume content exhausted
            if scrolls_done >= max_scrolls:
                reached_bottom = True

        return json.dumps({
            "status": "success",
            "action": "ai_read_and_scroll_to_bottom",
            "scrolls_done": scrolls_done,
            "reached_bottom": reached_bottom,
            "message": "Reached bottom of content" if reached_bottom
                       else f"Scrolled {scrolls_done} times, may need more"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def ai_check_bottom() -> str:
    """
    Check if the current view is at the bottom of the AI response.
    Takes two screenshots with a scroll attempt in between.
    If no change â†’ at bottom.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        img_before = pyautogui.screenshot()
        pyautogui.press("pagedown")
        time.sleep(0.8)
        img_after = pyautogui.screenshot()

        try:
            import numpy as _np
            before_arr = _np.array(img_before)
            after_arr = _np.array(img_after)
            diff = _np.sum(_np.abs(before_arr.astype(int) - after_arr.astype(int)))
            at_bottom = diff < 50000
        except Exception:
            at_bottom = False  # Can't determine, assume not

        # If we weren't at bottom, scroll back up to restore position
        if not at_bottom:
            pyautogui.press("pageup")
            time.sleep(0.3)

        return json.dumps({
            "status": "success",
            "action": "ai_check_bottom",
            "at_bottom": at_bottom
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def ai_full_interaction(query: str, max_read_scrolls: int = 30) -> str:
    """
    Full AI chat interaction cycle:
    1. Type and send the query
    2. Wait for response to start generating
    3. Scroll down reading all content until bottom
    4. Return status â€” ready for next query

    Use this for automated conversations with ChatGPT, Gemini web,
    Claude, DeepSeek, Grok, etc.
    """
    if not pyautogui:
        return json.dumps({"error": "pyautogui not installed"})
    try:
        results = {}

        # Step 1: Type and send
        send_result = json.loads(ai_type_and_send(query))
        results["send"] = send_result

        # Step 2: Wait for AI response to begin rendering
        time.sleep(4.0)

        # Step 3: Scroll through entire response
        scroll_result = json.loads(ai_read_and_scroll_to_bottom(
            max_scrolls=max_read_scrolls, scroll_pause=1.5))
        results["scroll"] = scroll_result

        # Step 4: Ready for next interaction
        results["status"] = "success"
        results["action"] = "ai_full_interaction"
        results["message"] = (
            f"Query sent. Scrolled {scroll_result.get('scrolls_done', 0)} times. "
            f"Bottom reached: {scroll_result.get('reached_bottom', False)}. "
            f"Screen now shows end of response â€” ready for follow-up."
        )

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# MASTER CONTROL MODE
# ======================================================================

def activate_master_control() -> str:
    """Activate master control mode â€” AI takes over mouse/keyboard."""
    global master_control_active, _master_stop_event
    master_control_active = True
    _master_stop_event.clear()

    # Start hotkey listener for Fn+F10 abort
    def _listen_abort():
        try:
            import keyboard as kb
            kb.wait("f10")
            deactivate_master_control()
            print("\nðŸ›‘ Master Control deactivated via F10")
        except Exception:
            pass

    threading.Thread(target=_listen_abort, daemon=True).start()

    print("\n" + "=" * 60)
    print("ðŸŽ® MASTER CONTROL MODE ACTIVATED")
    print("   AI now has mouse & keyboard control")
    print("   Say 'stop master control' or press F10 to deactivate")
    print("=" * 60)

    return json.dumps({
        "status": "activated",
        "message": "Master control is now active. I can see your screen and control mouse/keyboard. Tell me what to do, or I'll analyze the screen and act. Say 'stop master control' or press F10 to deactivate.",
        "screen": get_mouse_position(),
    })


def deactivate_master_control() -> str:
    """Deactivate master control mode."""
    global master_control_active
    master_control_active = False
    _master_stop_event.set()

    print("\nðŸ›‘ MASTER CONTROL MODE DEACTIVATED")

    return json.dumps({
        "status": "deactivated",
        "message": "Master control has been deactivated. I no longer control mouse/keyboard."
    })


def is_master_control_active() -> bool:
    """Check if master control is active."""
    return master_control_active


def computer_control(action: str, **kwargs) -> str:
    """
    Unified computer control entry point.

    Actions:
      Mouse: mouse_move, mouse_click, mouse_scroll, mouse_drag,
             mouse_move_relative, get_mouse_position
      Keyboard: type_text, press_key, hotkey
      App: open_app, close_app, list_windows, focus_window
      Browser: new_tab, close_tab, switch_tab, navigate, search
      Media: play_pause, skip, previous, volume, mute
      Master: activate_master_control, deactivate_master_control
      Screen: screenshot
    """
    action_map = {
        # Mouse
        "mouse_move": lambda: mouse_move(kwargs.get("x", 0), kwargs.get("y", 0),
                                          kwargs.get("duration", 0.3), kwargs.get("expected_window")),
        "mouse_move_relative": lambda: mouse_move_relative(
            kwargs.get("dx", 0), kwargs.get("dy", 0)),
        "mouse_click": lambda: mouse_click(
            kwargs.get("x"), kwargs.get("y"),
            kwargs.get("button", "left"), kwargs.get("clicks", 1), kwargs.get("expected_window")),
        "mouse_scroll": lambda: mouse_scroll(
            kwargs.get("amount", 3), kwargs.get("x"), kwargs.get("y")),
        "mouse_drag": lambda: mouse_drag(
            kwargs.get("start_x", 0), kwargs.get("start_y", 0),
            kwargs.get("end_x", 0), kwargs.get("end_y", 0)),
        "get_mouse_position": get_mouse_position,

        # Keyboard
        "type_text": lambda: type_text(kwargs.get("text", ""),
                                        kwargs.get("interval", 0.02), kwargs.get("expected_window")),
        "press_key": lambda: press_key(kwargs.get("key", "")),
        "hotkey": lambda: hotkey(*kwargs.get("keys", [])),

        # App
        "open_app": lambda: open_app(kwargs.get("app_name", "")),
        "close_app": lambda: close_app(kwargs.get("app_name", "")),
        "list_windows": list_windows,
        "focus_window": lambda: focus_window(kwargs.get("title", "")),

        # Browser
        "new_tab": lambda: browser_new_tab(kwargs.get("url")),
        "close_tab": browser_close_tab,
        "switch_tab": lambda: browser_switch_tab(kwargs.get("direction", "next")),
        "navigate": lambda: browser_navigate(kwargs.get("url", "")),
        "search": lambda: browser_search(kwargs.get("query", "")),

        # Media
        "play_pause": media_play_pause,
        "skip": media_skip,
        "previous": media_previous,
        "volume": lambda: media_volume(kwargs.get("direction", "up"),
                                        kwargs.get("amount", 5)),
        "mute": media_mute,

        # Master Control
        "activate_master_control": activate_master_control,
        "deactivate_master_control": deactivate_master_control,

        # Screen
        "screenshot": lambda: screenshot_region(
            kwargs.get("x", 0), kwargs.get("y", 0),
            kwargs.get("width", 0), kwargs.get("height", 0)),

        # AI Chat Interaction
        "ai_type_and_send": lambda: ai_type_and_send(
            kwargs.get("text", ""), kwargs.get("press_enter", True), kwargs.get("expected_window")),
        "ai_scroll_down": lambda: ai_scroll_down(kwargs.get("presses", 5)),
        "ai_scroll_up": lambda: ai_scroll_up(kwargs.get("presses", 5)),
        "ai_read_and_scroll_to_bottom": lambda: ai_read_and_scroll_to_bottom(
            kwargs.get("max_scrolls", 30), kwargs.get("scroll_pause", 1.5)),
        "ai_check_bottom": ai_check_bottom,
        "ai_full_interaction": lambda: ai_full_interaction(
            kwargs.get("text", ""), kwargs.get("max_scrolls", 30)),
    }

    handler = action_map.get(action)
    if not handler:
        return json.dumps({"error": f"Unknown action: {action}",
                          "available": list(action_map.keys())})
    try:
        return handler()
    except Exception as e:
        return json.dumps({"error": str(e), "action": action})
