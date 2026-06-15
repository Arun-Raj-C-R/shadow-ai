"""
face_recognition_tool.py - Real-time face recognition for Shadow AI camera feed.

Detects and recognizes faces in camera frames, draws green boxes with names,
and maintains a persistent face database (pickle-based, similar to memory vector DB).

Dependencies: face_recognition, dlib, opencv-python, numpy (all pre-installed)
"""

import os
# Limit CPU thread usage of dlib/numpy to avoid starving audio processing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import threading
import time
from datetime import datetime

import cv2
import numpy as np

# Attempt to import face_recognition (dlib-based)
try:
    import face_recognition as fr
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[FaceRec] WARNING: face_recognition library not available. Face recognition disabled.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
FACE_DB_FILE = os.path.join(KNOWN_FACES_DIR, "face_db.pkl")
RECOGNITION_TOLERANCE = 0.6       # Lower = stricter matching (0.4-0.6 range)
FACE_DETECTION_MODEL = "hog"      # 'hog' = fast CPU, 'cnn' = accurate GPU
MIN_FACE_SIZE = 30                # Minimum face size in pixels to detect
MAX_FACES_PER_FRAME = 5           # Limit faces per frame for performance

# ==============================================================================
# GLOBAL STATE (thread-safe)
# ==============================================================================

_face_db = {}                     # {name: [encoding1, encoding2, ...]}
_face_db_lock = threading.Lock()

_latest_camera_frame = None       # Updated by camera thread
_latest_frame_lock = threading.Lock()

_latest_screen_frame = None       # Updated by screen capture thread
_latest_screen_lock = threading.Lock()

_latest_annotated_frame = None    # Updated by face frame processor
_annotated_frame_lock = threading.Lock()

_face_names_cache = []            # Last detected face names (for AI context)
_face_cache_lock = threading.Lock()

_latest_detected_faces = []       # Recent detections: [{"name": str, "box": tuple, "encoding": np.ndarray, "frame": np.ndarray, "position": str}]
_detected_faces_lock = threading.Lock()

# Throttle: don't run face recognition on every single frame
_last_recognition_time = 0
_RECOGNITION_INTERVAL = 2.0       # Run face rec at most every 2 seconds
_cached_annotations = []          # Cache last recognition results for re-drawing

# Lazy warmup flag - dlib model loading is slow on first call
_dlib_warmed_up = False
_warmup_lock = threading.Lock()
_dlib_lock = threading.Lock()
_face_cascade = None
_cascade_lock = threading.Lock()

def _get_cascade_classifier():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    with _cascade_lock:
        if _face_cascade is None:
            try:
                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                _face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                print(f"[FaceRec] Error loading Haar Cascade XML: {e}")
    return _face_cascade


def _warmup_dlib():
    """Warm up dlib/face_recognition on first use to avoid slow first detection."""
    global _dlib_warmed_up
    if _dlib_warmed_up or not FACE_REC_AVAILABLE:
        return
    with _warmup_lock:
        if _dlib_warmed_up:
            return
        try:
            # Create a tiny dummy image and run detection to force model loading
            print("[FaceRec] Warming up face detection model (first-time load)...")
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            with _dlib_lock:
                fr.face_locations(dummy, model=FACE_DETECTION_MODEL)
            _dlib_warmed_up = True
            print("[FaceRec] [OK] Face detection model ready")
        except Exception as e:
            print(f"[FaceRec] Warmup error (non-fatal): {e}")
            _dlib_warmed_up = True  # Don't retry endlessly

# ==============================================================================
# DATABASE PERSISTENCE
# ==============================================================================

def _load_face_db() -> dict:
    """Load face database from pickle file."""
    try:
        if os.path.exists(FACE_DB_FILE):
            with open(FACE_DB_FILE, "rb") as f:
                db = pickle.load(f)
            if isinstance(db, dict):
                return db
    except Exception as e:
        print(f"[FaceRec] Error loading face DB: {e}")
    return {}


def _save_face_db():
    """Save current face database to pickle file. Thread-safe."""
    with _face_db_lock:
        try:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            tmp = FACE_DB_FILE + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(_face_db, f)
            if os.path.exists(FACE_DB_FILE):
                os.remove(FACE_DB_FILE)
            os.rename(tmp, FACE_DB_FILE)
        except Exception as e:
            print(f"[FaceRec] Error saving face DB: {e}")


# ==============================================================================
# CORE FACE OPERATIONS
# ==============================================================================

def _encode_faces(image_rgb: np.ndarray, locations=None, num_jitters=1):
    """Get face encoding(s) from an RGB image. Returns list of 128-dim encodings."""
    if not FACE_REC_AVAILABLE:
        return []
    try:
        with _dlib_lock:
            encodings = fr.face_encodings(image_rgb, known_face_locations=locations, num_jitters=num_jitters)
        return encodings
    except Exception as e:
        print(f"[FaceRec] Encoding error: {e}")
        return []


def register_face(name: str, image_bgr: np.ndarray, target_position: str = None) -> str:
    """
    Register a face from a BGR frame (camera or screen).
    Detects face, saves encoding to DB, saves crop image to disk.
    If target_position is provided, targets the face at that spatial location in a group picture.
    """
    global _face_db
    if not FACE_REC_AVAILABLE:
        return "Error: face_recognition library not available."
    if not name or not name.strip():
        return "Error: Name cannot be empty."

    name = name.strip()
    if name.lower() == "unknown":
        return "Error: Cannot register a face with the name 'Unknown'. Please provide the actual name of the person."

    try:
        _warmup_dlib()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with _dlib_lock:
            locations = fr.face_locations(image_rgb, model=FACE_DETECTION_MODEL)

        if not locations:
            return f"Error: No face detected in the frame. Make sure the person is clearly visible."

        h, w = image_bgr.shape[:2]
        face_loc = None

        if target_position:
            # Try to match a face whose spatial position matches the target
            target_pos_clean = target_position.lower().strip()
            
            # Helper to extract main directional keywords
            def get_dirs(s):
                return set(w for w in s.replace('-', ' ').split() if w in ("left", "right", "top", "bottom", "middle"))
                
            target_dirs = get_dirs(target_pos_clean)
            
            for loc in locations:
                pos = get_spatial_position(loc, w, h).lower()
                pos_dirs = get_dirs(pos)
                
                # Match if direct substring, or if directions are compatible subsets
                if (target_pos_clean in pos or pos in target_pos_clean) or \
                   (target_dirs and pos_dirs and (target_dirs.issubset(pos_dirs) or pos_dirs.issubset(target_dirs))):
                    face_loc = loc
                    break
                    
            if not face_loc:
                if len(locations) == 1:
                    face_loc = locations[0]
                else:
                    available_positions = [get_spatial_position(loc, w, h) for loc in locations]
                    return f"Error: No face found at position '{target_position}'. Available positions: {available_positions}"
        else:
            # Use the largest face (closest to camera)
            if len(locations) > 1:
                # Sort by face area (largest first)
                locations = sorted(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]), reverse=True)
            face_loc = locations[0]

        encodings = _encode_faces(image_rgb, [face_loc])

        if not encodings:
            return f"Error: Could not encode face. Try again with better lighting."

        encoding = encodings[0]

        # Add to database
        with _face_db_lock:
            if name not in _face_db:
                _face_db[name] = []
            _face_db[name].append(encoding)
            count = len(_face_db[name])

        # Save the face crop image
        top, right, bottom, left = face_loc
        # Add margin around the face
        margin = 30
        h, w = image_bgr.shape[:2]
        top_m = max(0, top - margin)
        right_m = min(w, right + margin)
        bottom_m = min(h, bottom + margin)
        left_m = max(0, left - margin)
        face_crop = image_bgr[top_m:bottom_m, left_m:right_m]

        # Save to person's folder
        person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_path = os.path.join(person_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_crop)

        # Persist database
        _save_face_db()

        print(f"[FaceRec] [OK] Registered '{name}' ({count} images total)")
        return f"Successfully registered face for '{name}'. Now have {count} reference image(s). Recognition will improve with more images."

    except Exception as e:
        import traceback
        print(f"[FaceRec] Register error: {e}")
        traceback.print_exc()
        return f"Error registering face: {e}"


def register_face_from_camera(name: str, position: str = None) -> str:
    """
    Register a face using the latest camera frame.
    Called by the AI's function calling tool.
    """
    with _latest_frame_lock:
        frame = _latest_camera_frame

    if frame is None:
        return "Error: No camera frame available. Make sure the camera shutter is open."

    return register_face(name, frame.copy(), target_position=position)


def set_latest_frame(frame_bgr: np.ndarray):
    """Thread-safe setter for the latest camera frame. Called from camera thread."""
    global _latest_camera_frame
    with _latest_frame_lock:
        _latest_camera_frame = frame_bgr


def register_face_from_screen(name: str, position: str = None) -> str:
    """
    Register a face using the latest screen frame.
    Called by the AI's function calling tool.
    """
    with _latest_screen_lock:
        frame = _latest_screen_frame

    if frame is None:
        return "Error: No screen frame available yet."

    return register_face(name, frame.copy(), target_position=position)


def set_latest_screen_frame(frame_bgr: np.ndarray):
    """Thread-safe setter for the latest screen frame. Called from screen capture thread."""
    global _latest_screen_frame
    with _latest_screen_lock:
        _latest_screen_frame = frame_bgr


def set_latest_annotated_frame(frame_bgr: np.ndarray):
    """Thread-safe setter for the latest annotated camera frame."""
    global _latest_annotated_frame
    with _annotated_frame_lock:
        _latest_annotated_frame = frame_bgr


def get_latest_annotated_frame() -> np.ndarray:
    """Thread-safe getter for the latest annotated camera frame."""
    with _annotated_frame_lock:
        return _latest_annotated_frame


# ==============================================================================
# SPATIAL MAPPING & IMAGE PREPROCESSING (Advanced Face Recognition)
# ==============================================================================

def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the Y (luminance) channel to balance harsh lighting and enhance facial details.
    """
    try:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        ycrcb = cv2.merge(channels)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    except Exception as e:
        print(f"[FaceRec] Preprocessing error: {e}")
        return image_bgr


def get_spatial_position(box: tuple, width: int, height: int) -> str:
    """
    Classify the center coordinates of a bounding box into detailed spatial regions:
    middle, left side, right side, top area, bottom area, or the four corners.
    """
    top, right, bottom, left = box
    cx = (left + right) / 2
    cy = (top + bottom) / 2

    # Boundaries dividing the frame into grid sections (35% / 65%)
    left_bound = width * 0.35
    right_bound = width * 0.65
    top_bound = height * 0.35
    bottom_bound = height * 0.65

    if cx < left_bound and cy < top_bound:
        return "top-left corner"
    elif cx > right_bound and cy < top_bound:
        return "top-right corner"
    elif cx < left_bound and cy > bottom_bound:
        return "bottom-left corner"
    elif cx > right_bound and cy > bottom_bound:
        return "bottom-right corner"
    elif cx < left_bound:
        return "left side"
    elif cx > right_bound:
        return "right side"
    elif cy < top_bound:
        return "top area"
    elif cy > bottom_bound:
        return "bottom area"
    else:
        return "middle"


# ==============================================================================
# FACE SPATIAL HISTORY QUEUE
# ==============================================================================

_face_history_queue = []          # List of (timestamp, [{"name": str, "position": str}])
_history_lock = threading.Lock()

def add_to_face_history(detected_faces: list):
    """
    Add current detected faces with their spatial positions to the history queue
    and prune entries older than 60 seconds.
    """
    import time
    now = time.time()
    
    # Extract only name and position for history context
    history_entries = []
    for face in detected_faces:
        history_entries.append({
            "name": face["name"],
            "position": face["position"]
        })
        
    with _history_lock:
        _face_history_queue.append((now, history_entries))
        # Keep only the last 60 seconds
        cutoff = now - 60.0
        while _face_history_queue and _face_history_queue[0][0] < cutoff:
            _face_history_queue.pop(0)


def get_face_history_context() -> str:
    """
    Format the last 60 seconds of face spatial history as a reverse-chronological
    ordered list (most recent first) for the AI context window.
    """
    import time
    now = time.time()
    
    with _history_lock:
        # Prune stale entries
        cutoff = now - 60.0
        while _face_history_queue and _face_history_queue[0][0] < cutoff:
            _face_history_queue.pop(0)
            
        if not _face_history_queue:
            return ""
            
        lines = []
        # Reverse to prioritize last seen data (newest/most recent first)
        for ts, faces in reversed(_face_history_queue):
            elapsed = int(now - ts)
            # Round to nearest multiple of 5 seconds to look cleaner
            elapsed_rounded = max(0, ((elapsed + 2) // 5) * 5)
            
            if not faces:
                desc = "No faces in view"
            else:
                desc = ", ".join([
                    f"{f['name']} on the {f['position']}" 
                    if "corner" in f['position'] or "side" in f['position'] or "area" in f['position'] 
                    else f"{f['name']} in the {f['position']}" 
                    for f in faces
                ])
                
            lines.append(f"  - {elapsed_rounded}s ago: {desc}")
            
        return "\n".join(lines)


# ==============================================================================
# FACE IDENTIFICATION
# ==============================================================================

def identify_faces(frame_bgr: np.ndarray, force: bool = False) -> list:
    """
    Detect and recognize all faces in a BGR frame.
    Returns list of dicts: {"name": str, "box": (top, right, bottom, left), "confidence": float, "position": str}
    """
    global _face_names_cache, _last_recognition_time, _cached_annotations

    if not FACE_REC_AVAILABLE:
        return []

    # Throttle: reuse cached results if called too frequently (unless forced)
    now = time.time()
    if not force:
        if now - _last_recognition_time < _RECOGNITION_INTERVAL:
            return _cached_annotations

    try:
        _warmup_dlib()
        
        # Preprocess frame to balance lighting and enhance details
        processed_frame = preprocess_image(frame_bgr)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using dlib HOG directly on the preprocessed RGB frame (extremely accurate)
        with _dlib_lock:
            locations = fr.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        
        if len(locations) == 0:
            if not force:
                _last_recognition_time = now
                _cached_annotations = []
                with _face_cache_lock:
                    _face_names_cache = []
            with _detected_faces_lock:
                _latest_detected_faces.clear()
            return []

        # Limit faces for performance
        locations = locations[:MAX_FACES_PER_FRAME]

        # Get encodings on full-resolution preprocessed RGB frame (extremely fast and precise)
        encodings = _encode_faces(rgb_frame, locations, num_jitters=1)

        results = []
        detected_faces = []
        names_in_view = []

        h, w = frame_bgr.shape[:2]

        with _face_db_lock:
            # Group reference encodings by name to run K-NN scoring
            person_db = {name: encs for name, encs in _face_db.items() if encs}

        for i, encoding in enumerate(encodings):
            top, right, bottom, left = locations[i]

            name = "Unknown"
            confidence = 0.0
            best_person = None
            best_score = 999.0
            best_min_dist = 999.0

            for person_name, enc_list in person_db.items():
                distances = fr.face_distance(enc_list, encoding)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    avg_dist = np.mean(distances)
                    # Advanced Weighted K-NN Score: 60% closest single match, 40% average group match
                    score = (min_dist * 0.6) + (avg_dist * 0.4)
                    
                    if score < best_score:
                        best_score = score
                        best_person = person_name
                        best_min_dist = min_dist

            # Strict threshold of 0.45 (confidence >= 0.55) to avoid misidentifying people
            if best_person and best_min_dist <= 0.45:
                name = best_person
                confidence = round(1.0 - best_min_dist, 2)

            position = get_spatial_position((top, right, bottom, left), w, h)

            # Public result (no raw frame or encoding)
            results.append({
                "name": name,
                "box": (top, right, bottom, left),
                "confidence": confidence,
                "position": position
            })
            names_in_view.append(name)

            # Cache details for active learning/confirmation
            detected_faces.append({
                "name": name,
                "box": (top, right, bottom, left),
                "confidence": confidence,
                "position": position,
                "encoding": encoding,
                "frame": frame_bgr.copy()
            })

        # Cache detections for confirm_face tool action
        with _detected_faces_lock:
            _latest_detected_faces.clear()
            _latest_detected_faces.extend(detected_faces)

        if not force:
            _last_recognition_time = now
            _cached_annotations = results
            with _face_cache_lock:
                _face_names_cache = names_in_view

        return results

    except Exception as e:
        print(f"[FaceRec] Identification error: {e}")
        return _cached_annotations  # Return cached on error


# ==============================================================================
# FRAME ANNOTATION (draws boxes + names on camera frames)
# ==============================================================================

def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Draw face recognition results on a BGR frame.
    - Known faces: GREEN box + name
    - Unknown faces: YELLOW box + "Unknown"
    Returns the annotated frame.
    """
    if not FACE_REC_AVAILABLE:
        return frame_bgr

    try:
        faces = identify_faces(frame_bgr)

        for face in faces:
            name = face["name"]
            top, right, bottom, left = face["box"]
            confidence = face["confidence"]

            if name == "Unknown":
                # Yellow for unknown
                box_color = (0, 255, 255)
                text_bg_color = (0, 200, 200)
            else:
                # Green for recognized
                box_color = (0, 255, 0)
                text_bg_color = (0, 200, 0)

            # Draw face rectangle
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), box_color, 2)

            # Build label text (name only, per user request)
            label = name

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_top = bottom + 5
            label_bottom = label_top + text_size[1] + 10
            cv2.rectangle(frame_bgr, (left, label_top), (left + text_size[0] + 10, label_bottom), text_bg_color, -1)

            # Draw label text (white on colored background)
            cv2.putText(frame_bgr, label, (left + 5, label_bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_bgr

    except Exception as e:
        # Never crash the camera pipeline
        return frame_bgr


# ==============================================================================
# LIGHTWEIGHT FRAME ANNOTATION (draws cached results, NO detection)
# ==============================================================================

def draw_cached_annotations(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Draw CACHED face recognition results on a BGR frame.
    Does NOT run any face detection - only draws boxes/names from previous results.
    This is extremely fast (~1ms) and safe to call from any thread.
    """
    try:
        cached = _cached_annotations
        if not cached:
            return frame_bgr

        for face in cached:
            name = face.get("name", "Unknown")
            box = face.get("box")
            if not box:
                continue
            top, right, bottom, left = box

            if name == "Unknown":
                box_color = (0, 255, 255)
                text_bg_color = (0, 200, 200)
            else:
                box_color = (0, 255, 0)
                text_bg_color = (0, 200, 0)

            # Draw face rectangle
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), box_color, 2)

            # Build label
            label = name

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_top = bottom + 5
            label_bottom = label_top + text_size[1] + 10
            cv2.rectangle(frame_bgr, (left, label_top), (left + text_size[0] + 10, label_bottom), text_bg_color, -1)

            # Draw label text (white on colored background)
            cv2.putText(frame_bgr, label, (left + 5, label_bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception:
        pass  # Never crash the camera pipeline

    return frame_bgr


# ==============================================================================
# SAFE FACE IDENTIFICATION (for use from separate thread)
# ==============================================================================

def identify_faces_safe() -> list:
    """
    Thread-safe face identification using the latest stored camera frame.
    Called from code5.py's face_scanner task via asyncio.to_thread().
    This keeps dlib completely isolated from the camera/OpenCV thread.
    Returns list of dicts: {"name": str, "confidence": float}
    """
    with _latest_frame_lock:
        frame = _latest_camera_frame

    if frame is None:
        return []

    try:
        return identify_faces(frame.copy())
    except Exception as e:
        print(f"[FaceRec] Safe identification error: {e}")
        return []


def identify_faces_on_screen_safe() -> list:
    """
    Thread-safe face identification using the latest stored screen frame.
    Called from code5.py via asyncio.to_thread() on user request.
    Returns list of dicts: {"name": str, "box": (top, right, bottom, left), "confidence": float, "position": str}
    """
    with _latest_screen_lock:
        frame = _latest_screen_frame

    if frame is None:
        return []

    try:
        return identify_faces(frame.copy(), force=True)
    except Exception as e:
        print(f"[FaceRec] Safe screen identification error: {e}")
        return []


# ==============================================================================
# DATABASE MANAGEMENT (for function calling)
# ==============================================================================

def get_known_faces() -> str:
    """List all known faces with image counts."""
    with _face_db_lock:
        if not _face_db:
            return "No faces registered yet. When you see an 'Unknown' face in the camera, ask the person their name and use register_face to save it."

        lines = ["[DB] Known Faces Database:"]
        for name, encodings in _face_db.items():
            person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_"))
            img_count = 0
            if os.path.isdir(person_dir):
                img_count = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
            lines.append(f"  - {name}: {len(encodings)} encodings, {img_count} images")

        lines.append(f"\nTotal: {len(_face_db)} people registered")
        return "\n".join(lines)


def delete_known_face(name: str) -> str:
    """Remove a person from the face database."""
    global _face_db

    if not name or not name.strip():
        return "Error: Name cannot be empty."

    name = name.strip()

    with _face_db_lock:
        if name not in _face_db:
            available = list(_face_db.keys())
            return f"Error: '{name}' not found in database. Known faces: {available}"

        del _face_db[name]

    _save_face_db()

    # Optionally remove images
    person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_"))
    removed_images = 0
    if os.path.isdir(person_dir):
        import shutil
        try:
            removed_images = len(os.listdir(person_dir))
            shutil.rmtree(person_dir)
        except Exception as e:
            print(f"[FaceRec] Error removing image folder: {e}")

    print(f"[FaceRec] [DELETED] Deleted '{name}' ({removed_images} images removed)")
    return f"Successfully deleted '{name}' from face database. Removed {removed_images} images."


def rename_known_face(old_name: str, new_name: str) -> str:
    """
    Rename a registered person in the face recognition database.
    This changes their name key in the pickle DB and renames their training crop image folder on disk.
    If the new name already exists, it merges their encodings and folder contents.
    """
    global _face_db
    if not old_name or not old_name.strip() or not new_name or not new_name.strip():
        return "Error: Both current name and new name must be provided."

    old_name = old_name.strip()
    new_name = new_name.strip()

    if old_name.lower() == new_name.lower():
        return f"Current name and new name are already the same: '{old_name}'"

    if new_name.lower() == "unknown":
        return "Error: Cannot rename a person to 'Unknown'."

    # Find the matching registered key (case-insensitive lookup to be user-friendly)
    actual_old_name = None
    with _face_db_lock:
        for k in _face_db.keys():
            if k.lower() == old_name.lower():
                actual_old_name = k
                break

    if not actual_old_name:
        with _face_db_lock:
            known = list(_face_db.keys())
        return f"Error: Person '{old_name}' not found in the face database. Registered names: {known}"

    # Find if the new name is already registered (case-insensitive)
    actual_new_name = None
    with _face_db_lock:
        for k in _face_db.keys():
            if k.lower() == new_name.lower():
                actual_new_name = k
                break

    with _face_db_lock:
        encodings_to_move = _face_db[actual_old_name]
        
        # If the target name already exists, we merge
        if actual_new_name:
            _face_db[actual_new_name].extend(encodings_to_move)
            del _face_db[actual_old_name]
            merge_mode = True
            resolved_new_name = actual_new_name
        else:
            _face_db[new_name] = encodings_to_move
            del _face_db[actual_old_name]
            merge_mode = False
            resolved_new_name = new_name

    _save_face_db()

    # Move/Rename directories on disk
    old_dir = os.path.join(KNOWN_FACES_DIR, actual_old_name.replace(" ", "_"))
    new_dir = os.path.join(KNOWN_FACES_DIR, resolved_new_name.replace(" ", "_"))
    
    moved_count = 0
    dir_msg = ""
    
    if os.path.exists(old_dir):
        if merge_mode:
            # Move files one by one to merge
            os.makedirs(new_dir, exist_ok=True)
            import shutil
            for filename in os.listdir(old_dir):
                src_file = os.path.join(old_dir, filename)
                if os.path.isfile(src_file):
                    # To prevent filename collision, we prepend timestamp or unique suffix if it exists
                    dest_file = os.path.join(new_dir, filename)
                    if os.path.exists(dest_file):
                        base, ext = os.path.splitext(filename)
                        dest_file = os.path.join(new_dir, f"{base}_merged_{int(time.time())}{ext}")
                    shutil.move(src_file, dest_file)
                    moved_count += 1
            try:
                os.rmdir(old_dir)
            except Exception as e:
                print(f"[FaceRec] Could not remove old directory: {e}")
            dir_msg = f"Merged {moved_count} image files from '{actual_old_name}' folder into '{resolved_new_name}' folder."
        else:
            # Simple directory rename
            try:
                if os.path.exists(new_dir):
                    # Just in case the folder existed but was not in the database
                    import shutil
                    for filename in os.listdir(old_dir):
                        src_file = os.path.join(old_dir, filename)
                        if os.path.isfile(src_file):
                            dest_file = os.path.join(new_dir, filename)
                            shutil.move(src_file, dest_file)
                    shutil.rmtree(old_dir)
                else:
                    os.rename(old_dir, new_dir)
                dir_msg = f"Renamed folder '{actual_old_name}' to '{resolved_new_name}'."
            except Exception as e:
                dir_msg = f"Renamed DB entry but folder rename encountered an issue: {e}"
    else:
        dir_msg = "No image folder found to rename."

    print(f"[FaceRec] Renamed '{actual_old_name}' to '{resolved_new_name}' ({dir_msg})")
    
    # Also update _latest_detected_faces in-place so that subsequent confirm_face etc. know the new name
    with _detected_faces_lock:
        for face in _latest_detected_faces:
            if face["name"].lower() == actual_old_name.lower():
                face["name"] = resolved_new_name

    if merge_mode:
        return f"Successfully merged '{actual_old_name}' into existing person '{resolved_new_name}'. {dir_msg}"
    else:
        return f"Successfully renamed '{actual_old_name}' to '{resolved_new_name}'. {dir_msg}"


def confirm_face(name: str, position: str = None) -> str:
    """
    Active learning feature: confirm or correct a recently detected face's identity.
    Locates the matching face in _latest_detected_faces, crops and saves it under the correct name,
    and adds its 128D encoding to the database.
    """
    global _face_db
    if not FACE_REC_AVAILABLE:
        return "Error: face_recognition library not available."
    if not name or not name.strip():
        return "Error: Name cannot be empty."

    name = name.strip()
    if name.lower() == "unknown":
        return "Error: Cannot register a face with the name 'Unknown'. Please provide the actual name."

    with _detected_faces_lock:
        if not _latest_detected_faces:
            return "Error: No faces have been detected recently. Make sure someone is in view of the camera or screen."

        target_face = None
        
        # 1. If position is specified, match by position
        if position:
            pos_clean = position.lower().strip()
            for face in _latest_detected_faces:
                if pos_clean in face["position"].lower():
                    target_face = face
                    break
            if not target_face:
                available_positions = [f["position"] for f in _latest_detected_faces]
                return f"Error: No face found at position '{position}'. Available positions: {available_positions}"
        # 2. If no position is specified but there is only 1 face detected, use it
        elif len(_latest_detected_faces) == 1:
            target_face = _latest_detected_faces[0]
        # 3. Otherwise, try to find a face matching the name or default to the first one
        else:
            for face in _latest_detected_faces:
                if face["name"].lower() == name.lower():
                    target_face = face
                    break
            if not target_face:
                target_face = _latest_detected_faces[0]

        # Extract data from target face
        frame = target_face["frame"]
        box = target_face["box"]
        encoding = target_face["encoding"]

    try:
        # Save face crop image
        top, right, bottom, left = box
        margin = 30
        h, w = frame.shape[:2]
        top_m = max(0, top - margin)
        right_m = min(w, right + margin)
        bottom_m = min(h, bottom + margin)
        left_m = max(0, left - margin)
        face_crop = frame[top_m:bottom_m, left_m:right_m]

        person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_path = os.path.join(person_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_crop)

        # Append encoding to DB
        with _face_db_lock:
            if name not in _face_db:
                _face_db[name] = []
            _face_db[name].append(encoding)
            count = len(_face_db[name])

        _save_face_db()
        print(f"[FaceRec] Active Learning confirmation: saved '{name}' crop and encoding ({count} total)")
        return f"Successfully confirmed/saved face for '{name}' at position '{target_face['position']}'. Now have {count} training images."

    except Exception as e:
        print(f"[FaceRec] Confirmation error: {e}")
        return f"Error confirming face: {e}"


def get_face_names_in_view() -> list:
    """Return the cached list of face names currently visible."""
    with _face_cache_lock:
        return list(_face_names_cache)


# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
_face_db = _load_face_db()

if FACE_REC_AVAILABLE:
    known_count = len(_face_db)
    known_names = list(_face_db.keys())
    if known_count > 0:
        print(f"[FaceRec] [OK] Loaded {known_count} known face(s): {known_names}")
    else:
        print(f"[FaceRec] [OK] Initialized (no faces registered yet)")
else:
    print(f"[FaceRec] [WARNING] Running without face recognition (library not available)")
