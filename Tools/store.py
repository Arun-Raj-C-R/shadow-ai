"""
ingest_memory_logs.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Reads a narrative memory log (.txt), uses Gemma to intelligently chunk AND
compress it in a SINGLE API call, then stores everything into:
  1. ChromaDB vector database  (vector_memory/)
  2. memory.txt                (flat append log)
  3. memory_index.json         (structured index used by memory agent)

Rate limit budget (Gemini 3.1 Flash-Lite Preview free tier):
  - 15 RPM  (requests per minute)
  - 250K TPM (tokens per minute)
  - 500 RPD  (requests per day)

Optimization strategy:
  - ONE combined chunk+compress API call for the entire narrative
  - ONE optional metadata-tagging call (can be skipped with --no-tag)
  - All other processing is local (zero API calls)
  - Built-in RPD tracker so you never accidentally burn your daily quota

Usage:
    python ingest_memory_logs.py --file my_memory_log.txt
    python ingest_memory_logs.py --file my_memory_log.txt --dry-run
    python ingest_memory_logs.py --file my_memory_log.txt --no-tag
    python ingest_memory_logs.py --status        # show today's API usage
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIG
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# override=False keeps existing env vars; verbose=False silences parse warnings
load_dotenv(override=False, verbose=False)
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("âŒ  GEMINI_API_KEY not found in environment. Check your .env file.")
    sys.exit(1)

MODEL_NAME = "gemma-3-27b-it"

# Rate limits (free tier)
RPM_LIMIT      = 15
RPD_LIMIT      = 500
RPM_SAFETY     = 12                   # stay under limit with a buffer
INTER_CALL_DELAY = 60 / RPM_SAFETY   # ~5 seconds between calls

BASE_DIR           = pathlib.Path(__file__).parent
MEMORY_FILE_PATH   = BASE_DIR / "memory.txt"
MEMORY_INDEX_PATH  = BASE_DIR / "memory_index.json"
VECTOR_DB_PATH     = BASE_DIR / "vector_memory"
QUOTA_TRACKER_PATH = BASE_DIR / ".quota_tracker.json"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# QUOTA TRACKER  (persists across runs)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_quota() -> Dict:
    today = str(datetime.date.today())
    if QUOTA_TRACKER_PATH.exists():
        try:
            data = json.loads(QUOTA_TRACKER_PATH.read_text())
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "calls_today": 0, "last_call_ts": 0.0}


def _save_quota(q: Dict):
    QUOTA_TRACKER_PATH.write_text(json.dumps(q, indent=2))


def _check_quota(q: Dict, n_calls: int = 1) -> bool:
    """Returns True if we have enough RPD quota remaining."""
    if q["calls_today"] + n_calls > RPD_LIMIT:
        remaining = RPD_LIMIT - q["calls_today"]
        print(f"âŒ  RPD limit reached! {q['calls_today']}/{RPD_LIMIT} used today.")
        print(f"   Remaining: {remaining} | Needed: {n_calls}")
        return False
    return True


def _rpm_wait(q: Dict):
    """Enforce inter-call spacing to respect RPM limit."""
    elapsed = time.time() - q["last_call_ts"]
    if elapsed < INTER_CALL_DELAY:
        wait = INTER_CALL_DELAY - elapsed
        print(f"â³  RPM spacing: waiting {wait:.1f}s...")
        time.sleep(wait)


def print_quota_status():
    q = _load_quota()
    used      = q["calls_today"]
    remaining = RPD_LIMIT - used
    pct       = (used / RPD_LIMIT) * 100
    filled    = int(pct / 5)
    bar       = "â–ˆ" * filled + "â–‘" * (20 - filled)
    print(f"\nðŸ“Š  API Quota Status  [{q['date']}]")
    print(f"    Used today : {used} / {RPD_LIMIT}")
    print(f"    Remaining  : {remaining}")
    print(f"    [{bar}] {pct:.1f}%\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA STRUCTURES  (mirrors memory_agent.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MemoryType(Enum):
    FACT       = "fact"
    PROCEDURE  = "procedure"
    PREFERENCE = "preference"
    PROTOCOL   = "protocol"
    EXPERIENCE = "experience"
    INSIGHT    = "insight"


@dataclass
class MemoryEntry:
    id:              str
    content:         str
    memory_type:     MemoryType
    timestamp:       str
    importance:      float
    access_count:    int
    categories:      List[str]
    compressed_size: int
    metadata:        Dict[str, Any] = field(default_factory=dict)
    vector_id:       Optional[str]  = field(default=None)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS  (all local â€” zero API cost)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_id(content: str) -> str:
    # Stable hash based on content only â€” same text always gets the same ID.
    # This is the deduplication key: re-ingesting the same chunk is a no-op.
    return hashlib.md5(content.strip().encode()).hexdigest()[:12]


def _is_duplicate(chunk: str, memory_index: Dict) -> bool:
    """Return True if this exact chunk is already stored."""
    chunk_id = _make_id(chunk)
    return chunk_id in memory_index


def _heuristic_type(text: str) -> MemoryType:
    lower = text.lower()
    if any(k in lower for k in ["always", "never", "prefer", "like", "dislike", "want", "hate"]):
        return MemoryType.PREFERENCE
    if any(k in lower for k in ["step", "how to", "procedure", "process", "workflow"]):
        return MemoryType.PROCEDURE
    if any(k in lower for k in ["protocol", "rule", "policy", "must", "standard", "required"]):
        return MemoryType.PROTOCOL
    if any(k in lower for k in ["insight", "realized", "learned", "discovered", "now understand"]):
        return MemoryType.INSIGHT
    if any(k in lower for k in ["happened", "did", "went", "tried", "today", "yesterday", "last week"]):
        return MemoryType.EXPERIENCE
    return MemoryType.FACT


def _importance(t: MemoryType) -> float:
    return {
        MemoryType.PROTOCOL:   0.9,
        MemoryType.PREFERENCE: 0.75,
        MemoryType.INSIGHT:    0.7,
        MemoryType.EXPERIENCE: 0.6,
        MemoryType.PROCEDURE:  0.6,
        MemoryType.FACT:       0.5,
    }.get(t, 0.5)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LLM CALL WRAPPER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _call_llm(system: str, user: str, label: str, quota: Dict) -> str:
    """
    Fires one API call with:
    - Pre-flight quota check
    - RPM spacing enforcement
    - Exponential backoff on 429
    - Quota recording on success
    """
    if not _check_quota(quota, n_calls=1):
        return ""

    _rpm_wait(quota)

    for attempt in range(3):
        try:
            # Gemma models on Google's endpoint do NOT support the "system" role.
            # Merging system instruction directly into the user message instead.
            combined_user = f"{system}\n\n---\n\n{user}"
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": combined_user}
                ]
            )
            result = (resp.choices[0].message.content or "").strip()

            # Record call
            quota["calls_today"] += 1
            quota["last_call_ts"] = time.time()
            _save_quota(quota)
            remaining = RPD_LIMIT - quota["calls_today"]
            print(f"   ðŸ“¡  [{label}] Call #{quota['calls_today']} used | {remaining} remaining today")

            return result

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 20 * (2 ** attempt)   # 20s â†’ 40s â†’ 80s
                print(f"â³  [{label}] 429 rate limit â€” backing off {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"âŒ  [{label}] API error: {e}")
                return ""

    print(f"âŒ  [{label}] All 3 retries exhausted.")
    return ""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” CHUNK + COMPRESS  (1 API call total)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

COMBINED_SYSTEM = """You are an expert memory architect for an AI assistant called Alfred.

You will receive a long narrative memory log written in natural language.

Your job â€” do BOTH in one pass:
1. SPLIT the narrative into discrete, self-contained memory units.
2. COMPRESS each unit using Sparse Priming Representation (SPR).

Rules per memory unit:
- Represents ONE coherent piece of information (fact, experience, insight, preference, procedure, or protocol)
- Fully self-contained â€” no dangling pronouns, no missing context
- Compressed to 1â€“3 dense sentences maximum
- Must preserve: names, numbers, relationships, decisions, preferences

Output ONLY a valid JSON array of strings (the compressed units).
No preamble. No explanation. No markdown code fences. Raw JSON only.

Example:
["Arun prefers dark terminal themes; dislikes white backgrounds.", "Alfred backend: FastAPI + uvicorn on port 8000, bound to 0.0.0.0.", "Perovskite cells degrade under humidity â€” hermetic encapsulation is critical."]
"""

def chunk_and_compress(narrative: str, quota: Dict) -> List[str]:
    print("ðŸ§   [Step 1] Chunk + Compress â€” sending to Gemma (1 API call)...")
    raw = _call_llm(COMBINED_SYSTEM, narrative, label="Chunk+Compress", quota=quota)

    if not raw:
        print("âš ï¸   API call failed. Falling back to local paragraph split (0 API calls).")
        return _local_fallback(narrative)

    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    raw = raw.rstrip("```").strip()

    try:
        chunks = json.loads(raw)
        if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
            chunks = [c.strip() for c in chunks if c.strip()]
            print(f"âœ…  Got {len(chunks)} compressed memory units.")
            return chunks
        raise ValueError(f"Unexpected structure: {type(chunks)}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"âš ï¸   JSON parse failed ({e}). Using local paragraph fallback.")
        return _local_fallback(narrative)


def _local_fallback(text: str) -> List[str]:
    chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    print(f"âš ï¸   Local fallback: {len(chunks)} paragraph chunks (0 API calls used).")
    return chunks


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” TYPE TAGGING  (1 API call, optional)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TAG_SYSTEM = """You receive a JSON array of compressed memory chunks.
Classify each chunk with ONE label from:
  fact | procedure | preference | protocol | experience | insight

Return ONLY a JSON array of label strings in the SAME ORDER as the input.
No preamble. Raw JSON only.

Example input:  ["Arun prefers dark mode.", "Restart Alfred with: uvicorn main:app"]
Example output: ["preference", "procedure"]
"""

def ai_tag_types(chunks: List[str], quota: Dict) -> List[str]:
    print("ðŸ·ï¸   [Step 2] Type tagging â€” 1 API call for all chunks...")
    raw = _call_llm(TAG_SYSTEM, json.dumps(chunks), label="Tagger", quota=quota)

    if not raw:
        print("âš ï¸   Tagging failed. Using local heuristic (free).")
        return [_heuristic_type(c).value for c in chunks]

    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        tags = json.loads(raw)
        if isinstance(tags, list) and len(tags) == len(chunks):
            print(f"âœ…  Got {len(tags)} type tags.")
            return [t.lower().strip() for t in tags]
        raise ValueError(f"Length mismatch: got {len(tags)}, expected {len(chunks)}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"âš ï¸   Tag parsing failed ({e}). Using local heuristic.")
        return [_heuristic_type(c).value for c in chunks]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STORAGE LAYER  (all local, no API calls)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_vector_collection():
    try:
        import chromadb
        from chromadb.config import Settings
        chroma = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        return chroma.get_or_create_collection(
            name="shadow_memories",
            metadata={"description": "SHADOW long-term memory storage"}
        )
    except ImportError:
        print("âŒ  ChromaDB not installed. Run: pip install chromadb")
        return None
    except Exception as e:
        print(f"âŒ  ChromaDB init failed: {e}")
        return None


def store_to_vector(collection, entry: MemoryEntry) -> bool:
    if collection is None:
        return False
    try:
        collection.upsert(
            documents=[entry.content],
            metadatas=[{
                "memory_id":       entry.id,
                "type":            entry.memory_type.value,
                "timestamp":       entry.timestamp,
                "importance":      str(entry.importance),
                "content_preview": entry.content[:100]
            }],
            ids=[entry.id]
        )
        return True
    except Exception as e:
        print(f"âŒ  [Vector] {entry.id}: {e}")
        return False


def append_to_memory_txt(entry: MemoryEntry):
    line = f"[{entry.id}|{entry.memory_type.value}|{entry.timestamp}] {entry.content}"
    with open(MEMORY_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n---\n")


def load_memory_index() -> Dict:
    if not MEMORY_INDEX_PATH.exists():
        return {}
    try:
        with open(MEMORY_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"âš ï¸   Could not load memory index: {e}")
        return {}


def save_memory_index(index: Dict):
    with open(MEMORY_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def add_to_index(index: Dict, entry: MemoryEntry):
    d = asdict(entry)
    d["memory_type"] = entry.memory_type.value
    index[entry.id] = d


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ingest(log_file: pathlib.Path, dry_run: bool = False, no_tag: bool = False):
    print(f"\n{'â”€'*60}")
    print(f"  Alfred Memory Log Ingestion")
    print(f"  File     : {log_file}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Dry run  : {dry_run}")
    print(f"  Rate limit: {RPM_LIMIT} RPM | {RPD_LIMIT} RPD")
    print(f"{'â”€'*60}")

    quota = _load_quota()
    print_quota_status()

    # Estimate and check budget
    calls_needed = 1 + (0 if no_tag else 1)   # 1 or 2 calls max
    if not dry_run and not _check_quota(quota, calls_needed):
        sys.exit(1)
    print(f"ðŸ“‹  This run will use at most {calls_needed} API call(s).\n")

    # Read file
    if not log_file.exists():
        print(f"âŒ  File not found: {log_file}")
        sys.exit(1)
    narrative = log_file.read_text(encoding="utf-8").strip()
    if not narrative:
        print("âŒ  Memory log file is empty.")
        sys.exit(1)
    print(f"ðŸ“„  Read {len(narrative):,} characters from {log_file.name}\n")

    # â”€â”€ DRY RUN: preview locally, zero API calls â”€â”€
    if dry_run:
        print("ðŸ”  [DRY RUN] Using local fallback â€” no API calls consumed.\n")
        chunks = _local_fallback(narrative)
        types  = [_heuristic_type(c).value for c in chunks]
        print(f"   Would store {len(chunks)} memory unit(s):\n")
        for i, (c, t) in enumerate(zip(chunks, types), 1):
            imp = _importance(MemoryType(t))
            print(f"   [{i}] type={t} | importance={imp}")
            print(f"        {c[:110]}...\n")
        print(f"âœ…  Dry run complete. No data written. No API calls used.\n")
        return

    # â”€â”€ STEP 1: Chunk + Compress (1 API call) â”€â”€
    chunks = chunk_and_compress(narrative, quota)
    if not chunks:
        print("âŒ  No chunks produced. Aborting.")
        sys.exit(1)

    # â”€â”€ STEP 2: Type tagging (1 API call, optional) â”€â”€
    if no_tag:
        print("ðŸ·ï¸   [Local] Heuristic type classification (0 API calls).")
        type_strings = [_heuristic_type(c).value for c in chunks]
    else:
        type_strings = ai_tag_types(chunks, quota)

    print(f"\nðŸ“¦  Storing {len(chunks)} memory units...\n")

    collection   = get_vector_collection()
    memory_index = load_memory_index()

    if collection:
        print(f"âœ…  Vector DB ready â€” {collection.count()} existing memories\n")
    else:
        print("âš ï¸   No vector DB. Only flat files will be updated.\n")

    stored  = 0
    skipped = 0
    report  = []

    duplicates = 0
    for i, (chunk, type_str) in enumerate(zip(chunks, type_strings), 1):
        print(f"[{i}/{len(chunks)}] {chunk[:80]}...")

        # â”€â”€ Deduplication check â”€â”€
        if _is_duplicate(chunk, memory_index):
            print(f"  â­ï¸   Skipped (duplicate â€” already in index)\n")
            duplicates += 1
            continue

        try:
            mem_type = MemoryType(type_str.lower())
        except ValueError:
            mem_type = _heuristic_type(chunk)

        entry = MemoryEntry(
            id              = _make_id(chunk),
            content         = chunk,
            memory_type     = mem_type,
            timestamp       = datetime.datetime.now().isoformat(),
            importance      = _importance(mem_type),
            access_count    = 0,
            categories      = ["narrative_log", log_file.stem],
            compressed_size = len(chunk),
            metadata        = {
                "source_file":   str(log_file),
                "chunk_index":   i,
                "ingest_date":   str(datetime.date.today())
            },
            vector_id       = None
        )

        vec_ok = store_to_vector(collection, entry)
        if vec_ok:
            entry.vector_id = entry.id
            print(f"  âœ…  Vector  : {entry.id}")

        try:
            append_to_memory_txt(entry)
            add_to_index(memory_index, entry)
            print(f"  âœ…  Stored  : type={mem_type.value} | importance={entry.importance}\n")
            stored += 1
            report.append({
                "id":         entry.id,
                "type":       mem_type.value,
                "importance": entry.importance,
                "content":    chunk
            })
        except Exception as e:
            print(f"  âŒ  File sync failed: {e}\n")
            skipped += 1

    # Single index write at the end
    save_memory_index(memory_index)
    print(f"ðŸ’¾  memory_index.json saved ({len(memory_index)} total entries)")

    # Save report
    report_path = BASE_DIR / f"ingest_report_{datetime.date.today()}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print_quota_status()

    calls_used = quota["calls_today"] - _load_quota().get("calls_today", quota["calls_today"])
    print(f"{'â”€'*60}")
    print(f"  âœ…  Done!")
    print(f"  Stored     : {stored}")
    print(f"  Duplicates : {duplicates} (skipped)")
    print(f"  Errors     : {skipped}")
    print(f"  API calls  : {calls_needed} used this run")
    print(f"  Report     : {report_path.name}")
    print(f"{'â”€'*60}\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ENTRY POINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a narrative memory log into Alfred's memory system."
    )
    parser.add_argument(
        "--file", "-f",
        type=pathlib.Path,
        help="Path to the .txt narrative memory log file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be stored â€” no writes, no API calls."
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip AI type tagging, use local heuristic instead (saves 1 API call)."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show today's API quota usage and exit."
    )

    args = parser.parse_args()

    if args.status:
        print_quota_status()
        sys.exit(0)

    if not args.file:
        parser.error("--file is required unless using --status.")

    ingest(args.file, dry_run=args.dry_run, no_tag=args.no_tag)