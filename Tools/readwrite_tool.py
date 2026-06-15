# readwrite_tool.py
"""
Read & Write RAG Tool for SHADOW/Shadow AI
==========================================
Monitors a folder for documents (PDF, DOCX, TXT, MD, CSV, etc.),
indexes them into a local ChromaDB vector store using sentence-transformers,
and provides RAG-based query/answer + document writing capabilities.

Dependencies: pip install chromadb sentence-transformers pypdf python-docx
"""

import os

# Suppress TensorFlow logging and oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger("readwrite_tool")

# â”€â”€ Paths â”€â”€
BASE_DIR = os.path.join(os.path.dirname(__file__), "Read&Write")
WRITE_DIR = os.path.join(BASE_DIR, "Write")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "readwrite_vectordb")
INDEX_FILE = os.path.join(VECTOR_DIR, "file_index.json")

# Ensure dirs exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(WRITE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# â”€â”€ Supported extensions â”€â”€
SUPPORTED_EXT = {".pdf", ".txt", ".md", ".csv", ".docx", ".doc", ".json", ".py", ".log"}

# â”€â”€ Lazy-loaded globals â”€â”€
_chroma_client = None
_collection = None
_embedding_fn = None
_file_index = {}  # {filepath: {"hash": ..., "chunks": int, "indexed_at": ...}}


# ======================================================================
# FILE INDEX â€” track what's already indexed
# ======================================================================

def _load_file_index() -> dict:
    global _file_index
    try:
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                _file_index = json.load(f)
    except Exception:
        _file_index = {}
    return _file_index


def _save_file_index():
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(_file_index, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save file index: {e}")


def _file_hash(filepath: str) -> str:
    """Quick hash of file to detect changes."""
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


# ======================================================================
# DOCUMENT READERS
# ======================================================================

def _read_pdf(filepath: str) -> str:
    """Read text from PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(filepath)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            return "[ERROR] pypdf or PyPDF2 not installed. Run: pip install pypdf"
    except Exception as e:
        return f"[ERROR reading PDF] {e}"


def _read_docx(filepath: str) -> str:
    """Read text from DOCX."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        return "[ERROR] python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        return f"[ERROR reading DOCX] {e}"


def _read_text(filepath: str) -> str:
    """Read plain text files."""
    try:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return "[ERROR] Could not decode file with any encoding"
    except Exception as e:
        return f"[ERROR reading file] {e}"


def read_document(filepath: str) -> str:
    """Read any supported document and return its text content."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return _read_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return _read_docx(filepath)
    elif ext in (".txt", ".md", ".csv", ".json", ".py", ".log"):
        return _read_text(filepath)
    else:
        return f"[ERROR] Unsupported file type: {ext}"


# ======================================================================
# CHUNKING
# ======================================================================

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if not text or len(text.strip()) < 10:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ======================================================================
# VECTOR STORE â€” ChromaDB + sentence-transformers
# ======================================================================

def _get_embedding_fn():
    """Lazy-load sentence-transformers embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        try:
            from chromadb.utils import embedding_functions
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.error(f"Could not load embedding function: {e}")
            raise
    return _embedding_fn


def _get_collection():
    """Lazy-load ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb
            _chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
            _collection = _chroma_client.get_or_create_collection(
                name="readwrite_docs",
                embedding_function=_get_embedding_fn(),
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Could not init ChromaDB: {e}")
            raise
    return _collection


# ======================================================================
# INDEXING
# ======================================================================

def index_document(filepath: str) -> str:
    """Index a single document into the vector store."""
    _load_file_index()

    if not os.path.exists(filepath):
        return json.dumps({"error": f"File not found: {filepath}"})

    # Check if already indexed with same hash
    fhash = _file_hash(filepath)
    if filepath in _file_index and _file_index[filepath].get("hash") == fhash:
        return json.dumps({
            "status": "already_indexed",
            "file": os.path.basename(filepath),
            "chunks": _file_index[filepath].get("chunks", 0)
        })

    # Read and chunk
    text = read_document(filepath)
    if text.startswith("[ERROR"):
        return json.dumps({"error": text})

    chunks = _chunk_text(text)
    if not chunks:
        return json.dumps({"error": "No extractable text found in document"})

    # Remove old entries if re-indexing
    collection = _get_collection()
    try:
        old_ids = [f"{filepath}__chunk_{i}" for i in range(500)]
        collection.delete(ids=old_ids)
    except Exception:
        pass

    # Add chunks
    ids = [f"{filepath}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{
        "source": filepath,
        "filename": os.path.basename(filepath),
        "chunk_index": i,
        "total_chunks": len(chunks)
    } for i in range(len(chunks))]

    # Batch insert (ChromaDB handles batching)
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end]
        )

    # Update index
    _file_index[filepath] = {
        "hash": fhash,
        "chunks": len(chunks),
        "indexed_at": datetime.now().isoformat(),
        "filename": os.path.basename(filepath)
    }
    _save_file_index()

    return json.dumps({
        "status": "indexed",
        "file": os.path.basename(filepath),
        "chunks": len(chunks),
        "text_length": len(text)
    })


def index_all_documents() -> str:
    """Scan Read&Write folder and index all supported documents."""
    results = []
    for fname in os.listdir(BASE_DIR):
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXT:
                result = index_document(fpath)
                results.append(json.loads(result))

    return json.dumps({
        "status": "scan_complete",
        "files_processed": len(results),
        "details": results
    })


# ======================================================================
# QUERY (RAG)
# ======================================================================

def query_documents(query: str, n_results: int = 5, filename_filter: str = None) -> str:
    """
    Query the indexed documents using semantic search.
    Returns relevant chunks with source info.
    """
    try:
        collection = _get_collection()

        where_filter = None
        if filename_filter:
            where_filter = {"filename": filename_filter}

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        if not results["documents"] or not results["documents"][0]:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant content found. Make sure documents are indexed first.",
                "query": query
            })

        output_chunks = []
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0], results["metadatas"][0]
        )):
            output_chunks.append({
                "rank": i + 1,
                "source": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "content": doc[:1500],  # Cap content length
            })

        return json.dumps({
            "status": "success",
            "query": query,
            "results": output_chunks
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# FILE LISTING
# ======================================================================

def list_read_files() -> str:
    """List all files in the Read&Write folder."""
    files = []
    for fname in os.listdir(BASE_DIR):
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            size_kb = os.path.getsize(fpath) / 1024
            indexed = fpath in _file_index
            files.append({
                "name": fname,
                "extension": ext,
                "size_kb": round(size_kb, 1),
                "supported": ext in SUPPORTED_EXT,
                "indexed": indexed
            })

    # Also list Write folder contents
    write_files = []
    if os.path.exists(WRITE_DIR):
        for fname in os.listdir(WRITE_DIR):
            fpath = os.path.join(WRITE_DIR, fname)
            if os.path.isfile(fpath):
                write_files.append({
                    "name": fname,
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1)
                })

    return json.dumps({
        "read_folder": BASE_DIR,
        "write_folder": WRITE_DIR,
        "read_files": files,
        "write_files": write_files,
        "total_read": len(files),
        "total_write": len(write_files)
    })


# ======================================================================
# WRITE â€” create or modify files in Write folder
# ======================================================================

def write_document(filename: str, content: str, mode: str = "create") -> str:
    """
    Write content to a file in the Write folder.
    mode: 'create' (overwrite), 'append', 'modify' (same as create)
    """
    try:
        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            safe_name = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = os.path.join(WRITE_DIR, safe_name)

        if mode == "append" and os.path.exists(filepath):
            with open(filepath, "a", encoding="utf-8") as f:
                f.write("\n\n" + content)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        return json.dumps({
            "status": "success",
            "action": "write_document",
            "file": safe_name,
            "path": filepath,
            "mode": mode,
            "size_kb": round(os.path.getsize(filepath) / 1024, 1)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def read_write_file(filename: str) -> str:
    """Read a file from the Write folder."""
    try:
        filepath = os.path.join(WRITE_DIR, filename)
        if not os.path.exists(filepath):
            return json.dumps({"error": f"File not found: {filename}"})
        content = _read_text(filepath)
        return json.dumps({
            "status": "success",
            "file": filename,
            "content": content[:5000]  # Cap
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# UNIFIED ENTRY POINT
# ======================================================================

import difflib

def _resolve_filename(filename: str, folder: str) -> str:
    """Fuzzy match filename in the given folder to handle AI voice transcription typos."""
    if not filename:
        return ""
    if os.path.exists(os.path.join(folder, filename)):
        return filename
        
    try:
        if not os.path.exists(folder):
            return filename
        files = os.listdir(folder)
        def clean(n):
            return os.path.splitext(n)[0].lower().replace("-", " ").replace("_", " ")
        
        target = clean(filename)
        best_match = None
        best_score = 0
        
        for f in files:
            score = difflib.SequenceMatcher(None, target, clean(f)).ratio()
            if score > best_score:
                best_score = score
                best_match = f
                
        if best_score > 0.5:
            return best_match
    except Exception:
        pass
    return filename

def readwrite_tool(action: str, **kwargs) -> str:
    """
    Unified Read&Write tool entry point.

    Actions:
      list_files     â€” List files in Read&Write and Write folders
      index_file     â€” Index a specific file (filename required)
      index_all      â€” Index all supported files in Read&Write folder
      query          â€” RAG query across indexed documents (query required)
      read_file      â€” Read raw content of a file (filename required)
      write_file     â€” Write/create a file in Write folder (filename, content required)
      append_file    â€” Append to existing file in Write folder
      read_write_file â€” Read a file from the Write folder
    """
    req_filename = kwargs.get("filename", "")
    if req_filename:
        if action in ["index_file", "read_file", "query"]:
            kwargs["filename"] = _resolve_filename(req_filename, BASE_DIR)
        elif action in ["read_write_file", "append_file"]:
            kwargs["filename"] = _resolve_filename(req_filename, WRITE_DIR)

    action_map = {
        "list_files": lambda: list_read_files(),

        "index_file": lambda: index_document(
            os.path.join(BASE_DIR, kwargs.get("filename", ""))
        ),

        "index_all": lambda: index_all_documents(),

        "query": lambda: query_documents(
            kwargs.get("query", ""),
            kwargs.get("n_results", 5),
            kwargs.get("filename", None)
        ),

        "read_file": lambda: json.dumps({
            "status": "success",
            "file": kwargs.get("filename", ""),
            "content": read_document(
                os.path.join(BASE_DIR, kwargs.get("filename", ""))
            )[:5000]
        }),

        "write_file": lambda: write_document(
            kwargs.get("filename", ""),
            kwargs.get("content", ""),
            mode="create"
        ),

        "append_file": lambda: write_document(
            kwargs.get("filename", ""),
            kwargs.get("content", ""),
            mode="append"
        ),

        "read_write_file": lambda: read_write_file(
            kwargs.get("filename", "")
        ),
    }

    handler = action_map.get(action)
    if not handler:
        return json.dumps({
            "error": f"Unknown action: {action}",
            "available": list(action_map.keys())
        })
    try:
        return handler()
    except Exception as e:
        return json.dumps({"error": str(e), "action": action})


# Load file index on import
_load_file_index()
