import os
# Kill ChromaDB telemetry before it initializes to prevent terminal crash
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pathlib
import json
import datetime
import hashlib
import threading
import re
import time
import concurrent.futures
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

# Knowledge Graph imports
from knowledge_graph import KnowledgeGraphManager
from entity_extractor import extract_entities_and_relations, quick_extract_names

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

# ==============================================================================
# --- ENHANCED MEMORY AGENT CONFIGURATION ---
# ==============================================================================

MEMORY_MODEL_NAME = "gemini-3.1-flash-lite-preview"

# --- Compression threshold ---
# Texts shorter than this are stored directly (no LLM compression needed)
# This eliminates ~80% of LLM calls since most store_memory calls are short
COMPRESS_THRESHOLD = 500  # chars

# --- Define paths ---
# Memory data lives in the 'memory' directory (sibling to Tools)
_CLI_DIR = pathlib.Path(__file__).parent.parent
BASE_DIR = _CLI_DIR / "memory"
# Ensure the memory directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE_PATH = BASE_DIR / "memory.txt"
PROTOCOL_FILE_PATH = _CLI_DIR / "logs" / "protocol.txt"
# Prompt files live in the prompts directory
SPR_PROMPT_PATH = _CLI_DIR / "prompts" / "spr_generator_prompt.txt"
RETRIEVAL_PROMPT_PATH = _CLI_DIR / "prompts" / "retrieval_synthesizer_prompt.txt"
SHADOW_PROMPT_PATH = _CLI_DIR / "prompts" / "shadow_system_prompt.txt"
MEMORY_INDEX_PATH = BASE_DIR / "memory_index.json"
MEMORY_ANALYTICS_PATH = BASE_DIR / "memory_analytics.json"
VECTOR_DB_PATH = BASE_DIR / "vector_memory"

# Configure the client
try:
    memory_openai_client = OpenAI(
        api_key=API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
except Exception as e:
    print(f"Error configuring Gemini for Memory Agent: {e}")
    exit()

# Thread pool for parallel I/O (store operations)
_io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="mem_io")

# ==============================================================================
# --- ENUMS AND DATA STRUCTURES ---
# ==============================================================================

class MemoryType(Enum):
    FACT = "fact"
    PROCEDURE = "procedure"
    PREFERENCE = "preference"
    PROTOCOL = "protocol"
    EXPERIENCE = "experience"
    INSIGHT = "insight"

class RetrievalStrategy(Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"
    CONTEXTUAL = "contextual"
    VECTOR = "vector"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class MemoryEntry:
    id: str
    content: str
    memory_type: MemoryType
    timestamp: str
    importance: float
    access_count: int
    categories: List[str]
    compressed_size: int
    metadata: Dict[str, Any] = None
    vector_id: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ==============================================================================
# --- VECTOR DATABASE INTEGRATION ---
# ==============================================================================

class VectorMemoryManager:
    def __init__(self, persist_directory: str = VECTOR_DB_PATH):
        self.persist_directory = persist_directory
        self.initialized = False
        self.client = None
        self.collection = None
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="shadow_memories",
                metadata={"description": "SHADOW long-term memory storage"}
            )
            
            self.initialized = True
            print("[OK] Vector database initialized successfully")
            
        except ImportError:
            print("[INFO] ChromaDB not installed. Vector storage disabled.")
            self.initialized = False
        except Exception as e:
            print(f"[ERROR] Vector database initialization failed: {e}")
            self.initialized = False

    def store_memory(self, memory_id: str, content: str, memory_type: str, metadata: Dict = None):
        if not self.initialized:
            return False
        
        try:
            vector_metadata = {
                "memory_id": str(memory_id),
                "type": str(memory_type),
                "timestamp": str(datetime.datetime.now().isoformat()),
                "content_preview": str(content[:100] + "..." if len(content) > 100 else content)
            }
        
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)):
                        vector_metadata[key] = ", ".join(str(item) for item in value)
                    else:
                        vector_metadata[key] = str(value) 
        
            self.collection.add(
                documents=[str(content)],
                metadatas=[vector_metadata],
                ids=[str(memory_id)]
            )
            return True
        
        except Exception as e:
            print(f"[ERR] Vector storage error: {e}")
            return False

    def semantic_search(self, query: str, n_results: int = 5, filters: Dict = None):
        if not self.initialized:
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'memory_id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"[ERR] Vector search error: {e}")
            return []

    def delete_memory(self, memory_id: str):
        if not self.initialized:
            return False
            
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"[ERR] Vector delete error: {e}")
            return False

    def get_collection_stats(self):
        if not self.initialized:
            return {"error": "Vector database not initialized"}
            
        try:
            count = self.collection.count()
            return {
                "total_memories": count,
                "vector_db_initialized": True,
                "storage_path": str(self.persist_directory)
            }
        except Exception as e:
            return {"error": str(e), "vector_db_initialized": False}

# ==============================================================================
# --- LOCAL HEURISTIC HELPERS (zero latency) ---
# ==============================================================================

def _heuristic_type(text: str) -> MemoryType:
    """Classify memory type locally â€” zero API cost, <1ms."""
    lower = text.lower()
    if any(k in lower for k in ["always", "never", "prefer", "like", "dislike", "want", "hate"]):
        return MemoryType.PREFERENCE
    if any(k in lower for k in ["step", "how to", "procedure", "process", "workflow"]):
        return MemoryType.PROCEDURE
    if any(k in lower for k in ["protocol", "rule", "policy", "must", "standard", "required"]):
        return MemoryType.PROTOCOL
    if any(k in lower for k in ["insight", "realized", "learned", "discovered", "understand"]):
        return MemoryType.INSIGHT
    if any(k in lower for k in ["happened", "did", "went", "tried", "today", "yesterday"]):
        return MemoryType.EXPERIENCE
    return MemoryType.FACT


def _local_compress(text: str) -> str:
    """Fast local compression â€” strips fluff, keeps facts. <1ms."""
    # Remove common conversational fluff
    fluff_patterns = [
        r"(?i)^(okay|ok|sure|yes|no|well|so|um|uh|hey|hi|hello|please|thanks|thank you)[,.\s]*",
        r"(?i)(i think|i believe|i feel like|it seems like|basically|essentially|you know)\s*",
        r"(?i)(could you|can you|would you|i want you to|i need you to)\s*",
    ]
    result = text
    for pattern in fluff_patterns:
        result = re.sub(pattern, "", result)
    
    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    
    # If still too long, take first 400 chars at sentence boundary
    if len(result) > 500:
        sentences = re.split(r'(?<=[.!?])\s+', result)
        compressed = ""
        for s in sentences:
            if len(compressed) + len(s) > 450:
                break
            compressed += s + " "
        result = compressed.strip() or result[:450]
    
    return result or text[:500]

# ==============================================================================
# --- CORE ENHANCED MEMORY AGENT ---
# ==============================================================================

class EnhancedMemoryAgent:
    def __init__(self):
        self.memory_index: Dict[str, MemoryEntry] = {}
        self.working_memory: List[MemoryEntry] = []
        self.analytics_data = defaultdict(lambda: {'access_count': 0, 'successful_uses': 0})
        self.feedback_loop: List[Dict] = []
        self.lock = threading.RLock()
        
        # Cache the SPR prompt so we don't read disk every call
        self._spr_prompt_cache = None
        self._retrieval_prompt_cache = None
        
        self.vector_db = VectorMemoryManager()
        self._load_memory_index()
        self._load_analytics()
        
        # --- Knowledge Graph ---
        try:
            self.knowledge_graph = KnowledgeGraphManager()
            print("[OK] Knowledge graph initialized")
        except Exception as e:
            print(f"[WARN] Knowledge graph init failed (non-fatal): {e}")
            self.knowledge_graph = None
        
        # Auto-sync: backfill any memories missing from vector DB (runs in background)
        if self.vector_db.initialized and self.memory_index:
            threading.Thread(target=self._sync_vector_db, daemon=True, name="mem_sync").start()

    def _sync_vector_db(self):
        """One-time startup sync: ensures all memories in index are also in ChromaDB."""
        try:
            existing_ids = set(self.vector_db.collection.get()['ids'])
            missing = {k: v for k, v in self.memory_index.items() if k not in existing_ids}
            
            if not missing:
                return  # Already in sync
            
            print(f"ðŸ”„ [Memory] Syncing {len(missing)} memories to vector DB...")
            
            BATCH = 100
            ids_b, docs_b, metas_b = [], [], []
            
            for mem_id, entry in missing.items():
                content = entry.content if hasattr(entry, 'content') else str(entry)
                if not content.strip():
                    continue
                
                mem_type = entry.memory_type.value if hasattr(entry, 'memory_type') else 'fact'
                timestamp = entry.timestamp if hasattr(entry, 'timestamp') else ''
                importance = entry.importance if hasattr(entry, 'importance') else 0.5
                
                ids_b.append(str(mem_id))
                docs_b.append(str(content))
                metas_b.append({
                    "memory_id": str(mem_id),
                    "type": str(mem_type),
                    "timestamp": str(timestamp),
                    "importance": str(importance),
                    "content_preview": str(content[:100])
                })
                
                if len(ids_b) >= BATCH:
                    self.vector_db.collection.add(documents=docs_b, metadatas=metas_b, ids=ids_b)
                    ids_b, docs_b, metas_b = [], [], []
            
            if ids_b:
                self.vector_db.collection.add(documents=docs_b, metadatas=metas_b, ids=ids_b)
            
            print(f"[OK] [Memory] Sync complete â€” vector DB now has {self.vector_db.collection.count()} memories")
        except Exception as e:
            print(f"[WARN] [Memory] Sync error (non-fatal): {e}")

    # ==========================================================================
    # --- MEMORY STORAGE LAYER (OPTIMIZED) ---
    # ==========================================================================

    def _create_memory_id(self, content: str) -> str:
        timestamp = datetime.datetime.now().isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:12]

    def _compress_data(self, data: str) -> str:
        """Smart compression: local for short texts, LLM only for long/complex data."""
        # SHORT TEXT -> skip LLM entirely, store as-is or with light local cleanup
        if len(data) < COMPRESS_THRESHOLD:
            return _local_compress(data)
        
        # LONG TEXT -> use LLM compression (this is background anyway, latency is OK)
        if self._spr_prompt_cache is None:
            self._spr_prompt_cache = self._load_prompt_from_file(SPR_PROMPT_PATH)
        spr_prompt = self._spr_prompt_cache or "Compress the following information into a concise, essential summary that preserves the core meaning and key details:"
        
        try:
            response = memory_openai_client.chat.completions.create(
                model=MEMORY_MODEL_NAME,
                messages=[
                    {"role": "system", "content": spr_prompt},
                    {"role": "user", "content": data}
                ]
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"Compression error: {e}")
            return _local_compress(data)

    def _categorize_memory(self, content: str) -> MemoryType:
        return _heuristic_type(content)  # Local heuristic, zero latency

    def _create_memory_entry(self, content: str, memory_type: MemoryType = None) -> MemoryEntry:
        if memory_type is None:
            memory_type = self._categorize_memory(content)
            
        return MemoryEntry(
            id=self._create_memory_id(content),
            content=content,
            memory_type=memory_type,
            timestamp=datetime.datetime.now().isoformat(),
            importance=0.7 if memory_type in [MemoryType.PROTOCOL, MemoryType.PREFERENCE] else 0.5,
            access_count=0,
            categories=["general"],
            compressed_size=len(content)
        )

    def _persist_memory(self, entry: MemoryEntry) -> str:
        try:
            memory_line = f"[{entry.id}|{entry.memory_type.value}] {entry.content}"
            with open(MEMORY_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(memory_line + "\n---\n")
            return "SUCCESS"
        except Exception as e:
            return f"FAILURE: {e}"

    def store_memory(self, raw_data: str, memory_type: str = None) -> str:
        """Optimized store: local compress for short texts, parallel I/O."""
        if not raw_data.strip():
            return "Agent: Nothing to store."

        t0 = time.perf_counter()
        
        # --- COMPRESS (fast for short texts) ---
        compressed_data = self._compress_data(raw_data)
        if not compressed_data:
            return "Agent: Compression resulted in empty data."

        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type.lower())
            except ValueError:
                mem_type = None

        entry = self._create_memory_entry(compressed_data, mem_type)

        # --- PARALLEL I/O: file + vector + index all at once ---
        file_future = _io_pool.submit(self._persist_memory, entry)
        
        vec_future = None
        if self.vector_db.initialized:
            vec_future = _io_pool.submit(
                self.vector_db.store_memory,
                entry.id, compressed_data, entry.memory_type.value,
                {"importance": str(entry.importance), "compressed_size": str(entry.compressed_size)}
            )

        # Wait for file write (critical path)
        storage_result = file_future.result(timeout=5)
        if not storage_result.startswith("SUCCESS"):
            return storage_result

        # Check vector result (non-critical)
        if vec_future:
            try:
                vector_success = vec_future.result(timeout=5)
                if vector_success:
                    entry.vector_id = entry.id
            except Exception:
                pass

        # Update index (fast, in-memory)
        with self.lock:
            self.memory_index[entry.id] = entry
            self.working_memory.append(entry)
            if len(self.working_memory) > 20:
                self.working_memory.pop(0)
            self._save_memory_index()

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[OK] [Memory] Stored: {entry.id} ({elapsed:.0f}ms)")
        
        # --- KNOWLEDGE GRAPH: Extract entities in background ---
        if self.knowledge_graph:
            _io_pool.submit(self._extract_and_update_graph, entry.id, compressed_data)
        
        return f"SUCCESS: Memory stored with ID {entry.id}"

    # ==========================================================================
    # --- KNOWLEDGE GRAPH INTEGRATION ---
    # ==========================================================================

    def _extract_and_update_graph(self, memory_id: str, content: str):
        """Background: extract entities from stored memory and update knowledge graph."""
        try:
            existing_names = [p.name for p in self.knowledge_graph.get_all_persons()]
            extracted = extract_entities_and_relations(content, existing_names)
            
            for person_info in extracted.get("persons", []):
                name = person_info.get("name", "").strip()
                if not name:
                    continue
                
                # Prepare aliases
                aliases = person_info.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                
                # Add or update person in graph
                pid = self.knowledge_graph.add_or_update_person({
                    "name": name,
                    "description": person_info.get("description", ""),
                    "disambiguation": person_info.get("disambiguation", ""),
                    "aliases": aliases,
                })
                
                if pid:
                    # Link this memory to the person
                    self.knowledge_graph.link_memory_to_person(pid, memory_id)
                    
                    # Add relationship to user if specified
                    rel_to_user = person_info.get("relation_to_user", "").strip()
                    if rel_to_user:
                        self.knowledge_graph.add_relationship(
                            "person_arun_raj", pid, rel_to_user, [memory_id]
                        )
                    
                    # Add relationships to other persons
                    for other_rel in person_info.get("relations_to_others", []):
                        other_name = other_rel.get("name", "").strip()
                        other_relation = other_rel.get("relation", "").strip()
                        if other_name and other_relation:
                            # Find or create the other person
                            other_pid = self.knowledge_graph.add_or_update_person(
                                {"name": other_name}
                            )
                            if other_pid:
                                self.knowledge_graph.link_memory_to_person(other_pid, memory_id)
                                self.knowledge_graph.add_relationship(
                                    pid, other_pid, other_relation, [memory_id]
                                )
            
            # Run transitive inference after adding new relationships
            self.knowledge_graph.infer_relationships()
            
        except Exception as e:
            print(f"[WARN] [Memory] Graph extraction error (non-fatal): {e}")

    def _graph_augmented_retrieve(self, query: str) -> str:
        """Augment retrieval with knowledge graph context."""
        if not self.knowledge_graph:
            return ""
        
        try:
            # Find persons mentioned in the query
            persons = self.knowledge_graph.find_persons_in_text(query)
            if not persons:
                # Also try quick name extraction against known names
                known_names = [p.name for p in self.knowledge_graph.get_all_persons()]
                mentioned = quick_extract_names(query, known_names)
                if mentioned:
                    for name in mentioned:
                        found = self.knowledge_graph.find_person(name)
                        persons.extend(found)
            
            if not persons:
                return ""
            
            # Build context from graph
            context_parts = ["[KNOWLEDGE GRAPH CONTEXT]"]
            seen_ids = set()
            for person in persons:
                if person.id in seen_ids:
                    continue
                seen_ids.add(person.id)
                
                # Get profile
                profile = self.knowledge_graph.get_person_profile(person.id)
                context_parts.append(profile)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"[WARN] [Memory] Graph retrieval error: {e}")
            return ""

    # ==========================================================================
    # --- OPTIMIZED MEMORY RETRIEVAL LAYER ---
    # ==========================================================================

    def retrieve_memory(self, query: str, context: str = "", strategy: RetrievalStrategy = RetrievalStrategy.AUTO) -> str:
        """Ultra-fast retrieval: vector-first, LLM-free when possible."""
        if not query.strip():
            return "Agent: Retrieval query missing."

        t0 = time.perf_counter()

        # â”€â”€ VECTOR-ONLY PATH (fast, <100ms) â”€â”€
        # When ChromaDB is available, use it exclusively.
        # The main AI (Gemini 3.1 Flash) is smart enough to synthesize raw results.
        if self.vector_db.initialized:
            result_text, result_count = self._vector_retrieve(query, n_results=7)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[SEARCH] [Memory] Vector retrieval: {result_count} results ({elapsed:.0f}ms)")
            
            if result_text:
                self._update_retrieval_analytics(query, True)
                # Augment with graph context
                graph_context = self._graph_augmented_retrieve(query)
                if graph_context:
                    result_text = graph_context + "\n\n" + result_text
                return result_text
            
            # Vector returned nothing â€” fall through to semantic
            print("[WARN] [Memory] Vector returned empty, falling back to semantic...")

        # â”€â”€ SEMANTIC FALLBACK (slow, only when vector DB is down or empty) â”€â”€
        results = self._semantic_retrieve(query, context)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[SEARCH] [Memory] Semantic retrieval ({elapsed:.0f}ms)")
        
        self._update_retrieval_analytics(query, bool(results and "No memories found" not in results))
        return f"[MEM] MEMORY RETRIEVED:\n{results}"

    def _vector_retrieve(self, query: str, n_results: int = 7) -> tuple:
        """Direct vector search â€” returns (formatted_string, count) tuple.
        
        ChromaDB's built-in sentence-transformer embeddings handle semantic matching.
        The main AI (Gemini 3.1) is more than capable of interpreting raw results.
        This eliminates the 3-10s LLM synthesis call entirely.
        """
        if not self.vector_db.initialized:
            return "", 0
        
        vector_results = self.vector_db.semantic_search(query, n_results=n_results)
        if not vector_results:
            return "", 0
        
        # Format results with relevance scores for the main AI to interpret
        lines = ["[MEM] MEMORY RETRIEVED (vector search):"]
        for i, r in enumerate(vector_results, 1):
            distance = r.get('distance', 0)
            # ChromaDB default uses squared L2. Typical range 0~1.5+ for sentence-transformers.
            # 0 = identical, 0.5 = very close, 1.0 = somewhat related, 1.5+ = weak match
            relevance = max(0, min(100, int((1 - min(distance, 1.5) / 1.5) * 100)))
            mem_type = r.get('metadata', {}).get('type', 'unknown')
            timestamp = r.get('metadata', {}).get('timestamp', '')
            date_str = timestamp[:10] if timestamp else ''
            
            lines.append(f"  [{i}] ({relevance}% match | {mem_type} | {date_str}) {r['content']}")
        
        return "\n".join(lines), len(vector_results)

    def _hybrid_retrieve(self, query: str, context: str) -> str:
        """Legacy hybrid path â€” kept for compatibility but not used by default."""
        if not self.vector_db.initialized:
            return self._semantic_retrieve(query, context)
        
        vector_results = self.vector_db.semantic_search(query, n_results=5)
        vector_content = "\n".join([f"â€¢ {r['content']}" for r in vector_results]) if vector_results else "No vector results."
        
        return f"VECTOR RESULTS:\n{vector_content}"

    def _semantic_retrieve(self, query: str, context: str) -> str:
        """LLM-based semantic search â€” ONLY used as fallback when vector DB is unavailable.
        
        Optimization: Instead of sending ALL 62KB of memory.txt, we:
        1. Split into entries
        2. Do local keyword pre-filtering to narrow down candidates
        3. Send only the top candidates to the LLM
        """
        memory_content = self._read_memory_file()
        if not memory_content.strip():
            return "No memories found."

        # --- LOCAL PRE-FILTER: keyword match to reduce what goes to LLM ---
        entries = [e.strip() for e in memory_content.split("---") if e.strip()]
        query_words = set(query.lower().split())
        
        # Score entries by keyword overlap
        scored = []
        for entry in entries:
            entry_lower = entry.lower()
            score = sum(1 for w in query_words if w in entry_lower)
            scored.append((score, entry))
        
        # Take top 20 entries by relevance (or all if fewer)
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered_entries = [e for _, e in scored[:20]]
        filtered_content = "\n---\n".join(filtered_entries)
        
        if self._retrieval_prompt_cache is None:
            self._retrieval_prompt_cache = self._load_prompt_from_file(RETRIEVAL_PROMPT_PATH)
        retrieval_prompt = self._retrieval_prompt_cache or "Extract and summarize the most relevant information from the following memories that directly addresses the user's query."
        
        try:
            response = memory_openai_client.chat.completions.create(
                model=MEMORY_MODEL_NAME,
                messages=[
                    {"role": "system", "content": retrieval_prompt},
                    {"role": "user", "content": f"Query: {query}\nContext: {context}\nMemories:\n{filtered_content}"}
                ]
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Retrieval error: {e}"

    # ==========================================================================
    # --- PROTOCOL MANAGEMENT ---
    # ==========================================================================

    def update_protocol(self, protocol_update: str) -> str:
        if not protocol_update.strip():
            return "Agent: No protocol update provided."

        print("[SYS] [Protocol] Processing protocol update...")
        compressed_protocol = self._compress_data(protocol_update)
        if not compressed_protocol:
            return "Agent: Protocol compression failed."

        entry = self._create_memory_entry(compressed_protocol, MemoryType.PROTOCOL)
        entry.importance = 0.9

        try:
            with open(PROTOCOL_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{entry.timestamp}] {entry.content}\n")
            
            self.memory_index[entry.id] = entry
            self.vector_db.store_memory(
                entry.id, compressed_protocol, "protocol", metadata={"importance": "0.9"} 
            )
            self._save_memory_index()
            return "SUCCESS: Protocol updated and stored."
        except Exception as e:
            return f"FAILURE: {e}"

    # ==========================================================================
    # --- UTILITY & FILESYSTEM OPERATIONS ---
    # ==========================================================================

    def _load_prompt_from_file(self, file_path: pathlib.Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    def _read_memory_file(self) -> str:
        if not MEMORY_FILE_PATH.exists():
            return ""
        try:
            with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    def _load_memory_index(self):
        if MEMORY_INDEX_PATH.exists():
            try:
                with open(MEMORY_INDEX_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for mem_id, mem_data in data.items():
                        mem_data['memory_type'] = MemoryType(mem_data['memory_type'])
                        self.memory_index[mem_id] = MemoryEntry(**mem_data)
            except Exception as e:
                print(f"Error loading index: {e}")

    def _save_memory_index(self):
        try:
            with open(MEMORY_INDEX_PATH, "w", encoding="utf-8") as f:
                serializable_data = {}
                for mem_id, entry in self.memory_index.items():
                    entry_dict = asdict(entry)
                    entry_dict['memory_type'] = entry.memory_type.value
                    serializable_data[mem_id] = entry_dict
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            print(f"Error saving index: {e}")

    def _load_analytics(self):
        if MEMORY_ANALYTICS_PATH.exists():
            try:
                with open(MEMORY_ANALYTICS_PATH, "r", encoding="utf-8") as f:
                    self.analytics_data.update(json.load(f))
            except:
                pass

    def _save_analytics(self):
        try:
            with open(MEMORY_ANALYTICS_PATH, "w", encoding="utf-8") as f:
                json.dump(dict(self.analytics_data), f, indent=2)
        except:
            pass

    def _update_retrieval_analytics(self, query: str, success: bool):
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        self.analytics_data[query_hash]['access_count'] += 1
        if success:
            self.analytics_data[query_hash]['successful_uses'] += 1
        self._save_analytics()

    def add_feedback(self, memory_id: str, feedback: str, was_helpful: bool):
        self.feedback_loop.append({
            'memory_id': memory_id,
            'feedback': feedback,
            'was_helpful': was_helpful,
            'timestamp': datetime.datetime.now().isoformat()
        })
        if memory_id in self.memory_index:
            if was_helpful:
                self.memory_index[memory_id].importance = min(1.0, self.memory_index[memory_id].importance + 0.1)
            else:
                self.memory_index[memory_id].importance = max(0.1, self.memory_index[memory_id].importance - 0.15)
            self._save_memory_index()

    def get_system_health(self) -> Dict[str, Any]:
        health = {
            "total_memories": len(self.memory_index),
            "working_memory_count": len(self.working_memory),
            "vector_database": self.vector_db.get_collection_stats()
        }
        if self.knowledge_graph:
            health["knowledge_graph"] = self.knowledge_graph.get_graph_summary()
        return health

# ==============================================================================
# --- GLOBAL INSTANCE AND EXPORTS ---
# ==============================================================================

enhanced_memory_agent = EnhancedMemoryAgent()

def run_store_logic(raw_data: str, memory_type: str = None) -> str:
    return enhanced_memory_agent.store_memory(raw_data, memory_type)

def run_retrieve_logic(query: str, context: str = "") -> str:
    return enhanced_memory_agent.retrieve_memory(query, context)

def run_update_protocol_logic(protocol_update: str) -> str:
    return enhanced_memory_agent.update_protocol(protocol_update)

def get_memory_health() -> Dict[str, Any]:
    return enhanced_memory_agent.get_system_health()

def add_memory_feedback(memory_id: str, feedback: str, was_helpful: bool):
    return enhanced_memory_agent.add_feedback(memory_id, feedback, was_helpful)

def find_similar_memories(content: str, n_results: int = 5) -> List[Dict]:
    return enhanced_memory_agent.vector_db.semantic_search(content, n_results)

def get_vector_stats() -> Dict[str, Any]:
    return enhanced_memory_agent.vector_db.get_collection_stats()

def rebuild_vector_database():
    pass # Disabled in lightweight version to prevent accidental quota burn

def load_prompt_from_file(file_path: pathlib.Path, is_optional: bool = False) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if is_optional:
            return ""
        print(f"Error: Prompt file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading prompt file {file_path}: {e}")
        return ""

# ==============================================================================
# --- KNOWLEDGE GRAPH TOOL EXPORTS ---
# ==============================================================================

def graph_add_person(name: str, description: str, disambiguation: str = "", aliases: str = "") -> str:
    """Add a person to the knowledge graph. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []
        person_id = enhanced_memory_agent.knowledge_graph.add_person(
            name=name,
            description=description,
            disambiguation=disambiguation,
            aliases=alias_list
        )
        if person_id:
            return f"SUCCESS: Person '{name}' added to knowledge graph with ID: {person_id}"
        return f"ERROR: Failed to add person '{name}'"
    except Exception as e:
        return f"ERROR: {e}"


def graph_add_relationship(person1_name: str, person2_name: str, relation_type: str) -> str:
    """Add a relationship between two people. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        kg = enhanced_memory_agent.knowledge_graph
        
        # Find or create person 1
        p1_matches = kg.find_person(person1_name)
        if p1_matches:
            p1_id = p1_matches[0].id
        else:
            p1_id = kg.add_person(name=person1_name)
        
        # Find or create person 2
        p2_matches = kg.find_person(person2_name)
        if p2_matches:
            p2_id = p2_matches[0].id
        else:
            p2_id = kg.add_person(name=person2_name)
        
        if not p1_id or not p2_id:
            return f"ERROR: Could not resolve both persons."
        
        success = kg.add_relationship(p1_id, p2_id, relation_type)
        if success:
            # Run inference after new relationship
            inferred = kg.infer_relationships()
            inferred_msg = f" ({len(inferred)} new relationships inferred)" if inferred else ""
            return f"SUCCESS: {person1_name} ->{relation_type}-> {person2_name} added.{inferred_msg}"
        return f"ERROR: Failed to add relationship."
    except Exception as e:
        return f"ERROR: {e}"


def graph_query_person(name_query: str, context: str = "") -> str:
    """Look up a person in the knowledge graph. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        kg = enhanced_memory_agent.knowledge_graph
        
        if context:
            persons = kg.disambiguate_name(name_query, context)
        else:
            persons = kg.find_person(name_query)
        
        if not persons:
            return f"No person found matching '{name_query}' in the knowledge graph."
        
        results = []
        for person in persons:
            profile = kg.get_person_profile(person.id)
            results.append(profile)
        
        if len(persons) > 1:
            header = f"Found {len(persons)} person(s) matching '{name_query}':\n"
            return header + "\n\n".join(results)
        
        return results[0]
    except Exception as e:
        return f"ERROR: {e}"


def graph_query_relationship(person_name: str, relation_type: str = "") -> str:
    """Find relationships for a person. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        kg = enhanced_memory_agent.knowledge_graph
        
        # Find the person
        persons = kg.find_person(person_name)
        if not persons:
            return f"No person found matching '{person_name}'."
        
        person = persons[0]
        
        if relation_type:
            # Query specific relationship type
            rels = kg.get_relationships(person.id, relation_type=relation_type)
        else:
            # Get all relationships
            rels = kg.get_relationships(person.id)
        
        if not rels:
            return f"No relationships found for '{person.name}'" + (f" of type '{relation_type}'" if relation_type else "")
        
        lines = [f"Relationships for {person.name}:"]
        for rel in rels:
            target = kg.get_person(rel.target_id)
            target_name = target.name if target else rel.target_id
            conf = f" ({rel.confidence:.0%})" if rel.confidence < 1.0 else ""
            inferred = " [inferred]" if rel.metadata.get("inferred") else ""
            lines.append(f"  -> {rel.relation_type}: {target_name}{conf}{inferred}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


def graph_get_connections(person_name: str, depth: int = 2) -> str:
    """Get network of connections around a person. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        kg = enhanced_memory_agent.knowledge_graph
        
        # Find the person
        persons = kg.find_person(person_name)
        if not persons:
            return f"No person found matching '{person_name}'."
        
        person = persons[0]
        depth = min(max(depth, 1), 3)  # Clamp to 1-3
        
        connections = kg.get_connected_persons(person.id, max_depth=depth)
        
        if not connections:
            return f"No connections found for '{person.name}' within {depth} degrees."
        
        lines = [f"Connections for {person.name} (up to {depth} degrees):"]
        for connected_person, dist, path in connections:
            # Build the path string
            path_names = []
            for pid in path:
                p = kg.get_person(pid)
                path_names.append(p.name if p else pid)
            
            # Get the relationship type for this edge
            rel_info = ""
            if dist == 1:
                rels = kg.get_relationships(person.id)
                for r in rels:
                    if r.target_id == connected_person.id:
                        rel_info = f" ({r.relation_type})"
                        break
            
            indent = "  " * dist
            lines.append(f"{indent}-> [{dist}Â° ] {connected_person.name}{rel_info}")
            if connected_person.description:
                lines.append(f"{indent}     {connected_person.description}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


def graph_get_summary() -> str:
    """Get summary of the entire knowledge graph. Called by shadow.py via function calling."""
    if not enhanced_memory_agent.knowledge_graph:
        return "ERROR: Knowledge graph not initialized."
    
    try:
        kg = enhanced_memory_agent.knowledge_graph
        summary = kg.get_graph_summary()
        
        lines = [
            "â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—",
            "  ðŸ•¸ï¸  KNOWLEDGE GRAPH SUMMARY",
            "â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•",
            f"  Total Persons: {summary['total_persons']}",
            f"  Total Relationships: {summary['total_relationships']}",
            f"  Inferred Relationships: {summary['inferred_relationships']}",
            f"  Total Linked Memories: {summary['total_linked_memories']}",
            f"  Graph Density: {summary['graph_density']:.3f}",
        ]
        
        if summary.get("node_types"):
            lines.append("\n  Node Types:")
            for ntype, count in summary["node_types"].items():
                lines.append(f"    {ntype}: {count}")
        
        if summary.get("relationship_types"):
            lines.append("\n  Relationship Types:")
            for rtype, count in summary["relationship_types"].items():
                lines.append(f"    {rtype}: {count}")
        
        if summary.get("recent_additions"):
            lines.append(f"\n  Recent Additions (last 7 days): {', '.join(summary['recent_additions'])}")
        
        # Also include a compact graph view
        graph_view = kg.export_for_prompt()
        if graph_view:
            lines.append(f"\n{graph_view}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"