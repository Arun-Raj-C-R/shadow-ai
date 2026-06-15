"""
Knowledge Graph Engine for Shadow AI Personal Assistant
=========================================================

Implements a directed knowledge graph for tracking people, organizations,
places, and their relationships. Integrates with the existing memory system
(ChromaDB + JSON index in memory_tools.py) via memory ID linking.

Architecture:
    - NetworkX DiGraph for in-memory graph operations
    - JSON file for persistence (memory/knowledge_graph.json)
    - Thread-safe via threading.RLock
    - Bidirectional edges: adding "A->girlfriend->B" auto-adds "B->boyfriend->A"
    - Transitive inference: e.g., girlfriend's parent -> in_law

Author: Shadow AI System
"""

import json
import re
import datetime
import threading
import pathlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

try:
    import networkx as nx
except ImportError:
    nx = None
    print("[GRAPH] WARNING: NetworkX not installed. Install with: pip install networkx")

# ==============================================================================
# --- PATHS ---
# ==============================================================================

_CLI_DIR = pathlib.Path(__file__).parent.parent
GRAPH_FILE_PATH = _CLI_DIR / "memory" / "knowledge_graph.json"

# ==============================================================================
# --- RELATIONSHIP TYPE SYSTEM ---
# ==============================================================================

RELATION_REVERSE_MAP = {
    # Symmetric
    "friend": "friend",
    "spouse": "spouse",
    "sibling": "sibling",
    "colleague": "colleague",
    "classmate": "classmate",
    "cousin": "cousin",
    "neighbor": "neighbor",
    "acquaintance": "acquaintance",
    "roommate": "roommate",

    # Asymmetric pairs
    "girlfriend": "boyfriend",
    "boyfriend": "girlfriend",
    "parent": "child",
    "child": "parent",
    "mother": "child",
    "father": "child",
    "son": "parent",
    "daughter": "parent",
    "teacher": "student",
    "student": "teacher",
    "mentor": "mentee",
    "mentee": "mentor",
    "boss": "subordinate",
    "subordinate": "boss",
    "uncle": "nephew_niece",
    "aunt": "nephew_niece",
    "nephew_niece": "uncle_aunt",
    "uncle_aunt": "nephew_niece",
    "grandparent": "grandchild",
    "grandchild": "grandparent",

    # In-law relations
    "mother_in_law": "son_in_law",
    "father_in_law": "son_in_law",
    "son_in_law": "parent_in_law",
    "daughter_in_law": "parent_in_law",
    "parent_in_law": "child_in_law",
    "child_in_law": "parent_in_law",
    "in_law": "in_law",

    # Professional / organizational
    "member_of": "has_member",
    "has_member": "member_of",
    "works_at": "employs",
    "employs": "works_at",
    "studies_at": "has_student",
    "has_student": "studies_at",
    "founded": "founded_by",
    "founded_by": "founded",

    # Generic fallback
    "knows": "known_by",
    "known_by": "knows",
    "related_to": "related_to",
}

# ==============================================================================
# --- TRANSITIVE INFERENCE RULES ---
# ==============================================================================

# Format: (A->rel1->B, B->rel2->C) => A->inferred_rel->C
INFERENCE_RULES = [
    # Partner's family
    ("girlfriend", "parent", "in_law"),
    ("girlfriend", "mother", "mother_in_law"),
    ("girlfriend", "father", "father_in_law"),
    ("girlfriend", "sibling", "in_law"),
    ("boyfriend", "parent", "in_law"),
    ("boyfriend", "mother", "mother_in_law"),
    ("boyfriend", "father", "father_in_law"),
    ("boyfriend", "sibling", "in_law"),
    ("spouse", "parent", "in_law"),
    ("spouse", "mother", "mother_in_law"),
    ("spouse", "father", "father_in_law"),
    ("spouse", "sibling", "in_law"),

    # Generational
    ("parent", "parent", "grandparent"),
    ("child", "child", "grandchild"),

    # Extended family
    ("parent", "sibling", "uncle_aunt"),
    ("parent", "child", "sibling"),          # My parent's child = my sibling
    ("sibling", "child", "nephew_niece"),
    ("parent", "parent", "grandparent"),

    # Friend-of-friend (low confidence)
    ("friend", "friend", "acquaintance"),
]

# ==============================================================================
# --- DATA MODELS ---
# ==============================================================================


@dataclass
class PersonNode:
    """Represents an entity node in the knowledge graph.

    Can represent a person, organization, place, or the 'self' node (Arun Raj).
    """
    id: str                                     # e.g., "person_arun_raj"
    name: str                                   # "Arun Raj"
    aliases: List[str] = field(default_factory=list)   # ["Arun", "AR"]
    description: str = ""                       # "BSc Physics, SRM University"
    disambiguation: str = ""                    # "Mount Seena College, BSc Physics"
    node_type: str = "person"                   # "self" | "person" | "organization" | "place"
    first_mentioned: str = ""                   # ISO timestamp
    last_updated: str = ""                      # ISO timestamp
    linked_memory_ids: List[str] = field(default_factory=list)  # Memory IDs from memory_index.json
    metadata: Dict[str, Any] = field(default_factory=dict)      # Flexible extra data

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonNode":
        """Reconstruct from a plain dict."""
        # Handle any extra keys gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class Relationship:
    """Represents a directed edge between two nodes in the knowledge graph."""
    source_id: str                              # "person_arun_raj"
    target_id: str                              # "person_divya"
    relation_type: str                          # "girlfriend"
    reverse_type: str = ""                      # "boyfriend" (auto-computed)
    confidence: float = 1.0                     # 0.0â€“1.0
    source_memory_ids: List[str] = field(default_factory=list)  # Which memories established this
    first_established: str = ""                 # ISO timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ==============================================================================
# --- KNOWLEDGE GRAPH MANAGER ---
# ==============================================================================


class KnowledgeGraphManager:
    """
    Core knowledge graph engine.

    Manages a directed graph of people and their relationships, with:
    - Thread-safe CRUD for persons and relationships
    - Bidirectional edge management (auto-reverse)
    - Transitive inference (e.g., girlfriend's parent -> in_law)
    - Memory linking to the existing memory_tools.py system
    - Fuzzy name search and text-based person detection
    - JSON persistence and NetworkX in-memory graph
    """

    def __init__(self, graph_path: pathlib.Path = None):
        """
        Initialize the Knowledge Graph Manager.

        Args:
            graph_path: Path to the JSON persistence file.
                        Defaults to memory/knowledge_graph.json
        """
        self.graph_path = graph_path or GRAPH_FILE_PATH
        self.lock = threading.RLock()

        if nx is None:
            raise ImportError(
                "[GRAPH] NetworkX is required. Install with: pip install networkx"
            )

        self.graph: nx.DiGraph = nx.DiGraph()
        self._person_cache: Dict[str, PersonNode] = {}  # id -> PersonNode
        self._name_index: Dict[str, List[str]] = {}     # lowercase name/alias -> [person_ids]

        self._load_graph()
        self._ensure_root_node()
        print(f"[GRAPH] Knowledge graph initialized â€” {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    # ==========================================================================
    # --- PERSISTENCE ---
    # ==========================================================================

    def _load_graph(self):
        """Load the graph from JSON persistence file."""
        if not self.graph_path.exists():
            print("[GRAPH] No existing graph found, starting fresh")
            self.graph = nx.DiGraph()
            self._person_cache = {}
            self._name_index = {}
            return

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.graph = nx.DiGraph()
            self._person_cache = {}
            self._name_index = {}

            # Reconstruct nodes
            for node_data in data.get("nodes", []):
                person = PersonNode.from_dict(node_data)
                self._person_cache[person.id] = person
                self.graph.add_node(person.id, **person.to_dict())
                self._index_person_name(person)

            # Reconstruct edges
            for edge_data in data.get("edges", []):
                rel = Relationship.from_dict(edge_data)
                self.graph.add_edge(
                    rel.source_id, rel.target_id,
                    **rel.to_dict()
                )

            print(f"[GRAPH] Loaded graph from {self.graph_path}")

        except json.JSONDecodeError as e:
            print(f"[GRAPH] ERROR: Corrupted graph file, starting fresh: {e}")
            self.graph = nx.DiGraph()
            self._person_cache = {}
            self._name_index = {}
        except Exception as e:
            print(f"[GRAPH] ERROR loading graph: {e}")
            self.graph = nx.DiGraph()
            self._person_cache = {}
            self._name_index = {}

    def _save_graph(self):
        """Persist the graph to JSON. Must be called with self.lock held."""
        try:
            # Ensure directory exists
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize nodes
            nodes = []
            for person_id, person in self._person_cache.items():
                nodes.append(person.to_dict())

            # Serialize edges
            edges = []
            for u, v, edge_data in self.graph.edges(data=True):
                edges.append(edge_data)

            data = {
                "version": "1.0",
                "last_saved": datetime.datetime.now().isoformat(),
                "nodes": nodes,
                "edges": edges,
            }

            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[GRAPH] ERROR saving graph: {e}")

    def _ensure_root_node(self):
        """Ensure the central 'self' node (Arun Raj) always exists."""
        root_id = "person_arun_raj"
        if root_id not in self._person_cache:
            root = PersonNode(
                id=root_id,
                name="Arun Raj",
                aliases=["Arun", "AR", "Arun Raj S"],
                description="BSc Physics, SRM University. Founder of Shadow AI. From Kerala, India.",
                disambiguation="Self â€” the user and creator of Shadow",
                node_type="self",
                first_mentioned=datetime.datetime.now().isoformat(),
                last_updated=datetime.datetime.now().isoformat(),
                linked_memory_ids=[],
                metadata={"is_root": True},
            )
            with self.lock:
                self._person_cache[root_id] = root
                self.graph.add_node(root_id, **root.to_dict())
                self._index_person_name(root)
                self._save_graph()
            print("[GRAPH] Root node 'person_arun_raj' created")

    # ==========================================================================
    # --- NAME INDEXING (for fast lookup) ---
    # ==========================================================================

    def _index_person_name(self, person: PersonNode):
        """Add a person's name and aliases to the name index for fast lookup."""
        keys = [person.name.lower()]
        for alias in person.aliases:
            keys.append(alias.lower())

        for key in keys:
            if key not in self._name_index:
                self._name_index[key] = []
            if person.id not in self._name_index[key]:
                self._name_index[key].append(person.id)

    def _deindex_person_name(self, person: PersonNode):
        """Remove a person's name and aliases from the name index."""
        keys = [person.name.lower()]
        for alias in person.aliases:
            keys.append(alias.lower())

        for key in keys:
            if key in self._name_index:
                self._name_index[key] = [
                    pid for pid in self._name_index[key] if pid != person.id
                ]
                if not self._name_index[key]:
                    del self._name_index[key]

    # ==========================================================================
    # --- PERSON ID GENERATION ---
    # ==========================================================================

    def _generate_person_id(self, name: str, disambiguation: str = "") -> str:
        """
        Generate a unique person ID from a name.

        Strategy:
            1. Slugify the name: lowercase, replace spaces/special chars with underscores
            2. Prefix with "person_"
            3. If disambiguation provided, append a key word from it
            4. If ID already exists (collision), append a numeric suffix

        Examples:
            "Gopika" -> "person_gopika"
            "Gopika" + "NIT Calicut" -> "person_gopika_nit"
            "Gopika" + "school friend" -> "person_gopika_school"
        """
        # Slugify the name
        slug = re.sub(r'[^a-z0-9\s]', '', name.lower().strip())
        slug = re.sub(r'\s+', '_', slug).strip('_')

        if not slug:
            slug = "unknown"

        base_id = f"person_{slug}"

        # If disambiguation provided, extract a keyword and append
        if disambiguation:
            disambig_slug = re.sub(r'[^a-z0-9\s]', '', disambiguation.lower().strip())
            disambig_words = disambig_slug.split()
            # Pick the first meaningful word (skip very short/common words)
            stop_words = {"the", "a", "an", "of", "in", "at", "from", "and", "or", "my", "is", "was"}
            keyword = ""
            for w in disambig_words:
                if w not in stop_words and len(w) > 1:
                    keyword = w
                    break
            if keyword:
                base_id = f"person_{slug}_{keyword}"

        # Check for collisions
        final_id = base_id
        counter = 2
        while final_id in self._person_cache:
            # If the existing person has the same name (case-insensitive), it's likely the same person
            existing = self._person_cache[final_id]
            if existing.name.lower() == name.lower().strip():
                # Same person, return existing ID (caller should use update instead)
                return final_id
            final_id = f"{base_id}_{counter}"
            counter += 1

        return final_id

    # ==========================================================================
    # --- PERSON CRUD ---
    # ==========================================================================

    def add_person(
        self,
        name: str,
        description: str = "",
        disambiguation: str = "",
        aliases: List[str] = None,
        node_type: str = "person",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add a new person to the knowledge graph.

        Args:
            name: Full name of the person
            description: Brief description (education, role, context)
            disambiguation: Context to differentiate people with similar names
            aliases: Alternative names/nicknames
            node_type: "person", "organization", "place", or "self"
            metadata: Additional flexible key-value data

        Returns:
            The person_id of the created (or existing) node
        """
        if not name or not name.strip():
            print("[GRAPH] ERROR: Cannot add person with empty name")
            return ""

        name = name.strip()
        person_id = self._generate_person_id(name, disambiguation)
        now = datetime.datetime.now().isoformat()

        with self.lock:
            # If person already exists, just return their ID
            if person_id in self._person_cache:
                existing = self._person_cache[person_id]
                if existing.name.lower() == name.lower():
                    print(f"[GRAPH] Person '{name}' already exists as {person_id}")
                    return person_id

            person = PersonNode(
                id=person_id,
                name=name,
                aliases=aliases or [],
                description=description,
                disambiguation=disambiguation,
                node_type=node_type,
                first_mentioned=now,
                last_updated=now,
                linked_memory_ids=[],
                metadata=metadata or {},
            )

            self._person_cache[person_id] = person
            self.graph.add_node(person_id, **person.to_dict())
            self._index_person_name(person)
            self._save_graph()

        print(f"[GRAPH] Added person: {name} -> {person_id}")
        return person_id

    def update_person(self, person_id: str, **updates) -> bool:
        """
        Update an existing person's attributes.

        Args:
            person_id: The ID of the person to update
            **updates: Key-value pairs to update (name, description, aliases, etc.)

        Returns:
            True if the person was found and updated, False otherwise
        """
        with self.lock:
            if person_id not in self._person_cache:
                print(f"[GRAPH] Person {person_id} not found for update")
                return False

            person = self._person_cache[person_id]

            # Remove old name index entries before update
            self._deindex_person_name(person)

            # Apply updates
            for key, value in updates.items():
                if hasattr(person, key) and key != "id":  # Never allow ID change
                    setattr(person, key, value)

            person.last_updated = datetime.datetime.now().isoformat()

            # Re-index with new names/aliases
            self._index_person_name(person)

            # Update the graph node attributes
            self.graph.nodes[person_id].update(person.to_dict())
            self._save_graph()

        print(f"[GRAPH] Updated person: {person_id}")
        return True

    def get_person(self, person_id: str) -> Optional[PersonNode]:
        """
        Retrieve a person by their ID.

        Returns:
            PersonNode if found, None otherwise
        """
        return self._person_cache.get(person_id)

    def get_all_persons(self) -> List[PersonNode]:
        """
        Get all persons in the knowledge graph.

        Returns:
            List of all PersonNode objects
        """
        return list(self._person_cache.values())

    def add_or_update_person(self, person_info: dict) -> str:
        """
        Upsert a person: find existing by name match or create new.

        Args:
            person_info: Dict with keys: name (required), description, disambiguation,
                        aliases, node_type, metadata

        Returns:
            The person_id of the created or updated node
        """
        name = person_info.get("name", "").strip()
        if not name:
            print("[GRAPH] ERROR: person_info must include 'name'")
            return ""

        # Try to find existing person by name
        existing = self.find_person(name)
        if existing:
            # Check for exact name match
            for person in existing:
                if person.name.lower() == name.lower():
                    # Update existing person
                    updates = {}
                    if person_info.get("description"):
                        updates["description"] = person_info["description"]
                    if person_info.get("aliases"):
                        # Merge aliases
                        merged_aliases = list(set(person.aliases + person_info["aliases"]))
                        updates["aliases"] = merged_aliases
                    if person_info.get("disambiguation"):
                        updates["disambiguation"] = person_info["disambiguation"]
                    if person_info.get("metadata"):
                        merged_meta = {**person.metadata, **person_info["metadata"]}
                        updates["metadata"] = merged_meta

                    if updates:
                        self.update_person(person.id, **updates)
                    return person.id

        # No exact match found â€” create new
        return self.add_person(
            name=name,
            description=person_info.get("description", ""),
            disambiguation=person_info.get("disambiguation", ""),
            aliases=person_info.get("aliases", []),
            node_type=person_info.get("node_type", "person"),
            metadata=person_info.get("metadata", {}),
        )

    def remove_person(self, person_id: str) -> bool:
        """
        Remove a person and all their relationships from the graph.

        Args:
            person_id: The ID of the person to remove

        Returns:
            True if the person was found and removed, False otherwise
        """
        with self.lock:
            if person_id not in self._person_cache:
                print(f"[GRAPH] Person {person_id} not found for removal")
                return False

            if person_id == "person_arun_raj":
                print("[GRAPH] Cannot remove the root node (person_arun_raj)")
                return False

            person = self._person_cache[person_id]
            self._deindex_person_name(person)
            del self._person_cache[person_id]
            self.graph.remove_node(person_id)  # Also removes all connected edges
            self._save_graph()

        print(f"[GRAPH] Removed person: {person_id}")
        return True

    # ==========================================================================
    # --- RELATIONSHIP CRUD ---
    # ==========================================================================

    def _get_reverse_type(self, relation_type: str) -> str:
        """Get the reverse relationship type. Falls back to 'related_to' if unknown."""
        return RELATION_REVERSE_MAP.get(relation_type, "related_to")

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        memory_ids: List[str] = None,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Add a relationship between two persons.

        Automatically creates the reverse relationship as well.
        For example, adding "A->girlfriend->B" also adds "B->boyfriend->A".

        Args:
            source_id: ID of the source person
            target_id: ID of the target person
            relation_type: Type of relationship (e.g., "friend", "girlfriend")
            memory_ids: List of memory IDs that established this relationship
            confidence: Confidence score (0.0â€“1.0)
            metadata: Additional metadata for the relationship

        Returns:
            True if the relationship was added successfully
        """
        with self.lock:
            if source_id not in self._person_cache:
                print(f"[GRAPH] Source person {source_id} not found")
                return False
            if target_id not in self._person_cache:
                print(f"[GRAPH] Target person {target_id} not found")
                return False
            if source_id == target_id:
                print("[GRAPH] Cannot create self-referencing relationship")
                return False

            now = datetime.datetime.now().isoformat()
            relation_type = relation_type.lower().strip()
            reverse_type = self._get_reverse_type(relation_type)

            # Check if relationship already exists -> update instead of duplicate
            if self.graph.has_edge(source_id, target_id):
                existing = self.graph[source_id][target_id]
                # Merge memory IDs
                existing_mems = existing.get("source_memory_ids", [])
                new_mems = list(set(existing_mems + (memory_ids or [])))
                existing["source_memory_ids"] = new_mems
                existing["confidence"] = max(existing.get("confidence", 0), confidence)
                existing["relation_type"] = relation_type
                existing["reverse_type"] = reverse_type
                print(f"[GRAPH] Updated existing relationship: {source_id} ->{relation_type}-> {target_id}")
            else:
                # Create forward edge
                forward_rel = Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    reverse_type=reverse_type,
                    confidence=confidence,
                    source_memory_ids=memory_ids or [],
                    first_established=now,
                    metadata=metadata or {},
                )
                self.graph.add_edge(source_id, target_id, **forward_rel.to_dict())

            # Create or update reverse edge
            if self.graph.has_edge(target_id, source_id):
                rev_existing = self.graph[target_id][source_id]
                rev_mems = rev_existing.get("source_memory_ids", [])
                rev_new_mems = list(set(rev_mems + (memory_ids or [])))
                rev_existing["source_memory_ids"] = rev_new_mems
                rev_existing["confidence"] = max(rev_existing.get("confidence", 0), confidence)
                rev_existing["relation_type"] = reverse_type
                rev_existing["reverse_type"] = relation_type
            else:
                reverse_rel = Relationship(
                    source_id=target_id,
                    target_id=source_id,
                    relation_type=reverse_type,
                    reverse_type=relation_type,
                    confidence=confidence,
                    source_memory_ids=memory_ids or [],
                    first_established=now,
                    metadata=metadata or {},
                )
                self.graph.add_edge(target_id, source_id, **reverse_rel.to_dict())

            self._save_graph()

        source_name = self._person_cache[source_id].name
        target_name = self._person_cache[target_id].name
        print(f"[GRAPH] Relationship added: {source_name} ->{relation_type}-> {target_name} "
              f"(reverse: {reverse_type})")
        return True

    def get_relationships(
        self,
        person_id: str,
        relation_type: str = None,
        direction: str = "outgoing",
    ) -> List[Relationship]:
        """
        Get all relationships for a person.

        Args:
            person_id: ID of the person
            relation_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of Relationship objects
        """
        if person_id not in self._person_cache:
            return []

        relationships = []

        if direction in ("outgoing", "both"):
            for _, target, edge_data in self.graph.out_edges(person_id, data=True):
                rel = Relationship.from_dict(edge_data)
                if relation_type is None or rel.relation_type == relation_type.lower():
                    relationships.append(rel)

        if direction in ("incoming", "both"):
            for source, _, edge_data in self.graph.in_edges(person_id, data=True):
                rel = Relationship.from_dict(edge_data)
                if relation_type is None or rel.relation_type == relation_type.lower():
                    # Avoid duplicates from bidirectional edges
                    if not any(r.source_id == rel.source_id and r.target_id == rel.target_id
                               for r in relationships):
                        relationships.append(rel)

        return relationships

    def remove_relationship(self, source_id: str, target_id: str) -> bool:
        """
        Remove a relationship between two persons (both directions).

        Args:
            source_id: ID of the source person
            target_id: ID of the target person

        Returns:
            True if the relationship was found and removed
        """
        with self.lock:
            removed = False

            if self.graph.has_edge(source_id, target_id):
                self.graph.remove_edge(source_id, target_id)
                removed = True

            if self.graph.has_edge(target_id, source_id):
                self.graph.remove_edge(target_id, source_id)
                removed = True

            if removed:
                self._save_graph()
                print(f"[GRAPH] Removed relationship: {source_id} â†” {target_id}")
            else:
                print(f"[GRAPH] No relationship found between {source_id} and {target_id}")

            return removed

    # ==========================================================================
    # --- MEMORY LINKING ---
    # ==========================================================================

    def link_memory_to_person(self, person_id: str, memory_id: str) -> bool:
        """
        Link a memory (from memory_tools.py) to a person node.

        Args:
            person_id: ID of the person
            memory_id: Memory ID from memory_index.json (e.g., "3f5709babad2")

        Returns:
            True if the link was created successfully
        """
        with self.lock:
            if person_id not in self._person_cache:
                print(f"[GRAPH] Person {person_id} not found for memory linking")
                return False

            person = self._person_cache[person_id]
            if memory_id not in person.linked_memory_ids:
                person.linked_memory_ids.append(memory_id)
                person.last_updated = datetime.datetime.now().isoformat()
                self.graph.nodes[person_id]["linked_memory_ids"] = person.linked_memory_ids
                self._save_graph()
                print(f"[GRAPH] Linked memory {memory_id} to {person.name}")
            return True

    def get_memories_for_person(self, person_id: str) -> List[str]:
        """
        Get all memory IDs linked to a person.

        Args:
            person_id: ID of the person

        Returns:
            List of memory IDs
        """
        person = self._person_cache.get(person_id)
        if person:
            return list(person.linked_memory_ids)
        return []

    def get_shared_memories(self, person_id_1: str, person_id_2: str) -> List[str]:
        """
        Get memory IDs that are linked to both persons.

        Useful for understanding shared context between two people.

        Args:
            person_id_1: ID of the first person
            person_id_2: ID of the second person

        Returns:
            List of shared memory IDs
        """
        mems_1 = set(self.get_memories_for_person(person_id_1))
        mems_2 = set(self.get_memories_for_person(person_id_2))
        return list(mems_1 & mems_2)

    # ==========================================================================
    # --- QUERY OPERATIONS ---
    # ==========================================================================

    def find_person(self, name_query: str) -> List[PersonNode]:
        """
        Find persons by name with case-insensitive fuzzy matching.

        Searches across both primary names and aliases. Supports partial matching.

        Args:
            name_query: Name or partial name to search for

        Returns:
            List of matching PersonNode objects, sorted by relevance
        """
        if not name_query or not name_query.strip():
            return []

        query_lower = name_query.lower().strip()
        results: List[Tuple[PersonNode, float]] = []  # (person, score)
        seen_ids = set()

        for person in self._person_cache.values():
            score = 0.0

            # Exact name match
            if person.name.lower() == query_lower:
                score = 1.0
            # Name starts with query
            elif person.name.lower().startswith(query_lower):
                score = 0.85
            # Query is contained in name
            elif query_lower in person.name.lower():
                score = 0.7

            # Check aliases
            for alias in person.aliases:
                alias_lower = alias.lower()
                if alias_lower == query_lower:
                    score = max(score, 0.95)
                elif alias_lower.startswith(query_lower):
                    score = max(score, 0.8)
                elif query_lower in alias_lower:
                    score = max(score, 0.65)

            if score > 0 and person.id not in seen_ids:
                results.append((person, score))
                seen_ids.add(person.id)

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return [person for person, _ in results]

    def find_persons_in_text(self, text: str) -> List[PersonNode]:
        """
        Find all known persons mentioned in a text string.

        Checks all person names and aliases against the text using
        case-insensitive word boundary matching.

        Args:
            text: The text to scan for person mentions

        Returns:
            List of PersonNode objects found in the text
        """
        if not text or not text.strip():
            return []

        text_lower = text.lower()
        found = []
        seen_ids = set()

        for person in self._person_cache.values():
            if person.id in seen_ids:
                continue

            # Check primary name
            names_to_check = [person.name] + person.aliases

            for name in names_to_check:
                if len(name) < 2:  # Skip single-char aliases
                    continue

                name_lower = name.lower()

                # Use word boundary matching to avoid false positives
                # e.g., "an" shouldn't match inside "manager"
                pattern = r'\b' + re.escape(name_lower) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(person)
                    seen_ids.add(person.id)
                    break

        return found

    def get_connected_persons(
        self,
        person_id: str,
        max_depth: int = 2,
    ) -> List[Tuple[PersonNode, int, List[str]]]:
        """
        BFS traversal from a person node to find connected persons.

        Args:
            person_id: Starting person ID
            max_depth: Maximum traversal depth (default: 2)

        Returns:
            List of tuples: (PersonNode, depth, path_from_start)
            where path is a list of person IDs from start to this node.
        """
        if person_id not in self._person_cache:
            return []

        results = []
        visited = {person_id}
        queue = deque()

        # Initialize BFS with direct neighbors
        for _, neighbor in self.graph.out_edges(person_id):
            if neighbor not in visited:
                queue.append((neighbor, 1, [person_id, neighbor]))
                visited.add(neighbor)

        while queue:
            current_id, depth, path = queue.popleft()
            person = self._person_cache.get(current_id)
            if person:
                results.append((person, depth, path))

            if depth < max_depth:
                for _, neighbor in self.graph.out_edges(current_id):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1, path + [neighbor]))
                        visited.add(neighbor)

        return results

    def query_by_relationship(
        self,
        relation_type: str,
        from_person: str = None,
    ) -> List[Tuple[PersonNode, PersonNode, Relationship]]:
        """
        Find all relationships of a specific type.

        Args:
            relation_type: The relationship type to search for
            from_person: Optional: only show relationships from this person

        Returns:
            List of tuples: (source_person, target_person, relationship)
        """
        relation_type = relation_type.lower().strip()
        results = []

        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get("relation_type") == relation_type:
                if from_person and u != from_person:
                    continue
                source = self._person_cache.get(u)
                target = self._person_cache.get(v)
                if source and target:
                    rel = Relationship.from_dict(edge_data)
                    results.append((source, target, rel))

        return results

    # ==========================================================================
    # --- DISAMBIGUATION ---
    # ==========================================================================

    def disambiguate_name(
        self,
        name: str,
        context_clues: str = "",
    ) -> List[PersonNode]:
        """
        Disambiguate a name using context clues.

        When multiple persons share a name, uses keyword overlap between
        context_clues and person descriptions/disambiguation to rank matches.

        Args:
            name: The ambiguous name
            context_clues: Additional context (e.g., "from college", "NIT", "physics")

        Returns:
            List of matching PersonNode objects sorted by relevance
        """
        candidates = self.find_person(name)

        if not candidates or not context_clues:
            return candidates

        context_words = set(re.findall(r'\w+', context_clues.lower()))

        scored = []
        for person in candidates:
            # Build a bag of words from person's description, disambiguation, metadata
            person_text = " ".join([
                person.description,
                person.disambiguation,
                " ".join(person.aliases),
                json.dumps(person.metadata) if person.metadata else "",
            ]).lower()
            person_words = set(re.findall(r'\w+', person_text))

            # Score = keyword overlap count
            overlap = len(context_words & person_words)

            # Bonus for exact disambiguation match
            if context_clues.lower() in person.disambiguation.lower():
                overlap += 5

            scored.append((person, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [person for person, _ in scored]

    # ==========================================================================
    # --- TRANSITIVE INFERENCE ---
    # ==========================================================================

    def infer_relationships(self) -> List[dict]:
        """
        Apply transitive inference rules to discover implicit relationships.

        For example:
            - If A->girlfriend->B and B->parent->C, infer A->in_law->C
            - If A->parent->B and B->parent->C, infer A->grandparent->C

        Only adds inferred relationships that don't already exist.
        Inferred relationships are marked with confidence=0.7 and metadata={"inferred": True}.

        Returns:
            List of dicts describing newly inferred relationships
        """
        inferred = []

        with self.lock:
            for rel1_type, rel2_type, inferred_type in INFERENCE_RULES:
                # Find all A->rel1->B edges
                for a, b, edge_ab in list(self.graph.edges(data=True)):
                    if edge_ab.get("relation_type") != rel1_type:
                        continue

                    # Find all B->rel2->C edges
                    for _, c, edge_bc in list(self.graph.out_edges(b, data=True)):
                        if edge_bc.get("relation_type") != rel2_type:
                            continue

                        # Skip self-loops and already-existing relationships
                        if a == c:
                            continue
                        if self.graph.has_edge(a, c):
                            # Check if the existing relationship is the same type
                            existing = self.graph[a][c]
                            if existing.get("relation_type") == inferred_type:
                                continue

                        # Infer the new relationship
                        a_name = self._person_cache.get(a, PersonNode(id=a, name=a)).name
                        b_name = self._person_cache.get(b, PersonNode(id=b, name=b)).name
                        c_name = self._person_cache.get(c, PersonNode(id=c, name=c)).name

                        now = datetime.datetime.now().isoformat()
                        reverse_type = self._get_reverse_type(inferred_type)

                        # Collect source memory IDs from both contributing edges
                        source_mems = list(set(
                            edge_ab.get("source_memory_ids", []) +
                            edge_bc.get("source_memory_ids", [])
                        ))

                        # Add forward inferred edge
                        forward_rel = Relationship(
                            source_id=a,
                            target_id=c,
                            relation_type=inferred_type,
                            reverse_type=reverse_type,
                            confidence=0.7,
                            source_memory_ids=source_mems,
                            first_established=now,
                            metadata={
                                "inferred": True,
                                "inference_chain": f"{a_name}->{rel1_type}->{b_name}->{rel2_type}->{c_name}",
                            },
                        )
                        self.graph.add_edge(a, c, **forward_rel.to_dict())

                        # Add reverse inferred edge
                        if not self.graph.has_edge(c, a):
                            reverse_rel = Relationship(
                                source_id=c,
                                target_id=a,
                                relation_type=reverse_type,
                                reverse_type=inferred_type,
                                confidence=0.7,
                                source_memory_ids=source_mems,
                                first_established=now,
                                metadata={
                                    "inferred": True,
                                    "inference_chain": f"{c_name}->{reverse_type}->{a_name} (reverse of inferred)",
                                },
                            )
                            self.graph.add_edge(c, a, **reverse_rel.to_dict())

                        inference_info = {
                            "from": a_name,
                            "to": c_name,
                            "relation": inferred_type,
                            "via": b_name,
                            "chain": f"{rel1_type} + {rel2_type} -> {inferred_type}",
                        }
                        inferred.append(inference_info)
                        print(f"[GRAPH] Inferred: {a_name} ->{inferred_type}-> {c_name} "
                              f"(via {b_name}: {rel1_type}+{rel2_type})")

            if inferred:
                self._save_graph()

        if inferred:
            print(f"[GRAPH] Inference complete: {len(inferred)} new relationships discovered")
        else:
            print("[GRAPH] Inference complete: no new relationships to infer")

        return inferred

    # ==========================================================================
    # --- UTILITY & REPORTING ---
    # ==========================================================================

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the knowledge graph.

        Returns:
            Dict with stats: total persons, relationships, types breakdown, etc.
        """
        # Count relationship types
        rel_types = {}
        for _, _, edge_data in self.graph.edges(data=True):
            rtype = edge_data.get("relation_type", "unknown")
            rel_types[rtype] = rel_types.get(rtype, 0) + 1

        # Count node types
        node_types = {}
        for person in self._person_cache.values():
            ntype = person.node_type
            node_types[ntype] = node_types.get(ntype, 0) + 1

        # Find recently added (last 7 days)
        recent_cutoff = (
            datetime.datetime.now() - datetime.timedelta(days=7)
        ).isoformat()
        recent_persons = [
            p.name for p in self._person_cache.values()
            if p.first_mentioned >= recent_cutoff
        ]

        # Count inferred relationships
        inferred_count = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("metadata", {}).get("inferred", False)
        )

        # Total linked memories
        total_linked_memories = sum(
            len(p.linked_memory_ids) for p in self._person_cache.values()
        )

        return {
            "total_persons": len(self._person_cache),
            "total_relationships": self.graph.number_of_edges(),
            "relationship_types": rel_types,
            "node_types": node_types,
            "inferred_relationships": inferred_count,
            "total_linked_memories": total_linked_memories,
            "recent_additions": recent_persons,
            "graph_density": (
                nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
            ),
        }

    def get_person_profile(self, person_id: str) -> str:
        """
        Generate a human-readable formatted profile for a person.

        Args:
            person_id: ID of the person

        Returns:
            Formatted profile string
        """
        person = self._person_cache.get(person_id)
        if not person:
            return f"[GRAPH] Person '{person_id}' not found."

        lines = [
            f"â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—",
            f"  ðŸ‘¤ {person.name}",
            f"  ID: {person.id}",
            f"  Type: {person.node_type}",
            f"â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•",
        ]

        if person.aliases:
            lines.append(f"  Aliases: {', '.join(person.aliases)}")
        if person.description:
            lines.append(f"  Description: {person.description}")
        if person.disambiguation:
            lines.append(f"  Context: {person.disambiguation}")

        # Relationships
        relationships = self.get_relationships(person_id)
        if relationships:
            lines.append("")
            lines.append("  ðŸ“Ž Relationships:")
            for rel in relationships:
                target = self._person_cache.get(rel.target_id)
                target_name = target.name if target else rel.target_id
                conf_str = f" ({rel.confidence:.0%})" if rel.confidence < 1.0 else ""
                inferred_tag = " [inferred]" if rel.metadata.get("inferred") else ""
                lines.append(f"    -> {rel.relation_type}: {target_name}{conf_str}{inferred_tag}")

        # Linked memories
        if person.linked_memory_ids:
            lines.append("")
            lines.append(f"  ðŸ§  Linked Memories: {len(person.linked_memory_ids)}")
            for mid in person.linked_memory_ids[:5]:  # Show first 5
                lines.append(f"    â€¢ {mid}")
            if len(person.linked_memory_ids) > 5:
                lines.append(f"    ... and {len(person.linked_memory_ids) - 5} more")

        # Timestamps
        lines.append("")
        lines.append(f"  First mentioned: {person.first_mentioned[:10] if person.first_mentioned else 'unknown'}")
        lines.append(f"  Last updated: {person.last_updated[:10] if person.last_updated else 'unknown'}")

        return "\n".join(lines)

    def export_for_prompt(self, person_id: str = "person_arun_raj", max_depth: int = 2) -> str:
        """
        Export a compact graph summary suitable for injection into an LLM prompt.

        Provides a concise view of the knowledge graph centered on a person,
        designed to fit within token limits.

        Args:
            person_id: Center person (default: Arun Raj)
            max_depth: How far to traverse

        Returns:
            Compact string representation of the graph
        """
        center = self._person_cache.get(person_id)
        if not center:
            return ""

        lines = [f"[Knowledge Graph â€” centered on {center.name}]"]

        # Direct relationships
        for rel in self.get_relationships(person_id):
            target = self._person_cache.get(rel.target_id)
            if target:
                lines.append(f"  {center.name} ->{rel.relation_type}-> {target.name}")

        # 2nd degree connections
        if max_depth >= 2:
            for _, neighbor_id in self.graph.out_edges(person_id):
                neighbor = self._person_cache.get(neighbor_id)
                if not neighbor:
                    continue
                for rel2 in self.get_relationships(neighbor_id):
                    if rel2.target_id != person_id:
                        target2 = self._person_cache.get(rel2.target_id)
                        if target2:
                            lines.append(
                                f"  {neighbor.name} ->{rel2.relation_type}-> {target2.name}"
                            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraphManager("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"path='{self.graph_path}')"
        )


# ==============================================================================
# --- MODULE-LEVEL CONVENIENCE (no global instance â€” instantiated externally) ---
# ==============================================================================

def create_graph_manager(graph_path: pathlib.Path = None) -> KnowledgeGraphManager:
    """
    Factory function to create a KnowledgeGraphManager instance.

    Args:
        graph_path: Optional custom path for the graph JSON file

    Returns:
        Initialized KnowledgeGraphManager
    """
    return KnowledgeGraphManager(graph_path=graph_path)
