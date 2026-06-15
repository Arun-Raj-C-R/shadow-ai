"""
Entity Extractor for Shadow Memory System.

Extracts person entities and relationships from memory text using an LLM.
Called after a memory is stored to populate the knowledge graph.
"""

import os
import re
import json
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

_extractor_client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
EXTRACTOR_MODEL = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """\
You are an entity-extraction engine for a personal AI assistant belonging to **Arun Raj**.
Given a memory text, extract every **person** mentioned (other than Arun Raj himself).

Rules:
1. For each person return: name, relation_to_user (if inferrable), a one-line description,
   disambiguation clues, aliases, and relations_to_others.
2. Use the EXISTING PERSONS list to match namesâ€”prefer an existing spelling/name over
   creating a new entry. If a reference like "my friend" or "she" clearly maps to an
   existing person from context, use that person's name.
3. If no persons are mentioned, return empty lists.
4. Output **only** valid JSON matching the schema belowâ€”no markdown fences, no commentary.

Schema:
{
  "persons": [
    {
      "name": "<string>",
      "relation_to_user": "<string or empty>",
      "description": "<brief string>",
      "disambiguation": "<string or empty>",
      "aliases": ["<string>"],
      "relations_to_others": [
        {"name": "<other person>", "relation": "<relation label>"}
      ]
    }
  ],
  "locations": [],
  "events": []
}
"""

EMPTY_RESULT: Dict = {"persons": [], "locations": [], "events": []}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quick_extract_names(text: str, known_names: List[str]) -> List[str]:
    """Fast local check: which known persons are mentioned in this text?

    Performs case-insensitive substring matchingâ€”no LLM call.
    """
    text_lower = text.lower()
    return [name for name in known_names if name.lower() in text_lower]


def _parse_llm_json(raw: str) -> Dict:
    """Parse JSON from LLM output, stripping optional markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    data = json.loads(cleaned)

    # Ensure top-level keys exist with correct defaults
    result: Dict = {
        "persons": data.get("persons", []),
        "locations": data.get("locations", []),
        "events": data.get("events", []),
    }

    # Normalise each person entry
    for p in result["persons"]:
        p.setdefault("name", "")
        p.setdefault("relation_to_user", "")
        p.setdefault("description", "")
        p.setdefault("disambiguation", "")
        p.setdefault("aliases", [])
        p.setdefault("relations_to_others", [])

    return result


def _text_likely_has_persons(text: str) -> bool:
    """Quick heuristic: does the text plausibly mention a person?"""
    # Any capitalised word that isn't at sentence start is a decent signal,
    # but the simplest reliable check is just length + presence of a proper noun pattern.
    if len(text) < 20:
        return False
    # Look for capitalised words (rough proxy for proper nouns) or relationship keywords
    if re.search(r"\b[A-Z][a-z]{1,}", text):
        return True
    relationship_keywords = {"friend", "brother", "sister", "mom", "dad", "mother",
                             "father", "girlfriend", "boyfriend", "wife", "husband",
                             "colleague", "boss", "uncle", "aunt", "cousin"}
    text_lower = text.lower()
    return any(kw in text_lower for kw in relationship_keywords)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_entities_and_relations(
    text: str, existing_persons: List[str]
) -> Dict:
    """Extract person entities and relationships from a memory text via LLM.

    Args:
        text: The memory text to analyse (already compressed/cleaned).
        existing_persons: Person names already in the knowledge graph.

    Returns:
        Dict with keys ``persons``, ``locations``, ``events``.
    """
    # Short-circuit: skip LLM for trivial / person-free text
    if not _text_likely_has_persons(text):
        print("[ENTITY] Skipped extraction â€“ text unlikely to contain persons.")
        return dict(EMPTY_RESULT)

    user_prompt = (
        f"EXISTING PERSONS: {json.dumps(existing_persons)}\n\n"
        f"MEMORY TEXT:\n{text}"
    )

    try:
        response = _extractor_client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        print(f"[ENTITY] Raw LLM response length: {len(raw)} chars")

        result = _parse_llm_json(raw)
        print(f"[ENTITY] Extracted {len(result['persons'])} person(s).")
        return result

    except json.JSONDecodeError:
        print("[ENTITY] Failed to parse LLM JSON response â€“ returning empty.")
        return dict(EMPTY_RESULT)
    except Exception as exc:
        print(f"[ENTITY] Extraction failed ({type(exc).__name__}): {exc}")
        return dict(EMPTY_RESULT)
