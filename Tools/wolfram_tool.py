# wolfram_tool.py
"""
Core Wolfram|Alpha API client with full error handling, logging, and JSON parsing.
Supports all endpoints: v2/query, v1/simple, v1/spoken, etc.
"""
import requests
import logging
from urllib.parse import urlencode
from typing import Dict, Any, Optional
import os

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Exceptions ---
class WolframAPIError(Exception):
    """Network or HTTP error."""
    pass

class WolframQueryError(Exception):
    """Wolfram returned success=False."""
    pass

# --- Client ---
class WolframAlphaAPI:
    BASE_URL_V2 = "http://api.wolframalpha.com/v2/query"
    BASE_URL_V1 = "http://api.wolframalpha.com/v1"

    def __init__(self, app_id: str):
        if not app_id or app_id.strip() in ["", "DEMO"]:
            raise ValueError("Valid AppID required. Get one at: https://developer.wolframalpha.com/portal/")
        self.app_id = app_id.strip()
        logging.info("WolframAlphaAPI initialized.")

    def _get(self, endpoint: str, params: Dict) -> Dict:
        params["appid"] = self.app_id
        url = f"{self.BASE_URL_V1}/{endpoint}" if endpoint.startswith("v1/") else f"{self.BASE_URL_V2}?{urlencode(params)}"
        try:
            response = requests.get(url, params=params if endpoint.startswith("v1/") else None, timeout=30)
            response.raise_for_status()
            return response.json() if "application/json" in response.headers.get("content-type", "") else {"raw": response.text}
        except requests.exceptions.RequestException as e:
            raise WolframAPIError(f"HTTP error: {e}")
        except ValueError:
            raise WolframAPIError("Invalid JSON response")

    def query(self, query_text: str, **kwargs) -> Dict:
        """Full Results API (v2/query)."""
        params = {"input": query_text, "output": "json"}
        params.update(kwargs)
        data = self._get("v2/query", params)
        result = data.get("queryresult", {})
        if not result.get("success", False):
            err = result.get("error", "Unknown")
            msg = err.get("msg", str(err)) if isinstance(err, dict) else str(err)
            raise WolframQueryError(msg)
        return result

    def simple(self, query_text: str) -> bytes:
        """Simple image API."""
        return requests.get(f"{self.BASE_URL_V1}/simple", params={"i": query_text, "appid": self.app_id}).content

    def spoken(self, query_text: str) -> str:
        """Spoken answer."""
        return requests.get(f"{self.BASE_URL_V1}/spoken", params={"i": query_text, "appid": self.app_id}).text.strip()

    def result(self, query_text: str) -> str:
        """Short answer."""
        return requests.get(f"{self.BASE_URL_V1}/result", params={"i": query_text, "appid": self.app_id}).text.strip()

    def recognizer(self, query_text: str) -> Dict:
        """Fast query recognizer."""
        return self._get("v1/recognizer", {"i": query_text})

    def summarybox(self, entity: str) -> str:
        """Summary box HTML."""
        return requests.get(f"{self.BASE_URL_V1}/summarybox", params={"i": entity, "appid": self.app_id}).text

    def calculator(self, form_data: Dict) -> str:
        """Instant calculator."""
        form_data["appid"] = self.app_id
        return requests.get(f"{self.BASE_URL_V1}/calculator", params=form_data).text