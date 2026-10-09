"""Minimal HTTP client for any OpenAI-compatible chat completions API.

Provider is configured through config/config.yaml; the API key always comes from
the LLM_API_KEY environment variable. Without a key the client is disabled and
the caller falls back to local NLP.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import requests

try:
    # Trust the OS certificate store so TLS-inspecting corporate proxies work.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


class LLMError(Exception):
    """Raised for connection, timeout, rate-limit and auth problems."""


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    # ---------------- public API ----------------

    @property
    def enabled(self) -> bool:
        """True when an API key is configured."""
        return bool(self.api_key)

    def complete(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """Send a prompt, return the parsed message content (JSON-decoded if possible)."""
        if not self.enabled:
            raise LLMError("No LLM_API_KEY configured.")
        return self._http_call(prompt, max_tokens)

    # ---------------- internals ----------------

    def _http_call(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise LLMError("The LLM request timed out. Please try again.") from exc
        except requests.exceptions.SSLError as exc:
            raise LLMError(
                "TLS certificate verification failed. If you are on a corporate "
                "network, install the 'truststore' package so Python trusts the "
                "system certificate store."
            ) from exc
        except requests.ConnectionError as exc:
            raise LLMError("Could not reach the LLM service. Check your connection.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if resp.status_code == 401:
            raise LLMError("LLM API key was rejected (401). Check LLM_API_KEY.")
        if resp.status_code == 429:
            raise LLMError("LLM rate limit reached (429). Wait a moment and retry.")
        if resp.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected LLM response structure") from exc

        if not content:
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
            if finish == "length":
                raise LLMError(
                    "The model hit its token limit before returning an answer. "
                    "Use a non-reasoning model or raise max_tokens."
                )
            raise LLMError("The LLM returned an empty response.")

        try:
            return _extract_json(content)
        except ValueError as exc:
            raise LLMError("LLM returned invalid JSON") from exc


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first balanced JSON object out of an LLM reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError("malformed JSON object") from exc
                if isinstance(obj, dict):
                    return obj
                raise ValueError("JSON is not an object")
    raise ValueError("unterminated JSON object")
