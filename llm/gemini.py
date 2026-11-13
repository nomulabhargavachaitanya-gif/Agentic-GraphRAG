"""Shared Gemini client with retry/backoff for free-tier rate limits."""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_client = None
_client_key = None


def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            val = str(st.secrets[key])
            os.environ[key] = val
            return val
    except Exception:
        pass
    return default


def available() -> bool:
    return bool(_get_secret("GEMINI_API_KEY"))


def _get_client():
    global _client, _client_key
    api_key = _get_secret("GEMINI_API_KEY")
    if _client is None or _client_key != api_key:
        from google import genai
        from google.genai import types
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),
        )
        _client_key = api_key
    return _client


def generate(prompt: str, system: str | None = None, json_mode: bool = False,
             max_tokens: int = 2048, retries: int = 5) -> str:
    """Generate text; retries with exponential backoff on 429/5xx."""
    from google.genai import types

    model = _get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=0.2,
        response_mime_type="application/json" if json_mode else None,
    )
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = _get_client().models.generate_content(
                model=model, contents=prompt, config=cfg)
            return resp.text or ""
        except Exception as e:
            msg = str(e)
            transient = any(t in msg for t in ("429", "500", "502", "503", "504", "DEADLINE", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""
