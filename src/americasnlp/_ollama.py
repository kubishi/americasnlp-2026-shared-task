"""Ollama backend helpers.

Local multimodal/text models served via the OpenAI-compatible
`/v1/chat/completions` endpoint that ollama exposes on
`http://127.0.0.1:11434/v1`. Works for both text-only models (passed to
the yaduha translator step) and multimodal models (used as the VLM in
the captioning pipeline).
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


def base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def is_ollama_model(model: str) -> bool:
    """Heuristic: explicit `ollama:` prefix, or a colon-tagged name like
    `qwen2.5vl:7b`, `llama3.1:8b`, `gemma3:27b` that's clearly an ollama
    image tag and not an OpenAI fine-tune ID (those start with `ft:`)."""
    if model.startswith("ollama:"):
        return True
    if model.startswith("ft:"):
        return False
    if ":" in model and not model.startswith(("gpt-", "claude-", "o1-", "o3-", "o4-")):
        return True
    return False


def normalize_ollama_model(model: str) -> str:
    return model.removeprefix("ollama:") if model.startswith("ollama:") else model


def make_openai_client_for_ollama() -> Any:
    """Return an `openai.OpenAI` client pointed at the local ollama server.
    Ollama doesn't enforce auth but the SDK requires *some* api_key string."""
    from openai import OpenAI
    return OpenAI(base_url=base_url(),
                  api_key=os.environ.get("OLLAMA_API_KEY", "ollama"))
