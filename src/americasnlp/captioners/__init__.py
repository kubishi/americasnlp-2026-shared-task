"""Captioner protocol shared by every method in the pipeline.

A `Captioner` returns a `CaptionResult`: the target-language caption plus
optional debug info (intermediate English, back-translated English,
the raw structured `SentenceList` the LLM emitted, etc.). The save-
everything principle here is deliberate: presentation-layer changes
(punctuation, formatting, prompt-tuning of back-translation) should
be doable without re-running the expensive VLM step. Captioners that
have access to the structured output should save it; downstream tools
(retranslate.py, fill_back_translation.py) can re-derive surface
forms from it cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass
class CaptionResult:
    target: str
    english_intermediate: Optional[str] = None  # None for direct baselines
    back_translation: Optional[str] = None  # pipeline only — sentence → English
    structured_json: Optional[list[dict]] = None  # serialized SentenceList; None for direct
    extras: dict[str, Any] = field(default_factory=dict)  # any other model-specific debug state


class Captioner(Protocol):
    """Maps `(record, image_path)` to a `CaptionResult`."""

    name: str

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        ...
