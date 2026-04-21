"""Captioner protocol shared by every method in the pipeline.

A `Captioner` returns a `CaptionResult`: the target-language caption plus
optional debug info (intermediate English, sentence type matched, etc.).
The pipeline method fills in `english_intermediate`; the direct baseline
leaves it None.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class CaptionResult:
    target: str
    english_intermediate: Optional[str] = None  # None for direct baselines


class Captioner(Protocol):
    """Maps `(record, image_path)` to a `CaptionResult`."""

    name: str

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        ...
