"""Extract training signals from image→target_caption pairs.

The agent uses these to ground its vocabulary and grammar choices in the
actual data we'll be evaluated on. No internet needed for this part.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from americasnlp.data import load_jsonl
from americasnlp.languages import LanguageConfig


@dataclass(frozen=True)
class TrainingExample:
    id: str
    target_caption: str
    spanish_caption: Optional[str]   # only present in pilot
    image_filename: str


def _coerce(record: dict) -> Optional[TrainingExample]:
    target = record.get("target_caption")
    if not target:
        return None
    return TrainingExample(
        id=record.get("id", ""),
        target_caption=target,
        spanish_caption=record.get("spanish_caption"),
        image_filename=record.get("filename", ""),
    )


def load_training_examples(
    lang: LanguageConfig,
    data_root: Path,
    splits: List[str] = ("dev", "pilot"),
    *,
    allowed_ids: Optional[set] = None,
) -> List[TrainingExample]:
    """Load (id, target_caption) pairs.

    If `allowed_ids` is given, only rows whose id is in that set are
    returned. Use this to enforce the train/val split: pass `train_ids` so
    the generator agent never sees validation rows.
    """
    out: List[TrainingExample] = []
    for split in splits:
        try:
            rows = load_jsonl(_jsonl_for_split(lang, data_root, split))
        except FileNotFoundError:
            continue
        for row in rows:
            if allowed_ids is not None and row.get("id") not in allowed_ids:
                continue
            ex = _coerce(row)
            if ex is not None:
                out.append(ex)
    return out


def _jsonl_for_split(lang: LanguageConfig, data_root: Path, split: str) -> Path:
    if split == "pilot":
        return data_root / "pilot" / f"{lang.key}.jsonl"
    return data_root / split / lang.key / f"{lang.key}.jsonl"


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Tokenize on Unicode word boundaries; strip combining marks for the key."""
    norm = unicodedata.normalize("NFC", text)
    return [m.group(0) for m in _TOKEN_RE.finditer(norm)]


def extract_content_words(
    examples: List[TrainingExample],
    *,
    min_freq: int = 2,
    top_n: Optional[int] = 200,
) -> List[tuple[str, int]]:
    """Frequency-rank content words across the training captions.

    Returns `[(token, count), ...]` sorted by descending count, filtered to
    tokens appearing at least `min_freq` times. Caller is responsible for any
    further linguistic filtering (stoplists are language-specific and we don't
    have them here — that's part of what the agent figures out).
    """
    counter: Counter[str] = Counter()
    for ex in examples:
        for tok in _tokenize(ex.target_caption):
            counter[tok.lower()] += 1
    ranked = [(t, c) for t, c in counter.most_common() if c >= min_freq]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked
