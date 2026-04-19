"""Per-language configuration for the AmericasNLP 2026 shared task.

Centralizes language metadata (ISO code, culture key, data quirks) so that
evaluation / baseline scripts can be language-agnostic.

The competition covers 5 languages:
    bribri  (bzd) — Bribri,          Costa Rica
    guarani (grn) — Guaraní,         Paraguay
    maya    (yua) — Yucatec Maya,    Mexico
    nahuatl (nlv) — Orizaba Nahuatl, Mexico
    wixarika (hch) — Wixárika,       Mexico

Guaraní stores `filename` as `data/guarani/images/...`; every other language
uses bare `images/...`. Normalization is applied in `resolve_image_path`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LanguageConfig:
    key: str          # short key used on the filesystem (e.g. "wixarika")
    iso: str          # ISO 639-3 code (e.g. "hch")
    name: str         # human-readable language name (e.g. "Wixárika")
    culture: str      # culture tag from dataset (e.g. "wixarika", "Nahua")

    def dev_dir(self, data_root: Path) -> Path:
        return data_root / "dev" / self.key

    def pilot_dir(self, data_root: Path) -> Path:
        # Only Wixárika has a pilot split in the released data.
        return data_root / "pilot"


LANGUAGES: Dict[str, LanguageConfig] = {
    "bribri":   LanguageConfig("bribri",   "bzd", "Bribri",          "bribri"),
    "guarani":  LanguageConfig("guarani",  "grn", "Guaraní",         "guarani"),
    "maya":     LanguageConfig("maya",     "yua", "Yucatec Maya",    "maya"),
    "nahuatl":  LanguageConfig("nahuatl",  "nlv", "Orizaba Nahuatl", "Nahua"),
    "wixarika": LanguageConfig("wixarika", "hch", "Wixárika",        "wixarika"),
}


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_image_path(record: dict, split_dir: Path) -> Path:
    """Resolve the on-disk image path for a dataset record.

    Guaraní records store `filename` as `data/guarani/images/grn_001.jpg`
    (a path relative to the dataset root). Everything else uses
    `images/<id>.<ext>` relative to the split directory.
    """
    filename = record["filename"]
    # Strip any leading `data/<lang>/` that appears in Guaraní records.
    if filename.startswith("data/"):
        parts = filename.split("/", 2)
        if len(parts) == 3:
            filename = parts[2]
    return split_dir / filename


def get_split_path(lang: LanguageConfig, split: str, data_root: Path) -> Path:
    """Return the directory that contains `<lang>.jsonl` and `images/` for a split."""
    if split == "pilot":
        return lang.pilot_dir(data_root)
    # "dev" or "test"
    return data_root / split / lang.key


def load_split(lang: LanguageConfig, split: str, data_root: Path) -> List[dict]:
    """Load a split's JSONL file. Raises FileNotFoundError if missing."""
    split_dir = get_split_path(lang, split, data_root)
    jsonl_path = split_dir / f"{lang.key}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    return load_jsonl(jsonl_path)


def iter_languages(keys: Optional[List[str]] = None) -> List[LanguageConfig]:
    if keys is None:
        return list(LANGUAGES.values())
    return [LANGUAGES[k] for k in keys]
