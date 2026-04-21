"""Registry of the 5 shared-task languages.

Pure metadata. Data-loading helpers live in `americasnlp.data`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LanguageConfig:
    key: str        # filesystem-friendly key (e.g. "wixarika"); matches the dev/ subdir
    iso: str        # ISO 639-3 code (e.g. "hch"); matches the yaduha language code
    name: str       # human-readable display name (e.g. "Wixárika")
    culture: str    # `culture` value as it appears in the dataset JSONL

    def dev_dir(self, data_root: Path) -> Path:
        return data_root / "dev" / self.key

    def test_dir(self, data_root: Path) -> Path:
        return data_root / "test" / self.key

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


def iter_languages(keys: Optional[Iterable[str]] = None) -> List[LanguageConfig]:
    if keys is None:
        return list(LANGUAGES.values())
    return [LANGUAGES[k] for k in keys]
