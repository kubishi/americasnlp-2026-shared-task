"""Dataset loading utilities for the AmericasNLP 2026 shared task.

JSONL parsing, image path resolution (with the Guaraní quirk), and image
encoding for vision models. Pillow is used to re-encode images so that
mislabelled extensions don't break the OpenAI vision API.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from americasnlp.languages import LanguageConfig


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_dir(lang: LanguageConfig, split: str, data_root: Path) -> Path:
    """Return the directory holding `<lang>.jsonl` and `images/` for one split."""
    if split == "pilot":
        return lang.pilot_dir(data_root)
    if split == "dev":
        return lang.dev_dir(data_root)
    if split == "test":
        return lang.test_dir(data_root)
    raise ValueError(f"unknown split: {split!r}")


def split_jsonl(lang: LanguageConfig, split: str, data_root: Path) -> Path:
    return split_dir(lang, split, data_root) / f"{lang.key}.jsonl"


def load_split(lang: LanguageConfig, split: str, data_root: Path) -> List[dict]:
    path = split_jsonl(lang, split, data_root)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    return load_jsonl(path)


def resolve_image_path(record: dict, base_dir: Path) -> Path:
    """Resolve a record's image to an on-disk path.

    Guaraní stores `filename` as `data/guarani/images/grn_001.jpg` (relative to the
    repo root); every other language stores `images/<id>.<ext>` relative to the
    split directory. We strip the `data/<lang>/` prefix when present.
    """
    filename = record["filename"]
    if filename.startswith("data/"):
        parts = filename.split("/", 2)
        if len(parts) == 3:
            filename = parts[2]
    return base_dir / filename


def image_data_url(path: Path) -> str:
    """Re-encode the image through Pillow, then base64-data-URL it.

    A handful of dev-set images (notably some Bribri ones) have wrong
    extensions — WebP/PNG bytes saved as `.jpg`. The OpenAI API rejects those
    with `invalid_image_format`. Going through Pillow normalizes everything
    to PNG and side-steps the issue.
    """
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def submission_row(record: dict, predicted_caption: str) -> dict:
    """Build a submission row in the format the shared task expects.

    Per the task README: same fields as the test JSONL plus `predicted_caption`.
    """
    keep = ("id", "filename", "split", "culture", "language", "iso_lang")
    out = {k: record[k] for k in keep if k in record}
    out["predicted_caption"] = predicted_caption
    return out


def existing_predictions(path: Optional[Path]) -> Dict[str, str]:
    """Read {id: predicted_caption} from a partially-written JSONL output."""
    if path is None or not path.exists():
        return {}
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("predicted_caption"):
                out[row["id"]] = row["predicted_caption"]
    return out
