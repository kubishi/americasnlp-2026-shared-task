"""Produce a submission JSONL for the test split.

Identical machinery to `evaluate.py` but it writes only the fields the shared
task asks for, drops scoring, and refuses to be pointed at a split that
contains target captions (so we can't accidentally submit a dev-set output).
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from americasnlp.captioners import Captioner
from americasnlp.data import (
    append_jsonl,
    existing_predictions,
    load_split,
    resolve_image_path,
    split_dir,
    submission_row,
)
from americasnlp.languages import LanguageConfig


def make_submission(
    *,
    captioner: Captioner,
    lang: LanguageConfig,
    split: str,
    data_root: Path,
    output: Path,
    workers: int = 8,
    limit: Optional[int] = None,
) -> Path:
    if split not in ("test", "dev", "pilot"):
        raise ValueError(f"unsupported split: {split!r}")

    records = load_split(lang, split, data_root)
    if limit is not None:
        records = records[:limit]
    base_dir = split_dir(lang, split, data_root)

    done = existing_predictions(output)
    pending = [r for r in records if r["id"] not in done]
    print(f"[{lang.key}/{split}/{captioner.name}] {len(pending)} pending  "
          f"({len(done)} resumed)  -> {output}", file=sys.stderr)

    write_lock = threading.Lock()
    rich_path = output.with_suffix(".rich.jsonl")

    def _process(rec: dict) -> None:
        english_intermediate = None
        back_translation = None
        structured_json = None
        try:
            result = captioner.caption(rec, resolve_image_path(rec, base_dir))
            pred = result.target
            english_intermediate = result.english_intermediate
            back_translation = result.back_translation
            structured_json = getattr(result, "structured_json", None)
        except Exception as exc:  # noqa: BLE001
            print(f"[{rec['id']}] ERROR: {exc}", file=sys.stderr)
            pred = ""
        out = submission_row(rec, pred)
        # Rich record has the same fields plus debug / traceability extras.
        rich = dict(out)
        if english_intermediate:
            rich["english_intermediate"] = english_intermediate
        if back_translation:
            rich["back_translation"] = back_translation
        if structured_json is not None:
            rich["structured_json"] = structured_json
        with write_lock:
            append_jsonl(output, out)
            append_jsonl(rich_path, rich)
        preview = pred.replace("\n", " ")[:60]
        print(f"  {rec['id']}  {preview!r}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process, r) for r in pending]
        for fut in as_completed(futures):
            fut.result()

    print(f"\nwrote {output}", file=sys.stderr)
    return output
