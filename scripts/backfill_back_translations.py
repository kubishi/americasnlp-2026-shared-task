"""Backfill `back_translation` onto existing pipeline dev JSONL rows.

For each row that has `english_intermediate` and a non-empty
`predicted_caption` but no `back_translation`, re-run PipelineTranslator
on the saved English intermediate to capture both the structured
forward parse and the back-translation. Only `back_translation` is
written into the row; the original `predicted_caption` and `chrf` are
left untouched (the new pass may produce a tiny surface-form drift
even at temperature 0; we don't want to silently rewrite scored runs).

The VLM step is skipped entirely (`english_intermediate` is already
saved), so this is much cheaper than a full re-run. Default translator
is the local `qwen2.5vl:32b` via Ollama — free, and per the
2026-04-23 open-weight scan it's the strongest local translator on hand.
The back-translation reflects whatever model is used here, *not* the
model that originally produced `predicted_caption`. For most uses
(qualitative inspection in the explorer) the local model is fine; pass
`--model claude-sonnet-4-5` if you want the back-translation to come
from the same model as the original run.

Usage:
    # Backfill one language using the local model (free):
    uv run python scripts/backfill_back_translations.py --language bribri

    # Sweep all five languages:
    for lang in bribri guarani maya nahuatl wixarika; do
        uv run python scripts/backfill_back_translations.py --language $lang
    done

    # Smoke-test on 2 rows:
    uv run python scripts/backfill_back_translations.py \\
        --language bribri --limit 2
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from americasnlp.captioners.pipeline import (  # noqa: E402
    _make_translator_agent,
)
from americasnlp.languages import LANGUAGES  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--model", default="qwen2.5vl:32b",
                   help="Translator model (default: qwen2.5vl:32b via local "
                        "Ollama — free). The back-translation reflects this "
                        "model, NOT necessarily the model that produced the "
                        "original predicted_caption.")
    p.add_argument("--source-config",
                   default="pipeline_claude-sonnet-4-5",
                   help="Filename stem identifying the source pipeline run "
                        "(default: pipeline_claude-sonnet-4-5). The script "
                        "loads results/dev/<lang>_dev_<source-config>.jsonl.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N pending rows (smoke testing)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--input", type=Path, default=None,
                   help="Override JSONL path (default: results/dev/"
                        "<lang>_dev_<source-config>.jsonl)")
    args = p.parse_args()

    lang = LANGUAGES[args.language]
    if args.input:
        path = args.input
    else:
        path = REPO_ROOT / "results" / "dev" / (
            f"{args.language}_dev_{args.source_config}.jsonl")
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return 1

    rows = load_rows(path)
    pending: list[dict] = [
        r for r in rows
        if r.get("english_intermediate")
        and r.get("predicted_caption")
        and not r.get("back_translation")
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[{args.language}/{args.source_config} via {args.model}] "
          f"{len(pending)} pending of {len(rows)} rows in {path.name}",
          file=sys.stderr)
    if not pending:
        return 0

    # Lazy-build translator once (uses the same plumbing as the captioner).
    from yaduha.loader import LanguageLoader
    from yaduha.translator.pipeline import PipelineTranslator
    language = LanguageLoader.load_language(lang.iso)
    translator = PipelineTranslator(
        agent=_make_translator_agent(args.model),
        SentenceType=language.sentence_types,
    )

    write_lock = threading.Lock()
    by_id = {r["id"]: r for r in rows}

    def _process(row: dict) -> Optional[str]:
        eng = row["english_intermediate"]
        try:
            t = translator.translate(eng)
            bt = (t.back_translation.source.strip()
                  if t.back_translation else "")
        except Exception as exc:  # noqa: BLE001
            print(f"[{row['id']}] ERROR: {exc}", file=sys.stderr)
            return None
        with write_lock:
            by_id[row["id"]]["back_translation"] = bt
            # Persist progressively so a Ctrl-C doesn't lose work.
            write_rows(path, list(by_id.values()))
        preview = bt.replace("\n", " ")[:60]
        print(f"  {row['id']}  {preview!r}", file=sys.stderr)
        return bt

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process, r) for r in pending]
        for fut in as_completed(futures):
            fut.result()

    n_now = sum(1 for r in by_id.values() if r.get("back_translation"))
    print(f"\n{path.name}: {n_now}/{len(rows)} rows have back_translation",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
