"""Regenerate back_translation from saved structured_json — no VLM call.

For each row in a rich submission file that has `structured_json` but no
`back_translation` (e.g. a row whose VLM call succeeded but whose BT
step rate-limited or otherwise failed), reconstruct the Pydantic
`Sentence` instances from the JSON and run yaduha's
`SentenceToEnglishTool` on each to populate the BT field.

Usage:
    uv run python scripts/regen_bt_from_structured.py \\
        --input results/submissions/bribri_one-step_gpt-5.rich.jsonl \\
        --language bribri
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from americasnlp.languages import LANGUAGES  # noqa: E402


def _clean_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if not re.search(r"[.!?]$", s):
        s += "."
    return s[0].upper() + s[1:]


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--bt-model", default="gpt-4o-mini")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    from americasnlp.captioners.pipeline import _make_translator_agent
    from yaduha.loader import LanguageLoader
    from yaduha.tool.sentence_to_english import SentenceToEnglishTool

    lang = LANGUAGES[args.language]
    language = LanguageLoader.load_language(lang.iso)
    sentence_types = list(language.sentence_types)
    bt_tool = SentenceToEnglishTool(
        agent=_make_translator_agent(args.bt_model),
        SentenceType=sentence_types,
    )

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    pending = [r for r in rows
               if r.get("structured_json")
               and not r.get("back_translation", "").strip()]
    print(f"[{args.language}/{args.input.name}] {len(pending)} pending of "
          f"{len(rows)} rows", file=sys.stderr)

    def _reconstruct(s_dict: dict):
        """Try each sentence type until one validates."""
        for SType in sentence_types:
            try:
                return SType.model_validate(s_dict)
            except Exception:  # noqa: BLE001
                continue
        return None

    write_lock = threading.Lock()
    by_id = {r["id"]: r for r in rows}

    def _process(row: dict) -> None:
        bts = []
        for s_dict in row.get("structured_json") or []:
            sent = _reconstruct(s_dict)
            if sent is None:
                continue
            try:
                bt = bt_tool(sent)
                bt_str = bt.content if hasattr(bt, "content") else str(bt)
                bt_str = _clean_sentence(bt_str.strip())
                if bt_str:
                    bts.append(bt_str)
            except Exception as exc:  # noqa: BLE001
                print(f"[{row['id']}] BT err: {exc}", file=sys.stderr)
        new_bt = " ".join(bts).strip()
        with write_lock:
            if new_bt:
                by_id[row["id"]]["back_translation"] = new_bt
            tmp = args.input.with_suffix(args.input.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(args.input)
        print(f"  {row['id']}  {(new_bt[:80] or '(failed)')}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process, r) for r in pending]
        for fut in as_completed(futs):
            fut.result()

    n_with_bt = sum(1 for r in rows if r.get("back_translation"))
    print(f"\n{args.input.name}: {n_with_bt}/{len(rows)} rows have BT",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
