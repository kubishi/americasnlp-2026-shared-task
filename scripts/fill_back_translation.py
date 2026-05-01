"""Fill `back_translation` on JSONL rows that don't have it.

Naive approach: translate the rendered target caption directly to English
using gpt-4o-mini. Different signal than yaduha's structured-form
back-translation (which goes from the SentenceList, not the rendered
string), but useful as an explorer-side "what does this caption say in
English" readout for one-step outputs that don't have a structured
intermediate available.

Usage:
    uv run python scripts/fill_back_translation.py \\
        --input results/dev/bribri_dev_one-step_gpt-5.jsonl \\
        --language bribri

    # Or pattern-fill all one-step outputs:
    for lang in bribri guarani maya nahuatl wixarika; do
        uv run python scripts/fill_back_translation.py \\
            --input results/dev/${lang}_dev_one-step_gpt-5.jsonl \\
            --language $lang
    done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from americasnlp.languages import LANGUAGES  # noqa: E402


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-translate rows even if they already have a "
                        "back_translation. Off by default.")
    args = p.parse_args()

    from openai import OpenAI
    client = OpenAI()
    lang = LANGUAGES[args.language]

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    pending = [
        r for r in rows
        if r.get("predicted_caption")
        and (args.overwrite or not r.get("back_translation"))
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[{args.language}/{args.input.name}] {len(pending)} pending of "
          f"{len(rows)} rows", file=sys.stderr)
    if not pending:
        return 0

    write_lock = threading.Lock()
    by_id = {r["id"]: r for r in rows}

    def _translate(row: dict) -> str:
        target = row["predicted_caption"].strip()
        if not target:
            return ""
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system",
                 "content": (
                     f"You are a literal translator. Translate the user's "
                     f"{lang.name} ({lang.iso}) caption into English as "
                     "literally as possible. Preserve bracketed placeholders "
                     "verbatim.\n\n"
                     "**If you do not know the language well enough to "
                     "translate accurately**, respond with exactly the "
                     "string 'unable to translate' (no other text). "
                     "DO NOT guess or fabricate a translation when you are "
                     "unfamiliar with the language — an honest 'unable to "
                     "translate' is preferred over a confident hallucination.\n\n"
                     "Output only the English translation OR the literal "
                     "disclaimer string — no commentary, no quotation marks."
                 )},
                {"role": "user", "content": target},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()

    def _process(row: dict) -> None:
        try:
            bt = _translate(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[{row['id']}] ERROR: {exc}", file=sys.stderr)
            return
        with write_lock:
            by_id[row["id"]]["back_translation"] = bt
            tmp = args.input.with_suffix(args.input.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in by_id.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(args.input)
        preview = bt.replace("\n", " ")[:80]
        print(f"  {row['id']}  {preview!r}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process, r) for r in pending]
        for fut in as_completed(futs):
            fut.result()

    n_with_bt = sum(1 for r in by_id.values() if r.get("back_translation"))
    print(f"\n{args.input.name}: {n_with_bt}/{len(rows)} rows have "
          "back_translation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
