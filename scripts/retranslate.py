"""Re-run only the translator step on saved English intermediates.

Reads an existing pipeline JSONL that has `english_intermediate` for every
row, runs PipelineTranslator against a different model, and writes a new
JSONL with refreshed `predicted_caption`, `chrf`, and `back_translation`.
The VLM step is skipped entirely — saves ~30% of pipeline cost vs a full
re-sweep when the VLM output is already what we want.

Usage:
    uv run python scripts/retranslate.py \\
        --input results/dev/<lang>_dev_pipeline_strict-schema_claude_vlm__gpt-4o-mini_translator.jsonl \\
        --output results/dev/<lang>_dev_pipeline_strict-schema_claude_vlm__gpt-4o_translator.jsonl \\
        --language <lang> --translator gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from sacrebleu.metrics.chrf import CHRF

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from americasnlp.captioners.pipeline import _make_translator_agent  # noqa: E402
from americasnlp.languages import LANGUAGES  # noqa: E402


def chrf_score(hyp: str, ref: str) -> float:
    if not hyp or not ref:
        return 0.0
    return CHRF(word_order=2).corpus_score([hyp], [[ref]]).score


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Existing pipeline JSONL with english_intermediate.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSONL path.")
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--translator", required=True,
                   help="Forward translator model (English -> SentenceList).")
    p.add_argument("--back-translator", default=None,
                   help="Optional separate back-translator model "
                        "(SentenceList -> English). Defaults to --translator. "
                        "Use a cheap model like gpt-4o-mini to keep cost down "
                        "when --translator is gpt-5 / claude.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    pending = [r for r in rows if r.get("english_intermediate")]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[{args.language}/{args.translator}] {len(pending)} rows from "
          f"{args.input.name}", file=sys.stderr)

    from yaduha.loader import LanguageLoader
    from yaduha.translator.pipeline import PipelineTranslator
    lang = LANGUAGES[args.language]
    language = LanguageLoader.load_language(lang.iso)
    bt_model = args.back_translator or args.translator
    translator = PipelineTranslator(
        agent=_make_translator_agent(args.translator),
        back_translation_agent=(
            _make_translator_agent(bt_model)
            if bt_model != args.translator else None
        ),
        SentenceType=language.sentence_types,
    )

    write_lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    def _process(row: dict) -> dict:
        eng = row["english_intermediate"]
        new = dict(row)
        try:
            t = translator.translate(eng)
            new["predicted_caption"] = t.target.strip()
            new["back_translation"] = (
                t.back_translation.source.strip()
                if t.back_translation else ""
            )
            if row.get("target_caption") and new["predicted_caption"]:
                new["chrf"] = chrf_score(
                    new["predicted_caption"], row["target_caption"])
            else:
                new["chrf"] = 0.0
        except Exception as exc:  # noqa: BLE001
            print(f"[{row['id']}] ERROR: {exc}", file=sys.stderr)
            new["predicted_caption"] = ""
            new["back_translation"] = ""
            new["chrf"] = 0.0
        with write_lock:
            out_rows.append(new)
            # Persist progressively
            tmp = args.output.with_suffix(args.output.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in sorted(out_rows, key=lambda x: x.get("id", "")):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(args.output)
        chrf = new.get("chrf", 0.0)
        preview = new["predicted_caption"].replace("\n", " ")[:60]
        print(f"  {row['id']}  chrf={chrf:5.2f}  {preview!r}", file=sys.stderr)
        return new

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process, r) for r in pending]
        for fut in as_completed(futs):
            fut.result()

    # Final summary
    scored = [r for r in out_rows
              if r.get("target_caption") and r.get("predicted_caption")]
    if scored:
        mean_chrf = sum(r.get("chrf", 0.0) for r in scored) / len(scored)
        corpus = CHRF(word_order=2).corpus_score(
            [r["predicted_caption"] for r in scored],
            [[r["target_caption"] for r in scored]],
        ).score
        print(f"\n{args.language}/{args.translator}: "
              f"N={len(scored)}  mean ChrF++={mean_chrf:.2f}  "
              f"corpus={corpus:.2f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
