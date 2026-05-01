"""Spot-check a "one short sentence" VLM prompt vs the current verbose prompt.

The current `CAPTION_SYSTEM_PROMPT_BASE` asks the VLM to cover everything
it sees in literal detail — that produces 4-8 sentence English, which the
translator turns into 4-8 target-language sentences. Gold captions are
typically 1-3 sentences (often 1). The hypothesis: just asking the VLM
for one short sentence recovers most of the length penalty, no training
needed.

This script runs the pipeline on N dev records per language with both
prompt styles ("detailed" = current, "concise" = new), for one or more
VLM/translator models, and prints per-row + summary ChrF++ with both
arms side-by-side. It does not write to results/dev/ so existing
artifacts are untouched.

Defaults are chosen for a quick free smoke test:
    uv run python scripts/spot_check_concise_prompt.py \\
        --languages bribri --per-lang 5 --models qwen2.5vl:32b

Add a closed-weight model for comparison (will hit the API, ~$0.05):
    uv run python scripts/spot_check_concise_prompt.py \\
        --languages bribri --per-lang 5 \\
        --models qwen2.5vl:32b claude-sonnet-4-5
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from statistics import mean
from typing import Optional

from dotenv import load_dotenv
from sacrebleu.metrics.chrf import CHRF

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


CONCISE_PROMPT_BASE = (
    "Write a single short photo caption for this image — one sentence, "
    "the kind of caption you'd see beside a news or travel photograph. "
    "State the main subject and what's happening. Aim for under 20 words. "
    "No commentary, no quotation marks, no bullet points — just the "
    "caption.\n\n"
)


def build_concise_prompt(language) -> str:
    """Like `_build_caption_prompt` but with the concise base."""
    from americasnlp.captioners.pipeline import (
        CAPTION_GRAMMAR_HEADER,
        _resolve_grammar_string,
    )
    grammar_block = _resolve_grammar_string(language)
    prompt = CONCISE_PROMPT_BASE
    if grammar_block:
        prompt += CAPTION_GRAMMAR_HEADER.format(
            name=language.name, grammar_block=grammar_block)
    return prompt


def chrf_score(hyp: str, ref: str) -> float:
    if not hyp or not ref:
        return 0.0
    return CHRF(word_order=2).corpus_score([hyp], [[ref]]).score


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["bribri"],
                   help="Language keys to spot-check (default: bribri only)")
    p.add_argument("--per-lang", type=int, default=5,
                   help="Records per language (default: 5)")
    p.add_argument("--models", nargs="+", default=["qwen2.5vl:32b"],
                   help="Models to use as both VLM + translator. Default: "
                        "qwen2.5vl:32b (free, local). Add closed-weight "
                        "models to compare e.g. claude-sonnet-4-5.")
    p.add_argument("--styles", nargs="+",
                   default=["detailed", "concise"],
                   choices=["detailed", "concise"],
                   help="Prompt styles to compare (default: both)")
    args = p.parse_args()

    from americasnlp.captioners.pipeline import PipelineCaptioner
    from americasnlp.data import load_split, resolve_image_path, split_dir
    from americasnlp.languages import LANGUAGES

    data_root = REPO_ROOT / "americasnlp2026" / "data"
    results: list[dict] = []

    for lang_key in args.languages:
        if lang_key not in LANGUAGES:
            print(f"unknown language: {lang_key}", file=sys.stderr)
            continue
        lang = LANGUAGES[lang_key]
        records = load_split(lang, "dev", data_root)[:args.per_lang]
        base_dir = split_dir(lang, "dev", data_root)

        for model in args.models:
            captioner = PipelineCaptioner(
                lang=lang,
                vlm_model=model,
                translator_model=model,
            )
            detailed_prompt = captioner._caption_prompt
            concise_prompt = build_concise_prompt(captioner._language)

            for style in args.styles:
                captioner._caption_prompt = (
                    concise_prompt if style == "concise" else detailed_prompt)
                print(f"\n=== {lang_key} / {model} / {style} ===",
                      file=sys.stderr)
                for rec in records:
                    img = resolve_image_path(rec, base_dir)
                    try:
                        cr = captioner.caption(rec, img)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{rec['id']}] ERROR: {exc}", file=sys.stderr)
                        continue
                    target = cr.target.strip()
                    gold = rec.get("target_caption", "").strip()
                    score = chrf_score(target, gold)
                    n_words_target = len(target.split())
                    n_words_gold = len(gold.split())
                    results.append({
                        "language": lang_key,
                        "model": model,
                        "style": style,
                        "id": rec["id"],
                        "english": cr.english_intermediate or "",
                        "target": target,
                        "gold": gold,
                        "chrf": score,
                        "n_words_target": n_words_target,
                        "n_words_gold": n_words_gold,
                    })
                    short_eng = cr.english_intermediate.replace("\n", " ")[:80] \
                        if cr.english_intermediate else ""
                    print(f"  {rec['id']}  chrf={score:5.2f}  "
                          f"len(eng)={len(cr.english_intermediate or ''):>4}  "
                          f"len(tgt)={len(target):>4}", file=sys.stderr)
                    print(f"    eng: {short_eng}", file=sys.stderr)
                    print(f"    tgt: {target[:100]}", file=sys.stderr)

    # Summary table
    if not results:
        print("no results", file=sys.stderr)
        return 1

    print()
    print("=" * 96)
    print(f"{'language':<10} {'model':<20} {'style':<10} {'n':>3} "
          f"{'mean_chrf':>10} {'avg_eng_len':>12} {'avg_tgt_len':>12} "
          f"{'avg_words':>10} {'gold_words':>11}")
    print("=" * 96)

    by_group: dict[tuple, list[dict]] = {}
    for r in results:
        by_group.setdefault(
            (r["language"], r["model"], r["style"]), []
        ).append(r)

    for (lang_key, model, style), rs in sorted(by_group.items()):
        n = len(rs)
        mean_chrf = mean(r["chrf"] for r in rs)
        avg_eng_len = mean(len(r["english"]) for r in rs)
        avg_tgt_len = mean(len(r["target"]) for r in rs)
        avg_words = mean(r["n_words_target"] for r in rs)
        gold_words = mean(r["n_words_gold"] for r in rs)
        print(f"{lang_key:<10} {model:<20} {style:<10} {n:>3} "
              f"{mean_chrf:>10.2f} {avg_eng_len:>12.0f} {avg_tgt_len:>12.0f} "
              f"{avg_words:>10.1f} {gold_words:>11.1f}")

    print()
    # Per-row pairwise comparison (same id, same model: detailed vs concise)
    print("Per-row chrF deltas (concise - detailed) for each (lang, model):")
    by_pair: dict[tuple, list[float]] = {}
    for (lang_key, model, _), _rs in by_group.items():
        if (lang_key, model) in by_pair:
            continue
        det = {r["id"]: r["chrf"] for r in by_group.get(
            (lang_key, model, "detailed"), [])}
        con = {r["id"]: r["chrf"] for r in by_group.get(
            (lang_key, model, "concise"), [])}
        deltas = [con[i] - det[i] for i in det.keys() & con.keys()]
        if deltas:
            by_pair[(lang_key, model)] = deltas

    for (lang_key, model), deltas in sorted(by_pair.items()):
        avg = mean(deltas)
        wins = sum(1 for d in deltas if d > 0)
        print(f"  {lang_key:<10} {model:<22}  "
              f"avg={avg:+6.2f}  wins {wins}/{len(deltas)}")

    print()
    print("=" * 96)
    print("Sample comparisons (gold | detailed → concise):")
    print("=" * 96)
    by_id_lang_model = {}
    for r in results:
        by_id_lang_model.setdefault(
            (r["language"], r["model"], r["id"]), {})[r["style"]] = r
    for (lang_key, model, rid), styles in by_id_lang_model.items():
        if not (styles.get("detailed") and styles.get("concise")):
            continue
        det = styles["detailed"]
        con = styles["concise"]
        gold = det["gold"]
        print(f"\n  {lang_key} / {model} / {rid}")
        print(f"    gold      ({det['n_words_gold']:>2} w): "
              f"{textwrap.shorten(gold, width=110)}")
        print(f"    detailed  ({det['n_words_target']:>2} w, "
              f"chrf {det['chrf']:5.2f}): "
              f"{textwrap.shorten(det['target'], width=110)}")
        print(f"    concise   ({con['n_words_target']:>2} w, "
              f"chrf {con['chrf']:5.2f}): "
              f"{textwrap.shorten(con['target'], width=110)}")

    # Optional: dump raw rows as JSONL for later inspection
    out_path = REPO_ROOT / "results" / "spot_check_concise_prompt.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nraw rows written to {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
