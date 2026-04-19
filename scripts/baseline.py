"""Language-agnostic direct-prompt baseline for AmericasNLP 2026.

Generates image captions in a target indigenous language using an OpenAI
vision model (default `gpt-4o-mini`). Supports:

    * Zero-shot direct prompting
    * k-shot in-context learning using held-out dev-set examples (image +
      target_caption pairs)
    * Incremental resume (skips rows already present in the output JSONL)
    * ChrF++ scoring when `target_caption` is available (dev split)

The output JSONL follows the shared task's submission format: identical to
the input but with an additional `predicted_caption` field. A parallel CSV
with per-example ChrF++ is written when references are available.

Usage:
    # Zero-shot on the Wixárika dev set
    uv run python scripts/baseline.py --language wixarika --split dev --shots 0

    # 3-shot in-context on the Maya dev set with gpt-4o
    uv run python scripts/baseline.py --language maya --split dev \
        --shots 3 --model gpt-4o

    # All 5 languages, 3-shot, dev split
    for l in bribri guarani maya nahuatl wixarika; do
        uv run python scripts/baseline.py --language $l --split dev --shots 3
    done
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from sacrebleu.metrics.chrf import CHRF

# Make `scripts/` imports work when invoked from the project root.
sys.path.insert(0, str(Path(__file__).parent))
from languages import (  # noqa: E402
    LANGUAGES,
    LanguageConfig,
    load_split,
    resolve_image_path,
    get_split_path,
)

load_dotenv()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_mime_suffix(path: Path) -> str:
    s = path.suffix.lower().lstrip(".")
    if s in ("jpg", "jpeg"):
        return "jpeg"
    if s == "png":
        return "png"
    if s == "webp":
        return "webp"
    return "jpeg"  # safe default


def image_data_url(path: Path) -> str:
    return f"data:image/{image_mime_suffix(path)};base64,{encode_image(path)}"


def compute_chrf(hypothesis: str, reference: str) -> float:
    chrf = CHRF(word_order=2)
    return chrf.corpus_score([hypothesis], [[reference]]).score


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You are an image captioning system for the {language} language "
    "(ISO 639-3: {iso}). Given an image, produce a single caption in "
    "{language}. Output ONLY the {language} caption text — no translation, "
    "no commentary, no quotation marks, no language labels. Keep the style "
    "consistent with the examples you are shown."
)


def build_system_prompt(lang: LanguageConfig) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(language=lang.name, iso=lang.iso)


def build_user_message(image_data_urls: List[str], few_shot_captions: List[str],
                       final_image_url: str) -> list:
    """Build a multimodal message with optional few-shot demonstrations.

    Layout:
        [image_1] Caption in <language>: <caption_1>
        [image_2] Caption in <language>: <caption_2>
        ...
        [image_final] Caption in <language>:
    """
    content: list = []
    assert len(image_data_urls) == len(few_shot_captions)
    for url, caption in zip(image_data_urls, few_shot_captions):
        content.append({"type": "image_url",
                        "image_url": {"url": url, "detail": "auto"}})
        content.append({"type": "text", "text": f"Caption: {caption}"})
    # Final image whose caption we want.
    content.append({"type": "image_url",
                    "image_url": {"url": final_image_url, "detail": "auto"}})
    content.append({"type": "text", "text": "Caption:"})
    return content


# -----------------------------------------------------------------------------
# Few-shot sampling
# -----------------------------------------------------------------------------

def sample_few_shot(records: List[dict], target_id: str, k: int,
                    rng: random.Random) -> List[dict]:
    """Pick k demonstrations from `records`, excluding the target example.

    Demos are sampled from the same split (usually dev). This is
    permitted by the competition rules for the dev set; for the test split
    you should pass a dev-set sample instead (see `load_demo_pool`).
    """
    if k <= 0:
        return []
    pool = [r for r in records if r.get("id") != target_id
            and r.get("target_caption")]
    if len(pool) <= k:
        return list(pool)
    return rng.sample(pool, k)


def load_demo_pool(lang: LanguageConfig, data_root: Path,
                   demo_split: str) -> List[dict]:
    """Load the pool of records we may draw few-shot demos from."""
    try:
        return [r for r in load_split(lang, demo_split, data_root)
                if r.get("target_caption")]
    except FileNotFoundError:
        return []


# -----------------------------------------------------------------------------
# Captioning
# -----------------------------------------------------------------------------

def caption_one(client, model: str, lang: LanguageConfig, record: dict,
                split_dir: Path, demo_pool: List[dict],
                demo_split_dir: Path, k: int, rng: random.Random,
                max_tokens: int) -> str:
    demos = sample_few_shot(demo_pool, record.get("id"), k, rng)
    demo_urls = [image_data_url(resolve_image_path(d, demo_split_dir))
                 for d in demos]
    demo_captions = [d["target_caption"] for d in demos]
    final_url = image_data_url(resolve_image_path(record, split_dir))

    messages = [
        {"role": "system", "content": build_system_prompt(lang)},
        {"role": "user", "content": build_user_message(
            demo_urls, demo_captions, final_url)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    # Guard against the occasional leading "Caption:" echo.
    if text.lower().startswith("caption:"):
        text = text[len("Caption:"):].strip()
    # Strip wrapping quotes if any.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“"):
        text = text[1:-1].strip()
    return text


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def read_existing_predictions(path: Path) -> dict:
    """Return {id: predicted_caption} from an existing JSONL output."""
    if not path.exists():
        return {}
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "predicted_caption" in row:
                preds[row["id"]] = row["predicted_caption"]
    return preds


def write_outputs(out_jsonl: Path, out_csv: Optional[Path],
                  results: List[dict]) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            # Competition submission: all original fields + predicted_caption.
            submit = {k: v for k, v in row.items() if k != "chrf_score"}
            f.write(json.dumps(submit, ensure_ascii=False) + "\n")
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "predicted_caption", "target_caption",
                             "chrf_score"])
            for row in results:
                writer.writerow([
                    row.get("id"),
                    row.get("predicted_caption", ""),
                    row.get("target_caption", ""),
                    f"{row.get('chrf_score', 0.0):.4f}",
                ])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct-prompt (+ few-shot) baseline for AmericasNLP 2026")
    parser.add_argument("--language", required=True, choices=list(LANGUAGES),
                        help="Target language key")
    parser.add_argument("--split", default="dev",
                        choices=["pilot", "dev", "test"],
                        help="Dataset split to caption (default: dev)")
    parser.add_argument("--shots", type=int, default=3,
                        help="Number of few-shot demonstrations (default: 3)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--data-root", default="americasnlp2026/data",
                        type=Path, help="Path to the dataset root")
    parser.add_argument("--demo-split", default=None,
                        help="Split to sample few-shot demos from "
                             "(default: same as --split for dev, 'dev' for test)")
    parser.add_argument("--output-dir", default="results/baseline",
                        type=Path, help="Where to write outputs")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers (default: 8)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Generation max_tokens cap (default: 512)")
    parser.add_argument("--seed", type=int, default=17,
                        help="RNG seed for few-shot sampling (default: 17)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N records (debugging)")
    args = parser.parse_args()

    # Lazy import so --help works without the openai package installed.
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run `uv sync`.",
              file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    lang = LANGUAGES[args.language]
    split_dir = get_split_path(lang, args.split, args.data_root)
    if not split_dir.exists():
        print(f"Error: split directory not found: {split_dir}", file=sys.stderr)
        sys.exit(1)

    records = load_split(lang, args.split, args.data_root)
    if args.limit:
        records = records[:args.limit]
    print(f"[{lang.key}] loaded {len(records)} records from {split_dir}")

    # Few-shot pool.
    demo_split = args.demo_split or ("dev" if args.split == "test" else args.split)
    demo_split_dir = get_split_path(lang, demo_split, args.data_root)
    demo_pool = load_demo_pool(lang, args.data_root, demo_split)
    print(f"[{lang.key}] demo pool: {len(demo_pool)} from split={demo_split}")
    if args.shots > 0 and not demo_pool:
        print(f"Warning: --shots {args.shots} requested but no demos available; "
              "falling back to zero-shot.", file=sys.stderr)

    # Output paths.
    tag = f"{lang.key}_{args.split}_{args.model}_shots{args.shots}"
    out_jsonl = args.output_dir / f"{tag}.jsonl"
    out_csv = args.output_dir / f"{tag}.csv" if args.split in ("pilot", "dev") \
        else None

    existing = read_existing_predictions(out_jsonl)
    if existing:
        print(f"[{lang.key}] {len(existing)} rows already predicted "
              f"— resuming")

    client = OpenAI(api_key=api_key)
    rng = random.Random(args.seed)
    lock = threading.Lock()
    results: List[dict] = []

    def process(record: dict) -> dict:
        rid = record.get("id")
        if rid in existing:
            pred = existing[rid]
        else:
            try:
                pred = caption_one(client, args.model, lang, record,
                                   split_dir, demo_pool, demo_split_dir,
                                   args.shots, rng, args.max_tokens)
            except Exception as e:  # network, rate limit, etc.
                print(f"[{lang.key}] {rid}: ERROR - {e}", file=sys.stderr)
                pred = ""
        row = dict(record)
        row["predicted_caption"] = pred
        if record.get("target_caption") and pred:
            row["chrf_score"] = compute_chrf(pred, record["target_caption"])
        else:
            row["chrf_score"] = 0.0
        with lock:
            print(f"[{lang.key}] {rid}: chrf={row['chrf_score']:.2f}  "
                  f"len={len(pred)}")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, r) for r in records]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r.get("id", ""))
    write_outputs(out_jsonl, out_csv, results)

    # Summary.
    scored = [r for r in results if r.get("target_caption") and r.get("predicted_caption")]
    if scored:
        scores = [r["chrf_score"] for r in scored]
        print()
        print("=" * 60)
        print(f"SUMMARY  language={lang.key}  split={args.split}  "
              f"model={args.model}  shots={args.shots}")
        print(f"  N={len(scores)}  mean ChrF++={sum(scores)/len(scores):.2f}  "
              f"min={min(scores):.2f}  max={max(scores):.2f}")
        # Corpus-level ChrF++ (concatenated reference/hypothesis).
        chrf = CHRF(word_order=2)
        corpus = chrf.corpus_score(
            [r["predicted_caption"] for r in scored],
            [[r["target_caption"] for r in scored]],
        )
        print(f"  corpus ChrF++={corpus.score:.2f}")
    print(f"\nSubmission JSONL: {out_jsonl}")
    if out_csv:
        print(f"Eval CSV:         {out_csv}")


if __name__ == "__main__":
    main()
