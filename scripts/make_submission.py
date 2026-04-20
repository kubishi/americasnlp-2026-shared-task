"""Generate a submission file in the AmericasNLP 2026 expected format.

From the task README:
  > Participants must generate a `predicted_caption` for each entry and include
  > it as an additional field in their submission file.

The input JSONL has the same schema as the dev set (see
`americasnlp2026/data/dev/<lang>/<lang>.jsonl`) but without `target_caption`.
This script reuses the captioning methods from `scripts/evaluate_dev.py`, so
any method that runs in the evaluator can produce a submission.

Usage
-----
    # Dry run on dev (where we have references — useful for smoke testing the
    # submission pipeline).
    uv run python scripts/make_submission.py \\
        --input americasnlp2026/data/dev/wixarika/wixarika.jsonl \\
        --language wixarika \\
        --method direct-target \\
        --output submissions/wixarika_dev_direct.jsonl

    # On the test set once it's released:
    uv run python scripts/make_submission.py \\
        --input path/to/test_wixarika.jsonl \\
        --language wixarika \\
        --method spanish-pivot \\
        --output submissions/wixarika_test.jsonl

The output JSONL has each input row verbatim plus a `predicted_caption` field.
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
from openai import OpenAI

# Share the captioning implementations with the evaluator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_dev import DEV_LANGUAGES, METHODS, _resolve_image_path  # type: ignore  # noqa: E402

load_dotenv()


def _iter_records(input_path: Path, image_root: Path) -> list[dict]:
    """Load submission-format records and resolve each to an on-disk image."""
    records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_image_path"] = _resolve_image_path(image_root, row["filename"])
            records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Input JSONL (dev or test format)."
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=list(DEV_LANGUAGES.keys()),
        help="Target language (drives cultural prompt).",
    )
    parser.add_argument(
        "--method",
        default="direct-target",
        choices=list(METHODS.keys()),
        help="Captioning method (default: direct-target).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for both caption and translation stages.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Directory containing the images referenced by `filename`. "
            "Defaults to the directory of --input (matches the dev layout)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent API calls (default: 8)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N examples (smoke test).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    image_root = args.image_root or args.input.parent
    records = _iter_records(args.input, image_root)
    if args.limit is not None:
        records = records[: args.limit]

    # Skip rows already present in the output (resumable).
    done_ids: set[str] = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done_ids.add(json.loads(line)["id"])

    pending = [r for r in records if r["id"] not in done_ids]
    print(
        f"{len(pending)} to caption ({len(done_ids)} already done); "
        f"method={args.method}, model={args.model}, workers={args.workers}"
    )
    if not pending:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Append mode; resumable
    out_lock = threading.Lock()
    client = OpenAI(api_key=api_key)
    caption_fn = METHODS[args.method]
    culture = DEV_LANGUAGES[args.language]

    def _process(rec: dict) -> dict:
        try:
            target, _spanish = caption_fn(client, args.model, culture, rec["_image_path"])
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['id']}] ERROR: {e}", file=sys.stderr)
            target = ""
        # Drop our internal `_image_path` and add `predicted_caption`.
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        out["predicted_caption"] = target
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process, rec): rec["id"] for rec in pending}
        done_count = 0
        for fut in as_completed(futures):
            out = fut.result()
            done_count += 1
            preview = out["predicted_caption"].replace("\n", " ")[:60]
            print(f"[{done_count}/{len(pending)}] {out['id']}: {preview!r}")
            with out_lock:
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nWrote submission to {args.output}")


if __name__ == "__main__":
    main()
