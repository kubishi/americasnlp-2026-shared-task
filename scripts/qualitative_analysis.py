"""Print best / worst / median samples for qualitative inspection.

Walks `results/dev/<lang>_dev_pipeline_claude-sonnet-4-5.jsonl`, picks a
few representative rows by ChrF (top, bottom, median), and prints
gold / English-intermediate / predicted / back-translation / ChrF
side-by-side. Designed to be eyeballed in a terminal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def show(row: dict, label: str) -> None:
    chrf = row.get("chrf", float("nan"))
    print(f"\n--- {label}  id={row.get('id')}  chrf={chrf:5.2f} ---")
    print(f"GOLD ({len(row.get('target_caption','').split()):>2}w):")
    print(f"  {row.get('target_caption','')}")
    eng = row.get("english_intermediate", "")
    print(f"\nVLM ENGLISH ({len(eng.split()):>2}w):")
    for line in eng.split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    pred = row.get("predicted_caption", "")
    print(f"\nPREDICTED TARGET ({len(pred.split()):>2}w):")
    print(f"  {pred}")
    bt = row.get("back_translation", "")
    if bt:
        print(f"\nBACK-TRANSLATION ({len(bt.split()):>2}w):")
        for line in bt.split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("\n(no back-translation yet)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True)
    p.add_argument("--config", default="pipeline_claude-sonnet-4-5")
    p.add_argument("--n-each", type=int, default=2,
                   help="Number of best, worst, median samples each (default 2)")
    args = p.parse_args()

    path = (REPO_ROOT / "results" / "dev"
            / f"{args.language}_dev_{args.config}.jsonl")
    rows = load(path)
    rows = [r for r in rows if r.get("predicted_caption")
            and r.get("target_caption") and r.get("chrf") is not None]
    rows.sort(key=lambda r: r["chrf"])

    print(f"\n{'=' * 76}")
    print(f" {args.language.upper()} / {args.config}  —  n={len(rows)}")
    print(f" chrF: min={rows[0]['chrf']:.2f}  median={median(r['chrf'] for r in rows):.2f}  max={rows[-1]['chrf']:.2f}")
    print("=" * 76)

    n = args.n_each
    print(f"\n### {n} WORST ###")
    for r in rows[:n]:
        show(r, "WORST")

    mid = len(rows) // 2
    print(f"\n\n### {n} MEDIAN ###")
    for r in rows[mid - n // 2: mid + (n - n // 2)]:
        show(r, "MEDIAN")

    print(f"\n\n### {n} BEST ###")
    for r in rows[-n:]:
        show(r, "BEST")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
