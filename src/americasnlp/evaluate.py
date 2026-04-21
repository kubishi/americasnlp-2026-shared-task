"""Run a captioner over a split and (when references exist) score it with ChrF++.

This is the dev-loop driver. For the test split, prefer `submit.py` — it
writes a clean submission JSONL and skips scoring entirely.
"""
from __future__ import annotations

import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sacrebleu.metrics.chrf import CHRF

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


@dataclass
class EvalResult:
    rows: List[dict]
    mean_chrf: float
    corpus_chrf: float
    n_scored: int
    mean_comet: Optional[float] = None


def _chrf(hyp: str, ref: str) -> float:
    return CHRF(word_order=2).corpus_score([hyp], [[ref]]).score


def _try_score_comet(scored_rows: List[dict]) -> Optional[List[float]]:
    """Score predictions with reference-based COMET. Returns per-row scores or None.

    Lazy-imports `comet` so the dep is truly optional. Uses the captioner's
    intermediate English caption (`english_intermediate`) as the source when
    present (pipeline method); falls back to the reference itself when absent
    (direct baselines), with a one-line warning. The metric is for our
    diagnostic use only — these languages aren't in COMET's training
    distribution, so treat absolute scores cautiously and trust deltas more
    than levels.
    """
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except ImportError:
        print("  COMET not installed \u2014 `uv sync --extra comet` to enable",
              file=sys.stderr)
        return None
    try:
        ckpt = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(ckpt)
    except Exception as exc:  # noqa: BLE001
        print(f"  COMET model load failed: {exc}", file=sys.stderr)
        return None

    n_with_english = sum(1 for r in scored_rows if r.get("english_intermediate"))
    if n_with_english == 0:
        print("  COMET: no `english_intermediate` available; using reference as "
              "source proxy. Scores will be optimistic.", file=sys.stderr)
    elif n_with_english < len(scored_rows):
        print(f"  COMET: {n_with_english}/{len(scored_rows)} rows have "
              "`english_intermediate`; mixing real source with ref-as-source "
              "fallback for the rest.", file=sys.stderr)

    data = [{
        "src": r.get("english_intermediate") or r["target_caption"],
        "mt":  r["predicted_caption"],
        "ref": r["target_caption"],
    } for r in scored_rows]
    try:
        out = model.predict(data, batch_size=8, gpus=0, progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  COMET scoring failed: {exc}", file=sys.stderr)
        return None
    return list(out.scores)


def evaluate(
    *,
    captioner: Captioner,
    lang: LanguageConfig,
    split: str,
    data_root: Path,
    output_jsonl: Path,
    output_csv: Optional[Path] = None,
    workers: int = 8,
    limit: Optional[int] = None,
    score_comet: bool = False,
    val_only: bool = False,
    train_frac: float = 0.6,
) -> EvalResult:
    """Caption the split, append to JSONL incrementally, optionally score with ChrF++.

    JSONL is appended to as soon as each prediction completes — re-running the
    same command resumes from the previous output without redoing API calls.

    `val_only=True` restricts scoring to the held-out validation rows (the
    rows the generator agent never saw). Use this for honest measurement
    of an agent-authored package.
    """
    records = load_split(lang, split, data_root)
    if val_only:
        from americasnlp.generator.split import split_dev
        s = split_dev(lang, data_root, train_frac=train_frac)
        records = [r for r in records if r["id"] in s.val]
        print(f"[{lang.key}/{split}] val-only: scoring {len(records)} held-out rows",
              file=sys.stderr)
    if limit is not None:
        records = records[:limit]
    base_dir = split_dir(lang, split, data_root)

    done = existing_predictions(output_jsonl)
    pending = [r for r in records if r["id"] not in done]
    print(f"[{lang.key}/{split}/{captioner.name}] {len(pending)} pending  "
          f"({len(done)} resumed)  workers={workers}", file=sys.stderr)

    rows_by_id: dict = {}
    write_lock = threading.Lock()

    def _process(rec: dict) -> dict:
        english_intermediate: Optional[str] = None
        try:
            result = captioner.caption(rec, resolve_image_path(rec, base_dir))
            pred = result.target
            english_intermediate = result.english_intermediate
        except Exception as exc:  # noqa: BLE001
            print(f"[{rec['id']}] ERROR: {exc}", file=sys.stderr)
            pred = ""
        out = submission_row(rec, pred)
        if english_intermediate:
            out["english_intermediate"] = english_intermediate
        if rec.get("target_caption"):
            out["target_caption"] = rec["target_caption"]
            out["chrf"] = _chrf(pred, rec["target_caption"]) if pred else 0.0
        with write_lock:
            append_jsonl(output_jsonl, out)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, rec): rec["id"] for rec in pending}
        for fut in as_completed(futures):
            row = fut.result()
            rows_by_id[row["id"]] = row
            preview = (row.get("predicted_caption") or "").replace("\n", " ")[:60]
            chrf = row.get("chrf")
            chrf_s = f"chrf={chrf:5.2f}" if chrf is not None else "chrf=    -"
            print(f"  {row['id']}  {chrf_s}  {preview!r}", file=sys.stderr)

    # Re-load everything from the JSONL so resumed rows are included in the summary.
    from americasnlp.data import load_jsonl
    rows = load_jsonl(output_jsonl)
    rows.sort(key=lambda r: r.get("id", ""))

    scored = [r for r in rows if r.get("target_caption") and r.get("predicted_caption")]
    mean_chrf = sum(r.get("chrf", 0.0) for r in scored) / len(scored) if scored else 0.0
    corpus_chrf = 0.0
    if scored:
        chrf = CHRF(word_order=2)
        corpus_chrf = chrf.corpus_score(
            [r["predicted_caption"] for r in scored],
            [[r["target_caption"] for r in scored]],
        ).score

    mean_comet: Optional[float] = None
    comet_scores: Optional[List[float]] = None
    if score_comet and scored:
        print("  scoring with COMET (Unbabel/wmt22-comet-da)...", file=sys.stderr)
        comet_scores = _try_score_comet(scored)
        if comet_scores is not None:
            for r, s in zip(scored, comet_scores):
                r["comet"] = float(s)
            mean_comet = sum(comet_scores) / len(comet_scores)

    if output_csv is not None and scored:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["id", "predicted_caption", "target_caption", "chrf"]
        if mean_comet is not None:
            cols.append("comet")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                row = [
                    r.get("id", ""),
                    r.get("predicted_caption", ""),
                    r.get("target_caption", ""),
                    f"{r.get('chrf', 0.0):.4f}",
                ]
                if mean_comet is not None:
                    row.append(f"{r.get('comet', 0.0):.4f}")
                w.writerow(row)

    print()
    print("=" * 60, file=sys.stderr)
    print(f"  language={lang.key}  split={split}  method={captioner.name}",
          file=sys.stderr)
    if scored:
        line = (f"  N={len(scored)}  mean ChrF++={mean_chrf:.2f}  "
                f"corpus ChrF++={corpus_chrf:.2f}")
        if mean_comet is not None:
            line += f"  mean COMET={mean_comet:.4f}"
        print(line, file=sys.stderr)
    else:
        print("  no scored rows (test split or no references)", file=sys.stderr)
    print(f"  jsonl: {output_jsonl}", file=sys.stderr)
    if output_csv:
        print(f"  csv:   {output_csv}", file=sys.stderr)

    return EvalResult(
        rows=rows,
        mean_chrf=mean_chrf,
        corpus_chrf=corpus_chrf,
        n_scored=len(scored),
        mean_comet=mean_comet,
    )
