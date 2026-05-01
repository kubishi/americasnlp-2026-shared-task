"""Measure per-record Anthropic token usage and project full submission cost.

Patches `anthropic.Anthropic.messages.create` so every API call's `usage`
field is recorded against a stage label (vlm, fwd, back). Runs the pipeline
captioner on N dev records per language and aggregates totals. Then prints:

  - per-call breakdown (input / output / cache reads / cache writes)
  - per-record averages by stage
  - projected USD cost for the full test split using current Sonnet 4.5
    pricing (input $3/MTok, output $15/MTok, cache write $3.75/MTok,
    cache read $0.30/MTok)

Run:
    uv run python scripts/probe_token_costs.py --per-lang 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# Sonnet 4.5 pricing (USD per 1M tokens) as of 2026-04.
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE_5M = 3.75
PRICE_CACHE_READ = 0.30


# Test-split row counts (from `wc -l` on each test JSONL).
TEST_ROWS = {
    "bribri": 267,
    "guarani": 110,
    "maya": 212,
    "nahuatl": 200,
    "wixarika": 201,
}


@dataclass
class CallRecord:
    stage: str  # vlm | fwd | back
    record_id: str
    language: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    elapsed_s: float

    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT / 1_000_000
            + self.output_tokens * PRICE_OUTPUT / 1_000_000
            + self.cache_write_tokens * PRICE_CACHE_WRITE_5M / 1_000_000
            + self.cache_read_tokens * PRICE_CACHE_READ / 1_000_000
        )


@dataclass
class Probe:
    records: List[CallRecord] = field(default_factory=list)
    current_stage: str = "fwd"  # the translator stages dominate; default
    current_id: str = "?"
    current_lang: str = "?"

    def add(self, usage: Any, elapsed: float) -> None:
        # `usage` fields per Anthropic SDK message response
        rec = CallRecord(
            stage=self.current_stage,
            record_id=self.current_id,
            language=self.current_lang,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            elapsed_s=elapsed,
        )
        self.records.append(rec)


def install_patch(probe: Probe) -> None:
    """Monkey-patch `anthropic.Anthropic.messages.create` to record usage."""
    import anthropic

    original = anthropic.resources.messages.Messages.create

    def wrapped(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        t0 = time.time()
        resp = original(self, *args, **kwargs)
        try:
            probe.add(resp.usage, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            print(f"  [probe] failed to record usage: {exc}", file=sys.stderr)
        return resp

    anthropic.resources.messages.Messages.create = wrapped  # type: ignore[assignment]


def run(per_lang: int, languages: Optional[List[str]]) -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from americasnlp.captioners.pipeline import PipelineCaptioner
    from americasnlp.data import load_split, resolve_image_path, split_dir
    from americasnlp.languages import LANGUAGES

    probe = Probe()
    install_patch(probe)

    lang_keys = languages or list(LANGUAGES)
    data_root = REPO_ROOT / "americasnlp2026" / "data"

    for key in lang_keys:
        if key not in LANGUAGES:
            print(f"unknown language: {key}", file=sys.stderr)
            continue
        lang = LANGUAGES[key]
        print(f"\n=== {key} ===", file=sys.stderr)
        captioner = PipelineCaptioner(
            lang=lang,
            vlm_model="claude-sonnet-4-5",
            translator_model="claude-sonnet-4-5",
        )
        records = load_split(lang, "dev", data_root)[:per_lang]
        base_dir = split_dir(lang, "dev", data_root)

        for rec in records:
            probe.current_id = rec["id"]
            probe.current_lang = key
            image_path = resolve_image_path(rec, base_dir)
            # We need to label calls by stage. Easiest: count calls before/after.
            n_before = len(probe.records)
            try:
                # Stage 1: VLM (Anthropic call #1 of this record)
                # Stage 2+: forward translator + per-sentence back-translation
                # We'll relabel after the fact since we know the order.
                probe.current_stage = "vlm"  # the first call is always the VLM
                result = captioner.caption(rec, image_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{rec['id']}] ERROR: {exc}", file=sys.stderr)
                continue

            # Relabel: of the calls made for this record, the 1st is VLM,
            # the 2nd is the forward translator (English->SentenceList),
            # and the rest are back-translations (one per sentence).
            new_calls = probe.records[n_before:]
            for i, c in enumerate(new_calls):
                if i == 0:
                    c.stage = "vlm"
                elif i == 1:
                    c.stage = "fwd"
                else:
                    c.stage = "back"

            preview = (result.target or "").replace("\n", " ")[:60]
            print(f"  {rec['id']}  calls={len(new_calls)}  {preview!r}",
                  file=sys.stderr)

    summarize(probe)


def summarize(probe: Probe) -> None:
    if not probe.records:
        print("\nno calls recorded", file=sys.stderr)
        return

    # Aggregate per-call by stage (across all records).
    by_stage: Dict[str, List[CallRecord]] = {"vlm": [], "fwd": [], "back": []}
    for c in probe.records:
        by_stage.setdefault(c.stage, []).append(c)

    # Aggregate per-record (one record = the calls sharing a record_id+language).
    per_rec: Dict[tuple, List[CallRecord]] = {}
    for c in probe.records:
        per_rec.setdefault((c.language, c.record_id), []).append(c)

    print()
    print("=" * 78)
    print("PER-CALL BREAKDOWN (by stage)")
    print("=" * 78)
    print(f"{'stage':<6} {'n':>4} {'avg_in':>8} {'avg_out':>8} "
          f"{'avg_cache_r':>12} {'avg_cache_w':>12} {'avg_$':>10}")
    for stage in ("vlm", "fwd", "back"):
        calls = by_stage.get(stage, [])
        if not calls:
            continue
        n = len(calls)
        avg_in = sum(c.input_tokens for c in calls) / n
        avg_out = sum(c.output_tokens for c in calls) / n
        avg_cr = sum(c.cache_read_tokens for c in calls) / n
        avg_cw = sum(c.cache_write_tokens for c in calls) / n
        avg_cost = sum(c.cost_usd() for c in calls) / n
        print(f"{stage:<6} {n:>4d} {avg_in:>8.0f} {avg_out:>8.0f} "
              f"{avg_cr:>12.0f} {avg_cw:>12.0f} {avg_cost:>10.5f}")

    print()
    print("=" * 78)
    print("PER-RECORD COST (by language)")
    print("=" * 78)
    print(f"{'lang':<10} {'n_rec':>6} {'avg_calls':>10} {'avg_$/rec':>12} "
          f"{'min_$':>9} {'max_$':>9}")

    by_lang: Dict[str, List[float]] = {}
    by_lang_call_count: Dict[str, List[int]] = {}
    for (lang, _rid), calls in per_rec.items():
        total = sum(c.cost_usd() for c in calls)
        by_lang.setdefault(lang, []).append(total)
        by_lang_call_count.setdefault(lang, []).append(len(calls))

    grand_costs: List[float] = []
    for lang, costs in by_lang.items():
        n = len(costs)
        avg = statistics.mean(costs)
        avg_calls = statistics.mean(by_lang_call_count[lang])
        print(f"{lang:<10} {n:>6d} {avg_calls:>10.1f} {avg:>12.5f} "
              f"{min(costs):>9.5f} {max(costs):>9.5f}")
        grand_costs.extend(costs)

    if grand_costs:
        overall = statistics.mean(grand_costs)
        print(f"{'OVERALL':<10} {len(grand_costs):>6d} "
              f"{'':>10} {overall:>12.5f}")

    print()
    print("=" * 78)
    print("PROJECTED TEST-SPLIT COST")
    print("=" * 78)
    print(f"{'lang':<10} {'rows':>6} {'avg_$/rec':>12} {'projected_$':>14}")
    grand_proj = 0.0
    for lang, costs in by_lang.items():
        rows = TEST_ROWS.get(lang, 0)
        avg = statistics.mean(costs)
        proj = avg * rows
        grand_proj += proj
        print(f"{lang:<10} {rows:>6d} {avg:>12.5f} {proj:>14.4f}")
    # Languages we didn't probe: project with overall mean.
    if grand_costs:
        overall = statistics.mean(grand_costs)
        for lang, rows in TEST_ROWS.items():
            if lang not in by_lang:
                proj = overall * rows
                grand_proj += proj
                print(f"{lang:<10} {rows:>6d} {overall:>12.5f} {proj:>14.4f}  "
                      "(extrapolated)")

    print(f"{'TOTAL':<10} {sum(TEST_ROWS.values()):>6d} {'':>12} "
          f"{grand_proj:>14.4f}")
    print()
    print("Pricing assumed (USD per 1M tokens):")
    print(f"  input            ${PRICE_INPUT}")
    print(f"  output           ${PRICE_OUTPUT}")
    print(f"  cache write 5m   ${PRICE_CACHE_WRITE_5M}")
    print(f"  cache read       ${PRICE_CACHE_READ}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=3,
                   help="dev records per language (default 3)")
    p.add_argument("--languages", nargs="*", default=None,
                   help="restrict to these LANGUAGES keys "
                        "(default: all 5 languages)")
    args = p.parse_args()
    run(args.per_lang, args.languages)


if __name__ == "__main__":
    main()
