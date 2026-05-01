"""Validate an AmericasNLP 2026 test-set submission JSONL.

Implements two checks:

  1. Row-count integrity (T-5 days message): bzd 267, grn 101, yua 212,
     nlv 200, hch 201 — reject silently truncated runs.
  2. Per-row sanity (T-4 days message): no empty `predicted_caption`,
     no encoding/Unicode issues, no `[english_lemma]` placeholder
     leakage from the pipeline captioner.

Plus the structural checks the task README implies but doesn't restate:
schema (exact 7 keys), id coverage vs the test split, byte-for-byte
fidelity of the 6 carry-over fields against the test JSONL.

Usage:
    python scripts/validate_submission.py results/submissions/bribri_pipeline_claude-sonnet-4-5.jsonl
    python scripts/validate_submission.py --all                # scan results/submissions/
    python scripts/validate_submission.py path.jsonl --language bribri
    python scripts/validate_submission.py path.jsonl --samples 10

Exits 0 if every file passes; non-zero (and prints a list) otherwise.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from americasnlp.languages import LANGUAGES, LanguageConfig  # noqa: E402

EXPECTED_COUNTS: Dict[str, int] = {
    "bribri":   267,
    "guarani":  101,  # 2026-04-30: upstream removed 9 duplicate IDs (was 110)
    "maya":     212,
    "nahuatl":  200,
    "wixarika": 201,
}

CARRYOVER_FIELDS: Tuple[str, ...] = (
    "id", "filename", "split", "culture", "language", "iso_lang",
)
EXPECTED_KEYS = set(CARRYOVER_FIELDS) | {"predicted_caption"}

PLACEHOLDER_RE = re.compile(r"\[[a-z][a-z0-9_]*\]")
REPLACEMENT_CHAR = "\ufffd"
MAX_REASONABLE_LEN = 2000


@dataclass
class Report:
    path: Path
    language: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    samples: List[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {ln}: invalid JSON ({exc})") from exc
    return rows


def _guess_language(path: Path) -> Optional[str]:
    """Filename convention is `{lang}_{method}_{vlm}.jsonl`."""
    stem = path.stem
    head = stem.split("_", 1)[0]
    return head if head in LANGUAGES else None


def _test_split_path(lang: LanguageConfig, data_root: Path) -> Path:
    return lang.test_dir(data_root) / f"{lang.key}.jsonl"


def validate_file(
    path: Path,
    *,
    language: Optional[str] = None,
    data_root: Path = REPO_ROOT / "americasnlp2026" / "data",
    samples: int = 5,
    seed: int = 17,
) -> Report:
    report = Report(path=path)

    if not path.exists():
        report.errors.append(f"file not found: {path}")
        return report

    lang_key = language or _guess_language(path)
    if lang_key is None:
        report.errors.append(
            f"could not infer language from filename {path.name!r}; "
            f"pass --language explicitly (one of {sorted(LANGUAGES)})"
        )
        return report
    if lang_key not in LANGUAGES:
        report.errors.append(
            f"unknown language {lang_key!r}; expected one of {sorted(LANGUAGES)}"
        )
        return report
    report.language = lang_key
    lang = LANGUAGES[lang_key]

    try:
        sub_rows = _load_jsonl(path)
    except ValueError as exc:
        report.errors.append(f"could not parse submission: {exc}")
        return report
    report.n_rows = len(sub_rows)

    test_path = _test_split_path(lang, data_root)
    if not test_path.exists():
        report.errors.append(f"test split not found at {test_path}")
        return report
    test_rows = _load_jsonl(test_path)
    test_by_id = {r["id"]: r for r in test_rows}

    expected = EXPECTED_COUNTS[lang_key]
    if len(test_by_id) != expected:
        report.warnings.append(
            f"local test split has {len(test_by_id)} rows but expected "
            f"{expected}; using whatever is on disk"
        )
    if len(sub_rows) != len(test_by_id):
        report.errors.append(
            f"row count mismatch: submission={len(sub_rows)}, "
            f"test={len(test_by_id)} (expected {expected})"
        )

    sub_ids = [r.get("id") for r in sub_rows]
    sub_id_set = set(sub_ids)
    if len(sub_id_set) != len(sub_ids):
        dupes = [i for i in sub_ids if sub_ids.count(i) > 1]
        report.errors.append(
            f"{len(set(dupes))} duplicate id(s) in submission, e.g. "
            f"{sorted(set(dupes))[:5]}"
        )
    missing = sorted(set(test_by_id) - sub_id_set)
    extra = sorted(sub_id_set - set(test_by_id))
    if missing:
        report.errors.append(
            f"{len(missing)} id(s) from test split missing in submission, "
            f"e.g. {missing[:5]}"
        )
    if extra:
        report.errors.append(
            f"{len(extra)} id(s) in submission not present in test split, "
            f"e.g. {extra[:5]}"
        )

    bad_keys: List[str] = []
    fidelity_mismatches: List[str] = []
    empty_pred_ids: List[str] = []
    non_str_pred_ids: List[str] = []
    multiline_pred_ids: List[str] = []
    long_pred_ids: List[str] = []
    placeholder_hits: List[Tuple[str, str]] = []
    replacement_char_ids: List[str] = []

    for row in sub_rows:
        rid = row.get("id", "<no id>")
        keys = set(row.keys())
        if keys != EXPECTED_KEYS:
            missing_k = sorted(EXPECTED_KEYS - keys)
            extra_k = sorted(keys - EXPECTED_KEYS)
            bad_keys.append(f"{rid}: missing={missing_k} extra={extra_k}")
            continue

        ref = test_by_id.get(rid)
        if ref is not None:
            for k in CARRYOVER_FIELDS:
                if row.get(k) != ref.get(k):
                    fidelity_mismatches.append(
                        f"{rid}.{k}: submission={row.get(k)!r} "
                        f"test={ref.get(k)!r}"
                    )
                    break

        pred = row.get("predicted_caption")
        if not isinstance(pred, str):
            non_str_pred_ids.append(rid)
            continue
        if not pred.strip():
            empty_pred_ids.append(rid)
            continue
        if "\n" in pred or "\t" in pred or "\r" in pred:
            multiline_pred_ids.append(rid)
        if len(pred) > MAX_REASONABLE_LEN:
            long_pred_ids.append(rid)
        if REPLACEMENT_CHAR in pred:
            replacement_char_ids.append(rid)
        m = PLACEHOLDER_RE.search(pred)
        if m:
            placeholder_hits.append((rid, m.group(0)))

    if bad_keys:
        report.errors.append(
            f"{len(bad_keys)} row(s) have wrong key set, e.g. {bad_keys[:3]}"
        )
    if fidelity_mismatches:
        report.errors.append(
            f"{len(fidelity_mismatches)} row(s) altered carry-over fields "
            f"vs test split, e.g. {fidelity_mismatches[:3]}"
        )
    if non_str_pred_ids:
        report.errors.append(
            f"{len(non_str_pred_ids)} row(s) have non-string "
            f"predicted_caption, e.g. {non_str_pred_ids[:5]}"
        )
    if empty_pred_ids:
        report.errors.append(
            f"{len(empty_pred_ids)} row(s) have empty predicted_caption, "
            f"e.g. {empty_pred_ids[:5]}"
        )
    if replacement_char_ids:
        report.errors.append(
            f"{len(replacement_char_ids)} row(s) contain Unicode replacement "
            f"char (U+FFFD), e.g. {replacement_char_ids[:5]}"
        )
    if placeholder_hits:
        sample = [f"{rid}:{m}" for rid, m in placeholder_hits[:5]]
        report.errors.append(
            f"{len(placeholder_hits)} row(s) appear to leak [english_lemma] "
            f"placeholders, e.g. {sample}"
        )
    if multiline_pred_ids:
        report.warnings.append(
            f"{len(multiline_pred_ids)} row(s) contain newline/tab/CR in "
            f"predicted_caption, e.g. {multiline_pred_ids[:5]}"
        )
    if long_pred_ids:
        report.warnings.append(
            f"{len(long_pred_ids)} row(s) have predicted_caption longer than "
            f"{MAX_REASONABLE_LEN} chars, e.g. {long_pred_ids[:5]}"
        )

    rng = random.Random(seed)
    pool = [r for r in sub_rows if r.get("predicted_caption")]
    n = min(samples, len(pool))
    report.samples = rng.sample(pool, n) if n else []

    return report


def _print_report(rep: Report) -> None:
    head = f"{rep.path}  [{rep.language or '?'}]"
    print(head)
    print("-" * len(head))
    expected = EXPECTED_COUNTS.get(rep.language or "", "?")
    print(f"  rows: {rep.n_rows} (expected {expected})")

    if rep.errors:
        print("  ERRORS:")
        for e in rep.errors:
            print(f"    - {e}")
    if rep.warnings:
        print("  warnings:")
        for w in rep.warnings:
            print(f"    - {w}")

    if rep.samples:
        print(f"  sample rows ({len(rep.samples)}):")
        for s in rep.samples:
            pred = (s.get("predicted_caption") or "").replace("\n", " \\n ")
            print(f"    {s.get('id')}  {pred[:140]}"
                  + ("..." if len(pred) > 140 else ""))

    print(f"  result: {'PASS' if rep.ok else 'FAIL'}")
    print()


def _gather_paths(args: argparse.Namespace) -> List[Path]:
    if args.all:
        sub_dir = REPO_ROOT / "results" / "submissions"
        if not sub_dir.exists():
            print(f"error: {sub_dir} does not exist (Diego hasn't pushed yet?)",
                  file=sys.stderr)
            sys.exit(2)
        paths = sorted(sub_dir.glob("*.jsonl"))
        if not paths:
            print(f"error: no .jsonl files in {sub_dir}", file=sys.stderr)
            sys.exit(2)
        return paths
    if not args.path:
        print("error: pass a path or --all", file=sys.stderr)
        sys.exit(2)
    return [Path(p) for p in args.path]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="*", help="submission JSONL(s) to validate")
    p.add_argument("--all", action="store_true",
                   help="validate every .jsonl under results/submissions/")
    p.add_argument("--language", default=None,
                   help="override language inference (e.g. bribri)")
    p.add_argument("--data-root",
                   default=REPO_ROOT / "americasnlp2026" / "data",
                   type=Path,
                   help="path to americasnlp2026/data")
    p.add_argument("--samples", type=int, default=5,
                   help="number of random rows to print (default: 5)")
    p.add_argument("--seed", type=int, default=17,
                   help="RNG seed for sampling (default: 17)")
    args = p.parse_args()

    paths = _gather_paths(args)
    reports = [
        validate_file(
            path,
            language=args.language,
            data_root=args.data_root,
            samples=args.samples,
            seed=args.seed,
        )
        for path in paths
    ]
    for rep in reports:
        _print_report(rep)

    failed = [r for r in reports if not r.ok]
    print(f"summary: {len(reports) - len(failed)}/{len(reports)} passed")
    if failed:
        print("failed: " + ", ".join(str(r.path.name) for r in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
