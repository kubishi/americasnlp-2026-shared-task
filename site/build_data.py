"""Build site/public/data/results.json from results/dev/*.jsonl files.

Walks every (language, configuration) JSONL in results/dev/, computes
per-language mean ChrF++, and emits one row per dev sample with all
predictions side-by-side. Image filenames are normalized to
`images/<lang>/<basename>` (Guaraní's `data/<lang>/images/...` quirk
handled here so the explorer just uses one path convention).

The site loads the resulting JSON at runtime — keep it lean.

Also materializes the dev images under `site/public/data/images/<lang>/`
as real files (not symlinks). Wrangler's symlink handling is unreliable;
real files keep the deploy deterministic. The image dir is gitignored.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "dev"
SUBMISSIONS = ROOT / "results" / "submissions"
DEV_DATA = ROOT / "americasnlp2026" / "data" / "dev"
TEST_DATA = ROOT / "americasnlp2026" / "data" / "test"
SITE_IMAGES = ROOT / "site" / "public" / "data" / "images"
SITE_TEST_IMAGES = ROOT / "site" / "public" / "data" / "test-images"
OUT = ROOT / "site" / "public" / "data" / "results.json"

# The single config we used for the test-set submission. The .rich.jsonl
# file alongside the clean submission file has back_translation +
# structured_json preserved; we surface it on the site so reviewers can
# see what each test prediction "means" in English without seeing gold.
SUBMISSION_RICH_FNAME = "{lang}_one-step_gpt-5.rich.jsonl"
SUBMISSION_LABEL = "gpt-5 one-step (test submission)"


# Display order for languages (matches the README headline tables).
LANGUAGES = [
    {"key": "bribri",   "iso": "bzd", "name": "Bribri",       "test_rows": 267},
    {"key": "guarani",  "iso": "grn", "name": "Guaraní",      "test_rows": 110},
    {"key": "maya",     "iso": "yua", "name": "Yucatec Maya", "test_rows": 212},
    {"key": "nahuatl",  "iso": "nlv", "name": "Orizaba Nahuatl", "test_rows": 200},
    {"key": "wixarika", "iso": "hch", "name": "Wixárika",     "test_rows": 201},
]


# Order matters — dictates table column order. `kind` controls the row marker
# in the aggregate table; `primary` selects the default in the explorer.
@dataclass
class ConfigSpec:
    id: str           # filename stem (without `<lang>_dev_` prefix)
    label: str
    method: str       # pipeline | direct | one-step
    model: str
    short: str        # short label for the agg-table column header
    kind: str         # ours | ours-local | baseline
    primary: bool = False


CONFIGS: list[ConfigSpec] = [
    # 2026-04-30/05-01 sweeps with the new strict-Literal schema,
    # hch CopularSentence removed, and proper_noun field.
    # The submission config is one-step gpt-5 with the minimal prompt
    # (v3) + structured-form back-translation via gpt-4o-mini.
    ConfigSpec("one-step-v3-minimal_gpt-5",
               "One-step · gpt-5 (minimal prompt — submission)",
               "one-step", "gpt-5",
               "gpt-5 v3", "ours", primary=True),
    ConfigSpec("one-step-v2_gpt-5",
               "One-step · gpt-5 (verbose prompt v2)",
               "one-step", "gpt-5",
               "gpt-5 v2", "ours"),
    ConfigSpec("pipeline_schema-v3_claude_vlm__gpt-4o_translator",
               "Pipeline · claude-vl + gpt-4o translator",
               "pipeline", "claude-sonnet-4-5+gpt-4o",
               "snt+4o", "ours"),
    ConfigSpec("pipeline_schema-v3_claude_vlm__gpt-4o-mini_translator",
               "Pipeline · claude-vl + gpt-4o-mini translator",
               "pipeline", "claude-sonnet-4-5+gpt-4o-mini",
               "snt+mini", "ours"),
    ConfigSpec("pipeline_schema-v3_claude_vlm__gpt-5_translator",
               "Pipeline · claude-vl + gpt-5 translator",
               "pipeline", "claude-sonnet-4-5+gpt-5",
               "snt+gpt5", "ours"),
    ConfigSpec("direct-shots3_claude-sonnet-4-5",
               "Direct 3-shot · claude-sonnet-4-5",
               "direct", "claude-sonnet-4-5",
               "direct-3", "baseline"),
    ConfigSpec("organizer_baseline",
               "Organizer baseline · Qwen3-VL → NLLB",
               "baseline", "qwen3-vl+nllb",
               "org", "baseline"),
]


# Organizer-baseline ChrF++ per language, sourced from the shared-task
# baseline runs. The organizer's NLLB pipeline doesn't cover yua, so it's
# left None there. We don't have per-row predictions from the organizer,
# so this config only contributes to the aggregate table — there are no
# samples in the explorer for it.
ORGANIZER_BASELINE = {
    "bribri":   7.57,
    "guarani":  20.82,
    "maya":     None,
    "nahuatl":  11.53,
    "wixarika": 17.77,
}


def normalize_image_path(filename: str, lang_key: str) -> str:
    """Return `images/<lang>/<basename>` regardless of source convention.

    Guaraní stores `data/guarani/images/grn_001.jpg`; everyone else
    stores `images/<id>.<ext>`. The site serves everything under
    `data/images/<lang>/` (symlinks point at the dev image dirs).
    """
    name = filename.split("/")[-1]
    return f"data/images/{lang_key}/{name}"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sync_image_dir(src_root: Path, dst_root: Path) -> int:
    """Copy each language's images from src_root/<lang>/images/ into
    dst_root/<lang>/. Idempotent (skip if size matches)."""
    n_copied = 0
    dst_root.mkdir(parents=True, exist_ok=True)
    for lang_key in (l["key"] for l in LANGUAGES):
        src_dir = src_root / lang_key / "images"
        dst_dir = dst_root / lang_key
        if dst_dir.is_symlink():
            dst_dir.unlink()
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            continue
        for src in src_dir.iterdir():
            if not src.is_file():
                continue
            dst = dst_dir / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dst)
            n_copied += 1
    return n_copied


def sync_images() -> int:
    """Sync dev + test image dirs into site/public/data/."""
    return (_sync_image_dir(DEV_DATA, SITE_IMAGES)
            + _sync_image_dir(TEST_DATA, SITE_TEST_IMAGES))


def main() -> int:
    # samples[lang_key][record_id] = {id, image, target_caption, predictions: {...}}
    samples: dict[str, dict[str, dict]] = defaultdict(dict)
    # aggregate[lang_key][config_id] = {n, mean_chrf}
    aggregate: dict[str, dict[str, dict]] = defaultdict(dict)

    n_loaded = 0
    for lang in LANGUAGES:
        for cfg in CONFIGS:
            stem = f"{lang['key']}_dev_{cfg.id}"
            path = RESULTS / f"{stem}.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            n_loaded += 1
            chrfs: list[float] = []
            for r in rows:
                rid = r.get("id")
                if not rid:
                    continue
                entry = samples[lang["key"]].setdefault(rid, {
                    "id": rid,
                    "image": normalize_image_path(r["filename"], lang["key"]),
                    "target_caption": r.get("target_caption", ""),
                    "predictions": {},
                })
                # Re-set target_caption if it was empty before (some rows
                # only have it on certain runs — pipeline always sets it).
                if not entry["target_caption"] and r.get("target_caption"):
                    entry["target_caption"] = r["target_caption"]
                pred = {
                    "caption": r.get("predicted_caption", ""),
                    "chrf": round(float(r.get("chrf", 0.0)), 3) if r.get("chrf") is not None else None,
                }
                if r.get("english_intermediate"):
                    pred["english"] = r["english_intermediate"]
                if r.get("back_translation"):
                    pred["back"] = r["back_translation"]
                entry["predictions"][cfg.id] = pred
                if r.get("chrf") is not None and r.get("predicted_caption"):
                    chrfs.append(float(r["chrf"]))
            if chrfs:
                aggregate[lang["key"]][cfg.id] = {
                    "n": len(chrfs),
                    "mean_chrf": round(mean(chrfs), 3),
                }
            print(f"  loaded {stem}.jsonl  n={len(rows)}  "
                  f"mean_chrf={aggregate[lang['key']].get(cfg.id, {}).get('mean_chrf', '—')}")

    # Inject the organizer-baseline numbers (no JSONL — just hard-coded
    # per-language means from the shared task's published baseline runs).
    for lang_key, score in ORGANIZER_BASELINE.items():
        if score is None:
            continue
        aggregate[lang_key]["organizer_baseline"] = {
            "n": 50,
            "mean_chrf": round(score, 3),
        }

    # Flatten samples to a list, sorted by (lang_order, id).
    lang_order = {l["key"]: i for i, l in enumerate(LANGUAGES)}
    samples_list: list[dict] = []
    for lang_key, by_id in samples.items():
        for rid, entry in by_id.items():
            entry["language"] = lang_key
            samples_list.append(entry)
    samples_list.sort(key=lambda e: (lang_order.get(e["language"], 99), e["id"]))

    # Sort configs once for the payload.
    configs_payload = [
        {
            "id": c.id,
            "label": c.label,
            "method": c.method,
            "model": c.model,
            "short": c.short,
            "kind": c.kind,
            "primary": c.primary,
        }
        for c in CONFIGS
    ]

    # Headline averages — per-config mean of per-language mean ChrF (over the
    # set of languages where that config has data). Mirrors the headline
    # numbers in README.md, except it's computed from per-row chrF rather
    # than corpus chrF, so values may differ by ~0.2 point.
    headline: list[dict] = []
    for cfg in CONFIGS:
        means = [
            aggregate[l["key"]].get(cfg.id, {}).get("mean_chrf")
            for l in LANGUAGES
        ]
        means = [m for m in means if m is not None]
        if means:
            headline.append({
                "config_id": cfg.id,
                "n_langs": len(means),
                "mean_chrf": round(mean(means), 3),
            })

    # Test-set submission predictions (one row per test sample, one config:
    # gpt-5 one-step + gpt-4o-mini back-translation).
    test_predictions: list[dict] = []
    for lang in LANGUAGES:
        rich = SUBMISSIONS / SUBMISSION_RICH_FNAME.format(lang=lang["key"])
        if not rich.exists():
            print(f"  skip test predictions for {lang['key']}: {rich.name} not found")
            continue
        for r in load_jsonl(rich):
            # The site puts test images under data/test-images/<lang>/<file>;
            # filename in the JSONL is `images/<id>.<ext>` so we just take
            # the basename.
            fname = r["filename"].split("/")[-1]
            test_predictions.append({
                "id": r["id"],
                "language": lang["key"],
                "image": f"data/test-images/{lang['key']}/{fname}",
                "predicted_caption": r.get("predicted_caption", ""),
                "back_translation": r.get("back_translation", ""),
            })
    test_predictions.sort(key=lambda e: (lang_order.get(e["language"], 99), e["id"]))

    payload = {
        "languages": LANGUAGES,
        "configs": configs_payload,
        "aggregate": aggregate,
        "headline": headline,
        "samples": samples_list,
        "submission_label": SUBMISSION_LABEL,
        "test_predictions": test_predictions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=None, separators=(",", ":")))

    size_kb = OUT.stat().st_size / 1024
    print(f"\n{n_loaded} JSONL files loaded → {len(samples_list)} dev samples, "
          f"{len(test_predictions)} test predictions")
    print(f"wrote {OUT.relative_to(ROOT)}: {size_kb:.0f} KB")

    n_copied = sync_images()
    print(f"copied {n_copied} new images into "
          f"site/public/data/{{images,test-images}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
