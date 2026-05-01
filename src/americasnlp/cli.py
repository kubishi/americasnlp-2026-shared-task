"""`python -m americasnlp ...` entrypoint.

Subcommands:
    evaluate  — run a captioner on dev/pilot, write JSONL+CSV, report ChrF++.
    submit    — run a captioner on test, write a clean submission JSONL.
    list      — list available languages and methods.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from americasnlp.captioners import Captioner
from americasnlp.evaluate import evaluate
from americasnlp.languages import LANGUAGES
from americasnlp.submit import make_submission


def _make_captioner(args: argparse.Namespace) -> Captioner:
    lang = LANGUAGES[args.language]
    if args.method == "pipeline":
        from americasnlp.captioners.pipeline import PipelineCaptioner
        return PipelineCaptioner(
            lang=lang,
            vlm_model=args.vlm,
            translator_model=args.translator or args.vlm,
        )
    if args.method == "one-step":
        from americasnlp.captioners.one_step import OneStepCaptioner
        return OneStepCaptioner(
            lang=lang,
            vlm_model=args.vlm,
            back_translator_model=args.translator,  # reused as BT slot for one-step
        )
    if args.method == "direct":
        from americasnlp.captioners.direct import DirectCaptioner
        return DirectCaptioner(
            lang=lang,
            data_root=args.data_root,
            vlm_model=args.vlm,
            shots=args.shots,
            seed=args.seed,
        )
    raise ValueError(f"unknown method: {args.method!r}")


def _default_output(*, kind: str, args: argparse.Namespace) -> Path:
    """Default {dev,submission} path conventions, so users rarely pass --output."""
    method_tag = args.method
    if args.method == "direct":
        method_tag = f"direct-shots{args.shots}"
    if kind == "dev":
        suffix = "_val" if getattr(args, "val_only", False) else ""
        return (args.output_root / "dev"
                / f"{args.language}_{args.split}{suffix}_{method_tag}_{args.vlm}.jsonl")
    return args.output_root / "submissions" / f"{args.language}_{method_tag}_{args.vlm}.jsonl"


def cmd_evaluate(args: argparse.Namespace) -> None:
    captioner = _make_captioner(args)
    output_jsonl = args.output or _default_output(kind="dev", args=args)
    output_csv = output_jsonl.with_suffix(".csv")
    evaluate(
        captioner=captioner,
        lang=LANGUAGES[args.language],
        split=args.split,
        data_root=args.data_root,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        workers=args.workers,
        limit=args.limit,
        score_comet=getattr(args, "comet", False),
        val_only=getattr(args, "val_only", False),
        train_frac=getattr(args, "train_frac", 0.6),
    )


def cmd_submit(args: argparse.Namespace) -> None:
    captioner = _make_captioner(args)
    output = args.output or _default_output(kind="submission", args=args)
    make_submission(
        captioner=captioner,
        lang=LANGUAGES[args.language],
        split=args.split,
        data_root=args.data_root,
        output=output,
        workers=args.workers,
        limit=args.limit,
    )


def cmd_list(_: argparse.Namespace) -> None:
    print("languages:")
    for key, lang in LANGUAGES.items():
        print(f"  {key:10s}  iso={lang.iso}  name={lang.name}")
    print("\nmethods:")
    print("  pipeline                 (LLM-RBMT — proposed system)")
    print("  direct --shots K         (zero/few-shot direct VLM prompting baseline)")


def cmd_generate_language(args: argparse.Namespace) -> None:
    from americasnlp.generator.agent import generate_language_package
    run = generate_language_package(
        iso=args.iso,
        repo_root=args.repo_root,
        data_root=args.data_root,
        model=args.model,
        effort=args.effort,
        max_iterations=args.max_iterations,
        overwrite_scaffold=args.overwrite_scaffold,
        train_frac=args.train_frac,
    )
    print(f"\npackage written to: {run.package_root}")
    print(f"iterations: {run.iterations}")
    print(f"validation passed: "
          f"{run.final_validation.passed if run.final_validation else 'unknown'}")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--method", required=True,
                   choices=["pipeline", "direct", "one-step"])
    p.add_argument("--vlm", default="gpt-4o-mini",
                   help="OpenAI vision model (default: gpt-4o-mini)")
    p.add_argument("--translator", default=None,
                   help="OpenAI model for the structured-translation step "
                        "(pipeline method only; defaults to --vlm)")
    p.add_argument("--shots", type=int, default=0,
                   help="Few-shot demonstrations (direct method only; default: 0)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N records (smoke testing)")
    p.add_argument("--data-root", default=Path("americasnlp2026/data"), type=Path)
    p.add_argument("--output-root", default=Path("results"), type=Path)
    p.add_argument("--output", default=None, type=Path,
                   help="Override the default output path")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="americasnlp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="run + score on dev/pilot")
    _add_common_args(p_eval)
    p_eval.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    p_eval.add_argument("--comet", action="store_true",
                        help="Also score with COMET (Unbabel/wmt22-comet-da). "
                             "Requires `uv sync --extra comet`. Slow on CPU.")
    p_eval.add_argument("--val-only", action="store_true",
                        help="Score only the held-out validation rows (rows "
                             "the generator agent never saw). Use this for "
                             "honest measurement of agent-authored packages.")
    p_eval.add_argument("--train-frac", type=float, default=0.6,
                        help="Train/val split fraction used by the generator "
                             "(default: 0.6 → 30 train / 20 val on dev)")
    p_eval.set_defaults(fn=cmd_evaluate)

    p_sub = sub.add_parser("submit", help="produce submission JSONL on test")
    _add_common_args(p_sub)
    p_sub.add_argument("--split", default="test", choices=["test", "dev", "pilot"])
    p_sub.set_defaults(fn=cmd_submit)

    p_list = sub.add_parser("list", help="list languages and methods")
    p_list.set_defaults(fn=cmd_list)

    p_gen = sub.add_parser(
        "generate-language",
        help="run the agent that authors a yaduha-{iso} package")
    p_gen.add_argument("--iso", required=True,
                       choices=sorted({l.iso for l in LANGUAGES.values()}),
                       help="ISO 639-3 code of the language to generate")
    p_gen.add_argument("--repo-root", default=Path("."), type=Path,
                       help="Repo root containing yaduha-* packages")
    p_gen.add_argument("--data-root", default=Path("americasnlp2026/data"),
                       type=Path, help="Path to the dataset root")
    p_gen.add_argument("--model", default="claude-opus-4-7",
                       help="Anthropic model (default: claude-opus-4-7)")
    p_gen.add_argument("--effort",
                       default="high",
                       choices=["low", "medium", "high", "xhigh", "max"],
                       help="output_config.effort (default: high)")
    p_gen.add_argument("--max-iterations", type=int, default=60,
                       help="Max model turns before bailing (default: 60)")
    p_gen.add_argument("--overwrite-scaffold", action="store_true",
                       help="Overwrite an existing skeleton in yaduha-{iso}/. "
                            "Off by default so partial work is preserved.")
    p_gen.add_argument("--train-frac", type=float, default=0.6,
                       help="Fraction of dev rows visible to the agent "
                            "(default: 0.6). Use 1.0 in submission mode "
                            "to let the agent see all dev rows.")
    p_gen.set_defaults(fn=cmd_generate_language)

    args = parser.parse_args()
    try:
        args.fn(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
