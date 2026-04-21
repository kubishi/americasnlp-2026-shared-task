"""Anthropic-driven agent loop that authors a yaduha-{iso} package.

Hands Claude the training captions, lets it browse the web (server-side
`web_search_20260209` + `web_fetch_20260209`), and exposes file-write +
validate tools so it can iterate. The loop ends on `end_turn` or after a
configurable maximum number of model iterations.

Run via the CLI:

    uv run americasnlp generate-language --iso bzd

The system prompt is cached — reruns reuse the prefix as long as the
training data and references don't change.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from americasnlp.data import (
    image_data_url,
    load_jsonl,
    load_split,
    resolve_image_path,
    split_dir,
)
from americasnlp.generator.extract import (
    extract_content_words,
    load_training_examples,
)
from americasnlp.generator.scaffold import scaffold_package
from americasnlp.generator.split import SplitIds, split_dev
from americasnlp.generator.validate import (
    PackageValidation,
    _reload_package,
    validate_package,
)
from americasnlp.languages import LANGUAGES, LanguageConfig


# ---------------------------------------------------------------------------
# References — what Claude has to mimic
# ---------------------------------------------------------------------------

REFERENCE_PACKAGES = ("yaduha-hch", "yaduha-ovp")

REFERENCE_NOTES_PER_ISO: Dict[str, str] = {
    "bzd": (
        "Bribri (bzd), Costa Rica. Verb-final (SOV) with split-ergative "
        "alignment, rich aspect, tonal marking. Suggested references: "
        "Constenla Umaña (2023), Gramática de la lengua bribri; Jara Murillo "
        "(2018), Diccionario bribri-español; SIL bzd resources."
    ),
    "grn": (
        "Guaraní (grn), Paraguay. Predominantly SVO, agglutinative, with "
        "subject-marking prefixes (a-, re-, o-, ja-, ro-, pe-). References: "
        "Gregores & Suárez (1967); Velázquez-Castillo (2002); Wiktionary "
        "Guaraní lemmas."
    ),
    "yua": (
        "Yucatec Maya (yua), Mexico. Predicate-initial (default VOS, also "
        "VSO/SVO), split-ergative, aspect/mood prefixes (k- IMPV, t- PFV, "
        "h- PFV). References: Bohnemeyer (2002); Lehmann grammar notes; SIL."
    ),
    "nlv": (
        "Orizaba Nahuatl (nlv), Mexico. Polysynthetic, predominantly "
        "verb-initial (VSO/VOS), heavy noun incorporation, "
        "subject-marking (ni-/ti-/ø-/ti-/an-/ø-). References: Andrews "
        "(2003); Launey (2011); Tuggy (1979) on Tetelcingo Nahuatl."
    ),
    "hch": (
        "Wixárika / Huichol (hch), Mexico. Reference implementation in "
        "yaduha-hch; expand vocabulary if regenerating."
    ),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PREFACE = """\
You are authoring a language package for the Yaduha framework. Yaduha
implements LLM-Assisted Rule-Based Machine Translation (LLM-RBMT): English
input is mapped to a Pydantic Sentence schema, then Python code on the
schema renders the target-language string deterministically. You are
producing the Pydantic schema + vocabulary for one language so it plugs
into yaduha's PipelineTranslator.

# Mandatory contract

Your package lives at `yaduha-{iso}/yaduha_{iso}/__init__.py` and
`yaduha-{iso}/yaduha_{iso}/vocab.py`. The pyproject.toml is already
written; do not modify it.

`__init__.py` MUST expose a module-level `language` of type
`yaduha.language.Language`. Match the conventions in the reference
implementations exactly: imports, enum names, Pydantic model names,
sentence-type names, the shape of `__str__()`, the shape of
`get_examples()`. Treat `yaduha-hch/yaduha_hch/__init__.py` as canon.

`vocab.py` MUST export `NOUNS`, `TRANSITIVE_VERBS`, `INTRANSITIVE_VERBS`
as `list[VocabEntry]`. Each `VocabEntry(english=..., target=...)`
target form must trace back to a citation in `vocab.py`'s header
comment — never invent target forms.

# Hard rules

1. Render unknown lemmas as the literal `[english_lemma]`. Do not invent.
2. Every target-language string in the package must come from `vocab.py`
   or be deterministically assembled from morphology rules in
   `__init__.py`. No LLM-generated target text.
3. Sentence renders must NOT contain `[english_lemma]` placeholders for
   lemmas that ARE in your `get_examples()` — your examples must use
   in-vocab lemmas.
4. Use the training-slice captions to ground your vocabulary choices.
   Tokens that appear with frequency ≥ 2 are high-priority.
5. **NO hardcoded shortcuts.** Do not write conditions of the form
   `if english == "...": return "..."`, do not embed image IDs or
   filenames anywhere in the package, do not key renders to specific
   training rows, do not memorize (English-input, target-output) pairs
   in a dict. Every render must be a pure function of the structured
   `Sentence` input. Validation rows are held out specifically to catch
   this kind of overfitting.
6. **Train/val isolation.** You only see training-slice captions via
   `read_training_captions`, `extract_content_words`, and
   `compare_pipeline_to_targets`. Validation rows exist but are
   inaccessible to you on purpose. Don't try to back into them by
   probing tools.

# Sentence inventory: decide based on the data, not by template

The reference implementations (`yaduha-hch`, `yaduha-ovp`) ship two
sentence types: `SubjectVerbSentence` and `SubjectVerbObjectSentence`.
That is a *floor*, not a target. The right inventory is whatever the
training captions actually require — and these are image captions, so
expect frequent:

- copular sentences ("X is Y", "X is a Y") for descriptions of objects
- locative sentences ("X is at/in/on Y") for scenes
- possessive sentences ("X has Y", "X's Y") for relationships
- adjectival modification ("the red X", "the small Y")
- multi-clause / coordination ("X does A and B")

## Recursive sentence types (sentences that contain other sentences)

If `compare_pipeline_to_targets` shows your package consistently losing
relations between clauses — coordinated facts becoming two unconnected
sentences, modifiers vanishing, subordinated clauses being dropped — the
fix is a sentence type whose fields *are themselves Sentences*. Generic
patterns to consider, instantiated however your target language
naturally expresses the relation:

- **Binary coordination.** A type with `left: Sentence`, `right: Sentence`,
  and a small enum of connectives (and / but / because / when / while).
  Each connective renders to whatever particle / clitic / juxtaposition
  the target language uses.
- **Modified noun phrase.** A noun whose `modifier` field is a
  `Sentence` (or smaller `Clause` sub-grammar) — covers relative
  clauses, attributive modification, possessor clauses.
- **Adverbial clause.** A main clause with an embedded subordinate
  `Sentence` and a relation marker (temporal, causal, conditional).

These are *patterns*, not mandates. Adopt the form that fits how your
specific target language actually combines clauses — Bribri's tone +
clause-final particle approach is different from Yucatec's sentential
clitics or Nahuatl's verbal suffixes. The goal is faithful
representation of the recurring patterns you see in training, not a
universal recursive grammar.

Keep recursion bounded — depth-1 (one Sentence inside another) is
usually enough and keeps the structured-output schema tractable. Don't
add a sentence type whose fields can recurse arbitrarily deep.

Conversely: do not add a sentence type that won't be exercised — every
type adds tokens to the schema and thinking load to the translator. If
a pattern is rare, it can fall back to a less-ideal but in-vocabulary
sentence type.

# Schema-size budget

The complete sentence-type schema is sent on every translator request as
the structured-output JSON Schema. That cost is paid per caption at
inference time. Keep the schema tractable:

- Don't add Pydantic enum values you won't use.
- Don't enumerate hundreds of `lemma` literals in Field descriptions —
  the reference uses a compact "Known: x, y, z. If unknown, pass the
  English word as a placeholder." pattern. Match it.
- Aim for a sentence-type set that round-trips on the smoke test (see
  workflow) without producing visible weirdness, *and* whose JSON Schema
  fits comfortably in a few thousand tokens.

If you find yourself adding a 5th or 6th sentence type, ask whether one
of the existing ones could be generalized instead.

# Workflow

Use the tools to research, draft, and iterate. A reasonable loop:

1. Read the reference yaduha-hch package end-to-end (both
   `__init__.py` and `vocab.py`).
2. Read 30+ training captions for this language. Sketch the sentence
   patterns you observe. Decide your sentence-type inventory.
3. Use `extract_content_words`, then `web_search` to gloss the high-
   frequency tokens and pin down word order, person marking, and
   tense/aspect.
4. Write `vocab.py` with citation-grounded entries (≥60 nouns, ≥20
   transitive verbs, ≥20 intransitive verbs is a good starting target;
   more is better when justified).
5. Write `__init__.py` with the sentence types you decided on, plus
   the `language = Language(...)` object.
6. **Smoke-test continuously.** Two complementary tools:
   - `test_translate_english("...")` lets you push hand-built English
     sentences through the package and inspect target + back-translation.
     Use it for targeted probes ("does the locative type fire when I
     write 'the X is on the Y'?").
   - `compare_pipeline_to_targets(n=4)` runs the full image-to-target
     pipeline on N random training rows and shows you `{target,
     vlm_english, predicted}` for each. Use this when you want to see
     what kinds of constructions the actual gold captions contain that
     your package isn't representing — when the predicted strings keep
     dropping a coordinator, flattening a relative clause, or losing a
     modifier across many examples, that's the strongest signal to add
     a recursive sentence type.
7. Call `validate_package` after structural changes. Stop iterating when:
   - `validate_package` returns `PASS: True` with zero placeholder leaks,
   - `compare_pipeline_to_targets` shows reasonable round-trips for
     several distinct caption patterns (predictions resemble the gold
     in structure even if vocabulary diverges), and
   - `test_translate_english` back-translations recover the input intent
     for the constructions you've added.

Do not over-engineer beyond what these checks demand.
"""


# ---------------------------------------------------------------------------
# Custom tool definitions
# ---------------------------------------------------------------------------

CUSTOM_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_reference_files",
        "description": ("List source files in a reference yaduha package "
                        "(`yaduha-hch` or `yaduha-ovp`)."),
        "input_schema": {
            "type": "object",
            "properties": {"package": {"type": "string",
                                       "enum": list(REFERENCE_PACKAGES)}},
            "required": ["package"],
        },
    },
    {
        "name": "read_reference_file",
        "description": ("Read a file from a reference yaduha package. "
                        "`relative_path` is relative to the package root, "
                        "e.g. `yaduha_hch/__init__.py`."),
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {"type": "string",
                            "enum": list(REFERENCE_PACKAGES)},
                "relative_path": {"type": "string"},
            },
            "required": ["package", "relative_path"],
        },
    },
    {
        "name": "read_training_captions",
        "description": (
            "Return up to N (target_caption only) entries from the training "
            "slice. Held-out validation rows are NEVER returned — they are "
            "used to score your finished package, so seeing them would "
            "contaminate the evaluation. Image IDs and filenames are also "
            "withheld so you cannot accidentally hardcode for a specific row."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1,
                                     "maximum": 200}},
            "required": [],
        },
    },
    {
        "name": "extract_content_words",
        "description": (
            "Frequency-rank tokens across the training-slice captions only "
            "(content + function words; you must classify them). Returns "
            "`[[token, count], ...]` sorted by count. Validation captions "
            "are excluded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_freq": {"type": "integer", "minimum": 1, "default": 2},
                "top_n": {"type": "integer", "minimum": 1, "default": 200},
            },
            "required": [],
        },
    },
    {
        "name": "list_package_files",
        "description": "List files currently inside the target yaduha-{iso} package.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_package_file",
        "description": ("Read a file inside the target yaduha-{iso} package. "
                        "`relative_path` is relative to the package root, "
                        "e.g. `yaduha_{iso}/__init__.py`."),
        "input_schema": {
            "type": "object",
            "properties": {"relative_path": {"type": "string"}},
            "required": ["relative_path"],
        },
    },
    {
        "name": "write_package_file",
        "description": ("Overwrite a file inside the target yaduha-{iso} "
                        "package. Use only for `yaduha_{iso}/__init__.py` and "
                        "`yaduha_{iso}/vocab.py`. Do NOT modify pyproject.toml."),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["relative_path", "content"],
        },
    },
    {
        "name": "validate_package",
        "description": ("Run the acceptance check on the target yaduha-{iso} "
                        "package. Returns a structured report with import "
                        "status, sentence-type renders, vocabulary counts, "
                        "and training-token coverage."),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "test_translate_english",
        "description": (
            "Run the in-progress yaduha-{iso} package end-to-end on an "
            "English sentence using the same `PipelineTranslator` the "
            "evaluation pipeline uses. Returns the rendered target-language "
            "string plus a back-translation. Use this to spot-check that "
            "your sentence types handle real captions: mentally translate "
            "a target caption you've read, pass the English in, and judge "
            "whether the output is plausible. If a frequent pattern hits "
            "the wrong sentence type or leaves bracketed placeholders, "
            "that's the signal to add or extend a sentence type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "english_caption": {
                    "type": "string",
                    "description": ("A short English sentence to push through "
                                    "your in-progress package."),
                },
            },
            "required": ["english_caption"],
        },
    },
    {
        "name": "compare_pipeline_to_targets",
        "description": (
            "Pick N random TRAINING-slice rows, run the in-progress package "
            "end-to-end on each (image -> VLM English -> translator -> "
            "target), and return side-by-side `{target, predicted}` pairs. "
            "Image IDs and filenames are intentionally withheld so you "
            "cannot key your renders to specific rows. Use this output to "
            "reason about what grammatical constructions the package is "
            "missing — recurring loss-of-meaning patterns (a relative "
            "clause becoming an unrelated SVO, a coordinated 'X and Y' "
            "splitting into two unconnected clauses, an adjectival "
            "modifier vanishing) are the signal to add a new sentence "
            "type. Validation rows are NEVER used here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 8,
                      "default": 4},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

@dataclass
class GeneratorContext:
    iso: str
    lang: LanguageConfig
    repo_root: Path
    data_root: Path
    package_root: Path
    split: SplitIds  # train_ids visible to agent; val_ids held back
    training_token_freq: List[tuple[str, int]] = field(default_factory=list)


def _safe_relative(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError(f"path {relative_path!r} escapes {root_resolved}")
    return candidate


def _execute_custom_tool(name: str, args: dict, ctx: GeneratorContext) -> str:
    if name == "list_reference_files":
        pkg = ctx.repo_root / args["package"]
        if not pkg.exists():
            return f"ERROR: reference package {args['package']!r} not found"
        skip = {"__pycache__", ".git", ".github"}
        files = sorted(p.relative_to(pkg).as_posix()
                       for p in pkg.rglob("*")
                       if p.is_file()
                       and not any(part in skip for part in p.parts)
                       and not p.name.startswith("."))
        return "\n".join(files)

    if name == "read_reference_file":
        root = ctx.repo_root / args["package"]
        try:
            target = _safe_relative(root, args["relative_path"])
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not target.exists() or not target.is_file():
            return f"ERROR: {args['relative_path']!r} not found in {args['package']}"
        return target.read_text(encoding="utf-8")

    if name == "read_training_captions":
        limit = int(args.get("limit", 60))
        examples = load_training_examples(
            ctx.lang, ctx.data_root, allowed_ids=set(ctx.split.train))
        rows = [{"target_caption": ex.target_caption} for ex in examples[:limit]]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    if name == "extract_content_words":
        min_freq = int(args.get("min_freq", 2))
        top_n = int(args.get("top_n", 200))
        examples = load_training_examples(
            ctx.lang, ctx.data_root, allowed_ids=set(ctx.split.train))
        ranked = extract_content_words(examples, min_freq=min_freq, top_n=top_n)
        return json.dumps([list(t) for t in ranked], ensure_ascii=False)

    if name == "list_package_files":
        if not ctx.package_root.exists():
            return "(package not yet scaffolded)"
        files = sorted(p.relative_to(ctx.package_root).as_posix()
                       for p in ctx.package_root.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        return "\n".join(files) or "(empty)"

    if name == "read_package_file":
        try:
            target = _safe_relative(ctx.package_root, args["relative_path"])
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not target.exists() or not target.is_file():
            return f"ERROR: {args['relative_path']!r} not found"
        return target.read_text(encoding="utf-8")

    if name == "write_package_file":
        rel = args["relative_path"]
        if rel == "pyproject.toml" or rel.endswith("/pyproject.toml"):
            return "ERROR: refusing to modify pyproject.toml"
        try:
            target = _safe_relative(ctx.package_root, rel)
        except ValueError as exc:
            return f"ERROR: {exc}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return f"wrote {rel} ({len(args['content'])} bytes)"

    if name == "validate_package":
        training_tokens = [t for t, _ in ctx.training_token_freq]
        result: PackageValidation = validate_package(
            ctx.iso, training_tokens=training_tokens)
        return result.summary()

    if name == "test_translate_english":
        return _smoke_test_translate(args["english_caption"], ctx)

    if name == "compare_pipeline_to_targets":
        n = int(args.get("n", 4))
        return _compare_pipeline_to_targets(n, ctx)

    return f"ERROR: unknown tool {name!r}"


def _compare_pipeline_to_targets(n: int, ctx: GeneratorContext) -> str:
    """Run the in-progress pipeline on N random training rows; return
    {target, predicted} pairs without IDs/filenames."""
    import random as _random
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("ERROR: ANTHROPIC_API_KEY is not set. The compare tool needs "
                "it to run the pipeline end-to-end.")

    try:
        from americasnlp.captioners.pipeline import PipelineCaptioner
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: PipelineCaptioner import failed: {exc}"

    train_ids = list(ctx.split.train)
    if not train_ids:
        return "ERROR: no training rows available."

    rng = _random.Random()
    sample_ids = rng.sample(train_ids, k=min(n, len(train_ids)))

    # Find the records (across dev + pilot) by id, remembering which split
    # each came from so we can resolve the image path correctly. The dev-set
    # JSONL marks pilot rows as `split: "train"`, which is not a directory we
    # have — so we don't trust the row's `split` field, we use the directory
    # where we actually loaded it from.
    by_id: Dict[str, tuple[dict, str]] = {}
    for split in ("dev", "pilot"):
        try:
            for r in load_split(ctx.lang, split, ctx.data_root):
                by_id[r["id"]] = (r, split)
        except FileNotFoundError:
            continue
    sampled = [(by_id[i][0], by_id[i][1]) for i in sample_ids
               if i in by_id and by_id[i][0].get("target_caption")]

    try:
        captioner = PipelineCaptioner(lang=ctx.lang)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: failed to construct PipelineCaptioner: {exc}"

    out: List[Dict[str, str]] = []
    for rec, split in sampled:
        base = split_dir(ctx.lang, split, ctx.data_root)
        img_path = resolve_image_path(rec, base)
        try:
            result = captioner.caption(rec, img_path)
            predicted = result.target
            english = result.english_intermediate or ""
        except Exception as exc:  # noqa: BLE001
            predicted = f"<ERROR: {type(exc).__name__}: {exc}>"
            english = ""
        out.append({
            "target":          rec["target_caption"],
            "vlm_english":     english,
            "predicted":       predicted,
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


SMOKE_TEST_MODEL = "gpt-4o-mini"


def _smoke_test_translate(english: str, ctx: GeneratorContext) -> str:
    """Run the in-progress yaduha-{iso} package end-to-end on an English caption.

    Always reloads the package first so successive calls reflect edits the
    agent just made. Returns a JSON blob with target + back-translation, or
    a descriptive ERROR string. Uses the same OpenAI model the evaluation
    pipeline uses, so what the agent sees here matches what eval will see.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ("ERROR: OPENAI_API_KEY is not set. The smoke-test tool "
                "needs it to drive the structured-output translation step. "
                "Use `validate_package` instead until the key is available.")

    try:
        _reload_package(f"yaduha_{ctx.iso}")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: package import failed: {type(exc).__name__}: {exc}"

    try:
        from yaduha.agent.openai import OpenAIAgent
        from yaduha.loader import LanguageLoader
        from yaduha.translator.pipeline import PipelineTranslator
    except ImportError as exc:
        return f"ERROR: yaduha import failed: {exc}"

    try:
        language = LanguageLoader.load_language(ctx.iso)
    except Exception as exc:  # noqa: BLE001
        return (f"ERROR: language not loadable via LanguageLoader: {exc}. "
                "Make sure `language = Language(...)` is exported.")

    if not getattr(language, "sentence_types", None):
        return ("ERROR: language has no sentence_types defined. Add at least "
                "one Sentence subclass and pass it to Language(...) before "
                "smoke-testing.")

    translator = PipelineTranslator(
        agent=OpenAIAgent(model=SMOKE_TEST_MODEL, api_key=api_key),
        SentenceType=language.sentence_types,
    )

    try:
        result = translator.translate(english)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: translation crashed: {type(exc).__name__}: {exc}"

    out = {
        "english_input": english,
        "target_output": result.target,
        "back_translation": (result.back_translation.source
                             if result.back_translation else None),
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

@dataclass
class GenerationRun:
    iso: str
    package_root: Path
    iterations: int
    final_validation: Optional[PackageValidation]
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read: int
    total_cache_write: int


def generate_language_package(
    iso: str,
    *,
    repo_root: Path,
    data_root: Path,
    model: str = "claude-opus-4-7",
    effort: str = "high",
    max_iterations: int = 60,
    overwrite_scaffold: bool = False,
    train_frac: float = 0.6,
) -> GenerationRun:
    """Generate a yaduha-{iso} package.

    `train_frac` controls the dev-set train/val split visible to the agent:
    the default 0.6 reserves ~40% of dev rows for held-out validation. Set
    to 1.0 in submission mode to let the agent train on all dev rows.
    """
    if iso not in {l.iso for l in LANGUAGES.values()}:
        raise ValueError(f"unknown ISO {iso!r}")
    lang = next(l for l in LANGUAGES.values() if l.iso == iso)

    # Scaffold first so the agent has somewhere to write.
    package_root = scaffold_package(
        iso=iso,
        display_name=lang.name,
        repo_root=repo_root,
        word_order_hint=REFERENCE_NOTES_PER_ISO.get(iso, ""),
        overwrite=overwrite_scaffold,
    )

    split = split_dev(lang, data_root, train_frac=train_frac)
    examples = load_training_examples(
        lang, data_root, allowed_ids=set(split.train))
    token_freq = extract_content_words(examples, min_freq=2, top_n=300)

    ctx = GeneratorContext(
        iso=iso,
        lang=lang,
        repo_root=repo_root,
        data_root=data_root,
        package_root=package_root,
        split=split,
        training_token_freq=token_freq,
    )

    notes = REFERENCE_NOTES_PER_ISO.get(iso, "")
    user_kickoff = (
        f"Improve the `yaduha-{iso}` package for {lang.name} "
        f"(ISO 639-3 `{iso}`).\n\n"
        f"Notes for this language:\n{notes}\n\n"
        f"You have access to {len(split.train)} training-slice "
        f"(image, target_caption) pairs. {len(split.val)} additional dev "
        f"rows are HELD OUT as the validation set — they will be used to "
        f"score your finished package and you will never see them.\n\n"
        f"Workflow:\n"
        f"  1. Read the current `yaduha_{iso}/__init__.py` and "
        f"`yaduha_{iso}/vocab.py` end-to-end. The package may already "
        f"be in good shape — assess what's there before touching anything. "
        f"Only consult the `yaduha-hch` reference if you need a "
        f"convention check.\n"
        f"  2. Inspect ~30 training captions and decide what's missing. "
        f"Common gaps after a first-pass package: limited vocabulary, "
        f"only flat (atomic) sentence types where coordination / "
        f"modification / subordination would help.\n"
        f"  3. Run `compare_pipeline_to_targets(n=4)` to see end-to-end "
        f"output on real training rows BEFORE editing. This tells you "
        f"where the actual losses are (vocab? sentence types? "
        f"morphology?) so you can edit precisely instead of rewriting.\n"
        f"  4. Edit incrementally — extend `vocab.py` with new "
        f"citation-grounded entries, add or generalize sentence types in "
        f"`__init__.py` (see system prompt for recursive-type patterns). "
        f"Re-run `compare_pipeline_to_targets` after each meaningful "
        f"change to confirm it's helping.\n"
        f"  5. Stop when `validate_package` returns `PASS: True` with "
        f"zero placeholder leaks AND `compare_pipeline_to_targets` shows "
        f"clear improvement over the starting state for several distinct "
        f"caption patterns. Do not over-engineer."
    )

    # System prompt is large + stable → cache it.
    system = [{
        "type": "text",
        "text": SYSTEM_PREFACE,
        "cache_control": {"type": "ephemeral"},
    }]

    # Use the older `_20250305` web_search (no dynamic-filtering /
    # code_execution loop). The `_20260209` version is more efficient but its
    # code-execution container_id flow is fragile when interleaved with
    # custom tools — we hit `pending tool uses generated by code execution`
    # 400s mid-run that don't recover. Older version is more reliable.
    tools: List[Dict[str, Any]] = [
        {"type": "web_search_20250305", "name": "web_search"},
        *CUSTOM_TOOLS,
    ]

    # Bump SDK auto-retries; on top of that we wrap the per-iteration call
    # in our own backoff loop for `overloaded_error` (Anthropic capacity blip
    # — happens periodically on Opus 4.7 under load).
    client = anthropic.Anthropic(max_retries=4)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_kickoff}
    ]

    total_in = total_out = total_cr = total_cw = 0
    iterations = 0
    container_id: Optional[str] = None  # set when web_search dynamic filtering kicks in

    import time as _time

    while iterations < max_iterations:
        iterations += 1
        print(f"\n[generator/{iso}] iteration {iterations}/{max_iterations}",
              file=sys.stderr)
        stream_kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=16000,
            system=system,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=messages,
        )
        if container_id is not None:
            stream_kwargs["container"] = container_id

        # Anthropic Opus 4.7 occasionally returns `overloaded_error` mid-
        # stream under load. The SDK retries pre-stream connect but not
        # mid-stream, so we wrap with explicit backoff.
        response = None
        for backoff_attempt in range(6):
            try:
                with client.messages.stream(**stream_kwargs) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True, file=sys.stderr)
                    response = stream.get_final_message()
                break
            except (anthropic.APIConnectionError,
                    anthropic.RateLimitError,
                    anthropic.InternalServerError) as exc:
                wait = min(60, 5 * (2 ** backoff_attempt))
                print(f"\n[generator] transient api error "
                      f"({type(exc).__name__}): {exc} "
                      f"— sleeping {wait}s and retrying ({backoff_attempt + 1}/6)",
                      file=sys.stderr)
                _time.sleep(wait)
            except anthropic.APIStatusError as exc:
                # `overloaded_error` and `rate_limit_error` can arrive
                # mid-stream with HTTP status 200, so InternalServerError
                # doesn't catch them. Inspect the body / message and retry
                # those; permanent client errors (400, 401, 403, 404, 413)
                # surface to the caller.
                body_str = str(exc).lower()
                is_transient = (
                    "overloaded" in body_str
                    or "rate_limit" in body_str
                    or (exc.status_code or 0) >= 500
                    or exc.status_code == 529
                )
                if is_transient:
                    wait = min(60, 5 * (2 ** backoff_attempt))
                    print(f"\n[generator] transient api error mid-stream "
                          f"({type(exc).__name__} {exc.status_code}): {exc} "
                          f"— sleeping {wait}s and retrying ({backoff_attempt + 1}/6)",
                          file=sys.stderr)
                    _time.sleep(wait)
                else:
                    print(f"\n[generator] permanent api error "
                          f"({type(exc).__name__} {exc.status_code}): {exc}",
                          file=sys.stderr)
                    raise
        else:
            print("\n[generator] retries exhausted; bailing", file=sys.stderr)
            break

        # Capture/refresh container_id when present (web_search_20260209 et al
        # use a server-side code-execution container; subsequent requests must
        # pass it back or the API 400s with "container_id is required").
        container = getattr(response, "container", None)
        if container is not None and getattr(container, "id", None):
            container_id = container.id

        usage = response.usage
        total_in += getattr(usage, "input_tokens", 0) or 0
        total_out += getattr(usage, "output_tokens", 0) or 0
        total_cr += getattr(usage, "cache_read_input_tokens", 0) or 0
        total_cw += getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Record the assistant's full reply (text + tool_use + thinking blocks).
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            print("\n[generator] stop_reason=end_turn", file=sys.stderr)
            break
        if response.stop_reason == "pause_turn":
            # Server-side tool hit iteration cap; resume by re-sending.
            continue

        # Execute every client-side tool_use the model emitted this turn.
        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = block.name
            if name in {"web_search", "web_fetch"}:
                # Server-side; we don't execute these.
                continue
            args = dict(block.input)
            try:
                result = _execute_custom_tool(name, args, ctx)
                is_error = False
            except Exception as exc:  # noqa: BLE001
                result = f"ERROR: {exc}"
                is_error = True
            preview = (result if isinstance(result, str) else str(result))[:200]
            print(f"\n[tool] {name}({json.dumps(args)[:100]}) -> {preview!r}",
                  file=sys.stderr)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
                "is_error": is_error,
            })

        if not tool_results:
            # Model stopped calling tools but didn't emit end_turn — break to
            # avoid an infinite loop.
            print("\n[generator] no tool results to send; ending", file=sys.stderr)
            break

        messages.append({"role": "user", "content": tool_results})

    final = validate_package(iso, training_tokens=[t for t, _ in token_freq])
    print("\n" + "=" * 60, file=sys.stderr)
    print(final.summary(), file=sys.stderr)
    print(f"tokens — in:{total_in} out:{total_out} "
          f"cache_read:{total_cr} cache_write:{total_cw}", file=sys.stderr)

    return GenerationRun(
        iso=iso,
        package_root=package_root,
        iterations=iterations,
        final_validation=final,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cache_read=total_cr,
        total_cache_write=total_cw,
    )
