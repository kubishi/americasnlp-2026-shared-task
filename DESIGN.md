# Design — AmericasNLP 2026 Shared Task

## Thesis

> **Given image→target-caption training pairs plus whatever it can find on the
> open web about the target language, an automated agent can generate a
> Yaduha-compatible language package — a Pydantic grammar with vocabulary —
> that combines with a vision-language model to produce grammatically
> guaranteed image captions in a low-resource Indigenous language without any
> fine-tuning.**

This is the entire pitch. Everything in the repo exists to test it.

The shared task gives us 5 Indigenous languages, ~50 dev examples per language,
and asks us to caption unseen images. These are *low-resource* — published
grammars and dictionaries exist (Bohnemeyer for Yucatec, Constenla Umaña for
Bribri, etc.), but parallel corpora are tiny and we have no native speakers
on the team. The conventional approach (fine-tune a multilingual seq2seq) is
not viable. The interesting question is: **how far can we go with
structured-output decoding constrained by a grammar that an agent authored
from public sources?**

The system has two halves:

1. **The generator** — `americasnlp generate-language --iso <code>` runs an
   agent (Anthropic Opus 4.7 with web search) over the training data and
   open-web references for the target language, and emits a complete
   `yaduha-{iso}/` package. This is the artifact we report on in the paper.
2. **The captioner** — `americasnlp evaluate --method pipeline` plugs that
   package into the LLM-RBMT pipeline. The competition just consumes its output.

## Pipeline

The submission pipeline for every language is exactly:

```
image
  ├─► [VLM]                                English caption
  ├─► [yaduha.EnglishToSentencesTool]      structured Pydantic Sentence
  └─► [Sentence.__str__()]                 target-language caption
```

The `EnglishToSentencesTool` step uses OpenAI's structured outputs to force
the LLM to emit a JSON object that conforms to the Pydantic schema we
authored. Because the schema *is* the grammar (vocabulary enums, morphology
features, sentence templates), the only thing the LLM is allowed to do is
*select valid grammatical structure*. The actual target-language string is
synthesized by Python code in `Sentence.__str__()`. The model never sees
or generates target-language tokens.

## Comparison baselines (not the proposed system, just for the paper)

For each language we will also report:

1. **zero-shot direct prompting** — VLM gets the image and a one-line prompt
   asking for a caption in the target language. No examples.
2. **few-shot direct prompting** — same, but the prompt also includes a
   handful of `(image, target_caption)` pairs sampled from the dev set.

These baselines exist to anchor the ChrF++ scores. We expect them to win on
ChrF++ in absolute terms (because they emit fluent surface forms even when
they're hallucinating) and to lose on grammaticality and semantic faithfulness
in the Stage-2 human eval (because they have no real knowledge of the
target language).

We do **not** ship the agentic translator from the original `yaduha`
framework. It degenerates into repetition loops and obscures the thesis.

## Repository layout

```
americasnlp-2026-shared-task/
├── README.md                          # public-facing pitch
├── DESIGN.md                          # this file
├── PROGRESS.md                        # team coordination
├── pyproject.toml                     # uv workspace
├── americasnlp2026/                   # task data (submodule)
├── yaduha/                            # framework (submodule)
├── yaduha-hch/                        # Wixárika (submodule, exists)
├── yaduha-ovp/                        # Owens Valley Paiute (submodule, kept for tests)
├── yaduha-bzd/                        # Bribri (NEW, in-tree until stable)
├── yaduha-grn/                        # Guaraní (NEW)
├── yaduha-yua/                        # Yucatec Maya (NEW)
├── yaduha-nlv/                        # Orizaba Nahuatl (NEW)
├── src/
│   └── americasnlp/
│       ├── __init__.py
│       ├── languages.py               # registry: key ↔ ISO ↔ display name ↔ data path
│       ├── data.py                    # JSONL loaders, image path resolution
│       ├── captioners/
│       │   ├── __init__.py            # Captioner protocol
│       │   ├── pipeline.py            # the proposed system (image → English → yaduha → target)
│       │   └── direct.py              # baselines (zero-shot, few-shot)
│       ├── evaluate.py                # ChrF++ on dev split
│       ├── submit.py                  # produce JSONL on test split
│       └── cli.py                     # `python -m americasnlp ...`
├── docs/
│   └── bootstrap_language.md          # playbook for authoring a yaduha-{lang}
├── results/
│   ├── dev/                           # dev-set evaluation outputs (CSVs + JSONL)
│   └── submissions/                   # final test-set submission JSONLs
└── scripts/                           # only thin shell wrappers
    └── run_all.sh                     # sweep all (language × method) for the dev set
```

The new code lives in `src/americasnlp/`. `scripts/` reduces to shell glue.
The yaduha-{lang} packages are submodules once they're real, in-tree until then.

## What we're salvaging vs. discarding

Salvage:
- `yaduha-hch/` — already runs; extend vocab + sentence types but keep the API.
- `scripts/languages.py` → moves to `src/americasnlp/languages.py` with minor cleanup.
- `scripts/baseline.py::caption_one` → moves to `src/americasnlp/captioners/direct.py`.
- `PROGRESS.md` — keep as a team coordination doc.

Discard:
- `main.py` — empty stub.
- `scripts/evaluate.py` — 533 lines, three methods including the agentic
  loop we're abandoning. Replaced by the new pipeline+CLI.
- `scripts/test_caption.py` — superseded; `PipelineTranslator` does the same thing.
- `scripts/make_submission.py` — broken (imports a nonexistent
  `evaluate_dev` module). Replaced by the new `submit.py`.

## Language packages

Each `yaduha-{iso}` package follows the layout that `yaduha-hch` already
demonstrates:

```
yaduha-{iso}/
├── pyproject.toml                  # entry point: yaduha.languages → {iso} = "yaduha_{iso}:language"
└── yaduha_{iso}/
    ├── __init__.py                 # enums, Pydantic models, sentence types, language object
    ├── vocab.py                    # NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS as VocabEntry lists
    └── prompts.py                  # natural-language grammar instructions (optional, but include it)
```

Minimum viable surface for each language to plug into the pipeline:

- Two sentence types: `SubjectVerbSentence`, `SubjectVerbObjectSentence`.
- Verb morphology: at least person (1/2/3), number (sg/pl), tense/aspect.
- ~50–100 lexical entries seeded from the dev-set captions and
  publicly-available wordlists (ASJP, SIL, Glottolog references).
- A `language = Language(code=..., name=..., sentence_types=(...))` object.

Out-of-vocabulary words render as `[english_lemma]` placeholders. This is the
existing `yaduha-hch` convention, and ChrF++ will penalize them — that's the
honest signal of vocabulary coverage gaps and we want it visible.

## Bootstrapping a new language package

The interesting research artifact is the **prompt** we hand a coding agent
to produce a `yaduha-{iso}` package. That prompt lives at
`docs/bootstrap_language.md` and contains:

1. The yaduha framework conventions (with `yaduha-hch` and `yaduha-ovp` as
   reference implementations).
2. The expected file layout.
3. Instructions to source vocabulary from the dev split + cite-able external
   resources.
4. A correctness checklist (the package must import, instantiate
   `language`, and produce a non-empty string for at least one
   `SubjectVerbSentence` example).

The paper will reproduce this prompt verbatim and report what the agent
produced for each language. That's the contribution.

## CLI surface

```bash
# Caption the dev split with the proposed system, report ChrF++.
uv run python -m americasnlp evaluate \
    --language wixarika \
    --method pipeline \
    --vlm gpt-4o-mini

# Same, but with the few-shot direct baseline.
uv run python -m americasnlp evaluate \
    --language wixarika --method direct --shots 3 --vlm gpt-4o-mini

# Produce the submission JSONL on the test split.
uv run python -m americasnlp submit \
    --language wixarika --method pipeline --vlm gpt-4o-mini \
    --output results/submissions/wixarika_pipeline.jsonl

# Sweep everything for the dev set.
bash scripts/run_all.sh
```

Output paths follow a consistent pattern:

```
results/dev/{language}_{method}_{vlm}[_shots{k}].{csv,jsonl}
results/submissions/{language}_{method}.jsonl   # test-set submission, no scoring
```

Both forms include resume-from-existing logic so a re-run only fills in
missing rows.

## Risks and what to do about them

- **ChrF++ will be low for the pipeline method.** The grammar models are
  small and produce simple sentences against ornate references. Mitigation:
  ship direct baselines too so we have a comparison row in the paper. Lean
  on the Stage-2 human eval for the grammaticality story. Don't try to
  win Stage 1.
- **Building 4 language packages by May 1 is tight.** The bootstrap prompt
  has to do most of the work; humans (us) only review and tweak.
  Fallback: if a yaduha-{lang} package doesn't materialize for some
  language, that language only gets the direct baselines submitted.
- **Bribri image format bug** is still open from the previous progress doc.
  Fix in the new `data.py` by re-encoding through Pillow on load.

## Out of scope

- Fine-tuning, LoRA, RAG, or any approach that requires gradient updates.
- The `AgenticTranslator` from yaduha. Removed.
- Building a `plot_results.py`. Numbers go in CSVs and into the paper directly.
- Anything Owens Valley Paiute beyond keeping the package available for
  framework-level tests.
