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

## Architectural alternatives considered

Two variants of the captioner share infrastructure with the proposed system
and are kept around in `src/americasnlp/captioners/` for measurement:

- **One-step** (`one_step.py`): the VLM emits the structured `Sentence` JSON
  directly via OpenAI structured outputs, skipping the English intermediate.
  Cheaper (one LLM call) and gives the VLM full image context. Empirically
  loses to the two-step pipeline at every model tier — VLMs are notably
  weaker at structured-output JSON than text translators, even with strong
  in-vocab steering. Documented as a measured architectural alternative,
  not the primary path.
- **Local Ollama backend**: any of the captioners can take an Ollama-hosted
  model (e.g. `qwen2.5vl:32b`) in any slot via model-name dispatch. Local
  multimodal as the VLM step is roughly equivalent to cloud `gpt-4o-mini`
  for free. Useful for low-cost iteration; insufficient quality for final
  submissions on its own.

## Comparison baselines (not the proposed system, just for the paper)

For each language we also report:

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

- **ChrF++ in the 10–25 range is realistic.** As of 2026-04-21, the
  pipeline beats the organizer baseline on bzd / nlv (and covers yua which
  the baseline can't), but loses on grn / hch where Sonnet has heavy
  pretraining exposure. Best-per-language average matches our few-shot
  direct baseline. Mitigation: ship direct baselines for comparison; lean
  on the Stage-2 human eval for the grammaticality story.
- **Per-language model selection costs.** Picking the strongest model per
  language requires a strong-model dev sweep (~$25 for `gpt-5`). Done once,
  the matrix predicts lift across the board so we don't need to repeat for
  marginal package changes. See README → Cost discipline.
- **JSON-parse failures on the structured-output step** were the original
  motivation for switching from Anthropic to OpenAI for the translator.
  OpenAI's native structured outputs eliminated them; ollama models without
  fine-tuning are unreliable here (matches the OVP weakmodels-paper finding).

## Future directions (deferred, not currently committed)

- **Auto-finetuning the translator step** ([`docs/auto_finetuning_spec.md`](docs/auto_finetuning_spec.md)).
  Port the OVP `feature/weakmodels` recipe (LoRA on Qwen2.5-3B) to our 5
  languages. Two paths described: local LoRA (Path A) for the open-weight
  paper story, or OpenAI fine-tune of `gpt-4o-mini` (Path B) for cheap
  high-quality submission inference. Both train on synthetic
  (English, structured Sentence) pairs sampled from the language package
  itself — no parallel image/text data needed.
- **VLM prompt engineering**: actively steer the English caption (or the
  one-step structured output) toward in-vocab lemmas via hypernym
  substitution guidance. Initial test on one-step lifted yua by +5.16 and
  grn by +3.97; same approach should help two-step.
- **Submission-mode generators**: re-run with `--train-frac 1.0` so the
  agent sees all dev rows before final test-set submissions.

## Out of scope (still)

- The `AgenticTranslator` from yaduha. Removed.
- Anything Owens Valley Paiute beyond keeping `yaduha-ovp` available as
  a read-only reference for the generator + fine-tuning experiments.
