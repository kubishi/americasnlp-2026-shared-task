# AmericasNLP 2026 — LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html). An agent reads the dev captions plus open-web references for each language and writes a Yaduha-compatible Pydantic grammar package. A VLM produces an English caption; that grammar deterministically renders it as a target-language caption. No training, no fine-tuning.

```
image → [VLM] → English → [EnglishToSentencesTool] → Sentence → Sentence.__str__() → target
```

## Results (dev set, ChrF++, N=50/lang)

| iso | organizer baseline | direct 3-shot (claude) | pipe claude-snt-4-5 | pipe gpt-4o-mini | pipe gpt-4o | pipe gpt-5 |
|---|--:|--:|--:|--:|--:|--:|
| bzd | 7.57 | 9.43 | **11.17** | 8.49 | 9.45 | 10.45 |
| grn | **20.82** | 18.37 | 15.70 | 13.08 | 12.72 | 17.18 |
| yua | — | 18.86 | **25.01** | 21.01 | 23.96 | 24.10 |
| nlv | 11.53 | 21.56 | 23.16 | 22.17 | 22.72 | **23.82** |
| hch | **17.77** | 18.14 | 12.59 | 11.07 | 11.47 | 15.52 |

**Averages over the 4 ChrF++-comparable languages (bzd / grn / nlv / hch):**

| | mean |
|---|--:|
| organizer baseline (Qwen3-VL → NLLB) | 14.42 |
| our direct 3-shot, claude-sonnet-4-5 | 16.87 |
| our pipeline, gpt-5 | 16.74 |
| our pipeline, claude-sonnet-4-5 | 15.66 |
| **our pipeline, best-per-lang** | **16.92** |

Best pipeline per language: bzd → claude-sonnet-4-5, grn → gpt-5, yua → claude-sonnet-4-5, nlv → gpt-5, hch → gpt-5. All 5 generated `yaduha-{iso}/` packages are in-tree, citation-grounded, and pass the validation gate (no `[english]` placeholder leaks).

## How it works

1. `generate-language --iso <code>` runs an Anthropic Opus 4.7 agent with these tools: web search, read reference yaduha-{hch,ovp} packages, read training-slice captions, extract content-word frequencies, write files inside the target package, `validate_package`, `test_translate_english`, `compare_pipeline_to_targets`.
2. The generator only ever sees a deterministic **training slice** (30 of 50 dev rows); the other 20 are held out for honest scoring. The system prompt forbids hardcoded `if english == "..."` shortcuts and keys everything to structured `Sentence` inputs.
3. The produced `yaduha-{iso}` package is a standalone Python package with Pydantic `Sentence` subclasses. `__str__()` is deterministic Python — no LLM at render time. OOV lemmas render as `[english_lemma]`.
4. `evaluate --method pipeline` runs image → VLM English → `PipelineTranslator` → target. Backend (OpenAI vs Anthropic) is dispatched on model name prefix (`claude-*` → Anthropic, anything else → OpenAI).

See [`DESIGN.md`](DESIGN.md) for the thesis, [`docs/bootstrap_language.md`](docs/bootstrap_language.md) for the generator workflow, [`PROGRESS.md`](PROGRESS.md) for team / journey notes.

## Setup

```bash
git clone --recurse-submodules https://github.com/kubishi/americasnlp-2026-shared-task.git
cd americasnlp-2026-shared-task
uv sync
cat <<EOF > .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
EOF
```

## Usage

```bash
# Generate / improve a language package
uv run americasnlp generate-language --iso bzd                 # 30/20 train/val split
uv run americasnlp generate-language --iso bzd --train-frac 1.0 # submission mode

# Dev-set evaluation
uv run americasnlp evaluate --language bribri --method pipeline               # default model
uv run americasnlp evaluate --language bribri --method pipeline --vlm gpt-4o-mini
uv run americasnlp evaluate --language bribri --method pipeline --vlm claude-sonnet-4-5
uv run americasnlp evaluate --language bribri --method pipeline --val-only    # 20 held-out rows

# Test-set submission JSONL (held until final)
uv run americasnlp submit --language bribri --method pipeline --vlm <model> \
    --output results/submissions/bribri_pipeline.jsonl
```

## Cost discipline (please follow this)

All eval costs scale with `langs × 50 dev rows × ~3 API calls/row`. Observed per full-dev sweep:

| sweep | cost |
|---|--:|
| 5 langs × `gpt-4o-mini` | ~$1 |
| 5 langs × `gpt-4o` | ~$10 |
| 5 langs × `claude-sonnet-4-5` | ~$5 |
| 5 langs × `gpt-5` | ~$25 |

**Default to `gpt-4o-mini` during iteration.** It's the floor of the model matrix, ~25× cheaper than `gpt-5` and ~10× cheaper than `gpt-4o`. Use it to:
- Smoke-test new generator runs / package changes (does the package even produce non-empty output?)
- Compare two language-package versions against each other (relative deltas are usually consistent across model tiers)
- Iterate on prompt or pipeline changes

**Use the table above as a lift predictor.** The matrix shows what the pipeline gains by upgrading `gpt-4o-mini → gpt-4o → gpt-5 / claude-sonnet-4-5`. Typical lifts for our pipeline:
- `gpt-4o-mini → gpt-4o`: +0.5 to +3 ChrF++
- `gpt-4o-mini → gpt-5`: +2 to +5 ChrF++
- `gpt-4o-mini → claude-sonnet-4-5`: +2 to +4 ChrF++ (and stronger on yua)

If a change improves `gpt-4o-mini` numbers, expect a similar relative improvement on the strong models. Don't re-sweep `gpt-5` / `claude-sonnet-4-5` unless the dev matrix is genuinely stale (e.g. after a major package or pipeline refactor).

**Reserve the strong models** (`gpt-5`, `claude-sonnet-4-5`, future Opus tiers) for:
- Final submission runs (best-per-lang, ~$30 for the test set)
- One sanity check after a major refactor to confirm the matrix shape still holds

## Repository layout

```
src/americasnlp/
  captioners/{pipeline,direct}.py   # backend-dispatched (OpenAI / Anthropic)
  generator/{agent,extract,validate,scaffold,split}.py
  evaluate.py submit.py cli.py data.py languages.py
yaduha/                              # framework (submodule)
yaduha-ovp/                          # reference package (submodule, read-only)
yaduha-{hch,bzd,grn,yua,nlv}/        # in-tree language packages
americasnlp2026/                     # task data + test set (submodule)
docs/bootstrap_language.md
results/dev/                         # eval artifacts (CSV + JSONL per lang × model)
DESIGN.md PROGRESS.md
```

## Pending

- Final submission: per-lang best model on the 990-row test set (deferred until final go-ahead).
- Optional: re-run generators with `--train-frac 1.0` (sees all dev) before final submission.
- Optional: try claude-opus-4-7 as the translator on dev (likely the strongest, but $$$).

## Sister projects

[yaduha](https://github.com/kubishi/yaduha-2) (framework) · [yaduha-ovp](https://github.com/kubishi/yaduha-ovp) · [conlang-claude](https://github.com/kubishi/conlang-claude) (same package shape for synthetic conlangs)
