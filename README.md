# AmericasNLP 2026 — LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html). An agent reads the dev captions plus open-web references for each language and writes a Yaduha-compatible Pydantic grammar package. A VLM produces an English caption; that grammar deterministically renders it as a target-language caption. No training, no fine-tuning.

```
image → [VLM] → English → [EnglishToSentencesTool] → Sentence → Sentence.__str__() → target
```

## Methods

| method | description | when to use |
|---|---|---|
| `pipeline` | The proposed system: image → VLM English → structured-output translator → render. Two LLM calls. Grammatical-by-construction. | Primary captioner. |
| `one-step` | Image → structured Sentence JSON directly via VLM (no English intermediate). One LLM call. | Diagnostic / cost-saving comparison. Currently weaker than two-step at every model tier. |
| `direct` | Image → target caption directly (no structure). Few-shot from dev. | Baseline only. Output is fluent-looking but unverified. |

Backend dispatch is by model-name prefix: `claude-*` → Anthropic; `gpt-*`, `o1-*`, `o3-*`, `ft:*` → OpenAI; ollama-style names like `qwen2.5vl:32b` → local Ollama (`http://127.0.0.1:11434`).

## Results (dev set, ChrF++, N=50/lang)

**Two-step pipeline across 4 cloud models:**

| iso | organizer baseline | direct 3-shot (claude) | claude-snt-4-5 | gpt-4o-mini | gpt-4o | gpt-5 |
|---|--:|--:|--:|--:|--:|--:|
| bzd | 7.57 | 9.43 | **10.72** | 8.53 | 9.45 | 10.45 |
| grn | **20.82** | 18.37 | 15.38 | 13.56 | 12.72 | 17.18 |
| yua | — | 18.86 | **24.51** | 21.84 | 23.96 | 24.10 |
| nlv | 11.53 | 21.56 | 23.16 | 22.20 | 22.72 | **23.82** |
| hch | **17.77** | 18.14 | 11.08 | 11.43 | 11.47 | 15.52 |

**Local-VLM and one-step variants.** For the qwen row, qwen2.5vl:32b acts as *both* the VLM and the structured-output translator — a fully-local, zero-cost configuration. The 2026-04-23 open-weight translator scan confirmed qwen2.5vl:32b is the strongest local translator of what's on hand; neither the text-only `qwen2.5:32b` sibling, `mixtral:8x22b`, `llama3.1:8b`, nor `llama4:latest` beat it on this task.

| iso | qwen2.5vl:32b (local, free) | one-step gpt-4o-mini |
|---|--:|--:|
| bzd | 7.92 | 3.25 |
| grn | 13.50 | 13.18 |
| yua | 18.64 | 20.17 |
| nlv | 21.69 | 16.97 |
| hch | 11.44 | 8.94 |

**Headline averages over the 4 ChrF++-comparable languages (bzd / grn / nlv / hch):**

| | mean |
|---|--:|
| organizer baseline (Qwen3-VL → NLLB) | 14.42 |
| our direct 3-shot, claude-sonnet-4-5 | 16.88 |
| our pipeline, gpt-4o-mini (cloud, ~$1/sweep) | 13.93 |
| our pipeline, qwen2.5vl:32b (local, free) | 13.64 |
| our pipeline, claude-sonnet-4-5 (cloud, ~$5/sweep) | 15.09 |
| our pipeline, gpt-5 (cloud, ~$25/sweep) | 16.74 |
| **our pipeline, best-per-lang (cloud)** | **16.81** |

**Best per-language pipeline picks:** bzd → claude-sonnet-4-5; grn → gpt-5; yua → claude-sonnet-4-5; nlv → gpt-5; hch → gpt-5.

**Key takeaways**
- We beat the organizer baseline (Qwen3-VL → NLLB) on bzd (+3.15) and nlv (+12.29); lose on grn (-3.64) and hch (-2.25); cover yua which their NLLB pipeline doesn't.
- `qwen2.5vl:32b` (local, as both VLM and translator) lands at 4-lang avg **13.64** — roughly matching our `gpt-4o-mini` cloud pipeline (13.93) at $0 cost, but still 0.78 below the organizer baseline's 14.42. Scan of locally-available open-weight translators found nothing stronger than qwen2.5vl:32b itself for the structured-output step.
- One-step (image → structured directly) loses to two-step at every model tier: the English intermediate is doing real work, and VLMs are weaker at structured-output JSON than text translators.
- Best-per-lang pipeline ties direct 3-shot on average and adds the grammaticality-by-construction property.
- The open-weight gap to gpt-5 / claude-3-shot does **not** live in the translator step — the languages where our pipeline underperforms the organizer baseline (grn, hch) are exactly the ones with the thinnest yaduha package coverage. Investment should go into language-package breadth, not translator-model substitution.

## How it works

1. `generate-language --iso <code>` runs an Anthropic Opus 4.7 agent with these tools: web search, read reference yaduha-{hch,ovp} packages, read training-slice captions, extract content-word frequencies, write package files, `validate_package`, `test_translate_english`, `compare_pipeline_to_targets`.
2. The generator only ever sees a deterministic **training slice** (30 of 50 dev rows); the other 20 are held out for honest scoring (`--val-only`). Bootstrap prompt forbids hardcoded `if english == "..."` shortcuts and keys everything to structured `Sentence` inputs.
3. The produced `yaduha-{iso}` package is a standalone Python package with Pydantic `Sentence` subclasses. `__str__()` is deterministic Python — no LLM at render time. OOV lemmas render as `[english_lemma]`.
4. `evaluate --method pipeline` runs image → VLM English → `PipelineTranslator` → target. `--method one-step` runs image → structured Sentence directly via OpenAI's structured outputs. `--method direct` is the few-shot text baseline.

See [`DESIGN.md`](DESIGN.md) for the thesis, [`docs/bootstrap_language.md`](docs/bootstrap_language.md) for the generator workflow, [`docs/auto_finetuning_spec.md`](docs/auto_finetuning_spec.md) for the proposed open-weight fine-tuning extension, [`PROGRESS.md`](PROGRESS.md) for team / journey notes.

## Setup

```bash
git clone --recurse-submodules https://github.com/kubishi/americasnlp-2026-shared-task.git
cd americasnlp-2026-shared-task
uv sync
cat <<EOF > .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
EOF
# For local Ollama models (qwen2.5vl, qwen2.5, etc.), have ollama running:
#   ollama serve &
#   ollama pull qwen2.5vl:32b
```

## Usage

```bash
# Generate / improve a language package
uv run americasnlp generate-language --iso bzd                  # 30/20 train/val split
uv run americasnlp generate-language --iso bzd --train-frac 1.0  # submission mode

# Dev-set evaluation
uv run americasnlp evaluate --language bribri --method pipeline                # default model
uv run americasnlp evaluate --language bribri --method pipeline --vlm gpt-4o-mini
uv run americasnlp evaluate --language bribri --method pipeline --vlm claude-sonnet-4-5
uv run americasnlp evaluate --language bribri --method pipeline --vlm qwen2.5vl:32b   # local
uv run americasnlp evaluate --language bribri --method one-step --vlm gpt-4o-mini
uv run americasnlp evaluate --language bribri --method pipeline --val-only     # 20 held-out

# Test-set submission JSONL (held until final)
uv run americasnlp submit --language bribri --method pipeline --vlm <model> \
    --output results/submissions/bribri_pipeline.jsonl
```

## Cost discipline (please follow)

All eval costs scale with `langs × 50 dev rows × ~3 API calls/row`. Observed per full-dev sweep:

| sweep | cost |
|---|--:|
| 5 langs × `qwen2.5vl:32b` (local) | $0 |
| 5 langs × `gpt-4o-mini` | ~$1 |
| 5 langs × `claude-sonnet-4-5` | ~$5 |
| 5 langs × `gpt-4o` | ~$10 |
| 5 langs × `gpt-5` | ~$25 |

**Default to `gpt-4o-mini` or `qwen2.5vl:32b` (local) during iteration.** Both are at the floor of the model matrix and cost ≪ 1 cent per row. Use them to:
- Smoke-test new generator runs / package changes
- Compare two language-package versions (relative deltas are usually consistent across model tiers)
- Iterate on prompt or pipeline changes

**Use the matrix above as a lift predictor.** Typical lifts when upgrading:
- `gpt-4o-mini → gpt-4o`: +0.5 to +3 ChrF++
- `gpt-4o-mini → gpt-5`: +2 to +5 ChrF++
- `gpt-4o-mini → claude-sonnet-4-5`: +2 to +4 ChrF++ (and stronger on yua)

If a change improves `gpt-4o-mini` numbers, expect a similar relative improvement on the strong models. Don't re-sweep `gpt-5` / `claude-sonnet-4-5` unless the dev matrix is genuinely stale (e.g. after a major package or pipeline refactor).

**Reserve the strong models** (`gpt-5`, `claude-sonnet-4-5`, `claude-opus-*`, future Opus tiers) for:
- Final submission runs (best-per-lang on the 990-row test set, ~$30 total)
- One sanity check after a major refactor to confirm the matrix shape still holds

## Repository layout

```
src/americasnlp/
  captioners/{pipeline,one_step,direct}.py   # backend-dispatched (OpenAI / Anthropic / Ollama)
  generator/{agent,extract,validate,scaffold,split}.py
  evaluate.py submit.py cli.py data.py languages.py
  _openai.py  _anthropic.py  _ollama.py       # backend helpers
yaduha/                                        # framework (submodule)
yaduha-ovp/                                    # reference package (submodule, read-only)
yaduha-{hch,bzd,grn,yua,nlv}/                  # in-tree language packages
americasnlp2026/                               # task data + test set (submodule)
docs/
  bootstrap_language.md                        # how to use the generator
  auto_finetuning_spec.md                      # proposed open-weight extension (not yet executed)
results/dev/                                   # eval artifacts (CSV + JSONL per lang × method × model)
DESIGN.md  PROGRESS.md
```

## Open directions (no commitments yet)

- **Prompt engineering on the VLM step**: actively steer the English caption / one-step output toward in-vocab lemmas via "use the simplest possible vocabulary" / hypernym-substitution guidance. Initial test on one-step lifted yua by +5.16 and grn by +3.97. Same idea may help two-step.
- **Auto-finetuning** ([`docs/auto_finetuning_spec.md`](docs/auto_finetuning_spec.md)): port the OVP `feature/weakmodels` recipe (LoRA-fine-tune Qwen2.5-3B as the structured-output translator, training data synthesized from the language package itself). Two paths: local LoRA (Path A) or OpenAI fine-tune of `gpt-4o-mini` (Path B). Currently on hold.
- **Expand language packages**: agent-driven vocab/sentence-type expansion for the languages where we currently lose to the organizer baseline (grn, hch).
- **Submission-mode generators**: re-run with `--train-frac 1.0` so the agent sees all dev rows before final test submissions.

## Sister projects

[yaduha](https://github.com/kubishi/yaduha-2) (framework) · [yaduha-ovp](https://github.com/kubishi/yaduha-ovp) (reference + `feature/weakmodels` open-weight branch) · [paper_yaduha_open_weight](https://github.com/kubishi/paper_yaduha_open_weight) (the WIP fine-tuning paper) · [conlang-claude](https://github.com/kubishi/conlang-claude)
