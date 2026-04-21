# AmericasNLP 2026 — LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html). An agent reads the dev captions plus open-web references for each language and writes a Yaduha-compatible Pydantic grammar package. A VLM produces an English caption; that grammar deterministically renders it as a target-language caption. No training, no fine-tuning.

```
image → [VLM] → English → [EnglishToSentencesTool] → Sentence → Sentence.__str__() → target
```

## Results (dev set)

| lang | organizer baseline | ours pipeline (LLM-RBMT) | ours direct (3-shot) |
|---|--:|--:|--:|
| bzd (Bribri)          | 7.57  | **11.17** (N=48) | 9.43 (N=50) |
| grn (Guaraní)         | 20.82 | 15.70 (N=49)    | 18.37 (N=50) |
| yua (Yucatec Maya)    | —     | **25.01** (N=49) | 18.59 (N=44) |
| nlv (Orizaba Nahuatl) | 11.53 | **23.16** (N=50) | 21.56 (N=50) |
| hch (Wixárika)        | 17.77 | 12.59 (N=44)*   | 18.14 (N=50) |
| **avg (4 measurable)** | **14.43** | **18.16** | 16.88 |

*Metric: ChrF++ (the shared task's Stage-1 ranking metric). Organizer baseline = Qwen3-VL-8B + Sheffield 2023 NLLB MT. Ours uses Claude Sonnet 4.5 for VLM + structured outputs.*

*hch N=44/50: sporadic JSON-parse errors from yaduha's prompt-based `AnthropicAgent` — expected to close the gap once we swap in `client.messages.parse()`.*

**Held-out validation (20 rows/lang, never shown to the generator agent):**

| lang | pipeline | direct | Δ |
|---|--:|--:|--:|
| bzd | **10.77** | 8.74 | +2.03 |
| grn | 15.15    | **18.96** | -3.81 |
| yua | **24.49** | 19.85 | +4.64 |
| nlv | **23.95** | 22.07 | +1.88 |
| hch | 11.87    | **17.89** | -6.02 |

## How it works

1. `generate-language --iso <code>` runs an **Anthropic Opus 4.7 agent** with these tools: web search, read reference yaduha-{hch,ovp} packages, read training-slice captions, extract content-word frequencies, write files inside the target package, `validate_package`, `test_translate_english`, `compare_pipeline_to_targets`. The agent inspects the data, drafts a grammar, and iterates until validation + smoke tests look good.
2. The generator only ever sees a deterministic **training slice** (30 of 50 dev rows); the other 20 are held out for honest scoring. The system prompt forbids hardcoded `if english == "..."` shortcuts and keys everything to structured `Sentence` inputs.
3. The produced `yaduha-{iso}` package is a standalone Python package with Pydantic `Sentence` subclasses. `__str__()` on each sentence is deterministic Python — no LLM at render time. OOV lemmas render as `[english_lemma]`.
4. `evaluate --method pipeline` runs image → Sonnet-4.5 VLM → English → `PipelineTranslator` (also Sonnet-4.5) → target. `evaluate --method direct --shots K` is the baseline.

See [`DESIGN.md`](DESIGN.md) for the thesis, [`docs/bootstrap_language.md`](docs/bootstrap_language.md) for the generator workflow, and [`PROGRESS.md`](PROGRESS.md) for team notes.

## Setup

```bash
git clone --recurse-submodules https://github.com/kubishi/americasnlp-2026-shared-task.git
cd americasnlp-2026-shared-task
uv sync
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

Optional COMET scoring: `uv sync --extra comet` then `evaluate --comet`.

## Usage

```bash
# Generate / improve a language package
uv run americasnlp generate-language --iso bzd           # uses 30/20 train/val split
uv run americasnlp generate-language --iso bzd --train-frac 1.0   # submission mode (all dev)

# Evaluate on the dev set
uv run americasnlp evaluate --language bribri --method pipeline
uv run americasnlp evaluate --language bribri --method direct --shots 3
uv run americasnlp evaluate --language bribri --method pipeline --val-only   # held-out 20 rows

# Produce a test-set submission JSONL
uv run americasnlp submit --language bribri --method pipeline \
    --output results/submissions/bribri_pipeline.jsonl
```

## Repository layout

```
src/americasnlp/
  captioners/{pipeline,direct}.py   # LLM-RBMT + direct prompting
  generator/{agent,extract,validate,scaffold,split}.py   # the generator
  evaluate.py submit.py cli.py data.py languages.py
yaduha/ yaduha-hch/ yaduha-ovp/ yaduha-{bzd,grn,yua,nlv}/   # framework + language packages
americasnlp2026/                                            # task data (submodule)
docs/bootstrap_language.md
results/dev/                        # eval artifacts
DESIGN.md PROGRESS.md
```

## Pending

- Replace yaduha's prompt-based `AnthropicAgent` structured outputs with `client.messages.parse(output_format=...)` — should close the hch gap (6 of 50 rows currently hit JSON-parse errors).
- Run generators with `--train-frac 1.0` for the final submission packages.
- Run captioner on the test set on 2026-05-01.

## Sister projects

[yaduha](https://github.com/kubishi/yaduha-2) (framework) · [yaduha-hch](https://github.com/kubishi/yaduha-hch) · [yaduha-ovp](https://github.com/kubishi/yaduha-ovp) · [conlang-claude](https://github.com/kubishi/conlang-claude) (same package shape for synthetic conlangs)
