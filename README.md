# AmericasNLP 2026 — LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html). An agent reads the dev captions plus open-web references for each language and writes a Yaduha-compatible Pydantic grammar package. The grammar's lemma fields are typed as `Literal[...]` enums, so a strong VLM (gpt-5) emits structured `SentenceList` output that Python deterministically renders into the target language. No training, no fine-tuning, no silent OOV stripping — schema-enforced grammaticality end-to-end.

```
image → [VLM with strict-Literal schema] → SentenceList → Sentence.__str__() → target
```

## Methods

| method | description | when to use |
|---|---|---|
| `one-step` | Single VLM call: image + schema → `SentenceList` JSON via OpenAI structured outputs. | **Primary captioner (with gpt-5 as VLM).** |
| `pipeline` | Two-step: image → VLM English → structured-output translator → render. | Useful for cheap iteration with claude-vl + gpt-4o-mini; loses at peak quality. |
| `direct` | Image → target caption directly (no structure). Few-shot. | Baseline only. Output unverified. |

Backend dispatch is by model-name prefix: `claude-*` → Anthropic; `gpt-*`, `o1-*`, `o3-*`, `ft:*` → OpenAI; ollama-style names like `qwen2.5vl:32b` → local Ollama (`http://127.0.0.1:11434`).

## Architectural decisions (2026-04-30)

Several invariants the pipeline now enforces:

- **Strict-Literal lemma typing.** Every lemma field in every package is typed `Literal["dog", "cat", ...]` from the package's vocab list. The LLM's structured output cannot emit OOV terms — Pydantic rejects them at validation time. This is what kills the "silent OOV stripping" failure mode the original yaduha-hch package suffered from.
- **No bracket-stripping.** The `clean_text` function inside yaduha's `PipelineTranslator` no longer strips `[english_placeholder]` patterns. Combined with strict-Literal typing, placeholders should never appear at the surface level — but if they do, they're visible (a transparency property, not noise).
- **Proper-noun escape hatch.** Every package's `Noun` model has `proper_noun: Optional[str]` for genuine named entities (Mercado 4, Asunción, etc.). Lemma fields stay strict; the `proper_noun` slot is the carefully-prompted exception.
- **hch CopularSentence removed.** Wixárika has no Adjective concept in our package, so `CopularSentence` was being abused as a fallback for property predication ("the bag is green"), producing degenerate "X is X" tautology cascades. Removing the type forces property sentences to be dropped at parse time — honest given the grammar's coverage.

## Results (dev set, ChrF++, N=50/lang)

Headline sweep, 2026-04-30, with the schema fixes above:

| iso | organizer baseline | gpt-5 one-step | claude-vl + gpt-4o-mini | claude-vl + gpt-4o |
|---|--:|--:|--:|--:|
| bzd | 7.57 | 9.62 | 10.38 | **10.97** |
| grn | **20.82** | **18.57** | 15.32 | 15.63 |
| yua | — | **25.09** | 16.48 | 17.02 |
| nlv | 11.53 | **22.89** | 16.81 | 16.79 |
| hch | **17.77** | **16.08** | 15.56 | 14.73 |
| **5-lang mean** | — | **18.45** | 14.91 | 15.03 |
| **4-lang mean (excl yua)** | **14.42** | **16.79** | 14.52 | 14.53 |

**Best-per-lang picks (mixed configs):**

| iso | best config | ChrF |
|---|---|--:|
| bzd | claude-vl + gpt-4o | 10.97 |
| grn | gpt-5 one-step | 18.57 |
| yua | gpt-5 one-step | 25.09 |
| nlv | gpt-5 one-step | 22.89 |
| hch | gpt-5 one-step | 16.08 |
| **mean (5-lang)** | | **18.71** |
| **mean (4-lang excl yua)** | | **16.86** |

**Vs organizer baseline (Qwen3-VL → NLLB):** **+2.44 ChrF** on the 4 comparable languages, plus full coverage of yua which their NLLB pipeline doesn't translate.

Per-language wins / losses:
- Wins: bzd (+3.40), nlv (+11.36), yua (full coverage)
- Losses: grn (-2.25), hch (-1.69)

**Key takeaways**

- **gpt-5 one-step is the strongest single config** on 4 of 5 languages (everything except bzd). Single API call, no two-step pipeline, the VLM with the schema in context handles structured output natively.
- **Open-weight VLM substitution costs ~2 ChrF** but isn't competitive at the peak — `qwen2.5vl:32b` as both VLM and translator lands around 14.6 mean. With the new strict-Literal schema, open-weight *translators* fail outright (parse errors on every row); the schema strictness is incompatible with non-fine-tuned local models.
- **The grn/hch losses are language-package coverage gaps** (vocabulary breadth + grn morphology + hch's missing Adjective concept), not translator-model issues. Same diagnosis as the previous version of this README; the schema fixes address grammaticality but not coverage.
- **Cost picture:** dev sweep with gpt-5 one-step ≈ $7 (250 rows); test submission ≈ $29 (990 rows). Two-step sweeps with gpt-5 are 2-3× more expensive because the schema gets shipped on every call.

## How it works

1. `generate-language --iso <code>` runs an Anthropic Opus 4.7 agent with these tools: web search, read reference yaduha-{hch,ovp} packages, read training-slice captions, extract content-word frequencies, write package files, `validate_package`, `test_translate_english`, `compare_pipeline_to_targets`.
2. The generator only ever sees a deterministic **training slice** (30 of 50 dev rows); the other 20 are held out for honest scoring (`--val-only`). Bootstrap prompt forbids hardcoded `if english == "..."` shortcuts and keys everything to structured `Sentence` inputs.
3. The produced `yaduha-{iso}` package is a standalone Python package with Pydantic `Sentence` subclasses. Lemma fields are typed `Literal[...]` from the package's vocab list, so the LLM cannot emit OOV terms. `__str__()` is deterministic Python — no LLM at render time. Genuine proper nouns can pass through verbatim via the `proper_noun: Optional[str]` slot on `Noun`, narrowly prompted to discourage abuse.
4. `evaluate --method one-step --vlm gpt-5` runs the strongest config: a single OpenAI structured-output call with image + schema. `--method pipeline` runs the two-step variant for cheap iteration. `--method direct` is the few-shot text baseline.

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

Observed per full-dev sweep (250 rows, after the 2026-04-30 schema fixes):

| sweep | cost |
|---|--:|
| `pipeline` claude-vl + gpt-4o-mini translator + gpt-4o-mini BT | ~$3 |
| `pipeline` claude-vl + gpt-4o translator + gpt-4o-mini BT | ~$5 |
| `pipeline` claude-vl + gpt-5 translator + gpt-4o-mini BT | ~$15 |
| `pipeline` qwen2.5vl:32b VLM + gpt-4o-mini translator + gpt-4o-mini BT | ~$1 |
| `one-step` gpt-5 (single call) | **~$7** |

**Test-set submission cost (990 rows):**

| config | cost |
|---|--:|
| `one-step` gpt-5 on all 5 langs | ~$29 |
| best-per-lang (gpt-5 one-step on 4 + claude-vl+gpt-4o on bzd) | ~$30 |
| `pipeline` gpt-5 two-step on all 5 langs (with BT) | ~$100 |

**Default to `gpt-4o-mini` translator during iteration.** Cheap and saturates the structured-output task on these schemas; switching to gpt-4o or gpt-5 in the translator slot doesn't reliably move the needle.

**For peak quality, use gpt-5 one-step.** It dominates the matrix on 4/5 languages.

**Reserve open-weight translator slots** for the qwen2.5vl:32b VLM step (free, near-cloud quality on the VLM side). Open-weight translators are unreliable with strict-Literal schemas — they fail validation on most rows.

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

- **Morphological renderer for grn**: implement the t/r/h consonant alternation for possessed nouns and verb subject prefixes. Currently the package emits base forms (e.g. `tetã` for "country") while gold uses possessed forms (`ñane retã` = "our country"); chrF doesn't credit the overlap. ~2-3 hours of careful Guaraní morphology work.
- **Adjective concept for hch**: Wixárika doesn't use English-style attributive adjectives heavily, but a small adjective slot + property-as-stative-verb pattern could absorb some content currently dropped. The 2026-04-30 vocab probe found web-sourced Wixárika color terms don't appear in dev gold — needs a real Wixárika dictionary, not blog references.
- **Cultural-NER prompting**: for grn especially, gold often names specific places ("Mercado 4", "Panteón de los Héroes") that the VLM doesn't visually identify. The `proper_noun` infrastructure is in place; the bottleneck is VLM-side recognition.
- **Auto-finetuning** ([`docs/auto_finetuning_spec.md`](docs/auto_finetuning_spec.md)): port the OVP `feature/weakmodels` recipe (LoRA-fine-tune Qwen2.5-3B as the structured-output translator, training data synthesized from the language package itself). Pre-2026-04-30 schema, this was on hold; with strict Literal schemas, it's now necessary if we want viable open-weight translators.

## Sister projects

[yaduha](https://github.com/kubishi/yaduha-2) (framework) · [yaduha-ovp](https://github.com/kubishi/yaduha-ovp) (reference + `feature/weakmodels` open-weight branch) · [paper_yaduha_open_weight](https://github.com/kubishi/paper_yaduha_open_weight) (the WIP fine-tuning paper) · [conlang-claude](https://github.com/kubishi/conlang-claude)
