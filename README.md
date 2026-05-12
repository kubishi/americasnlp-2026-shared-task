# AmericasNLP 2026 — LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html).

## Approach

```
image → [VLM with strict-Literal schema] → SentenceList → Sentence.__str__() → target
```

For each target language, an agent writes a Yaduha-compatible Pydantic grammar package. Lemma fields are typed `Literal[...]` enums drawn from the package's vocab list, so a VLM emits structured `SentenceList` JSON that Python deterministically renders into the target language. No training, no fine-tuning, no silent OOV — Pydantic rejects out-of-vocabulary lemmas at validation time.

Proper nouns escape through a narrow `proper_noun: Optional[str]` slot on `Noun`.

## Final submission

**gpt-5 `one-step`** (single VLM call: image + schema → `SentenceList`) on all 5 languages.

Dev results (ChrF++, N=50):

| iso | baseline | ours |
|---|--:|--:|
| bzd | 7.57 | 9.62 |
| grn | 20.82 | 18.57 |
| yua | — | 25.09 |
| nlv | 11.53 | 22.89 |
| hch | 17.77 | 16.08 |

**+2.44 ChrF** vs the organizer baseline on the 4 comparable languages, plus full yua coverage. Submission cost: ~$29 (990 rows).

## Usage

```bash
git clone --recurse-submodules https://github.com/kubishi/americasnlp-2026-shared-task.git
cd americasnlp-2026-shared-task
uv sync
echo "OPENAI_API_KEY=sk-..." > .env

# Generate language package (Anthropic agent, 30/20 train/val split)
uv run americasnlp generate-language --iso bzd

# Evaluate on dev
uv run americasnlp evaluate --language bribri --method one-step --vlm gpt-5

# Test submission
uv run americasnlp submit --language bribri --method one-step --vlm gpt-5 \
    --output results/submissions/bribri.jsonl
```

See [`DESIGN.md`](DESIGN.md) for the thesis and [`docs/bootstrap_language.md`](docs/bootstrap_language.md) for the generator workflow.
