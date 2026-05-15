# AmericasNLP 2026 Shared Task: LLM-Assisted RBMT for Image Captioning

Entry in the [AmericasNLP 2026 Shared Task on Cultural Image Captioning](https://americasnlp.org/2026_st.html).

## Approach

```
image → [VLM with strict-Literal schema] → SentenceList → Sentence.__str__() → target
```

For each target language, an agent writes a Yaduha-compatible Pydantic grammar package. Lemma fields are typed `Literal[...]` enums drawn from the package's vocab list, so a VLM emits structured `SentenceList` JSON that Python deterministically renders into the target language.

## System Description and Results

See [anlp26.kubshi.com](https://anlp26.kubishi.com) for the full system description, dev results, and test submission details. In brief:

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
