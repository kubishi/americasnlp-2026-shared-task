# AmericasNLP 2026 Shared Task: Wixárika Image Captioning

Evaluation framework for the [AmericasNLP 2026 Shared Task](https://github.com/AmericasNLP/americasnlp2026) on culturally grounded image captioning in Wixárika (Huichol, ISO 639-3: `hch`).

Uses the [Yaduha](https://github.com/kubishi/yaduha-2) structured translation framework to generate Wixárika captions from images via OpenAI vision models, then evaluates against reference translations using ChrF++.

## Project Structure

```
.
├── americasnlp2026/        # Shared task data (submodule)
│   └── data/pilot/
│       ├── wixarika.jsonl   # 20 pilot examples (Spanish + Wixárika captions)
│       └── images/wixarika/ # Source images
├── yaduha/                 # Yaduha translation framework (submodule)
├── yaduha-hch/             # Wixárika language package (submodule)
├── yaduha-ovp/             # Owens Valley Paiute language package (submodule)
├── scripts/
│   ├── evaluate.py         # Main evaluation script
│   └── plot_results.py     # Results visualization
├── results/
│   └── evaluation_results.csv
└── pyproject.toml          # uv workspace config
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/kubishi/americasnlp-2026-shared-task.git
cd americasnlp-2026-shared-task
uv sync
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Evaluation

### Captioning Methods

Three approaches are compared for generating Wixárika captions from images:

| Method | Description |
|--------|-------------|
| **structured** | Image sent to OpenAI vision model with structured outputs constrained to `yaduha-hch` sentence types (`SubjectVerbSentence`, `SubjectVerbObjectSentence`). Produces grammatically valid Wixárika via Pydantic models. |
| **translator-pipeline** | Image captioned in English, then translated to Wixárika via `PipelineTranslator` using structured language outputs. |
| **translator-agentic** | Image captioned in English, then translated to Wixárika via `AgenticTranslator` with vocabulary/grammar system prompt. Free-form generation, less constrained. |

### Running

```bash
# Run evaluation (default: gpt-4o-mini)
uv run python scripts/evaluate.py

# With a specific model
uv run python scripts/evaluate.py --model gpt-4o

# Custom output path
uv run python scripts/evaluate.py --output results/eval_gpt4o.csv
```

The script saves results incrementally and skips already-completed entries on re-run.

```bash
# Generate plot from results
uv run python scripts/plot_results.py
```

## Results

Pilot evaluation on 20 Wixárika image captions using `gpt-4o-mini`:

| Method | Mean ChrF++ | Min | Max | Errors |
|--------|------------|-----|-----|--------|
| structured | 8.55 | 3.87 | 14.30 | 0/20 |
| translator-pipeline | 10.49 | 3.59 | 15.47 | 0/20 |
| translator-agentic | 10.60* | 5.92 | 16.02 | 6/20 |

*\*Mean computed over 14 successful translations only. 6 examples hit the output token limit.*

### Observations

- **translator-pipeline** is the most reliable method, completing all 20 examples with the highest overall mean ChrF++ (10.49).
- **translator-agentic** achieves the highest individual scores when it works (max 16.02), but frequently hits the output token limit due to repetitive generation. The model tends to loop when producing free-form Wixárika text.
- **structured** is the most constrained approach. It always produces valid output but is limited by the grammar model's coverage — concepts outside the vocabulary (e.g., "tractor", "green beans", "taco") appear as bracketed English placeholders.
- All methods struggle with complex reference sentences. The Wixárika references in the pilot set contain sophisticated morphology (multi-prefix verb forms, clause chains, pragmatic particles) that far exceeds what the current grammar model can generate.
- ChrF++ scores in the 8-15 range reflect partial lexical overlap rather than fluent translation. This is expected given the minimal grammar and vocabulary (~63 entries) in `yaduha-hch`.

### Key Limitations

- **Vocabulary coverage**: 37 nouns, 13 transitive verbs, 13 intransitive verbs — insufficient for open-domain captioning
- **Grammar model**: Only models SV and SOV sentence patterns; no subordination, relativization, or complex verb morphology
- **Agentic instability**: The free-form translator is prone to repetitive degeneration without output length constraints
- **No Spanish path**: The pipeline currently goes Image -> English -> Wixárika, losing potential cognate/cultural alignment from the Spanish captions

## Dependencies

- [yaduha](https://github.com/kubishi/yaduha-2) — Structured translation framework
- [yaduha-hch](https://github.com/kubishi/yaduha-hch) — Wixárika language package
- [sacrebleu](https://github.com/mjpost/sacrebleu) — ChrF++ scoring
- [matplotlib](https://matplotlib.org/) — Results visualization
- [openai](https://github.com/openai/openai-python) — Vision model API

## License

See individual submodule repositories for license details.
