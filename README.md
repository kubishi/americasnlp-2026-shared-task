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
| **structured** | Image sent to OpenAI vision model with structured outputs constrained to `yaduha-hch` sentence types (`SubjectVerbSentence`, `SubjectVerbObjectSentence`). Outputs are guaranteed to conform to the grammar model defined in `yaduha-hch`. |
| **translator-pipeline** | Image captioned in English, then translated to Wixárika via `PipelineTranslator` using structured language outputs. |
| **translator-agentic** | Image captioned in English, then translated to Wixárika via `AgenticTranslator` with vocabulary/grammar system prompt. Free-form generation, less constrained. |

> **Note on linguistic accuracy:** The `yaduha-hch` language package was written using Claude, using the 20 pilot training examples and publicly available grammar references (Iturrioz Leza, ASJP, SIL). The authors do not speak Wixárika. The structured methods guarantee correctness *according to the written grammar model* (valid morphological concatenation, correct person prefixes, etc.), but we cannot guarantee that the model itself accurately represents the language. The grammar and vocabulary should be reviewed by a Wixárika speaker.

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
- **structured** is the most constrained approach. It always produces valid output but is limited by the grammar model's coverage, concepts outside the vocabulary (e.g., "tractor", "green beans", "taco") appear as bracketed English placeholders.
- All methods struggle with complex reference sentences. The Wixárika references in the pilot set contain sophisticated morphology (multi-prefix verb forms, clause chains, pragmatic particles) that far exceeds what the current grammar model can generate.
- ChrF++ scores in the 8-15 range reflect partial lexical overlap rather than fluent translation. This is expected given the minimal grammar and vocabulary (~63 entries) in `yaduha-hch`.

## TODO

### Fix agentic translator bugs
- The `translator-agentic` method hits the output token limit on 6/20 examples due to repetitive degeneration (looping phrases like `'i-kwai-t+ háne` endlessly)
- `OpenAIAgent` was missing `max_tokens` support (now added with 4096 default): need to verify this resolves the failures and push the fix upstream to yaduha
- Investigate whether the `AgenticTranslator` needs better stop/truncation logic or if the system prompt needs restructuring to prevent loops
- Pydantic serialization warnings (`PydanticSerializationUnexpectedValue` on `parsed` field): harmless but should be fixed in yaduha's `AgentResponse` type annotations

### Improve the Wixárika language model (`yaduha-hch`)
- **Expand vocabulary**: Currently only 37 nouns, 13 transitive verbs, 13 intransitive verbs: many image concepts fall back to bracketed English placeholders
- **Add sentence types**: Only SV and SOV patterns are modeled; add support for locative sentences, possessive constructions, copular sentences, and multi-clause structures
- **Richer morphology**: Model verb serialization, applicatives, causatives, and more complex prefix stacking (the reference translations use forms like `me yu ku há arit+wat+` that are well beyond current coverage)
- **Evaluation-driven development**: Use the 20 pilot reference translations as a test suite. Measure ChrF++ improvement as vocabulary and grammar are expanded. Look for additional Wixárika corpora or wordlists (Iturrioz Leza grammars, SIL resources, ASJP) to bootstrap coverage

### Explore other approaches and prepare for additional languages
- Try different base models (`gpt-4o` vs `gpt-4o-mini`) and compare cost/quality tradeoffs
- Experiment with few-shot prompting using the pilot training examples as in-context demonstrations
- Investigate fine-tuning or retrieval-augmented generation for low-resource translation
- The shared task will announce additional languages: design the evaluation pipeline to be language-agnostic so new `yaduha-*` packages can be plugged in with minimal changes
- Consider ensemble methods (e.g., structured + agentic with reranking by ChrF++ against back-translations)

## Dependencies

- [yaduha](https://github.com/kubishi/yaduha-2): Structured translation framework
- [yaduha-hch](https://github.com/kubishi/yaduha-hch): Wixárika language package
- [sacrebleu](https://github.com/mjpost/sacrebleu): ChrF++ scoring
- [matplotlib](https://matplotlib.org/): Results visualization
- [openai](https://github.com/openai/openai-python): Vision model API

## License

See individual submodule repositories for license details.
