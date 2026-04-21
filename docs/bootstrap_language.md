# Generating a `yaduha-{iso}` package

This is the core research artifact of the project. The captioning pipeline
is a thin wrapper around `yaduha`; the *interesting* claim is that the
language packages it consumes can be authored end-to-end by an agent given
only:

1. The training-set image→target_caption pairs for the language.
2. Whatever the agent can find on the open web (published grammars,
   dictionaries, SIL/ASJP/Glottolog references, Wiktionary).

The system description paper will report the prompt verbatim, the
generation traces (token usage, iterations, web queries), and the captioning
quality of the resulting packages.

## Run it

```bash
# Authenticate Anthropic and OpenAI:
echo OPENAI_API_KEY=sk-...     >> .env
echo ANTHROPIC_API_KEY=sk-...  >> .env

# Generate one language. Bribri shown; works for any of {bzd, grn, yua, nlv}.
uv run americasnlp generate-language --iso bzd

# Knobs:
uv run americasnlp generate-language --iso bzd \
    --model claude-opus-4-7 \
    --effort high \
    --max-iterations 60 \
    --overwrite-scaffold     # only if you want to wipe an existing draft
```

The agent runs to either:
- `validate_package` returning `PASS: True` with zero placeholder leaks, then
  emitting `end_turn`, **or**
- `--max-iterations` (default 60) being reached.

Either way, a `yaduha-{iso}/` directory is left on disk with whatever the
agent produced. Inspect it, run the acceptance check manually if needed, and
either keep iterating or merge.

## What the agent has access to

| Tool                           | Side          | Purpose                                                          |
|--------------------------------|---------------|------------------------------------------------------------------|
| `web_search` / `web_fetch`     | server        | Open-web research with citations (Anthropic-hosted)              |
| `list_reference_files`         | client        | List source files in `yaduha-hch` or `yaduha-ovp`               |
| `read_reference_file`          | client        | Read source from a reference package (the canonical conventions) |
| `read_training_captions`       | client        | Up to N (`id`, `target_caption`) pairs from dev + pilot          |
| `extract_content_words`        | client        | Frequency-rank tokens across the training captions               |
| `list_package_files`           | client        | List files inside the in-progress `yaduha-{iso}/`                |
| `read_package_file`            | client        | Read a file inside `yaduha-{iso}/`                               |
| `write_package_file`           | client        | Overwrite a file inside `yaduha-{iso}/` (refuses pyproject.toml) |
| `validate_package`             | client        | Acceptance check — import, render, vocab counts, coverage        |

Source: `src/americasnlp/generator/agent.py`. Implementation of each tool:
`_execute_custom_tool` in the same file.

## The acceptance contract (what `validate_package` checks)

```
package: yaduha-{iso}
importable: True/False                ← module imports without error
language.name: '...'                  ← `language` is a yaduha.Language
sentence_types:
  SubjectVerbSentence: N examples, 0 placeholder leak(s), 0 empty render(s)
  SubjectVerbObjectSentence: N examples, 0 placeholder leak(s), 0 empty render(s)
vocab: NOUNS=N, TRANSITIVE_VERBS=N, INTRANSITIVE_VERBS=N
training-token coverage: X.X%
PASS: True
```

A package "passes" when:
- It imports cleanly.
- `language.sentence_types` is non-empty.
- Every sentence type's `get_examples()` renders to a non-empty string.

A package can pass with `placeholder_leaks > 0` (out-of-vocab lemmas in the
package's *own* examples), but the loop is instructed to push leaks to zero.
Training-token coverage isn't a pass/fail criterion — it's a signal the
agent uses to decide whether the vocabulary is rich enough.

## Cost & reproducibility

- Default model is `claude-opus-4-7` with adaptive thinking and `effort=high`.
  Drop to `claude-sonnet-4-6` if Opus runs are too expensive.
- The system prompt is cached (`cache_control: ephemeral`) so re-runs against
  the same training data reuse the prefix — verify with the printed
  `cache_read` token count.
- The reference package (`yaduha-hch`) is pinned via the submodule, so an
  agent run is reproducible up to model version + web-search non-determinism.

## Reviewing the agent's output

Before merging:

- [ ] `uv run americasnlp evaluate --language {key} --method pipeline --limit 5 --split dev`
      returns sane-looking captions.
- [ ] `vocab.py` has a header comment listing its sources.
- [ ] Every distinct target form in `vocab.py` is attested by at least one
      cited reference (spot-check 5 random entries).
- [ ] `__init__.py` does not contain any heuristic-rendered target strings —
      every target token must trace to `vocab.py` or to a morphology rule.

## Why an agent for this and not just a hand-written package?

Three reasons, in priority order:

1. **The contribution is the workflow, not the linguistic accuracy.** We
   are explicitly making a claim about what an agent can produce given a
   small training set + open-web research. That claim only means anything
   if the workflow is reproducible end-to-end, not "we wrote it ourselves
   then tested an agent's prompt against the same task."
2. **Coverage scales.** AmericasNLP adds languages most years. A working
   generator drops the per-language onboarding cost from weeks to one
   command + a code review.
3. **Honest baselines.** Hand-tuned packages would over-fit to the dev set
   and exaggerate the pipeline's quality. A generator-authored package
   gives a clean read on the LLM-RBMT paradigm itself.

If the agent fails for a particular language, that's a result worth
reporting — it tells us where the workflow's limits are.
