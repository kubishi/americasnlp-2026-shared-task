# Auto-Finetuning Spec — Porting yaduha-ovp/feature/weakmodels to AmericasNLP 5 langs

> **Status: deferred.** Spec only — not currently being executed. Decision pending whether to proceed with Path A (open-weight LoRA, paper-narrative-strong) or Path B (OpenAI fine-tune of `gpt-4o-mini`, lower friction).

The [`feature/weakmodels` branch of yaduha-ovp](https://github.com/kubishi/yaduha-ovp/tree/feature/weakmodels), accompanied by the [paper draft](https://github.com/kubishi/paper_yaduha_open_weight), demonstrates that a LoRA-fine-tuned Qwen2.5-3B can match or beat `gpt-4o-mini` on the LLM-RBMT structured-output translation step for OVP, running locally on a consumer GPU. This document spells out what it would take to apply the same recipe to our five AmericasNLP languages.

## What the OVP recipe does

Six datagen stages, all using primitives from yaduha + a strong-model "paraphraser" (gpt-4o-mini in the paper):

1. **Structure sampling.** Stratified sampling from each language's `Sentence` schema (1–3 clauses, SV vs SVO, plain vs nominalized, in-vocab vs OOV).
2. **Paraphrase generation.** Backward translator renders each structured target as canonical English; gpt-4o-mini rewrites it under 8 transforms (passivize, add_relative_clause, idiomize, etc.).
3. **OOV substitution.** Curated `(OOV english, in-vocab hypernym)` pairs (e.g. `chihuahua → dog`) teach the model when to substitute and when to preserve as `[english]`.
4. **Proper-noun coreference.** Personal names always preserved as `[Name]`; multi-clause coreference cases included.
5. **Backward training pairs.** Each structured target also yields one clean + one masked backward training pair (for the `structured → English` adapter).
6. **Assembly.** Stratified 90/10 train/val split, ~4500 pairs from 750 sampled targets.

LoRA fine-tune Qwen2.5-3B-Instruct with [Outlines](https://github.com/dottxt-ai/outlines)-style grammar-constrained decoding at inference. Forward + backward adapters separately.

**OVP result:** ft-qwen2.5:3b leads `gpt-4o-mini` on COMET_c (0.473 vs 0.448), 0 parse failures, runs locally.

## What we need to add per language

Three concrete code changes per `yaduha-{iso}` package:

1. **`Sentence.masked_copy()`** — return a deep-copied Sentence with vocabulary-constrained fields (Noun heads, Verb lemmas) replaced by role-tagged sentinels (`[NOUN]`, `[VERB]`), plus the list of original OOV tokens. Required by the OOV-substitution datagen and the comparator-arm evaluation.
2. **`Sentence.str_masked()`** — render but with placeholders held as sentinels. Default impl just calls `__str__`; only override if the language emits OOV `[english]` fallbacks.
3. **`Sentence.sample_iter(n)` / `sample(n)`** — random structured sentence generation. Some of our generated packages already have this from the agent's modeling of yaduha-hch; we should audit.

Plus one shared piece across all 5 languages:

4. **OOV lemma list** — `(english, hypernym_in_vocab)` pairs, e.g. `(chef, cook)`, `(mansion, house)`. Per-language because vocab differs. The agent could generate these via a dedicated tool: read vocab.py, propose 50–100 OOV-hypernym pairs whose hypernym is in vocab.

## Estimated effort

| step | effort | who |
|---|---|---|
| Add `masked_copy` / `str_masked` to `yaduha` core | already done (in `feature/weakmodels` branch) | pull from upstream |
| Update our `yaduha-{iso}` packages to override `masked_copy` per Sentence type | ~30 min × 5 langs (if done by hand) | agent could do this in a focused run |
| Generate OOV lemma list per language | ~10 min × 5 langs (agent or hand) | agent |
| Run the datagen pipeline (paraphrase via gpt-4o-mini) | ~$5 × 5 langs ≈ $25 | scripted |
| LoRA fine-tune Qwen2.5-3B per language (forward) | ~1 hr × 5 langs on our 2× RTX 5000 | scripted (axolotl / unsloth / TRL) |
| Train one shared backward adapter (structured-JSON → English is language-agnostic enough that one adapter might work across) | ~1 hr | scripted |
| Hook the adapters into our captioner (ollama backend already present) | ~1 hr | hand |
| Re-run dev eval matrix with the fine-tuned adapter | runs locally, free | scripted |

Total: roughly **a day of automation work + a few hours of GPU time**, plus ~$25 in OpenAI calls for paraphrase generation.

## Two parallel paths

**Path A: Port the OVP recipe verbatim.**
- Pull weakmodels-branch code from yaduha + yaduha-ovp/experiments
- Adapt the `experiments/datagen/` scripts to take `--lang` arg
- Override `masked_copy()` per package
- Train one adapter per language

**Path B: Use OpenAI's fine-tuning API to fine-tune `gpt-4o-mini` on the same datagen output.**
- Same datagen pipeline
- Format pairs as OpenAI fine-tune JSONL
- ~$3-5/lang to fine-tune; cheap inference at $1.50/1M tokens (vs $25/1M for gpt-5)
- No GPU needed locally; integrates trivially into existing captioner
- Produces a `ft:gpt-4o-mini-2024-07-18:org:our-name:abc123` model ID

Path B is lower-friction (no LoRA infra, no quantization, no Outlines) and gives us a model that's both stronger than off-the-shelf gpt-4o-mini AND cheaper to call than gpt-5. For paper-narrative purposes Path A is more interesting (true open-weight story), but for actual leaderboard performance Path B may be the higher-leverage bet.

## Concrete proposal

1. **Local VLM is already wired** (this commit) — qwen2.5vl:7b and :32b available via `--vlm qwen2.5vl:32b`. Free, decent quality, sweep just kicked off.
2. **Auto-finetuning** — propose Path B first (faster, cheaper, lower-friction), measure lift on dev, then decide whether Path A is worth the additional work for the paper story.
3. **Decision point**: if Path B's fine-tuned `gpt-4o-mini` matches or beats `claude-sonnet-4-5` / `gpt-5` on dev, we have our submission models at submission-affordable cost.

## Open questions for the team

- Do we want the paper to feature open-weight (Path A) or just leaderboard performance (Path B)? Affects effort allocation.
- The paper draft is currently OVP-only; extending to 5 AmericasNLP langs could be a separate paper or an appendix.
- For Path A, where do we host the fine-tuned adapters so reviewers can reproduce?
