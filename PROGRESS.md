# Project Progress — AmericasNLP 2026 Shared Task

Working doc for team coordination. Everyone: please add updates under your own section and keep it terse. Dates in YYYY-MM-DD.

## Competition Timeline (from [task page](https://americasnlp.org/2026_st.html))

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-02-20 | Pilot data + baseline | done |
| 2026-03-01 | Dev sets released (50 ex / language) | done — **integrated 2026-04-20** |
| 2026-04-01 | Surprise language | announced |
| 2026-04-13 | Orizaba Nahuatl added | announced |
| **2026-04-20** | **Test sets released** | **today** |
| **2026-05-01** | **Submission deadline** | **11 days out** |
| 2026-05-08 | Winner announcement | — |
| 2026-05-13 | System description paper due | — |
| 2026-05-22 | Camera-ready | — |

**Target languages (5):** Bribri, Guaraní, Yucatec Maya, Wixárika, Orizaba Nahuatl.
**Evaluation:** ChrF++ for all systems (Stage 1), then human eval of top-5 (Stage 2).
**System direction (set 2026-04-20):** LLM-assisted RBMT — image → VLM English caption → yaduha `PipelineTranslator` constrained by per-language Pydantic grammar → target caption. Direct VLM prompting (zero-shot, few-shot) shipped only as comparison baselines. The agentic translator is dropped. See `DESIGN.md`.

---

## Team Status

### Diego Cuadros — team lead
- **What I worked on (2026-04-15):**
  - Built `scripts/test_caption.py` — the pipeline connection between a structured sentence and the target language.
  - Function `translate_structured_sentence(sentence: Dict, language_code: str) -> SentenceTranslationResult` takes a structured sentence dict (the JSON output of `EnglishToSentencesTool`) and a language code, loads the matching language package via `LanguageLoader`, tries each of the language's Pydantic sentence types in order (`model_validate`), and calls `str(sentence)` to render the target-language string — exactly the rendering step used internally by `PipelineTranslator.translate()`.
  - Defined two Pydantic schemas: `SentenceTranslationRequest` (typed input) and `SentenceTranslationResult` (typed output with `target`, `sentence_type`, `source`, `language_code`).
  - Verified smoke test against OVP (Owens Valley Paiute) with both SV and SVO sentence dicts; target-language strings rendered correctly.
- **What I worked on (2026-04-19):**
  - Rewrote `scripts/evaluate.py` to be fully language-agnostic:
    - Added `LANGUAGE_CONFIG` dict covering all 5 competition languages with yaduha codes, display names, and data file paths for both pilot and dev splits.
    - Added `--language` and `--split` CLI args; output CSVs now auto-named `results/{language}_{split}_{model}.csv`.
    - Inlined `translate_structured_sentence` (from `test_caption.py`) directly into `evaluate.py` — replaces `PipelineTranslator` as the rendering step.
    - Replaced `PipelineTranslator` with new `PipelineCaptioner` class: image → English caption → `EnglishToSentencesTool` → `translate_structured_sentence` per sentence.
    - Replaced `StructuredCaptioner` to use `translate_structured_sentence` instead of bare `str(sentence)`.
    - Added `AgenticCaptioner` with language-specific prompt loading via `importlib` (falls back to generic prompt if no `yaduha_{code}.prompts` module exists).
    - Graceful degradation: if no yaduha package is installed for the selected language, `structured` and `translator-pipeline` are skipped automatically; only `translator-agentic` runs.
    - Removed all dead imports (`from http import client`, `from pydoc import text`, `from base64 import b64encode`, `import openai`, `import time`).
    - Fixed stale-CSV crash: reader now validates expected column headers before parsing.
    - Added `import openai.resources.chat` warm-up to prevent threading deadlock on first API call (was killing examples 1–4 on every run).
  - Diagnosed bribri image format mismatch: 3 images (`bzd_000.jpg`, `bzd_006.jpg`, `bzd_012.jpg`) have wrong extensions — WebP/unknown bytes stored as `.jpg`, causing OpenAI to reject them with `invalid_image_format`. Fix: normalize all images through Pillow before encoding (pending).
- **Current blockers:**
  - Bribri image format mismatch needs fix in `baseline.py` before bribri dev run can complete.
  - `translate_structured_sentence` + Nick's bracket-omission fix still need an end-to-end test against Wixárika dev via `evaluate.py`.
- **Questions for the team:**
  - Azul / Amanda: once you have a language file ready, share the `language_code` and a sample structured sentence so I can verify `translate_structured_sentence` works end-to-end.
- **Next steps:**
  - Fix image encoding in `baseline.py` using Pillow to handle mismatched extensions.
  - Run `evaluate.py --language wixarika --split dev` end-to-end with Nick to validate the full pipeline.
  - Pull `plot_results.py` and `smoke_test.py` from `remotes/origin/claude/wonderful-ptolemy-Rp9bs` into main (still missing).

### Amanda Avalos
- **What I worked on:**
- **Current blockers:**
- **Questions for the team:**
- **Next steps:**

### Azul Alpizar
- **What I worked on:**
- **Current blockers:**
- **Questions for the team:**
- **Next steps:**

### Nick Leeds
- **What I worked on (2026-04-15):**
  - Modified `pipeline.py` to omit untranslatable words entirely instead of leaving `[english_word]` bracket placeholders — prevents ChrF++ deductions from English tokens in output.
- **What I worked on (2026-04-19):**
  - Merged `baseline.py` and `languages.py` from agent branch `claude/wonderful-ptolemy-Rp9bs` into `main` (commit `1a8509c`) — gives the team a working multi-language submission pipeline for all 5 languages.
  - Ran `baseline.py` against Wixárika dev set (3-shot, gpt-4o-mini); results at `results/baseline/wixarika_dev_gpt-4o-mini_shots3.csv` (50 examples).
- **What I worked on (2026-04-20):**
  - Added make_submission.py from claude/wonderful-ptolemy-HcIbP, just 'merged' the file not the brach"
- **What I worked on (2026-04-27):**
  - Created `validate_submission.py` - checks that a submission folder has the right files, line counts, and format before we send it in.
  - Created `make_all_submissions.sh` - one command that builds submission folders for all 5 languages so we don't have to run `make_submission.py` five times by hand.
  - Tested both files with the current resources avalible.
- **Current blockers:**
  - Waiting on Diego to push test-set submission JSONLs to `results/submissions/`. Until those land, `validate_submission.py` has only been smoke-tested on synthetic data; the real per-language sanity check row counts, empty preds, encoding, etc cannot be run. He said he will have pushed by tommorow. 
- **Questions for the team:**
  - N/A talked to Diego over text
- **Next steps:**


### Faezeh Dehghan Tarzjani
- **What I worked on:**
- **Current blockers:**
- **Questions for the team:**
- **Next steps:**

### Jared Coleman (supervisor)
- **Notes / decisions needed:**
- **2026-04-23 — open-weight translator scan + qualitative failure-mode analysis:**
  - Qualitatively inspected qwen2.5vl:32b pipeline outputs. English intermediates are fine and rich; the target-language renders are where it fails — sparse Sentence shells, lexical ruts (same lemma reused across Sentences), and 4/250 hard-empty outputs. Representative examples at `hch_021`, `grn_044`, `yua_002`, `hch_044`.
  - Ran an open-weight translator scan on nahuatl dev (10 rows, qwen2.5vl:32b as VLM, various translators): qwen2.5vl:32b (21.69 full set) ≈ gpt-4o-mini cloud (21.52) > qwen2.5:32b text (18.54) ≈ mixtral:8x22b (17.18) ≈ llama3.1:8b (17.15) >> llama4 (9.52, degenerate single-sentence failure mode) >> qwen2.5:7b (6.09). **qwen2.5vl:32b is already the strongest local translator of what's pulled.** Surprise finding: text-only qwen2.5:32b is *worse* than its vision-language sibling at this structured-output task.
  - Decision: **stop iterating on translator-model substitution.** The gap to premium cloud doesn't live in the translator step. The languages where qwen pipeline loses to the organizer baseline (grn −7.32, hch −6.33) are exactly the two with the thinnest yaduha packages — investment belongs in language-package coverage.
  - Re-ran full-dev pipeline sweep with qwen2.5vl:32b acting as both VLM and translator (previous sweep had gpt-4o-mini as translator). New qwen-only avg **14.64** over 5 langs (4-lang 13.64); narrowly loses to organizer baseline 14.42. Result CSVs/JSONLs under `results/dev/*_pipeline_qwen2.5vl:32b.*` updated; README tables refreshed for current numbers across all cloud models too.
  - Back-burnered: defensive retry on empty Sentence shells (patch-shaped, ~1.6% of rows); translator fine-tuning (expected low ROI given captioner already produces grammar-conforming source sentences).
  - Candidates worth pulling next (not currently local): `qwen3:32b` (reported strong JSON/instruction-following lift over 2.5), `gemma3:27b`, `llama3.3:70b`, `command-r:35b`.
- **2026-04-21 (afternoon) — local VLM, one-step exploration, fine-tuning spec:**
  - **Local VLM via Ollama works.** Pulled `qwen2.5vl:7b` and `qwen2.5vl:32b`; added an `_ollama.py` backend that the captioner dispatches via model-name pattern (anything with a `:` tag that isn't `gpt-`/`claude-`/`ft:`). Full-dev sweep with `qwen2.5vl:32b` as the VLM (gpt-4o-mini still as translator) lands at avg **15.05** vs gpt-4o-mini cloud-VLM **15.16** — essentially tied for $0. Free win for low-cost iteration.
  - **One-step variant tested.** Built `OneStepCaptioner` that has the VLM emit `SentenceList` JSON directly via OpenAI structured outputs (no English intermediate). Lost on every language at gpt-4o-mini (avg ~11 vs two-step's 15). Stronger prompt steering toward in-vocab lemmas lifted yua by +5.16 and grn by +3.97 but didn't catch up to two-step. Conclusion: the English intermediate is doing real work; VLMs are weaker at structured-output JSON than text translators. Logged as a documented architectural alternative; not the primary path.
  - **Auto-finetuning spec written** (`docs/auto_finetuning_spec.md`). Port of the `feature/weakmodels` recipe from `kubishi/yaduha-ovp` + the WIP paper at `kubishi/paper_yaduha_open_weight`. Two paths: (A) local LoRA on Qwen2.5-3B per language; (B) OpenAI fine-tune of `gpt-4o-mini`. Currently on hold pending decision. Applies to the **translator step** (image→English VLM unchanged); fine-tuning data is synthesized from the yaduha-{iso} packages themselves (sample structured Sentence → render to canonical English → paraphrase) so no parallel image data needed.
  - **Code organization**: backend dispatchers split into `_openai.py`, `_anthropic.py`, `_ollama.py`. New `OneStepCaptioner` lives in `captioners/one_step.py`. `--method` choices now `pipeline | one-step | direct`.
- **2026-04-21 — model exploration + cost discipline:**
  - Test set released (990 rows: bzd 267, grn 110, yua 212, nlv 200, hch 201). Submodule bumped, format already matches our submission writer.
  - Captioners refactored to dispatch on model-name prefix: `claude-*` → Anthropic, anything else → OpenAI. Both VLM and translator paths.
  - Generated 4 model variants of dev pipeline (claude-sonnet-4-5, gpt-4o-mini, gpt-4o, gpt-5). Best per language: bzd → claude (11.17, +3.60 over baseline), grn → gpt-5 (17.18, still -3.64 vs baseline), yua → claude (25.01, baseline n/a), nlv → gpt-5 (23.82, +12.29), hch → gpt-5 (15.52, -2.25). Best-per-lang average 16.92 vs organizer baseline 14.42.
  - **Cost discipline going forward (in README):** default to `gpt-4o-mini` for iteration (~$1/sweep). Reserve `gpt-5` (~$25/sweep) and `claude-sonnet-4-5` (~$5/sweep) for final submission and one sanity-check post-refactor. Use the dev matrix to predict lift; don't re-sweep strong models for marginal changes.
  - Slack updates sent to Diego + Nick on the new direction + dev numbers.
  - Test-set submissions deferred until we're ready to finalize.
- **2026-04-20 — direction reset:**
  - Reframed the project around the LLM-RBMT thesis: a coding agent authors a
    Yaduha-compatible Pydantic grammar per language, the VLM only emits English,
    and the rendering step is deterministic. Direct VLM prompting (zero / 3-shot)
    stays as comparison baselines. Agentic translation removed.
  - Restructured the repo: `src/americasnlp/` package with a `python -m
    americasnlp {evaluate,submit}` CLI, a `Captioner` protocol, and clean dev /
    submission writers. See `DESIGN.md`.
  - Initialized submodules (none of the previous code actually ran without them).
  - Pulled the latest americasnlp2026 submodule so dev sets are present.
  - Bribri image-format bug fixed at the source: `data.image_data_url` re-encodes
    every image through Pillow, so mislabelled `.jpg`/WebP files no longer break
    the OpenAI vision API.
  - Stubbed `yaduha-{bzd,grn,yua,nlv}` as minimum-viable grammar packages so the
    pipeline runs end-to-end for all 5 languages today. They are honest stubs —
    correct typological word order, ~15-noun / 10-verb starter vocabularies, no
    morphology beyond bare lemma + tense particle. ChrF++ from the pipeline on
    these will be near floor until the bootstrap workflow expands them.
  - Wrote `docs/bootstrap_language.md` — the Claude prompt for authoring a real
    `yaduha-{iso}` package and the acceptance checklist. **This is the artifact
    we will report on in the system description paper.**
  - Deleted dead code: `main.py`, `scripts/{evaluate,test_caption,make_submission,baseline}.py`.
    All recoverable from git.

---

## Agent Review — 2026-04-14

### Status snapshot
- **Infra works, submission does not.** Baseline pipeline (`scripts/evaluate.py`) runs end-to-end on the 20-example Wixárika **pilot** set with three methods: `structured`, `translator-pipeline`, `translator-agentic`.
- **Best current result (gpt-4o-mini on pilot):** `translator-pipeline` at ChrF++ **10.49** (20/20 completed). `translator-agentic` hits 10.60 on 14/20 (6 token-limit failures). `structured` at 8.55.
- **Critical gaps vs. competition:**
  1. Only Wixárika is wired up. We need 4 more languages (Bribri, Guaraní, Yucatec Maya, Orizaba Nahuatl) by **May 1**.
  2. Dev sets (50 ex/language, released 2026-03-01) are **not being used**. Our only eval is the 20-example pilot.
  3. No commits from team members in `main` — only the initial project setup by the supervisor. No evidence in the repo of individual experiments.
  4. Last Slack activity in `#shared-task-competition` was 2026-03-30 (cancelled meeting). Radio silence for ~2 weeks.
- **Risk level: HIGH.** Six days until surprise test drops, 17 days until submission. At current pace we will not submit for 4/5 languages.

### What the data shows
- ChrF++ scores are in the 3–16 range. For context, reference translations use rich Wixárika morphology (`me yu ku há arit+wat+`-style multi-prefix verbs) that the current grammar model (~63 lex entries, only SV/SVO) cannot produce. Bracketed English placeholders (`[tractor]`, `[taco]`) appear frequently.
- `translator-agentic` degenerates into loops (`'iya 'i-kwai-t+ 'i-ikata-t+ 'i-ya-t+ 'uxi` repeated until token cap). This is the single biggest fixable defect in the current code.
- `structured` is ceilinged by vocabulary/grammar coverage in `yaduha-hch`, not by model ability.

### Priority queue (most-leverage items first)

**P0 — blockers for any submission:**
- [ ] **Pull dev sets and add them to eval loop.** `americasnlp2026/data/dev/{bribri,guarani,maya,nahuatl,wixarika}` exist in the submodule. Run the current Wixárika pipeline against `dev/wixarika` and report ChrF++ — **owner: Diego** (fastest, understands the codebase).
- [ ] **Ship a language-agnostic baseline that works for all 5 languages** even if it's weak. The competition page accepts ChrF++ scores; a 3-ChrF submission beats no submission. Simplest path: direct prompting of gpt-4o / Qwen3-VL with a few in-context examples from each dev set — **owner: Amanda + Azul** (pair up, one notebook per language).
- [ ] **Decide on model strategy.** Options: (a) OpenAI only, (b) Qwen3-VL-8B-Instruct (Diego already shared the HF link 2026-03-24), (c) ensemble. Needs Jared's input on budget/compute. **Owner: Jared decision, Nick prototype Qwen3-VL.**

**P1 — quality improvements once baselines exist:**
- [ ] Fix `translator-agentic` token-limit loop. `max_tokens=4096` is already in README TODO, but the root cause (repetitive degeneration) needs a proper fix: stronger stop conditions, lower repetition penalty, or shorter sentence-by-sentence prompting. **Owner: Faezeh** (good first code task).
- [ ] Few-shot prompting using pilot training examples as in-context demos. This is the single highest-ROI experiment per dollar of compute. **Owner: Amanda or Azul.**
- [ ] Expand `yaduha-hch` vocabulary + add locative / copular / possessive sentence types (see README TODO). This only helps `structured` and is slow — **defer unless a Wixárika speaker can be recruited.**

**P2 — post-submission / paper:**
- [ ] System description paper draft (due 2026-05-13, 12 days after submission). Start outlining now.
- [ ] Ablation table: per-method × per-language ChrF++ on dev set.
- [ ] Human eval strategy — top-5 systems get human-graded (Stage 2). Unlikely to place top-5 at current trajectory; do not optimize for this.

### Code quality issues spotted
- `scripts/evaluate.py` has dead imports (`from http import client`, `from pydoc import text`, `from base64 import b64encode`) — looks like IDE auto-imports. Clean up during the refactor.
- `scripts/plot_results.py` is referenced in README but **does not exist**. Either write it or remove the reference.
- The captioner dict hardcodes Wixárika (`yaduha_hch`, `SubjectVerbSentence`, etc.). Parameterize by `--language` so the same script handles all 5 languages.
- No unit tests. Not worth writing pre-submission, but a smoke test that loads every language and captions 1 image would catch breakage during refactors.

### Guidance on open questions
- **"Should we extend `yaduha-hch` to cover more grammar?"** — Not before April 20. Low-ROI vs. fixing `agentic` degeneration or adding few-shot prompting to `pipeline`. Revisit for the paper.
- **"Should we build separate `yaduha-bribri`, `yaduha-guarani`, etc.?"** — No. Building Yaduha-style structured grammars for 4 new languages in 17 days is not feasible. Use direct VLM prompting for the new languages; keep Yaduha only for Wixárika (and only if it actually wins on dev).
- **"Which VLM?"** — Needs Jared's call. gpt-4o-mini results are weak; gpt-4o may help but costs ~10x. Qwen3-VL is free to run locally if we have a GPU. Worth benchmarking both on Wixárika dev before committing.

### Questions for Jared (need decisions)
1. Compute budget? Can we run gpt-4o across all languages × dev + test, or are we constrained to gpt-4o-mini?
2. Is local GPU available for Qwen3-VL-8B experiments? If so, who has access?
3. Should we attempt all 5 languages, or focus on a subset (e.g., Wixárika + 2 others) and submit strong results for fewer?
4. Is there anyone we can reach for native-speaker validation for any of the 5 languages?

### Suggested schedule to hit May 1
- **By 2026-04-16 (Wed):** Dev-set eval running on Wixárika, one working notebook per language for direct-prompt baselines.
- **By 2026-04-18 (Fri):** Qwen3-VL vs. gpt-4o-mini decision made; few-shot prompting integrated.
- **2026-04-20 (Mon):** Test set drops. Run submission pipeline same day; iterate only on prompt/model choices.
- **By 2026-04-28:** Frozen submission files for all 5 languages.
- **2026-05-01:** Submit with buffer. Do not push changes on submission day.
- **2026-05-02 → 05-13:** Paper draft.
