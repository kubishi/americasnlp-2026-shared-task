# Project Progress — AmericasNLP 2026 Shared Task

Working doc for team coordination. Everyone: please add updates under your own section and keep it terse. Dates in YYYY-MM-DD.

## Competition Timeline (from [task page](https://turing.iimas.unam.mx/americasnlp/2026_st.html))

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-02-20 | Pilot data + baseline | done |
| 2026-03-01 | Dev sets released (50 ex / language) | done — **not yet integrated** |
| 2026-04-01 | Surprise language | announced |
| 2026-04-13 | Orizaba Nahuatl added | announced (yesterday) |
| **2026-04-20** | **Test sets released** | **TODAY** |
| **2026-05-01** | **Submission deadline** | **11 days out** |
| 2026-05-08 | Winner announcement | — |
| 2026-05-13 | System description paper due | — |
| 2026-05-22 | Camera-ready | — |

**Target languages (5):** Bribri, Guaraní, Yucatec Maya, Wixárika, Orizaba Nahuatl.
**Evaluation:** ChrF++ for all systems (Stage 1), then human eval of top-5 (Stage 2).

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
- **Current blockers:**
  - Remaining 4 languages (bribri, guaraní, maya, nahuatl) not yet baselined — bribri blocked by image format issue; others ready to run.
- **Questions for the team:**
- **Next steps:**
  - Run `evaluate.py --language wixarika --split dev` end-to-end with Diego to validate `translate_structured_sentence` + bracket-omission fix together.
  - Run baseline for guaraní, maya, nahuatl once bribri image fix is in.

### Faezeh Dehghan Tarzjani
- **What I worked on:**
- **Current blockers:**
- **Questions for the team:**
- **Next steps:**

### Jared Coleman (supervisor)
- **Notes / decisions needed:**

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

---

## Agent Review — 2026-04-20

> **TEST SET DROPS TODAY. T-11 days to submission (May 1). T-23 days to paper (May 13).**

### Δ since last review (2026-04-14 in this file; 2026-04-19 on branch `ecstatic-babbage-0erka`)

| Who | Activity since Apr 14 | Key deliverables |
|-----|----------------------|-----------------|
| **Diego** | Rewrote `evaluate.py` to be language-agnostic (Apr 19). Ran 3-shot baseline for **all 5 languages** on dev. Diagnosed bribri image bug. Updated PROGRESS.md. | `evaluate.py` rewrite, 5× dev baseline CSVs |
| **Nick** | Fixed bracket placeholders in `pipeline.py` (Apr 15). Merged `baseline.py` + `languages.py` from agent branch (Apr 19). Updated PROGRESS.md (Apr 18). | `baseline.py`, `languages.py`, bracket-omit fix |
| **Amanda** | Zero commits. Zero PROGRESS.md updates. Zero visible activity. | — |
| **Azul** | Zero commits. Zero PROGRESS.md updates. Zero visible activity. | — |
| **Faezeh** | Zero commits. Zero PROGRESS.md updates. Zero visible activity. | — |
| **Jared** | Paper draft in progress (7 pages as of Apr 18, per Slack). Set up Claude daily check-in routine. Posted Issue #1. | Paper draft, daily agent check-ins |

### Baseline results — 3-shot gpt-4o-mini on dev (50 examples/language)

| Language | Our ChrF++ | Organizer Baseline | Gap | Status |
|----------|-----------|-------------------|-----|--------|
| Nahuatl | **17.08** | 11.53 | **+5.55** | Above organizer ✓ |
| Wixárika | **15.94** | 17.77 | −1.83 | Close — improvable |
| Guaraní | **14.30** | 20.82 | −6.52 | Below — needs work |
| Bribri | **11.95** | 7.57 | **+4.38** | Above organizer ✓ |
| Maya | **10.41** | n/a | — | No organizer baseline |

**Key insight:** We are already **above** the organizer baselines for Bribri and Nahuatl with a simple 3-shot prompt. Wixárika is within striking distance. Guaraní is the weakest spot — 6.5 points below organizer. These scores are from `baseline.py` (direct VLM captioning), not the structured/pipeline methods.

### Pilot-set results (evaluate.py — 20 Wixárika examples, gpt-4o-mini)

| Method | Mean ChrF++ | Completions |
|--------|------------|-------------|
| structured | 6.94 | 20/20 |
| translator-pipeline | 10.28 | 20/20 |
| translator-agentic | 11.28 | 11/20 (9 token-limit failures) |

The structured methods still only have Wixárika data (yaduha-hch) and have not been tested on dev. The agentic method's looping bug remains unfixed.

### What went RIGHT since Apr 14

1. **We now have a working multi-language submission pipeline.** `baseline.py` + `languages.py` + `make_submission.py` (on branch `wonderful-ptolemy-HcIbP`) cover all 5 languages end-to-end.
2. **Dev-set baselines exist for all 5 languages.** First time we have ChrF++ numbers on the actual evaluation data.
3. **Diego's evaluate.py rewrite is a major step.** Language-agnostic, graceful degradation, dead imports cleaned up, threading bug fixed.
4. **Nick's bracket-omit fix** removes English placeholders that were hurting ChrF++.
5. **Paper draft is underway** (Jared, 7 pages — empirical story, methodology, related work).
6. **Diego + Nick are collaborating actively** — concrete plan to wire up structured pipeline with bracket fix.

### What still needs to happen

**P0 — THIS WEEKEND (Apr 20–21): Generate submission files**

- [ ] **Merge `make_submission.py` from `claude/wonderful-ptolemy-HcIbP` into main.** This script generates the required submission JSONL format. Without it we cannot submit. — **Owner: Diego or Nick** (5-minute cherry-pick)
- [ ] **Run `baseline.py` against test sets the moment they drop.** Use the same 3-shot gpt-4o-mini config that produced the dev baselines. Generate submission files for all 5 languages. This is our safety-net submission. — **Owner: Diego**
- [ ] **Fix bribri image format bug** (3 images with wrong extensions). Add Pillow-based image normalization to `baseline.py`. — **Owner: Diego** (already diagnosed, just needs Pillow encode step)
- [ ] **Guaraní improvement.** ChrF++ 14.30 vs organizer 20.82 is a 6.5-point gap. Try: (a) increase shots to 5, (b) use gpt-4o instead of gpt-4o-mini, (c) adjust prompt (Guaraní has unique orthography). Even a 3-point gain matters. — **Owner: Nick**

**P1 — Apr 22–28: Iterate on quality**

- [ ] **Run gpt-4o on all 5 languages** and compare ChrF++ vs gpt-4o-mini. If the budget allows, gpt-4o is likely our best single-model strategy. This is the highest-ROI experiment. — **Owner: Diego** (needs Jared's compute budget decision)
- [ ] **Test evaluate.py structured + pipeline methods on Wixárika dev.** Nick's bracket-omit fix + Diego's refactored pipeline should improve Wixárika ChrF++ beyond the 15.94 baseline. — **Owner: Nick + Diego** (as planned)
- [ ] **Try `few_shot_baseline.py` from `claude/wonderful-ptolemy-UOQaf`.** Leave-one-out few-shot with CLIP similarity retrieval — might outperform random shot selection. — **Owner: anyone with API access**
- [ ] **Amanda / Azul / Faezeh:** If you can contribute anything in the next 11 days — prompt engineering, running baselines, error analysis, paper writing — please surface it NOW. Update your PROGRESS.md section or post in Slack. Any contribution helps at this stage.

**P2 — Paper track (parallel, start now)**

- [ ] **Paper outline.** Jared has a 7-page draft. Team members: read it and contribute your section (system description for your component). — **Owner: everyone**
- [ ] **Ablation table.** Per-method × per-language × per-model ChrF++ on dev. The data is there; just needs formatting. — **Owner: whoever writes the results section**
- [ ] **Merge `plot_results.py` and `analyze_results.py`** from `claude/wonderful-ptolemy-UOQaf` — needed for figures in the paper.

### Unmerged agent branches — cleanup needed

Several Claude-authored branches have useful scripts that are NOT on main:

| Branch | Key files | Status |
|--------|-----------|--------|
| `wonderful-ptolemy-HcIbP` | `make_submission.py`, `evaluate_dev.py`, `cultural_prompts.py` | **Must merge** — submission pipeline |
| `wonderful-ptolemy-UOQaf` | `plot_results.py`, `analyze_results.py`, `few_shot_baseline.py` | Should merge — paper figures + experiments |
| `wonderful-ptolemy-Rp9bs` | `baseline.py`, `languages.py`, `smoke_test.py` | Partially merged (baseline+languages on main via Nick) |

**Action: Diego or Nick — cherry-pick `make_submission.py` from `wonderful-ptolemy-HcIbP` today.**

### Code quality notes

- ✅ Dead imports in `evaluate.py` — **fixed** by Diego (Apr 19)
- ✅ `evaluate.py` language parameterization — **done** by Diego (Apr 19)
- ⬜ `main.py` is 0 bytes — delete or use as entry point
- ⬜ `plot_results.py` still missing on main (exists on `wonderful-ptolemy-UOQaf`)
- ⬜ README still references `plot_results.py` as if it exists
- ⬜ README needs `git submodule update --init --recursive` in setup instructions

### Questions for Jared (URGENT — decisions needed for submission)

1. **Compute budget:** Can we run gpt-4o for all 5 langs × test? Dev baselines show gpt-4o-mini is competitive on 3/5 languages, so even gpt-4o on just Guaraní + Wixárika (weakest vs organizer) would help.
2. **Are we submitting all 5 languages?** Agent recommendation: YES — we have baselines for all 5 and beat organizers on 2. No reason to skip any.
3. **Paper draft:** Can you share the draft link with the team so everyone can contribute? (Slack message Apr 18 had it behind Cloudflare Access.)
4. **Scope decision on structured methods:** The structured/pipeline methods (yaduha-based) are only available for Wixárika. Should we invest time testing them on dev, or focus all effort on improving the direct-prompt baseline for all 5 languages?

### Revised schedule — final sprint

| Date | Milestone | Owner |
|------|-----------|-------|
| **Apr 20 (today)** | Test sets drop. Merge `make_submission.py`. Fix bribri images. | Diego, Nick |
| **Apr 21** | Run baseline on all 5 test sets. Generate first submission files. | Diego |
| **Apr 22–24** | gpt-4o experiment. Guaraní prompt tuning. Structured pipeline on Wixárika dev. | Diego, Nick |
| **Apr 25–27** | Final iteration. Compare methods. Pick best per language. | Diego, Nick |
| **Apr 28** | Freeze submission files. No code changes after this. | All |
| **Apr 29–30** | Buffer. Final review. | Diego |
| **May 1** | Submit. | Diego |
| **May 2–13** | Paper draft and submission. | All |
