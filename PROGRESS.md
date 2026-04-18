# Project Progress — AmericasNLP 2026 Shared Task

Working doc for team coordination. Everyone: please add updates under your own section and keep it terse. Dates in YYYY-MM-DD.

## Competition Timeline (from [task page](https://turing.iimas.unam.mx/americasnlp/2026_st.html))

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-02-20 | Pilot data + baseline | done |
| 2026-03-01 | Dev sets released (50 ex / language) | done — **not yet integrated** |
| 2026-04-01 | Surprise language | announced |
| 2026-04-13 | Orizaba Nahuatl added | announced (yesterday) |
| **2026-04-20** | **Test sets released** | **6 days out** |
| **2026-05-01** | **Submission deadline** | **17 days out** |
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
- **Current blockers:**
  - Bracket placeholders: when a vocabulary word has no entry in the language's vocab, `str(sentence)` renders it as `[english_word]` (e.g. `[tractor]`). These brackets will cost ChrF++ points. Need to decide: strip brackets + word, or blank the whole slot.
  - Function currently tries sentence types in registration order; a dict with both `verb` and `object` fields may match `SubjectVerbSentence` first (Pydantic ignores extra fields by default), silently dropping the object. Need to clarify with Nick whether we should prefer the most-specific type match.
- **Questions for the team:**
  - Nick: how do you want to handle the bracket placeholder issue — drop the bracketed token entirely, replace with empty string, or leave as-is and accept the ChrF++ hit?
      - Response: As we discussed, I think the best way to handle this is to omit the word entirly. We discussed other methods, and wrote them down - but for the time being this is best.  
  - Azul / Amanda: once you have a language file ready, can you share the `language_code` and a sample structured sentence so I can verify `translate_structured_sentence` works end-to-end for that language?
- **Next steps:**
  - Wire `translate_structured_sentence` into the full caption pipeline (image → caption → structured sentence → target string) alongside Nick.
  - Add blank-out logic for bracketed placeholders per the workflow spec.
  - Parameterize `scripts/evaluate.py` by `--language` so the same script runs all 5 languages (currently hardcoded to Wixárika).

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
- **What I worked on:**
Changed pipeline.py to omit words it can't translate
- **Current blockers:**
- **Questions for the team:**
- **Next steps:**
- Add `translate_structured_sentence` into the full caption pipeline with Diego. 


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
