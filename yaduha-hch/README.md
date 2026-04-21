# yaduha-hch

Wixárika (Huichol) language package for the [Yaduha](https://github.com/kubishi/yaduha-2) structured translation framework.

Wixárika (ISO 639-3: `hch`) is a Uto-Aztecan language spoken by the Wixárika people primarily in the Sierra Madre Occidental of western Mexico (Jalisco, Nayarit, Durango, Zacatecas). This package was developed for the [AmericasNLP 2026 Shared Task](https://github.com/AmericasNLP/americasnlp2026) on culturally grounded image captioning.

## Installation

```bash
pip install yaduha-hch
```

Or as an editable install for development:

```bash
pip install -e .
```

The package registers itself as a Yaduha language plugin via the `yaduha.languages` entry point. Once installed, it is automatically discoverable:

```python
from yaduha.loader import LanguageLoader

language = LanguageLoader.load_language("hch")
```

## What's Included

### Vocabulary (`vocab.py`)

63 entries organized by semantic domain: 37 nouns, 13 transitive verbs, 13 intransitive verbs. Coverage is oriented toward the image captioning domain (people, animals, food/agriculture, nature, built environment, clothing/artifacts).

### Grammar Model (`__init__.py`)

- **Word order**: SOV (Subject-Object-Verb) for transitive clauses, SV for intransitive
- **Person agreement**: 6-person system (1sg, 1pl, 2sg, 2pl, 3sg, 3pl) with subject prefixes on verbs (`ne-`, `te-`, `pe-`, `xe-`, `∅`, `me-`) and object prefixes for transitives
- **Tense/aspect**: Present (unmarked), past (`-k+`), progressive (`-t+`), habitual (`-ame`)
- **Number**: Singular/plural with semantically-conditioned plural suffixes (`-ri` for female humans and domestic animals, `-tsi` for male humans and small animates, `-xi` for inanimates, `-te` for objects and body parts, `-ma` for kinship terms)
- **Two sentence types**: `SubjectVerbSentence` (intransitive) and `SubjectVerbObjectSentence` (transitive)

### LLM Prompts (`prompts.py`)

System prompts for LLM-based translation including vocabulary lists, sentence structure documentation, and few-shot examples. Supports both English and Spanish as source languages.

## Orthography

This package uses the practical orthography common among Wixárika writers:

- **`+`** represents the high central unrounded vowel /ɨ/ (the sixth vowel of Wixárika)
- **Acute accents** (`á`, `é`, `í`, `ú`) mark non-default stress (default stress is penultimate)
- Long vowels are written by doubling: `aa`, `ee`, `ii`, `uu`, `++`
- **`'`** represents the glottal stop /ʔ/
- **`x`** represents a retroflex fricative
- **`ts`** represents an affricate

## Data Sources

### Primary: AmericasNLP 2026 Pilot Data

20 Wixárika-Spanish parallel image captions from the [AmericasNLP 2026 Shared Task](https://github.com/AmericasNLP/americasnlp2026) pilot set. These provided:

- Direct vocabulary extraction through Spanish-Wixárika alignment (nouns like `uká`/woman, `tsapa`/fish, `ikwai`/food, `paapá`/tortilla, `puritu`/horse, `wakaitsixi`/bulls, `muxatsi`/sheep, `kiekari`/community; verbs like `ikata`/carry, `mati`/harvest, `wipa`/embroider, `ekwá`/wash, `tsunawa`/rest, `hupú`/live)
- Morphological pattern identification: the `me-` third person plural prefix, `-t+` progressive suffix, `-k+` past suffix, and `-ame` habitual suffix are all directly observable in the training captions
- Plural morphology: `uká` → `ukari` (women), `wakaitsi` → `wakaitsixi` (bulls) visible in the data
- Confirmation of SOV word order tendencies

### Grammar References

- **Iturrioz Leza, J. L. & Gómez López, P.** *Gramática Wixárika.* Munich: Lincom Europa. — Reference grammar informing the person prefix system, tense/aspect suffixes, and plural formation rules.
- **Grimes, J. E. (1964)** *Huichol Syntax.* The Hague: Mouton. (SIL archives) — Early syntactic description confirming SOV order and verb morphology template.
- **Mager, M., Mager, E., Medina-Urrea, A., Meza, I., & Kann, K. (2018)** "Lost in Translation: Analysis of Information Loss During Machine Translation of Public Health Database Content." *Journal of Intelligent & Fuzzy Systems.* — Describes a probabilistic FST morphological analyzer for Wixárika with 8 prefix positions and 23 suffix positions.
- **Iturrioz Leza, J. L. (2004)** Contrastive analysis of nominal number in Spanish and Wixárika, published in *Energeia Online.* — Primary source for the semantically-conditioned plural suffix system (`-ri`, `-tsi`, `-xi`, `-te`, `-ma` and compound suffixes like `-rixi`, `-texi`, `-teri`).
- **Banerji, N. (2015)** "Huichol (Wixarika) Word Accent." SSRN. — Documents the privative tone system where accent position can distinguish tense.

### Vocabulary Databases

- **ASJP Database** (Automated Similarity Judgment Program) — Huichol wordlist providing core vocabulary: independent pronouns (`ne` = I, `eki` = you, `tame` = we), and basic nouns (`há` = water, `tai` = fire, `táu` = sun).
- **native-languages.org** — Huichol pronunciation guide and basic word list, cross-referenced for animal and nature terms.
- **rankeamexico.com/lengua-huichol** — 100+ word Wixárika-Spanish vocabulary list used to supplement and verify entries (animals, plants, numbers, kinship terms).
- **SIL International archives** — Huichol texts, dictionary materials, and the *Vocabulario Huichol-castellano* (catalog #54282).

### Parallel Corpus

- **Mager et al., Wixárika-Spanish Parallel Corpus** (https://opencor.gitlab.io/corpora/mager18wixarika/) — Used as reference for morphological segmentation patterns and verb prefix slot ordering.

## Limitations

- The grammar model is simplified compared to the full complexity of Wixárika verbal morphology (which can have up to 20 morphemes per verb). Modal/evidential prefixes (`p+-` first-hand, `mi-` hearsay), directional prefixes, applicative suffixes, and causative morphology are not yet modeled.
- Vocabulary was extracted from a small pilot dataset (20 sentences) and supplemented with dictionary sources that may reflect different dialects. Wixárika has four recognized dialect groups (north, south, east, west).
- Some vocabulary entries are tentative and should be verified by native speakers, particularly verb stems which may appear in inflected forms in the training data.
- The `tsapa` (fish) entry may be dialect-specific or species-specific; other sources give `kechí` or `kesi` as the generic term for fish.

## License

See LICENSE.md for details.
