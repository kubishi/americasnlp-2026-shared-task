"""Bribri (bzd) — Yaduha language package.

Bribri is a Chibchan language of Costa Rica. Key grammatical facts modelled
here (following Jara Murillo & García Segura 2013/2018 and Constenla Umaña
2023):

  * Default constituent order is **SOV** (verb-final).
  * Alignment is **split-ergative**: nominal agents of transitive verbs in
    imperfective-aspect clauses are marked with the ergative particle **tö**
    (this package inserts *tö* after a full-NP transitive subject).
  * Predicate adjectives take **no copula**:  "Ù sulë" = "The house is pretty".
  * Equative / class-assigning clauses use the copula **dör**:
    "Ie' dör awá" = "He is a shaman".
  * Locative / existential clauses use the positional-auxiliary **tso'**
    (glossed 'be.located / exist'):  "Dù tso' kàl kĩ" = "A bird is on the
    tree".  The postposition follows its NP (Bribri is postpositional):
    **a̱**  'at, in',  **ki̱**  'on',  **wa**  'with'.
  * Possession uses the **wa̱** construction:  "Ie' wa̱ cha̱mù̱ tso'"
    = 'he has coffee' (lit. 'at-him coffee is').  The possessor appears
    clause-initially with *wa̱* and the possessee stays with *tso'*.
  * Adjectives follow the head noun:  "chìchi mã̀t" = 'red dog'.

Verbs are rendered in their infinitive (-ök / -ũk) citation form. The remote
imperfective suffix -ke and the perfective inflection are deliberately NOT
generated: they require stem information (final-vowel + tone) we do not have
a lookup for, and the infinitive form is attested as a clause-final
predicate in Bribri teaching materials.
"""
from __future__ import annotations

import re
from enum import Enum
from random import choice, randint
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_bzd.vocab import (
    ADJECTIVES,
    INTRANSITIVE_VERBS,
    NOUNS,
    TRANSITIVE_VERBS,
)

# ---------------------------------------------------------------------------
# LOOKUPS
# ---------------------------------------------------------------------------
NOUN_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in NOUNS}
ADJ_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in ADJECTIVES}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in INTRANSITIVE_VERBS}

# Closed lemma vocabularies — typed as Literal so the LLM's structured
# output cannot emit out-of-vocabulary terms. The captioner prompt steers
# the VLM toward in-vocab hypernyms; this is the schema-level enforcement.
NounLemma = Literal[*tuple(sorted(NOUN_LOOKUP))]  # type: ignore[valid-type]
AdjectiveLemma = Literal[*tuple(sorted(ADJ_LOOKUP))]  # type: ignore[valid-type]
TransitiveVerbLemma = Literal[*tuple(sorted(TRANSITIVE_VERB_LOOKUP))]  # type: ignore[valid-type]
IntransitiveVerbLemma = Literal[*tuple(sorted(INTRANSITIVE_VERB_LOOKUP))]  # type: ignore[valid-type]


def get_noun_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_adjective_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in ADJ_LOOKUP:
        return ADJ_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_transitive_verb_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_intransitive_verb_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


_PLACEHOLDER_RE = re.compile(r"\s*\[[^\]]*\]\s*")


def _clean(s: str) -> str:
    """Final whitespace/placeholder scrub.

    * Bracketed placeholder tokens (e.g. ``[cap]``) are removed along with
      the whitespace that surrounds them — otherwise stripping them would
      leave adjacent particles dangling with a double space.
    * Any remaining run of whitespace is collapsed to a single space.
    """
    s = _PLACEHOLDER_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# GRAMMATICAL ENUMERATIONS
# ---------------------------------------------------------------------------
class Number(str, Enum):
    singular = "singular"
    plural = "plural"


class Person(str, Enum):
    first_sg  = "I"
    first_pl  = "we"
    second_sg = "you"
    second_pl = "you (plural)"
    third_sg  = "he/she/it"
    third_pl  = "they"


# Independent subject pronouns. [J-AB L4, L9; LB]
SUBJECT_PRONOUNS: Dict[Person, str] = {
    Person.first_sg:  "ye'",
    Person.first_pl:  "se'",    # generic/inclusive 1pl — very frequent in DEV
    Person.second_sg: "be'",
    Person.second_pl: "be'pa",
    Person.third_sg:  "ie'",
    Person.third_pl:  "ie'pa",
}


class TenseAspect(str, Enum):
    """Bribri expresses aspect primarily via suffix + auxiliary. For this
    surface-level package we realize:
      * present (unmarked, default)     — bare V-INF
      * progressive                     — "S tso' V-INF"  (positional aux)
      * past  (remote)                  — bare V-INF (form is stem-specific)
    """
    present     = "present"
    progressive = "progressive"
    past        = "past"


class Postposition(str, Enum):
    at     = "at"       # → a̱   (general locative 'at, in, to')
    on     = "on"       # → ki̱  ('on top of')
    in_    = "in"       # → a̱   (general locative)
    with_  = "with"     # → wa  (instrumental / comitative)


POSTPOSITION_FORMS: Dict[Postposition, str] = {
    Postposition.at:    "a̱",
    Postposition.on:    "ki̱",
    Postposition.in_:   "a̱",
    Postposition.with_: "wa",
}


# Plural: Bribri uses the suffix -pa for human/animate plurals, and
# typically leaves inanimate plurals unmarked (mass-like). [J-AB L9]
HUMAN_ANIMATE = {
    "person", "people", "man", "woman", "child", "children", "boy", "girl",
    "baby", "father", "mother", "sister", "brother", "uncle", "grandmother",
    "grandfather", "family", "relative", "elder", "old_woman", "old_man",
    "friend", "hunter", "doctor", "shaman", "student", "teacher",
    "foreigner", "Bribri", "Bribri_person", "group",
}


def get_plural_form(lemma: str) -> str:
    target = get_noun_target(lemma)
    if not target or target.startswith("["):
        return target
    if lemma in HUMAN_ANIMATE:
        # Human/animate plural with -pa; avoid double-marking.
        if target.endswith("pa"):
            return target
        return f"{target}pa"
    return target  # inanimate plural = bare form


# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class Adjective(BaseModel):
    lemma: AdjectiveLemma = Field(
        ...,
        description=(
            "A descriptive / color / quality lemma. Pick the closest "
            "match from the enum. Use a hypernym if the literal property "
            "isn't listed (e.g. 'crimson' → 'red')."
        ),
    )


class Noun(BaseModel):
    head: NounLemma = Field(
        ...,
        description=(
            "A noun lemma. Pick the closest match from the enum. Use a "
            "hypernym if the literal noun isn't listed (e.g. 'chihuahua' → "
            "'dog', 'mansion' → 'house', 'shaman' → 'person'). When you "
            "set 'proper_noun', still pick the closest hypernym here as "
            "a type hint."
        ),
    )
    proper_noun: Optional[str] = Field(
        default=None,
        description=(
            "Optional verbatim string for proper nouns (named entities) "
            "that lack an in-vocab lemma — e.g. 'Mercado 4', 'Panteón de "
            "los Héroes', 'Maria', 'Talamanca'. When set, this string is "
            "rendered verbatim INSTEAD OF the 'head' lemma. **Use only "
            "for actual named entities. Do NOT use as a placeholder for "
            "unknown common nouns — pick a hypernym from the lemma list "
            "instead.**"
        ),
    )
    number: Number = Number.singular
    modifier: Optional[Adjective] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional postnominal adjective modifier, e.g. 'the red dog' "
                "→ Noun(head='dog', modifier=Adjective(lemma='red')). "
                "In Bribri, adjectives follow the head noun: 'chìchi mã̀t'."
            )
        },
    )


class Verb(BaseModel):
    lemma: str
    tense_aspect: TenseAspect = TenseAspect.present


class TransitiveVerb(Verb):
    lemma: TransitiveVerbLemma = Field(
        ...,
        description=(
            "A transitive verb lemma. Pick the closest match from the "
            "enum; use a hypernym if the literal action isn't listed."
        ),
    )


class IntransitiveVerb(Verb):
    lemma: IntransitiveVerbLemma = Field(
        ...,
        description=(
            "An intransitive verb lemma. Pick the closest match from the "
            "enum; use a hypernym if the literal action isn't listed."
        ),
    )


# ---------------------------------------------------------------------------
# RENDER HELPERS
# ---------------------------------------------------------------------------
def _render_np(np: Union[Noun, Person]) -> str:
    """Render a noun phrase (may include an adjectival modifier)."""
    if isinstance(np, Person):
        return SUBJECT_PRONOUNS[np]
    if isinstance(np, Noun):
        # Proper-noun escape hatch: render verbatim if set.
        if np.proper_noun:
            head_str = np.proper_noun.strip()
        elif np.number == Number.plural:
            head_str = get_plural_form(np.head)
        else:
            head_str = get_noun_target(np.head)
        # Adjectives follow head in Bribri.
        if np.modifier is not None:
            adj = get_adjective_target(np.modifier.lemma)
            return _clean(f"{head_str} {adj}")
        return head_str
    return ""


def _np_is_known(np: Union[Noun, Person]) -> bool:
    """True if the noun phrase will render to *concrete* target-language
    content. Proper-noun-overridden NPs are also considered known (the
    verbatim string is real content even if not from the lemma list).
    """
    if isinstance(np, Person):
        return True
    if isinstance(np, Noun):
        if np.proper_noun:
            return True
        return bool(np.head) and (np.head in NOUN_LOOKUP)
    return False


# ---------------------------------------------------------------------------
# SENTENCE TYPES
# ---------------------------------------------------------------------------
class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """Intransitive sentence — Bribri SV word order.

    Rendering:
      * present/past:   "S V-INF"
      * progressive:    "S tso' V-INF"     (positional auxiliary)
    """
    subject: Union[Noun, Person]
    verb: IntransitiveVerb

    def __str__(self) -> str:
        subj = _render_np(self.subject)
        stem = get_intransitive_verb_target(self.verb.lemma)
        parts = [subj]
        if self.verb.tense_aspect == TenseAspect.progressive:
            parts.append("tso'")
        parts.append(stem)
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            verb = IntransitiveVerb(
                lemma=choice(list(INTRANSITIVE_VERB_LOOKUP.keys())),
                tense_aspect=choice(list(TenseAspect)),
            )
            yield cls(subject=subject, verb=verb)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        return [
            (
                "The woman walks.",
                cls(
                    subject=Noun(head="woman"),
                    verb=IntransitiveVerb(lemma="walk"),
                ),
            ),
            (
                "The children are playing.",
                cls(
                    subject=Noun(head="child", number=Number.plural),
                    verb=IntransitiveVerb(lemma="play", tense_aspect=TenseAspect.progressive),
                ),
            ),
            (
                "I sleep.",
                cls(
                    subject=Person.first_sg,
                    verb=IntransitiveVerb(lemma="sleep"),
                ),
            ),
            (
                "The old woman dances.",
                cls(
                    subject=Noun(head="old_woman"),
                    verb=IntransitiveVerb(lemma="dance"),
                ),
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """Transitive sentence — Bribri SOV with ergative **tö** on NP agents.

    Rendering:
      * nominal subject:    "S tö O V-INF"
      * pronominal subject: "S O V-INF"      (pronoun-subject imperfectives
                                              drop the overt ergative particle)
      * progressive adds leading auxiliary "tso'": "S tö O tso' V-INF"
    """
    subject: Union[Noun, Person]
    object: Union[Noun, Person]
    verb: TransitiveVerb

    def __str__(self) -> str:
        subj = _render_np(self.subject)
        obj = _render_np(self.object)
        stem = get_transitive_verb_target(self.verb.lemma)

        # Ergative marker only when (a) subject is a full NP, and
        # (b) the NP is known vocab — avoids dangling `tö` after a
        # placeholder gets stripped.
        use_erg = isinstance(self.subject, Noun) and _np_is_known(self.subject)

        parts: List[str] = []
        if subj:
            parts.append(subj)
        if use_erg:
            parts.append("tö")
        if obj:
            parts.append(obj)
        if self.verb.tense_aspect == TenseAspect.progressive:
            parts.append("tso'")
        if stem:
            parts.append(stem)
        return _clean(" ".join(parts))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbObjectSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            verb = TransitiveVerb(
                lemma=choice(list(TRANSITIVE_VERB_LOOKUP.keys())),
                tense_aspect=choice(list(TenseAspect)),
            )
            if randint(0, 1) == 0:
                obj = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            else:
                obj = choice(list(Person))
            yield cls(subject=subject, verb=verb, object=obj)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "The man eats corn.",
                cls(
                    subject=Noun(head="man"),
                    object=Noun(head="corn"),
                    verb=TransitiveVerb(lemma="eat"),
                ),
            ),
            (
                "The woman is grinding cacao.",
                cls(
                    subject=Noun(head="woman"),
                    object=Noun(head="cacao"),
                    verb=TransitiveVerb(lemma="grind", tense_aspect=TenseAspect.progressive),
                ),
            ),
            (
                "I see the red dog.",
                cls(
                    subject=Person.first_sg,
                    object=Noun(head="dog", modifier=Adjective(lemma="red")),
                    verb=TransitiveVerb(lemma="see"),
                ),
            ),
        ]


class CopularSentence(Sentence["CopularSentence"]):
    """Copular / predicative sentence.

    Two sub-patterns, determined by complement type:
      * **Equative / class-assigning** (NP complement):
            "S dör NP"           — 'S is (a) NP'
      * **Descriptive / qualifying** (Adjective complement):
            "S ADJ"              — 'S is ADJ'   (no copula in Bribri)

    DEV attestations:
        "Sku' wö̀a krôrô"     — The dog is round        (ADJ)
        "Ù sulë"              — The house is pretty     (ADJ)
        "Se' dör stë̀"         — We are art             (NP + dör)
    """
    subject: Union[Noun, Person]
    complement: Union[Noun, Adjective]

    def __str__(self) -> str:
        subj = _render_np(self.subject)
        if isinstance(self.complement, Adjective):
            adj = get_adjective_target(self.complement.lemma)
            return _clean(" ".join(p for p in [subj, adj] if p))
        # Noun complement — use the equative copula dör.
        comp = _render_np(self.complement)
        parts = [subj]
        # Only emit `dör` if the complement is known vocab — otherwise the
        # placeholder gets stripped and leaves a dangling copula.
        if comp and _np_is_known(self.complement):
            parts.append("dör")
            parts.append(comp)
        elif comp:
            parts.append(comp)
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["CopularSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            if randint(0, 1) == 0:
                complement = Adjective(lemma=choice(list(ADJ_LOOKUP.keys())))
            else:
                complement = Noun(head=choice(list(NOUN_LOOKUP.keys())))
            yield cls(subject=subject, complement=complement)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "CopularSentence"]]:
        return [
            (
                "The house is pretty.",
                cls(
                    subject=Noun(head="house"),
                    complement=Adjective(lemma="pretty"),
                ),
            ),
            (
                "The dog is round.",
                cls(
                    subject=Noun(head="dog"),
                    complement=Adjective(lemma="round"),
                ),
            ),
            (
                "He is a shaman.",
                cls(
                    subject=Person.third_sg,
                    complement=Noun(head="shaman"),
                ),
            ),
            (
                "The pants are yellow.",
                cls(
                    subject=Noun(head="pants"),
                    complement=Adjective(lemma="yellow"),
                ),
            ),
        ]


class LocativeSentence(Sentence["LocativeSentence"]):
    """Locative & existential sentence built on the positional copula **tso'**.

    Rendering:
      * existential (no location):   "S tso'"                 — 'there is (an) S'
      * locative (with location):    "S tso' L POSTP"         — 'S is at/on/with L'

    DEV attestations:
        "Dù tso' kàl kĩ"              — A bird is on the tree
        "Ù tso' ka̱nò̱ ki̱"              — A house is on the river
        "Chkö̀ a̱ àrros tso'"           — There is rice to eat
    """
    subject: Union[Noun, Person]
    location: Optional[Noun] = None
    postposition: Postposition = Postposition.at

    def __str__(self) -> str:
        subj = _render_np(self.subject)
        parts: List[str] = []
        if subj:
            parts.append(subj)
        parts.append("tso'")
        # Only emit the location NP + postposition if the NP is known vocab —
        # otherwise the placeholder gets stripped and the postposition dangles.
        if self.location is not None and _np_is_known(self.location):
            loc = _render_np(self.location)
            if loc:
                parts.append(loc)
                parts.append(POSTPOSITION_FORMS[self.postposition])
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["LocativeSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            if randint(0, 1) == 0:
                location = None
            else:
                location = Noun(head=choice(list(NOUN_LOOKUP.keys())))
            postposition = choice(list(Postposition))
            yield cls(subject=subject, location=location, postposition=postposition)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "LocativeSentence"]]:
        return [
            (
                "A bird is on the tree.",
                cls(
                    subject=Noun(head="bird"),
                    location=Noun(head="tree"),
                    postposition=Postposition.on,
                ),
            ),
            (
                "The house is by the river.",
                cls(
                    subject=Noun(head="house"),
                    location=Noun(head="river"),
                    postposition=Postposition.at,
                ),
            ),
            (
                "There is cacao.",
                cls(
                    subject=Noun(head="cacao"),
                ),
            ),
        ]


class PossessiveSentence(Sentence["PossessiveSentence"]):
    """Possessive / 'have' sentence — Bribri uses the **wa̱** construction,
    not a dedicated verb 'to have'. The possessor is marked with the
    possessive postposition *wa̱* and the possessee is the grammatical
    subject of the positional copula *tso'*.

    Template:                "POSSESSOR wa̱ POSSESSEE tso'"
    Literal gloss:           'at-POSSESSOR POSSESSEE is'
    Idiomatic gloss:         'POSSESSOR has POSSESSEE'

    DEV attestations:
        "Ie' wa̱ cha̱mù̱ tso' ie' ù a̱"   — he has coffee in his house
        "Ie' wa̱ dalì dàmi̱"              — he has many (things)

    Use this type for English 'X has Y', 'X's Y', 'X owns Y'.
    """
    possessor: Union[Noun, Person]
    possessee: Noun

    def __str__(self) -> str:
        poss = _render_np(self.possessor)
        thing = _render_np(self.possessee)
        parts: List[str] = []
        # Only emit 'wa̱' if the possessor is *known* vocab — otherwise it
        # would dangle after the framework strips a [placeholder].
        if _np_is_known(self.possessor) and poss:
            parts.append(poss)
            parts.append("wa̱")
        elif poss:
            parts.append(poss)
        if thing:
            parts.append(thing)
        parts.append("tso'")
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["PossessiveSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                possessor = choice(list(Person))
            else:
                possessor = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                )
            possessee = Noun(
                head=choice(list(NOUN_LOOKUP.keys())),
                number=choice(list(Number)),
            )
            yield cls(possessor=possessor, possessee=possessee)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "PossessiveSentence"]]:
        return [
            (
                "He has coffee.",
                cls(
                    possessor=Person.third_sg,
                    possessee=Noun(head="coffee"),
                ),
            ),
            (
                "The woman has a dog.",
                cls(
                    possessor=Noun(head="woman"),
                    possessee=Noun(head="dog"),
                ),
            ),
            (
                "I have many friends.",
                cls(
                    possessor=Person.first_sg,
                    possessee=Noun(head="friend", number=Number.plural),
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# LANGUAGE
# ---------------------------------------------------------------------------
language = Language(
    code="bzd",
    name="Bribri",
    sentence_types=(
        SubjectVerbSentence,
        SubjectVerbObjectSentence,
        CopularSentence,
        LocativeSentence,
        PossessiveSentence,
    ),
)
