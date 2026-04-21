"""Guaraní (grn) — Yaduha language package.

Paraguayan Guaraní is predominantly SVO and agglutinative.  Verbs take
person-marking prefixes for the subject (and separate object prefixes
which we gloss over here):

    a-   1sg      ja-  1pl incl     o-   3 (sg/pl; 3pl optionally +-kuéra)
    re-  2sg      ro-  1pl excl
    pe-  2pl

Tense/aspect marking is largely analytic; we support three values:
    present  -- unmarked (the default in narration)
    past     -- suffix "-kuri" (recent past; Gregores & Suárez 1967 §4.5)
    future   -- suffix "-ta"

Plural number on nouns is marked with the clitic "-kuéra".

Sentence inventory (tuned to image-caption training data):
    1. SubjectVerbSentence           – intransitive SV
    2. SubjectVerbObjectSentence     – transitive SVO (canonical Guaraní order)
    3. CopularSentence               – predicate nominal "X ha'e (peteĩ) Y"
    4. LocativeSentence              – "X oĩ Y-pe" existential / location
    5. AndSentence                   – "CLAUSE_1 ha CLAUSE_2" (binary coord.)

Noun modification: the `Noun` model carries two optional slots —
    * `modifier`  — an adjective lemma placed *after* the head noun
                    (Guaraní attributive order: "óga morotĩ" = "white house";
                     Gregores & Suárez 1967 §5.2).
    * `material`  — an "ojejapóva X-gui" participial phrase
                    (~"which is made of X"; productive in the training
                     captions: "ta'anga ojejapóva yvyrágui" = "figure
                     made of wood").

Copular and locative sentences were added because the training captions
are image descriptions of the form "a chipa, made of starch and cheese",
"a statue which is in the square", etc. — they predominate over bare
action sentences.  AndSentence was added because the VLM front-end
segments image descriptions into many short clauses; coordinating
related facts preserves meaning that would otherwise be lost.

References
----------
Gregores, E. & Suárez, J. (1967) *A description of colloquial Guaraní*.
Velázquez-Castillo, M. (2002) "Grammatical relations in active systems."
Wiktionary Guaraní lemmas:  https://en.wiktionary.org/wiki/Category:Guarani_lemmas
"""
from __future__ import annotations

import re
from enum import Enum
from random import choice, randint, random
from typing import Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_grn.vocab import (
    ADJECTIVES,
    INTRANSITIVE_VERBS,
    NOUNS,
    TRANSITIVE_VERBS,
)


# ---------------------------------------------------------------------------
# Whitespace / punctuation tidy pass
# ---------------------------------------------------------------------------
_ORPHAN_POSTPOS_RE = re.compile(r"(^|\s)-[^\s-]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _tidy(s: str) -> str:
    """Collapse whitespace and drop orphaned postposition suffixes.

    If the VLM hands us a partial Noun (empty head with modifier / material
    set, or an empty location NP), earlier rendering can leave doubled
    whitespace or a lone "-pe" floating at the start of a constituent.  This
    pass normalises both so the final target string stays well-formed.
    """
    s = _WS_RE.sub(" ", s).strip()
    # Remove tokens like " -pe", " -gui" that appear after an empty NP.
    s = _ORPHAN_POSTPOS_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
NOUN_LOOKUP: Dict[str, VocabEntry]              = {e.english: e for e in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry]   = {e.english: e for e in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in INTRANSITIVE_VERBS}
ADJECTIVE_LOOKUP: Dict[str, VocabEntry]         = {e.english: e for e in ADJECTIVES}


def get_noun_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
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


def get_verb_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_adjective_target(lemma: str) -> str:
    if not lemma:
        return ""
    if lemma in ADJECTIVE_LOOKUP:
        return ADJECTIVE_LOOKUP[lemma].target
    return f"[{lemma}]"


# ---------------------------------------------------------------------------
# Grammatical categories
# ---------------------------------------------------------------------------
class Number(str, Enum):
    singular = "singular"
    plural = "plural"


class TenseAspect(str, Enum):
    present = "present"
    past    = "past"
    future  = "future"

    def get_suffix(self) -> str:
        if self == TenseAspect.present:
            return ""
        if self == TenseAspect.past:
            return "kuri"
        if self == TenseAspect.future:
            return "ta"
        raise ValueError(f"Unknown tense/aspect: {self}")


class Person(str, Enum):
    first_sg  = "I"
    first_pl  = "we"
    second_sg = "you"
    second_pl = "you (plural)"
    third_sg  = "he/she/it"
    third_pl  = "they"


# Independent subject pronouns (used as free-standing subjects).
SUBJECT_PRONOUNS: Dict[Person, str] = {
    Person.first_sg:  "che",
    Person.first_pl:  "ñande",     # inclusive; most common narrative 1pl
    Person.second_sg: "nde",
    Person.second_pl: "peẽ",
    Person.third_sg:  "ha'e",
    Person.third_pl:  "ha'ekuéra",
}

# Subject-marking prefixes on verbs.
SUBJECT_PREFIXES: Dict[Person, str] = {
    Person.first_sg:  "a",
    Person.first_pl:  "ja",
    Person.second_sg: "re",
    Person.second_pl: "pe",
    Person.third_sg:  "o",
    Person.third_pl:  "o",         # 3pl is 3sg prefix + optional "-kuéra"
}

# Existential / locative copula "ime ~ ĩ" inflected for person (Gregores &
# Suárez 1967 §4.4; Wiktionary: oĩ).
LOCATIVE_COPULA: Dict[Person, str] = {
    Person.first_sg:  "aime",
    Person.first_pl:  "jaime",
    Person.second_sg: "reime",
    Person.second_pl: "peime",
    Person.third_sg:  "oĩ",
    Person.third_pl:  "oĩ",
}


def _pluralize_noun(target: str) -> str:
    """Append the standard plural clitic -kuéra."""
    if not target:
        return target
    if target.startswith("[") and target.endswith("]"):
        return target  # leave placeholders untouched
    return f"{target}kuéra"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Noun(BaseModel):
    head: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A noun lemma. Known: "
                + ", ".join(e.english for e in NOUNS)
                + ". If unknown, pass the English word as a placeholder."
            )
        },
    )
    number: Number = Number.singular
    modifier: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional attributive adjective lemma, rendered AFTER the head "
                "noun (Guaraní 'óga morotĩ' = 'white house'). Known: "
                + ", ".join(e.english for e in ADJECTIVES)
                + ". If unknown, pass the English word as a placeholder. "
                "Leave null if the noun has no adjectival modifier."
            )
        },
    )
    material: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional material-of noun lemma; rendered as 'ojejapóva X-gui' "
                "(~'made of X') after the noun. Use for phrases like 'a basket "
                "of wicker', 'a jar of clay'. Known noun lemmas same as head. "
                "Leave null if the noun has no material phrase."
            )
        },
    )


class Verb(BaseModel):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A verb lemma. Known: "
                + ", ".join(e.english for e in TRANSITIVE_VERBS + INTRANSITIVE_VERBS)
                + ". If unknown, pass the English word as a placeholder."
            )
        },
    )
    tense_aspect: TenseAspect = TenseAspect.present


class TransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A transitive verb lemma. Known: "
                + ", ".join(e.english for e in TRANSITIVE_VERBS)
                + ". If unknown, pass the English word as a placeholder."
            )
        },
    )


class IntransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "An intransitive verb lemma. Known: "
                + ", ".join(e.english for e in INTRANSITIVE_VERBS)
                + ". If unknown, pass the English word as a placeholder."
            )
        },
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_noun(noun: Noun) -> str:
    """Render an NP: HEAD [MODIFIER] [ojejapóva MATERIAL-gui] [kuéra].

    If the head is missing we return "" outright, so modifier/material never
    surface as a decapitated NP.
    """
    head = get_noun_target(noun.head)
    if not head:
        return ""
    parts: List[str] = [head]
    if noun.modifier:
        adj = get_adjective_target(noun.modifier)
        if adj:
            parts.append(adj)
    if noun.material:
        mat = get_noun_target(noun.material)
        if mat:
            parts.append(f"ojejapóva {mat}-gui")
    np_str = " ".join(parts)
    if noun.number == Number.plural:
        # Plural clitic attaches to the rightmost element of the NP.
        if noun.material:
            # "X-gui kuéra" reads awkwardly; put -kuéra on the head instead.
            parts[0] = _pluralize_noun(head)
            np_str = " ".join(parts)
        else:
            parts[-1] = _pluralize_noun(parts[-1])
            np_str = " ".join(parts)
    return np_str


def _subject_person(subject: Union[Noun, Person]) -> Person:
    if isinstance(subject, Person):
        return subject
    if subject.number == Number.plural:
        return Person.third_pl
    return Person.third_sg


def _render_subject(subject: Union[Noun, Person]) -> str:
    if isinstance(subject, Person):
        return SUBJECT_PRONOUNS[subject]
    return _render_noun(subject)


def _render_verb(verb: Verb, subject: Union[Noun, Person], transitive: bool) -> str:
    stem = (get_transitive_verb_target(verb.lemma)
            if transitive else get_intransitive_verb_target(verb.lemma))
    if not stem:
        return ""
    prefix = SUBJECT_PREFIXES[_subject_person(subject)]
    suffix = verb.tense_aspect.get_suffix()
    form = f"{prefix}{stem}"
    if suffix:
        form = f"{form}-{suffix}"
    # Mark 3pl explicitly with -kuéra when the subject is an overt plural noun
    # or a 3pl pronoun (common Paraguayan Guaraní usage).
    if _subject_person(subject) == Person.third_pl:
        form = f"{form} hikuái"
    return form


def _random_noun(allow_modifier: bool = True) -> Noun:
    kwargs = dict(
        head=choice(list(NOUN_LOOKUP.keys())),
        number=choice(list(Number)),
    )
    if allow_modifier and random() < 0.3:
        kwargs["modifier"] = choice(list(ADJECTIVE_LOOKUP.keys()))
    return Noun(**kwargs)


# ---------------------------------------------------------------------------
# Sentence types
# ---------------------------------------------------------------------------
class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """Intransitive clause — Guaraní default order: Subject Verb."""
    subject: Union[Noun, Person]
    verb: IntransitiveVerb

    def __str__(self) -> str:
        subj = _render_subject(self.subject)
        v = _render_verb(self.verb, self.subject, transitive=False)
        return _tidy(" ".join(p for p in [subj, v] if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject: Union[Noun, Person] = choice(list(Person))
            else:
                subject = _random_noun()
            verb = IntransitiveVerb(
                lemma=choice(list(INTRANSITIVE_VERB_LOOKUP.keys())),
                tense_aspect=choice(list(TenseAspect)),
            )
            yield cls(subject=subject, verb=verb)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        return [
            (
                "The man walks.",
                cls(
                    subject=Noun(head="man", number=Number.singular),
                    verb=IntransitiveVerb(lemma="walk",
                                          tense_aspect=TenseAspect.present),
                ),
            ),
            (
                "I sleep.",
                cls(
                    subject=Person.first_sg,
                    verb=IntransitiveVerb(lemma="sleep",
                                          tense_aspect=TenseAspect.present),
                ),
            ),
            (
                "The small birds fly.",
                cls(
                    subject=Noun(head="bird", number=Number.plural,
                                 modifier="small"),
                    verb=IntransitiveVerb(lemma="fly",
                                          tense_aspect=TenseAspect.present),
                ),
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """Transitive clause — Guaraní default order: Subject Verb Object (SVO)."""
    subject: Union[Noun, Person]
    verb: TransitiveVerb
    object: Union[Noun, Person]

    def __str__(self) -> str:
        subj = _render_subject(self.subject)
        v = _render_verb(self.verb, self.subject, transitive=True)
        if isinstance(self.object, Person):
            # Pronominal object: use the independent pronoun after the verb.
            obj = SUBJECT_PRONOUNS[self.object]
        else:
            obj = _render_noun(self.object)
        return _tidy(" ".join(p for p in [subj, v, obj] if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbObjectSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject: Union[Noun, Person] = choice(list(Person))
            else:
                subject = _random_noun()
            verb = TransitiveVerb(
                lemma=choice(list(TRANSITIVE_VERB_LOOKUP.keys())),
                tense_aspect=choice(list(TenseAspect)),
            )
            if randint(0, 1) == 0:
                obj: Union[Noun, Person] = _random_noun()
            else:
                obj = choice(list(Person))
            yield cls(subject=subject, verb=verb, object=obj)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "The woman makes chipa.",
                cls(
                    subject=Noun(head="woman", number=Number.singular),
                    verb=TransitiveVerb(lemma="make",
                                        tense_aspect=TenseAspect.present),
                    object=Noun(head="chipa", number=Number.singular),
                ),
            ),
            (
                "The man has a red hat.",
                cls(
                    subject=Noun(head="man", number=Number.singular),
                    verb=TransitiveVerb(lemma="have",
                                        tense_aspect=TenseAspect.present),
                    object=Noun(head="hat", number=Number.singular,
                                modifier="red"),
                ),
            ),
            (
                "The workers carry a basket of wicker.",
                cls(
                    subject=Noun(head="worker", number=Number.plural),
                    verb=TransitiveVerb(lemma="carry",
                                        tense_aspect=TenseAspect.present),
                    object=Noun(head="basket", number=Number.singular,
                                material="wicker"),
                ),
            ),
        ]


class CopularSentence(Sentence["CopularSentence"]):
    """Predicate-nominal 'X is (a) Y'.

    Guaraní uses the identificational particle "ha'e" as the copula.  An
    indefinite predicate is typically introduced by "peteĩ" ("a / one").
    Rendered as:  SUBJECT  ha'e  [peteĩ]  COMPLEMENT
    """
    subject: Union[Noun, Person]
    complement: Noun
    indefinite: bool = Field(
        default=True,
        description="If True, insert 'peteĩ' before the complement (English 'a/an')."
    )

    def __str__(self) -> str:
        subj = _render_subject(self.subject)
        comp = _render_noun(self.complement)
        parts = [subj, "ha'e"]
        if self.indefinite and comp:
            parts.append("peteĩ")
        if comp:
            parts.append(comp)
        return _tidy(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["CopularSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject: Union[Noun, Person] = choice(list(Person))
            else:
                subject = _random_noun()
            comp = _random_noun()
            comp.number = Number.singular
            yield cls(subject=subject, complement=comp,
                      indefinite=bool(randint(0, 1)))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "CopularSentence"]]:
        return [
            (
                "Chipa is a food.",
                cls(
                    subject=Noun(head="chipa", number=Number.singular),
                    complement=Noun(head="food", number=Number.singular),
                    indefinite=True,
                ),
            ),
            (
                "The bird is a hummingbird.",
                cls(
                    subject=Noun(head="bird", number=Number.singular),
                    complement=Noun(head="hummingbird", number=Number.singular),
                    indefinite=True,
                ),
            ),
            (
                "The flower is medicine.",
                cls(
                    subject=Noun(head="flower", number=Number.singular),
                    complement=Noun(head="medicine", number=Number.singular),
                    indefinite=False,
                ),
            ),
        ]


class LocativeSentence(Sentence["LocativeSentence"]):
    """Existential / locative clause 'X is at Y'.

    Uses the locative copula `ime ~ ĩ` (inflected for person) followed by
    the location NP bearing the postposition "-pe" (general locative 'at,
    in, to').  Rendered as:  SUBJECT  (COP)  LOCATION-pe
    """
    subject: Union[Noun, Person]
    location: Noun

    def __str__(self) -> str:
        subj = _render_subject(self.subject)
        cop = LOCATIVE_COPULA[_subject_person(self.subject)]
        loc_np = _render_noun(self.location)
        loc = f"{loc_np}-pe" if loc_np else ""
        return _tidy(" ".join(p for p in [subj, cop, loc] if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["LocativeSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject: Union[Noun, Person] = choice(list(Person))
            else:
                subject = _random_noun()
            loc = _random_noun(allow_modifier=False)
            loc.number = Number.singular
            yield cls(subject=subject, location=loc)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "LocativeSentence"]]:
        return [
            (
                "The statue is in the market.",
                cls(
                    subject=Noun(head="statue", number=Number.singular),
                    location=Noun(head="market", number=Number.singular),
                ),
            ),
            (
                "The white flowers are in the yard.",
                cls(
                    subject=Noun(head="flower", number=Number.plural,
                                 modifier="white"),
                    location=Noun(head="yard", number=Number.singular),
                ),
            ),
            (
                "I am at home.",
                cls(
                    subject=Person.first_sg,
                    location=Noun(head="home", number=Number.singular),
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# Binary coordination:  CLAUSE_1 ha CLAUSE_2
# ---------------------------------------------------------------------------
# A restricted Union keeps the JSON Schema small while still covering the
# four main clause types; we intentionally exclude AndSentence itself from
# the Union so recursion is bounded to depth 1.
_CLAUSE_UNION = Union[
    SubjectVerbSentence,
    SubjectVerbObjectSentence,
    CopularSentence,
    LocativeSentence,
]


class AndSentence(Sentence["AndSentence"]):
    """Binary clausal coordination: 'CLAUSE_1 ha CLAUSE_2'.

    Guaraní coordinates clauses with the particle `ha` ("and").  The
    same particle coordinates NPs, but this type is only for clause-level
    coordination; coordinated NPs are better expressed by giving the noun
    a modifier or by generating two separate AndSentences.

    Use this when the English input expresses two related facts about the
    same scene ("X carries the basket AND Y cooks the food").
    """
    left: _CLAUSE_UNION = Field(
        ...,
        description="The left clause of the coordination.",
    )
    right: _CLAUSE_UNION = Field(
        ...,
        description="The right clause of the coordination.",
    )

    def __str__(self) -> str:
        left = str(self.left).rstrip(".").strip()
        right = str(self.right).rstrip(".").strip()
        if left and right:
            return _tidy(f"{left} ha {right}")
        return _tidy(left or right)

    @classmethod
    def sample_iter(cls, n: int) -> Generator["AndSentence", None, None]:
        builders = [
            SubjectVerbSentence,
            SubjectVerbObjectSentence,
            CopularSentence,
            LocativeSentence,
        ]
        for _ in range(n):
            left = next(choice(builders).sample_iter(1))
            right = next(choice(builders).sample_iter(1))
            yield cls(left=left, right=right)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "AndSentence"]]:
        return [
            (
                "The man walks and the woman sings.",
                cls(
                    left=SubjectVerbSentence(
                        subject=Noun(head="man", number=Number.singular),
                        verb=IntransitiveVerb(lemma="walk",
                                              tense_aspect=TenseAspect.present),
                    ),
                    right=SubjectVerbSentence(
                        subject=Noun(head="woman", number=Number.singular),
                        verb=IntransitiveVerb(lemma="sing",
                                              tense_aspect=TenseAspect.present),
                    ),
                ),
            ),
            (
                "The basket is on the table and the pot is in the oven.",
                cls(
                    left=LocativeSentence(
                        subject=Noun(head="basket", number=Number.singular),
                        location=Noun(head="table", number=Number.singular),
                    ),
                    right=LocativeSentence(
                        subject=Noun(head="pot", number=Number.singular),
                        location=Noun(head="oven", number=Number.singular),
                    ),
                ),
            ),
            (
                "The figure is made of wood and the worker paints it.",
                cls(
                    left=CopularSentence(
                        subject=Noun(head="figure", number=Number.singular),
                        complement=Noun(head="wood", number=Number.singular),
                        indefinite=False,
                    ),
                    right=SubjectVerbObjectSentence(
                        subject=Noun(head="worker", number=Number.singular),
                        verb=TransitiveVerb(lemma="paint",
                                            tense_aspect=TenseAspect.present),
                        object=Person.third_sg,
                    ),
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# Language object
# ---------------------------------------------------------------------------
language = Language(
    code="grn",
    name="Guaraní",
    sentence_types=(
        SubjectVerbSentence,
        SubjectVerbObjectSentence,
        CopularSentence,
        LocativeSentence,
        AndSentence,
    ),
)
