"""Yucatec Maya (yua) language package for the Yaduha framework.

Design notes
------------
Yucatec Maya (Maayáaj t'aan) is a head-marking, predicate-initial Mayan
language spoken in the Yucatán Peninsula. The grammar this package
models, grounded in Bohnemeyer (2002) and Lehmann's grammar notes:

* Default constituent order: predicate first, then arguments. For
  transitive clauses VOS ~ VSO ~ SVO all occur; image-caption corpora
  overwhelmingly show SVO / preverbal-topic S (e.g. "Máak ku tsiktik
  kaax"), so SVO is used here.
* Aspect-first inflection. The imperfective ``k-``, perfective ``t-``
  and progressive ``táan`` particles fuse with Set A (ergative) person
  markers (``in, a, u, k, a, u``) giving ``kin / ka / ku`` etc.
* Split ergativity: Set A marks transitive subjects and (in incompletive
  aspect) intransitive subjects; Set B / absolutive marks intransitive
  subjects in completive / stative. 3sg Set B is zero — hence
  ``ku weenel`` = "is sleeping" with no subject clitic.
* Noun-phrase internal order: [numeral-classifier] [adjective] [HEAD].
  Numeral classifiers are obligatory: ``Juntúul`` for animates,
  ``Junkúul`` for plants / trees, ``Junp'éel`` for general inanimates.
* Plural: ``-o'ob`` suffix on the head noun.
* Existential / locative predication uses ``yaan`` ("there is / exists")
  which is by far the most frequent predicate in the training captions,
  so it gets its own sentence type. Its negation is ``mina'an``.
* Possession is expressed with the Set A ergative prefix on the
  possessed noun and an existential: ``N yaan u M`` = "N has M"
  (literally "N, its-M exists"). Training row:
  ``Junkúul k'áax yaan u loolo'ob`` = "A plant has flowers".
  Before a vowel-initial stem, 3sg ``u`` fuses to ``y-``:
  ``u + ich → yich`` (``Le che'o' ya'ab u yich``).
* Demonstrative/copular predication uses a zero copula:
  ``Le lela' junp'éel N`` = "This is a N".
* Clause coordination uses the particles ``yéetel`` (and / with),
  ``ba'ale'`` (but), ``beyxan`` (also), or the locative discourse
  adverbs ``paachile'`` ("in the back / then"), ``táanile'``
  ("in front / first"), ``tu tséele'`` ("at its side").

Sentence-type inventory (chosen from the training data):

1. :class:`SubjectVerbSentence` — intransitive SV.
2. :class:`SubjectVerbObjectSentence` — transitive SVO.
3. :class:`ExistentialLocativeSentence` — "N yaan [Prep N]" (or mina'an
   negation). Covers the majority of the training captions.
4. :class:`PossessiveSentence` — "N yaan u M" / "N mina'an u M" with
   3sg possessor morphophonology.
5. :class:`PredicativeSentence` — zero-copula NP-is-a-NP.
6. :class:`CoordinatedSentence` — binary-recursive clause joiner with
   attested connectives (yéetel, ba'ale', beyxan, paachile', táanile',
   tu tséele'). Bounded to depth 1.

References: Bohnemeyer (2002); Lehmann grammar notes; SIL Mexico;
Bricker, Po'ot Yah & Dzul de Po'ot (1998).
"""
from __future__ import annotations

from enum import Enum
from random import choice, randint
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from yaduha.language import Language, Sentence, VocabEntry

from yaduha_yua.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS


# ============================================================================
# LOOKUPS
# ============================================================================

NOUN_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in NOUNS}
T_VERB_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in TRANSITIVE_VERBS}
I_VERB_LOOKUP: Dict[str, VocabEntry] = {e.english: e for e in INTRANSITIVE_VERBS}

# Closed lemma vocabularies — typed as Literal so the LLM's structured
# output cannot emit out-of-vocabulary terms.
NounLemma = Literal[*tuple(sorted(NOUN_LOOKUP))]  # type: ignore[valid-type]
TransitiveVerbLemma = Literal[*tuple(sorted(T_VERB_LOOKUP))]  # type: ignore[valid-type]
IntransitiveVerbLemma = Literal[*tuple(sorted(I_VERB_LOOKUP))]  # type: ignore[valid-type]


def get_noun_target(lemma: str) -> str:
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_transitive_verb_target(lemma: str) -> str:
    if lemma in T_VERB_LOOKUP:
        return T_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_intransitive_verb_target(lemma: str) -> str:
    if lemma in I_VERB_LOOKUP:
        return I_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


# ============================================================================
# NOUN CLASSIFICATION (for numeral-classifier selection)
# ============================================================================
# The three productive numeral classifiers:
#   -túul  for animate referents (people, animals)
#   -kúul  for plants, trees, flowers
#   -p'éel general / default for inanimate objects
# The classifier attaches to the numeral stem (``jun-`` = one),
# giving ``Juntúul``, ``Junkúul``, ``Junp'éel``.

ANIMATE_NOUNS = {
    "person", "man", "woman", "lady", "child", "elder", "people",
    "boy", "girl",
    "cat", "dog", "chicken", "rooster", "bird", "butterfly", "fish",
    "ant", "horse", "cow", "pig", "deer", "snake", "animal",
}

PLANT_NOUNS = {
    "tree", "flower", "plant", "wall_plant", "corn_plant", "papaya",
    "sapodilla", "banana", "palm", "seed",
}


def _noun_classifier(lemma: str) -> str:
    """Return the numeral classifier suffix appropriate for ``lemma``."""
    if lemma in ANIMATE_NOUNS:
        return "túul"
    if lemma in PLANT_NOUNS:
        return "kúul"
    return "p'éel"


# ============================================================================
# GRAMMATICAL ENUMS
# ============================================================================

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


INDEPENDENT_PRONOUNS: Dict[Person, str] = {
    Person.first_sg:  "teen",
    Person.first_pl:  "to'on",
    Person.second_sg: "teech",
    Person.second_pl: "te'ex",
    Person.third_sg:  "leti'",
    Person.third_pl:  "leti'ob",
}


# Set A (ergative / possessive) pronominal prefixes.
SET_A: Dict[Person, str] = {
    Person.first_sg:  "in",
    Person.first_pl:  "k",
    Person.second_sg: "a",
    Person.second_pl: "a",
    Person.third_sg:  "u",
    Person.third_pl:  "u",
}

# Plural enclitic for Set A (attached to the verb / clause):
SET_A_PLURAL: Dict[Person, str] = {
    Person.first_pl:  "o'on",
    Person.second_pl: "e'ex",
    Person.third_pl:  "o'ob",
}


class TenseAspect(str, Enum):
    imperfective = "imperfective"   # ku / kin / ka ... habitual, generic
    progressive  = "progressive"    # táan u / táan in ... ongoing
    perfective   = "perfective"     # tu / tin / ta ... completed


def aspect_person_particle(aspect: TenseAspect, person: Person) -> str:
    """Return the fused aspect + Set A particle (e.g. ``ku``, ``tin``)."""
    erg = SET_A[person]
    if aspect == TenseAspect.imperfective:
        return {"in": "kin", "a": "ka", "u": "ku", "k": "k"}[erg]
    if aspect == TenseAspect.perfective:
        return {"in": "tin", "a": "ta", "u": "tu", "k": "t"}[erg]
    if aspect == TenseAspect.progressive:
        return f"táan {erg}"
    raise ValueError(aspect)


class Adjective(str, Enum):
    """Colour and dimension adjectives attested in the training captions.

    A few extra English colour terms (blue, brown, gray, dark) are
    included defensively so that VLM-emitted modifiers don't crash the
    structured-output parser. Yucatec colour vocabulary is smaller than
    English (ya'ax covers blue-green; boox covers black/dark/brown), so
    these extras simply map onto the closest attested Yucatec form.
    """
    white   = "white"
    black   = "black"
    red     = "red"
    green   = "green"
    blue    = "blue"
    yellow  = "yellow"
    pink    = "pink"
    brown   = "brown"
    gray    = "gray"
    dark    = "dark"
    big     = "big"
    small   = "small"
    long    = "long"
    tall    = "tall"
    old     = "old"
    new     = "new"
    many    = "many"
    dry     = "dry"
    dead    = "dead"
    good    = "good"


ADJECTIVE_FORMS: Dict[Adjective, str] = {
    Adjective.white:  "sak",        # TRAIN: 018, 043, 045, 049
    Adjective.black:  "boox",       # TRAIN: 004, 031, 048
    Adjective.red:    "chak",       # TRAIN: 014, 035
    Adjective.green:  "ya'ax",      # TRAIN: 011, 017, 033
    Adjective.blue:   "ya'ax",      # Yucatec ya'ax conflates blue & green (LEH, BRICK)
    Adjective.yellow: "k'an",       # TRAIN: 014, 015
    Adjective.pink:   "sakchak",    # TRAIN: 010, 013, 045, 051
    Adjective.brown:  "boox",       # boox = dark/brown (BRICK) — approximation
    Adjective.gray:   "boox",       # boox (dark) — approximation
    Adjective.dark:   "boox",       # boox = dark/black (BRICK)
    Adjective.big:    "nojoch",     # TRAIN: 022
    Adjective.small:  "chan",       # TRAIN: 027, 034, 039, 042
    Adjective.long:   "chowak",     # TRAIN: 029, 039
    Adjective.tall:   "ka'anal",    # TRAIN: 015, 033, 034
    Adjective.old:    "úuchben",    # TRAIN: 040
    Adjective.new:    "túumben",    # BRICK
    Adjective.many:   "ya'ab",      # TRAIN: 016, 020, 046, 047
    Adjective.dry:    "tikin",      # TRAIN: 012
    Adjective.dead:   "kíimen",     # TRAIN: 005
    Adjective.good:   "ma'alob",    # BRICK
}


class Preposition(str, Enum):
    """Spatial relators built on possessed body-part nouns."""
    in_         = "in"            # ichil
    on          = "on"            # yóok'ol
    under       = "under"         # yáanal
    next_to     = "next_to"       # tu tséel
    in_front_of = "in_front_of"   # tu táan
    behind      = "behind"        # tu paach
    at          = "at"            # ti'
    with_       = "with"          # yéetel


PREPOSITION_FORMS: Dict[Preposition, str] = {
    Preposition.in_:         "ichil",       # TRAIN: 002, 007, 038, 041
    Preposition.on:          "yóok'ol",     # TRAIN: 026, 029, 031, 042, 044, 051
    Preposition.under:       "yáanal",      # TRAIN: 032, 049
    Preposition.next_to:     "tu tséel",    # TRAIN: 019, 041, 042, 045, 046
    Preposition.in_front_of: "tu táan",     # TRAIN: 003, 014, 021
    Preposition.behind:      "tu paach",    # TRAIN: 012, 017, 019, 039, 042, 047
    Preposition.at:          "ti'",         # TRAIN: passim
    Preposition.with_:       "yéetel",      # TRAIN: passim
}


class Connective(str, Enum):
    """Attested clause-linking particles for :class:`CoordinatedSentence`.

    All six are drawn from the training captions and function to join
    two independent clauses describing different parts of a scene.
    """
    and_         = "and"            # yéetel  (TRAIN: 012, 013, 016, 017...)
    but          = "but"            # ba'ale' (TRAIN: 026)
    also         = "also"           # beyxan  (TRAIN: 010, 011)
    behind_it    = "behind_it"      # paachile' — "in the back..."
    in_front_it  = "in_front_it"    # táanile' — "in the front..."
    next_to_it   = "next_to_it"     # tu tséele' — "at its side..."


CONNECTIVE_FORMS: Dict[Connective, str] = {
    Connective.and_:        "yéetel",
    Connective.but:         "ba'ale'",
    Connective.also:        "beyxan",
    Connective.behind_it:   "paachile'",
    Connective.in_front_it: "táanile'",
    Connective.next_to_it:  "tu tséele'",
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

_KNOWN_NOUNS_HINT = ", ".join(e.english for e in NOUNS)
_KNOWN_TVERBS_HINT = ", ".join(e.english for e in TRANSITIVE_VERBS)
_KNOWN_IVERBS_HINT = ", ".join(e.english for e in INTRANSITIVE_VERBS)


class Noun(BaseModel):
    """A Yucatec noun phrase.

    Renders as ``[Numeral-Classifier] [adjective] head[-o'ob]``.
    When ``definite`` is True, the noun is bracketed by the distal
    clitics ``le ... o'`` instead ("le che'o'" = "the tree").
    """
    head: NounLemma = Field(
        ...,
        description=(
            "A noun lemma. Pick the closest match from the enum; use a "
            "hypernym if the literal noun isn't listed (e.g. 'chihuahua' → "
            "'dog'). When you set 'proper_noun', still pick the closest "
            "hypernym here as a type hint."
        ),
    )
    proper_noun: Optional[str] = Field(
        default=None,
        description=(
            "Optional verbatim string for proper nouns (named entities) "
            "that lack an in-vocab lemma — e.g. 'Chichén Itzá', 'Mérida', "
            "'Maria', 'Maaya t'aan'. When set, this string is rendered "
            "verbatim INSTEAD OF the 'head' lemma. **Use only for actual "
            "named entities. Do NOT use as a placeholder for unknown "
            "common nouns — pick a hypernym from the lemma list instead.**"
        ),
    )
    number: Number = Number.singular
    definite: bool = Field(
        default=False,
        description="If True, mark the NP as definite with 'le ... o'' instead of a numeral classifier.",
    )
    modifier: Optional[Adjective] = Field(
        default=None,
        description="Optional adjective modifier. See the Adjective enum for known values.",
    )


class Verb(BaseModel):
    lemma: str
    tense_aspect: TenseAspect = TenseAspect.imperfective


class TransitiveVerb(Verb):
    lemma: TransitiveVerbLemma = Field(
        ...,
        description=(
            "A transitive verb lemma. Pick the closest match from the enum."
        ),
    )


class IntransitiveVerb(Verb):
    lemma: IntransitiveVerbLemma = Field(
        ...,
        description=(
            "An intransitive verb lemma (includes positional/stative -kbal "
            "forms). Pick the closest match from the enum."
        ),
    )


class PrepositionalPhrase(BaseModel):
    """A spatial / comitative phrase: ``preposition + noun``."""
    preposition: Preposition
    noun: "Noun"


# ============================================================================
# NOUN-PHRASE RENDERING
# ============================================================================

_VOWELS = set("aeiouáéíóúAEIOUÁÉÍÓÚ")

# Positional / stative roots (those already ending in -kbal) that
# should NOT take an aspect particle — they are already predicative.
_STATIVE_STEMS = {"kulukbal", "wa'akbal", "chilikbal", "ch'uyukbal",
                  "p'ukukbal", "much'ukbal"}


def _pluralize(stem: str) -> str:
    """Add the nominal plural suffix ``-o'ob`` to a noun stem."""
    return stem + "o'ob"


def _render_noun(n: Noun) -> str:
    """Render a :class:`Noun` as a complete target-language NP."""
    if n.proper_noun:
        # Proper nouns render verbatim, no classifier or pluralization.
        return n.proper_noun.strip()
    head = get_noun_target(n.head)
    if head.startswith("["):
        return head

    if n.definite:
        if n.number == Number.plural:
            if n.modifier is not None:
                adj = ADJECTIVE_FORMS[n.modifier]
                return f"le {adj} {head}o'obo'"
            return f"le {head}o'obo'"
        if n.modifier is not None:
            adj = ADJECTIVE_FORMS[n.modifier]
            return f"le {adj} {head}o'"
        return f"le {head}o'"

    # Indefinite: numeral + classifier
    classifier = _noun_classifier(n.head)
    numeral_cls = {
        "túul":  "Juntúul",
        "kúul":  "Junkúul",
        "p'éel": "Junp'éel",
    }[classifier]

    if n.number == Number.plural:
        head_form = _pluralize(head)
        if n.modifier is not None:
            adj = ADJECTIVE_FORMS[n.modifier]
            return f"ya'ab {adj} {head_form}"
        return f"ya'ab {head_form}"

    if n.modifier is not None:
        adj = ADJECTIVE_FORMS[n.modifier]
        return f"{numeral_cls} {adj} {head}"
    return f"{numeral_cls} {head}"


def _render_possessed_noun(n: Noun, possessor: Person = Person.third_sg) -> str:
    """Render a possessed NP: Set A prefix + head [+ adjective after head].

    Attested pattern from TRAIN: ``u loolo'ob`` ("its flowers"),
    ``u lool k'an`` ("its yellow flower" — adjective *follows* the
    possessed head), ``u yich`` (3sg ``u`` + vowel-initial ``ich``
    fuses to ``yich``).
    """
    head = get_noun_target(n.head)
    if head.startswith("["):
        return head

    if n.number == Number.plural:
        head = _pluralize(head)

    erg = SET_A[possessor]

    # 3sg ``u`` fuses to ``y-`` before a vowel-initial stem.
    # 1sg ``in`` becomes ``inw-`` and 2sg ``a`` becomes ``aw-`` before vowels.
    first = head[0] if head else ""
    if first in _VOWELS:
        if erg == "u":
            prefixed = f"y{head}"
        elif erg == "in":
            prefixed = f"inw{head}"
        elif erg == "a":
            prefixed = f"aw{head}"
        else:
            prefixed = f"{erg} {head}"
    else:
        prefixed = f"{erg} {head}"

    if n.modifier is not None:
        adj = ADJECTIVE_FORMS[n.modifier]
        # Adjective follows the head in possessed NPs (as attested).
        return f"{prefixed} {adj}"
    return prefixed


def _render_subject_or_pronoun(x: Union[Noun, Person]) -> str:
    if isinstance(x, Person):
        return INDEPENDENT_PRONOUNS[x]
    return _render_noun(x)


def _subject_person(x: Union[Noun, Person]) -> Person:
    if isinstance(x, Person):
        return x
    if x.number == Number.plural:
        return Person.third_pl
    return Person.third_sg


def _render_intransitive_verb(verb: IntransitiveVerb, subj_person: Person) -> str:
    stem = get_intransitive_verb_target(verb.lemma)
    if stem in _STATIVE_STEMS:
        return stem
    if stem.startswith("["):
        return stem
    particle = aspect_person_particle(verb.tense_aspect, subj_person)
    plural_clitic = SET_A_PLURAL.get(subj_person, "")
    if plural_clitic:
        return f"{particle} {stem} {plural_clitic}"
    return f"{particle} {stem}"


def _render_transitive_verb(verb: TransitiveVerb, subj_person: Person) -> str:
    stem = get_transitive_verb_target(verb.lemma)
    if stem.startswith("["):
        return stem
    particle = aspect_person_particle(verb.tense_aspect, subj_person)
    return f"{particle} {stem}"


def _render_pp(pp: PrepositionalPhrase) -> str:
    return f"{PREPOSITION_FORMS[pp.preposition]} {_render_noun(pp.noun)}"


# ============================================================================
# SENTENCE TYPES
# ============================================================================

class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """Intransitive sentence: Subject + Verb.

    Renders in SV order: *Juntúul miis ku weenel* "A cat is sleeping".
    Stative positional predicates (``kulukbal``, ``wa'akbal``,
    ``chilikbal`` ...) are predicative on their own — no aspect marker.
    """
    subject: Union[Noun, Person]
    verb: IntransitiveVerb

    def __str__(self) -> str:
        subj = _render_subject_or_pronoun(self.subject)
        verb = _render_intransitive_verb(self.verb, _subject_person(self.subject))
        return f"{subj} {verb}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP)),
                    number=choice(list(Number)),
                )
            yield cls(
                subject=subject,
                verb=IntransitiveVerb(
                    lemma=choice(list(I_VERB_LOOKUP)),
                    tense_aspect=choice(list(TenseAspect)),
                ),
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        return [
            (
                "A cat is sleeping.",
                cls(
                    subject=Noun(head="cat"),
                    verb=IntransitiveVerb(
                        lemma="sleep", tense_aspect=TenseAspect.imperfective
                    ),
                ),
            ),
            (
                "A man walks.",
                cls(
                    subject=Noun(head="man"),
                    verb=IntransitiveVerb(
                        lemma="walk", tense_aspect=TenseAspect.imperfective
                    ),
                ),
            ),
            (
                "The cat is lying down.",
                cls(
                    subject=Noun(head="cat", definite=True),
                    verb=IntransitiveVerb(
                        lemma="lie", tense_aspect=TenseAspect.imperfective
                    ),
                ),
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """Transitive sentence: Subject + Verb + Object (SVO).

    Renders *N ku V-ik N*: the aspect + Set A particle selects the
    incompletive transitive stem stored in ``vocab.py`` (already
    carrying the ``-ik`` / ``-tik`` suffix).
    """
    subject: Union[Noun, Person]
    verb: TransitiveVerb
    object: Union[Noun, Person]

    def __str__(self) -> str:
        subj = _render_subject_or_pronoun(self.subject)
        verb = _render_transitive_verb(self.verb, _subject_person(self.subject))
        obj = _render_subject_or_pronoun(self.object)
        return f"{subj} {verb} {obj}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbObjectSentence", None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(head=choice(list(NOUN_LOOKUP)))
            if randint(0, 1) == 0:
                obj = choice(list(Person))
            else:
                obj = Noun(head=choice(list(NOUN_LOOKUP)))
            yield cls(
                subject=subject,
                verb=TransitiveVerb(
                    lemma=choice(list(T_VERB_LOOKUP)),
                    tense_aspect=choice(list(TenseAspect)),
                ),
                object=obj,
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "A person cooks a chicken.",
                cls(
                    subject=Noun(head="person"),
                    verb=TransitiveVerb(
                        lemma="cook", tense_aspect=TenseAspect.imperfective
                    ),
                    object=Noun(head="chicken"),
                ),
            ),
            (
                "A man ties the firewood.",
                cls(
                    subject=Noun(head="man"),
                    verb=TransitiveVerb(
                        lemma="tie", tense_aspect=TenseAspect.imperfective
                    ),
                    object=Noun(head="firewood"),
                ),
            ),
            (
                "A person shells white beans.",
                cls(
                    subject=Noun(head="person"),
                    verb=TransitiveVerb(
                        lemma="shell", tense_aspect=TenseAspect.imperfective
                    ),
                    object=Noun(head="white_bean", number=Number.plural),
                ),
            ),
        ]


class ExistentialLocativeSentence(Sentence["ExistentialLocativeSentence"]):
    """Existential / locative sentence: *N yaan [Prep N]* (or *mina'an*).

    This is the default predication for image-caption descriptions in
    Yucatec Maya: ``yaan`` ("there is / exists") is followed optionally
    by a locative prepositional phrase. Negation uses ``mina'an``
    (TRAIN: ``mina'an muunyal`` "there are no clouds").

    Examples:
      * "Junp'éel naj yaan tu tséel le che'o'" — A house is next to the tree.
      * "Junkúul sakchak lool yaan yóok'ol le bejo'" — A pink flower is on the road.
      * "Le che'o' mina'an u le'" — via the PossessiveSentence sibling.
    """
    subject: Noun
    location: Optional[PrepositionalPhrase] = Field(
        default=None,
        description="Optional spatial / locative PP placing the subject.",
    )
    negated: bool = Field(
        default=False,
        description=(
            "If True, render with 'mina'an' (does not exist) instead of "
            "'yaan'. Use for 'there is no X' / 'X is not here'."
        ),
    )

    def __str__(self) -> str:
        subj = _render_noun(self.subject)
        existential = "mina'an" if self.negated else "yaan"
        if self.location is None:
            return f"{subj} {existential}".strip()
        pp = _render_pp(self.location)
        return f"{subj} {existential} {pp}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["ExistentialLocativeSentence", None, None]:
        for _ in range(n):
            subject = Noun(
                head=choice(list(NOUN_LOOKUP)),
                number=choice(list(Number)),
            )
            if randint(0, 1) == 0:
                loc = None
            else:
                loc = PrepositionalPhrase(
                    preposition=choice(list(Preposition)),
                    noun=Noun(head=choice(list(NOUN_LOOKUP)), definite=True),
                )
            yield cls(subject=subject, location=loc, negated=bool(randint(0, 1)))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "ExistentialLocativeSentence"]]:
        return [
            (
                "A pink flower is on the road.",
                cls(
                    subject=Noun(head="flower", modifier=Adjective.pink),
                    location=PrepositionalPhrase(
                        preposition=Preposition.on,
                        noun=Noun(head="road", definite=True),
                    ),
                ),
            ),
            (
                "A house is next to the tree.",
                cls(
                    subject=Noun(head="house"),
                    location=PrepositionalPhrase(
                        preposition=Preposition.next_to,
                        noun=Noun(head="tree", definite=True),
                    ),
                ),
            ),
            (
                "There are no clouds.",
                cls(
                    subject=Noun(head="cloud", number=Number.plural),
                    negated=True,
                ),
            ),
        ]


class PossessiveSentence(Sentence["PossessiveSentence"]):
    """Possessive predication: *N yaan u M* "N has M".

    Yucatec expresses possession with an existential whose possessed NP
    is marked by a Set A ergative prefix. Attested in training:

    * ``Junkúul k'áax yaan u loolo'ob de sakchak`` — A plant has pink flowers.
    * ``Junkúul ché' yaan u lool k'an`` — A tree has a yellow flower.
    * ``Le che'o' ya'ab u yich`` — The tree has many fruits (``u + ich → yich``).
    * ``Le che'o' mina'an u le'`` — The tree has no leaves.

    Negation uses ``mina'an`` in place of ``yaan`` (*mina'an u le'*).
    """
    possessor: Union[Noun, Person]
    possessed: Noun = Field(
        ...,
        description=(
            "The possessed noun. 'u-' / 'y-' agreement is added "
            "automatically based on the possessor."
        ),
    )
    negated: bool = Field(
        default=False,
        description="If True, use 'mina'an' instead of 'yaan' for 'does not have'.",
    )

    def __str__(self) -> str:
        possessor = _render_subject_or_pronoun(self.possessor)
        person = _subject_person(self.possessor)
        head = _render_possessed_noun(self.possessed, person)
        existential = "mina'an" if self.negated else "yaan"
        return f"{possessor} {existential} {head}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["PossessiveSentence", None, None]:
        for _ in range(n):
            possessor = Noun(
                head=choice(list(NOUN_LOOKUP)),
                definite=bool(randint(0, 1)),
            )
            possessed = Noun(
                head=choice(list(NOUN_LOOKUP)),
                number=choice(list(Number)),
            )
            yield cls(
                possessor=possessor,
                possessed=possessed,
                negated=bool(randint(0, 1)),
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "PossessiveSentence"]]:
        return [
            (
                "A plant has pink flowers.",
                cls(
                    possessor=Noun(head="plant"),
                    possessed=Noun(
                        head="flower",
                        number=Number.plural,
                        modifier=Adjective.pink,
                    ),
                ),
            ),
            (
                "The tree has many fruits.",
                cls(
                    possessor=Noun(head="tree", definite=True),
                    possessed=Noun(
                        head="fruit",
                        number=Number.plural,
                        modifier=Adjective.many,
                    ),
                ),
            ),
            (
                "The tree has no leaves.",
                cls(
                    possessor=Noun(head="tree", definite=True),
                    possessed=Noun(head="leaf"),
                    negated=True,
                ),
            ),
        ]


class PredicativeSentence(Sentence["PredicativeSentence"]):
    """Zero-copula predicative sentence: *Le lela' [NP]* = "This is a NP".

    Yucatec has no overt copula for identificational predicates; the
    subject is juxtaposed with the predicate NP. This sentence type
    handles common caption openers like "Le lela' junp'éel sakchak lol"
    ("This is a pink flower").
    """
    subject: Union[Noun, Person] = Field(
        default_factory=lambda: Noun(head="person"),
        description="Subject NP or pronoun; ignored when 'demonstrative' is True.",
    )
    predicate: Noun
    demonstrative: bool = Field(
        default=True,
        description=(
            "If True, render the subject as the demonstrative 'Le lela'' "
            "('this one') rather than expanding the subject NP."
        ),
    )

    def __str__(self) -> str:
        if self.demonstrative:
            subj = "Le lela'"
        else:
            subj = _render_subject_or_pronoun(self.subject)
        pred = _render_noun(self.predicate)
        return f"{subj} {pred}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["PredicativeSentence", None, None]:
        for _ in range(n):
            pred = Noun(
                head=choice(list(NOUN_LOOKUP)),
                modifier=choice([None] + list(Adjective)),
            )
            yield cls(predicate=pred)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "PredicativeSentence"]]:
        return [
            (
                "This is a pink flower.",
                cls(predicate=Noun(head="flower", modifier=Adjective.pink)),
            ),
            (
                "This is a light.",
                cls(predicate=Noun(head="light")),
            ),
            (
                "This is a black container.",
                cls(predicate=Noun(head="container", modifier=Adjective.black)),
            ),
        ]


# ---------------------------------------------------------------------------
# Recursive (depth-1) coordination
# ---------------------------------------------------------------------------

# The "atomic" (non-coordinated) sentence types that may appear inside a
# CoordinatedSentence's left/right slots. Using an explicit Union here
# keeps the recursion bounded to depth 1 — CoordinatedSentence cannot
# contain another CoordinatedSentence, so the structured-output schema
# stays tractable.
AtomicSentence = Union[
    SubjectVerbSentence,
    SubjectVerbObjectSentence,
    ExistentialLocativeSentence,
    PossessiveSentence,
    PredicativeSentence,
]


class CoordinatedSentence(Sentence["CoordinatedSentence"]):
    """Binary clause coordination: *LEFT CONNECTIVE RIGHT*.

    Image captions in the training data frequently stack two atomic
    clauses with a connective particle — ``yéetel`` "and", ``ba'ale'``
    "but", ``beyxan`` "also", or the locative discourse adverbs
    ``paachile'`` "behind/after (it)", ``táanile'`` "in front of (it)",
    ``tu tséele'`` "at its side". This sentence type captures that
    pattern while keeping recursion to depth 1: the ``left`` and
    ``right`` slots accept any atomic sentence but *not* another
    CoordinatedSentence, so the JSON Schema stays compact.

    Examples (rendered):
      * "Juntúul miis ku weenel yéetel Juntúul peek' chilikbal"
        — A cat is sleeping and a dog is lying down.
      * "Le che'o' mina'an u le' ba'ale' ya'ab u yich"
        — The tree has no leaves but it has many fruits.
    """
    left: AtomicSentence = Field(
        ...,
        description="The first clause (any non-coordinated sentence type).",
    )
    connective: Connective = Field(
        ...,
        description=(
            "The linking particle. 'and' = yéetel, 'but' = ba'ale', "
            "'also' = beyxan, 'behind_it' = paachile', "
            "'in_front_it' = táanile', 'next_to_it' = tu tséele'."
        ),
    )
    right: AtomicSentence = Field(
        ...,
        description="The second clause (any non-coordinated sentence type).",
    )

    def __str__(self) -> str:
        left = str(self.left).strip().rstrip(".")
        right = str(self.right).strip().rstrip(".")
        particle = CONNECTIVE_FORMS[self.connective]
        return f"{left} {particle} {right}".strip()

    @classmethod
    def sample_iter(cls, n: int) -> Generator["CoordinatedSentence", None, None]:
        # Sample simple existential / possessive clauses for left & right.
        for _ in range(n):
            left = ExistentialLocativeSentence(
                subject=Noun(head=choice(list(NOUN_LOOKUP))),
            )
            right = ExistentialLocativeSentence(
                subject=Noun(head=choice(list(NOUN_LOOKUP))),
                location=PrepositionalPhrase(
                    preposition=choice(list(Preposition)),
                    noun=Noun(head=choice(list(NOUN_LOOKUP)), definite=True),
                ),
            )
            yield cls(
                left=left,
                connective=choice(list(Connective)),
                right=right,
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "CoordinatedSentence"]]:
        return [
            (
                "A cat is sleeping and a dog is lying down.",
                cls(
                    left=SubjectVerbSentence(
                        subject=Noun(head="cat"),
                        verb=IntransitiveVerb(lemma="sleep"),
                    ),
                    connective=Connective.and_,
                    right=SubjectVerbSentence(
                        subject=Noun(head="dog"),
                        verb=IntransitiveVerb(lemma="lie"),
                    ),
                ),
            ),
            (
                "The tree has no leaves but it has many fruits.",
                cls(
                    left=PossessiveSentence(
                        possessor=Noun(head="tree", definite=True),
                        possessed=Noun(head="leaf"),
                        negated=True,
                    ),
                    connective=Connective.but,
                    right=PossessiveSentence(
                        possessor=Noun(head="tree", definite=True),
                        possessed=Noun(
                            head="fruit",
                            number=Number.plural,
                            modifier=Adjective.many,
                        ),
                    ),
                ),
            ),
            (
                "A house is next to the tree; behind it there are stones.",
                cls(
                    left=ExistentialLocativeSentence(
                        subject=Noun(head="house"),
                        location=PrepositionalPhrase(
                            preposition=Preposition.next_to,
                            noun=Noun(head="tree", definite=True),
                        ),
                    ),
                    connective=Connective.behind_it,
                    right=ExistentialLocativeSentence(
                        subject=Noun(head="stone", number=Number.plural),
                    ),
                ),
            ),
        ]


language = Language(
    code="yua",
    name="Yucatec Maya",
    sentence_types=(
        SubjectVerbSentence,
        SubjectVerbObjectSentence,
        ExistentialLocativeSentence,
        PossessiveSentence,
        PredicativeSentence,
        CoordinatedSentence,
    ),
)
