"""Orizaba Nahuatl (nlv) language package for the Yaduha framework.

Orizaba Nahuatl is a Uto-Aztecan (Nahuan) variety spoken in the Sierra de
Zongolica region of Veracruz, Mexico. It is polysynthetic, predominantly
verb-initial (VSO/VOS), with heavy noun incorporation, obligatory
subject-marking on the verb (ni- / ti- / ø- / ti- / an- / ø-), and
object cross-referencing prefixes (nech- / mits- / ki- / tech- / amech- /
kin-). Tense/aspect is marked by a combination of prefixes (preterit o-)
and suffixes (-k / -tok / -s). Plural 3rd-person subject agreement is
indicated by a verb-final suffix -j / -ej / -kej / -tokej / -skej.

Image captions in the training data also rely heavily on non-verbal
predication — "Se X ika/ipan/inauak Y" — so we model that pattern
explicitly as an ExistentialSentence alongside the intransitive and
transitive clause types. Coordinated noun phrases ("altepetlakamej uan
topilmej") are modelled by letting any Noun carry an optional
``conjoined`` Noun joined with ``uan``.

References:
  - Andrews, J. R. (2003). *Introduction to Classical Nahuatl* (Revised).
  - Launey, M. (2011). *An Introduction to Classical Nahuatl*.
  - Tuggy, D. (1979). *Tetelcingo Nahuatl*. In Langacker (ed.),
    *Studies in Uto-Aztecan Grammar*. SIL.
  - americasnlp2026 dev/pilot gold captions for nlv.
"""
from __future__ import annotations

from enum import Enum
from random import choice, randint
from typing import Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from yaduha.language import Language, Sentence, VocabEntry

from yaduha_nlv.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# ----------------------------------------------------------------------------
# Lookups
# ----------------------------------------------------------------------------
NOUN_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in INTRANSITIVE_VERBS}


def get_noun_target(lemma: str) -> str:
    if lemma in NOUN_LOOKUP:
        return NOUN_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_transitive_verb_target(lemma: str) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_intransitive_verb_target(lemma: str) -> str:
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


def get_verb_target(lemma: str) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    if lemma in INTRANSITIVE_VERB_LOOKUP:
        return INTRANSITIVE_VERB_LOOKUP[lemma].target
    return f"[{lemma}]"


# ----------------------------------------------------------------------------
# Noun pluralisation
# ----------------------------------------------------------------------------
# Hand-curated plurals for the most frequent animate / irregular nouns, all
# attested either in the nlv dev captions or in the cited references.
PLURAL_DEFAULTS: Dict[str, str] = {
    "woman":        "siuamej",        # nlv_006, nlv_017
    "man":          "tlakamej",       # nlv_017
    "person":       "maseualmej",     # nlv_005
    "people":       "maseualmej",     # nlv_005
    "young_man":    "telpochmej",     # nlv_004
    "young_woman":  "ichpochmej",
    "boy":          "topilmej",       # nlv_031
    "child":        "konemej",
    "townspeople":  "altepetlakamej", # nlv_031
    "villagers":    "altepetlakamej", # nlv_031
    "dog":          "itskuintimej",
    "cat":          "mistomej",
    "horse":        "kauayomej",
    "cow":          "kuakuemej",
    "chicken":      "piyomej",
    "bird":         "totomej",
    "fish":         "michimej",
    "snake":        "kouamej",
    "spider":       "tokamej",
    "worm":         "okuilmej",       # xonokuilmej nlv_019
    "insect":       "yolkamej",
    "mantis":       "mantismej",
    "tree":         "kuaumej",
    "branch":       "kuaumamej",
    "plant":        "xiuimej",        # xiuitl pl.
    "flower":       "xochimej",
    "marigold":     "senpoalxochimej",
    "stone":        "temej",          # nlv_028
    "cloud":        "mixmej",
    "mountain":     "tepemej",        # tepetl pl.
    "hill":         "tepemej",
    "house":        "kalmej",         # nlv_042
    "building":     "kalmej",
    "road":         "ojmej",
    "path":         "ojmej",
    "star":         "sitlalimej",
    "friend":       "ikniuan",
    "chile":        "chilmej",
    "chair":        "ikpalmej",
    "bench":        "ikpalmej",
    "mask":         "xayakamej",
}


def get_plural_form(lemma: str) -> str:
    """Return the plural target form for a noun lemma.

    Falls back to an Orizaba-default rule: strip the absolutive suffix
    (-tl / -tli / -li / -in), then append -mej.
    """
    if lemma in PLURAL_DEFAULTS:
        return PLURAL_DEFAULTS[lemma]
    target = get_noun_target(lemma)
    if target.startswith("["):
        return target  # unknown lemma — leave placeholder intact
    stem = target
    for suf in ("tli", "tl", "li", "in"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return f"{stem}mej"


# ----------------------------------------------------------------------------
# Grammatical enumerations
# ----------------------------------------------------------------------------
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


# Independent (free) subject pronouns. Overt pronouns are optional in
# Nahuatl — the verb's subject prefix is the primary exponent — but they
# appear in focus / disambiguation contexts.
SUBJECT_PRONOUNS: Dict[Person, str] = {
    Person.first_sg:  "neh",
    Person.first_pl:  "tejwan",
    Person.second_sg: "teh",
    Person.second_pl: "amejwan",
    Person.third_sg:  "yeh",
    Person.third_pl:  "yejwan",
}

# Subject-agreement prefixes on the verb. 3rd person is zero-marked;
# plurality is instead signalled by the verb-final plural suffix.
SUBJECT_PREFIXES: Dict[Person, str] = {
    Person.first_sg:  "ni",
    Person.first_pl:  "ti",
    Person.second_sg: "ti",
    Person.second_pl: "an",
    Person.third_sg:  "",
    Person.third_pl:  "",
}

# Object-cross-reference prefixes. These sit between the subject prefix
# and the verb stem.
OBJECT_PREFIXES: Dict[Person, str] = {
    Person.first_sg:  "nech",
    Person.first_pl:  "tech",
    Person.second_sg: "mits",
    Person.second_pl: "amech",
    Person.third_sg:  "ki",
    Person.third_pl:  "kin",
}


class TenseAspect(str, Enum):
    present    = "present"      # bare stem (+ plural -j for 3pl)
    past       = "past"          # o-…-k (+ -ej for 3pl)
    stative    = "stative"       # …-tok (+ -ej for 3pl) — "is V-ing / is V-ed"
    future     = "future"        # …-s (+ -kej for 3pl)

    def get_prefix(self) -> str:
        return "o" if self == TenseAspect.past else ""

    def get_suffix(self, is_plural_subject: bool) -> str:
        if self == TenseAspect.present:
            return "j" if is_plural_subject else ""
        if self == TenseAspect.past:
            return "kej" if is_plural_subject else "k"
        if self == TenseAspect.stative:
            return "tokej" if is_plural_subject else "tok"
        if self == TenseAspect.future:
            return "skej" if is_plural_subject else "s"
        raise ValueError(f"Invalid tense/aspect: {self}")


class Preposition(str, Enum):
    with_      = "with"       # ika
    on         = "on"         # ipan
    in_        = "in"         # ijtik
    next_to    = "next_to"    # inauak
    under      = "under"      # itlanpa
    at         = "at"         # itech
    near       = "near"       # inakastlan


PREPOSITION_WORDS: Dict[Preposition, str] = {
    Preposition.with_:   "ika",
    Preposition.on:      "ipan",
    Preposition.in_:     "ijtik",
    Preposition.next_to: "inauak",
    Preposition.under:   "itlanpa",
    Preposition.at:      "itech",
    Preposition.near:    "inakastlan",
}


class Quantifier(str, Enum):
    one    = "one"     # se
    some   = "some"    # seki / sekimej
    many   = "many"    # miakej
    two    = "two"     # ome
    three  = "three"   # eyi


def _quantifier_word(q: Quantifier, animate_plural: bool) -> str:
    # "sekimej" and "miakej" are used before animate plurals; before
    # inanimate / collective nouns, "seki" and "miak" appear instead.
    if q == Quantifier.one:
        return "se"
    if q == Quantifier.some:
        return "sekimej" if animate_plural else "seki"
    if q == Quantifier.many:
        return "miakej" if animate_plural else "miak"
    if q == Quantifier.two:
        return "ome"
    if q == Quantifier.three:
        return "eyi"
    raise ValueError(f"Unknown quantifier: {q}")


def _quantifier_forces_plural(q: Quantifier) -> bool:
    return q != Quantifier.one


class Adjective(str, Enum):
    big        = "big"         # ueyi
    small      = "small"       # tepitsin
    beautiful  = "beautiful"   # kualtsin
    black      = "black"       # tliltik
    white      = "white"       # istak
    yellow     = "yellow"      # kostik
    red        = "red"         # chichiltik
    green      = "green"       # xoxoktik
    blue       = "blue"        # texotik  (SIL Orizaba colour term)
    gray       = "gray"        # nextik   ("ash-coloured")
    pink       = "pink"        # tlatlauik ("pinkish-red")
    brown      = "brown"       # kafentik  ("coffee-coloured"; productive -tik)
    dark       = "dark"        # yayauik
    new        = "new"         # yankuik
    old        = "old"         # soltik
    tall       = "tall"        # uejkapantik
    wooden     = "wooden"      # kuautik   ("wood-like"; productive -tik)
    dry        = "dry"         # uakik
    sweet      = "sweet"       # tsopelik
    sour       = "sour"        # xokok
    far        = "far"         # uejka
    cold       = "cold"        # seuetok


ADJECTIVE_WORDS: Dict[Adjective, str] = {
    Adjective.big:        "ueyi",
    Adjective.small:      "tepitsin",
    Adjective.beautiful:  "kualtsin",
    Adjective.black:      "tliltik",
    Adjective.white:      "istak",
    Adjective.yellow:     "kostik",
    Adjective.red:        "chichiltik",
    Adjective.green:      "xoxoktik",
    Adjective.blue:       "texotik",
    Adjective.gray:       "nextik",
    Adjective.pink:       "tlatlauik",
    Adjective.brown:      "kafentik",
    Adjective.dark:       "yayauik",
    Adjective.new:        "yankuik",
    Adjective.old:        "soltik",
    Adjective.tall:       "uejkapantik",
    Adjective.wooden:     "kuautik",
    Adjective.dry:        "uakik",
    Adjective.sweet:      "tsopelik",
    Adjective.sour:       "xokok",
    Adjective.far:        "uejka",
    Adjective.cold:       "seuetok",
}


# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------
class Verb(BaseModel):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A verb lemma (transitive or intransitive). Known verbs: "
                + ", ".join(entry.english for entry in TRANSITIVE_VERBS + INTRANSITIVE_VERBS)
                + ". If the exact verb is not in this list, pass the English "
                "lemma as a placeholder."
            )
        },
    )
    tense_aspect: TenseAspect


class TransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A transitive verb lemma. Known: "
                + ", ".join(entry.english for entry in TRANSITIVE_VERBS)
                + ". If unknown, pass the English lemma as a placeholder."
            )
        },
    )


class IntransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "An intransitive verb lemma. Known: "
                + ", ".join(entry.english for entry in INTRANSITIVE_VERBS)
                + ". If unknown, pass the English lemma as a placeholder."
            )
        },
    )


class Noun(BaseModel):
    head: str = Field(
        ...,
        json_schema_extra={
            "description": (
                "A noun lemma. Known: "
                + ", ".join(entry.english for entry in NOUNS)
                + ". If unknown, pass the English lemma as a placeholder."
            )
        },
    )
    number: Number = Number.singular
    adjective: Optional[Adjective] = None
    # For nominal coordination "X uan Y". Depth-1 only: the conjoined noun
    # should not itself carry another conjoined noun; if the translator
    # provides one anyway the renderer just chains the strings.
    conjoined: Optional["Noun"] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional second noun joined to this one with 'uan' (and). "
                "Use when the English phrase is of the form 'X and Y' "
                "functioning as a single NP (subject, object, or oblique). "
                "Leave null if not applicable."
            )
        },
    )


Noun.model_rebuild()


# ----------------------------------------------------------------------------
# Shared rendering helpers
# ----------------------------------------------------------------------------
def _clean(s: str) -> str:
    """Collapse internal whitespace and trim. Removes gaps introduced when
    a placeholder NP was silently dropped from a sentence."""
    return " ".join(s.split()).strip()


def _render_noun(n: Noun) -> str:
    head = get_plural_form(n.head) if n.number == Number.plural else get_noun_target(n.head)
    if n.adjective is not None:
        rendered = f"{ADJECTIVE_WORDS[n.adjective]} {head}"
    else:
        rendered = head
    if n.conjoined is not None:
        rendered = f"{rendered} uan {_render_noun(n.conjoined)}"
    return rendered


def _render_nominal(x: Union[Noun, Person]) -> str:
    if isinstance(x, Person):
        return SUBJECT_PRONOUNS[x]
    return _render_noun(x)


def _subject_person(x: Union[Noun, Person]) -> Person:
    if isinstance(x, Person):
        return x
    if x.conjoined is not None:
        # Coordinated NP triggers plural agreement.
        return Person.third_pl
    return Person.third_pl if x.number == Number.plural else Person.third_sg


def _object_person(x: Union[Noun, Person]) -> Person:
    if isinstance(x, Person):
        return x
    if x.conjoined is not None:
        return Person.third_pl
    return Person.third_pl if x.number == Number.plural else Person.third_sg


def _is_plural(p: Person) -> bool:
    return p in (Person.first_pl, Person.second_pl, Person.third_pl)


def _is_bare_placeholder(s: str) -> bool:
    return s.startswith("[") and s.endswith("]")


def _assemble_verb_form(prefix: str, agreement: str, stem: str, suffix: str) -> str:
    """Concatenate verb parts.

    If the stem is an unknown-lemma placeholder like ``[run]``, we fall
    back to rendering the placeholder on its own rather than gluing
    agreement morphology directly onto the brackets, which produces
    mojibake like ``ki[run]`` or a bare ``ki`` once downstream code
    strips brackets. Keep morphology attached to real Nahuatl stems only.
    """
    if _is_bare_placeholder(stem):
        return stem
    return "".join(p for p in (prefix, agreement, stem, suffix) if p)


def _render_intransitive_verb(v: IntransitiveVerb, subject: Union[Noun, Person]) -> str:
    person = _subject_person(subject)
    return _assemble_verb_form(
        v.tense_aspect.get_prefix(),
        SUBJECT_PREFIXES[person],
        get_intransitive_verb_target(v.lemma),
        v.tense_aspect.get_suffix(_is_plural(person)),
    )


def _render_transitive_verb(
    v: TransitiveVerb,
    subject: Union[Noun, Person],
    obj: Union[Noun, Person],
) -> str:
    subj_person = _subject_person(subject)
    obj_person = _object_person(obj)
    stem = get_transitive_verb_target(v.lemma)
    if _is_bare_placeholder(stem):
        return stem
    agreement = f"{SUBJECT_PREFIXES[subj_person]}{OBJECT_PREFIXES[obj_person]}"
    return _assemble_verb_form(
        v.tense_aspect.get_prefix(),
        agreement,
        stem,
        v.tense_aspect.get_suffix(_is_plural(subj_person)),
    )


# ----------------------------------------------------------------------------
# Sentence types
# ----------------------------------------------------------------------------
class ExistentialSentence(Sentence["ExistentialSentence"]):
    """Non-verbal / existential description: "A/Some/Many X [PREP Y]".

    This is the single most frequent caption shape in the nlv training
    data — e.g. "Se ouamila ika itlatsakuil.", "Miakej kalmej ipan tepetl.",
    "Se ixtlauak ika miakej kaltsitsintin.", "Ateskatl ika ateskilitl
    inauak kuajyo." The quantifier (se / seki / sekimej / miakej / ome /
    eyi) selects singular vs. plural marking on the head noun.
    """
    quantifier: Quantifier
    subject: Noun
    preposition: Optional[Preposition] = None
    modifier: Optional[Noun] = None

    def __str__(self) -> str:
        # Force the subject's number to agree with the quantifier.
        forced_plural = _quantifier_forces_plural(self.quantifier)
        subj = self.subject.model_copy(update={
            "number": Number.plural if forced_plural else Number.singular,
        })
        animate_plural = forced_plural  # default: pluraliser is animate
        q_word = _quantifier_word(self.quantifier, animate_plural)
        subj_str = _render_noun(subj)
        # If the whole subject head is a placeholder, fall back to the
        # quantifier only — otherwise downstream bracket-stripping leaves
        # a lonely "se" / "miakej" with no NP.
        subject_str = (
            q_word if _is_bare_placeholder(subj_str) else f"{q_word} {subj_str}"
        )

        if self.preposition is None or self.modifier is None:
            return _clean(subject_str)

        prep = PREPOSITION_WORDS[self.preposition]
        mod_str = _render_noun(self.modifier)
        # Drop the whole PP if the modifier is just an unresolved
        # placeholder — a bare "itech [podium]" with the bracket stripped
        # downstream leaves a dangling preposition.
        if _is_bare_placeholder(mod_str):
            return _clean(subject_str)
        return _clean(f"{subject_str} {prep} {mod_str}")

    @classmethod
    def sample_iter(cls, n: int) -> Generator["ExistentialSentence", None, None]:
        noun_keys = list(NOUN_LOOKUP.keys())
        for _ in range(n):
            subj = Noun(head=choice(noun_keys))
            if randint(0, 1) == 0:
                yield cls(quantifier=choice(list(Quantifier)), subject=subj)
            else:
                yield cls(
                    quantifier=choice(list(Quantifier)),
                    subject=subj,
                    preposition=choice(list(Preposition)),
                    modifier=Noun(head=choice(noun_keys)),
                )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "ExistentialSentence"]]:
        return [
            (
                "A cornfield with a fence.",
                cls(
                    quantifier=Quantifier.one,
                    subject=Noun(head="cornfield"),
                    preposition=Preposition.with_,
                    modifier=Noun(head="fence"),
                ),
            ),
            (
                "Many houses on a mountain.",
                cls(
                    quantifier=Quantifier.many,
                    subject=Noun(head="house", number=Number.plural),
                    preposition=Preposition.on,
                    modifier=Noun(head="mountain"),
                ),
            ),
            (
                "A beautiful marigold.",
                cls(
                    quantifier=Quantifier.one,
                    subject=Noun(head="marigold", adjective=Adjective.beautiful),
                ),
            ),
            (
                "A spider on a flower.",
                cls(
                    quantifier=Quantifier.one,
                    subject=Noun(head="spider"),
                    preposition=Preposition.on,
                    modifier=Noun(head="flower"),
                ),
            ),
            (
                "Some villagers and boys.",
                cls(
                    quantifier=Quantifier.some,
                    subject=Noun(
                        head="villagers",
                        number=Number.plural,
                        conjoined=Noun(head="boy", number=Number.plural),
                    ),
                ),
            ),
        ]


class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """Intransitive clause: Subject + Verb.

    Nahuatl is head-marking and the verb is the sole obligatory element;
    the overt subject NP is optional. We render Subject + Verb here
    because captions in the training data that contain an intransitive
    action consistently front the subject (e.g. "Se siuatsintli nemi ipan
    se teyo ojtli.").

    A coordinated subject like "a woman and a man walk" is expressed by
    setting ``conjoined`` on the subject Noun; the verb then takes 3pl
    agreement.
    """
    subject: Union[Noun, Person]
    verb: IntransitiveVerb

    def __str__(self) -> str:
        verb_str = _render_intransitive_verb(self.verb, self.subject)
        # If the verb lemma is unknown, the whole clause would render as
        # "<subject> [verb]" and downstream bracket-stripping leaves a
        # bare NP with a stray space. Falling back to a pure nominal
        # ("subject") line is more useful for evaluation than a mutilated
        # verb phrase.
        if _is_bare_placeholder(verb_str):
            if isinstance(self.subject, Noun):
                return _clean(_render_noun(self.subject))
            return ""
        subject_str = _render_nominal(self.subject)
        if _is_bare_placeholder(subject_str):
            subject_str = ""
        # Pronominal subjects are usually dropped; keep them only when
        # the user explicitly passed a Person (focus).
        parts = [subject_str, verb_str] if subject_str else [verb_str]
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbSentence", None, None]:
        noun_keys = list(NOUN_LOOKUP.keys())
        i_verb_keys = list(INTRANSITIVE_VERB_LOOKUP.keys())
        for _ in range(n):
            if randint(0, 1) == 0:
                subj: Union[Noun, Person] = choice(list(Person))
            else:
                subj = Noun(head=choice(noun_keys), number=choice(list(Number)))
            yield cls(
                subject=subj,
                verb=IntransitiveVerb(
                    lemma=choice(i_verb_keys),
                    tense_aspect=choice(list(TenseAspect)),
                ),
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        return [
            (
                "The woman walks on the road.",  # paraphrase of nlv_021
                cls(
                    subject=Noun(head="woman"),
                    verb=IntransitiveVerb(lemma="walk", tense_aspect=TenseAspect.present),
                ),
            ),
            (
                "I sleep.",
                cls(
                    subject=Person.first_sg,
                    verb=IntransitiveVerb(lemma="sleep", tense_aspect=TenseAspect.present),
                ),
            ),
            (
                "The people live.",
                cls(
                    subject=Noun(head="people", number=Number.plural),
                    verb=IntransitiveVerb(lemma="live", tense_aspect=TenseAspect.present),
                ),
            ),
            (
                "The flowers bloomed.",
                cls(
                    subject=Noun(head="flower", number=Number.plural),
                    verb=IntransitiveVerb(lemma="bloom", tense_aspect=TenseAspect.past),
                ),
            ),
            (
                "A woman and a man walk.",
                cls(
                    subject=Noun(
                        head="woman",
                        conjoined=Noun(head="man"),
                    ),
                    verb=IntransitiveVerb(lemma="walk", tense_aspect=TenseAspect.present),
                ),
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """Transitive clause: Subject + Verb (+ Object NP).

    The verb carries a subject-agreement prefix and an object-cross-
    reference prefix (usually ki- for 3sg). The overt object NP
    co-occurs with the ki-/kin- prefix, e.g. "Se ichpoka kitejteki
    nakatl." ("A young woman cuts the meat"). Coordinated subjects or
    objects are expressed by setting ``conjoined`` on the relevant Noun.
    """
    subject: Union[Noun, Person]
    verb: TransitiveVerb
    object: Union[Noun, Person]

    def __str__(self) -> str:
        verb_str = _render_transitive_verb(self.verb, self.subject, self.object)
        subject_str = _render_nominal(self.subject)
        # Pronominal objects are marked on the verb only.
        object_str = "" if isinstance(self.object, Person) else _render_noun(self.object)
        # If the verb stem is unknown, fall back to a nominal-only render
        # rather than a subject + bracket + object string that gets
        # stripped into "subj  obj" downstream.
        if _is_bare_placeholder(verb_str):
            pieces = []
            if not _is_bare_placeholder(subject_str):
                pieces.append(subject_str)
            if object_str and not _is_bare_placeholder(object_str):
                pieces.append(object_str)
            return _clean(" ".join(pieces))
        # Drop subject / object NPs that are nothing but a bare placeholder.
        if _is_bare_placeholder(subject_str):
            subject_str = ""
        if _is_bare_placeholder(object_str):
            object_str = ""
        parts = [subject_str, verb_str, object_str]
        return _clean(" ".join(p for p in parts if p))

    @classmethod
    def sample_iter(cls, n: int) -> Generator["SubjectVerbObjectSentence", None, None]:
        noun_keys = list(NOUN_LOOKUP.keys())
        t_verb_keys = list(TRANSITIVE_VERB_LOOKUP.keys())
        for _ in range(n):
            if randint(0, 1) == 0:
                subj: Union[Noun, Person] = choice(list(Person))
            else:
                subj = Noun(head=choice(noun_keys), number=choice(list(Number)))
            if randint(0, 1) == 0:
                obj: Union[Noun, Person] = choice(list(Person))
            else:
                obj = Noun(head=choice(noun_keys), number=choice(list(Number)))
            yield cls(
                subject=subj,
                verb=TransitiveVerb(
                    lemma=choice(t_verb_keys),
                    tense_aspect=choice(list(TenseAspect)),
                ),
                object=obj,
            )

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "The young woman cuts the meat.",  # paraphrase of nlv_018
                cls(
                    subject=Noun(head="young_woman"),
                    verb=TransitiveVerb(lemma="cut", tense_aspect=TenseAspect.present),
                    object=Noun(head="meat"),
                ),
            ),
            (
                "They planted the trees.",  # paraphrase of nlv_050
                cls(
                    subject=Noun(head="people", number=Number.plural),
                    verb=TransitiveVerb(lemma="plant", tense_aspect=TenseAspect.past),
                    object=Noun(head="tree", number=Number.plural),
                ),
            ),
            (
                "I see the dog.",
                cls(
                    subject=Person.first_sg,
                    verb=TransitiveVerb(lemma="see", tense_aspect=TenseAspect.present),
                    object=Noun(head="dog"),
                ),
            ),
            (
                "The man has a white shirt.",
                cls(
                    subject=Noun(head="man"),
                    verb=TransitiveVerb(lemma="have", tense_aspect=TenseAspect.present),
                    object=Noun(head="shirt", adjective=Adjective.white),
                ),
            ),
            (
                "The villagers and boys bury the one who died.",
                cls(
                    subject=Noun(
                        head="villagers",
                        number=Number.plural,
                        conjoined=Noun(head="boy", number=Number.plural),
                    ),
                    verb=TransitiveVerb(lemma="bury", tense_aspect=TenseAspect.future),
                    object=Noun(head="person"),
                ),
            ),
        ]


language = Language(
    code="nlv",
    name="Orizaba Nahuatl",
    sentence_types=(
        ExistentialSentence,
        SubjectVerbSentence,
        SubjectVerbObjectSentence,
    ),
)
