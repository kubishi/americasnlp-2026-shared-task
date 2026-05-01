from pydantic import BaseModel, Field
from typing import Dict, Generator, List, Literal, Optional, Tuple, Type, Union
from enum import Enum
from random import choice, randint

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_hch.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Lookup dictionaries for easy access
NOUN_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in INTRANSITIVE_VERBS}

# Closed lemma vocabularies — typed as Literal so the LLM's structured
# output cannot emit out-of-vocabulary terms.
NounLemma = Literal[*tuple(sorted(NOUN_LOOKUP))]  # type: ignore[valid-type]
TransitiveVerbLemma = Literal[*tuple(sorted(TRANSITIVE_VERB_LOOKUP))]  # type: ignore[valid-type]
IntransitiveVerbLemma = Literal[*tuple(sorted(INTRANSITIVE_VERB_LOOKUP))]  # type: ignore[valid-type]
VerbLemma = Literal[*tuple(sorted(TRANSITIVE_VERB_LOOKUP | INTRANSITIVE_VERB_LOOKUP))]  # type: ignore[valid-type]


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


# Plural suffix rules: Wixárika has semantically-conditioned plural suffixes
# -ri: female humans, domestic animals, flowers
# -tsi: male humans, small animate beings
# -xi: inanimate objects, elderly females
# -te: body parts, ritual/everyday objects
# -ma: kinship terms
# -tari: large collectives of humans
PLURAL_DEFAULTS = {
    "woman": "ukari",
    "man": "tevítsi",
    "child": "'uxitsi",
    "boy": "tevitsi",
    "girl": "'uxitsi",
    "baby": "nunutsixi",
    "person": "wixáritari",
    "people": "te+teri",
    "young_person": "temaik+tsi",
    "shaman": "mara'akate",
    "elder": "kawiterutsixi",
    "dog": "chíkitsi",
    "horse": "purituxi",
    "bull": "wakaitsixi",
    "cow": "tewáxixi",
    "sheep": "muxatsixi",
    "goat": "muxaxi",
    "bird": "wikitsi",
    "fish": "tsapari",
    "deer": "máxari",
    "rabbit": "tatsiutsi",
    "snake": "kutsixi",
    "chicken": "tsikerutsixi",
    "turkey": "turukitsi",
    "animal": "t+rit+",
    "butterfly": "kukurutsi",
    "tortilla": "paapári",
    "flower": "tutúri",
    "tree": "kiyete",
    "plant": "iwekate",
    "stone": "tetete",
    "mountain": "kieriete",
    "clothing": "kemarite",
    "clothes": "kemarite",
    "bead": "chakirate",
    "arrow": "muwierite",
    "book": "xapate",
    "basket": "kiriwate",
}


def get_plural_form(lemma: str) -> str:
    """Get the plural form of a noun."""
    if lemma in PLURAL_DEFAULTS:
        return PLURAL_DEFAULTS[lemma]
    target = get_noun_target(lemma)
    if target.startswith("["):
        return target
    # Default: -te (collective/generic plural for inanimates)
    return f"{target}te"


# ============================================================================
# GRAMMATICAL ENUMERATIONS
# ============================================================================

class Number(str, Enum):
    singular = "singular"
    plural = "plural"


class TenseAspect(str, Enum):
    present = "present"
    past = "past"
    progressive = "progressive"
    habitual = "habitual"

    def get_suffix(self) -> str:
        """Return the TAM suffix for verb forms."""
        if self == TenseAspect.present:
            return ""  # present is unmarked
        elif self == TenseAspect.past:
            return "k+"
        elif self == TenseAspect.progressive:
            return "t+"
        elif self == TenseAspect.habitual:
            return "ame"
        raise ValueError("Invalid tense/aspect")


class LocativeRelation(str, Enum):
    """Spatial relations expressed through postpositions on the location noun.
    Wixárika uses postpositions (suffixed to the noun), not prepositions.
    """
    at = "at"          # -tsie (general locative, 'at/on')
    in_ = "in"         # -kewa (inside, 'in')
    on = "on"          # -tsie (on the surface of)
    near = "near"      # -'aurie (beside, next to)
    to = "to"          # -pait+ (direction, 'toward')
    from_ = "from"     # -kaku (source, 'from')

    def get_postposition(self) -> str:
        return {
            LocativeRelation.at: "tsie",
            LocativeRelation.in_: "kewa",
            LocativeRelation.on: "tsie",
            LocativeRelation.near: "'aurie",
            LocativeRelation.to: "pait+",
            LocativeRelation.from_: "kaku",
        }[self]


class Connective(str, Enum):
    """Clause-linking connectives in Wixárika."""
    and_ = "and"            # metá (and)
    also = "also"           # ya xeik+a (and also)
    but = "but"             # metá (contrastive)
    because = "because"     # kename (because, that)
    when = "when"           # tsie (when)
    while_ = "while"        # yak+ (while)

    def get_particle(self) -> str:
        return {
            Connective.and_: "metá",
            Connective.also: "ya xeik+a",
            Connective.but: "metá",
            Connective.because: "kename",
            Connective.when: "tsie",
            Connective.while_: "yak+",
        }[self]


# ============================================================================
# PRONOUN SYSTEM
# ============================================================================

class Person(str, Enum):
    first_sg = "I"
    first_pl = "we"
    second_sg = "you"
    second_pl = "you (plural)"
    third_sg = "he/she/it"
    third_pl = "they"

# Independent subject pronouns
SUBJECT_PRONOUNS: Dict[Person, str] = {
    Person.first_sg: "ne",
    Person.first_pl: "tame",
    Person.second_sg: "eki",
    Person.second_pl: "xame",
    Person.third_sg: "",       # 3SG subject often zero-marked
    Person.third_pl: "",       # 3PL uses me- prefix on verb instead
}

# Person-marking prefixes on verbs (subject agreement)
SUBJECT_PREFIXES: Dict[Person, str] = {
    Person.first_sg: "ne",
    Person.first_pl: "te",
    Person.second_sg: "pe",
    Person.second_pl: "xe",
    Person.third_sg: "",       # zero-marked
    Person.third_pl: "me",
}

# Object pronoun prefixes on verbs
OBJECT_PREFIXES: Dict[Person, str] = {
    Person.first_sg: "ne",
    Person.first_pl: "te",
    Person.second_sg: "pe",
    Person.second_pl: "xe",
    Person.third_sg: "",       # zero 3sg obj when NP object present
    Person.third_pl: "wa",
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Verb(BaseModel):
    lemma: VerbLemma = Field(
        ...,
        description=(
            "A verb lemma (transitive or intransitive). Pick the closest "
            "match from the enum; use a hypernym if the literal action "
            "isn't listed."
        ),
    )
    tense_aspect: TenseAspect

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
            "An intransitive verb lemma. Pick the closest match from the enum."
        ),
    )

class Noun(BaseModel):
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
            "that lack an in-vocab lemma — e.g. 'Tatewari', 'Wirikuta', "
            "'Maria', a place name. When set, this string is rendered "
            "verbatim INSTEAD OF the 'head' lemma. **Use only for "
            "actual named entities. Do NOT use as a placeholder for "
            "unknown common nouns — pick a hypernym from the lemma "
            "list instead.**"
        ),
    )
    number: Number


# ============================================================================
# HELPER RENDERING FUNCTIONS
# ============================================================================

def render_noun(noun: Noun) -> str:
    """Render a Noun (with number inflection)."""
    if noun.proper_noun:
        return noun.proper_noun.strip()
    if noun.number == Number.plural:
        return get_plural_form(noun.head)
    return get_noun_target(noun.head)


def render_subject_independent(subject) -> str:
    """Render an independent subject word (noun phrase or free pronoun)."""
    if isinstance(subject, Person):
        return SUBJECT_PRONOUNS[subject]
    elif isinstance(subject, Noun):
        return render_noun(subject)
    return ""


def _subject_person_for_agreement(subject) -> Person:
    if isinstance(subject, Person):
        return subject
    elif isinstance(subject, Noun):
        return Person.third_pl if subject.number == Number.plural else Person.third_sg
    return Person.third_sg


def _object_person_for_agreement(obj) -> Person:
    if isinstance(obj, Person):
        return obj
    elif isinstance(obj, Noun):
        return Person.third_pl if obj.number == Number.plural else Person.third_sg
    return Person.third_sg


def render_intransitive_verb(verb: IntransitiveVerb, subject_person: Person) -> str:
    prefix = SUBJECT_PREFIXES[subject_person]
    stem = get_intransitive_verb_target(verb.lemma)
    suffix = verb.tense_aspect.get_suffix()
    if not prefix and not suffix:
        return stem
    return f"{prefix}{stem}{suffix}"


def render_transitive_verb(
    verb: TransitiveVerb,
    subject_person: Person,
    object_person: Person,
    object_is_np: bool,
) -> str:
    subj_prefix = SUBJECT_PREFIXES[subject_person]
    # When a full NP object is present, 3sg object agreement is zero;
    # 3pl object still marks with "wa-".
    if object_is_np and object_person == Person.third_sg:
        obj_prefix = ""
    else:
        obj_prefix = OBJECT_PREFIXES[object_person]
    stem = get_transitive_verb_target(verb.lemma)
    suffix = verb.tense_aspect.get_suffix()
    return f"{subj_prefix}{obj_prefix}{stem}{suffix}"


# ============================================================================
# SENTENCE TYPES
# ============================================================================

class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """Intransitive sentence: Subject + Verb.

    Word order: Subject Verb (SV). The verb carries a subject-agreement
    prefix (ne-/te-/pe-/xe-/Ø-/me-) and a tense/aspect suffix.
    """
    subject: Union[Noun, Person]
    verb: IntransitiveVerb

    def __str__(self) -> str:
        subj_str = render_subject_independent(self.subject)
        verb_str = render_intransitive_verb(
            self.verb, _subject_person_for_agreement(self.subject)
        )
        parts = [p for p in [subj_str, verb_str] if p]
        return " ".join(parts)

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbSentence', None, None]:
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
                "The woman rests.",
                SubjectVerbSentence(
                    subject=Noun(head="woman", number=Number.singular),
                    verb=IntransitiveVerb(
                        lemma="rest", tense_aspect=TenseAspect.progressive
                    ),
                ),
            ),
            (
                "I work.",
                SubjectVerbSentence(
                    subject=Person.first_sg,
                    verb=IntransitiveVerb(
                        lemma="work", tense_aspect=TenseAspect.present
                    ),
                ),
            ),
            (
                "They live.",
                SubjectVerbSentence(
                    subject=Person.third_pl,
                    verb=IntransitiveVerb(
                        lemma="live", tense_aspect=TenseAspect.habitual
                    ),
                ),
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """Transitive sentence: Subject + Object + Verb (SOV).

    The verb carries a subject-agreement prefix, an object-agreement
    prefix (for pronominal or 3pl objects), the stem, and a tense/aspect
    suffix.
    """
    subject: Union[Noun, Person]
    verb: TransitiveVerb
    object: Union[Noun, Person]

    def __str__(self) -> str:
        subj_str = render_subject_independent(self.subject)

        object_is_np = isinstance(self.object, Noun)
        if object_is_np:
            obj_str = render_noun(self.object)  # type: ignore[arg-type]
        else:
            obj_str = ""  # pronominal object is marked on the verb

        verb_str = render_transitive_verb(
            self.verb,
            _subject_person_for_agreement(self.subject),
            _object_person_for_agreement(self.object),
            object_is_np,
        )
        parts = [p for p in [subj_str, obj_str, verb_str] if p]
        return " ".join(parts)

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbObjectSentence', None, None]:
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
    def sample(cls, n: int) -> List['SubjectVerbObjectSentence']:
        return list(cls.sample_iter(n))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "The woman washes the clothing.",
                SubjectVerbObjectSentence(
                    subject=Noun(head="woman", number=Number.singular),
                    verb=TransitiveVerb(
                        lemma="wash", tense_aspect=TenseAspect.progressive
                    ),
                    object=Noun(head="clothing", number=Number.singular),
                ),
            ),
            (
                "They carry the corn.",
                SubjectVerbObjectSentence(
                    subject=Person.third_pl,
                    verb=TransitiveVerb(
                        lemma="carry", tense_aspect=TenseAspect.past
                    ),
                    object=Noun(head="corn", number=Number.singular),
                ),
            ),
            (
                "The women cook the fish.",
                SubjectVerbObjectSentence(
                    subject=Noun(head="woman", number=Number.plural),
                    verb=TransitiveVerb(
                        lemma="cook", tense_aspect=TenseAspect.progressive
                    ),
                    object=Noun(head="fish", number=Number.singular),
                ),
            ),
        ]


class CopularSentence(Sentence["CopularSentence"]):
    """Equative sentence: "X is a Y" — class-membership only.

    Rendered as: subject predicate p+h+k+
    The particle "p+h+k+" is the Wixárika copula ('it is').

    **Use this only when the predicate is a different category of thing
    from the subject** ("the man is a shaman", "the building is a
    school"). Do NOT use for property predication ("the bag is green",
    "the house is big") — Wixárika expresses properties through verbs
    or context, not via a copula. If the English says "X is COLOR" or
    "X is BIG/SMALL/OLD/etc.", omit that sentence entirely; the
    descriptor cannot be carried into Wixárika by this schema.
    """
    subject: Union[Noun, Person]
    predicate: Noun

    def __str__(self) -> str:
        subj_str = render_subject_independent(self.subject)
        pred_str = render_noun(self.predicate)
        # Skip degenerate "X is X" tautologies: when the LLM falls back to
        # CopularSentence for an English sentence the schema can't carry
        # (e.g. "the bag is green"), the predicate slot collapses onto the
        # subject. Returning "" here drops the noise from the rendered output.
        if subj_str and pred_str and subj_str == pred_str:
            return ""
        parts = [p for p in [subj_str, pred_str, "p+h+k+"] if p]
        return " ".join(parts)

    @classmethod
    def sample_iter(cls, n: int) -> Generator['CopularSentence', None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            predicate = Noun(
                head=choice(list(NOUN_LOOKUP.keys())),
                number=Number.singular,
            )
            yield cls(subject=subject, predicate=predicate)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "CopularSentence"]]:
        return [
            (
                "The animal is a deer.",
                CopularSentence(
                    subject=Noun(head="animal", number=Number.singular),
                    predicate=Noun(head="deer", number=Number.singular),
                ),
            ),
            (
                "The food is a tortilla.",
                CopularSentence(
                    subject=Noun(head="food", number=Number.singular),
                    predicate=Noun(head="tortilla", number=Number.singular),
                ),
            ),
            (
                "I am a shaman.",
                CopularSentence(
                    subject=Person.first_sg,
                    predicate=Noun(head="shaman", number=Number.singular),
                ),
            ),
        ]


class LocativeSentence(Sentence["LocativeSentence"]):
    """Locative / existential sentence: "X is at/in/on/near Y".

    Rendered as: subject location-POSTPOSITION p+ma
    Wixárika uses postpositions (suffixes) on the location noun
    (-tsie 'at/on', -kewa 'in', -'aurie 'beside', -pait+ 'toward',
    -kaku 'from'), with the positional verb "ma" ('to be at') carrying
    a subject-agreement prefix.

    Use this for scene captions of the form "X is at/in/on Y"
    (e.g. "the woman is in the house", "children are on the road").
    """
    subject: Union[Noun, Person]
    location: Noun
    relation: LocativeRelation

    def __str__(self) -> str:
        subj_str = render_subject_independent(self.subject)
        loc_noun = render_noun(self.location)
        postp = self.relation.get_postposition()
        if loc_noun.startswith("["):
            loc_str = f"{loc_noun} {postp}"
        else:
            loc_str = f"{loc_noun}{postp}"

        # Existential verb 'ma' with subject-agreement prefix.
        # 3sg uses the affirmative prefix p+- ; 3pl uses me-.
        subj_person = _subject_person_for_agreement(self.subject)
        if subj_person == Person.third_pl:
            verb_str = "memama"
        elif subj_person == Person.third_sg:
            verb_str = "p+ma"
        else:
            subj_prefix = SUBJECT_PREFIXES[subj_person]
            verb_str = f"{subj_prefix}ma"

        parts = [p for p in [subj_str, loc_str, verb_str] if p]
        return " ".join(parts)

    @classmethod
    def sample_iter(cls, n: int) -> Generator['LocativeSentence', None, None]:
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = choice(list(Person))
            else:
                subject = Noun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                )
            location = Noun(
                head=choice(list(NOUN_LOOKUP.keys())),
                number=Number.singular,
            )
            relation = choice(list(LocativeRelation))
            yield cls(subject=subject, location=location, relation=relation)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "LocativeSentence"]]:
        return [
            (
                "The woman is in the house.",
                LocativeSentence(
                    subject=Noun(head="woman", number=Number.singular),
                    location=Noun(head="house", number=Number.singular),
                    relation=LocativeRelation.in_,
                ),
            ),
            (
                "The children are on the road.",
                LocativeSentence(
                    subject=Noun(head="child", number=Number.plural),
                    location=Noun(head="road", number=Number.singular),
                    relation=LocativeRelation.on,
                ),
            ),
            (
                "The dog is near the tree.",
                LocativeSentence(
                    subject=Noun(head="dog", number=Number.singular),
                    location=Noun(head="tree", number=Number.singular),
                    relation=LocativeRelation.near,
                ),
            ),
        ]


# Flat (non-recursive) sentence type alias. Used as the children of
# CoordinatedSentence so that recursion is bounded to depth 1.
FlatSentence = Union[
    SubjectVerbSentence,
    SubjectVerbObjectSentence,
    CopularSentence,
    LocativeSentence,
]


class CoordinatedSentence(Sentence["CoordinatedSentence"]):
    """Two clauses joined by a connective: "X and Y", "X also Y", etc.

    Recursive: `left` and `right` are themselves Sentence instances
    (of one of the flat sentence types). Use this when a caption
    describes two facts about a scene that should be linked — e.g.
    "the woman cooks food and the man carries corn", or a scene with
    a subject in a location and a further action.

    Recursion is bounded to depth 1: the children must be flat
    sentence types (not another CoordinatedSentence).
    """
    left: FlatSentence
    right: FlatSentence
    connective: Connective

    def __str__(self) -> str:
        left_str = str(self.left).rstrip(".").strip()
        right_str = str(self.right).rstrip(".").strip()
        particle = self.connective.get_particle()
        # If either child rendered to empty (e.g. a CopularSentence whose
        # subject==predicate was dropped), fall back to the surviving
        # child rather than emit a dangling particle.
        if left_str and right_str:
            return f"{left_str} {particle} {right_str}"
        return left_str or right_str

    @classmethod
    def sample_iter(cls, n: int) -> Generator['CoordinatedSentence', None, None]:
        for _ in range(n):
            left = next(SubjectVerbSentence.sample_iter(1))
            right = next(SubjectVerbObjectSentence.sample_iter(1))
            yield cls(left=left, right=right, connective=choice(list(Connective)))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "CoordinatedSentence"]]:
        return [
            (
                "The woman cooks food and the man carries corn.",
                CoordinatedSentence(
                    left=SubjectVerbObjectSentence(
                        subject=Noun(head="woman", number=Number.singular),
                        verb=TransitiveVerb(
                            lemma="cook", tense_aspect=TenseAspect.progressive
                        ),
                        object=Noun(head="food", number=Number.singular),
                    ),
                    right=SubjectVerbObjectSentence(
                        subject=Noun(head="man", number=Number.singular),
                        verb=TransitiveVerb(
                            lemma="carry", tense_aspect=TenseAspect.progressive
                        ),
                        object=Noun(head="corn", number=Number.singular),
                    ),
                    connective=Connective.and_,
                ),
            ),
            (
                "The children play, and also the dog runs.",
                CoordinatedSentence(
                    left=SubjectVerbSentence(
                        subject=Noun(head="child", number=Number.plural),
                        verb=IntransitiveVerb(
                            lemma="play", tense_aspect=TenseAspect.progressive
                        ),
                    ),
                    right=SubjectVerbSentence(
                        subject=Noun(head="dog", number=Number.singular),
                        verb=IntransitiveVerb(
                            lemma="run", tense_aspect=TenseAspect.progressive
                        ),
                    ),
                    connective=Connective.also,
                ),
            ),
            (
                "The woman is in the house and she cooks food.",
                CoordinatedSentence(
                    left=LocativeSentence(
                        subject=Noun(head="woman", number=Number.singular),
                        location=Noun(head="house", number=Number.singular),
                        relation=LocativeRelation.in_,
                    ),
                    right=SubjectVerbObjectSentence(
                        subject=Person.third_sg,
                        verb=TransitiveVerb(
                            lemma="cook", tense_aspect=TenseAspect.progressive
                        ),
                        object=Noun(head="food", number=Number.singular),
                    ),
                    connective=Connective.and_,
                ),
            ),
        ]


language = Language(
    code="hch",
    name="Wixárika",
    sentence_types=(
        SubjectVerbSentence,
        SubjectVerbObjectSentence,
        # CopularSentence removed: hch has no Adjective concept, so the
        # LLM was using CopularSentence as a fallback for any "X is Y"
        # English sentence and producing tautological "X is X" or
        # nonsense "X is unrelated-noun" output. Without an Adjective
        # type, English property sentences (the bag is green, the
        # house is yellow) are dropped at parse time, which is the
        # honest behavior given the grammar's coverage.
        LocativeSentence,
        CoordinatedSentence,
    ),
)
