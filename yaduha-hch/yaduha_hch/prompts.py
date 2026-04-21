from typing import Iterable, List, Type, TYPE_CHECKING
from yaduha_hch.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS
from yaduha_hch import SubjectVerbSentence, SubjectVerbObjectSentence

if TYPE_CHECKING:
    from yaduha.language import Sentence

SYSTEM_PROMPT_PREFIX = (
    "You are a translator that translates English or Spanish sentences into Wixárika (Huichol). "
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible. "
    "It doesn't need to be perfect and you can leave untranslated words in brackets if necessary.\n"
    "Wixárika uses the character + to represent the high central vowel /ɨ/.\n"
    "Wixárika is an SOV (Subject-Object-Verb) language with polysynthetic verb morphology.\n"
)

TOOL_USE_INSTRUCTION = (
    "You may also have access to tools that can help you produce a better translation. "
    "Use these tools as needed. You can make one or many tool calls (in parallel and/or sequentially) "
    "until you decide to respond.\n"
)

VOCABULARY_PROMPT = (
    "You use the following vocabulary to translate user input sentences into Wixárika.\n"
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible.\n"
    "It doesn't need to be perfect and you can leave untranslated words in brackets if necessary.\n"
    "# Vocabulary\n"
    "## Nouns: \n" + "\n".join([f"{noun.target}: {noun.english}" for noun in NOUNS]) + "\n"
    "## Transitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in TRANSITIVE_VERBS]) + "\n"
    "## Intransitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in INTRANSITIVE_VERBS]) + "\n"
)

SENTENCE_STRUCTURE_PROMPT = (
    "# Sentence Structure\n"
    "## Wixárika follows SOV word order:\n"
    "Intransitive (SV): [subject] [subject_prefix]-[verb stem]-[tense suffix]\n"
    "Transitive (SOV): [subject] [object] [subject_prefix]-[object_prefix]-[verb stem]-[tense suffix]\n"
    "## Person Prefixes on Verbs:\n"
    "1SG: ne-, 1PL: te-, 2SG: pe-, 2PL: xe-, 3SG: (zero), 3PL: me-\n"
    "## Tense/Aspect Suffixes:\n"
    "Present: (unmarked), Past: -k+, Progressive: -t+, Habitual: -ame\n"
    "## Plural Suffixes on Nouns:\n"
    "-ri (female humans, domestic animals, flowers), -tsi (male humans, small animates), "
    "-xi (inanimates), -te (objects, body parts), -ma (kinship)\n"
    "## Postpositions:\n"
    "-tsie (on, over, at), -k+ (by, with, for), -pai (toward, in area of)\n"
)

def get_prompt(include_vocab: bool,
               has_tools: bool = False,
               include_examples: Iterable[Type["Sentence"]] | None = None) -> str:
    include_examples = include_examples or []
    system_prompt = SYSTEM_PROMPT_PREFIX
    if has_tools:
        system_prompt += TOOL_USE_INSTRUCTION
    if include_vocab:
        system_prompt += VOCABULARY_PROMPT
    system_prompt += SENTENCE_STRUCTURE_PROMPT
    for sentence_cls in include_examples:
        for source, example_sentence in sentence_cls.get_examples():
            system_prompt += (
                "\n# Example\n"
                f"English: {source}\n"
                f"Wixárika: {example_sentence}\n"
            )

    return system_prompt
