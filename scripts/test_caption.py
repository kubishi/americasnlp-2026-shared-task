import os
import sys
from typing import Any, Dict, List

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — make yaduha and language packages importable when running this
# script directly without installing them.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("yaduha", "yaduha-ovp", "yaduha-hch"):
    _p = os.path.join(_REPO_ROOT, _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Pydantic schemas for the function's I/O
# ---------------------------------------------------------------------------

class SentenceTranslationRequest(BaseModel):
    """Input to translate_structured_sentence."""
    sentence: Dict[str, Any]
    language_code: str


class SentenceTranslationResult(BaseModel):
    """Output of translate_structured_sentence."""
    source: Dict[str, Any]
    language_code: str
    sentence_type: str
    target: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def translate_structured_sentence(
    sentence: Dict[str, Any],
    language_code: str,
) -> SentenceTranslationResult:
    """Translate a structured sentence to the target language.

    Mirrors the rendering step of ``PipelineTranslator.translate()``:
    parse the dict into one of the language's Sentence types, then call
    ``str(sentence)`` to obtain the target-language string.

    Args:
        sentence:       Dict matching one of the language's sentence-type
                        schemas (the JSON output of ``EnglishToSentencesTool``).
        language_code:  Language code registered via the ``yaduha.languages``
                        entry-point group (e.g. ``"ovp"``).

    Returns:
        SentenceTranslationResult with the rendered target-language string.

    Raises:
        ValueError: If the sentence cannot be parsed by any sentence type for
                    the requested language.
    """
    from yaduha.loader import LanguageLoader  # type: ignore[import]

    language = LanguageLoader.load_language(language_code)

    errors: List[str] = []
    for SentenceType in language.sentence_types:
        try:
            parsed = SentenceType.model_validate(sentence)
            target = str(parsed)
            return SentenceTranslationResult(
                source=sentence,
                language_code=language_code,
                sentence_type=SentenceType.__name__,
                target=target,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{SentenceType.__name__}: {exc}")

    raise ValueError(
        f"Could not parse sentence into any type for language '{language_code}'.\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1 — intransitive (SubjectVerbSentence in OVP)
    sv_sentence: Dict[str, Any] = {
        "subject": {
            "head": "coyote",
            "proximity": "distal",
            "plurality": "singular",
        },
        "verb": {
            "lemma": "run",
            "tense_aspect": "present_simple",
        },
    }

    result1 = translate_structured_sentence(sv_sentence, language_code="ovp")
    print(f"[{result1.sentence_type}] {result1.target}")

    # Example 2 — transitive (SubjectVerbObjectSentence in OVP)
    svo_sentence: Dict[str, Any] = {
        "subject": "you",
        "verb": {
            "lemma": "read",
            "tense_aspect": "present_simple",
        },
        "object": {
            "head": "mountain",
            "proximity": "distal",
            "plurality": "plural",
        },
    }

    result2 = translate_structured_sentence(svo_sentence, language_code="ovp")
    print(f"[{result2.sentence_type}] {result2.target}")
