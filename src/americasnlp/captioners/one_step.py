"""One-step VLM-direct-to-structured captioner (for comparison vs two-step).

Skips the English intermediate: the VLM sees the image and emits a
`SentenceList` JSON conforming to the target language's grammar schema
via OpenAI's structured outputs (`response_format=PydanticModel`). The
deterministic Python `__str__()` then renders the target string.

Trade-off vs the two-step pipeline:
- Pro: VLM has direct image context, no information loss in English.
- Pro: one fewer LLM call, lower cost / latency.
- Con: VLMs are weaker at structured-output JSON than text models.
- Con: schema (sentence types + lemma lists) goes on every image request.

Currently OpenAI-only (gpt-4o, gpt-4o-mini, gpt-5).
"""
from __future__ import annotations

import functools
import operator
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, create_model

from yaduha.loader import LanguageLoader

from americasnlp._openai import image_data_url, model_kwargs
from americasnlp.captioners import CaptionResult
from americasnlp.languages import LanguageConfig


SYSTEM_PROMPT = (
    "You are a structured image-captioning system for the {name} language "
    "(ISO 639-3: {iso}). Look at the image and emit a JSON object "
    "conforming to the schema.\n\n"
    "Each item in `sentences` should describe one fact about the image "
    "(subject + verb, copular, locative, possessive, etc.). Use 3–6 "
    "sentences total covering: who/what is in the image, what they are "
    "doing, where they are, and notable properties (colour, size, "
    "material).\n\n"
    "**CRITICAL — use in-vocab lemmas wherever possible.** Each "
    "schema field's description lists the lemmas the language actually "
    "knows. You should be aggressive about mapping what you see to those "
    "lemmas, even when the literal English word is more specific:\n"
    "  - 'chihuahua', 'puppy', 'hound' → use 'dog'\n"
    "  - 'mansion', 'cottage', 'cabin', 'hut' → use 'house'\n"
    "  - 'sit', 'crouch', 'kneel', 'recline' → use 'sit' (or whatever "
    "the closest in-vocab posture verb is)\n"
    "  - 'shaman', 'priest', 'elder' → use 'person' or 'old_man' if "
    "those are in-vocab\n"
    "  - 'bicycle', 'wagon', 'carretón' → use the nearest in-vocab "
    "object word\n"
    "Decompose unfamiliar concepts into combinations of in-vocab words "
    "where possible. **Only emit an English placeholder lemma when no "
    "reasonable in-vocab substitution exists.** A grammatical, slightly "
    "imprecise sentence is better than one full of `[english]` "
    "placeholders."
)


@dataclass
class OneStepCaptioner:
    """VLM directly emits SentenceList JSON; no English intermediate."""

    lang: LanguageConfig
    vlm_model: str = "gpt-4o-mini"
    name: str = "one-step"

    _client: OpenAI = field(init=False, repr=False)
    _sentence_list_type: type[BaseModel] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)
        language = LanguageLoader.load_language(self.lang.iso)
        # Build a SentenceList[union-of-the-language's-sentence-types].
        types = tuple(language.sentence_types)
        union = types[0] if len(types) == 1 else functools.reduce(operator.or_, types)
        self._sentence_list_type = create_model(
            "SentenceList",
            sentences=(list[union], ...),  # type: ignore[valid-type]
            __base__=BaseModel,
        )

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        prompt = SYSTEM_PROMPT.format(name=self.lang.name, iso=self.lang.iso)
        resp = self._client.chat.completions.parse(
            model=self.vlm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": image_data_url(image_path),
                                   "detail": "auto"}},
                    {"type": "text",
                     "text": "Describe this image by emitting the SentenceList JSON."},
                ]},
            ],
            response_format=self._sentence_list_type,
            **model_kwargs(self.vlm_model, max_out=1024),
        )
        parsed: Any = resp.choices[0].message.parsed
        if parsed is None:
            return CaptionResult(target="", english_intermediate=None)
        target = " ".join(str(s) for s in parsed.sentences).strip()
        return CaptionResult(target=target, english_intermediate=None)
