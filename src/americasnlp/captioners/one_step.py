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
    "You are an image-captioning system for {name} ({iso}). Describe "
    "the image by emitting a SentenceList that conforms to the schema. "
    "Pick the closest in-vocab lemma for each concept; use a hypernym "
    "from the listed enum when the literal word isn't listed (e.g. "
    "chihuahua → dog, mansion → house)."
)


@dataclass
class OneStepCaptioner:
    """VLM directly emits SentenceList JSON; no English intermediate.

    `back_translator_model` (optional): if set, after the structured
    `SentenceList` is parsed, run yaduha's `SentenceToEnglishTool` on
    each `Sentence` using this model to produce a back-translation
    from the structured form. The back-translation is honest (LLM
    reads the structure, not the rendered target), parallel to what
    PipelineTranslator does for the two-step path.
    """

    lang: LanguageConfig
    vlm_model: str = "gpt-4o-mini"
    back_translator_model: str | None = None
    name: str = "one-step"

    _client: OpenAI = field(init=False, repr=False)
    _sentence_list_type: type[BaseModel] = field(init=False, repr=False)
    _bt_tool: object = field(init=False, repr=False, default=None)

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
        # Lazy-build the back-translator tool if requested.
        if self.back_translator_model:
            from yaduha.tool.sentence_to_english import SentenceToEnglishTool
            from americasnlp.captioners.pipeline import _make_translator_agent
            self._bt_tool = SentenceToEnglishTool(
                agent=_make_translator_agent(self.back_translator_model),
                SentenceType=language.sentence_types,
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
        target = " ".join(_clean_sentence(str(s)) for s in parsed.sentences).strip()

        back_translation: str | None = None
        if self._bt_tool is not None:
            try:
                bts = []
                for sent in parsed.sentences:
                    bt = self._bt_tool(sent)
                    bt_str = bt.content if hasattr(bt, "content") else str(bt)
                    bt_str = _clean_sentence(bt_str.strip())
                    if bt_str:
                        bts.append(bt_str)
                back_translation = " ".join(bts).strip() or None
            except Exception:  # noqa: BLE001
                back_translation = None

        # Save the raw structured output so presentation-layer changes
        # (punctuation, BT prompts, lemma-substitution display) don't
        # require re-running the VLM call.
        try:
            structured_json = [s.model_dump(mode="json") for s in parsed.sentences]
        except Exception:  # noqa: BLE001
            structured_json = None

        return CaptionResult(
            target=target,
            english_intermediate=None,
            back_translation=back_translation,
            structured_json=structured_json,
        )


def _clean_sentence(s: str) -> str:
    """Same shape as yaduha's PipelineTranslator clean_text: trim, ensure
    final punctuation, capitalize first letter. Keeps one-step output
    cosmetically aligned with the two-step path."""
    import re
    s = s.strip()
    if not s:
        return s
    if not re.search(r"[.!?]$", s):
        s += "."
    return s[0].upper() + s[1:]
