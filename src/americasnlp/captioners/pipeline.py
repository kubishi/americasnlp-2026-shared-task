"""LLM-RBMT image captioning — the proposed system.

Pipeline:
    image
        --[VLM]------------------> English caption
        --[EnglishToSentencesTool]-> structured Pydantic Sentence(s)
        --[Sentence.__str__()]----> target-language caption

Backend (OpenAI vs Anthropic) is dispatched on model name: anything that
starts with `claude` or `anthropic` uses Anthropic; everything else uses
OpenAI. OpenAI gives us native structured outputs; Anthropic is currently
stronger on vision and reasoning per our dev numbers, so per-language
model choice matters.

OOV lemmas render as `[english_lemma]` placeholders.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from yaduha.loader import LanguageLoader
from yaduha.translator.pipeline import PipelineTranslator

from americasnlp.captioners import CaptionResult
from americasnlp.languages import LanguageConfig


CAPTION_SYSTEM_PROMPT = (
    "You describe a photograph in 3–6 short, literal English sentences. "
    "Use only simple grammar: subject–verb, subject–verb–object, "
    "'X is Y' (predicate adjective or noun), 'X is at/in/on Y' (locative), "
    "or 'X has Y' (possessive). Each sentence should make exactly one "
    "claim. Cover: who or what is in the image, what they are doing, "
    "where they are, and notable properties (colour, size, material). "
    "Prefer common, concrete nouns and verbs. No commentary, no quotation "
    "marks, no bullet points — just the sentences, one per line."
)


def _is_anthropic_model(model: str) -> bool:
    return model.startswith(("claude", "anthropic"))


def _vlm_caption_english_openai(model: str, image_path: Path) -> str:
    from openai import OpenAI
    from americasnlp._openai import image_data_url, model_kwargs
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_data_url(image_path), "detail": "auto"}},
                {"type": "text", "text": "Describe this image as instructed."},
            ]},
        ],
        **model_kwargs(model, max_out=512),
    )
    return (resp.choices[0].message.content or "").strip()


def _vlm_caption_english_anthropic(model: str, image_path: Path) -> str:
    import anthropic
    from americasnlp._anthropic import image_block
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=[{"type": "text", "text": CAPTION_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            image_block(image_path),
            {"type": "text", "text": "Describe this image as instructed."},
        ]}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def _vlm_caption_english(model: str, image_path: Path) -> str:
    text = (_vlm_caption_english_anthropic(model, image_path)
            if _is_anthropic_model(model)
            else _vlm_caption_english_openai(model, image_path))
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“"):
        text = text[1:-1].strip()
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


def _make_translator_agent(model: str) -> Any:
    """Construct the right yaduha Agent for the structured-output translator."""
    if _is_anthropic_model(model):
        from americasnlp._anthropic import AnthropicAgent
        return AnthropicAgent(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    from yaduha.agent.openai import OpenAIAgent
    from americasnlp._openai import is_reasoning_model
    kwargs = {"model": model, "api_key": os.environ["OPENAI_API_KEY"]}
    if is_reasoning_model(model):
        kwargs["temperature"] = 1.0
    return OpenAIAgent(**kwargs)


@dataclass
class PipelineCaptioner:
    """The proposed system. VLM + yaduha translator per language.

    `vlm_model` and `translator_model` accept either an OpenAI model
    (`gpt-4o-mini`, `gpt-4o`, `gpt-5`) or an Anthropic model
    (`claude-sonnet-4-5`, `claude-opus-4-7`, ...).
    """

    lang: LanguageConfig
    vlm_model: str = "claude-sonnet-4-5"
    translator_model: Optional[str] = None  # defaults to vlm_model
    name: str = "pipeline"

    def __post_init__(self) -> None:
        self.translator_model = self.translator_model or self.vlm_model
        # Required key per backend
        if _is_anthropic_model(self.vlm_model) or _is_anthropic_model(self.translator_model):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
        if (not _is_anthropic_model(self.vlm_model)
                or not _is_anthropic_model(self.translator_model)):
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set")
        self._language = LanguageLoader.load_language(self.lang.iso)
        self._translator = PipelineTranslator(
            agent=_make_translator_agent(self.translator_model),
            SentenceType=self._language.sentence_types,
        )

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        english = _vlm_caption_english(self.vlm_model, image_path)
        translation = self._translator.translate(english)
        return CaptionResult(
            target=translation.target.strip(),
            english_intermediate=english,
        )
