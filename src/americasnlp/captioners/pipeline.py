"""LLM-RBMT image captioning — the proposed system.

Pipeline:
    image
        --[VLM]------------------> English caption
        --[EnglishToSentencesTool]-> structured Pydantic Sentence(s)
        --[Sentence.__str__()]----> target-language caption

The VLM only emits English. The English-to-target step is constrained by the
yaduha language package's Pydantic grammar via the LLM's structured outputs,
so the resulting target string is grammatical *by construction*.

OOV lemmas render as `[english_lemma]` placeholders (the existing
yaduha-{hch} convention) — visible signal of vocabulary gaps, not a bug.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic

from yaduha.loader import LanguageLoader
from yaduha.translator.pipeline import PipelineTranslator

from americasnlp._anthropic import AnthropicAgent, image_block
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


def _vlm_caption_english(client: anthropic.Anthropic, model: str,
                         image_path: Path) -> str:
    """Single VLM call: image -> 3–6 short English sentences (one per line)."""
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=[{"type": "text",
                 "text": CAPTION_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": [
                image_block(image_path),
                {"type": "text",
                 "text": "Describe this image as instructed."},
            ],
        }],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“"):
        text = text[1:-1].strip()
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


@dataclass
class PipelineCaptioner:
    """The proposed system. Anthropic VLM + yaduha translator per language."""

    lang: LanguageConfig
    vlm_model: str = "claude-sonnet-4-5"
    translator_model: str = "claude-sonnet-4-5"
    name: str = "pipeline"

    def __post_init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; create a .env or export it")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._language = LanguageLoader.load_language(self.lang.iso)
        self._translator = PipelineTranslator(
            agent=AnthropicAgent(model=self.translator_model, api_key=api_key),
            SentenceType=self._language.sentence_types,
        )

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        english = _vlm_caption_english(self._client, self.vlm_model, image_path)
        translation = self._translator.translate(english)
        return CaptionResult(
            target=translation.target.strip(),
            english_intermediate=english,
        )
