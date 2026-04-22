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


CAPTION_SYSTEM_PROMPT_BASE = (
    "Describe the photograph in literal English. Cover what you see: "
    "people, animals, objects, what's happening, where it is, and "
    "notable properties. No commentary, no quotation marks, no bullet "
    "points — just the description.\n\n"
)

CAPTION_VOCAB_HEADER = (
    "The downstream translator's English vocabulary is limited. **When the "
    "literal word for what you see falls outside the listed lemmas, prefer "
    "a hypernym from the list** (e.g. chihuahua → dog, mansion → house, "
    "shaman → person). When the literal word IS in the list, use it as-is. "
    "The goal is to minimize unrenderable words downstream, not to "
    "oversimplify accurate descriptions.\n\n"
    "Available {name} vocabulary:\n{vocab_block}"
)


def _autobuild_vocab_string(language: Any) -> str:
    """Default vocabulary description: bullet lists of lemma names from the
    package's NOUNS/TRANSITIVE_VERBS/INTRANSITIVE_VERBS/ADJECTIVES.

    Used when the language package doesn't define its own `get_vocab()`.
    """
    iso = language.code
    try:
        import importlib
        vocab = importlib.import_module(f"yaduha_{iso}.vocab")
    except Exception:  # noqa: BLE001
        return ""

    def _lemmas(attr: str) -> list[str]:
        entries = getattr(vocab, attr, None)
        if entries is None:
            return []
        out: list[str] = []
        for e in entries:
            if hasattr(e, "english"):
                out.append(e.english)
            elif hasattr(e, "value") and isinstance(e.value, str):
                out.append(e.value)
        return out

    sections: list[tuple[str, list[str]]] = [
        ("nouns", _lemmas("NOUNS")),
        ("transitive verbs", _lemmas("TRANSITIVE_VERBS")),
        ("intransitive verbs", _lemmas("INTRANSITIVE_VERBS")),
        ("adjectives", _lemmas("ADJECTIVES")),
    ]
    lines = [f"  - {label}: {', '.join(sorted(items))}"
             for label, items in sections if items]
    return "\n".join(lines)


def _resolve_vocab_string(language: Any) -> str:
    """Return the per-language vocabulary description for the caption prompt.

    Each `yaduha_{iso}` package may optionally define a top-level
    `get_vocab() -> str` callable that returns a free-form description of
    its vocabulary, morphology rules, register notes, common compounds,
    etc. The captioner uses that string verbatim when present. Otherwise
    we fall back to a default bullet list assembled from the package's
    NOUNS/TRANSITIVE_VERBS/INTRANSITIVE_VERBS/ADJECTIVES.
    """
    iso = language.code
    try:
        import importlib
        mod = importlib.import_module(f"yaduha_{iso}")
        getter = getattr(mod, "get_vocab", None)
        if callable(getter):
            result = getter()
            if isinstance(result, str) and result.strip():
                return result.strip()
    except Exception:  # noqa: BLE001
        pass
    return _autobuild_vocab_string(language)


def _build_caption_prompt(language: Any) -> str:
    vocab_block = _resolve_vocab_string(language)
    if not vocab_block:
        return CAPTION_SYSTEM_PROMPT_BASE
    return CAPTION_SYSTEM_PROMPT_BASE + CAPTION_VOCAB_HEADER.format(
        name=language.name,
        vocab_block=vocab_block,
    )


def _is_anthropic_model(model: str) -> bool:
    return model.startswith(("claude", "anthropic"))


def _vlm_caption_english_openai(model: str, image_path: Path,
                                system_prompt: str,
                                use_ollama: bool = False) -> str:
    from americasnlp._openai import image_data_url, model_kwargs
    if use_ollama:
        from americasnlp._ollama import (
            make_openai_client_for_ollama, normalize_ollama_model,
        )
        client = make_openai_client_for_ollama()
        eff_model = normalize_ollama_model(model)
        kwargs = {"max_tokens": 512, "temperature": 0.0}
    else:
        from openai import OpenAI
        client = OpenAI()
        eff_model = model
        kwargs = model_kwargs(model, max_out=512)
    resp = client.chat.completions.create(
        model=eff_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_data_url(image_path), "detail": "auto"}},
                {"type": "text", "text": "Describe this image as instructed."},
            ]},
        ],
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def _vlm_caption_english_anthropic(model: str, image_path: Path,
                                   system_prompt: str) -> str:
    import anthropic
    from americasnlp._anthropic import image_block
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [
            image_block(image_path),
            {"type": "text", "text": "Describe this image as instructed."},
        ]}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def _vlm_caption_english(model: str, image_path: Path,
                         system_prompt: str) -> str:
    from americasnlp._ollama import is_ollama_model
    if _is_anthropic_model(model):
        text = _vlm_caption_english_anthropic(model, image_path, system_prompt)
    elif is_ollama_model(model):
        text = _vlm_caption_english_openai(
            model, image_path, system_prompt, use_ollama=True)
    else:
        text = _vlm_caption_english_openai(model, image_path, system_prompt)
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“"):
        text = text[1:-1].strip()
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


def _make_translator_agent(model: str) -> Any:
    """Construct the right yaduha Agent for the structured-output translator.

    Routes to Anthropic for `claude-*`, Ollama (via the OpenAI-compatible
    /v1 endpoint) for ollama-style names like `qwen2.5:7b`, and the
    cloud OpenAI API for everything else.
    """
    if _is_anthropic_model(model):
        from americasnlp._anthropic import AnthropicAgent
        return AnthropicAgent(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    from americasnlp._ollama import (
        base_url, is_ollama_model, normalize_ollama_model,
    )
    from yaduha.agent.openai import OpenAIAgent
    if is_ollama_model(model):
        # yaduha's OpenAIAgent has a hard-coded `model: Literal[...]` so we
        # subclass it with `model: str` and point the OpenAI client at the
        # local ollama server via the OPENAI_BASE_URL env var.
        from pydantic import Field
        from typing import ClassVar
        from yaduha.agent.openai import OpenAIAgent as _Up
        class _OllamaAgent(_Up):
            model: str = Field(...)  # type: ignore[assignment]
            name: ClassVar[str] = "ollama_openai_agent"
        # Set the base URL for this process so yaduha's `OpenAI(api_key=...)`
        # picks it up via the env-var fallback.
        os.environ.setdefault("OPENAI_BASE_URL", base_url())
        return _OllamaAgent(model=normalize_ollama_model(model),
                            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
                            temperature=0.0)
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
        self._caption_prompt = _build_caption_prompt(self._language)
        self._translator = PipelineTranslator(
            agent=_make_translator_agent(self.translator_model),
            SentenceType=self._language.sentence_types,
        )

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        english = _vlm_caption_english(
            self.vlm_model, image_path, self._caption_prompt)
        translation = self._translator.translate(english)
        return CaptionResult(
            target=translation.target.strip(),
            english_intermediate=english,
        )
