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

CAPTION_GRAMMAR_HEADER = (
    "After you describe the image, a downstream step will recast your "
    "English to fit the {name} grammar below. Producing English that's "
    "naturally compatible with these patterns helps that step, but "
    "don't twist the description into knots — clarity wins over "
    "alignment.\n\n"
    "**When the literal word for what you see falls outside the listed "
    "lemmas, prefer a hypernym from the list** (e.g. chihuahua → dog, "
    "mansion → house, shaman → person). When the literal word IS in "
    "the list, use it as-is.\n\n"
    "{name} grammar:\n{grammar_block}"
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


def _autobuild_grammar_string(language: Any) -> str:
    """Auto-derived grammar description from each Sentence type's Pydantic
    spec. Lists each sentence type's fields (with type names), then lists
    every distinct sub-model and enum referenced anywhere — once each —
    expanded with their fields and descriptions (which already include
    the package's vocabulary lists, since vocab.py lemmas are surfaced
    via each lemma-typed Field's `json_schema_extra['description']`).

    No per-language work needed; the spec IS the grammar.
    """
    import enum
    import typing

    from pydantic import BaseModel

    referenced_models: dict[str, type[BaseModel]] = {}
    referenced_enums: dict[str, type[enum.Enum]] = {}

    def _record(annotation: Any) -> None:
        origin = typing.get_origin(annotation)
        if origin is typing.Union:
            for a in typing.get_args(annotation):
                if a is not type(None):
                    _record(a)
            return
        if origin in (list, tuple):
            for a in typing.get_args(annotation):
                _record(a)
            return
        if isinstance(annotation, type):
            if issubclass(annotation, enum.Enum):
                referenced_enums[annotation.__name__] = annotation
            elif issubclass(annotation, BaseModel):
                if annotation.__name__ in referenced_models:
                    return
                referenced_models[annotation.__name__] = annotation
                # Recurse into this sub-model's fields
                for finfo in annotation.model_fields.values():
                    if finfo.annotation is not None:
                        _record(finfo.annotation)

    def _format(annotation: Any) -> str:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Union:
            parts = [a for a in args if a is not type(None)]
            if len(parts) == 1:
                return _format(parts[0])
            return " | ".join(_format(p) for p in parts)
        if origin in (list, tuple):
            return f"list of {_format(args[0])}" if args else "list"
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = [m.value if isinstance(m.value, str) else m.name
                      for m in annotation]
            if len(values) <= 8:
                return f"one of {{{', '.join(values)}}}"
            return annotation.__name__
        return getattr(annotation, "__name__", str(annotation))

    def _field_description(finfo: Any) -> str:
        if finfo.description:
            return finfo.description
        extra = getattr(finfo, "json_schema_extra", None) or {}
        if isinstance(extra, dict) and isinstance(extra.get("description"), str):
            return extra["description"]
        return ""

    sentence_types = list(getattr(language, "sentence_types", ()))

    # Pass 1: collect referenced submodels and enums
    for SType in sentence_types:
        for finfo in SType.model_fields.values():
            if finfo.annotation is not None:
                _record(finfo.annotation)

    out: list[str] = ["Sentence patterns:"]
    for SType in sentence_types:
        out.append(f"  {SType.__name__}:")
        for fname, finfo in SType.model_fields.items():
            type_str = _format(finfo.annotation) if finfo.annotation else "?"
            out.append(f"    - {fname}: {type_str}")

    if referenced_models:
        out.append("")
        out.append("Field types (vocabulary is encoded in each "
                   "lemma field's description):")
        # Skip the sentence types themselves to avoid duplicating
        sentence_type_names = {S.__name__ for S in sentence_types}
        for name, model in referenced_models.items():
            if name in sentence_type_names:
                continue
            out.append(f"  {name}:")
            for fname, finfo in model.model_fields.items():
                type_str = _format(finfo.annotation) if finfo.annotation else "?"
                desc = _field_description(finfo)
                line = f"    - {fname}: {type_str}"
                if desc:
                    line += f"\n        {desc}"
                out.append(line)

    return "\n".join(out)


def _resolve_grammar_string(language: Any) -> str:
    """Per-language grammar description for the caption prompt.

    Each `yaduha_{iso}` package may define a top-level `get_grammar()
    -> str` callable that returns a free-form schema description (or any
    other guidance about which sentence patterns the package handles).
    Captioner uses it verbatim. Otherwise falls back to introspecting
    the language's `sentence_types` Pydantic schemas.
    """
    iso = language.code
    try:
        import importlib
        mod = importlib.import_module(f"yaduha_{iso}")
        getter = getattr(mod, "get_grammar", None)
        if callable(getter):
            result = getter()
            if isinstance(result, str) and result.strip():
                return result.strip()
    except Exception:  # noqa: BLE001
        pass
    return _autobuild_grammar_string(language)


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
    grammar_block = _resolve_grammar_string(language)
    prompt = CAPTION_SYSTEM_PROMPT_BASE
    if grammar_block:
        prompt += CAPTION_GRAMMAR_HEADER.format(
            name=language.name, grammar_block=grammar_block)
    return prompt


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
