"""Direct VLM prompting baselines (zero-shot and few-shot).

Comparison only — *not* the proposed system. The VLM is asked to emit a
caption directly in the target Indigenous language with no grammatical
guarantees. Few-shot demonstrations are sampled from the dev set when
available (dev split is permitted for training per the shared-task rules).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import anthropic

from americasnlp._anthropic import image_block
from americasnlp.captioners import CaptionResult
from americasnlp.data import (
    load_split,
    resolve_image_path,
    split_dir,
)
from americasnlp.languages import LanguageConfig


SYSTEM_PROMPT_TEMPLATE = (
    "You are an image captioning system for {name} (ISO 639-3: {iso}). "
    "Given an image, produce a single caption in {name}. "
    "Output ONLY the {name} caption text — no translation, no commentary, "
    "no quotation marks, no language labels. "
    "Match the style of the example captions you are shown."
)


@dataclass
class DirectCaptioner:
    """Direct prompting baseline (zero- or few-shot)."""

    lang: LanguageConfig
    data_root: Path
    vlm_model: str = "claude-sonnet-4-5"
    shots: int = 0
    demo_split: str = "dev"
    seed: int = 17
    max_tokens: int = 256
    name: str = ""

    _rng: random.Random = field(init=False, repr=False)
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _demo_pool: List[dict] = field(init=False, repr=False)
    _demo_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; create a .env or export it")
        self._rng = random.Random(self.seed)
        self._client = anthropic.Anthropic(api_key=api_key)
        self.name = "direct" if self.shots == 0 else f"direct-shots{self.shots}"
        try:
            pool = load_split(self.lang, self.demo_split, self.data_root)
        except FileNotFoundError:
            pool = []
        self._demo_pool = [r for r in pool if r.get("target_caption")]
        self._demo_dir = split_dir(self.lang, self.demo_split, self.data_root)

    def _sample_demos(self, target_id: Optional[str]) -> List[dict]:
        if self.shots <= 0 or not self._demo_pool:
            return []
        eligible = [r for r in self._demo_pool if r.get("id") != target_id]
        if len(eligible) <= self.shots:
            return list(eligible)
        return self._rng.sample(eligible, self.shots)

    def caption(self, record: dict, image_path: Path) -> CaptionResult:
        demos = self._sample_demos(record.get("id"))
        content: list = []
        for d in demos:
            content.append(image_block(resolve_image_path(d, self._demo_dir)))
            content.append({"type": "text", "text": f"Caption: {d['target_caption']}"})
        content.append(image_block(image_path))
        content.append({"type": "text", "text": "Caption:"})

        resp = self._client.messages.create(
            model=self.vlm_model,
            max_tokens=self.max_tokens,
            system=[{"type": "text",
                     "text": SYSTEM_PROMPT_TEMPLATE.format(
                         name=self.lang.name, iso=self.lang.iso),
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        if text.lower().startswith("caption:"):
            text = text[len("caption:"):].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“"):
            text = text[1:-1].strip()
        return CaptionResult(target=text, english_intermediate=None)
