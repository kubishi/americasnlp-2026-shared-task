"""Anthropic helpers: relaxed-model `AnthropicAgent` + vision image block.

Yaduha's `AnthropicAgent.model` is a hard `Literal` of older model IDs. We
need newer models (Sonnet 4.5+, Opus 4.7) so we subclass with a `str` field.
We also need the Anthropic vision message shape, which the upstream agent
doesn't produce since it serializes content as plain text.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import ClassVar

from PIL import Image
from pydantic import Field

from yaduha.agent.anthropic import AnthropicAgent as _UpstreamAnthropicAgent


class AnthropicAgent(_UpstreamAnthropicAgent):
    """Same behavior as upstream, but accepts any model ID string."""
    model: str = Field(..., description="Anthropic model ID")  # type: ignore[assignment]
    name: ClassVar[str] = "anthropic_agent"


def image_block(path: Path, *,
                max_dim: int = 1568,
                quality: int = 85) -> dict:
    """Anthropic vision message block — Pillow-normalized JPEG, downsized.

    The Anthropic vision API caps each image at 5 MB base64. Bribri images in
    the dev set go up to ~9MP (21 MB as PNG), so we resize to `max_dim`
    on the long side (Anthropic's recommended ceiling for sharp inference)
    and re-encode as JPEG. Pillow handles whatever the actual format is, so
    Bribri's mislabelled `.jpg` files (some are WebP/PNG bytes) still load.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": encoded,
        },
    }
