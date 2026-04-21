"""OpenAI helpers: image data URL with Pillow normalization + resize.

OpenAI's vision API accepts base64 data URLs in the chat-completions
`image_url` content type. We resize to max 1568px on the long side and
re-encode as JPEG for the same reasons we do with Anthropic — Bribri's
mislabelled `.jpg` files (some are WebP/PNG bytes) and the 5 MB-ish
practical ceiling on per-request image size.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def model_kwargs(model: str, *, max_out: int) -> dict:
    """Return the right kwargs for `chat.completions.create`.

    gpt-5 and the o-series only accept the default temperature (1) and
    use `max_completion_tokens`. Reasoning models also burn output
    tokens internally for hidden reasoning before producing visible
    text, so we ~8x the visible cap to leave room.
    """
    if is_reasoning_model(model):
        return {"max_completion_tokens": max_out * 8}
    return {"max_tokens": max_out, "temperature": 0.0}


def image_data_url(path: Path, *,
                   max_dim: int = 1568,
                   quality: int = 85) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
