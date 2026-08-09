"""Geometry fixes applied between the phone and the virtual camera.

Phones hand over frames in whatever orientation the sensor felt like, and the
virtual camera has one fixed resolution for its whole lifetime. So every frame
is rotated/mirrored to taste and then forced to exactly the target size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

FitMode = str  # "contain" | "cover" | "stretch"
VALID_FIT_MODES = ("contain", "cover", "stretch")


@dataclass(frozen=True)
class Transform:
    """Orientation correction. ``rotate`` is counter-clockwise degrees."""

    rotate: int = 0
    hflip: bool = False
    vflip: bool = False

    def __post_init__(self) -> None:
        if self.rotate % 90 != 0:
            raise ValueError("rotate must be a multiple of 90")

    @property
    def quarter_turns(self) -> int:
        return (self.rotate // 90) % 4

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if self.quarter_turns:
            out = np.rot90(out, self.quarter_turns)
        if self.hflip:
            out = out[:, ::-1]
        if self.vflip:
            out = out[::-1, :]
        # rot90/slicing produce views; the virtual camera needs real memory.
        return np.ascontiguousarray(out)


def fit(
    frame: np.ndarray,
    width: int,
    height: int,
    mode: FitMode = "contain",
    background: int = 0,
) -> np.ndarray:
    """Resize ``frame`` to exactly ``width`` x ``height``.

    ``contain`` letterboxes to preserve aspect ratio, ``cover`` fills and
    centre-crops, ``stretch`` ignores aspect ratio entirely.
    """
    if width <= 0 or height <= 0:
        raise ValueError("target size must be positive")
    if mode not in VALID_FIT_MODES:
        raise ValueError(f"mode must be one of {VALID_FIT_MODES}")

    src_h, src_w = frame.shape[:2]
    if src_w == width and src_h == height:
        return np.ascontiguousarray(frame)

    if mode == "stretch":
        return _resize(frame, width, height)

    scale = (
        min(width / src_w, height / src_h)
        if mode == "contain"
        else max(width / src_w, height / src_h)
    )
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = _resize(frame, new_w, new_h)

    if mode == "cover":
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return np.ascontiguousarray(
            resized[top : top + height, left : left + width]
        )

    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
