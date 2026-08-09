"""JPEG decoding to RGB arrays.

Pillow is the dependable baseline. OpenCV, when installed, decodes noticeably
faster, so it is used opportunistically without becoming a hard dependency.
"""

from __future__ import annotations

import io

import numpy as np

try:  # pragma: no cover - depends on optional install
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from PIL import Image


class DecodeError(Exception):
    """The bytes handed over were not a decodable JPEG."""


def jpeg_to_rgb(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into an ``(H, W, 3)`` uint8 RGB array."""
    if not data:
        raise DecodeError("empty frame")

    if cv2 is not None:  # pragma: no cover - exercised only when cv2 present
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is not None:
            return np.ascontiguousarray(bgr[:, :, ::-1])

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # Pillow raises a variety of types here
        raise DecodeError(f"could not decode JPEG ({exc})") from exc


def rgb_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB array back to JPEG. Used by tests and the self-test."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")
    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=int(quality)
    )
    return buffer.getvalue()
