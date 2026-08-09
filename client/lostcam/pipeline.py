"""The shared path from JPEG bytes to the virtual camera.

Both transports — pull mode and push mode — funnel here, so orientation,
scaling, frame accounting and error tolerance behave identically no matter how
the frame arrived.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .decode import DecodeError, jpeg_to_rgb
from .transform import Transform, fit
from .virtualcam import Sink


@dataclass
class Stats:
    """Frame accounting, safe to read from another thread."""

    frames: int = 0
    dropped: int = 0
    decode_errors: int = 0
    bytes_in: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _last_report: float = field(default_factory=time.monotonic)
    _last_frames: int = 0

    @property
    def elapsed(self) -> float:
        return max(1e-9, time.monotonic() - self.started_at)

    @property
    def average_fps(self) -> float:
        return self.frames / self.elapsed

    def instant_fps(self) -> float:
        """FPS since the previous call. Resets its own window."""
        now = time.monotonic()
        window = now - self._last_report
        if window <= 0:
            return 0.0
        fps = (self.frames - self._last_frames) / window
        self._last_report = now
        self._last_frames = self.frames
        return fps


class FramePipeline:
    """Decode → orient → resize → send, with a one-frame latest-wins queue.

    Dropping stale frames rather than queueing them is the whole point: a
    webcam that is a second behind is worse than one that skipped a frame.
    """

    def __init__(
        self,
        sink: Sink,
        transform: Transform | None = None,
        fit_mode: str = "contain",
        stats: Stats | None = None,
    ) -> None:
        self.sink = sink
        self.transform = transform or Transform()
        self.fit_mode = fit_mode
        self.stats = stats or Stats()

    def submit(self, jpeg: bytes) -> bool:
        """Process one JPEG frame. Returns whether it reached the sink.

        A frame that fails to decode is counted and skipped — a single corrupt
        frame on a lossy link must not tear down a working stream.
        """
        self.stats.bytes_in += len(jpeg)
        try:
            frame = jpeg_to_rgb(jpeg)
        except DecodeError:
            self.stats.decode_errors += 1
            return False

        frame = self.transform.apply(frame)
        frame = fit(frame, self.sink.width, self.sink.height, self.fit_mode)
        self.sink.send(frame)
        self.stats.frames += 1
        return True


class LatestFrameBuffer:
    """A one-slot, latest-wins handoff between a reader and a sender thread.

    Used when the sink must be driven at a steady rate (the virtual camera
    prefers regular timing) while frames arrive in bursts.
    """

    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.dropped = 0
        self._closed = False

    def put(self, jpeg: bytes) -> None:
        with self._lock:
            if self._frame is not None:
                self.dropped += 1
            self._frame = jpeg
        self._event.set()

    def get(self, timeout: float | None = None) -> bytes | None:
        """Take the most recent frame, waiting up to ``timeout`` seconds."""
        if not self._event.wait(timeout):
            return None
        with self._lock:
            frame, self._frame = self._frame, None
            self._event.clear()
        return frame

    def close(self) -> None:
        self._closed = True
        self._event.set()

    @property
    def closed(self) -> bool:
        return self._closed


def blank_frame(width: int, height: int, value: int = 0) -> np.ndarray:
    """A solid frame, used to hold the camera open before the phone connects."""
    return np.full((height, width, 3), value, dtype=np.uint8)
