"""Microphone transport.

Same honesty as the video side: LostCam installs no audio driver. It pulls raw
PCM from the sender and plays it into an output device you nominate. Point that
at a loopback device and the phone's mic becomes a system microphone:

- Windows: VB-CABLE (``CABLE Input``), then select ``CABLE Output`` as the mic.
- Linux: a PipeWire/PulseAudio null sink, then use its ``.monitor`` as the mic.

``sounddevice`` is optional; without it the stream can still be pulled and
measured, which is what the tests do.
"""

from __future__ import annotations

import http.client
import threading
import urllib.parse
from dataclasses import dataclass

import numpy as np

BYTES_PER_SAMPLE = 2  # s16le


class AudioError(Exception):
    """The audio stream could not be opened or played."""


@dataclass(frozen=True)
class AudioFormat:
    rate: int = 44100
    channels: int = 1

    @property
    def frame_bytes(self) -> int:
        """Bytes for one sample across all channels."""
        return BYTES_PER_SAMPLE * self.channels

    def duration(self, byte_count: int) -> float:
        """Seconds of audio a byte count represents."""
        return byte_count / (self.frame_bytes * self.rate)


def parse_audio_content_type(value: str | None) -> AudioFormat:
    """Read rate/channels out of ``audio/L16; rate=44100; channels=1``."""
    fmt = AudioFormat()
    if not value:
        return fmt
    rate, channels = fmt.rate, fmt.channels
    for part in value.split(";")[1:]:
        key, _, raw = part.strip().partition("=")
        key = key.strip().lower()
        raw = raw.strip().strip('"')
        try:
            parsed = int(raw)
        except ValueError:
            continue
        if key == "rate" and parsed > 0:
            rate = parsed
        elif key == "channels" and parsed > 0:
            channels = parsed
    return AudioFormat(rate=rate, channels=channels)


def s16le_to_float32(data: bytes, channels: int = 1) -> np.ndarray:
    """Convert raw s16le bytes to ``(frames, channels)`` float32 in [-1, 1].

    Trailing bytes from a partial sample are discarded rather than misaligning
    the whole stream.
    """
    frame_bytes = BYTES_PER_SAMPLE * channels
    usable = len(data) - (len(data) % frame_bytes)
    samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, channels)


class AudioPuller:
    """Streams ``/audio`` from the sender and hands chunks to a callback."""

    def __init__(
        self,
        host: str,
        port: int = 4747,
        path: str = "/audio",
        token: str | None = None,
        timeout: float = 10.0,
        chunk_bytes: int = 4096,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.token = token
        self.timeout = timeout
        self.chunk_bytes = chunk_bytes
        self.format = AudioFormat()
        self.bytes_in = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self, on_chunk: callable) -> None:
        path = self.path
        if self.token:
            path += "?" + urllib.parse.urlencode({"token": self.token})
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request("GET", path, headers={"User-Agent": "LostCam/1.0"})
            response = conn.getresponse()
            if response.status != 200:
                raise AudioError(
                    f"audio stream unavailable: HTTP {response.status} "
                    f"{response.reason}"
                )
            self.format = parse_audio_content_type(response.getheader("Content-Type"))
            while not self._stop.is_set():
                chunk = response.read(self.chunk_bytes)
                if not chunk:
                    return
                self.bytes_in += len(chunk)
                on_chunk(chunk, self.format)
        except (OSError, http.client.HTTPException) as exc:
            raise AudioError(f"audio stream failed: {exc}") from exc
        finally:
            conn.close()


class Speaker:
    """Plays PCM chunks into an output device via ``sounddevice``."""

    def __init__(self, device: str | int | None = None) -> None:
        self.device = device
        self._stream = None
        self._format: AudioFormat | None = None

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def list_devices() -> list[str]:
        try:
            import sounddevice
        except Exception as exc:
            raise AudioError(
                "listing audio devices needs sounddevice: "
                "pip install -e client[audio]"
            ) from exc
        out = []
        for index, info in enumerate(sounddevice.query_devices()):
            if info.get("max_output_channels", 0) > 0:
                out.append(f"{index}: {info['name']}")
        return out

    def write(self, chunk: bytes, fmt: AudioFormat) -> None:
        if self._stream is None or self._format != fmt:
            self._open(fmt)
        assert self._stream is not None
        self._stream.write(s16le_to_float32(chunk, fmt.channels))

    def _open(self, fmt: AudioFormat) -> None:
        try:
            import sounddevice
        except Exception as exc:
            raise AudioError(
                "audio playback needs sounddevice: pip install -e client[audio]"
            ) from exc
        self.close()
        try:
            self._stream = sounddevice.OutputStream(
                samplerate=fmt.rate,
                channels=fmt.channels,
                dtype="float32",
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise AudioError(f"could not open audio output: {exc}") from exc
        self._format = fmt

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._stream = None
