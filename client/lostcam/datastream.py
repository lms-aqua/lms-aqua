"""The sensor and AR data channel: NDJSON in, typed samples out.

Implements the consumer requirements from docs/PROTOCOL.md §6.4 — split reads,
skip-don't-die on a bad line, tolerate unknown channels and fields, and detect
loss from ``seq`` gaps rather than assuming delivery.
"""

from __future__ import annotations

import http.client
import json
import math
import threading
import urllib.parse
from dataclasses import dataclass, field

from .netutil import read_available

DEFAULT_MAX_LINE_BYTES = 1 * 1024 * 1024
READ_CHUNK = 16384


class DataStreamError(Exception):
    """The data channel could not be opened, or the stream broke."""


@dataclass(frozen=True)
class Sample:
    """One record off the data channel.

    ``raw`` keeps every field, including ones this version has never heard of,
    so a bridge can forward what it cannot interpret.
    """

    t: int
    seq: int
    channel: str
    raw: dict

    def get(self, key: str, default: object = None) -> object:
        return self.raw.get(key, default)

    @property
    def is_ar(self) -> bool:
        return self.channel.startswith("ar.")

    def flatten(self, prefix: str = "") -> dict[str, float | str | bool]:
        """Flatten to scalar leaves, for CSV columns and OSC addresses.

        ``{"blend": {"jawOpen": 0.4}}`` becomes ``{"blend.jawOpen": 0.4}`` and
        ``{"q": [1,2]}`` becomes ``{"q.0": 1, "q.1": 2}``.
        """
        out: dict[str, float | str | bool] = {}
        _flatten_into(self.raw, prefix, out)
        return out


def _flatten_into(value: object, prefix: str, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_into(item, f"{prefix}{key}.", out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_into(item, f"{prefix}{index}.", out)
    else:
        key = prefix[:-1] if prefix.endswith(".") else prefix
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value


class NDJSONParser:
    """Incremental newline-delimited JSON parser.

    Counts malformed lines instead of raising: a single corrupt line on a live
    telemetry stream is not a reason to drop the connection.
    """

    def __init__(self, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> None:
        self.max_line_bytes = max_line_bytes
        self._buf = bytearray()
        self.bad_lines = 0

    def feed(self, chunk: bytes) -> list[dict]:
        if chunk:
            self._buf += chunk
        records: list[dict] = []
        while True:
            index = self._buf.find(b"\n")
            if index < 0:
                break
            line = bytes(self._buf[:index])
            del self._buf[: index + 1]
            record = self._parse(line)
            if record is not None:
                records.append(record)
        if len(self._buf) > self.max_line_bytes:
            # A producer emitting an unterminated megabyte is broken; drop the
            # partial line rather than growing without bound.
            self._buf.clear()
            self.bad_lines += 1
        return records

    def _parse(self, line: bytes) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            record = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.bad_lines += 1
            return None
        if not isinstance(record, dict):
            self.bad_lines += 1
            return None
        return record

    @property
    def buffered(self) -> int:
        return len(self._buf)


def to_sample(record: dict) -> Sample | None:
    """Validate a record into a ``Sample``, or ``None`` if it is not one."""
    channel = record.get("ch")
    if not isinstance(channel, str) or not channel:
        return None
    t = record.get("t")
    seq = record.get("seq")
    if not isinstance(t, (int, float)) or isinstance(t, bool):
        return None
    if not isinstance(seq, (int, float)) or isinstance(seq, bool):
        seq = 0
    return Sample(t=int(t), seq=int(seq), channel=channel, raw=record)


@dataclass
class DataStats:
    samples: int = 0
    dropped: int = 0
    bad_lines: int = 0
    per_channel: dict[str, int] = field(default_factory=dict)
    _last_seq: int = 0

    def observe(self, sample: Sample) -> None:
        self.samples += 1
        self.per_channel[sample.channel] = self.per_channel.get(sample.channel, 0) + 1
        if sample.seq and self._last_seq and sample.seq > self._last_seq + 1:
            self.dropped += sample.seq - self._last_seq - 1
        if sample.seq:
            self._last_seq = max(self._last_seq, sample.seq)


class DataPuller:
    """Streams ``/data`` from a sender and hands samples to a callback."""

    def __init__(
        self,
        host: str,
        port: int = 4747,
        path: str = "/data",
        channels: list[str] | None = None,
        hz: int | None = None,
        token: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.channels = channels
        self.hz = hz
        self.token = token
        self.timeout = timeout
        self.stats = DataStats()
        self._stop = threading.Event()

    @property
    def request_path(self) -> str:
        params: dict[str, str] = {}
        if self.channels:
            params["ch"] = ",".join(self.channels)
        if self.hz:
            params["hz"] = str(self.hz)
        if self.token:
            params["token"] = self.token
        if not params:
            return self.path
        return f"{self.path}?{urllib.parse.urlencode(params)}"

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run_once(self, on_sample: callable) -> None:
        headers = {"User-Agent": "LostCam/1.0", "Accept": "application/x-ndjson"}
        if self.token:
            headers["X-LostCam-Token"] = self.token
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            try:
                conn.request("GET", self.request_path, headers=headers)
                response = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise DataStreamError(f"could not open /data: {exc}") from exc

            if response.status == 404:
                raise DataStreamError(
                    "this sender has no /data endpoint — it is a v1 sender "
                    "(or DroidCam), which carries video only"
                )
            if response.status != 200:
                raise DataStreamError(
                    f"/data returned HTTP {response.status} {response.reason}"
                )

            parser = NDJSONParser()
            seen_bad = 0
            while not self._stop.is_set():
                try:
                    chunk = read_available(response, READ_CHUNK)
                except (TimeoutError, OSError) as exc:
                    raise DataStreamError(f"data stream read failed: {exc}") from exc
                if not chunk:
                    return
                for record in parser.feed(chunk):
                    sample = to_sample(record)
                    if sample is None:
                        self.stats.bad_lines += 1
                        continue
                    self.stats.observe(sample)
                    on_sample(sample)
                # Fold in lines the parser rejected since the last read.
                self.stats.bad_lines += parser.bad_lines - seen_bad
                seen_bad = parser.bad_lines
        finally:
            conn.close()

    def run_forever(
        self,
        on_sample: callable,
        retry_delay: float = 1.0,
        max_retry_delay: float = 15.0,
        on_error: callable | None = None,
    ) -> None:
        delay = retry_delay
        while not self._stop.is_set():
            try:
                self.run_once(on_sample)
                delay = retry_delay
            except DataStreamError as exc:
                if on_error:
                    on_error(exc)
            if self._stop.is_set():
                return
            self._stop.wait(delay)
            delay = min(max_retry_delay, delay * 2)


# -- small maths helpers consumers keep needing ------------------------------


#: Above this |sin(yaw)| the rotation is at gimbal lock and pitch/roll are no
#: longer independently determined. 0.99999 corresponds to about 0.26° from the
#: pole, comfortably outside sensor noise but inside where the maths degenerates.
GIMBAL_LOCK_THRESHOLD = 0.99999


def quaternion_to_euler(q: list[float]) -> tuple[float, float, float]:
    """``[x,y,z,w]`` to ``(pitch, yaw, roll)`` in degrees.

    Present because senders may omit ``euler`` and because doing this wrong is a
    rite of passage.

    **Gimbal lock is handled explicitly, and it has to be.** At yaw = ±90° only
    the *sum* of pitch and roll is determined — infinitely many (pitch, roll)
    pairs describe the same rotation. The naive formula computes
    ``atan2(0, 1 - 2(x² + y²))``, whose second argument lands on either side of
    zero depending on the platform's ``sin``: measured as ``0.0`` on Linux and
    ``-2.2e-16`` on Windows, giving 0° and 180° for the identical input. Both are
    correct, which is precisely the problem — a sender and a consumer on
    different machines would disagree.

    So at the pole the convention is pinned: roll is defined as 0 and the whole
    remaining rotation goes into pitch. That is the standard resolution, it is
    deterministic across platforms, and the iOS and Android senders implement the
    same rule so all three agree byte for byte.
    """
    if len(q) != 4:
        raise ValueError("quaternion must have 4 components [x,y,z,w]")
    x, y, z, w = (float(v) for v in q)

    sin_yaw = 2.0 * (w * y - z * x)
    # Clamp before asin: a denormalised quaternion can push this past 1 and
    # produce a NaN that then poisons every downstream consumer.
    sin_yaw = max(-1.0, min(1.0, sin_yaw))
    yaw = math.asin(sin_yaw)

    if abs(sin_yaw) >= GIMBAL_LOCK_THRESHOLD:
        # At the pole, fold everything into pitch and define roll as zero.
        pitch = 2.0 * math.atan2(x, w)
        # Keep pitch in (-180, 180] rather than letting it wrap to ±360.
        if pitch > math.pi:
            pitch -= 2.0 * math.pi
        elif pitch < -math.pi:
            pitch += 2.0 * math.pi
        roll = 0.0
    else:
        pitch = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def pose_translation(pose: list[float]) -> tuple[float, float, float]:
    """Extract the translation from a 16-element column-major 4x4 matrix."""
    if len(pose) != 16:
        raise ValueError("pose must have 16 elements (4x4 column-major)")
    return (float(pose[12]), float(pose[13]), float(pose[14]))
