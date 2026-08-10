"""Depth frame streaming (docs/PROTOCOL.md §7).

The payload is a row-major u16 little-endian millimetre raster, where 0 means
"no measurement" rather than "zero distance". Everything here preserves that
distinction, because conflating the two turns absent data into a wall 0 mm from
the lens.
"""

from __future__ import annotations

import http.client
import threading
import urllib.parse
from dataclasses import dataclass

import numpy as np

from .mjpeg import parse_boundary
from .netutil import read_available

READ_CHUNK = 65536
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024


class DepthError(Exception):
    """The depth stream is unavailable or malformed."""


@dataclass(frozen=True)
class DepthFrame:
    """One depth raster plus the intrinsics needed to project it."""

    millimetres: np.ndarray  # (H, W) uint16, 0 = no measurement
    timestamp_ms: int
    intrinsics: tuple[float, float, float, float] | None  # fx, fy, cx, cy

    @property
    def width(self) -> int:
        return int(self.millimetres.shape[1])

    @property
    def height(self) -> int:
        return int(self.millimetres.shape[0])

    @property
    def valid_mask(self) -> np.ndarray:
        """True where a real measurement exists."""
        return self.millimetres > 0

    def metres(self) -> np.ndarray:
        """Float metres with invalid pixels as NaN.

        NaN rather than 0 so that a mean over the frame cannot be silently
        dragged toward zero by pixels the sensor never measured.
        """
        out = self.millimetres.astype(np.float32) / 1000.0
        out[~self.valid_mask] = np.nan
        return out

    @property
    def coverage(self) -> float:
        """Fraction of pixels carrying a measurement. A data-quality signal."""
        total = self.millimetres.size
        return float(self.valid_mask.sum() / total) if total else 0.0


def parse_depth_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.split(b"\r\n"):
        name, sep, value = line.partition(b":")
        if not sep:
            continue
        headers[name.strip().lower().decode("ascii", "replace")] = (
            value.strip().decode("ascii", "replace")
        )
    return headers


def parse_depth_geometry(headers: dict[str, str]) -> tuple[int, int, str]:
    """Read ``X-LostCam-Depth: <w>x<h>; format=u16mm``."""
    raw = headers.get("x-lostcam-depth", "")
    if not raw:
        raise DepthError("depth part is missing the X-LostCam-Depth header")
    dimensions, _, rest = raw.partition(";")
    width_text, _, height_text = dimensions.strip().partition("x")
    try:
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise DepthError(f"malformed depth dimensions: {dimensions!r}") from exc
    if width <= 0 or height <= 0:
        raise DepthError(f"invalid depth dimensions: {width}x{height}")

    fmt = "u16mm"
    for part in rest.split(";"):
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "format":
            fmt = value.strip() or fmt
    return width, height, fmt


def parse_intrinsics(headers: dict[str, str]) -> tuple[float, float, float, float] | None:
    raw = headers.get("x-lostcam-intrinsics")
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return None
    try:
        fx, fy, cx, cy = (float(p) for p in parts)
    except ValueError:
        return None
    return (fx, fy, cx, cy)


def decode_depth(payload: bytes, width: int, height: int,
                 fmt: str = "u16mm") -> np.ndarray:
    """Decode a raw depth payload into an ``(H, W)`` uint16 array."""
    if fmt != "u16mm":
        raise DepthError(f"unsupported depth format {fmt!r}; expected u16mm")
    expected = width * height * 2
    if len(payload) != expected:
        raise DepthError(
            f"depth payload is {len(payload)} bytes, expected {expected} "
            f"for {width}x{height} u16"
        )
    return np.frombuffer(payload, dtype="<u2").reshape(height, width)


class DepthParser:
    """Incremental multipart parser for the depth stream.

    Separate from MJPEGParser because a depth part is not self-delimiting: there
    are no SOI/EOI markers to scan for, so Content-Length is mandatory and the
    per-part headers must be read to learn the raster size.
    """

    def __init__(self, boundary: str,
                 max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> None:
        self.boundary = boundary.lstrip("-")
        self.max_frame_bytes = max_frame_bytes
        self._buf = bytearray()
        self._pending: dict | None = None

    def feed(self, chunk: bytes) -> list[DepthFrame]:
        if chunk:
            self._buf += chunk
        frames: list[DepthFrame] = []
        while True:
            frame = self._next()
            if frame is None:
                break
            frames.append(frame)
        if len(self._buf) > self.max_frame_bytes:
            raise DepthError(
                f"buffered {len(self._buf)} bytes without a complete depth frame"
            )
        return frames

    def _next(self) -> DepthFrame | None:
        if self._pending is None:
            marker = b"--" + self.boundary.encode("ascii")
            index = self._buf.find(marker)
            if index < 0:
                if len(self._buf) > len(marker):
                    del self._buf[: len(self._buf) - len(marker)]
                return None
            start = index + len(marker)
            end = self._buf.find(b"\r\n\r\n", start)
            if end < 0:
                if index > 0:
                    del self._buf[:index]
                return None
            headers = parse_depth_headers(bytes(self._buf[start:end]))
            del self._buf[: end + 4]

            width, height, fmt = parse_depth_geometry(headers)
            try:
                length = int(headers.get("content-length", ""))
            except ValueError as exc:
                raise DepthError("depth part has no usable Content-Length") from exc
            if length > self.max_frame_bytes:
                raise DepthError(f"depth part declares {length} bytes, over cap")

            self._pending = {
                "length": length,
                "width": width,
                "height": height,
                "format": fmt,
                "timestamp": _int_or_zero(headers.get("x-lostcam-timestamp")),
                "intrinsics": parse_intrinsics(headers),
            }

        pending = self._pending
        if len(self._buf) < pending["length"]:
            return None
        payload = bytes(self._buf[: pending["length"]])
        del self._buf[: pending["length"]]
        self._pending = None

        array = decode_depth(payload, pending["width"], pending["height"],
                             pending["format"])
        return DepthFrame(
            millimetres=array,
            timestamp_ms=pending["timestamp"],
            intrinsics=pending["intrinsics"],
        )


def _int_or_zero(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


class DepthPuller:
    """Streams ``/depth`` from a sender and hands frames to a callback."""

    def __init__(self, host: str, port: int = 4747, path: str = "/depth",
                 token: str | None = None, timeout: float = 15.0) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.token = token
        self.timeout = timeout
        self.frames = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self, on_frame: callable) -> None:
        path = self.path
        headers = {"User-Agent": "LostCam/1.0"}
        if self.token:
            path += "?" + urllib.parse.urlencode({"token": self.token})
            headers["X-LostCam-Token"] = self.token

        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            try:
                conn.request("GET", path, headers=headers)
                response = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise DepthError(f"could not open /depth: {exc}") from exc

            if response.status == 404:
                raise DepthError(
                    "this sender has no /depth endpoint — depth needs a LiDAR "
                    "device (iPhone/iPad Pro) with the depth channel enabled"
                )
            if response.status != 200:
                raise DepthError(
                    f"/depth returned HTTP {response.status} {response.reason}"
                )

            boundary = parse_boundary(response.getheader("Content-Type"))
            if not boundary:
                raise DepthError("depth stream has no multipart boundary")
            parser = DepthParser(boundary)

            while not self._stop.is_set():
                try:
                    chunk = read_available(response, READ_CHUNK)
                except (TimeoutError, OSError) as exc:
                    raise DepthError(f"depth read failed: {exc}") from exc
                if not chunk:
                    return
                for frame in parser.feed(chunk):
                    self.frames += 1
                    on_frame(frame)
        finally:
            conn.close()
