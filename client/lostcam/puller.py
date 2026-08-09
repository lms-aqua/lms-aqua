"""Pull mode: connect out to a phone that is serving MJPEG.

This is the DroidCam-compatible direction. It works against the LostCam iOS
app and, because it speaks nothing but HTTP and MJPEG, against DroidCam's own
``http://<ip>:4747/video`` endpoint too.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass

from .mjpeg import MJPEGError, MJPEGParser, parse_boundary
from .pipeline import FramePipeline

DEFAULT_PORT = 4747
READ_CHUNK = 65536


class ConnectionFailed(Exception):
    """The phone could not be reached, or refused the stream."""


@dataclass(frozen=True)
class Source:
    """Where to pull from, and what to ask for."""

    host: str
    port: int = DEFAULT_PORT
    path: str = "/video"
    token: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    quality: int | None = None
    camera: str | None = None
    timeout: float = 10.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.request_path}"

    @property
    def request_path(self) -> str:
        params = {}
        if self.width:
            params["w"] = self.width
        if self.height:
            params["h"] = self.height
        if self.fps:
            params["fps"] = self.fps
        if self.quality:
            params["q"] = self.quality
        if self.camera:
            params["cam"] = self.camera
        if self.token:
            params["token"] = self.token
        if not params:
            return self.path
        return f"{self.path}?{urllib.parse.urlencode(params)}"

    @property
    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": "LostCam/1.0", "Accept": "multipart/x-mixed-replace"}
        if self.token:
            headers["X-LostCam-Token"] = self.token
        return headers


def probe_info(source: Source) -> dict | None:
    """Fetch ``/info``. Returns ``None`` if the sender does not implement it."""
    conn = http.client.HTTPConnection(source.host, source.port, timeout=source.timeout)
    try:
        path = "/info"
        if source.token:
            path += "?" + urllib.parse.urlencode({"token": source.token})
        conn.request("GET", path, headers=source.headers)
        response = conn.getresponse()
        body = response.read(64 * 1024)
        if response.status != 200:
            return None
        return json.loads(body.decode("utf-8"))
    except (OSError, http.client.HTTPException, ValueError, UnicodeDecodeError):
        return None
    finally:
        conn.close()


def resolve_size(source: Source, info: dict | None) -> tuple[int, int] | None:
    """Work out the virtual camera size from explicit flags, then ``/info``."""
    if source.width and source.height:
        return source.width, source.height
    if info:
        video = info.get("video")
        if isinstance(video, dict):
            width, height = video.get("width"), video.get("height")
            if (
                isinstance(width, int)
                and isinstance(height, int)
                and width > 0
                and height > 0
            ):
                return width, height
    return None


class Puller:
    """Streams from one source into a pipeline until stopped."""

    def __init__(self, source: Source, pipeline: FramePipeline) -> None:
        self.source = source
        self.pipeline = pipeline
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run_once(self) -> None:
        """One connection attempt; returns when the stream ends or stops."""
        conn = http.client.HTTPConnection(
            self.source.host, self.source.port, timeout=self.source.timeout
        )
        try:
            try:
                conn.request("GET", self.source.request_path, headers=self.source.headers)
                response = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise ConnectionFailed(f"{self.source.url}: {exc}") from exc

            if response.status == 401:
                raise ConnectionFailed(
                    f"{self.source.url}: 401 Unauthorized — the sender wants a "
                    "token (pass --token)"
                )
            if response.status != 200:
                raise ConnectionFailed(
                    f"{self.source.url}: HTTP {response.status} {response.reason}"
                )

            content_type = response.getheader("Content-Type")
            parser = MJPEGParser(parse_boundary(content_type))
            self._pump(response, parser)
        finally:
            conn.close()

    def _pump(self, response: http.client.HTTPResponse, parser: MJPEGParser) -> None:
        while not self._stop.is_set():
            try:
                chunk = response.read(READ_CHUNK)
            except TimeoutError as exc:
                raise ConnectionFailed(f"read timed out: {exc}") from exc
            except OSError as exc:
                raise ConnectionFailed(f"read failed: {exc}") from exc
            if not chunk:
                return  # sender closed the stream
            try:
                frames = parser.feed(chunk)
            except MJPEGError as exc:
                raise ConnectionFailed(f"malformed MJPEG stream: {exc}") from exc
            for frame in frames:
                if self._stop.is_set():
                    return
                self.pipeline.submit(frame)

    def run_forever(
        self,
        retry_delay: float = 1.0,
        max_retry_delay: float = 15.0,
        on_error: callable | None = None,
    ) -> None:
        """Reconnect with backoff until stopped.

        Wi-Fi drops, phones lock, apps get backgrounded. A webcam that gives up
        on the first hiccup is not useful, so the default is to keep trying.
        """
        delay = retry_delay
        while not self._stop.is_set():
            try:
                self.run_once()
                delay = retry_delay  # a successful connection resets backoff
            except ConnectionFailed as exc:
                if on_error:
                    on_error(exc)
            if self._stop.is_set():
                return
            self._stop.wait(delay)
            delay = min(max_retry_delay, delay * 2)


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Poll a TCP port until it accepts. Used by the CLI and the smoke test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False
