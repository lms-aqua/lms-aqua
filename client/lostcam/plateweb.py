"""A live plate dashboard you can leave open next to the printer.

Serves one page on localhost and pushes each measured plate state to it over
Server-Sent Events. SSE rather than a WebSocket because the traffic is entirely
one-way and SSE needs no handshake, no framing and no masking — the browser side
is three lines of ``EventSource``.

Latest-wins per client, like everything else here: a browser tab that fell behind
wants the current state of the plate, not a backlog of stale ones.
"""

from __future__ import annotations

import base64
import json
import socketserver
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from .netutil import is_disconnect
from .plate import HeightMap, PlateCalibration, PlateState

DEFAULT_PORT = 8770
WEB_ROOT = Path(__file__).resolve().parent / "web"
# Send a comment this often when nothing has changed, so an idle connection is
# not dropped by a proxy or by the browser's own timeout.
KEEPALIVE_SECONDS = 15.0


def encode_height_map(height_map: HeightMap) -> dict:
    """Encode a height map for the browser.

    Base64 of the u16 raster rather than a JSON array of numbers. The saving is
    not mainly in bytes — a mostly-flat 63x63 grid is about 10 KB either way —
    it is that the cost is *fixed* at 2 bytes per cell however tall the print
    gets, and that the browser decodes straight into a ``Uint16Array`` the canvas
    path indexes, instead of parsing four thousand separate numbers into a JS
    array on every frame.

    The ``+1`` offset is the same convention as the dataset export, so 0 keeps
    meaning "no measurement" rather than "0 mm".
    """
    grid = height_map.to_u16_mm()
    return {
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "cell_mm": height_map.cell_mm,
        "format": "u16mm+1",
        "data": base64.b64encode(grid.astype("<u2").tobytes()).decode("ascii"),
    }


def build_payload(state: PlateState, height_map: HeightMap | None,
                  calibration: PlateCalibration,
                  timestamp_ms: int = 0,
                  extra: dict | None = None) -> dict:
    """Assemble everything the page needs for one update."""
    payload: dict = {
        "t": int(timestamp_ms),
        "plate": {
            "width_mm": calibration.plate_width_mm,
            "height_mm": calibration.plate_height_mm,
            "cell_mm": calibration.cell_mm,
            "tilt_degrees": round(calibration.plane.tilt_degrees, 2),
        },
    }
    payload.update(state.as_dict())
    if height_map is not None:
        payload["map"] = encode_height_map(height_map)
    if extra:
        payload.update(extra)
    return payload


class _Subscriber:
    """One open browser tab."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.pending: bytes | None = None
        self.alive = True
        self.dropped = 0

    def offer(self, frame: bytes) -> None:
        with self.lock:
            if self.pending is not None:
                # Latest wins: the plate's current state is the only useful one.
                self.dropped += 1
            self.pending = frame
        self.event.set()

    def take(self, timeout: float) -> bytes | None:
        if not self.event.wait(timeout):
            return None
        with self.lock:
            frame, self.pending = self.pending, None
            self.event.clear()
        return frame

    def close(self) -> None:
        self.alive = False
        self.event.set()


class PlateWebServer:
    """Serves the dashboard and streams plate states to it."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self.published = 0
        self._latest: bytes = b"{}"
        self._subscribers: list[_Subscriber] = []
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        handler = _make_handler(self)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

            def handle_error(self, request, client_address) -> None:
                """A closed tab is not an error worth a traceback."""
                if not is_disconnect(sys.exc_info()[1]):
                    super().handle_error(request, client_address)

        self._server = Server((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            daemon=True,
        )
        self._thread.start()

    @property
    def bound_port(self) -> int:
        if not self._server:
            return self.port
        return self._server.socket.getsockname()[1]

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.bound_port}/"

    @property
    def client_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._subscribers if s.alive)

    def stop(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.close()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> PlateWebServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- publishing ----------------------------------------------------------

    def publish(self, payload: dict) -> None:
        """Push one update to every open tab."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # Kept so a tab opened later shows the plate immediately rather than an
        # empty page until the next depth frame.
        self._latest = body
        self.published += 1

        frame = b"data: " + body + b"\n\n"
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.offer(frame)

    def publish_state(self, state: PlateState, height_map: HeightMap | None,
                      calibration: PlateCalibration, timestamp_ms: int = 0,
                      extra: dict | None = None) -> None:
        self.publish(build_payload(state, height_map, calibration,
                                   timestamp_ms, extra))

    @property
    def latest(self) -> bytes:
        return self._latest

    def _register(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers.append(subscriber)

    def _unregister(self, subscriber: _Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


def _make_handler(server: PlateWebServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LostCam-PlateWeb/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            route = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
            if route == "/healthz":
                self._bytes(200, b"ok", "text/plain; charset=utf-8")
            elif route == "/":
                self._page()
            elif route == "/state.json":
                # A polling fallback, and handy for `curl | jq` at the terminal.
                self._bytes(200, server.latest, "application/json")
            elif route == "/events":
                self._events()
            else:
                self._bytes(404, b"not found", "text/plain; charset=utf-8")

        def _bytes(self, status: int, body: bytes, content_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self.close_connection = True

        def _page(self) -> None:
            path = WEB_ROOT / "plate.html"
            try:
                body = path.read_bytes()
            except OSError:
                self._bytes(500, b"dashboard page is missing from the install",
                            "text/plain; charset=utf-8")
                return
            self._bytes(200, body, "text/html; charset=utf-8")

        def _events(self) -> None:
            subscriber = _Subscriber()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                # No transform, or a proxy may buffer the stream into uselessness.
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                # Send the last known state at once, so a tab opened between
                # depth frames is not blank.
                self.wfile.write(b"data: " + server.latest + b"\n\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                return

            server._register(subscriber)
            try:
                while subscriber.alive:
                    frame = subscriber.take(KEEPALIVE_SECONDS)
                    if not subscriber.alive:
                        break
                    # A comment line is a valid SSE keep-alive the browser ignores.
                    self.wfile.write(frame if frame else b": keepalive\n\n")
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                server._unregister(subscriber)
                subscriber.close()
                self.close_connection = True

    return Handler
