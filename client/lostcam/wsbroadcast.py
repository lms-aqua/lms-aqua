"""A WebSocket broadcast server, so browsers can consume the data channel.

The phone serves plain HTTP NDJSON, which a browser page on a different origin
cannot read conveniently. This turns the stream into a WebSocket that any local
page can subscribe to, and adds the CORS header that makes it usable.

Slow clients are dropped rather than allowed to apply back-pressure: telemetry
is only worth having live, and a dashboard that fell behind wants the newest
sample, not a backlog.
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler

from . import wsproto
from .datastream import Sample

DEFAULT_PORT = 8765
MAX_QUEUED_BYTES = 1 * 1024 * 1024


class _Client:
    """One connected browser."""

    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.alive = True
        self.dropped = 0
        self._lock = threading.Lock()

    def send_text(self, payload: bytes) -> bool:
        if not self.alive:
            return False
        with self._lock:
            try:
                self.connection.sendall(
                    wsproto.encode_frame(payload, wsproto.OP_TEXT)
                )
                return True
            except OSError:
                self.alive = False
                return False

    def close(self) -> None:
        self.alive = False
        try:
            self.connection.close()
        except OSError:
            pass


class WSBroadcastServer:
    """Accepts WebSocket clients and pushes every sample to all of them."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self.sent = 0
        self.dropped = 0
        self._clients: list[_Client] = []
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        handler = _make_handler(self)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

    @property
    def bound_port(self) -> int:
        if not self._server:
            return self.port
        return self._server.socket.getsockname()[1]

    @property
    def client_count(self) -> int:
        with self._lock:
            return sum(1 for client in self._clients if client.alive)

    def stop(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.close()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> WSBroadcastServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- broadcasting --------------------------------------------------------

    def _register(self, client: _Client) -> None:
        with self._lock:
            self._clients.append(client)

    def _unregister(self, client: _Client) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def broadcast(self, sample: Sample) -> None:
        payload = json.dumps(sample.raw, separators=(",", ":")).encode("utf-8")
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            if not client.send_text(payload):
                self.dropped += 1
                self._unregister(client)
        self.sent += 1

    def as_sink(self):
        """Wrap as a bridge sink."""
        from .bridge import CallbackSink

        return CallbackSink(self.broadcast, name="websocket")


def _make_handler(server: WSBroadcastServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LostCam-Bridge/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            route = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
            if route == "/healthz":
                self._text(200, "ok")
            elif route in ("/", "/ws"):
                self._upgrade()
            else:
                self._text(404, "not found")

        def _text(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def _upgrade(self) -> None:
            key = self.headers.get("Sec-WebSocket-Key")
            if not key or (self.headers.get("Upgrade") or "").lower() != "websocket":
                self._text(400, "expected a WebSocket upgrade")
                return
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", wsproto.accept_key(key))
            self.end_headers()
            self.wfile.flush()

            client = _Client(self.connection)
            server._register(client)
            try:
                self._drain(client)
            finally:
                server._unregister(client)
                client.close()
                self.close_connection = True

        def _drain(self, client: _Client) -> None:
            """Read and discard client frames; we only care about the close."""
            decoder = wsproto.FrameDecoder()
            self.connection.settimeout(60.0)
            while client.alive:
                try:
                    chunk = self.connection.recv(4096)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                try:
                    for message in decoder.feed(chunk):
                        if message.opcode == wsproto.OP_CLOSE:
                            return
                        if message.opcode == wsproto.OP_PING:
                            client.send_text(b"")  # keep-alive is enough
                except wsproto.WSError:
                    return

    return Handler
