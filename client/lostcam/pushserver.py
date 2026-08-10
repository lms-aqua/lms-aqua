"""Push mode: the desktop listens, the phone's browser connects and sends.

DroidCam's phone-is-the-server model is impossible for a web page — a browser
can only make outbound connections. So this reverses the direction: the desktop
serves the sender page over HTTPS and accepts JPEG frames over a WebSocket.
That is what lets any phone with a browser act as the camera, Android included.
"""

from __future__ import annotations

import base64
import hmac
import json
import socketserver
import sys
import threading
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import wsproto
from .netutil import is_disconnect
from .tls import ensure_cert, server_context

DEFAULT_PORT = 8443
WEB_ROOT = Path(__file__).resolve().parent / "web"


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    # A phone that walks out of Wi-Fi range leaves a half-open socket; without
    # this the accept queue fills with sockets nobody will ever read.
    request_queue_size = 16

    def handle_error(self, request, client_address) -> None:
        """Stay quiet about clients that simply hung up.

        A phone leaving Wi-Fi mid-stream is the normal way these connections end,
        and the default behaviour is to print a traceback for it. Anything that is
        not a teardown is still reported — swallowing real errors would hide
        genuine faults.
        """
        if not is_disconnect(sys.exc_info()[1]):
            super().handle_error(request, client_address)


FrameHandler = Callable[[bytes], None]
HelloHandler = Callable[[dict], None]


class PushServer:
    """Serves the sender page and turns inbound WebSocket frames into calls."""

    def __init__(
        self,
        on_frame: FrameHandler,
        on_hello: HelloHandler | None = None,
        port: int = DEFAULT_PORT,
        host: str = "0.0.0.0",
        token: str | None = None,
        use_tls: bool = True,
        cert_dir: Path | None = None,
        cert_hosts: list[str] | None = None,
    ) -> None:
        self.on_frame = on_frame
        self.on_hello = on_hello
        self.port = port
        self.host = host
        self.token = token
        self.use_tls = use_tls
        self.cert_path: Path | None = None
        self.key_path: Path | None = None
        self.clients = 0
        self._lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

        if use_tls:
            self.cert_path, self.key_path = ensure_cert(
                cert_dir, cert_hosts or ["localhost"]
            )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        handler = _make_handler(self)
        self._server = _Server((self.host, self.port), handler)
        if self.use_tls:
            assert self.cert_path and self.key_path
            context = server_context(self.cert_path, self.key_path)
            self._server.socket = context.wrap_socket(
                self._server.socket, server_side=True
            )
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

    @property
    def bound_port(self) -> int:
        """The real port, which matters when 0 was requested."""
        if not self._server:
            return self.port
        return self._server.socket.getsockname()[1]

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> PushServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- helpers -------------------------------------------------------------

    def scheme(self) -> str:
        return "https" if self.use_tls else "http"

    def url_for(self, host: str) -> str:
        url = f"{self.scheme()}://{host}:{self.bound_port}/"
        if self.token:
            url += "?" + urllib.parse.urlencode({"token": self.token})
        return url

    def authorized(self, query: dict[str, list[str]], header: str | None) -> bool:
        """Constant-time token check; open when no token is configured."""
        if not self.token:
            return True
        supplied = header or (query.get("token") or [""])[0]
        return hmac.compare_digest(supplied or "", self.token)

    def _client_delta(self, delta: int) -> None:
        with self._lock:
            self.clients = max(0, self.clients + delta)


def _make_handler(server: PushServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LostCam/1.0"

        # BaseHTTPRequestHandler logs every request to stderr, which is noise
        # at 30 requests a second.
        def log_message(self, fmt: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            route = parsed.path.rstrip("/") or "/"

            if route == "/healthz":
                self._send_text(200, "ok")
                return

            if not server.authorized(query, self.headers.get("X-LostCam-Token")):
                self._send_text(401, "unauthorized: bad or missing token")
                return

            if route == "/":
                self._send_page()
            elif route == "/ws":
                self._do_websocket()
            else:
                self._send_text(404, "not found")

        def _send_text(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_page(self) -> None:
            page = WEB_ROOT / "index.html"
            try:
                payload = page.read_bytes()
            except OSError:
                self._send_text(500, "sender page is missing from the install")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        # -- websocket -------------------------------------------------------

        def _do_websocket(self) -> None:
            key = self.headers.get("Sec-WebSocket-Key")
            upgrade = (self.headers.get("Upgrade") or "").lower()
            if not key or upgrade != "websocket":
                self._send_text(400, "expected a WebSocket upgrade")
                return
            try:
                base64.b64decode(key, validate=True)
            except Exception:
                self._send_text(400, "malformed Sec-WebSocket-Key")
                return

            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", wsproto.accept_key(key))
            self.end_headers()
            self.wfile.flush()

            server._client_delta(1)
            try:
                self._pump_websocket()
            finally:
                server._client_delta(-1)
                self.close_connection = True

        def _pump_websocket(self) -> None:
            decoder = wsproto.FrameDecoder()
            self.connection.settimeout(30.0)
            while True:
                try:
                    chunk = self.connection.recv(65536)
                except TimeoutError:
                    if not self._send_ws(wsproto.encode_frame(b"", wsproto.OP_PING)):
                        return
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                try:
                    messages = decoder.feed(chunk)
                except wsproto.WSError:
                    self._send_ws(wsproto.encode_close(1002, "protocol error"))
                    return
                for message in messages:
                    if not self._handle_message(message):
                        return

        def _handle_message(self, message: wsproto.Message) -> bool:
            if message.opcode == wsproto.OP_CLOSE:
                self._send_ws(wsproto.encode_close(1000, "bye"))
                return False
            if message.opcode == wsproto.OP_PING:
                return self._send_ws(
                    wsproto.encode_frame(message.payload, wsproto.OP_PONG)
                )
            if message.opcode == wsproto.OP_PONG:
                return True
            if message.is_binary:
                server.on_frame(message.payload)
                return True
            if message.is_text:
                return self._handle_control(message)
            return True

        def _handle_control(self, message: wsproto.Message) -> bool:
            try:
                payload = json.loads(message.text())
            except (ValueError, UnicodeDecodeError):
                return True  # ignore junk rather than drop a working stream
            if not isinstance(payload, dict):
                return True
            kind = payload.get("type")
            if kind == "hello":
                if server.on_hello:
                    server.on_hello(payload)
                return self._send_ws(
                    wsproto.encode_frame(
                        json.dumps({"type": "ready"}).encode("utf-8"), wsproto.OP_TEXT
                    )
                )
            if kind == "bye":
                self._send_ws(wsproto.encode_close(1000, "bye"))
                return False
            return True

        def _send_ws(self, data: bytes) -> bool:
            try:
                self.connection.sendall(data)
            except OSError:
                return False
            return True

    return Handler
