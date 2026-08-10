"""A software sender that behaves like the phone.

This exists for two reasons. It makes the entire pull-mode path testable in CI
with no device attached, and it gives a user a way to prove their virtual camera
works before blaming the phone: run ``lostcam mocksender`` on one terminal and
``lostcam pull 127.0.0.1`` on another. If the moving test pattern shows up in
Zoom, the desktop half is fine.

It implements the pull side of docs/PROTOCOL.md: /video, /audio, /info, / and
UDP discovery.
"""

from __future__ import annotations

import hmac
import json
import math
import socketserver
import struct
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler

import numpy as np

from .decode import rgb_to_jpeg
from .discovery import DISCOVERY_PORT, Responder
from .netutil import is_disconnect

BOUNDARY = "lostcamframe"
DEPTH_BOUNDARY = "lostcamdepth"

# The synthetic plate sits 400 mm from the camera, with a growing object on it.
MOCK_PLATE_MM = 400


MOCK_NOZZLE_MM = 55  # height above the plate of the mock hotend


def synth_depth(width: int, height: int, phase: float,
                nozzle: bool = True) -> np.ndarray:
    """A synthetic depth raster shaped like what a printer rig really sees.

    Three things, each of which the consumer has to handle correctly:

    * A **slowly growing print** in the middle. Growth is deliberately slow — a
      couple of millimetres per second — because that is the rate a real print
      grows at, and because the whole basis of rejecting machinery is that the
      machine moves orders of magnitude faster than the part. A mock whose print
      shot up 40 mm per second would be correctly classified as machinery, and
      would have made this a test of nothing.
    * A **moving nozzle**, taller than the print and somewhere different each
      frame, so the machinery filter is exercised rather than assumed.
    * A **border of zeros**, because real sensors return nothing at grazing
      angles and a consumer that treats 0 as a distance produces nonsense.
    """
    raster = np.full((height, width), MOCK_PLATE_MM, dtype=np.uint16)

    # The print: rises steadily and monotonically, capped so it stays plausible.
    growth = int(min(45, 5 + phase * 2.0))
    box_h, box_w = max(1, height // 3), max(1, width // 3)
    top, left = (height - box_h) // 2, (width - box_w) // 2
    raster[top : top + box_h, left : left + box_w] = MOCK_PLATE_MM - growth

    if nozzle:
        # A hotend sweeping across the bed: small, tall, and never in the same
        # place twice.
        nozzle_w = max(2, width // 10)
        nozzle_h = max(2, height // 10)
        span = max(1, width - nozzle_w - 2)
        nozzle_left = 1 + int((0.5 + 0.5 * math.sin(phase * 3.0)) * span)
        nozzle_top = max(1, top - nozzle_h - 1)
        raster[
            nozzle_top : nozzle_top + nozzle_h,
            nozzle_left : nozzle_left + nozzle_w,
        ] = MOCK_PLATE_MM - MOCK_NOZZLE_MM

    raster[0, :] = 0
    raster[-1, :] = 0
    raster[:, 0] = 0
    raster[:, -1] = 0
    return raster

MOCK_CHANNELS = (
    "attitude",
    "motion",
    "ar.world",
    "ar.face",
    "light",
    "battery",
)

# A handful of real ARKit blendshape names, so a consumer's mapping table is
# exercised rather than a made-up vocabulary.
MOCK_BLENDSHAPES = (
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "browInnerUp",
)


def synth_samples(phase: float, seq: int, t_ms: int) -> list[dict]:
    """Generate one round of plausible samples across the mock channels.

    Values move smoothly with ``phase`` so a consumer can tell a live stream
    from a frozen one, and the shapes match docs/SENSORS.md exactly — this is
    the reference the client tests assert against.
    """
    sin, cos = math.sin(phase), math.cos(phase)
    records: list[dict] = []

    def add(channel: str, **fields: object) -> None:
        nonlocal seq
        records.append({"t": t_ms, "seq": seq, "ch": channel, **fields})
        seq += 1

    # A quaternion rotating about y, normalised by construction.
    half = phase / 2.0
    add(
        "attitude",
        q=[0.0, round(math.sin(half), 6), 0.0, round(math.cos(half), 6)],
        euler=[0.0, round(math.degrees(phase) % 360.0 - 180.0, 3), 0.0],
        ref="magnetic",
        accuracy="high",
    )
    add(
        "motion",
        accel=[round(0.02 * sin, 5), round(0.02 * cos, 5), round(0.98 + 0.01 * sin, 5)],
        gravity=[0.0, 0.0, -1.0],
        rot=[round(0.01 * cos, 5), 0.0, round(-0.01 * sin, 5)],
        mag=[round(21.0 + sin, 3), round(-8.0 + cos, 3), 40.0],
        magAccuracy="high",
    )
    # Column-major 4x4 with the translation in elements 12..14.
    add(
        "ar.world",
        pose=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            round(0.5 * sin, 5), 1.4, round(-0.5 * cos, 5), 1.0,
        ],
        state="normal",
        features=800 + int(50 * sin),
        intrinsics=[1440.0, 1440.0, 960.0, 540.0],
        resolution=[1920, 1080],
    )
    # Only non-zero coefficients, as the spec requires.
    blend = {}
    for index, name in enumerate(MOCK_BLENDSHAPES):
        value = round(max(0.0, math.sin(phase + index)) * 0.9, 4)
        if value > 0.0:
            blend[name] = value
    add(
        "ar.face",
        tracked=True,
        blend=blend,
        transform=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            round(0.01 * sin, 5), -0.04, -0.38, 1.0,
        ],
        look=[round(0.03 * sin, 5), round(-0.02 * cos, 5), -1.0],
    )
    add("light", lumens=round(950.0 + 80.0 * sin, 2), kelvin=round(6100.0 + 200.0 * cos, 1))
    add("battery", level=0.82, charging=False, thermal="nominal")
    return records


def render_pattern(width: int, height: int, phase: float) -> np.ndarray:
    """A frame that is obviously moving, so a frozen stream is obvious too.

    Colour gradient background, a bar that sweeps horizontally, and a block
    whose brightness tracks the phase.
    """
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[:, :, 0] = grid_x
    frame[:, :, 1] = grid_y
    frame[:, :, 2] = 128 + 127 * math.sin(phase)

    bar_x = int((0.5 + 0.5 * math.sin(phase)) * max(0, width - 1))
    half = max(2, width // 40)
    left, right = max(0, bar_x - half), min(width, bar_x + half)
    frame[:, left:right, :] = 255

    box = max(8, min(width, height) // 8)
    level = 255 * (0.5 + 0.5 * math.cos(phase))
    frame[:box, :box, :] = level

    return frame.astype(np.uint8)


class MockSender:
    """Serves the pull-mode endpoints on a real TCP port."""

    def __init__(
        self,
        port: int = 4747,
        host: str = "127.0.0.1",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        quality: int = 80,
        token: str | None = None,
        device: str = "LostCam Mock Sender",
        discovery: bool = False,
        discovery_port: int = DISCOVERY_PORT,
        audio_rate: int = 44100,
        depth: bool = True,
        depth_size: tuple[int, int] = (64, 48),
        depth_fps: int = 10,
        nozzle: bool = True,
    ) -> None:
        self.depth = depth
        self.depth_size = depth_size
        self.depth_fps = depth_fps
        self.depth_served = 0
        # A moving hotend in the depth stream by default, because a mock without
        # one lets a consumer look correct while being unable to cope with the
        # single most common obstruction on a real printer.
        self.nozzle = nozzle
        self.port = port
        self.host = host
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.quality = quality
        self.token = token
        self.device = device
        self.audio_rate = audio_rate
        self.frames_served = 0
        self.samples_served = 0
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._responder: Responder | None = None
        self._want_discovery = discovery
        self._discovery_port = discovery_port

    # -- info ----------------------------------------------------------------

    def info(self) -> dict:
        return {
            "product": "LostCam",
            "protocol": 2,
            "device": self.device,
            "platform": "mock",
            "cameras": ["back", "front"],
            "video": {"width": self.width, "height": self.height, "fps": self.fps},
            "audio": {"rate": self.audio_rate, "channels": 1, "format": "s16le"},
            "channels": list(MOCK_CHANNELS),
            "capture": {"locks": "locked", "position": "back"},
            "depth": (
                {
                    "available": True,
                    "width": self.depth_size[0],
                    "height": self.depth_size[1],
                    "format": "u16mm",
                    "source": "mock",
                }
                if self.depth
                else {"available": False, "source": "none"}
            ),
        }

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        handler = _make_handler(self)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

            def handle_error(self, request, client_address) -> None:
                """Stay quiet about clients that simply hung up.

                A consumer disconnecting mid-stream is the normal way these
                streams end, and socketserver's default is to dump a traceback
                for it. Anything that is *not* a teardown still gets reported,
                because swallowing real errors would hide genuine faults.
                """
                if not is_disconnect(sys.exc_info()[1]):
                    super().handle_error(request, client_address)

        self._server = Server((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

        if self._want_discovery:
            payload = dict(self.info(), port=self.bound_port)
            self._responder = Responder(payload, port=self._discovery_port)
            try:
                self._responder.start()
            except OSError:
                self._responder = None  # port busy; discovery is optional

    @property
    def bound_port(self) -> int:
        if not self._server:
            return self.port
        return self._server.socket.getsockname()[1]

    @property
    def video_url(self) -> str:
        return f"http://{self.host}:{self.bound_port}/video"

    def stop(self) -> None:
        if self._responder:
            self._responder.stop()
            self._responder = None
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> MockSender:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def authorized(self, query: dict[str, list[str]], header: str | None) -> bool:
        if not self.token:
            return True
        supplied = header or (query.get("token") or [""])[0]
        return hmac.compare_digest(supplied or "", self.token)


def _make_handler(sender: MockSender) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LostCam-MockSender/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            route = parsed.path.rstrip("/") or "/"

            if not sender.authorized(query, self.headers.get("X-LostCam-Token")):
                self._send_json(401, {"error": "unauthorized"})
                return

            if route == "/info":
                self._send_json(200, sender.info())
            elif route == "/video":
                self._stream_video(query)
            elif route == "/audio":
                self._stream_audio()
            elif route == "/data":
                self._stream_data(query)
            elif route == "/depth":
                if not sender.depth:
                    self._send_json(404, {"error": "depth not available"})
                else:
                    self._stream_depth()
            elif route == "/":
                self._send_status()
            else:
                self._send_json(404, {"error": "not found"})

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_status(self) -> None:
            body = (
                f"<!doctype html><meta charset=utf-8><title>LostCam mock sender</title>"
                f"<h1>LostCam mock sender</h1>"
                f"<p>{sender.width}x{sender.height} @ {sender.fps} fps</p>"
                f'<p><a href="/video">/video</a> &middot; '
                f'<a href="/info">/info</a></p>'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_video(self, query: dict[str, list[str]]) -> None:
            width = _int_param(query, "w", sender.width)
            height = _int_param(query, "h", sender.height)
            fps = max(1, _int_param(query, "fps", sender.fps))
            quality = _int_param(query, "q", sender.quality)

            interval = 1.0 / fps
            next_at = time.monotonic()
            phase = 0.0
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    jpeg = rgb_to_jpeg(render_pattern(width, height, phase), quality)
                    header = (
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n"
                        f"X-LostCam-Timestamp: {int(time.monotonic() * 1000)}\r\n"
                        f"\r\n"
                    ).encode("ascii")
                    self.wfile.write(header + jpeg + b"\r\n")
                    sender.frames_served += 1
                    phase += 2 * math.pi / max(2, fps * 2)
                    next_at += interval
                    time.sleep(max(0.0, next_at - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # the client hung up, which is the normal ending

        def _stream_data(self, query: dict[str, list[str]]) -> None:
            requested = (query.get("ch") or [""])[0]
            wanted = {c.strip() for c in requested.split(",") if c.strip()}
            hz = max(1, min(240, _int_param(query, "hz", 30)))

            interval = 1.0 / hz
            next_at = time.monotonic()
            phase = 0.0
            seq = 1
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    t_ms = int(time.monotonic() * 1000)
                    records = synth_samples(phase, seq, t_ms)
                    # Filter first, then number: seq counts records actually
                    # sent, so a channel subset does not look like packet loss
                    # to a consumer reading gaps as drops (PROTOCOL.md §6).
                    kept = [
                        record
                        for record in records
                        if not wanted or record["ch"] in wanted
                    ]
                    for record in kept:
                        record["seq"] = seq
                        seq += 1
                    lines = [
                        json.dumps(record, separators=(",", ":")) + "\n"
                        for record in kept
                    ]
                    if lines:
                        self.wfile.write("".join(lines).encode("utf-8"))
                        sender.samples_served += len(lines)
                    phase += 2 * math.pi / max(2, hz * 2)
                    next_at += interval
                    time.sleep(max(0.0, next_at - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _stream_depth(self) -> None:
            width, height = sender.depth_size
            interval = 1.0 / max(1, sender.depth_fps)
            next_at = time.monotonic()
            phase = 0.0
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={DEPTH_BOUNDARY}",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    raster = synth_depth(width, height, phase,
                                         nozzle=sender.nozzle)
                    payload = raster.astype("<u2").tobytes()
                    header = (
                        f"--{DEPTH_BOUNDARY}\r\n"
                        f"Content-Type: application/octet-stream\r\n"
                        f"Content-Length: {len(payload)}\r\n"
                        f"X-LostCam-Timestamp: {int(time.monotonic() * 1000)}\r\n"
                        f"X-LostCam-Depth: {width}x{height}; format=u16mm\r\n"
                        f"X-LostCam-Intrinsics: 360.0,360.0,"
                        f"{width / 2:.1f},{height / 2:.1f}\r\n"
                        f"\r\n"
                    ).encode("ascii")
                    self.wfile.write(header + payload + b"\r\n")
                    sender.depth_served += 1
                    phase += 0.1
                    next_at += interval
                    time.sleep(max(0.0, next_at - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _stream_audio(self) -> None:
            rate = sender.audio_rate
            chunk_samples = rate // 20  # 50 ms
            step = 2 * math.pi * 440.0 / rate
            index = 0
            try:
                self.send_response(200)
                self.send_header("Content-Type", f"audio/L16; rate={rate}; channels=1")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    values = [
                        int(12000 * math.sin(step * (index + i)))
                        for i in range(chunk_samples)
                    ]
                    index += chunk_samples
                    self.wfile.write(struct.pack(f"<{len(values)}h", *values))
                    time.sleep(chunk_samples / rate)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def _int_param(query: dict[str, list[str]], key: str, default: int) -> int:
    raw = (query.get(key) or [None])[0]
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
