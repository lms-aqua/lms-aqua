"""End-to-end tests over real sockets.

These are the tests that prove the thing actually works: a real HTTP server
streaming real MJPEG over a real TCP connection into a real decode/transform
pipeline, and a real WebSocket client pushing frames the other way.

Nothing here needs a phone or a virtual camera — NullSink stands in for the
device, which is why this suite runs unchanged in CI.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time

import pytest

from lostcam import wsproto
from lostcam.bridge import Bridge, CallbackSink
from lostcam.datastream import DataPuller, DataStreamError
from lostcam.discovery import Responder, discover, parse_reply
from lostcam.mocksender import MockSender, render_pattern, synth_samples
from lostcam.pipeline import FramePipeline
from lostcam.puller import ConnectionFailed, Puller, Source, probe_info, wait_for_port
from lostcam.pushserver import PushServer
from lostcam.transform import Transform
from lostcam.virtualcam import NullSink
from lostcam.wsbroadcast import WSBroadcastServer

pytestmark = pytest.mark.slow

# CI runners are slow and shared; give the streams room without making a green
# run depend on wall-clock luck.
SETTLE = 6.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout: float = SETTLE, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def sender():
    server = MockSender(port=0, width=320, height=240, fps=30)
    server.start()
    assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
    yield server
    server.stop()


class TestPullModeEndToEnd:
    def test_frames_reach_the_sink(self, sender):
        sink = NullSink(320, 240)
        pipeline = FramePipeline(sink)
        puller = Puller(Source("127.0.0.1", sender.bound_port), pipeline)

        thread = threading.Thread(target=puller.run_once, daemon=True)
        thread.start()
        try:
            assert wait_until(lambda: sink.frames >= 5), (
                f"only {sink.frames} frames arrived"
            )
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        assert pipeline.stats.decode_errors == 0
        assert pipeline.stats.bytes_in > 0
        assert sink.last_frame is not None
        assert sink.last_frame.shape == (240, 320, 3)

    def test_frames_are_not_static(self, sender):
        """A frozen stream is the classic silent failure; prove motion."""
        frames = []
        sink = NullSink(320, 240)

        class Recording(FramePipeline):
            def submit(self, jpeg: bytes) -> bool:
                result = super().submit(jpeg)
                if result and len(frames) < 12:
                    frames.append(self.sink.last_frame.copy())
                return result

        puller = Puller(Source("127.0.0.1", sender.bound_port), Recording(sink))
        thread = threading.Thread(target=puller.run_once, daemon=True)
        thread.start()
        try:
            assert wait_until(lambda: len(frames) >= 8)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        assert not (frames[0] == frames[-1]).all(), "the stream never changed"

    def test_transform_and_scaling_applied_end_to_end(self, sender):
        sink = NullSink(480, 640)  # portrait sink from a landscape source
        pipeline = FramePipeline(sink, Transform(rotate=90), fit_mode="contain")
        puller = Puller(Source("127.0.0.1", sender.bound_port), pipeline)

        thread = threading.Thread(target=puller.run_once, daemon=True)
        thread.start()
        try:
            assert wait_until(lambda: sink.frames >= 3)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        assert sink.last_frame.shape == (640, 480, 3)

    def test_requested_size_is_honoured(self, sender):
        source = Source("127.0.0.1", sender.bound_port, width=160, height=120)
        sink = NullSink(160, 120)
        pipeline = FramePipeline(sink, fit_mode="stretch")
        puller = Puller(source, pipeline)

        thread = threading.Thread(target=puller.run_once, daemon=True)
        thread.start()
        try:
            assert wait_until(lambda: sink.frames >= 3)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)
        assert sink.last_frame.shape == (120, 160, 3)

    def test_info_endpoint_describes_the_sender(self, sender):
        info = probe_info(Source("127.0.0.1", sender.bound_port))
        assert info is not None
        assert info["product"] == "LostCam"
        assert info["protocol"] == 2
        assert info["video"] == {"width": 320, "height": 240, "fps": 30}
        assert "ar.face" in info["channels"]

    def test_connection_refused_is_reported_clearly(self):
        puller = Puller(Source("127.0.0.1", free_port()), FramePipeline(NullSink(8, 8)))
        with pytest.raises(ConnectionFailed):
            puller.run_once()

    def test_404_path_is_reported(self, sender):
        source = Source("127.0.0.1", sender.bound_port, path="/nope")
        puller = Puller(source, FramePipeline(NullSink(8, 8)))
        with pytest.raises(ConnectionFailed, match="404"):
            puller.run_once()


class TestTokenGating:
    def test_correct_token_is_accepted(self):
        server = MockSender(port=0, width=64, height=48, token="letmein")
        server.start()
        try:
            assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
            info = probe_info(
                Source("127.0.0.1", server.bound_port, token="letmein")
            )
            assert info is not None and info["product"] == "LostCam"
        finally:
            server.stop()

    def test_missing_token_is_rejected(self):
        server = MockSender(port=0, width=64, height=48, token="letmein")
        server.start()
        try:
            assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
            puller = Puller(
                Source("127.0.0.1", server.bound_port),
                FramePipeline(NullSink(64, 48)),
            )
            with pytest.raises(ConnectionFailed, match="401"):
                puller.run_once()
        finally:
            server.stop()

    def test_wrong_token_is_rejected(self):
        server = MockSender(port=0, width=64, height=48, token="letmein")
        server.start()
        try:
            assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
            assert probe_info(
                Source("127.0.0.1", server.bound_port, token="wrong")
            ) is None
        finally:
            server.stop()


class TestDataChannelEndToEnd:
    def test_samples_arrive_and_parse(self, sender):
        received = []
        puller = DataPuller("127.0.0.1", sender.bound_port, hz=60)
        thread = threading.Thread(
            target=lambda: puller.run_once(received.append), daemon=True
        )
        thread.start()
        try:
            assert wait_until(lambda: len(received) >= 12), (
                f"only {len(received)} samples arrived"
            )
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        channels = {s.channel for s in received}
        assert "attitude" in channels
        assert "ar.face" in channels
        assert puller.stats.bad_lines == 0
        assert puller.stats.dropped == 0, "seq numbers had gaps"

    def test_channel_filter_is_honoured(self, sender):
        received = []
        puller = DataPuller(
            "127.0.0.1", sender.bound_port, channels=["ar.face"], hz=60
        )
        thread = threading.Thread(
            target=lambda: puller.run_once(received.append), daemon=True
        )
        thread.start()
        try:
            assert wait_until(lambda: len(received) >= 5)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        assert {s.channel for s in received} == {"ar.face"}
        # A filtered subset must still be contiguously numbered, or the
        # consumer reads the gaps as dropped samples.
        assert puller.stats.dropped == 0, "channel filtering faked packet loss"
        seqs = [s.seq for s in received]
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))

    def test_blendshapes_have_expected_shape(self, sender):
        received = []
        puller = DataPuller("127.0.0.1", sender.bound_port, channels=["ar.face"])
        thread = threading.Thread(
            target=lambda: puller.run_once(received.append), daemon=True
        )
        thread.start()
        try:
            assert wait_until(lambda: len(received) >= 3)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        blend = received[-1].get("blend")
        assert isinstance(blend, dict) and blend
        for name, value in blend.items():
            assert isinstance(name, str)
            assert 0.0 <= value <= 1.0, f"{name}={value} outside 0..1"

    def test_pose_is_16_elements(self, sender):
        received = []
        puller = DataPuller("127.0.0.1", sender.bound_port, channels=["ar.world"])
        thread = threading.Thread(
            target=lambda: puller.run_once(received.append), daemon=True
        )
        thread.start()
        try:
            assert wait_until(lambda: len(received) >= 2)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)

        assert len(received[-1].get("pose")) == 16

    def test_missing_data_endpoint_explains_v1_sender(self):
        """A DroidCam-style sender has no /data; say so usefully."""
        server = MockSender(port=0, width=64, height=48)
        server.start()
        try:
            assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
            puller = DataPuller("127.0.0.1", server.bound_port, path="/nonexistent")
            with pytest.raises(DataStreamError, match="v1 sender|HTTP 404"):
                puller.run_once(lambda s: None)
        finally:
            server.stop()

    def test_bridge_receives_from_a_live_stream(self, sender):
        seen = []
        bridge = Bridge()
        bridge.add(CallbackSink(seen.append))
        puller = DataPuller("127.0.0.1", sender.bound_port, hz=60)

        thread = threading.Thread(
            target=lambda: puller.run_once(bridge.handle), daemon=True
        )
        thread.start()
        try:
            assert wait_until(lambda: len(seen) >= 10)
        finally:
            puller.stop()
            thread.join(timeout=SETTLE)
        assert bridge.samples >= 10
        bridge.close()


class TestCaptureEndToEnd:
    """The dataset path: three live streams into one aligned recording."""

    def test_records_frames_depth_and_telemetry_together(self, sender, tmp_path):
        from lostcam.capture import CaptureOptions, CaptureSession
        from lostcam.dataset import DatasetWriter, load_depth, read_manifest
        from lostcam.vision import ROI, FrameAnalyser

        analyser = FrameAnalyser(roi=ROI(0, 0, 160, 120))
        writer = DatasetWriter(tmp_path / "ds")
        # 45 frames at 30 fps is ~1.5s, long enough for the 10 fps depth
        # stream to connect and deliver — a 6-frame capture would finish first.
        options = CaptureOptions(warmup_frames=0, max_frames=45)
        session = CaptureSession(
            Source("127.0.0.1", sender.bound_port), writer, analyser, options,
            want_depth=True, want_data=True, hz=60,
        )
        session.start()
        try:
            assert wait_until(lambda: session.pipeline.written >= 45,
                              timeout=30.0), (
                f"only wrote {session.pipeline.written} frames"
            )
        finally:
            session.stop()
            summary = writer.finalise()

        assert summary["frames"] >= 45
        records = read_manifest(tmp_path / "ds")
        assert len(records) >= 45

        # Every frame must have its image on disk and metrics computed.
        for record in records:
            assert (tmp_path / "ds" / record["file"]).exists()
            assert "metrics" in record
            assert record["metrics"]["sharpness"] >= 0

        # Telemetry must have been aligned onto at least one frame.
        with_samples = [r for r in records if r.get("samples")]
        assert with_samples, "no sensor samples were attached to any frame"
        channels = set()
        for record in with_samples:
            channels.update(record["samples"])
        assert "attitude" in channels

        # Depth must have been attached and be reloadable.
        with_depth = [r for r in records if r.get("depth")]
        assert with_depth, "no depth frames were attached"
        raster = load_depth(tmp_path / "ds", with_depth[-1])
        assert raster is not None
        assert raster.shape == (48, 64)
        # The mock plate sits at 400 mm with a raised block and a zero border.
        assert raster.max() == 400
        assert (raster == 0).any(), "the invalid border should be preserved"

    def test_depth_metrics_measure_height_above_the_plate(self, sender, tmp_path):
        from lostcam.capture import CaptureOptions, CaptureSession
        from lostcam.dataset import DatasetWriter, read_manifest
        from lostcam.mocksender import MOCK_PLATE_MM
        from lostcam.vision import FrameAnalyser

        analyser = FrameAnalyser()
        # Tell the analyser where the empty plate is, as --calibrate-plate would.
        analyser.plate_reference_mm = float(MOCK_PLATE_MM)

        writer = DatasetWriter(tmp_path / "ds")
        session = CaptureSession(
            Source("127.0.0.1", sender.bound_port), writer, analyser,
            CaptureOptions(warmup_frames=0, max_frames=45),
            want_depth=True, want_data=False,
        )
        session.start()
        try:
            assert wait_until(lambda: session.pipeline.written >= 45, timeout=30.0)
        finally:
            session.stop()
            writer.finalise()

        heights = [
            record["metrics"]["height_max_mm"]
            for record in read_manifest(tmp_path / "ds")
            if record.get("metrics", {}).get("height_max_mm") is not None
        ]
        assert heights, "no heights were derived from the depth frames"
        # The synthetic print rises between 10 mm and 50 mm above the plate.
        assert all(0 <= height <= 60 for height in heights), heights
        assert max(heights) >= 5

    def test_every_n_subsamples_the_stream(self, sender, tmp_path):
        from lostcam.capture import CaptureOptions, CaptureSession
        from lostcam.dataset import DatasetWriter
        from lostcam.vision import FrameAnalyser

        writer = DatasetWriter(tmp_path / "ds")
        session = CaptureSession(
            Source("127.0.0.1", sender.bound_port), writer, FrameAnalyser(),
            CaptureOptions(every_n=5, warmup_frames=0, max_frames=3),
            want_depth=False, want_data=False,
        )
        session.start()
        try:
            assert wait_until(lambda: session.pipeline.written >= 3, timeout=20.0)
        finally:
            session.stop()
            writer.finalise()

        assert session.pipeline.skipped >= 8, "frames should have been skipped"

    def test_capture_survives_a_sender_with_no_depth(self, tmp_path):
        from lostcam.capture import CaptureOptions, CaptureSession
        from lostcam.dataset import DatasetWriter
        from lostcam.vision import FrameAnalyser

        server = MockSender(port=0, width=64, height=48, depth=False)
        server.start()
        try:
            assert wait_for_port("127.0.0.1", server.bound_port, timeout=SETTLE)
            writer = DatasetWriter(tmp_path / "ds")
            session = CaptureSession(
                Source("127.0.0.1", server.bound_port), writer, FrameAnalyser(),
                CaptureOptions(warmup_frames=0, max_frames=3),
                want_depth=True, want_data=False,
            )
            session.start()
            try:
                assert wait_until(lambda: session.pipeline.written >= 3, timeout=20.0)
            finally:
                session.stop()
                summary = writer.finalise()
            # Video keeps recording; only depth is absent.
            assert summary["frames"] >= 3
            assert summary["depth_frames"] == 0
        finally:
            server.stop()


class TestSynthSamples:
    """The mock sender is the reference implementation of the schema."""

    def test_every_record_has_required_fields(self):
        for record in synth_samples(0.5, 1, 1000):
            assert isinstance(record["t"], int)
            assert isinstance(record["seq"], int)
            assert isinstance(record["ch"], str)

    def test_seq_numbers_are_contiguous(self):
        records = synth_samples(0.0, 10, 0)
        assert [r["seq"] for r in records] == list(range(10, 10 + len(records)))

    def test_quaternion_is_normalised(self):
        (attitude,) = [r for r in synth_samples(1.0, 1, 0) if r["ch"] == "attitude"]
        q = attitude["q"]
        assert sum(component**2 for component in q) == pytest.approx(1.0, abs=1e-4)

    def test_values_change_with_phase(self):
        first = synth_samples(0.0, 1, 0)
        later = synth_samples(1.5, 1, 0)
        assert first != later

    def test_blendshapes_are_sparse_and_in_range(self):
        (face,) = [r for r in synth_samples(0.8, 1, 0) if r["ch"] == "ar.face"]
        assert all(0.0 < v <= 1.0 for v in face["blend"].values())

    def test_pattern_shape_and_motion(self):
        first = render_pattern(64, 48, 0.0)
        later = render_pattern(64, 48, 1.2)
        assert first.shape == (48, 64, 3)
        assert not (first == later).all()


class TestPushModeEndToEnd:
    """Push mode with TLS off — TLS is covered separately in test_tls.py."""

    def test_pushed_frames_reach_the_pipeline(self):
        from lostcam.decode import rgb_to_jpeg

        received: list[bytes] = []
        hellos: list[dict] = []
        port = free_port()
        server = PushServer(
            on_frame=received.append,
            on_hello=hellos.append,
            port=port,
            host="127.0.0.1",
            use_tls=False,
        )
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            client = _WSClient("127.0.0.1", port)
            client.connect()
            client.send_text(json.dumps({"type": "hello", "width": 64, "height": 48}))
            jpeg = rgb_to_jpeg(render_pattern(64, 48, 0.3))
            for _ in range(3):
                client.send_binary(jpeg)
            assert wait_until(lambda: len(received) >= 3)
            client.close()
        finally:
            server.stop()

        assert hellos and hellos[0]["width"] == 64
        assert all(frame.startswith(b"\xff\xd8") for frame in received)

    def test_healthz_needs_no_token(self):
        port = free_port()
        server = PushServer(
            on_frame=lambda f: None, port=port, host="127.0.0.1",
            token="secret", use_tls=False,
        )
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            import http.client

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=SETTLE)
            conn.request("GET", "/healthz")
            assert conn.getresponse().status == 200
            conn.close()
        finally:
            server.stop()

    def test_sender_page_requires_the_token(self):
        import http.client

        port = free_port()
        server = PushServer(
            on_frame=lambda f: None, port=port, host="127.0.0.1",
            token="secret", use_tls=False,
        )
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=SETTLE)
            conn.request("GET", "/")
            assert conn.getresponse().status == 401
            conn.close()

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=SETTLE)
            conn.request("GET", "/?token=secret")
            response = conn.getresponse()
            assert response.status == 200
            assert b"LostCam Sender" in response.read()
            conn.close()
        finally:
            server.stop()

    def test_sender_page_is_served(self):
        import http.client

        port = free_port()
        server = PushServer(on_frame=lambda f: None, port=port,
                            host="127.0.0.1", use_tls=False)
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=SETTLE)
            conn.request("GET", "/")
            response = conn.getresponse()
            body = response.read()
            assert response.status == 200
            # The page must warn about the secure-context requirement.
            assert b"getUserMedia" in body or b"camera access needs HTTPS" in body
            conn.close()
        finally:
            server.stop()


class TestWSBroadcast:
    def test_browser_client_receives_samples(self):
        from lostcam.datastream import Sample

        port = free_port()
        server = WSBroadcastServer(port=port)
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            client = _WSClient("127.0.0.1", port, path="/ws")
            client.connect()
            assert wait_until(lambda: server.client_count == 1)

            server.broadcast(Sample(1, 1, "motion", {"t": 1, "seq": 1, "ch": "motion"}))
            payload = client.read_text(timeout=SETTLE)
            assert json.loads(payload)["ch"] == "motion"
            client.close()
        finally:
            server.stop()

    def test_disconnected_client_is_dropped_not_fatal(self):
        from lostcam.datastream import Sample

        port = free_port()
        server = WSBroadcastServer(port=port)
        server.start()
        try:
            assert wait_for_port("127.0.0.1", port, timeout=SETTLE)
            client = _WSClient("127.0.0.1", port, path="/ws")
            client.connect()
            assert wait_until(lambda: server.client_count == 1)
            client.hard_close()

            # Broadcasting to a dead socket must not raise.
            for _ in range(30):
                server.broadcast(Sample(1, 1, "x", {"ch": "x"}))
            assert wait_until(lambda: server.client_count == 0)
        finally:
            server.stop()


class TestDiscovery:
    def test_responder_answers_a_probe(self):
        info = {"product": "LostCam", "port": 4747, "device": "Test", "platform": "mock"}
        port = free_port()
        responder = Responder(info, port=port, bind_host="127.0.0.1")
        responder.start()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(SETTLE)
            sock.sendto(b"LOSTCAM_DISCOVER_V1", ("127.0.0.1", port))
            payload, addr = sock.recvfrom(8192)
            sock.close()
            found = parse_reply(payload, addr[0])
            assert found is not None and found.device == "Test"
        finally:
            responder.stop()

    def test_responder_ignores_unrelated_traffic(self):
        info = {"product": "LostCam", "port": 4747}
        port = free_port()
        responder = Responder(info, port=port, bind_host="127.0.0.1")
        responder.start()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)
            sock.sendto(b"who are you", ("127.0.0.1", port))
            with pytest.raises(socket.timeout):
                sock.recvfrom(8192)
            sock.close()
        finally:
            responder.stop()

    def test_discover_never_raises_without_a_network(self):
        """Containers and guest networks refuse broadcast; that is not a crash."""
        assert isinstance(discover(timeout=0.3), list)


# -- a minimal WebSocket client, so the tests exercise the real handshake -----


class _WSClient:
    def __init__(self, host: str, port: int, path: str = "/ws") -> None:
        self.host = host
        self.port = port
        self.path = path
        self.sock: socket.socket | None = None
        # Reading server→client frames, which are unmasked by spec.
        self._decoder = wsproto.FrameDecoder(require_mask=False)
        self._pending: list[wsproto.Message] = []

    def connect(self, timeout: float = SETTLE) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))

        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError("server closed during the handshake")
            buffer += chunk
        head, _, rest = buffer.partition(b"\r\n\r\n")
        assert b"101" in head.split(b"\r\n")[0], head
        assert wsproto.accept_key(key).encode() in head, "bad Sec-WebSocket-Accept"
        if rest:
            self._pending += self._decoder.feed(rest)

    def _frame(self, payload: bytes, opcode: int) -> bytes:
        header = bytearray()
        header.append(0x80 | opcode)
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        return bytes(header) + wsproto.mask(payload)

    def send_binary(self, payload: bytes) -> None:
        assert self.sock
        self.sock.sendall(self._frame(payload, wsproto.OP_BINARY))

    def send_text(self, text: str) -> None:
        assert self.sock
        self.sock.sendall(self._frame(text.encode("utf-8"), wsproto.OP_TEXT))

    def read_text(self, timeout: float = SETTLE) -> str:
        assert self.sock
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in list(self._pending):
                self._pending.remove(message)
                if message.is_text and message.payload:
                    return message.text()
            self.sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                chunk = self.sock.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                break
            self._pending += self._decoder.feed(chunk)
        raise AssertionError("no text frame arrived in time")

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.sendall(
                    self._frame(struct.pack("!H", 1000), wsproto.OP_CLOSE)
                )
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    def hard_close(self) -> None:
        """Vanish without a close frame, like a phone leaving Wi-Fi."""
        if self.sock:
            try:
                self.sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            except OSError:
                pass
            self.sock.close()
            self.sock = None
