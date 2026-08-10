"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import secrets
import signal
import sys
import threading
import time
from pathlib import Path

from . import DEFAULT_PULL_PORT, DEFAULT_PUSH_PORT, __version__
from .adb import AdbError, Forward, list_devices
from .audio import AudioError, AudioPuller, Speaker
from .bridge import Bridge, CSVSink, JSONLSink, OSCSink, UDPJSONSink
from .capture import (
    CaptureOptions,
    CaptureSession,
    build_analyser,
    build_config,
    calibrate_plate,
    run_plate_scan,
)
from .dataset import DatasetError, DatasetWriter
from .datastream import DataPuller, DataStreamError, Sample
from .decode import DecodeError, jpeg_to_rgb
from .depthstream import DepthError, DepthPuller
from .discovery import discover, local_addresses
from .pipeline import FramePipeline, Stats
from .plate import PlateCalibration, PlateError, PlateMapper
from .puller import ConnectionFailed, Puller, Source, probe_info, resolve_size
from .pushserver import PushServer
from .transform import VALID_FIT_MODES, Transform
from .virtualcam import VirtualCameraError, install_hint, open_sink
from .vision import CalibrationError
from .wsbroadcast import WSBroadcastServer

FALLBACK_SIZE = (1280, 720)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lostcam",
        description="Use your phone as a webcam. Feeds a virtual camera on "
        "Windows (OBS Virtual Camera) and Linux (v4l2loopback).",
    )
    parser.add_argument("--version", action="version", version=f"lostcam {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_output_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--width", type=int, help="virtual camera width")
        sub.add_argument("--height", type=int, help="virtual camera height")
        sub.add_argument("--fps", type=int, default=30, help="virtual camera fps")
        sub.add_argument(
            "--rotate", type=int, default=0, choices=[0, 90, 180, 270],
            help="rotate counter-clockwise by this many degrees",
        )
        sub.add_argument("--hflip", action="store_true", help="mirror horizontally")
        sub.add_argument("--vflip", action="store_true", help="mirror vertically")
        sub.add_argument(
            "--fit", default="contain", choices=list(VALID_FIT_MODES),
            help="how to reconcile frame size with camera size (default: contain)",
        )
        sub.add_argument(
            "--backend", help="pyvirtualcam backend (v4l2loopback, obs, unitycapture)"
        )
        sub.add_argument("--device", help="explicit device, e.g. /dev/video2")
        sub.add_argument(
            "--no-vcam", action="store_true",
            help="decode but do not open a virtual camera (for testing)",
        )
        sub.add_argument("--quiet", action="store_true", help="no periodic stats")

    pull = subparsers.add_parser(
        "pull", help="connect to a phone that is serving MJPEG (iOS app, DroidCam)"
    )
    pull.add_argument("host", nargs="?", help="phone IP; omit to auto-discover")
    pull.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
    pull.add_argument("--path", default="/video", help="stream path (default /video)")
    pull.add_argument("--token", help="shared secret, if the sender requires one")
    pull.add_argument("--quality", type=int, help="requested JPEG quality, 1-100")
    pull.add_argument("--camera", choices=["back", "front"], help="requested camera")
    pull.add_argument(
        "--usb", action="store_true",
        help="tunnel over USB with adb instead of Wi-Fi (Android)",
    )
    pull.add_argument("--serial", help="adb device serial, when several are attached")
    pull.add_argument(
        "--audio", action="store_true", help="also play the phone's microphone"
    )
    pull.add_argument("--audio-device", help="output device for --audio")
    pull.add_argument(
        "--once", action="store_true", help="do not reconnect when the stream ends"
    )
    add_output_flags(pull)

    serve = subparsers.add_parser(
        "serve", help="serve the browser sender page and accept pushed frames"
    )
    serve.add_argument("--port", type=int, default=DEFAULT_PUSH_PORT)
    serve.add_argument("--host", default="0.0.0.0", help="interface to bind")
    serve.add_argument(
        "--token", nargs="?", const="", help="require a token (generated if empty)"
    )
    serve.add_argument(
        "--no-tls", action="store_true",
        help="serve plain HTTP — the phone's camera will NOT work, localhost only",
    )
    add_output_flags(serve)

    mock = subparsers.add_parser(
        "mocksender", help="pretend to be a phone; verifies the desktop half alone"
    )
    mock.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--width", type=int, default=1280)
    mock.add_argument("--height", type=int, default=720)
    mock.add_argument("--fps", type=int, default=30)
    mock.add_argument("--token")
    mock.add_argument(
        "--discovery", action="store_true", help="also answer discovery probes"
    )

    def add_data_source_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("host", help="phone IP (or 127.0.0.1 with --usb)")
        sub.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
        sub.add_argument("--token", help="shared secret, if the sender requires one")
        sub.add_argument(
            "--channels",
            help="comma-separated subset, e.g. ar.face,attitude (default: all)",
        )
        sub.add_argument("--hz", type=int, help="per-channel rate cap")
        sub.add_argument(
            "--once", action="store_true", help="do not reconnect when the stream ends"
        )

    data = subparsers.add_parser(
        "data", help="print the phone's sensor/AR data channel as NDJSON"
    )
    add_data_source_flags(data)
    data.add_argument(
        "--pretty", action="store_true", help="indent each record (easier to read)"
    )
    data.add_argument(
        "--summary", action="store_true",
        help="print per-channel rates instead of the records themselves",
    )

    bridge = subparsers.add_parser(
        "bridge",
        help="re-emit the data channel to UDP / OSC / WebSocket / disk for other apps",
    )
    add_data_source_flags(bridge)
    bridge.add_argument(
        "--udp", nargs="?", const="127.0.0.1:9001", metavar="HOST:PORT",
        help="send each sample as a JSON datagram (default 127.0.0.1:9001)",
    )
    bridge.add_argument(
        "--osc", nargs="?", const="127.0.0.1:9000", metavar="HOST:PORT",
        help="send OSC bundles for TouchDesigner/Blender/Unity (default 127.0.0.1:9000)",
    )
    bridge.add_argument(
        "--ws", nargs="?", const="8765", metavar="PORT",
        help="serve a WebSocket browsers can subscribe to (default 8765)",
    )
    bridge.add_argument("--jsonl", metavar="FILE", help="append every sample to a file")
    bridge.add_argument("--csv", metavar="DIR", help="write one CSV per channel")

    record = subparsers.add_parser(
        "record", help="record the data channel to disk and stop after a while"
    )
    add_data_source_flags(record)
    record.add_argument("--out", default="lostcam-recording", help="output directory")
    record.add_argument(
        "--seconds", type=float, help="stop after this long (default: until Ctrl-C)"
    )

    scan = subparsers.add_parser(
        "scan",
        help="set up: look at an EMPTY build plate and save its geometry",
    )
    scan.add_argument("host", help="phone IP (or 127.0.0.1 with adb forward)")
    scan.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
    scan.add_argument("--token", help="shared secret, if the sender requires one")
    scan.add_argument(
        "--plate-mm", type=float, required=True, metavar="MM",
        help="plate width in millimetres (Ender 3: 220, Prusa MK4: 250)",
    )
    scan.add_argument(
        "--plate-depth-mm", type=float, metavar="MM",
        help="plate depth in millimetres, if it is not square",
    )
    scan.add_argument(
        "--cell-mm", type=float, default=None, metavar="MM",
        help="height-map resolution. Default: derived from the sensor's own "
             "sample spacing, which is what you want — a finer grid than the "
             "sensor can fill breaks objects apart and detects nothing",
    )
    scan.add_argument(
        "--frames", type=int, default=20,
        help="depth frames to average over (default 20)",
    )
    scan.add_argument("--out", default="plate.json", help="where to save the profile")
    scan.add_argument(
        "--force", action="store_true",
        help="save the profile even if the scan reported warnings",
    )

    plate = subparsers.add_parser(
        "plate",
        help="live view of what is on the plate, using a saved scan",
    )
    plate.add_argument("host", help="phone IP")
    plate.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
    plate.add_argument("--token", help="shared secret, if the sender requires one")
    plate.add_argument("--plate", default="plate.json", help="saved scan profile")
    plate.add_argument(
        "--threshold-mm", type=float, default=4.0,
        help="minimum height above the plate to count as an object (default 4)",
    )
    plate.add_argument(
        "--min-mm2", type=float, default=25.0,
        help="minimum footprint to count as an object (default 25mm²)",
    )
    plate.add_argument("--json", action="store_true", help="emit NDJSON, not a table")
    plate.add_argument("--seconds", type=float, help="stop after this long")

    capture = subparsers.add_parser(
        "capture",
        help="record an aligned dataset (frames + depth + telemetry) for training",
    )
    capture.add_argument("host", help="phone IP (or 127.0.0.1 with adb forward)")
    capture.add_argument("--port", type=int, default=DEFAULT_PULL_PORT)
    capture.add_argument("--token", help="shared secret, if the sender requires one")
    capture.add_argument("--out", default="lostcam-dataset", help="output directory")
    capture.add_argument(
        "--overwrite", action="store_true",
        help="replace the output directory if it already has data in it",
    )
    capture.add_argument(
        "--roi", metavar="X,Y,W,H",
        help="build-plate region in pixels; measurements are restricted to it",
    )
    capture.add_argument(
        "--plate-mm", type=float, metavar="MM",
        help="plate width in millimetres (with --roi) to calibrate mm-per-pixel",
    )
    capture.add_argument(
        "--plate-height-mm", type=float, metavar="MM",
        help="plate depth in millimetres, if it is not square",
    )
    capture.add_argument(
        "--every", type=int, default=1, metavar="N",
        help="keep every Nth frame (default 1). Use for long prints",
    )
    capture.add_argument("--frames", type=int, help="stop after this many frames")
    capture.add_argument("--seconds", type=float, help="stop after this long")
    capture.add_argument(
        "--warmup", type=int, default=5, metavar="N",
        help="discard the first N frames while metering settles (default 5)",
    )
    capture.add_argument(
        "--no-depth", action="store_true", help="do not record depth even if offered"
    )
    capture.add_argument(
        "--no-data", action="store_true", help="do not record the sensor channels"
    )
    capture.add_argument(
        "--no-metrics", action="store_true",
        help="skip per-frame analysis (no decode; faster, but no features)",
    )
    capture.add_argument(
        "--channels", help="comma-separated data channels to record (default: all)"
    )
    capture.add_argument("--hz", type=int, help="per-channel rate cap")
    capture.add_argument(
        "--calibrate-plate", action="store_true",
        help="measure the empty plate's depth first, to record heights in mm",
    )
    capture.add_argument("--label", help="label written on every frame record")
    capture.add_argument("--notes", default="", help="free text stored in dataset.json")
    capture.add_argument(
        "--plate", metavar="FILE",
        help="saved scan profile (from 'lostcam scan'). Enables plate mapping: "
             "per-frame object detection, heights, footprints and volumes",
    )
    capture.add_argument(
        "--threshold-mm", type=float, default=4.0,
        help="minimum height above the plate to count as an object (default 4)",
    )
    capture.add_argument(
        "--min-mm2", type=float, default=25.0,
        help="minimum object footprint in mm² (default 25)",
    )
    capture.add_argument(
        "--no-height-maps", action="store_true",
        help="measure objects but do not save the per-frame height-map rasters",
    )

    subparsers.add_parser("discover", help="find LostCam senders on the network")
    subparsers.add_parser("devices", help="list Android devices visible to adb")
    subparsers.add_parser(
        "doctor", help="check the virtual camera and audio setup on this machine"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "pull": cmd_pull,
        "serve": cmd_serve,
        "mocksender": cmd_mocksender,
        "data": cmd_data,
        "bridge": cmd_bridge,
        "record": cmd_record,
        "capture": cmd_capture,
        "scan": cmd_scan,
        "plate": cmd_plate,
        "discover": cmd_discover,
        "devices": cmd_devices,
        "doctor": cmd_doctor,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


# -- pull --------------------------------------------------------------------


def cmd_pull(args: argparse.Namespace) -> int:
    host, port = args.host, args.port

    if args.usb:
        try:
            with Forward(port, port, args.serial) as fwd:
                print(f"USB: forwarding 127.0.0.1:{port} to {fwd.serial} port {port}")
                return _run_pull(args, "127.0.0.1", port)
        except AdbError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if not host:
        print("No host given, looking for senders on the network…")
        found = discover()
        if not found:
            print(
                "error: no senders found. Give the phone's IP explicitly, e.g.\n"
                "  lostcam pull 192.168.1.42",
                file=sys.stderr,
            )
            return 2
        for entry in found:
            print(f"  found {entry.label}")
        host, port = found[0].host, found[0].port
        print(f"Using {host}:{port}")

    return _run_pull(args, host, port)


def _run_pull(args: argparse.Namespace, host: str, port: int) -> int:
    source = Source(
        host=host,
        port=port,
        path=args.path,
        token=args.token,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality,
        camera=args.camera,
    )

    info = probe_info(source)
    if info:
        print(f"Sender: {info.get('device', 'unknown')} ({info.get('platform', '?')})")
    size = resolve_size(source, info) or _probe_size_from_frame(source) or FALLBACK_SIZE

    try:
        sink = open_sink(
            size[0], size[1], args.fps,
            backend=args.backend, device=args.device, no_vcam=args.no_vcam,
        )
    except VirtualCameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    pipeline = FramePipeline(
        sink,
        Transform(args.rotate, args.hflip, args.vflip),
        args.fit,
    )
    puller = Puller(source, pipeline)

    print(f"Pulling {source.url} → {getattr(sink, 'device', 'null')} "
          f"at {size[0]}x{size[1]}")
    if args.no_vcam:
        print("(--no-vcam: frames are decoded and discarded)")

    audio_puller = _maybe_start_audio(args, host, port)
    reporter = _start_reporter(pipeline.stats, quiet=args.quiet)
    _install_stop_handler(puller)

    try:
        if args.once:
            try:
                puller.run_once()
            except ConnectionFailed as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 4
        else:
            puller.run_forever(
                on_error=lambda exc: print(f"disconnected: {exc}; retrying…",
                                           file=sys.stderr)
            )
    finally:
        reporter.set()
        if audio_puller:
            audio_puller.stop()
        sink.close()
        _final_report(pipeline.stats)
    return 0


def _probe_size_from_frame(source: Source) -> tuple[int, int] | None:
    """Fall back to decoding one frame when /info is unavailable.

    A DroidCam sender has no /info endpoint, so this is the path that makes
    LostCam work against it.
    """
    frames: list[bytes] = []

    class _Grab:
        stats = Stats()

        def submit(self, jpeg: bytes) -> bool:
            frames.append(jpeg)
            return True

    probe = Puller(source, _Grab())  # type: ignore[arg-type]

    def watch() -> None:
        deadline = time.monotonic() + 8.0
        while not frames and time.monotonic() < deadline:
            time.sleep(0.05)
        probe.stop()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        probe.run_once()
    except ConnectionFailed as exc:
        print(f"warning: could not probe frame size ({exc})", file=sys.stderr)
        return None
    finally:
        probe.stop()
        watcher.join(timeout=1.0)

    if not frames:
        return None
    try:
        frame = jpeg_to_rgb(frames[0])
    except DecodeError:
        return None
    return frame.shape[1], frame.shape[0]


def _maybe_start_audio(
    args: argparse.Namespace, host: str, port: int
) -> AudioPuller | None:
    if not getattr(args, "audio", False):
        return None
    if not Speaker.available():
        print(
            "warning: --audio needs sounddevice (pip install -e client[audio]); "
            "continuing without audio",
            file=sys.stderr,
        )
        return None

    puller = AudioPuller(host, port, token=args.token)
    speaker = Speaker(args.audio_device)

    def run() -> None:
        try:
            puller.run_once(lambda chunk, fmt: speaker.write(chunk, fmt))
        except AudioError as exc:
            print(f"audio stopped: {exc}", file=sys.stderr)
        finally:
            speaker.close()

    threading.Thread(target=run, daemon=True).start()
    print("Audio: streaming the phone's microphone to the chosen output device")
    return puller


# -- serve -------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    token = args.token
    if token == "":  # --token with no value
        token = secrets.token_urlsafe(9)

    # No size is chosen up front: the camera is opened from the sender's own
    # dimensions (hello, or failing that the first decoded frame), because a
    # virtual camera's resolution is fixed once opened.
    state: dict[str, object] = {"sink": None, "pipeline": None}
    lock = threading.Lock()
    transform = Transform(args.rotate, args.hflip, args.vflip)

    def ensure_pipeline(w: int, h: int) -> FramePipeline | None:
        """Open the camera once the first frame's real size is known.

        A virtual camera's resolution is fixed for its lifetime, so it cannot be
        opened until the phone has said what it is sending.
        """
        with lock:
            if state["pipeline"] is not None:
                return state["pipeline"]  # type: ignore[return-value]
            try:
                sink = open_sink(
                    w, h, args.fps,
                    backend=args.backend, device=args.device, no_vcam=args.no_vcam,
                )
            except VirtualCameraError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return None
            pipeline = FramePipeline(sink, transform, args.fit)
            state["sink"] = sink
            state["pipeline"] = pipeline
            print(f"Virtual camera open: {getattr(sink, 'device', 'null')} at {w}x{h}")
            return pipeline

    def on_hello(payload: dict) -> None:
        w = payload.get("width")
        h = payload.get("height")
        device = payload.get("device", "a phone")
        print(f"Sender connected: {str(device)[:60]} ({w}x{h})")
        if args.width and args.height:
            ensure_pipeline(args.width, args.height)
        elif isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            ensure_pipeline(w, h)

    def on_frame(jpeg: bytes) -> None:
        pipeline = state["pipeline"]
        if pipeline is None:
            # No hello, or it lacked dimensions: take them from the frame.
            try:
                frame = jpeg_to_rgb(jpeg)
            except DecodeError:
                return
            pipeline = ensure_pipeline(
                args.width or frame.shape[1], args.height or frame.shape[0]
            )
            if pipeline is None:
                return
        pipeline.submit(jpeg)  # type: ignore[union-attr]

    server = PushServer(
        on_frame=on_frame,
        on_hello=on_hello,
        port=args.port,
        host=args.host,
        token=token,
        use_tls=not args.no_tls,
        cert_hosts=local_addresses() + ["localhost", "127.0.0.1"],
    )

    try:
        server.start()
    except OSError as exc:
        print(f"error: could not bind {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 3

    _print_serve_banner(server, args)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.is_set():
            stop.wait(5.0)
            pipeline = state["pipeline"]
            if pipeline is not None and not args.quiet:
                stats = pipeline.stats  # type: ignore[union-attr]
                print(
                    f"  {stats.frames} frames, {stats.instant_fps():.1f} fps, "
                    f"{server.clients} sender(s)"
                )
    finally:
        server.stop()
        sink = state["sink"]
        if sink is not None:
            sink.close()  # type: ignore[union-attr]
        pipeline = state["pipeline"]
        if pipeline is not None:
            _final_report(pipeline.stats)  # type: ignore[union-attr]
    return 0


def _print_serve_banner(server: PushServer, args: argparse.Namespace) -> None:
    print(f"LostCam push server listening on port {server.bound_port}")
    addresses = local_addresses()
    if addresses:
        print("\nOpen one of these on your phone (same Wi-Fi):")
        for address in addresses:
            print(f"  {server.url_for(address)}")
    else:
        print(f"\nOpen: {server.url_for('<this-machine-ip>')}")

    if server.use_tls:
        print(
            "\nThe certificate is self-signed, so the phone will warn once —\n"
            "  Chrome: Advanced → Proceed;  Safari: Show Details → visit this website.\n"
            "This is required: browsers only expose the camera in a secure context."
        )
        if server.cert_path:
            from .tls import fingerprint

            try:
                print(f"  cert SHA-256: {fingerprint(server.cert_path)}")
            except Exception:
                pass
    else:
        print(
            "\nWARNING: --no-tls means the phone's browser will refuse camera\n"
            "access (getUserMedia needs a secure context). Useful only for\n"
            "testing the server from localhost."
        )
    if args.token:
        print("\nA token is required; it is already embedded in the URLs above.")
    print("\nPress Ctrl-C to stop.\n")


# -- data channel ------------------------------------------------------------


def _make_data_puller(args: argparse.Namespace) -> DataPuller:
    channels = (
        [c.strip() for c in args.channels.split(",") if c.strip()]
        if args.channels
        else None
    )
    return DataPuller(
        args.host, args.port, channels=channels, hz=args.hz, token=args.token
    )


def _run_data_puller(
    args: argparse.Namespace, puller: DataPuller, on_sample: callable
) -> int:
    stop = threading.Event()

    def handler(*_: object) -> None:
        print("\nStopping…", file=sys.stderr)
        puller.stop()
        stop.set()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - not on the main thread
        pass

    try:
        if args.once:
            try:
                puller.run_once(on_sample)
            except DataStreamError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 4
        else:
            puller.run_forever(
                on_sample,
                on_error=lambda exc: print(
                    f"data channel dropped: {exc}; retrying…", file=sys.stderr
                ),
            )
    finally:
        stats = puller.stats
        print(
            f"\nTotal: {stats.samples} samples, {stats.dropped} dropped "
            f"(from seq gaps), {stats.bad_lines} unparseable line(s).",
            file=sys.stderr,
        )
        if stats.per_channel:
            for channel, count in sorted(stats.per_channel.items()):
                print(f"  {channel}: {count}", file=sys.stderr)
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    puller = _make_data_puller(args)

    if args.summary:
        last = [time.monotonic()]

        def on_sample(sample: Sample) -> None:
            now = time.monotonic()
            if now - last[0] >= 1.0:
                last[0] = now
                counts = ", ".join(
                    f"{ch}={n}" for ch, n in sorted(puller.stats.per_channel.items())
                )
                print(f"{puller.stats.samples} samples  {counts}")
    else:
        indent = 2 if args.pretty else None

        def on_sample(sample: Sample) -> None:
            print(json.dumps(sample.raw, indent=indent), flush=True)

    print(
        f"Reading http://{args.host}:{args.port}{puller.request_path}",
        file=sys.stderr,
    )
    return _run_data_puller(args, puller, on_sample)


def cmd_bridge(args: argparse.Namespace) -> int:
    bridge = Bridge()
    ws_server: WSBroadcastServer | None = None

    if args.udp:
        host, port = _split_host_port(args.udp, 9001)
        bridge.add(UDPJSONSink(host, port))
        print(f"UDP JSON  → {host}:{port}")
    if args.osc:
        host, port = _split_host_port(args.osc, 9000)
        bridge.add(OSCSink(host, port))
        print(f"OSC       → {host}:{port}  (addresses like /lostcam/ar/face/blend/jawOpen)")
    if args.ws:
        try:
            ws_port = int(args.ws)
        except ValueError:
            print(f"error: --ws wants a port number, got {args.ws!r}", file=sys.stderr)
            return 2
        ws_server = WSBroadcastServer(ws_port)
        try:
            ws_server.start()
        except OSError as exc:
            print(f"error: could not bind WebSocket port {ws_port} ({exc})",
                  file=sys.stderr)
            return 3
        bridge.add(ws_server.as_sink())
        print(f"WebSocket → ws://127.0.0.1:{ws_server.bound_port}/")
    if args.jsonl:
        bridge.add(JSONLSink(args.jsonl))
        print(f"JSONL     → {args.jsonl}")
    if args.csv:
        bridge.add(CSVSink(args.csv))
        print(f"CSV       → {args.csv}/<channel>.csv")

    if not bridge.sinks:
        print(
            "error: a bridge with no outputs does nothing. Add at least one of\n"
            "  --udp [HOST:PORT]  --osc [HOST:PORT]  --ws [PORT]  --jsonl FILE  --csv DIR\n"
            "To just look at the stream, use: lostcam data <host>",
            file=sys.stderr,
        )
        return 2

    puller = _make_data_puller(args)
    print(f"\nSource: http://{args.host}:{args.port}{puller.request_path}")
    print("Press Ctrl-C to stop.\n")

    reporter_stop = threading.Event()

    def report() -> None:
        while not reporter_stop.wait(5.0):
            line = bridge.summary()
            if ws_server:
                line += f"; ws clients={ws_server.client_count}"
            print(f"  {line}")

    threading.Thread(target=report, daemon=True).start()

    try:
        return _run_data_puller(args, puller, bridge.handle)
    finally:
        reporter_stop.set()
        bridge.close()
        if ws_server:
            ws_server.stop()
        _warn_about_skipped_columns(bridge)


def cmd_record(args: argparse.Namespace) -> int:
    out = Path(args.out)
    bridge = Bridge()
    bridge.add(CSVSink(out))
    bridge.add(JSONLSink(out / "samples.jsonl"))
    puller = _make_data_puller(args)

    # Say where the data is going. Telemetry recorded silently is a bug.
    print(f"Recording to {out.resolve()}")
    print(f"  {out}/<channel>.csv  and  {out}/samples.jsonl")
    print(f"Source: http://{args.host}:{args.port}{puller.request_path}")
    timer: threading.Timer | None = None
    if args.seconds:
        print(f"Stopping automatically after {args.seconds:g}s.")
        timer = threading.Timer(args.seconds, puller.stop)
        # Daemon and cancelled on exit: otherwise a recording that ends early
        # keeps the process alive until the timer fires.
        timer.daemon = True
        timer.start()
    print()

    try:
        return _run_data_puller(args, puller, bridge.handle)
    finally:
        if timer is not None:
            timer.cancel()
        bridge.close()
        _warn_about_skipped_columns(bridge)
        print(f"Recording written to {out.resolve()}")


def _warn_about_skipped_columns(bridge: Bridge) -> None:
    """Report late columns, and be precise that the values were kept.

    Sparse channels (blendshapes send only non-zero coefficients) routinely
    introduce keys after the header is fixed. The values go to the extra_json
    column rather than being dropped, so this is a layout note — but it has to
    say so accurately, not point at a JSONL file that may not exist.
    """
    for sink in bridge.sinks:
        late = getattr(sink, "late_columns", None)
        if not late:
            continue
        extra_column = getattr(sink, "EXTRA_COLUMN", "extra_json")
        for channel, columns in sorted(late.items()):
            safe = channel.replace(".", "_")
            print(
                f"note: {safe}.csv saw column(s) after its header was written: "
                f"{', '.join(sorted(columns))}. Their values are preserved in "
                f"the '{extra_column}' column of the affected rows.",
                file=sys.stderr,
            )


def _split_host_port(value: str, default_port: int) -> tuple[str, int]:
    if ":" in value:
        host, _, raw = value.rpartition(":")
        try:
            return (host or "127.0.0.1"), int(raw)
        except ValueError:
            return value, default_port
    return value, default_port


# -- plate scanning and mapping ----------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    source = Source(host=args.host, port=args.port, token=args.token)

    info = probe_info(source)
    if info and not _sender_offers_depth(info):
        print(
            "error: this sender is not offering depth.\n"
            "  Plate mapping needs a LiDAR device (iPhone Pro / iPad Pro) with the\n"
            "  depth channel switched on in the app.",
            file=sys.stderr,
        )
        return 2

    print("Point the camera at the build plate and CLEAR IT COMPLETELY.")
    print("The scan measures the empty plate, so anything left on it becomes part")
    print("of the plate and every later height will be wrong.\n")
    print(f"Scanning {args.frames} depth frames from {source.host}:{source.port}…")

    report = run_plate_scan(
        source, args.plate_mm, args.plate_depth_mm,
        cell_mm=args.cell_mm, frames=args.frames,
    )

    if not report.ok:
        print("\nScan failed:", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 4

    print("\n" + report.summary())

    if report.warnings and not args.force:
        print(
            "\nThe scan produced warnings. Fix them and rescan, or pass --force to\n"
            "save it anyway — the measurements will still be produced, but their\n"
            "accuracy will be limited by whatever the warnings describe.",
            file=sys.stderr,
        )
        return 5

    assert report.calibration is not None
    path = report.calibration.save(args.out)
    print(f"\nSaved plate profile to {path}")
    print(f"Now record with it:\n  lostcam capture {args.host} --plate {path} "
          f"--out runs/my-print")
    return 0


def cmd_plate(args: argparse.Namespace) -> int:
    """Live plate readout — the fastest way to check a scan is any good."""
    try:
        calibration = PlateCalibration.load(args.plate)
    except PlateError as exc:
        print(f"error: {exc}\n\nRun 'lostcam scan <host> --plate-mm 220' first.",
              file=sys.stderr)
        return 2

    source = Source(host=args.host, port=args.port, token=args.token)
    mapper = PlateMapper(calibration, threshold_mm=args.threshold_mm,
                         min_footprint_mm2=args.min_mm2)
    puller = DepthPuller(source.host, source.port, token=source.token)

    if not args.json:
        print(f"Plate: {calibration.plate_width_mm:.0f} x "
              f"{calibration.plate_height_mm:.0f} mm, "
              f"{calibration.plane.tilt_degrees:.0f}° off head-on, "
              f"grid {calibration.cell_mm:g} mm")
        print("Press Ctrl-C to stop.\n")

    state = {"last": 0.0}

    def on_frame(frame) -> None:
        try:
            result = mapper.process(frame.millimetres)
        except PlateError as exc:
            print(f"  plate error: {exc}", file=sys.stderr)
            return

        if args.json:
            document = dict(result.as_dict(), t=frame.timestamp_ms)
            print(json.dumps(document, separators=(",", ":")), flush=True)
            return

        # Throttle the table so a 10 Hz depth stream does not scroll away.
        now = time.monotonic()
        if now - state["last"] < 1.0:
            return
        state["last"] = now
        _print_plate_table(result, mapper)

    stop = threading.Event()

    def handler(*_: object) -> None:
        puller.stop()
        stop.set()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - not on the main thread
        pass

    timer: threading.Timer | None = None
    if args.seconds:
        timer = threading.Timer(args.seconds, handler)
        # Daemon, so a stream that dies early cannot hold the process open until
        # the timer fires — which for --seconds 3600 would be a one-hour hang.
        timer.daemon = True
        timer.start()

    try:
        puller.run_once(on_frame)
    except DepthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    finally:
        if timer is not None:
            timer.cancel()
    return 0


def _print_plate_table(result, mapper: PlateMapper) -> None:
    coverage = result.coverage * 100
    print(
        f"[{result.object_count} object(s)]  tallest {result.tallest_mm:6.1f} mm  "
        f"occupied {result.occupied_mm2:8.0f} mm²  "
        f"volume {result.total_volume_mm3 / 1000:8.1f} cm³  "
        f"map {coverage:4.0f}%"
    )
    if not result.objects:
        # An empty plate is the expected state half the time, and saying so beats
        # printing nothing and looking hung.
        print("    plate is clear")
        return
    print(f"    {'id':>4} {'x,y mm':>14} {'size mm':>13} {'h mm':>7} "
          f"{'area mm²':>9} {'vol cm³':>8} {'solid':>6}")
    for item in result.objects[:8]:
        label = item.track_id if item.track_id is not None else item.object_id
        print(
            f"    {label:>4} "
            f"{item.centre_u_mm:6.0f},{item.centre_v_mm:<7.0f} "
            f"{item.bbox_u_mm:5.0f}x{item.bbox_v_mm:<7.0f} "
            f"{item.height_max_mm:7.1f} {item.footprint_mm2:9.0f} "
            f"{item.volume_mm3 / 1000:8.1f} {item.solidity:6.2f}"
        )


# -- dataset capture ---------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> int:
    source = Source(host=args.host, port=args.port, token=args.token)

    try:
        analyser = build_analyser(args.roi, args.plate_mm, args.plate_height_mm)
    except (ValueError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    info = probe_info(source)
    if info:
        print(f"Sender: {info.get('device', 'unknown')} "
              f"({info.get('platform', '?')}, protocol {info.get('protocol', '?')})")
        _report_capture_hygiene(info, args)
    else:
        print("warning: no /info from the sender; recording anyway",
              file=sys.stderr)

    size = resolve_size(source, info) or FALLBACK_SIZE
    want_depth = not args.no_depth and _sender_offers_depth(info)

    if args.calibrate_plate:
        if not want_depth:
            print("error: --calibrate-plate needs the depth stream, which this "
                  "sender is not offering", file=sys.stderr)
            return 2
        print("Measuring the empty plate — keep it clear…")
        reference = calibrate_plate(source, analyser, (size[1], size[0]))
        if reference is None:
            print(
                "warning: could not establish a plate reference (not enough valid "
                "depth). Heights will be omitted; distances are still recorded.",
                file=sys.stderr,
            )
        else:
            print(f"Plate reference: {reference:.0f} mm from the camera")

    channels = (
        [c.strip() for c in args.channels.split(",") if c.strip()]
        if args.channels else None
    )

    # Plate mapping, when a scan profile was supplied.
    plate_calibration = None
    mapper = None
    if args.plate:
        try:
            plate_calibration = PlateCalibration.load(args.plate)
        except PlateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not want_depth:
            print(
                "error: --plate needs the depth stream, which this sender is not\n"
                "  offering. Enable the depth channel in the app, or drop --plate.",
                file=sys.stderr,
            )
            return 2
        mapper = PlateMapper(plate_calibration, threshold_mm=args.threshold_mm,
                             min_footprint_mm2=args.min_mm2)
        print(f"Plate mapping: {plate_calibration.plate_width_mm:.0f}x"
              f"{plate_calibration.plate_height_mm:.0f} mm at "
              f"{plate_calibration.cell_mm:g} mm/cell, "
              f"objects over {args.threshold_mm:g} mm tall")

    config = build_config(
        source, analyser, args.notes, want_depth, info or {},
        plate=plate_calibration,
        save_height_maps=not args.no_height_maps,
    )
    try:
        writer = DatasetWriter(args.out, config, overwrite=args.overwrite)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    options = CaptureOptions(
        every_n=max(1, args.every),
        max_frames=args.frames,
        seconds=args.seconds,
        warmup_frames=max(0, args.warmup),
        analyse=not args.no_metrics,
    )
    session = CaptureSession(
        source, writer, analyser, options,
        want_depth=want_depth, want_data=not args.no_data,
        channels=channels, hz=args.hz, mapper=mapper,
    )
    session.pipeline.label = args.label

    print(f"\nRecording to {Path(args.out).resolve()}")
    print(f"  video  {source.url}")
    print(f"  depth  {'yes' if want_depth else 'no'}")
    print(f"  data   {'no' if args.no_data else (args.channels or 'all channels')}")
    if analyser.roi:
        print(f"  ROI    {analyser.roi.as_dict()}")
    if analyser.calibration:
        print(f"  scale  {analyser.calibration.mm_per_pixel_x:.4f} mm/px")
    if options.every_n > 1:
        print(f"  keeping every {options.every_n}th frame")
    print("\nPress Ctrl-C to stop.\n")

    stop = threading.Event()

    def handler(*_: object) -> None:
        print("\nStopping…", file=sys.stderr)
        stop.set()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - not on the main thread
        pass

    session.start()
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while not stop.is_set():
            stop.wait(5.0)
            if session.finished:
                break
            if deadline and time.monotonic() >= deadline:
                break
            print(f"  {session.summary_line()}")
    finally:
        # No return from this block: a return here would swallow whatever
        # exception brought us in, and losing that would hide the real fault.
        session.stop()
        summary = writer.finalise()
        print(f"\nWrote {summary['frames']} frames "
              f"({summary['depth_frames']} with depth) over "
              f"{summary['duration_ms'] / 1000:.1f}s "
              f"at {summary['average_fps']:.1f} fps.")
        print(f"Dataset: {Path(args.out).resolve()}")
        print(f"  manifest: {args.out}/manifest.jsonl (one JSON record per frame)")
        if summary["frames"] == 0:
            print(
                "\nNo frames were recorded. Check that the sender is streaming and "
                "that the address is right — 'lostcam pull <host> --no-vcam' is the "
                "quickest way to confirm the video path works.",
                file=sys.stderr,
            )

    return 0 if summary["frames"] else 4


def _sender_offers_depth(info: dict | None) -> bool:
    if not info:
        return False
    depth = info.get("depth")
    return isinstance(depth, dict) and bool(depth.get("available"))


def _report_capture_hygiene(info: dict, args: argparse.Namespace) -> None:
    """Warn about the settings that quietly ruin a vision dataset.

    Auto exposure and auto focus drifting over a long run is the single most
    common way a capture ends up unusable, and it is invisible until training.
    """
    capture = info.get("capture")
    if isinstance(capture, dict) and capture.get("locks") == "auto":
        print(
            "warning: the sender reports auto exposure/white balance/focus.\n"
            "         Over a long print the same scene will drift in brightness "
            "and sharpness,\n"
            "         and a model then learns the camera's reaction instead of "
            "the print.\n"
            "         In the app: 'Focus on centre and lock' before recording.",
            file=sys.stderr,
        )
    if not args.roi:
        print(
            "note: no --roi given, so metrics cover the whole frame. Restricting "
            "them to the\n"
            "      build plate makes them far more useful — most of the frame is "
            "the room.",
            file=sys.stderr,
        )


# -- other commands ----------------------------------------------------------


def cmd_mocksender(args: argparse.Namespace) -> int:
    from .mocksender import MockSender

    sender = MockSender(
        port=args.port, host=args.host, width=args.width, height=args.height,
        fps=args.fps, token=args.token, discovery=args.discovery,
    )
    try:
        sender.start()
    except OSError as exc:
        print(f"error: could not bind {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 3
    print(f"Mock sender on http://{args.host}:{sender.bound_port}/")
    print(f"  video: {sender.video_url}")
    print(f"  test with: lostcam pull {args.host} --port {sender.bound_port}")
    print("\nPress Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(5.0)
            print(f"  served {sender.frames_served} frames")
    except KeyboardInterrupt:
        pass
    finally:
        sender.stop()
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    print("Broadcasting for LostCam senders…")
    found = discover()
    if not found:
        print(
            "No senders replied. Check both devices are on the same Wi-Fi, that\n"
            "the sender app is streaming, and that the network is not blocking\n"
            "broadcast traffic (many guest networks do). You can always connect\n"
            "directly: lostcam pull <phone-ip>"
        )
        return 1
    for entry in found:
        print(f"  {entry.label}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    try:
        devices = list_devices()
    except AdbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not devices:
        print("No devices attached. Connect by USB and enable USB debugging.")
        return 1
    for device in devices:
        note = "" if device.usable else "  <- unlock the phone / accept the prompt"
        print(f"  {device.serial}  {device.state}{note}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"lostcam {__version__} on {sys.platform}, Python {sys.version.split()[0]}")

    print("\nVirtual camera:")
    try:
        import pyvirtualcam  # noqa: F401

        print("  pyvirtualcam: installed")
    except ImportError:
        print("  pyvirtualcam: MISSING — pip install -e client[vcam]")

    try:
        with open_sink(640, 480, 30) as sink:
            print(f"  opened OK: {getattr(sink, 'device', '?')}")
    except VirtualCameraError as exc:
        print(f"  could not open: {exc}")
        print("\n" + install_hint())

    print("\nAudio:")
    if Speaker.available():
        try:
            devices = Speaker.list_devices()
            print(f"  sounddevice: installed, {len(devices)} output device(s)")
            for line in devices[:12]:
                print(f"    {line}")
        except AudioError as exc:
            print(f"  sounddevice error: {exc}")
    else:
        print("  sounddevice: not installed — pip install -e client[audio]")

    print("\nUSB (adb):")
    try:
        devices = list_devices()
        print(f"  adb: available, {len(devices)} device(s)")
        for device in devices:
            print(f"    {device.serial}  {device.state}")
    except AdbError as exc:
        print(f"  {exc}".replace("\n", "\n  "))

    print("\nNetwork addresses:", ", ".join(local_addresses()) or "none detected")
    return 0


# -- shared helpers ----------------------------------------------------------


def _start_reporter(stats: Stats, quiet: bool = False, interval: float = 5.0):
    stop = threading.Event()
    if quiet:
        return stop

    def run() -> None:
        while not stop.wait(interval):
            print(
                f"  {stats.frames} frames, {stats.instant_fps():.1f} fps, "
                f"{stats.bytes_in / 1e6:.1f} MB in, "
                f"{stats.decode_errors} decode error(s)"
            )

    threading.Thread(target=run, daemon=True).start()
    return stop


def _final_report(stats: Stats) -> None:
    print(
        f"\nTotal: {stats.frames} frames in {stats.elapsed:.1f}s "
        f"({stats.average_fps:.1f} fps average), "
        f"{stats.decode_errors} decode error(s)."
    )


def _install_stop_handler(puller: Puller) -> None:
    def handler(*_: object) -> None:
        print("\nStopping…", file=sys.stderr)
        puller.stop()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - not on the main thread
        pass
