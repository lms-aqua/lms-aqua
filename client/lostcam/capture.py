"""The capture session: video + depth + telemetry into one aligned dataset.

Three streams run concurrently on their own threads and meet at the video frame:
each frame is written together with the freshest depth raster and the latest
value of every enabled data channel, all stamped with the sender's monotonic
clock so the alignment is verifiable after the fact rather than assumed.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .dataset import DatasetConfig, DatasetWriter, DepthHolder, SampleAligner
from .datastream import DataPuller, DataStreamError
from .decode import DecodeError, jpeg_to_rgb
from .depthstream import DepthError, DepthPuller
from .pipeline import Stats
from .puller import ConnectionFailed, Puller, Source
from .vision import ROI, Calibration, FrameAnalyser


@dataclass
class CaptureOptions:
    every_n: int = 1
    max_frames: int | None = None
    seconds: float | None = None
    warmup_frames: int = 0
    analyse: bool = True


class _RecordingPipeline:
    """Stands in for FramePipeline, writing to a dataset instead of a camera.

    Implements the same ``submit`` contract so the existing Puller drives it
    unchanged — the network and MJPEG parsing code has no idea a dataset is on
    the other end.
    """

    def __init__(self, writer: DatasetWriter, analyser: FrameAnalyser,
                 aligner: SampleAligner, depth: DepthHolder,
                 options: CaptureOptions) -> None:
        self.writer = writer
        self.analyser = analyser
        self.aligner = aligner
        self.depth = depth
        self.options = options
        self.stats = Stats()
        self.written = 0
        self.skipped = 0
        self.seen = 0
        self.label: str | None = None
        self._stop = threading.Event()

    @property
    def finished(self) -> bool:
        if self._stop.is_set():
            return True
        return bool(self.options.max_frames
                    and self.written >= self.options.max_frames)

    def stop(self) -> None:
        self._stop.set()

    def submit(self, jpeg: bytes) -> bool:
        self.seen += 1
        self.stats.bytes_in += len(jpeg)

        # Warmup exists because the first frames after a stream opens are the
        # worst ones: metering and, if unlocked, focus are still settling.
        if self.seen <= self.options.warmup_frames:
            return False
        if self.options.every_n > 1 and (self.seen % self.options.every_n) != 0:
            self.skipped += 1
            return False
        if self.finished:
            return False

        timestamp = int(time.monotonic() * 1000)
        frame: np.ndarray | None = None
        metrics = None

        depth_array, depth_timestamp, depth_intrinsics = self.depth.take(timestamp)

        if self.options.analyse:
            try:
                frame = jpeg_to_rgb(jpeg)
            except DecodeError:
                self.stats.decode_errors += 1
                return False
            metrics = self.analyser.analyse(frame, depth_array)

        self.writer.write_frame(
            jpeg=jpeg,
            timestamp_ms=timestamp,
            metrics=metrics,
            samples=self.aligner.snapshot(timestamp),
            depth=depth_array,
            depth_timestamp_ms=depth_timestamp if depth_array is not None else None,
            depth_intrinsics=depth_intrinsics,
            label=self.label,
        )
        self.written += 1
        self.stats.frames += 1
        return True


class CaptureSession:
    """Runs the three streams and writes the dataset."""

    def __init__(self, source: Source, writer: DatasetWriter,
                 analyser: FrameAnalyser, options: CaptureOptions,
                 want_depth: bool = True, want_data: bool = True,
                 channels: list[str] | None = None, hz: int | None = None) -> None:
        self.source = source
        self.writer = writer
        self.analyser = analyser
        self.options = options
        self.want_depth = want_depth
        self.want_data = want_data
        self.channels = channels
        self.hz = hz

        self.aligner = SampleAligner()
        self.depth_holder = DepthHolder()
        self.pipeline = _RecordingPipeline(writer, analyser, self.aligner,
                                          self.depth_holder, options)
        self.video = Puller(source, self.pipeline)
        self.data: DataPuller | None = None
        self.depth: DepthPuller | None = None
        self.warnings: list[str] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self.want_data:
            self.data = DataPuller(self.source.host, self.source.port,
                                   channels=self.channels, hz=self.hz,
                                   token=self.source.token)
            self._spawn("data", self._run_data)
        if self.want_depth:
            self.depth = DepthPuller(self.source.host, self.source.port,
                                    token=self.source.token)
            self._spawn("depth", self._run_depth)
        self._spawn("video", self._run_video)

    def _spawn(self, name: str, target) -> None:
        thread = threading.Thread(target=target, name=f"lostcam-{name}", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _run_video(self) -> None:
        # Reconnects: a capture that gives up on one Wi-Fi hiccup is useless for
        # a run measured in hours.
        self.video.run_forever(
            on_error=lambda exc: self._warn(f"video reconnecting: {exc}")
        )

    def _run_data(self) -> None:
        assert self.data
        try:
            self.data.run_forever(
                self.aligner.observe,
                on_error=lambda exc: self._warn(f"data reconnecting: {exc}"),
            )
        except DataStreamError as exc:
            self._warn(f"data channel unavailable: {exc}")

    def _run_depth(self) -> None:
        assert self.depth

        def on_frame(frame) -> None:
            self.depth_holder.observe(frame.millimetres, frame.timestamp_ms,
                                      frame.intrinsics)

        # Depth is genuinely optional — most devices have no LiDAR — so a missing
        # endpoint is a note, not a failure, and the recording continues.
        while not self._stop.is_set():
            try:
                self.depth.run_once(on_frame)
            except DepthError as exc:
                self._warn(f"depth unavailable: {exc}")
                return
            if self._stop.is_set():
                return
            self._stop.wait(1.0)

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        print(f"  note: {message}", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()
        self.pipeline.stop()
        self.video.stop()
        if self.data:
            self.data.stop()
        if self.depth:
            self.depth.stop()
        for thread in self._threads:
            thread.join(timeout=3.0)

    @property
    def finished(self) -> bool:
        return self.pipeline.finished

    def tag(self, kind: str, note: str = "") -> None:
        self.writer.write_event(kind, note, int(time.monotonic() * 1000))

    def summary_line(self) -> str:
        parts = [
            f"{self.pipeline.written} frames",
            f"{self.writer.depth_count} depth",
            f"{len(self.aligner.channels)} channels",
        ]
        if self.pipeline.skipped:
            parts.append(f"{self.pipeline.skipped} skipped by --every")
        if self.pipeline.stats.decode_errors:
            parts.append(f"{self.pipeline.stats.decode_errors} decode errors")
        return ", ".join(parts)


def calibrate_plate(source: Source, analyser: FrameAnalyser,
                    colour_shape: tuple[int, int],
                    frames: int = 10, timeout: float = 20.0) -> float | None:
    """Sample the empty plate's depth to establish a height reference.

    Returns None when depth is unavailable or too sparse to trust — a guessed
    reference would make every height in the dataset wrong by the same unknown
    amount, which is worse than having no heights at all.
    """
    collected: list[np.ndarray] = []
    puller = DepthPuller(source.host, source.port, token=source.token)

    def on_frame(frame) -> None:
        collected.append(frame.millimetres)
        if len(collected) >= frames:
            puller.stop()

    thread = threading.Thread(
        target=lambda: _quietly(puller.run_once, on_frame), daemon=True
    )
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    puller.stop()
    thread.join(timeout=2.0)

    if not collected:
        return None
    return analyser.calibrate_plate_reference(collected, colour_shape)


def _quietly(function, *args) -> None:
    try:
        function(*args)
    except (DepthError, ConnectionFailed, OSError):
        return


def build_analyser(roi_text: str | None, plate_mm: float | None,
                   plate_height_mm: float | None) -> FrameAnalyser:
    """Assemble the analyser from the CLI's calibration flags."""
    roi = ROI.parse(roi_text) if roi_text else None
    calibration = None
    if plate_mm:
        if roi is None:
            raise ValueError(
                "--plate-mm needs --roi as well: millimetres per pixel is only "
                "meaningful over a region of known size"
            )
        calibration = Calibration.from_plate(roi, plate_mm, plate_height_mm)
    return FrameAnalyser(roi=roi, calibration=calibration)


def build_config(source: Source, analyser: FrameAnalyser, notes: str,
                 save_depth: bool, sender_info: dict | None) -> DatasetConfig:
    return DatasetConfig(
        source=source.url,
        roi=analyser.roi,
        calibration=analyser.calibration,
        plate_reference_mm=analyser.plate_reference_mm,
        save_depth=save_depth,
        notes=notes,
        sender_info=sender_info or {},
    )
