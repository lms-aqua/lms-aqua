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
from .plate import (
    PlateCalibration,
    PlateError,
    PlateMapper,
    PlateState,
    ScanReport,
    scan_plate,
)
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
                 options: CaptureOptions,
                 mapper: PlateMapper | None = None,
                 on_plate: callable | None = None) -> None:
        self.writer = writer
        self.analyser = analyser
        self.aligner = aligner
        self.depth = depth
        self.options = options
        self.mapper = mapper
        # Called with each newly measured plate state, for the live dashboard.
        self.on_plate = on_plate
        self.stats = Stats()
        self.written = 0
        self.skipped = 0
        self.seen = 0
        self.label: str | None = None
        self.last_plate: PlateState | None = None
        self.plate_errors = 0
        self.plate_frames = 0
        self.plate_reused = 0
        self.last_plate_error = ""
        self._mapped_depth_timestamp: int | None = None
        self._mapped_plate: PlateState | None = None
        self._mapped_height_map = None
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

        # Plate mapping: what is on the plate, measured in millimetres. Only when
        # a calibration was supplied — it cannot be guessed from a single frame.
        plate: PlateState | None = None
        height_map = None
        if self.mapper is not None and depth_array is not None:
            # Depth arrives at roughly a third of the video rate, so the same
            # raster gets offered to several consecutive frames. Mapping it once
            # per depth frame rather than once per video frame is what keeps the
            # cost bounded, stops the dataset filling with identical height maps,
            # and makes the tracker's age_frames count observations rather than
            # video frames.
            is_new_depth = depth_timestamp != self._mapped_depth_timestamp
            if is_new_depth:
                self._mapped_depth_timestamp = depth_timestamp
                try:
                    self._mapped_plate = self.mapper.process(depth_array)
                    self._mapped_height_map = self.mapper.last_height_map
                    self.plate_frames += 1
                except PlateError as exc:
                    # A single bad depth frame must not end a multi-hour capture.
                    self.plate_errors += 1
                    self.last_plate_error = str(exc)
                    self._mapped_plate = None
                    self._mapped_height_map = None
            else:
                self.plate_reused += 1

            # The measurements are attached to every frame the depth aligns with —
            # they are the best available answer for that moment — but the raster
            # itself is written once, by the frame that triggered the mapping.
            plate = self._mapped_plate
            if plate is not None:
                self.last_plate = plate
            if is_new_depth:
                height_map = self._mapped_height_map
                # Notify once per measurement, not once per video frame, so the
                # dashboard updates at the depth rate rather than duplicating.
                if self.on_plate is not None and plate is not None:
                    try:
                        self.on_plate(plate, self._mapped_height_map)
                    except Exception:
                        # A dashboard client must never break a recording.
                        pass

        self.writer.write_frame(
            jpeg=jpeg,
            timestamp_ms=timestamp,
            metrics=metrics,
            samples=self.aligner.snapshot(timestamp),
            depth=depth_array,
            depth_timestamp_ms=depth_timestamp if depth_array is not None else None,
            depth_intrinsics=depth_intrinsics,
            label=self.label,
            plate=plate.as_dict() if plate is not None else None,
            height_map=height_map.to_u16_mm() if height_map is not None else None,
            height_map_cell_mm=(
                self.mapper.calibration.cell_mm if self.mapper is not None else None
            ),
        )
        self.written += 1
        self.stats.frames += 1
        return True


class CaptureSession:
    """Runs the three streams and writes the dataset."""

    def __init__(self, source: Source, writer: DatasetWriter,
                 analyser: FrameAnalyser, options: CaptureOptions,
                 want_depth: bool = True, want_data: bool = True,
                 channels: list[str] | None = None, hz: int | None = None,
                 mapper: PlateMapper | None = None,
                 on_plate: callable | None = None) -> None:
        self.source = source
        self.writer = writer
        self.analyser = analyser
        self.options = options
        self.want_depth = want_depth
        self.want_data = want_data
        self.channels = channels
        self.hz = hz
        self.mapper = mapper

        self.aligner = SampleAligner()
        self.depth_holder = DepthHolder()
        self.pipeline = _RecordingPipeline(writer, analyser, self.aligner,
                                          self.depth_holder, options, mapper,
                                          on_plate=on_plate)
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
        plate = self.pipeline.last_plate
        if plate is not None:
            parts.append(
                f"plate: {plate.object_count} object(s), "
                f"tallest {plate.tallest_mm:.0f}mm"
            )
        if self.pipeline.skipped:
            parts.append(f"{self.pipeline.skipped} skipped by --every")
        if self.pipeline.stats.decode_errors:
            parts.append(f"{self.pipeline.stats.decode_errors} decode errors")
        if self.pipeline.plate_errors:
            parts.append(f"{self.pipeline.plate_errors} plate errors")
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


def collect_depth_frames(source: Source, count: int = 20,
                         timeout: float = 30.0
                         ) -> tuple[list, tuple | None, str | None]:
    """Gather depth frames and their intrinsics, for the setup scan.

    Returns ``(frames, intrinsics, error)``. The error is surfaced rather than
    swallowed: a 401, a wrong port and a device without LiDAR all produce "no
    frames", and telling the user the wrong one of those wastes their afternoon.

    The intrinsics come from the stream's own headers rather than being guessed —
    they describe the depth raster, which is not the colour camera's.
    """
    frames: list = []
    intrinsics: list = [None]
    failure: list = [None]
    puller = DepthPuller(source.host, source.port, token=source.token)

    def on_frame(frame) -> None:
        frames.append(frame.millimetres)
        if frame.intrinsics is not None:
            intrinsics[0] = frame.intrinsics
        if len(frames) >= count:
            puller.stop()

    def run() -> None:
        try:
            puller.run_once(on_frame)
        except DepthError as exc:
            failure[0] = str(exc)
        except (ConnectionFailed, OSError) as exc:
            failure[0] = f"could not reach {source.host}:{source.port} ({exc})"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    puller.stop()
    thread.join(timeout=2.0)
    return frames, intrinsics[0], failure[0]


def run_plate_scan(source: Source, plate_width_mm: float,
                   plate_height_mm: float | None = None,
                   cell_mm: float | None = None, frames: int = 20,
                   timeout: float = 30.0) -> ScanReport:
    """The setup step: look at an empty plate and work out its geometry."""
    collected, intrinsics, failure = collect_depth_frames(source, frames, timeout)
    if not collected:
        if failure:
            # Report what actually went wrong, not an assumption about hardware.
            return ScanReport(None, 0, [failure])
        return ScanReport(None, 0, [
            "no depth frames arrived, and the connection reported no error. The "
            "sender must be a LiDAR device (iPhone/iPad Pro) with the depth "
            "channel switched on in the app."
        ])
    if intrinsics is None:
        return ScanReport(None, len(collected), [
            "the depth stream carried no X-LostCam-Intrinsics header, so the "
            "raster cannot be unprojected into millimetres"
        ])
    return scan_plate(collected, intrinsics, plate_width_mm, plate_height_mm,
                      cell_mm=cell_mm)


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
                 save_depth: bool, sender_info: dict | None,
                 plate: PlateCalibration | None = None,
                 save_height_maps: bool = True) -> DatasetConfig:
    return DatasetConfig(
        source=source.url,
        roi=analyser.roi,
        calibration=analyser.calibration,
        plate_reference_mm=analyser.plate_reference_mm,
        save_depth=save_depth,
        # Height maps are only meaningful with a plate calibration, so they are
        # off when there is not one rather than writing empty grids.
        save_height_maps=save_height_maps and plate is not None,
        notes=notes,
        sender_info=sender_info or {},
        plate_calibration=plate.as_dict() if plate is not None else None,
    )
