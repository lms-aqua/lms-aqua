"""Recording aligned frames, depth and telemetry as a training dataset.

The output is deliberately boring and self-describing: numbered files plus a
JSONL manifest with one record per frame. No custom container, no index that can
drift out of sync with the files, nothing that needs this library to read back.
A `for line in open("manifest.jsonl")` loop is the whole reader.

Alignment is by the shared monotonic clock from PROTOCOL.md §6.3, which is the
reason that clock is specified the way it is: a frame and a sensor sample can be
matched after the fact, without either side having to agree on wall time.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datastream import Sample
from .vision import ROI, Calibration, FrameMetrics


class DatasetError(Exception):
    """The dataset could not be written."""


# MARK: - Sample alignment


class SampleAligner:
    """Holds the latest value of each channel, to attach to the next frame.

    Nearest-previous rather than interpolated: the channels are a mix of
    continuous quantities and discrete events, and interpolating a tracking state
    or a plane event would invent data. Every attached value is one the sender
    actually reported, and its age is recorded so a consumer can reject stale
    ones.
    """

    def __init__(self, max_age_ms: int = 1000) -> None:
        self.max_age_ms = max_age_ms
        self._latest: dict[str, Sample] = {}
        self._lock = threading.Lock()
        self.dropped_stale = 0

    def observe(self, sample: Sample) -> None:
        with self._lock:
            self._latest[sample.channel] = sample

    def snapshot(self, at_ms: int) -> dict[str, dict]:
        """Channel → sample body, for samples fresh enough to be relevant."""
        with self._lock:
            items = list(self._latest.items())
        out: dict[str, dict] = {}
        for channel, sample in items:
            age = at_ms - sample.t
            # A negative age means the sample is slightly ahead of the frame,
            # which is normal for concurrent streams and is not staleness.
            if age > self.max_age_ms:
                self.dropped_stale += 1
                continue
            body = {k: v for k, v in sample.raw.items() if k not in ("ch",)}
            body["age_ms"] = age
            out[channel] = body
        return out

    @property
    def channels(self) -> set[str]:
        with self._lock:
            return set(self._latest)


class DepthHolder:
    """Keeps the most recent depth frame for attachment to the next video frame."""

    def __init__(self, max_age_ms: int = 500) -> None:
        self.max_age_ms = max_age_ms
        self._frame: np.ndarray | None = None
        self._timestamp = 0
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._lock = threading.Lock()

    def observe(self, millimetres: np.ndarray, timestamp_ms: int,
                intrinsics: tuple[float, float, float, float] | None) -> None:
        with self._lock:
            self._frame = millimetres
            self._timestamp = timestamp_ms
            self._intrinsics = intrinsics

    def take(self, at_ms: int):
        """The freshest depth frame, or None if there is nothing recent."""
        with self._lock:
            if self._frame is None:
                return None, 0, None
            if at_ms - self._timestamp > self.max_age_ms:
                return None, 0, None
            return self._frame, self._timestamp, self._intrinsics


# MARK: - Dataset writing


@dataclass
class DatasetConfig:
    """Everything needed to reproduce how a recording was made."""

    source: str = ""
    roi: ROI | None = None
    calibration: Calibration | None = None
    plate_reference_mm: float | None = None
    save_depth: bool = True
    # Off unless a plate calibration was supplied: a height map is meaningless
    # without one, and defaulting to True created an empty directory for every
    # non-plate recording.
    save_height_maps: bool = False
    jpeg_passthrough: bool = True
    notes: str = ""
    sender_info: dict = field(default_factory=dict)
    plate_calibration: dict | None = None

    def as_dict(self) -> dict:
        document: dict = {
            "source": self.source,
            "save_depth": self.save_depth,
            "save_height_maps": self.save_height_maps,
            "jpeg_passthrough": self.jpeg_passthrough,
        }
        if self.plate_calibration:
            document["plate"] = self.plate_calibration
        if self.roi:
            document["roi"] = self.roi.as_dict()
        if self.calibration:
            document["calibration"] = self.calibration.as_dict()
        if self.plate_reference_mm is not None:
            document["plate_reference_mm"] = round(self.plate_reference_mm, 2)
        if self.notes:
            document["notes"] = self.notes
        if self.sender_info:
            document["sender"] = self.sender_info
        return document


class DatasetWriter:
    """Writes frames, depth and a manifest into a directory.

    Layout::

        <root>/
          dataset.json        how the recording was made
          manifest.jsonl      one record per frame
          frames/000001.jpg
          depth/000001.u16    raw u16 millimetres, width/height in the manifest
          events.jsonl        operator tags, if any

    The JPEG is written exactly as the phone sent it whenever possible, so the
    recorded bytes are the bytes that were transmitted — re-encoding would add a
    second generation of compression artefacts to the thing being measured.
    """

    MANIFEST = "manifest.jsonl"
    METADATA = "dataset.json"
    EVENTS = "events.jsonl"

    def __init__(self, root: str | Path, config: DatasetConfig | None = None,
                 overwrite: bool = False) -> None:
        self.root = Path(root)
        self.config = config or DatasetConfig()
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                raise DatasetError(
                    f"{self.root} already exists and is not empty. Pass "
                    f"--overwrite to replace it, or choose another directory — "
                    f"mixing two runs in one dataset silently corrupts it."
                )
            shutil.rmtree(self.root)

        self.frames_dir = self.root / "frames"
        self.depth_dir = self.root / "depth"
        self.height_maps_dir = self.root / "height_maps"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_height_maps:
            self.height_maps_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now(timezone.utc)
        self.frame_count = 0
        self.depth_count = 0
        self.height_map_count = 0
        self.event_count = 0
        self._first_timestamp: int | None = None
        self._last_timestamp: int | None = None
        self._lock = threading.Lock()

        self._manifest = (self.root / self.MANIFEST).open("w", encoding="utf-8")
        self._events = (self.root / self.EVENTS).open("w", encoding="utf-8")
        self._write_metadata()

    # -- writing ------------------------------------------------------------

    def write_frame(self, jpeg: bytes, timestamp_ms: int,
                    metrics: FrameMetrics | None = None,
                    samples: dict[str, dict] | None = None,
                    depth: np.ndarray | None = None,
                    depth_timestamp_ms: int | None = None,
                    depth_intrinsics: tuple[float, float, float, float] | None = None,
                    label: str | None = None,
                    plate: dict | None = None,
                    height_map: np.ndarray | None = None,
                    height_map_cell_mm: float | None = None) -> int:
        """Write one frame and its manifest record. Returns the frame index."""
        with self._lock:
            self.frame_count += 1
            index = self.frame_count
            if self._first_timestamp is None:
                self._first_timestamp = timestamp_ms
            self._last_timestamp = timestamp_ms

        name = f"{index:06d}"
        frame_path = self.frames_dir / f"{name}.jpg"
        frame_path.write_bytes(jpeg)

        record: dict = {
            "frame": index,
            "file": f"frames/{name}.jpg",
            "t": timestamp_ms,
            "bytes": len(jpeg),
        }
        if self._first_timestamp is not None:
            record["t_rel"] = timestamp_ms - self._first_timestamp
        if metrics is not None:
            record["metrics"] = metrics.as_dict()
        if label:
            record["label"] = label

        if depth is not None and self.config.save_depth:
            depth_path = self.depth_dir / f"{name}.u16"
            # Written little-endian regardless of host byte order, matching the
            # wire format, so a dataset copied between machines still reads.
            depth_path.write_bytes(depth.astype("<u2").tobytes())
            self.depth_count += 1
            record["depth"] = {
                "file": f"depth/{name}.u16",
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
                "format": "u16mm",
                "t": depth_timestamp_ms if depth_timestamp_ms is not None else 0,
            }
            if depth_intrinsics:
                record["depth"]["intrinsics"] = [round(v, 4)
                                                 for v in depth_intrinsics]
            if depth_timestamp_ms is not None:
                record["depth"]["skew_ms"] = timestamp_ms - depth_timestamp_ms

        if samples:
            record["samples"] = samples

        # The measured plate: object list plus whole-plate totals, in millimetres.
        if plate is not None:
            record["plate"] = plate

        if height_map is not None and self.config.save_height_maps:
            map_path = self.height_maps_dir / f"{name}.u16"
            map_path.write_bytes(height_map.astype("<u2").tobytes())
            self.height_map_count += 1
            record["height_map"] = {
                "file": f"height_maps/{name}.u16",
                "width": int(height_map.shape[1]),
                "height": int(height_map.shape[0]),
                # Offset by 1 so 0 can mean "no measurement" — see
                # HeightMap.to_u16_mm. A reader must subtract 1 from non-zero
                # cells to get millimetres.
                "format": "u16mm+1",
                "cell_mm": height_map_cell_mm,
            }

        self._manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Flushed per frame: a recording interrupted by a crash or a power cut is
        # exactly when the data matters, and a buffered manifest loses the tail.
        self._manifest.flush()
        return index

    def write_event(self, kind: str, note: str = "",
                    timestamp_ms: int | None = None) -> None:
        """Record an operator tag — "failure started", "purge done", and so on."""
        self.event_count += 1
        record = {
            "event": kind,
            "note": note,
            "frame": self.frame_count,
            "t": timestamp_ms if timestamp_ms is not None else 0,
            "wall": datetime.now(timezone.utc).isoformat(),
        }
        self._events.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._events.flush()

    # -- metadata -----------------------------------------------------------

    def _write_metadata(self) -> None:
        # Only describe the directories this recording actually produces, so the
        # metadata cannot promise a height_maps/ that does not exist.
        layout = {
            "manifest": self.MANIFEST,
            "events": self.EVENTS,
            "frames": "frames/NNNNNN.jpg",
        }
        if self.config.save_depth:
            layout["depth"] = (
                "depth/NNNNNN.u16 (row-major u16 little-endian millimetres, "
                "0 = no measurement)"
            )
        if self.config.save_height_maps:
            layout["height_maps"] = (
                "height_maps/NNNNNN.u16 (top-down orthographic grid in plate "
                "coordinates, u16 little-endian, 0 = no measurement, otherwise "
                "millimetres above the plate PLUS ONE)"
            )

        document = {
            "product": "LostCam",
            "dataset_version": 1,
            "protocol": 2,
            "started_at": self.started_at.isoformat(),
            "config": self.config.as_dict(),
            "layout": layout,
        }
        (self.root / self.METADATA).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

    def finalise(self) -> dict:
        """Close the files and stamp the summary into dataset.json."""
        try:
            self._manifest.flush()
            self._manifest.close()
        except OSError:
            pass
        try:
            self._events.flush()
            self._events.close()
        except OSError:
            pass

        duration_ms = 0
        if self._first_timestamp is not None and self._last_timestamp is not None:
            duration_ms = self._last_timestamp - self._first_timestamp

        summary = {
            "frames": self.frame_count,
            "depth_frames": self.depth_count,
            "height_maps": self.height_map_count,
            "events": self.event_count,
            "duration_ms": duration_ms,
            "average_fps": round(self.frame_count / (duration_ms / 1000.0), 3)
            if duration_ms > 0 else 0.0,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

        path = self.root / self.METADATA
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = {}
        document["summary"] = summary
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return summary

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finalise()


# MARK: - Reading back


def read_manifest(root: str | Path) -> list[dict]:
    """Read a manifest. Present so the format has a reference reader."""
    path = Path(root) / DatasetWriter.MANIFEST
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                # A truncated last line is what an interrupted recording looks
                # like; the rest of the dataset is still perfectly good.
                continue
    return records


def load_height_map(root: str | Path, record: dict) -> np.ndarray | None:
    """Load a frame's height map as float32 millimetres, NaN where unmeasured.

    Undoes the ``+1`` offset the u16 export uses to keep 0 meaning "absent".
    Returning NaN rather than 0 for absent cells matters: an occluded cell is not
    a flat one, and averaging zeros into a height would understate every object.
    """
    meta = record.get("height_map")
    if not isinstance(meta, dict):
        return None
    path = Path(root) / meta["file"]
    if not path.exists():
        return None
    raw = np.frombuffer(path.read_bytes(), dtype="<u2")
    grid = raw.reshape(int(meta["height"]), int(meta["width"]))
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    measured = grid > 0
    out[measured] = grid[measured].astype(np.float32) - 1.0
    return out


def load_depth(root: str | Path, record: dict) -> np.ndarray | None:
    """Load the depth raster for a manifest record."""
    depth = record.get("depth")
    if not isinstance(depth, dict):
        return None
    path = Path(root) / depth["file"]
    if not path.exists():
        return None
    raw = np.frombuffer(path.read_bytes(), dtype="<u2")
    return raw.reshape(int(depth["height"]), int(depth["width"]))
