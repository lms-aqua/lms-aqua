"""Depth parsing, sample alignment and dataset writing tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lostcam.dataset import (
    DatasetConfig,
    DatasetError,
    DatasetWriter,
    DepthHolder,
    SampleAligner,
    load_depth,
    read_manifest,
)
from lostcam.datastream import Sample
from lostcam.depthstream import (
    DepthError,
    DepthFrame,
    DepthParser,
    decode_depth,
    parse_depth_geometry,
    parse_intrinsics,
)
from lostcam.vision import ROI, Calibration, FrameMetrics

BOUNDARY = "lostcamdepth"


def depth_part(raster: np.ndarray, timestamp: int = 1000,
               intrinsics: str = "360.0,360.0,32.0,24.0") -> bytes:
    payload = raster.astype("<u2").tobytes()
    header = (
        f"--{BOUNDARY}\r\n"
        f"Content-Type: application/octet-stream\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"X-LostCam-Timestamp: {timestamp}\r\n"
        f"X-LostCam-Depth: {raster.shape[1]}x{raster.shape[0]}; format=u16mm\r\n"
        f"X-LostCam-Intrinsics: {intrinsics}\r\n\r\n"
    ).encode("ascii")
    return header + payload + b"\r\n"


class TestDepthHeaders:
    def test_parses_geometry(self):
        assert parse_depth_geometry(
            {"x-lostcam-depth": "320x240; format=u16mm"}
        ) == (320, 240, "u16mm")

    def test_defaults_format_when_absent(self):
        assert parse_depth_geometry({"x-lostcam-depth": "64x48"})[2] == "u16mm"

    def test_missing_header_raises(self):
        with pytest.raises(DepthError, match="X-LostCam-Depth"):
            parse_depth_geometry({})

    @pytest.mark.parametrize("value", ["axb", "0x10", "10x0", "320"])
    def test_rejects_bad_dimensions(self, value):
        with pytest.raises(DepthError):
            parse_depth_geometry({"x-lostcam-depth": value})

    def test_parses_intrinsics(self):
        assert parse_intrinsics(
            {"x-lostcam-intrinsics": "1,2,3,4"}
        ) == (1.0, 2.0, 3.0, 4.0)

    @pytest.mark.parametrize("value", [None, "1,2,3", "a,b,c,d", ""])
    def test_bad_intrinsics_become_none(self, value):
        headers = {"x-lostcam-intrinsics": value} if value is not None else {}
        assert parse_intrinsics(headers) is None


class TestDecodeDepth:
    def test_round_trips_values(self):
        raster = np.array([[0, 1000], [65535, 250]], dtype=np.uint16)
        decoded = decode_depth(raster.astype("<u2").tobytes(), 2, 2)
        assert np.array_equal(decoded, raster)

    def test_wrong_size_raises(self):
        with pytest.raises(DepthError, match="expected"):
            decode_depth(b"\x00\x00", 10, 10)

    def test_unknown_format_raises(self):
        with pytest.raises(DepthError, match="unsupported depth format"):
            decode_depth(b"\x00\x00", 1, 1, fmt="f32m")

    def test_little_endian_is_enforced(self):
        """Byte order must not depend on the machine reading the dataset."""
        decoded = decode_depth(b"\xe8\x03", 1, 1)
        assert decoded[0, 0] == 1000


class TestDepthFrame:
    def test_metres_marks_invalid_as_nan(self):
        raster = np.array([[0, 1000]], dtype=np.uint16)
        frame = DepthFrame(raster, 0, None)
        metres = frame.metres()
        assert np.isnan(metres[0, 0])
        assert metres[0, 1] == pytest.approx(1.0)

    def test_coverage_counts_valid_pixels(self):
        raster = np.array([[0, 0, 500, 500]], dtype=np.uint16)
        assert DepthFrame(raster, 0, None).coverage == pytest.approx(0.5)

    def test_dimensions(self):
        frame = DepthFrame(np.zeros((48, 64), dtype=np.uint16), 0, None)
        assert (frame.width, frame.height) == (64, 48)


class TestDepthParser:
    def test_parses_one_frame(self):
        raster = np.full((4, 8), 400, dtype=np.uint16)
        frames = DepthParser(BOUNDARY).feed(depth_part(raster))
        assert len(frames) == 1
        assert np.array_equal(frames[0].millimetres, raster)
        assert frames[0].timestamp_ms == 1000
        assert frames[0].intrinsics == (360.0, 360.0, 32.0, 24.0)

    def test_parses_several_frames_in_one_read(self):
        raster = np.full((2, 2), 100, dtype=np.uint16)
        stream = depth_part(raster) + depth_part(raster) + depth_part(raster)
        assert len(DepthParser(BOUNDARY).feed(stream)) == 3

    def test_frame_split_across_reads(self):
        raster = np.full((4, 4), 250, dtype=np.uint16)
        stream = depth_part(raster)
        for split in range(1, len(stream)):
            parser = DepthParser(BOUNDARY)
            frames = parser.feed(stream[:split]) + parser.feed(stream[split:])
            assert len(frames) == 1, f"failed at split {split}"
            assert np.array_equal(frames[0].millimetres, raster)

    def test_incomplete_payload_is_withheld(self):
        raster = np.full((4, 4), 250, dtype=np.uint16)
        stream = depth_part(raster)
        assert DepthParser(BOUNDARY).feed(stream[:-10]) == []

    def test_oversized_declared_length_raises(self):
        header = (
            f"--{BOUNDARY}\r\nContent-Length: 999999999\r\n"
            f"X-LostCam-Depth: 4x4; format=u16mm\r\n\r\n"
        ).encode("ascii")
        with pytest.raises(DepthError, match="over cap"):
            DepthParser(BOUNDARY).feed(header)

    def test_zero_pixels_are_preserved_as_no_measurement(self):
        raster = np.zeros((2, 2), dtype=np.uint16)
        (frame,) = DepthParser(BOUNDARY).feed(depth_part(raster))
        assert frame.coverage == 0.0


class TestSampleAligner:
    def sample(self, channel: str, t: int) -> Sample:
        return Sample(t, 1, channel, {"t": t, "seq": 1, "ch": channel, "v": t})

    def test_keeps_latest_per_channel(self):
        aligner = SampleAligner()
        aligner.observe(self.sample("motion", 100))
        aligner.observe(self.sample("motion", 200))
        snapshot = aligner.snapshot(200)
        assert snapshot["motion"]["v"] == 200

    def test_records_sample_age(self):
        aligner = SampleAligner()
        aligner.observe(self.sample("motion", 100))
        assert aligner.snapshot(150)["motion"]["age_ms"] == 50

    def test_drops_stale_samples(self):
        """A one-second-old tracking state should not be attached to a frame."""
        aligner = SampleAligner(max_age_ms=100)
        aligner.observe(self.sample("motion", 0))
        assert "motion" not in aligner.snapshot(500)
        assert aligner.dropped_stale == 1

    def test_slightly_future_samples_are_not_stale(self):
        # Concurrent streams routinely deliver a sample just ahead of a frame.
        aligner = SampleAligner(max_age_ms=100)
        aligner.observe(self.sample("motion", 120))
        assert "motion" in aligner.snapshot(100)

    def test_channel_key_is_not_duplicated_in_the_body(self):
        aligner = SampleAligner()
        aligner.observe(self.sample("motion", 10))
        assert "ch" not in aligner.snapshot(10)["motion"]

    def test_tracks_seen_channels(self):
        aligner = SampleAligner()
        aligner.observe(self.sample("motion", 1))
        aligner.observe(self.sample("ar.face", 1))
        assert aligner.channels == {"motion", "ar.face"}


class TestDepthHolder:
    def test_returns_the_latest_frame(self):
        holder = DepthHolder()
        holder.observe(np.full((2, 2), 5, dtype=np.uint16), 100, (1, 2, 3, 4))
        frame, timestamp, intrinsics = holder.take(120)
        assert frame is not None and frame[0, 0] == 5
        assert timestamp == 100
        assert intrinsics == (1, 2, 3, 4)

    def test_stale_frame_is_not_returned(self):
        holder = DepthHolder(max_age_ms=50)
        holder.observe(np.zeros((2, 2), dtype=np.uint16), 0, None)
        assert holder.take(500)[0] is None

    def test_empty_holder(self):
        assert DepthHolder().take(0)[0] is None


class TestDatasetWriter:
    def jpeg(self) -> bytes:
        return b"\xff\xd8fake-jpeg-body\xff\xd9"

    def test_writes_frame_and_manifest(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1000)
        writer.finalise()

        assert (tmp_path / "ds" / "frames" / "000001.jpg").read_bytes() == self.jpeg()
        records = read_manifest(tmp_path / "ds")
        assert len(records) == 1
        assert records[0]["frame"] == 1
        assert records[0]["file"] == "frames/000001.jpg"
        assert records[0]["t"] == 1000

    def test_jpeg_bytes_are_stored_verbatim(self, tmp_path):
        """Re-encoding would add a second generation of artefacts."""
        writer = DatasetWriter(tmp_path / "ds")
        original = self.jpeg()
        writer.write_frame(original, timestamp_ms=1)
        writer.finalise()
        stored = (tmp_path / "ds" / "frames" / "000001.jpg").read_bytes()
        assert stored == original

    def test_frame_numbering_is_sequential_and_padded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        for index in range(3):
            writer.write_frame(self.jpeg(), timestamp_ms=index)
        writer.finalise()
        for name in ("000001.jpg", "000002.jpg", "000003.jpg"):
            assert (tmp_path / "ds" / "frames" / name).exists()

    def test_relative_timestamp_is_recorded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=5000)
        writer.write_frame(self.jpeg(), timestamp_ms=5200)
        writer.finalise()
        records = read_manifest(tmp_path / "ds")
        assert records[0]["t_rel"] == 0
        assert records[1]["t_rel"] == 200

    def test_metrics_are_embedded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        metrics = FrameMetrics(mean=120.5, sharpness=88.0)
        writer.write_frame(self.jpeg(), timestamp_ms=1, metrics=metrics)
        writer.finalise()
        record = read_manifest(tmp_path / "ds")[0]
        assert record["metrics"]["mean"] == pytest.approx(120.5)
        assert record["metrics"]["sharpness"] == pytest.approx(88.0)

    def test_samples_are_embedded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1,
                           samples={"motion": {"accel": [0, 0, 1], "age_ms": 5}})
        writer.finalise()
        record = read_manifest(tmp_path / "ds")[0]
        assert record["samples"]["motion"]["accel"] == [0, 0, 1]

    def test_depth_is_written_and_reloadable(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        raster = np.array([[0, 400], [401, 65535]], dtype=np.uint16)
        writer.write_frame(self.jpeg(), timestamp_ms=1000, depth=raster,
                           depth_timestamp_ms=980, depth_intrinsics=(1, 2, 3, 4))
        writer.finalise()

        record = read_manifest(tmp_path / "ds")[0]
        assert record["depth"]["width"] == 2
        assert record["depth"]["height"] == 2
        assert record["depth"]["format"] == "u16mm"
        # Skew is recorded so a consumer can reject badly aligned pairs.
        assert record["depth"]["skew_ms"] == 20

        reloaded = load_depth(tmp_path / "ds", record)
        assert reloaded is not None
        assert np.array_equal(reloaded, raster)

    def test_depth_can_be_disabled(self, tmp_path):
        config = DatasetConfig(save_depth=False)
        writer = DatasetWriter(tmp_path / "ds", config)
        writer.write_frame(self.jpeg(), timestamp_ms=1,
                           depth=np.zeros((2, 2), dtype=np.uint16))
        writer.finalise()
        assert "depth" not in read_manifest(tmp_path / "ds")[0]

    def test_label_is_recorded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1, label="printing")
        writer.finalise()
        assert read_manifest(tmp_path / "ds")[0]["label"] == "printing"

    def test_metadata_records_the_configuration(self, tmp_path):
        config = DatasetConfig(
            source="http://phone:4747/video",
            roi=ROI(10, 20, 100, 100),
            calibration=Calibration.from_plate(ROI(0, 0, 440, 440), 220.0),
            plate_reference_mm=400.0,
            notes="ender 3, pla",
        )
        writer = DatasetWriter(tmp_path / "ds", config)
        writer.finalise()

        document = json.loads((tmp_path / "ds" / "dataset.json").read_text())
        assert document["config"]["roi"] == {"x": 10, "y": 20,
                                            "width": 100, "height": 100}
        assert document["config"]["calibration"]["mm_per_pixel_x"] == pytest.approx(0.5)
        assert document["config"]["plate_reference_mm"] == pytest.approx(400.0)
        assert document["config"]["notes"] == "ender 3, pla"

    def test_summary_is_stamped_on_finalise(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1000)
        writer.write_frame(self.jpeg(), timestamp_ms=2000)
        summary = writer.finalise()

        assert summary["frames"] == 2
        assert summary["duration_ms"] == 1000
        assert summary["average_fps"] == pytest.approx(2.0)
        document = json.loads((tmp_path / "ds" / "dataset.json").read_text())
        assert document["summary"]["frames"] == 2

    def test_events_are_recorded(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1)
        writer.write_event("failure", "spaghetti started", timestamp_ms=1200)
        writer.finalise()

        lines = (tmp_path / "ds" / "events.jsonl").read_text().strip().splitlines()
        event = json.loads(lines[0])
        assert event["event"] == "failure"
        assert event["note"] == "spaghetti started"
        # Anchored to a frame so the tag can be found in the video.
        assert event["frame"] == 1

    def test_refuses_to_mix_two_runs_in_one_directory(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1)
        writer.finalise()

        with pytest.raises(DatasetError, match="already exists"):
            DatasetWriter(tmp_path / "ds")

    def test_overwrite_replaces_the_previous_run(self, tmp_path):
        writer = DatasetWriter(tmp_path / "ds")
        for _ in range(3):
            writer.write_frame(self.jpeg(), timestamp_ms=1)
        writer.finalise()

        second = DatasetWriter(tmp_path / "ds", overwrite=True)
        second.write_frame(self.jpeg(), timestamp_ms=1)
        second.finalise()
        assert len(read_manifest(tmp_path / "ds")) == 1

    def test_manifest_survives_a_truncated_last_line(self, tmp_path):
        """An interrupted recording is exactly when the data matters."""
        writer = DatasetWriter(tmp_path / "ds")
        writer.write_frame(self.jpeg(), timestamp_ms=1)
        writer.write_frame(self.jpeg(), timestamp_ms=2)
        writer.finalise()

        path = tmp_path / "ds" / "manifest.jsonl"
        text = path.read_text()
        path.write_text(text + '{"frame":3,"file":"frames/00')

        records = read_manifest(tmp_path / "ds")
        assert len(records) == 2

    def test_context_manager_finalises(self, tmp_path):
        with DatasetWriter(tmp_path / "ds") as writer:
            writer.write_frame(self.jpeg(), timestamp_ms=1)
        document = json.loads((tmp_path / "ds" / "dataset.json").read_text())
        assert document["summary"]["frames"] == 1
