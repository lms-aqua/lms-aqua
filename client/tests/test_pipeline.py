"""Transform, decode and pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest

from lostcam.decode import DecodeError, jpeg_to_rgb, rgb_to_jpeg
from lostcam.pipeline import FramePipeline, LatestFrameBuffer, Stats, blank_frame
from lostcam.transform import Transform, fit
from lostcam.virtualcam import NullSink, VirtualCameraError


def gradient(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frame[0, 0] = [255, 0, 0]  # a corner marker to track orientation
    return frame


class TestDecode:
    def test_round_trip_preserves_shape(self):
        frame = gradient(64, 48)
        decoded = jpeg_to_rgb(rgb_to_jpeg(frame, quality=95))
        assert decoded.shape == (48, 64, 3)
        assert decoded.dtype == np.uint8

    def test_round_trip_is_close(self):
        frame = gradient(32, 32)
        decoded = jpeg_to_rgb(rgb_to_jpeg(frame, quality=95))
        # JPEG is lossy; a generous bound still catches channel swaps.
        assert np.abs(decoded.astype(int) - frame.astype(int)).mean() < 12

    def test_channel_order_is_rgb(self):
        red = np.zeros((8, 8, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        decoded = jpeg_to_rgb(rgb_to_jpeg(red, quality=100))
        assert decoded[4, 4, 0] > 200
        assert decoded[4, 4, 1] < 60
        assert decoded[4, 4, 2] < 60

    def test_empty_data_raises(self):
        with pytest.raises(DecodeError, match="empty"):
            jpeg_to_rgb(b"")

    def test_garbage_raises(self):
        with pytest.raises(DecodeError):
            jpeg_to_rgb(b"\xff\xd8definitely not a jpeg\xff\xd9")

    def test_encode_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            rgb_to_jpeg(np.zeros((4, 4), dtype=np.uint8))


class TestTransform:
    def test_identity_by_default(self):
        frame = gradient(8, 6)
        assert np.array_equal(Transform().apply(frame), frame)

    def test_rotate_90_swaps_axes(self):
        frame = gradient(8, 6)
        assert Transform(rotate=90).apply(frame).shape == (8, 6, 3)

    def test_rotate_180_is_double_flip(self):
        frame = gradient(8, 6)
        assert np.array_equal(Transform(rotate=180).apply(frame), frame[::-1, ::-1])

    def test_rotate_360_is_identity(self):
        frame = gradient(8, 6)
        assert np.array_equal(Transform(rotate=360).apply(frame), frame)

    def test_hflip_moves_corner_marker(self):
        frame = gradient(8, 6)
        out = Transform(hflip=True).apply(frame)
        assert np.array_equal(out[0, -1], frame[0, 0])

    def test_vflip_moves_corner_marker(self):
        frame = gradient(8, 6)
        out = Transform(vflip=True).apply(frame)
        assert np.array_equal(out[-1, 0], frame[0, 0])

    def test_output_is_contiguous(self):
        """pyvirtualcam needs real memory, not a numpy view."""
        out = Transform(rotate=90, hflip=True).apply(gradient(8, 6))
        assert out.flags["C_CONTIGUOUS"]

    def test_non_right_angle_rejected(self):
        with pytest.raises(ValueError, match="multiple of 90"):
            Transform(rotate=45)


class TestFit:
    def test_exact_size_passes_through(self):
        frame = gradient(32, 24)
        assert fit(frame, 32, 24).shape == (24, 32, 3)

    @pytest.mark.parametrize("mode", ["contain", "cover", "stretch"])
    def test_always_returns_exact_target(self, mode):
        frame = gradient(40, 30)
        assert fit(frame, 64, 64, mode).shape == (64, 64, 3)

    def test_contain_letterboxes_with_background(self):
        frame = np.full((10, 10, 3), 255, dtype=np.uint8)
        out = fit(frame, 40, 20, "contain")
        assert out[10, 0, 0] == 0  # left bar is background
        assert out[10, 20, 0] == 255  # centre is image

    def test_contain_preserves_aspect_ratio(self):
        frame = np.full((10, 20, 3), 255, dtype=np.uint8)  # 2:1
        out = fit(frame, 100, 100, "contain")
        lit_rows = np.where(out[:, 50, 0] > 128)[0]
        assert 45 <= len(lit_rows) <= 55  # half the height, as 2:1 demands

    def test_cover_fills_every_pixel(self):
        frame = np.full((10, 20, 3), 255, dtype=np.uint8)
        out = fit(frame, 100, 100, "cover")
        assert (out > 128).all()

    def test_upscale_works(self):
        assert fit(gradient(8, 8), 64, 64).shape == (64, 64, 3)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            fit(gradient(8, 8), 16, 16, "squish")

    @pytest.mark.parametrize("size", [(0, 10), (10, 0), (-4, 4)])
    def test_invalid_target_rejected(self, size):
        with pytest.raises(ValueError, match="must be positive"):
            fit(gradient(8, 8), *size)


class TestFramePipeline:
    def test_submits_frame_to_sink(self):
        sink = NullSink(64, 48)
        pipeline = FramePipeline(sink)
        assert pipeline.submit(rgb_to_jpeg(gradient(64, 48))) is True
        assert sink.frames == 1
        assert pipeline.stats.frames == 1

    def test_resizes_to_sink_dimensions(self):
        sink = NullSink(100, 100)
        FramePipeline(sink).submit(rgb_to_jpeg(gradient(64, 48)))
        assert sink.last_frame is not None
        assert sink.last_frame.shape == (100, 100, 3)

    def test_rotation_is_applied_before_fitting(self):
        sink = NullSink(48, 64)
        pipeline = FramePipeline(sink, Transform(rotate=90), fit_mode="stretch")
        pipeline.submit(rgb_to_jpeg(gradient(64, 48)))
        assert sink.last_frame.shape == (64, 48, 3)

    def test_corrupt_frame_is_counted_not_raised(self):
        """One bad frame on a lossy link must not tear down the stream."""
        sink = NullSink(64, 48)
        pipeline = FramePipeline(sink)
        assert pipeline.submit(b"\xff\xd8garbage\xff\xd9") is False
        assert pipeline.stats.decode_errors == 1
        assert sink.frames == 0
        # and a good frame still works afterwards
        assert pipeline.submit(rgb_to_jpeg(gradient(64, 48))) is True

    def test_counts_bytes_in(self):
        sink = NullSink(16, 16)
        pipeline = FramePipeline(sink)
        jpeg = rgb_to_jpeg(gradient(16, 16))
        pipeline.submit(jpeg)
        assert pipeline.stats.bytes_in == len(jpeg)


class TestNullSink:
    def test_counts_frames(self):
        sink = NullSink(4, 4)
        sink.send(blank_frame(4, 4))
        sink.send(blank_frame(4, 4))
        assert sink.frames == 2

    def test_context_manager(self):
        with NullSink(4, 4) as sink:
            sink.send(blank_frame(4, 4))
        assert sink.frames == 1


class TestVirtualCameraGuards:
    def test_mismatched_frame_size_is_rejected(self):
        """A wrongly sized frame must fail loudly, not corrupt the device."""

        class Fake:
            width, height = 64, 48

            def send(self, frame):
                raise AssertionError("should not be reached")

        # Exercise the same guard VirtualCamera.send applies.
        from lostcam.virtualcam import VirtualCamera

        camera = VirtualCamera.__new__(VirtualCamera)
        camera.width, camera.height = 64, 48
        with pytest.raises(VirtualCameraError, match="camera expects"):
            VirtualCamera.send(camera, blank_frame(32, 32))


class TestLatestFrameBuffer:
    def test_returns_most_recent_and_counts_drops(self):
        buffer = LatestFrameBuffer()
        buffer.put(b"old")
        buffer.put(b"new")
        assert buffer.get(timeout=0.1) == b"new"
        assert buffer.dropped == 1

    def test_empty_get_times_out(self):
        assert LatestFrameBuffer().get(timeout=0.01) is None

    def test_get_clears_the_slot(self):
        buffer = LatestFrameBuffer()
        buffer.put(b"one")
        assert buffer.get(timeout=0.1) == b"one"
        assert buffer.get(timeout=0.01) is None


class TestStats:
    def test_average_fps_is_positive_after_frames(self):
        stats = Stats()
        stats.frames = 10
        assert stats.average_fps > 0

    def test_instant_fps_resets_window(self):
        stats = Stats()
        stats.frames = 5
        assert stats.instant_fps() >= 0
        assert stats.instant_fps() == pytest.approx(0, abs=1e-6)
