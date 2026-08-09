"""Vision metrics, ROI and calibration tests.

These matter more than usual: every number here ends up as a training feature, so
a silent unit or convention error would be baked into a model rather than
crashing something.
"""

from __future__ import annotations

import numpy as np
import pytest

from lostcam.vision import (
    ROI,
    Calibration,
    CalibrationError,
    FrameAnalyser,
    exposure_clipping,
    sharpness,
    to_grey,
)


def flat(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def checkerboard(width: int, height: int, size: int = 4) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = (((xs // size) + (ys // size)) % 2 * 255).astype(np.uint8)
    return np.dstack([pattern] * 3)


class TestROI:
    def test_parses_text(self):
        assert ROI.parse("10,20,300,400") == ROI(10, 20, 300, 400)

    def test_tolerates_spaces(self):
        assert ROI.parse(" 1, 2 ,3, 4 ") == ROI(1, 2, 3, 4)

    @pytest.mark.parametrize("text", ["1,2,3", "a,b,c,d", "", "1,2,3,4,5"])
    def test_rejects_malformed(self, text):
        with pytest.raises(CalibrationError):
            ROI.parse(text)

    @pytest.mark.parametrize("args", [(0, 0, 0, 10), (0, 0, 10, 0), (-1, 0, 5, 5)])
    def test_rejects_impossible_geometry(self, args):
        with pytest.raises(CalibrationError):
            ROI(*args)

    def test_crop_returns_the_region(self):
        frame = flat(100, 80)
        frame[20:40, 10:30] = 255
        cropped = ROI(10, 20, 20, 20).crop(frame)
        assert cropped.shape == (20, 20, 3)
        assert (cropped == 255).all()

    def test_clipping_keeps_the_roi_inside_the_frame(self):
        """A stale ROI must not index out of range when the size changes."""
        clipped = ROI(50, 50, 500, 500).clipped_to((80, 100, 3))
        assert clipped.x + clipped.width <= 100
        assert clipped.y + clipped.height <= 80

    def test_crop_with_oversized_roi_still_works(self):
        assert ROI(0, 0, 9999, 9999).crop(flat(20, 10)).shape == (10, 20, 3)

    def test_scaled_maps_into_a_smaller_raster(self):
        # The depth raster is a fraction of the colour frame's size.
        # y is 50*0.25 = 12.5, which round() takes to 12 (banker's rounding);
        # a half-pixel either way is irrelevant at depth-raster resolution.
        scaled = ROI(100, 50, 200, 100).scaled(0.25, 0.25)
        assert scaled == ROI(25, 12, 50, 25)

    def test_scaled_never_collapses_to_zero(self):
        assert ROI(0, 0, 2, 2).scaled(0.01, 0.01).width >= 1


class TestCalibration:
    def test_from_plate_computes_mm_per_pixel(self):
        # A 220 mm Ender plate filling 440 px is 0.5 mm per pixel.
        calibration = Calibration.from_plate(ROI(0, 0, 440, 440), 220.0)
        assert calibration.mm_per_pixel_x == pytest.approx(0.5)
        assert calibration.mm_per_pixel_y == pytest.approx(0.5)

    def test_non_square_plate(self):
        calibration = Calibration.from_plate(ROI(0, 0, 400, 200), 200.0, 100.0)
        assert calibration.mm_per_pixel_x == pytest.approx(0.5)
        assert calibration.mm_per_pixel_y == pytest.approx(0.5)

    def test_reference_string_records_the_derivation(self):
        calibration = Calibration.from_plate(ROI(0, 0, 440, 440), 220.0)
        assert "220" in calibration.reference and "440" in calibration.reference

    def test_area_conversion(self):
        calibration = Calibration(0.5, 0.5)
        assert calibration.area_mm2(400) == pytest.approx(100.0)

    def test_length_conversion(self):
        assert Calibration(0.5, 0.5).length_mm(10) == pytest.approx(5.0)

    @pytest.mark.parametrize("args", [(0, 1), (1, 0), (-1, 1)])
    def test_rejects_non_positive_scale(self, args):
        with pytest.raises(CalibrationError):
            Calibration(*args)

    def test_zero_plate_width_rejected(self):
        with pytest.raises(CalibrationError):
            Calibration.from_plate(ROI(0, 0, 10, 10), 0)


class TestGreyscale:
    def test_rgb_to_luma_weights(self):
        red = np.zeros((2, 2, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        green = np.zeros((2, 2, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        # Green contributes more luma than red — the Rec. 601 weighting.
        assert to_grey(green).mean() > to_grey(red).mean()

    def test_grey_input_passes_through(self):
        assert to_grey(np.full((4, 4), 100, dtype=np.uint8)).mean() == 100

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            to_grey(np.zeros((4, 4, 1), dtype=np.uint8))


class TestSharpness:
    def test_flat_image_has_no_detail(self):
        assert sharpness(to_grey(flat(32, 32))) == pytest.approx(0.0)

    def test_checkerboard_is_sharper_than_flat(self):
        assert sharpness(to_grey(checkerboard(32, 32))) > 100

    def test_blurring_reduces_sharpness(self):
        sharp = to_grey(checkerboard(64, 64, size=8))
        # A crude 3x3 box blur, which is enough to move the metric.
        blurred = sharp.copy()
        blurred[1:-1, 1:-1] = (
            sharp[:-2, 1:-1] + sharp[2:, 1:-1] + sharp[1:-1, :-2]
            + sharp[1:-1, 2:] + sharp[1:-1, 1:-1]
        ) / 5.0
        assert sharpness(blurred) < sharpness(sharp)

    def test_tiny_images_do_not_crash(self):
        assert sharpness(np.zeros((1, 1), dtype=np.float32)) == 0.0
        assert sharpness(np.zeros((0, 0), dtype=np.float32)) == 0.0


class TestExposureClipping:
    def test_detects_blown_highlights(self):
        black, white = exposure_clipping(to_grey(flat(10, 10, 255)))
        assert white == pytest.approx(1.0)
        assert black == pytest.approx(0.0)

    def test_detects_crushed_blacks(self):
        black, white = exposure_clipping(to_grey(flat(10, 10, 0)))
        assert black == pytest.approx(1.0)

    def test_mid_grey_is_not_clipped(self):
        black, white = exposure_clipping(to_grey(flat(10, 10, 128)))
        assert black == 0.0 and white == 0.0

    def test_empty_region(self):
        assert exposure_clipping(np.zeros((0, 0), dtype=np.float32)) == (0.0, 0.0)


class TestFrameAnalyser:
    def test_reports_brightness_and_spread(self):
        metrics = FrameAnalyser().analyse(flat(32, 32, 100))
        assert metrics.mean == pytest.approx(100, abs=1)
        assert metrics.std == pytest.approx(0, abs=1e-3)

    def test_first_frame_has_no_difference(self):
        metrics = FrameAnalyser().analyse(flat(16, 16))
        assert metrics.diff_mean == 0.0
        assert metrics.diff_fraction == 0.0

    def test_identical_frames_show_no_change(self):
        analyser = FrameAnalyser()
        analyser.analyse(flat(16, 16, 100))
        metrics = analyser.analyse(flat(16, 16, 100))
        assert metrics.diff_mean == pytest.approx(0.0)

    def test_changed_frame_is_detected(self):
        analyser = FrameAnalyser()
        analyser.analyse(flat(16, 16, 50))
        metrics = analyser.analyse(flat(16, 16, 200))
        assert metrics.diff_mean > 100
        assert metrics.diff_fraction == pytest.approx(1.0)

    def test_roi_restricts_the_measurement(self):
        """The whole point of the ROI: measure the plate, not the room."""
        frame = flat(100, 100, 0)
        frame[40:60, 40:60] = 200  # a bright patch, only inside the ROI
        analyser = FrameAnalyser(roi=ROI(40, 40, 20, 20))
        assert analyser.analyse(frame).mean == pytest.approx(200, abs=1)
        # Without the ROI the same frame reads as almost black.
        assert FrameAnalyser().analyse(frame).mean < 20

    def test_reset_clears_the_difference_reference(self):
        analyser = FrameAnalyser()
        analyser.analyse(flat(16, 16, 50))
        analyser.reset()
        assert analyser.analyse(flat(16, 16, 200)).diff_mean == 0.0

    def test_frame_size_change_does_not_crash_differencing(self):
        analyser = FrameAnalyser()
        analyser.analyse(flat(16, 16))
        metrics = analyser.analyse(flat(32, 32))
        assert metrics.diff_mean == 0.0

    def test_metrics_dict_omits_absent_depth_fields(self):
        metrics = FrameAnalyser().analyse(flat(8, 8))
        as_dict = metrics.as_dict()
        assert "mean" in as_dict
        assert "depth_min_mm" not in as_dict


class TestDepthMetrics:
    def plate(self, width=32, height=24, distance=400, growth=0):
        raster = np.full((height, width), distance, dtype=np.uint16)
        if growth:
            raster[8:16, 8:16] = distance - growth
        raster[0, :] = 0  # invalid border, as a real sensor returns
        return raster

    def test_reports_distance_statistics(self):
        metrics = FrameAnalyser().analyse(flat(32, 24), self.plate())
        assert metrics.depth_min_mm == 400
        assert metrics.depth_max_mm == 400
        assert metrics.depth_median_mm == 400

    def test_invalid_pixels_are_excluded_not_counted_as_zero_distance(self):
        """Treating 0 as a distance would put a wall at the lens."""
        metrics = FrameAnalyser().analyse(flat(32, 24), self.plate())
        assert metrics.depth_min_mm == 400
        assert metrics.depth_coverage < 1.0

    def test_coverage_reflects_valid_fraction(self):
        raster = np.full((10, 10), 500, dtype=np.uint16)
        raster[:5, :] = 0
        metrics = FrameAnalyser().analyse(flat(10, 10), raster)
        assert metrics.depth_coverage == pytest.approx(0.5)

    def test_all_invalid_depth_reports_zero_coverage_and_no_stats(self):
        metrics = FrameAnalyser().analyse(
            flat(10, 10), np.zeros((10, 10), dtype=np.uint16)
        )
        assert metrics.depth_coverage == 0.0
        assert metrics.depth_min_mm is None

    def test_heights_need_a_plate_reference(self):
        analyser = FrameAnalyser()
        metrics = analyser.analyse(flat(32, 24), self.plate(growth=30))
        assert metrics.height_max_mm is None, "no reference means no heights"

    def test_height_is_measured_from_the_plate_reference(self):
        analyser = FrameAnalyser()
        analyser.plate_reference_mm = 400
        metrics = analyser.analyse(flat(32, 24), self.plate(growth=30))
        assert metrics.height_max_mm == pytest.approx(30.0)

    def test_things_behind_the_plate_are_not_negative_heights(self):
        analyser = FrameAnalyser()
        analyser.plate_reference_mm = 400
        further = np.full((24, 32), 450, dtype=np.uint16)
        metrics = analyser.analyse(flat(32, 24), further)
        assert metrics.height_max_mm == 0.0

    def test_depth_roi_is_rescaled_to_the_depth_raster(self):
        """The ROI is in colour pixels; depth is smaller. Reuse would be wrong."""
        colour = flat(320, 240)
        depth = np.full((60, 80), 500, dtype=np.uint16)
        depth[15:30, 20:40] = 300  # the region the scaled ROI should land on
        analyser = FrameAnalyser(roi=ROI(80, 60, 80, 60))
        metrics = analyser.analyse(colour, depth)
        assert metrics.depth_min_mm == 300

    def test_plate_reference_calibration_uses_median_of_medians(self):
        analyser = FrameAnalyser()
        frames = [self.plate(distance=d) for d in (398, 400, 402, 401, 399)]
        reference = analyser.calibrate_plate_reference(frames, (24, 32))
        assert reference == pytest.approx(400, abs=1.5)

    def test_calibration_refuses_when_there_is_too_little_data(self):
        """A guessed reference makes every later height wrong by the same amount."""
        analyser = FrameAnalyser()
        assert analyser.calibrate_plate_reference([self.plate()], (24, 32)) is None
        assert analyser.plate_reference_mm is None

    def test_calibration_ignores_frames_that_are_mostly_invalid(self):
        analyser = FrameAnalyser()
        empty = np.zeros((24, 32), dtype=np.uint16)
        frames = [empty, empty, empty, self.plate(), self.plate()]
        assert analyser.calibrate_plate_reference(frames, (24, 32)) is None
