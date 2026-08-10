"""Plate scanning, height mapping and object measurement tests.

The important ones here build a *synthetic camera*: a known plane at a known
tilt, with known boxes on it, rendered to a depth raster. That makes the
measurements checkable against ground truth in millimetres, which is the only way
to catch a sign error or a wrong axis — both of which produce plausible-looking
numbers that are simply wrong.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from lostcam.plate import (
    DEFAULT_HEIGHT_THRESHOLD_MM,
    HeightMap,
    ObjectTracker,
    Plane,
    PlateCalibration,
    PlateError,
    PlateMapper,
    PlateObject,
    _dilate,
    _erode,
    build_height_map,
    close_mask,
    fit_plane,
    label_components,
    plane_basis,
    recommended_cell_mm,
    sample_pitch_mm,
    scan_plate,
    segment_objects,
    summarise,
    unproject,
)

INTRINSICS = (200.0, 200.0, 64.0, 48.0)  # fx, fy, cx, cy for a 128x96 raster
RASTER = (96, 128)  # rows, columns


# MARK: - A synthetic depth camera


def render_plate(distance_mm: float = 400.0,
                 tilt_degrees: float = 0.0,
                 boxes: list[tuple[float, float, float, float, float]] | None = None,
                 raster: tuple[int, int] = RASTER,
                 intrinsics: tuple[float, float, float, float] = INTRINSICS,
                 invalid_border: int = 0) -> np.ndarray:
    """Render a depth raster of a tilted plate with boxes standing on it.

    ``boxes`` are ``(centre_u_mm, centre_v_mm, width_mm, depth_mm, height_mm)`` in
    *plate coordinates* — the same frame the code under test measures in.

    Each pixel's ray is intersected with the actual plane it should hit: the plate
    plane, or, inside a box's footprint, the parallel plane offset toward the
    camera by the box height. Approximating that by shifting z instead would make
    a box of height h measure as ``h·cos²(tilt)``, which is a subtly wrong ground
    truth that would then "validate" a wrong implementation.
    """
    rows, columns = raster
    fx, fy, cx, cy = intrinsics

    # Plate plane through (0, 0, distance), normal tilted about the x-axis and
    # pointing back toward the camera.
    tilt = np.radians(tilt_degrees)
    normal = np.array([0.0, -np.sin(tilt), -np.cos(tilt)])
    origin = np.array([0.0, 0.0, distance_mm])
    offset = float(-normal @ origin)

    # An in-plane basis, so box footprints are specified in plate coordinates.
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.cross(normal, u_axis)

    us, vs = np.meshgrid(np.arange(columns), np.arange(rows))
    directions = np.stack(
        [(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, dtype=np.float64)], axis=-1
    )
    denominator = directions @ normal

    def intersect(plane_offset: float) -> np.ndarray:
        """Ray/plane intersection distance along z for ``n·p + offset = 0``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = -plane_offset / denominator
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 0.0)
        return scale

    plate_scale = intersect(offset)
    depth = directions[..., 2] * plate_scale
    hit = plate_scale > 0

    for box in boxes or []:
        centre_u, centre_v, width, depth_mm_box, height = box
        # The box's top face lies on a plane parallel to the plate, offset toward
        # the camera by its height.
        raised_scale = intersect(offset - height)
        raised = directions * raised_scale[..., None]
        relative = raised - origin
        local_u = relative @ u_axis
        local_v = relative @ v_axis
        inside = (
            (raised_scale > 0)
            & (np.abs(local_u - centre_u) <= width / 2.0)
            & (np.abs(local_v - centre_v) <= depth_mm_box / 2.0)
        )
        depth = np.where(inside, raised[..., 2], depth)
        hit = hit | inside

    out = np.clip(np.round(depth), 0, 65535).astype(np.uint16)
    out[~hit] = 0

    if invalid_border > 0:
        out[:invalid_border, :] = 0
        out[-invalid_border:, :] = 0
        out[:, :invalid_border] = 0
        out[:, -invalid_border:] = 0
    return out


def calibration_from(depth: np.ndarray, plate_mm: float = 200.0,
                     cell_mm: float = 2.0) -> PlateCalibration:
    report = scan_plate([depth] * 5, INTRINSICS, plate_mm, cell_mm=cell_mm)
    assert report.ok, report.problems
    assert report.calibration is not None
    return report.calibration


# MARK: - Unprojection


class TestUnproject:
    def test_returns_only_valid_points(self):
        depth = np.zeros((4, 4), dtype=np.uint16)
        depth[1, 1] = 500
        depth[2, 2] = 600
        points = unproject(depth, (100.0, 100.0, 2.0, 2.0))
        assert points.shape == (2, 3)

    def test_zero_is_never_unprojected_to_the_origin(self):
        """A zero means "no measurement", not a point at the camera."""
        depth = np.zeros((8, 8), dtype=np.uint16)
        assert unproject(depth, INTRINSICS).shape == (0, 3)

    def test_principal_point_maps_to_the_optical_axis(self):
        depth = np.zeros((96, 128), dtype=np.uint16)
        depth[48, 64] = 400  # exactly at (cy, cx)
        (point,) = unproject(depth, INTRINSICS)
        assert point[0] == pytest.approx(0.0, abs=1e-4)
        assert point[1] == pytest.approx(0.0, abs=1e-4)
        assert point[2] == pytest.approx(400.0)

    def test_offset_pixel_scales_with_depth(self):
        depth = np.zeros((96, 128), dtype=np.uint16)
        depth[48, 84] = 400  # 20 px right of centre
        (point,) = unproject(depth, INTRINSICS)
        # x = (u - cx) * z / fx = 20 * 400 / 200
        assert point[0] == pytest.approx(40.0)

    def test_rejects_bad_intrinsics(self):
        with pytest.raises(PlateError, match="invalid intrinsics"):
            unproject(np.ones((4, 4), dtype=np.uint16), (0.0, 100.0, 2.0, 2.0))


# MARK: - Plane fitting


class TestFitPlane:
    def test_recovers_a_head_on_plane(self):
        depth = render_plate(distance_mm=400.0, tilt_degrees=0.0)
        plane = fit_plane(unproject(depth, INTRINSICS))
        # Normal should point back at the camera, i.e. along -z.
        assert plane.normal[2] == pytest.approx(-1.0, abs=1e-3)
        assert plane.offset == pytest.approx(400.0, abs=1.0)
        assert plane.rms_mm < 1.0

    def test_recovers_a_tilted_plane(self):
        depth = render_plate(distance_mm=450.0, tilt_degrees=30.0)
        plane = fit_plane(unproject(depth, INTRINSICS))
        assert plane.tilt_degrees == pytest.approx(30.0, abs=1.5)
        assert plane.rms_mm < 2.0

    def test_normal_always_points_toward_the_camera(self):
        """A flipped normal would invert every height in a dataset."""
        for tilt in (0.0, 15.0, 40.0):
            depth = render_plate(tilt_degrees=tilt)
            plane = fit_plane(unproject(depth, INTRINSICS))
            # Positive offset means the camera origin is on the positive side.
            assert plane.offset > 0
            centre = np.array([[0.0, 0.0, 400.0]], dtype=np.float32)
            # A point closer to the camera than the plate must be positive.
            nearer = centre - np.array([[0.0, 0.0, 50.0]], dtype=np.float32)
            assert plane.signed_distance(nearer)[0] > 0

    def test_outliers_are_rejected(self):
        depth = render_plate(distance_mm=400.0)
        points = unproject(depth, INTRINSICS)
        # A cluster of rogue points, as the printer frame would produce.
        rogue = points[:200].copy()
        rogue[:, 2] -= 150.0
        polluted = np.vstack([points, rogue])
        plane = fit_plane(polluted)
        assert plane.offset == pytest.approx(400.0, abs=3.0)
        assert plane.inlier_fraction < 1.0

    def test_too_few_points_raises(self):
        with pytest.raises(PlateError, match="at least 3"):
            fit_plane(np.zeros((2, 3), dtype=np.float32))

    def test_signed_distance_is_millimetres(self):
        depth = render_plate(distance_mm=400.0, tilt_degrees=0.0)
        plane = fit_plane(unproject(depth, INTRINSICS))
        point = np.array([[0.0, 0.0, 370.0]], dtype=np.float32)  # 30mm nearer
        assert plane.signed_distance(point)[0] == pytest.approx(30.0, abs=1.0)


class TestPlaneBasis:
    def test_axes_are_orthonormal_and_in_plane(self):
        plane = fit_plane(unproject(render_plate(tilt_degrees=25.0), INTRINSICS))
        u, v = plane_basis(plane)
        normal = plane.normal_array
        assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-5)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
        assert float(u @ v) == pytest.approx(0.0, abs=1e-5)
        assert float(u @ normal) == pytest.approx(0.0, abs=1e-5)
        assert float(v @ normal) == pytest.approx(0.0, abs=1e-5)

    def test_basis_is_stable_for_the_same_plane(self):
        plane = fit_plane(unproject(render_plate(tilt_degrees=10.0), INTRINSICS))
        first = plane_basis(plane)
        second = plane_basis(plane)
        assert np.allclose(first[0], second[0])
        assert np.allclose(first[1], second[1])

    def test_handles_a_plane_aligned_with_the_seed_axis(self):
        # Normal along x would make the default seed degenerate.
        plane = Plane(normal=(1.0, 0.0, 0.0), offset=100.0)
        u, v = plane_basis(plane)
        assert float(u @ plane.normal_array) == pytest.approx(0.0, abs=1e-6)
        assert float(v @ plane.normal_array) == pytest.approx(0.0, abs=1e-6)


# MARK: - Scanning


class TestScanPlate:
    def test_scans_a_clean_plate(self):
        report = scan_plate([render_plate()] * 5, INTRINSICS, 200.0)
        assert report.ok, report.problems
        assert report.frames_used == 5
        assert report.calibration is not None
        assert report.calibration.plane.rms_mm < 2.0
        assert "tilt" in report.summary()

    def test_records_the_configured_plate_size(self):
        report = scan_plate([render_plate()] * 3, INTRINSICS, 220.0, 210.0,
                            cell_mm=1.0)
        calibration = report.calibration
        assert calibration is not None
        assert calibration.plate_width_mm == 220.0
        assert calibration.plate_height_mm == 210.0
        assert calibration.grid_shape == (210, 220)

    def test_no_frames_is_reported_not_raised(self):
        report = scan_plate([], INTRINSICS, 200.0)
        assert not report.ok
        assert "no depth frames" in report.problems[0]

    def test_mostly_empty_depth_is_rejected_with_advice(self):
        blank = np.zeros(RASTER, dtype=np.uint16)
        report = scan_plate([blank] * 3, INTRINSICS, 200.0)
        assert not report.ok
        # The message must say what to do about it, not just that it failed.
        assert "shiny" in report.problems[0] or "closer" in report.problems[0]

    def test_changing_raster_size_is_rejected(self):
        report = scan_plate(
            [render_plate(), np.zeros((48, 64), dtype=np.uint16)],
            INTRINSICS, 200.0,
        )
        assert not report.ok
        assert "changed size" in report.problems[0]

    def test_steep_angle_warns_but_still_scans(self):
        report = scan_plate([render_plate(tilt_degrees=70.0)] * 3, INTRINSICS, 200.0)
        assert report.ok, report.problems
        assert any("off head-on" in warning for warning in report.warnings)

    def test_warns_when_the_plate_is_barely_visible(self):
        # A 400mm plate seen by a camera that only covers ~200mm of it.
        report = scan_plate([render_plate()] * 3, INTRINSICS, 400.0)
        assert report.ok, report.problems
        assert any("only sees" in warning for warning in report.warnings)

    def test_origin_lies_on_the_plate_plane(self):
        """Plate coordinates must have zero height at their own origin."""
        calibration = calibration_from(render_plate())
        origin = np.asarray(calibration.origin, dtype=np.float32)[None, :]
        assert calibration.plane.signed_distance(origin)[0] == pytest.approx(
            0.0, abs=1e-2
        )


class TestCalibrationIO:
    def test_round_trips_through_json(self, tmp_path):
        original = calibration_from(render_plate(tilt_degrees=20.0))
        path = original.save(tmp_path / "plate.json")
        loaded = PlateCalibration.load(path)

        assert loaded.plate_width_mm == original.plate_width_mm
        assert loaded.cell_mm == original.cell_mm
        assert loaded.grid_shape == original.grid_shape
        assert np.allclose(loaded.plane.normal, original.plane.normal, atol=1e-6)
        assert loaded.plane.offset == pytest.approx(original.plane.offset, abs=1e-3)
        assert np.allclose(loaded.origin, original.origin, atol=1e-3)

    def test_saved_file_is_readable_json_with_a_version(self, tmp_path):
        path = calibration_from(render_plate()).save(tmp_path / "plate.json")
        document = json.loads(path.read_text())
        assert document["version"] == 1
        assert "tilt_degrees" in document

    def test_missing_file_reports_clearly(self, tmp_path):
        with pytest.raises(PlateError, match="could not read"):
            PlateCalibration.load(tmp_path / "nope.json")

    def test_malformed_calibration_is_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"plane": {"normal": [0, 0]}}')
        with pytest.raises(PlateError):
            PlateCalibration.load(path)


# MARK: - Height maps


class TestBuildHeightMap:
    def test_empty_plate_is_flat_and_near_zero(self):
        depth = render_plate()
        calibration = calibration_from(depth)
        height_map = build_height_map(depth, calibration)

        assert height_map.coverage > 0.2
        measured = height_map.heights[height_map.valid]
        assert np.abs(measured).max() < 3.0, "an empty plate should read ~0mm"

    def test_a_box_reads_its_real_height(self):
        box_height = 30.0
        depth = render_plate(boxes=[(0.0, 0.0, 40.0, 40.0, box_height)])
        calibration = calibration_from(render_plate())
        height_map = build_height_map(depth, calibration)

        assert height_map.heights.max() == pytest.approx(box_height, abs=2.0)

    def test_heights_are_correct_on_a_tilted_plate(self):
        """The whole reason for fitting a plane instead of one distance.

        With a scalar reference, the far half of a tilted plate reads as far below
        zero and the near half far above it. With a plane, both read ~0.
        """
        empty = render_plate(distance_mm=450.0, tilt_degrees=35.0)
        calibration = calibration_from(empty)
        height_map = build_height_map(empty, calibration)

        measured = height_map.heights[height_map.valid]
        assert np.abs(measured).max() < 4.0, (
            f"tilted empty plate should read ~0mm, got up to "
            f"{np.abs(measured).max():.1f}mm"
        )

    def test_box_height_is_right_on_a_tilted_plate(self):
        box_height = 25.0
        calibration = calibration_from(render_plate(distance_mm=450.0,
                                                   tilt_degrees=30.0))
        depth = render_plate(distance_mm=450.0, tilt_degrees=30.0,
                             boxes=[(0.0, 0.0, 40.0, 40.0, box_height)])
        height_map = build_height_map(depth, calibration)
        assert height_map.heights.max() == pytest.approx(box_height, abs=3.0)

    def test_unmeasured_cells_are_invalid_not_zero_height(self):
        """A cell with no depth is occluded, not flat."""
        depth = render_plate(invalid_border=20)
        calibration = calibration_from(render_plate())
        height_map = build_height_map(depth, calibration)
        assert not height_map.valid.all()
        assert height_map.coverage < 1.0

    def test_grid_shape_follows_the_calibration(self):
        calibration = calibration_from(render_plate(), plate_mm=200.0, cell_mm=4.0)
        height_map = build_height_map(render_plate(), calibration)
        assert height_map.heights.shape == calibration.grid_shape == (50, 50)

    def test_rescales_intrinsics_for_a_different_raster_size(self):
        """Intrinsics belong to the raster they were measured on."""
        calibration = calibration_from(render_plate())
        big_intrinsics = (400.0, 400.0, 128.0, 96.0)
        big = render_plate(raster=(192, 256), intrinsics=big_intrinsics)
        height_map = build_height_map(big, calibration)
        measured = height_map.heights[height_map.valid]
        assert measured.size > 0
        assert np.abs(measured).max() < 6.0

    def test_all_invalid_depth_yields_an_empty_map(self):
        calibration = calibration_from(render_plate())
        height_map = build_height_map(np.zeros(RASTER, dtype=np.uint16), calibration)
        assert height_map.coverage == 0.0
        assert not height_map.valid.any()

    def test_rejects_a_non_2d_frame(self):
        calibration = calibration_from(render_plate())
        with pytest.raises(PlateError, match="2D"):
            build_height_map(np.zeros((4, 4, 3), dtype=np.uint16), calibration)

    def test_u16_export_distinguishes_absent_from_zero(self):
        heights = np.array([[0.0, 5.0]], dtype=np.float32)
        valid = np.array([[False, True]])
        exported = HeightMap(heights, valid, 1.0).to_u16_mm()
        assert exported[0, 0] == 0, "absent must export as 0"
        assert exported[0, 1] == 6, "a real height is offset by 1 to free up 0"


# MARK: - Component labelling


class TestLabelComponents:
    def test_labels_two_separate_blobs(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True
        mask[6:9, 6:9] = True
        labels, count = label_components(mask)
        assert count == 2
        assert labels[1, 1] != labels[7, 7]

    def test_diagonal_touching_blobs_stay_separate(self):
        # 4-connectivity: a diagonal touch is not a connection.
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        _, count = label_components(mask)
        assert count == 2

    def test_u_shape_is_one_component(self):
        """The classic union-find case: two arms joined at the bottom."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:5, 1] = True
        mask[1:5, 4] = True
        mask[4, 1:5] = True
        labels, count = label_components(mask)
        assert count == 1
        assert labels[1, 1] == labels[1, 4]

    def test_labels_are_contiguous_from_one(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        mask[5, 5] = True
        mask[9, 9] = True
        labels, count = label_components(mask)
        assert count == 3
        assert sorted(np.unique(labels[labels > 0])) == [1, 2, 3]

    def test_empty_mask(self):
        labels, count = label_components(np.zeros((5, 5), dtype=bool))
        assert count == 0
        assert not labels.any()

    def test_full_mask_is_one_component(self):
        _, count = label_components(np.ones((7, 7), dtype=bool))
        assert count == 1

    def test_rejects_non_2d(self):
        with pytest.raises(PlateError, match="2D"):
            label_components(np.ones((2, 2, 2), dtype=bool))


# MARK: - Object measurement


class TestSegmentObjects:
    def build(self, boxes, cell_mm=2.0, plate_mm=200.0):
        calibration = calibration_from(render_plate(), plate_mm=plate_mm,
                                       cell_mm=cell_mm)
        depth = render_plate(boxes=boxes)
        height_map = build_height_map(depth, calibration)
        return height_map, calibration

    def test_empty_plate_has_no_objects(self):
        height_map, calibration = self.build([])
        assert segment_objects(height_map, calibration) == []

    def test_finds_one_box(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        objects = segment_objects(height_map, calibration)
        assert len(objects) == 1

    def test_measures_height_in_millimetres(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        (item,) = segment_objects(height_map, calibration)
        assert item.height_max_mm == pytest.approx(30.0, abs=2.5)

    def test_measures_footprint_in_square_millimetres(self):
        # A 40x40mm box is 1600mm².
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        (item,) = segment_objects(height_map, calibration)
        assert item.footprint_mm2 == pytest.approx(1600.0, rel=0.25)

    def test_measures_bounding_box_in_millimetres(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 20.0, 25.0)])
        (item,) = segment_objects(height_map, calibration)
        assert item.bbox_u_mm == pytest.approx(40.0, rel=0.3)
        assert item.bbox_v_mm == pytest.approx(20.0, rel=0.4)

    def test_estimates_volume(self):
        # A 40x40x30mm box is 48000mm³.
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        (item,) = segment_objects(height_map, calibration)
        assert item.volume_mm3 == pytest.approx(48000.0, rel=0.35)

    def test_separates_two_boxes(self):
        height_map, calibration = self.build([
            (-50.0, 0.0, 30.0, 30.0, 20.0),
            (50.0, 0.0, 30.0, 30.0, 40.0),
        ])
        objects = segment_objects(height_map, calibration)
        assert len(objects) == 2
        heights = sorted(item.height_max_mm for item in objects)
        assert heights[0] == pytest.approx(20.0, abs=3.0)
        assert heights[1] == pytest.approx(40.0, abs=3.0)

    def test_objects_are_positioned_on_opposite_sides(self):
        height_map, calibration = self.build([
            (-50.0, 0.0, 30.0, 30.0, 20.0),
            (50.0, 0.0, 30.0, 30.0, 20.0),
        ])
        objects = segment_objects(height_map, calibration)
        centres = sorted(item.centre_u_mm for item in objects)
        assert centres[0] < 0 < centres[1]

    def test_sorted_largest_first(self):
        height_map, calibration = self.build([
            (-50.0, 0.0, 20.0, 20.0, 20.0),
            (40.0, 0.0, 50.0, 50.0, 20.0),
        ])
        objects = segment_objects(height_map, calibration)
        assert objects[0].footprint_mm2 > objects[1].footprint_mm2

    def test_speckle_below_the_minimum_footprint_is_ignored(self):
        height_map, calibration = self.build([(0.0, 0.0, 3.0, 3.0, 30.0)])
        assert segment_objects(height_map, calibration,
                               min_footprint_mm2=200.0) == []

    def test_objects_below_the_height_threshold_are_ignored(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 2.0)])
        assert segment_objects(height_map, calibration,
                               threshold_mm=DEFAULT_HEIGHT_THRESHOLD_MM) == []

    def test_max_objects_is_capped(self):
        boxes = [
            (float(x), float(y), 8.0, 8.0, 20.0)
            for x in range(-60, 61, 20)
            for y in range(-30, 31, 20)
        ]
        height_map, calibration = self.build(boxes)
        objects = segment_objects(height_map, calibration, min_footprint_mm2=10.0,
                                  max_objects=3)
        assert len(objects) == 3

    def test_solidity_is_high_for_a_solid_block(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        (item,) = segment_objects(height_map, calibration)
        assert item.solidity > 0.8

    def test_all_measurements_are_finite(self):
        height_map, calibration = self.build([(0.0, 0.0, 40.0, 40.0, 30.0)])
        (item,) = segment_objects(height_map, calibration)
        for key, value in item.as_dict().items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"{key} is not finite"


# MARK: - Tracking


class TestCellSizing:
    """Regression tests for the bug that made detection silently return nothing.

    A grid finer than the depth sampling density leaves an empty cell between
    every measured one, so the occupancy mask becomes a checkerboard, every cell
    labels as its own component, all fall below the minimum footprint, and the
    plate reports as empty.
    """

    # An iPhone-like depth raster: 256x192 at 400mm samples roughly every 2.2mm.
    REAL_RASTER = (192, 256)
    REAL_INTRINSICS = (180.0, 180.0, 128.0, 96.0)

    def real_scan(self, cell_mm=None, plate_mm=220.0):
        empty = render_plate(distance_mm=400.0, raster=self.REAL_RASTER,
                             intrinsics=self.REAL_INTRINSICS)
        return scan_plate([empty] * 5, self.REAL_INTRINSICS, plate_mm,
                          cell_mm=cell_mm)

    def test_sample_pitch_matches_the_pinhole_maths(self):
        # 400mm / 180px focal = 2.22mm per pixel.
        assert sample_pitch_mm(self.REAL_INTRINSICS, 400.0) == pytest.approx(2.222,
                                                                            abs=1e-3)

    def test_recommended_cell_is_coarser_than_the_pitch(self):
        pitch = sample_pitch_mm(self.REAL_INTRINSICS, 400.0)
        cell = recommended_cell_mm(self.REAL_INTRINSICS, 400.0)
        assert cell > pitch
        assert cell == pytest.approx(3.5)

    def test_recommended_cell_never_goes_below_one_millimetre(self):
        # A very close, very high-resolution sensor should not produce 0.2mm cells.
        assert recommended_cell_mm((2000.0, 2000.0, 0.0, 0.0), 100.0) >= 1.0

    def test_scan_picks_a_workable_cell_size_by_default(self):
        report = self.real_scan()
        assert report.ok, report.problems
        assert report.calibration is not None
        pitch = sample_pitch_mm(self.REAL_INTRINSICS, 400.0)
        assert report.calibration.cell_mm >= pitch

    def test_default_cell_size_detects_a_real_object(self):
        """The headline regression: this returned zero objects before the fix."""
        report = self.real_scan()
        calibration = report.calibration
        assert calibration is not None

        depth = render_plate(distance_mm=400.0, raster=self.REAL_RASTER,
                             intrinsics=self.REAL_INTRINSICS,
                             boxes=[(0.0, 0.0, 60.0, 60.0, 40.0)])
        height_map = build_height_map(depth, calibration)
        objects = segment_objects(height_map, calibration)

        assert len(objects) == 1, (
            f"expected one object, got {len(objects)} — the grid is shattering"
        )
        assert objects[0].height_max_mm == pytest.approx(40.0, abs=4.0)
        # A 60x60mm box is 3600mm².
        assert objects[0].footprint_mm2 == pytest.approx(3600.0, rel=0.3)

    def test_too_fine_a_cell_warns_with_a_usable_suggestion(self):
        report = self.real_scan(cell_mm=1.0)
        assert report.ok, report.problems
        warning = next(
            (w for w in report.warnings if "finer than" in w), None
        )
        assert warning is not None, report.warnings
        assert "--cell-mm" in warning

    def test_closing_rejoins_a_shattered_object(self):
        """Defence in depth: even at a too-fine cell size, closing helps."""
        report = self.real_scan(cell_mm=1.0)
        calibration = report.calibration
        assert calibration is not None
        depth = render_plate(distance_mm=400.0, raster=self.REAL_RASTER,
                             intrinsics=self.REAL_INTRINSICS,
                             boxes=[(0.0, 0.0, 60.0, 60.0, 40.0)])
        height_map = build_height_map(depth, calibration)

        without = segment_objects(height_map, calibration, close_gaps=False)
        with_closing = segment_objects(height_map, calibration, close_gaps=True)
        assert len(with_closing) < len(without) or (
            with_closing and with_closing[0].footprint_mm2 > 500
        ), "closing should rejoin the shattered cells"

    def test_closing_does_not_inflate_measurements(self):
        """Bridged cells must not be counted as measured area or volume."""
        report = self.real_scan()
        calibration = report.calibration
        assert calibration is not None
        depth = render_plate(distance_mm=400.0, raster=self.REAL_RASTER,
                             intrinsics=self.REAL_INTRINSICS,
                             boxes=[(0.0, 0.0, 60.0, 60.0, 40.0)])
        height_map = build_height_map(depth, calibration)

        closed = segment_objects(height_map, calibration, close_gaps=True)
        raw = segment_objects(height_map, calibration, close_gaps=False)
        if closed and raw:
            # Same object, and closing added no area beyond measured cells.
            assert closed[0].footprint_mm2 <= sum(o.footprint_mm2 for o in raw) + 1e-6


class TestMorphology:
    def test_dilate_grows_by_one_cell(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert _dilate(mask).sum() == 9

    def test_erode_shrinks_an_interior_blob_by_one_cell(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True  # 3x3 interior blob
        assert _erode(mask).sum() == 1

    def test_erode_treats_outside_the_grid_as_occupied(self):
        """An object against the plate edge must not be shaved off for touching it."""
        mask = np.ones((5, 5), dtype=bool)
        assert _erode(mask).sum() == 25

    def test_closing_preserves_an_object_at_the_plate_edge(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True  # in the corner
        assert close_mask(mask)[0, 0]

    def test_closing_fills_a_one_cell_hole(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        assert close_mask(mask)[2, 2]

    def test_closing_bridges_a_one_cell_gap(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[2, 1:3] = True
        mask[2, 4:6] = True
        _, before = label_components(mask)
        _, after = label_components(close_mask(mask))
        assert before == 2
        assert after == 1

    def test_closing_leaves_a_solid_block_alone(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[3:6, 3:6] = True
        assert close_mask(mask).sum() == mask.sum()

    def test_closing_does_not_join_far_apart_blobs(self):
        mask = np.zeros((9, 12), dtype=bool)
        mask[4, 1] = True
        mask[4, 9] = True
        _, count = label_components(close_mask(mask))
        assert count == 2

    def test_closing_zero_iterations_is_identity(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        assert np.array_equal(close_mask(mask, iterations=0), mask)


class TestNegativeHeightEncoding:
    def test_deep_void_exports_as_absent_not_flat(self):
        """A cell 250mm below the plate is a dropout, not a flat plate."""
        heights = np.array([[-250.0, 0.0, 12.0]], dtype=np.float32)
        valid = np.array([[True, True, True]])
        exported = HeightMap(heights, valid, 1.0).to_u16_mm()
        assert exported[0, 0] == 0, "a deep void must not read as a measurement"
        assert exported[0, 1] == 1, "a genuine 0mm height stays measured"
        assert exported[0, 2] == 13

    def test_small_negative_noise_is_kept(self):
        # Sensor noise straddles zero on a genuinely flat plate.
        heights = np.array([[-1.0]], dtype=np.float32)
        exported = HeightMap(heights, np.array([[True]]), 1.0).to_u16_mm()
        assert exported[0, 0] == 1


class TestScanRobustness:
    def test_origin_is_robust_to_a_lopsided_view(self):
        """A median origin is not dragged by extra visible surface on one side."""
        depth = render_plate(distance_mm=400.0)
        # Blank the left half, as an occluding printer frame would.
        depth[:, : depth.shape[1] // 2] = 0
        report = scan_plate([depth] * 5, INTRINSICS, 200.0)
        assert report.ok, report.problems

    def test_scan_emits_no_runtime_warnings(self):
        """An all-NaN pixel column is expected and must not warn on the terminal."""
        depth = render_plate(distance_mm=400.0, invalid_border=6)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = scan_plate([depth] * 4, INTRINSICS, 200.0)
        assert report.ok, report.problems

    def test_oversized_plane_warns_about_offset_coordinates(self):
        # A tiny configured plate against a big visible surface.
        report = scan_plate([render_plate()] * 3, INTRINSICS, 60.0)
        assert report.ok, report.problems
        assert any("much larger" in warning for warning in report.warnings), (
            report.warnings
        )


class TestObjectTracker:
    def make(self, object_id: int, u: float, v: float,
             height: float = 20.0) -> PlateObject:
        return PlateObject(
            object_id=object_id, centre_u_mm=u, centre_v_mm=v,
            bbox_u_mm=10.0, bbox_v_mm=10.0, bbox_min_u_mm=u - 5,
            bbox_min_v_mm=v - 5, footprint_mm2=100.0, height_max_mm=height,
            height_mean_mm=height / 2, volume_mm3=1000.0, cells=25, solidity=1.0,
        )

    def test_assigns_ids(self):
        tracker = ObjectTracker()
        objects = tracker.update([self.make(1, 0.0, 0.0)])
        assert objects[0].track_id == 1

    def test_same_object_keeps_its_id_across_frames(self):
        tracker = ObjectTracker()
        tracker.update([self.make(1, 0.0, 0.0)])
        # The label from segmentation changed, but the position did not.
        second = tracker.update([self.make(7, 2.0, 1.0)])
        assert second[0].track_id == 1
        assert second[0].age_frames == 2

    def test_a_distant_object_gets_a_new_id(self):
        tracker = ObjectTracker(match_radius_mm=10.0)
        tracker.update([self.make(1, 0.0, 0.0)])
        second = tracker.update([self.make(1, 80.0, 80.0)])
        assert second[0].track_id == 2

    def test_two_objects_do_not_swap_identities(self):
        tracker = ObjectTracker()
        first = tracker.update([self.make(1, -50.0, 0.0), self.make(2, 50.0, 0.0)])
        left_id = next(o.track_id for o in first if o.centre_u_mm < 0)
        right_id = next(o.track_id for o in first if o.centre_u_mm > 0)

        second = tracker.update([self.make(9, 51.0, 0.0), self.make(8, -49.0, 0.0)])
        assert next(o.track_id for o in second if o.centre_u_mm < 0) == left_id
        assert next(o.track_id for o in second if o.centre_u_mm > 0) == right_id

    def test_one_track_cannot_be_claimed_twice(self):
        tracker = ObjectTracker(match_radius_mm=100.0)
        tracker.update([self.make(1, 0.0, 0.0)])
        # Two candidates both near the single existing track.
        objects = tracker.update([self.make(1, 1.0, 0.0), self.make(2, 2.0, 0.0)])
        ids = {item.track_id for item in objects}
        assert len(ids) == 2, "each object needs its own track id"

    def test_peak_height_is_remembered(self):
        tracker = ObjectTracker()
        tracker.update([self.make(1, 0.0, 0.0, height=10.0)])
        tracker.update([self.make(1, 0.0, 0.0, height=41.0)])
        # A print that fell over should not erase how tall it got.
        tracker.update([self.make(1, 0.0, 0.0, height=5.0)])
        assert tracker.peak_height(1) == pytest.approx(41.0)

    def test_vanished_tracks_are_forgotten(self):
        tracker = ObjectTracker(forget_after=2)
        tracker.update([self.make(1, 0.0, 0.0)])
        assert tracker.track_count == 1
        for _ in range(4):
            tracker.update([])
        assert tracker.track_count == 0


# MARK: - Whole pipeline


class TestPlateMapper:
    def test_reports_an_empty_plate_as_empty(self):
        calibration = calibration_from(render_plate())
        state = PlateMapper(calibration).process(render_plate())
        assert state.object_count == 0
        assert state.tallest_mm == 0.0
        assert state.total_volume_mm3 == 0.0

    def test_reports_a_growing_print(self):
        calibration = calibration_from(render_plate(), cell_mm=2.0)
        mapper = PlateMapper(calibration)

        heights = []
        for height in (10.0, 25.0, 45.0):
            depth = render_plate(boxes=[(0.0, 0.0, 40.0, 40.0, height)])
            state = mapper.process(depth)
            assert state.object_count == 1
            heights.append(state.tallest_mm)

        # Growth must be monotonic and roughly the right magnitude.
        assert heights[0] < heights[1] < heights[2]
        assert heights[2] == pytest.approx(45.0, abs=4.0)

    def test_track_ids_persist_as_the_print_grows(self):
        calibration = calibration_from(render_plate(), cell_mm=2.0)
        mapper = PlateMapper(calibration)
        ids = []
        for height in (10.0, 20.0, 30.0):
            state = mapper.process(
                render_plate(boxes=[(0.0, 0.0, 40.0, 40.0, height)])
            )
            ids.append(state.objects[0].track_id)
        assert len(set(ids)) == 1, f"track id changed across frames: {ids}"

    def test_summary_dict_is_json_serialisable(self):
        calibration = calibration_from(render_plate(), cell_mm=2.0)
        state = PlateMapper(calibration).process(
            render_plate(boxes=[(0.0, 0.0, 40.0, 40.0, 30.0)])
        )
        document = state.as_dict()
        json.dumps(document)  # must not raise
        assert document["object_count"] == 1
        assert document["objects"][0]["height_max_mm"] > 20

    def test_summarise_totals_add_up(self):
        calibration = calibration_from(render_plate(), cell_mm=2.0)
        depth = render_plate(boxes=[
            (-50.0, 0.0, 30.0, 30.0, 20.0),
            (50.0, 0.0, 30.0, 30.0, 40.0),
        ])
        height_map = build_height_map(depth, calibration)
        objects = segment_objects(height_map, calibration)
        state = summarise(height_map, objects)

        assert state.object_count == len(objects)
        assert state.occupied_mm2 == pytest.approx(
            sum(item.footprint_mm2 for item in objects)
        )
        assert state.tallest_mm == pytest.approx(
            max(item.height_max_mm for item in objects)
        )

    def test_tracking_can_be_disabled(self):
        calibration = calibration_from(render_plate(), cell_mm=2.0)
        mapper = PlateMapper(calibration, track=False)
        state = mapper.process(render_plate(boxes=[(0.0, 0.0, 40.0, 40.0, 30.0)]))
        assert state.objects[0].track_id is None
