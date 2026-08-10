"""Build-plate mapping: scan the plate once, then measure what is on it.

The idea is to do the hard part once, at setup, and make every later frame cheap
and metric.

**Setup** (`lostcam scan`) points the phone at an *empty* plate and fits a plane
to the depth points. That plane, plus the camera intrinsics, defines a plate
coordinate system in millimetres.

**Then** every depth frame is turned into a top-down **height map**: an
orthographic grid in plate coordinates holding the height above the plate. From
that grid, objects fall out as connected components, and each one has a real
footprint in mm², a real height in mm, and a real volume in mm³.

The grid's cell size is derived from the sensor, not chosen for tidiness — see
``recommended_cell_mm``. A grid finer than the depth sampling density leaves an
empty cell between every measured one and finds nothing at all.

Why a plane and not a single distance
------------------------------------
A phone on a tripod looks at the plate at an angle, so the plate's *depth* varies
across the frame — the far edge is genuinely further away. Subtracting one
scalar distance would report the far half of an empty plate as being below the
plate and the near half as above it. Fitting a plane and measuring along its
normal is the only way the numbers mean anything for a camera that is not
perfectly head-on.

Why an orthographic grid and not pixels
---------------------------------------
Perspective makes a pixel worth more millimetres far away than near. Resampling
into a plate-coordinate grid removes that entirely: every cell is the same
physical size, so areas and volumes are sums rather than approximations, and the
resulting map is a fixed-scale 2.5D image — which is a far better input to a model
than a perspective view whose scale depends on where in the frame something is.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Heights below this are treated as plate, not object. LiDAR noise at this range
# is a few millimetres, so anything lower cannot be distinguished from the plate.
DEFAULT_HEIGHT_THRESHOLD_MM = 4.0
# Components smaller than this are noise speckle, not prints.
DEFAULT_MIN_FOOTPRINT_MM2 = 25.0
# Only a fallback. The cell size should be derived from the sensor — see
# recommended_cell_mm — because a grid finer than the sampling density shatters
# into disconnected cells and finds nothing.
DEFAULT_CELL_MM = 3.0
# How much coarser than the raw sample pitch a cell should be. Below ~1.4 the
# grid develops holes between samples; much above it throws away resolution.
CELL_PITCH_SAFETY = 1.5


def sample_pitch_mm(intrinsics: tuple[float, float, float, float],
                    distance_mm: float) -> float:
    """Millimetres between neighbouring depth samples at a given distance.

    One pixel subtends ``distance / focal_length`` millimetres, so this is the
    real spatial resolution of the depth data — the number that decides how fine
    a height-map grid can usefully be.
    """
    fx, fy, _, _ = intrinsics
    focal = min(abs(fx), abs(fy))
    if focal <= 0 or distance_mm <= 0:
        raise PlateError("cannot compute sample pitch from these intrinsics")
    return float(distance_mm / focal)


def recommended_cell_mm(intrinsics: tuple[float, float, float, float],
                        distance_mm: float,
                        safety: float = CELL_PITCH_SAFETY) -> float:
    """Pick a height-map cell size the sensor can actually fill.

    This exists because the obvious choice — 1 mm, because millimetres are nice —
    is wrong on real hardware. An iPhone's depth raster at 400 mm samples roughly
    every 2 mm, so a 1 mm grid leaves an empty cell between every measured one:
    the occupancy mask becomes a checkerboard, connected-component labelling finds
    one component per cell, every one falls below the minimum footprint, and the
    whole plate reports as empty. Silently.
    """
    pitch = sample_pitch_mm(intrinsics, distance_mm)
    # Round up to a tidy half-millimetre so profiles are readable.
    raw = pitch * safety
    return float(max(1.0, np.ceil(raw * 2.0) / 2.0))


class PlateError(Exception):
    """The plate could not be scanned, or a calibration is unusable."""


# MARK: - Geometry


def unproject(depth_mm: np.ndarray,
              intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    """Turn a depth raster into an ``(N, 3)`` point cloud in millimetres.

    Camera convention: +x right, +y down, +z away from the camera, which is what
    the pinhole model with these intrinsics implies. Only pixels with a real
    measurement are returned — a zero means "no measurement" and must never be
    unprojected into a point at the origin.
    """
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise PlateError(f"invalid intrinsics: fx={fx}, fy={fy}")

    height, width = depth_mm.shape
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    vs, us = np.nonzero(valid)
    z = depth_mm[vs, us].astype(np.float32)
    x = (us.astype(np.float32) - cx) * z / fx
    y = (vs.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


@dataclass(frozen=True)
class Plane:
    """A plane as a unit normal and offset: ``normal · p + offset = 0``.

    The normal always points *toward the camera*, so a point in front of the
    plate has a positive signed distance. That convention is fixed here rather
    than left to callers, because getting it backwards silently inverts every
    height in a dataset.
    """

    normal: tuple[float, float, float]
    offset: float
    inlier_fraction: float = 1.0
    rms_mm: float = 0.0

    @property
    def normal_array(self) -> np.ndarray:
        return np.asarray(self.normal, dtype=np.float32)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Height above the plane, in millimetres, for ``(N, 3)`` points."""
        return points @ self.normal_array + self.offset

    def as_dict(self) -> dict:
        return {
            "normal": [round(v, 8) for v in self.normal],
            "offset": round(self.offset, 6),
            "inlier_fraction": round(self.inlier_fraction, 4),
            "rms_mm": round(self.rms_mm, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Plane:
        normal = data.get("normal")
        if not isinstance(normal, list) or len(normal) != 3:
            raise PlateError("plane normal must be three numbers")
        return cls(
            normal=(float(normal[0]), float(normal[1]), float(normal[2])),
            offset=float(data["offset"]),
            inlier_fraction=float(data.get("inlier_fraction", 1.0)),
            rms_mm=float(data.get("rms_mm", 0.0)),
        )

    @property
    def tilt_degrees(self) -> float:
        """Angle between the plate normal and the camera's optical axis.

        Reported because it is the number that tells you whether the rig is
        roughly top-down (small) or steeply oblique (large), which determines how
        much of the plate the sensor can actually see.
        """
        # The camera looks along +z, so a plate facing the camera has a normal
        # near (0, 0, -1).
        axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cosine = float(np.clip(np.dot(self.normal_array, axis), -1.0, 1.0))
        return float(np.degrees(np.arccos(abs(cosine))))


def fit_plane(points: np.ndarray, iterations: int = 3,
              inlier_mm: float = 8.0) -> Plane:
    """Least-squares plane fit, refined by discarding outliers.

    Plain least squares would be dragged by the printer frame, the bed clips and
    whatever else is in view, so the fit is repeated against the points that
    agreed with the previous pass. This is a cheap stand-in for RANSAC and is
    enough when the plate is most of what the sensor sees — which the setup step
    asks the user to arrange.
    """
    if points.shape[0] < 3:
        raise PlateError(
            f"need at least 3 depth points to fit a plane, got {points.shape[0]}"
        )

    working = points.astype(np.float64)
    normal = np.array([0.0, 0.0, -1.0])
    centroid = working.mean(axis=0)
    inlier_fraction = 1.0
    rms = 0.0

    for _ in range(max(1, iterations)):
        if working.shape[0] < 3:
            break
        centroid = working.mean(axis=0)
        centred = working - centroid
        # The plane normal is the direction of least variance, which is the
        # smallest right singular vector.
        try:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise PlateError(f"plane fit failed: {exc}") from exc
        normal = vt[-1]
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise PlateError("degenerate plane fit (zero normal)")
        normal = normal / norm

        distances = (points.astype(np.float64) - centroid) @ normal
        keep = np.abs(distances) <= inlier_mm
        inlier_fraction = float(keep.mean())
        rms = float(np.sqrt(np.mean(distances[keep] ** 2))) if keep.any() else 0.0
        if keep.sum() < 3:
            break
        working = points.astype(np.float64)[keep]

    offset = float(-normal @ centroid)

    # Fix the sign so the normal points toward the camera at the origin, making
    # heights above the plate positive.
    if offset < 0:
        normal = -normal
        offset = -offset

    return Plane(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset=offset,
        inlier_fraction=inlier_fraction,
        rms_mm=rms,
    )


def plane_basis(plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes.

    The rotation about the normal is arbitrary but *stable*: it is derived from
    the camera's x-axis, so a fixed camera yields the same plate axes every run
    and coordinates are comparable across sessions. It is not aligned to the
    plate's physical edges — that would need edge detection, and a stable
    arbitrary frame is enough for measuring what is on the plate.
    """
    normal = plane.normal_array.astype(np.float64)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(candidate, normal)) > 0.9:
        # The camera's x-axis is nearly along the normal; pick another seed.
        candidate = np.array([0.0, 1.0, 0.0])
    u = candidate - np.dot(candidate, normal) * normal
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        raise PlateError("could not build a plate basis")
    u = u / u_norm
    v = np.cross(normal, u)
    return u.astype(np.float32), v.astype(np.float32)


# MARK: - Calibration


@dataclass
class PlateCalibration:
    """Everything the setup scan establishes. Saved to JSON and reused."""

    plane: Plane
    intrinsics: tuple[float, float, float, float]
    depth_size: tuple[int, int]  # width, height of the depth raster
    origin: tuple[float, float, float]  # plate-coordinate origin, camera mm
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    plate_width_mm: float
    plate_height_mm: float
    cell_mm: float = DEFAULT_CELL_MM
    # Observed plate extent, which is how much of it the sensor actually saw.
    observed_extent_mm: tuple[float, float] = (0.0, 0.0)
    scanned_frames: int = 0
    notes: str = ""

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Grid rows, columns — one cell per ``cell_mm`` square."""
        rows = max(1, int(round(self.plate_height_mm / self.cell_mm)))
        columns = max(1, int(round(self.plate_width_mm / self.cell_mm)))
        return rows, columns

    @property
    def cell_area_mm2(self) -> float:
        return self.cell_mm * self.cell_mm

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "plane": self.plane.as_dict(),
            "intrinsics": [round(v, 4) for v in self.intrinsics],
            "depth_size": list(self.depth_size),
            "origin": [round(v, 4) for v in self.origin],
            "u_axis": [round(v, 8) for v in self.u_axis],
            "v_axis": [round(v, 8) for v in self.v_axis],
            "plate_width_mm": self.plate_width_mm,
            "plate_height_mm": self.plate_height_mm,
            "cell_mm": self.cell_mm,
            "observed_extent_mm": [round(v, 2) for v in self.observed_extent_mm],
            "scanned_frames": self.scanned_frames,
            "tilt_degrees": round(self.plane.tilt_degrees, 2),
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n",
                          encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> PlateCalibration:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlateError(f"could not read plate calibration {path}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> PlateCalibration:
        try:
            intrinsics = tuple(float(v) for v in data["intrinsics"])
            depth_size = tuple(int(v) for v in data["depth_size"])
            if len(intrinsics) != 4 or len(depth_size) != 2:
                raise PlateError("malformed intrinsics or depth_size")
            return cls(
                plane=Plane.from_dict(data["plane"]),
                intrinsics=(intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]),
                depth_size=(depth_size[0], depth_size[1]),
                origin=tuple(float(v) for v in data["origin"]),  # type: ignore[arg-type]
                u_axis=tuple(float(v) for v in data["u_axis"]),  # type: ignore[arg-type]
                v_axis=tuple(float(v) for v in data["v_axis"]),  # type: ignore[arg-type]
                plate_width_mm=float(data["plate_width_mm"]),
                plate_height_mm=float(data["plate_height_mm"]),
                cell_mm=float(data.get("cell_mm", DEFAULT_CELL_MM)),
                observed_extent_mm=tuple(
                    float(v) for v in data.get("observed_extent_mm", [0.0, 0.0])
                ),  # type: ignore[arg-type]
                scanned_frames=int(data.get("scanned_frames", 0)),
                notes=str(data.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlateError(f"malformed plate calibration: {exc}") from exc


@dataclass
class ScanReport:
    """What the setup scan found, and whether it is good enough to use."""

    calibration: PlateCalibration | None
    frames_used: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.calibration is not None and not self.problems

    def summary(self) -> str:
        if not self.calibration:
            return "scan failed: " + "; ".join(self.problems)
        calibration = self.calibration
        lines = [
            f"Plate plane fitted from {self.frames_used} frame(s):",
            f"  tilt from head-on:  {calibration.plane.tilt_degrees:.1f}°",
            f"  fit residual (RMS): {calibration.plane.rms_mm:.1f} mm",
            f"  inliers:            {calibration.plane.inlier_fraction * 100:.0f}%",
            f"  plate seen:         {calibration.observed_extent_mm[0]:.0f} x "
            f"{calibration.observed_extent_mm[1]:.0f} mm "
            f"(configured {calibration.plate_width_mm:.0f} x "
            f"{calibration.plate_height_mm:.0f} mm)",
            f"  grid:               {calibration.grid_shape[1]} x "
            f"{calibration.grid_shape[0]} cells at {calibration.cell_mm:g} mm",
        ]
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)


def scan_plate(depth_frames: list[np.ndarray],
               intrinsics: tuple[float, float, float, float],
               plate_width_mm: float,
               plate_height_mm: float | None = None,
               cell_mm: float | None = None,
               min_coverage: float = 0.25) -> ScanReport:
    """Fit the plate plane from frames of an *empty* plate.

    Returns a report rather than raising, because "the scan was not good enough"
    is an ordinary outcome the operator needs to act on — move the camera, add
    light, clear the plate — and a stack trace is a poor way to say that.
    """
    problems: list[str] = []
    warns: list[str] = []

    usable = [frame for frame in depth_frames if frame is not None and frame.size]
    if not usable:
        return ScanReport(None, 0, ["no depth frames arrived — is the depth "
                                    "channel enabled and is this a LiDAR device?"])

    shapes = {frame.shape for frame in usable}
    if len(shapes) > 1:
        return ScanReport(None, 0,
                          [f"depth frames changed size mid-scan: {sorted(shapes)}"])

    height_px, width_px = usable[0].shape
    coverages = [float((frame > 0).mean()) for frame in usable]
    mean_coverage = float(np.mean(coverages))
    if mean_coverage < min_coverage:
        problems.append(
            f"only {mean_coverage * 100:.0f}% of the depth frame returned a "
            f"measurement (need {min_coverage * 100:.0f}%). Move closer, reduce "
            f"the viewing angle, or light the plate — dark, shiny and glass beds "
            f"reflect the beam away from the sensor"
        )
        return ScanReport(None, len(usable), problems)

    # Median across frames per pixel: LiDAR is noisy frame to frame, and a median
    # is unbothered by the stray dropouts at plate edges.
    stack = np.stack([frame.astype(np.float32) for frame in usable])
    stack[stack == 0] = np.nan
    # A pixel that never returned anything is all-NaN, which is expected here and
    # not worth a RuntimeWarning on the user's terminal. errstate does not cover
    # this one — nanmedian raises it through the warnings module.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        averaged = np.nanmedian(stack, axis=0)
    averaged = np.nan_to_num(averaged, nan=0.0)

    points = unproject(averaged, intrinsics)
    if points.shape[0] < 100:
        return ScanReport(None, len(usable),
                          [f"only {points.shape[0]} valid depth points; too few "
                           f"to fit a plate plane"])

    try:
        plane = fit_plane(points)
    except PlateError as exc:
        return ScanReport(None, len(usable), [str(exc)])

    if plane.rms_mm > 15.0:
        warns.append(
            f"the plate fit residual is {plane.rms_mm:.0f} mm, which is high — "
            f"the plate may not be the dominant surface in view, or the plate was "
            f"not empty"
        )
    if plane.inlier_fraction < 0.5:
        warns.append(
            f"only {plane.inlier_fraction * 100:.0f}% of points lie on the fitted "
            f"plane; a lot of what the sensor sees is not the plate"
        )
    tilt = plane.tilt_degrees
    if tilt > 60:
        warns.append(
            f"the camera is {tilt:.0f}° off head-on. Heights still work, but the "
            f"far side of the plate will be poorly sampled and partly occluded by "
            f"whatever is printed in front of it"
        )

    # Keep only the points that actually lie on the plate, so the origin and the
    # observed extent describe the plate rather than the whole scene.
    heights = plane.signed_distance(points)
    on_plate = points[np.abs(heights) <= max(10.0, plane.rms_mm * 3.0)]
    if on_plate.shape[0] < 50:
        on_plate = points

    u_axis, v_axis = plane_basis(plane)
    # Median, not mean: the fitted plane extends past the bed onto whatever the
    # printer stands on, and a mean is dragged toward whichever side of the bed
    # happens to have more visible surface.
    origin = np.median(on_plate, axis=0).astype(np.float32)
    # Project the origin onto the plane, so plate coordinates have zero height at
    # the origin rather than an offset baked in.
    origin = origin - plane.signed_distance(origin[None, :])[0] * plane.normal_array

    local_u = (on_plate - origin) @ u_axis
    local_v = (on_plate - origin) @ v_axis
    observed = (
        float(np.percentile(local_u, 98) - np.percentile(local_u, 2)),
        float(np.percentile(local_v, 98) - np.percentile(local_v, 2)),
    )

    height_mm = plate_height_mm if plate_height_mm else plate_width_mm

    # Derive the grid resolution from what the sensor can actually resolve. This
    # is the difference between measuring objects and reporting an empty plate.
    distance = float(abs(plane.offset)) or 400.0
    suggested = recommended_cell_mm(intrinsics, distance)
    if cell_mm is None:
        cell_mm = suggested
    else:
        pitch = sample_pitch_mm(intrinsics, distance)
        if cell_mm < pitch:
            warns.append(
                f"a {cell_mm:g} mm grid is finer than this sensor's {pitch:.1f} mm "
                f"sample spacing at {distance:.0f} mm, so the height map will have "
                f"gaps between measured cells and objects may break apart or go "
                f"undetected. Use --cell-mm {suggested:g} or larger"
            )

    if observed[0] < plate_width_mm * 0.6 or observed[1] < height_mm * 0.6:
        warns.append(
            f"the sensor only sees about {observed[0]:.0f} x {observed[1]:.0f} mm "
            f"of a {plate_width_mm:.0f} x {height_mm:.0f} mm plate. Objects outside "
            f"that area will not be measured"
        )
    elif observed[0] > plate_width_mm * 1.6 or observed[1] > height_mm * 1.6:
        # The plane fit found a surface much bigger than the bed, which usually
        # means the bed sits on a larger flat surface that got included. Heights
        # are still right, but the plate-coordinate origin is the centre of
        # everything flat rather than the centre of the bed.
        warns.append(
            f"the flat surface in view is about {observed[0]:.0f} x "
            f"{observed[1]:.0f} mm, much larger than the {plate_width_mm:.0f} x "
            f"{height_mm:.0f} mm plate. Heights are unaffected, but plate "
            f"coordinates are centred on the whole visible surface, so object "
            f"positions may be offset from the true bed centre. Crop the view to "
            f"the bed for positions you can trust"
        )

    calibration = PlateCalibration(
        plane=plane,
        intrinsics=intrinsics,
        depth_size=(width_px, height_px),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        u_axis=(float(u_axis[0]), float(u_axis[1]), float(u_axis[2])),
        v_axis=(float(v_axis[0]), float(v_axis[1]), float(v_axis[2])),
        plate_width_mm=float(plate_width_mm),
        plate_height_mm=float(height_mm),
        cell_mm=float(cell_mm),
        observed_extent_mm=observed,
        scanned_frames=len(usable),
    )
    return ScanReport(calibration, len(usable), problems, warns)


# MARK: - Height mapping


@dataclass
class HeightMap:
    """A top-down orthographic map of the plate, in millimetres.

    ``heights`` holds the height above the plate per cell, and ``valid`` marks
    which cells had any depth point land in them. The two are separate because a
    cell with no measurement is not a cell of height zero — the plate might be
    occluded there, and treating it as flat would erase exactly the shadow an
    object casts.
    """

    heights: np.ndarray  # (rows, cols) float32, mm above plate
    valid: np.ndarray  # (rows, cols) bool
    cell_mm: float
    points_used: int = 0

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def occupancy(self, threshold_mm: float = DEFAULT_HEIGHT_THRESHOLD_MM
                  ) -> np.ndarray:
        """Cells that hold something taller than the noise floor."""
        return self.valid & (self.heights >= threshold_mm)

    def to_u16_mm(self, floor_mm: float = -2.0) -> np.ndarray:
        """Export as u16 millimetres, 0 meaning no measurement.

        Same convention as the depth wire format, so the same reader works and
        "absent" never gets confused with "flat".

        Heights below ``floor_mm`` are exported as *absent* rather than clamped to
        zero. A cell reading well below the plate is not a flat plate — it is a
        hole in the bed, a reflective dropout, or a bad plane fit — and clamping
        would turn "I cannot measure this" into a confident measurement of a
        surface that is not there. A couple of millimetres of slack is allowed
        because sensor noise straddles zero on a genuinely flat plate.
        """
        out = np.zeros(self.heights.shape, dtype=np.uint16)
        usable = (
            self.valid
            & np.isfinite(self.heights)
            & (self.heights >= floor_mm)
        )
        clipped = np.clip(self.heights[usable], 0, 65534)
        # +1 so a genuine zero height is distinguishable from "no measurement".
        out[usable] = (clipped + 1).astype(np.uint16)
        return out


def build_height_map(depth_mm: np.ndarray,
                     calibration: PlateCalibration) -> HeightMap:
    """Resample a depth frame into the plate's orthographic height grid."""
    if depth_mm.ndim != 2:
        raise PlateError("depth frame must be a 2D raster")

    intrinsics = calibration.intrinsics
    expected = (calibration.depth_size[1], calibration.depth_size[0])
    if depth_mm.shape != expected:
        # The intrinsics belong to the raster they were measured on, so a
        # different-sized frame needs them rescaled rather than reused.
        scale_x = depth_mm.shape[1] / calibration.depth_size[0]
        scale_y = depth_mm.shape[0] / calibration.depth_size[1]
        intrinsics = (
            intrinsics[0] * scale_x, intrinsics[1] * scale_y,
            intrinsics[2] * scale_x, intrinsics[3] * scale_y,
        )

    rows, columns = calibration.grid_shape
    heights = np.zeros((rows, columns), dtype=np.float32)
    valid = np.zeros((rows, columns), dtype=bool)

    points = unproject(depth_mm, intrinsics)
    if points.shape[0] == 0:
        return HeightMap(heights, valid, calibration.cell_mm, 0)

    origin = np.asarray(calibration.origin, dtype=np.float32)
    u_axis = np.asarray(calibration.u_axis, dtype=np.float32)
    v_axis = np.asarray(calibration.v_axis, dtype=np.float32)

    relative = points - origin
    local_u = relative @ u_axis
    local_v = relative @ v_axis
    height = calibration.plane.signed_distance(points)

    # Plate coordinates are centred on the origin, so shift into grid indices.
    half_width = calibration.plate_width_mm / 2.0
    half_height = calibration.plate_height_mm / 2.0
    column = np.floor((local_u + half_width) / calibration.cell_mm).astype(np.int64)
    row = np.floor((local_v + half_height) / calibration.cell_mm).astype(np.int64)

    inside = (
        (column >= 0) & (column < columns) & (row >= 0) & (row < rows)
        & np.isfinite(height)
    )
    if not inside.any():
        return HeightMap(heights, valid, calibration.cell_mm, 0)

    column = column[inside]
    row = row[inside]
    height = height[inside].astype(np.float32)

    flat = row * columns + column
    # Maximum per cell: several depth points land in one cell, and the tallest is
    # the one that describes what is there. A mean would smear an object's edge
    # into the plate around it.
    accumulator = np.full(rows * columns, -np.inf, dtype=np.float32)
    np.maximum.at(accumulator, flat, height)
    touched = np.isfinite(accumulator)

    heights.reshape(-1)[touched] = accumulator[touched]
    valid.reshape(-1)[touched] = True
    return HeightMap(heights, valid, calibration.cell_mm, int(height.size))


# MARK: - Object segmentation


def close_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological closing — dilate then erode — with a 3x3 element.

    Defence in depth against a grid that is slightly sparser than it looks.
    Bridging a one-cell gap turns a shattered object back into one component
    without inventing area, because the erode step gives back what the dilate
    step added anywhere it was not filling a hole.

    Hand-rolled with shifted slices so the client keeps its two dependencies.
    """
    if mask.ndim != 2:
        raise PlateError("mask must be 2D")
    out = mask
    for _ in range(max(0, iterations)):
        out = _erode(_dilate(out))
    return out


def _shifted_or(mask: np.ndarray) -> np.ndarray:
    padded = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool
    )
    padded[1:-1, 1:-1] = mask
    result = np.zeros_like(padded)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            result[1:-1, 1:-1] |= padded[dy:dy + mask.shape[0],
                                         dx:dx + mask.shape[1]]
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    return _shifted_or(mask)[1:-1, 1:-1]


def _erode(mask: np.ndarray) -> np.ndarray:
    """Erode by one cell, treating everything outside the grid as *occupied*.

    Erosion is dilation of the complement, inverted — and because the dilation
    pads with False, the complement's padding behaves as "outside is occupied".
    That is the boundary rule we want: an object sitting against the edge of the
    plate must not be shaved off just for touching the edge, which is what
    textbook zero-padded erosion would do.
    """
    return ~_dilate(~mask)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected components in a boolean mask.

    Union-find in two passes. Written out rather than pulled from scipy so the
    client keeps its two dependencies — the grid is a few hundred cells square,
    which this handles in milliseconds.
    """
    if mask.ndim != 2:
        raise PlateError("mask must be 2D")
    rows, columns = mask.shape
    labels = np.zeros((rows, columns), dtype=np.int32)
    parent: list[int] = [0]

    def find(label: int) -> int:
        root = label
        while parent[root] != root:
            root = parent[root]
        # Path compression, so repeated lookups stay near-constant.
        while parent[label] != root:
            parent[label], label = root, parent[label]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    next_label = 1
    for row in range(rows):
        for column in range(columns):
            if not mask[row, column]:
                continue
            up = labels[row - 1, column] if row > 0 else 0
            left = labels[row, column - 1] if column > 0 else 0
            if up and left:
                labels[row, column] = min(up, left)
                union(up, left)
            elif up:
                labels[row, column] = up
            elif left:
                labels[row, column] = left
            else:
                labels[row, column] = next_label
                parent.append(next_label)
                next_label += 1

    if next_label == 1:
        return labels, 0

    # Resolve to root labels, then renumber so they run 1..n with no gaps.
    resolved = np.zeros(next_label, dtype=np.int32)
    for label in range(1, next_label):
        resolved[label] = find(label)
    unique_roots = sorted({int(resolved[label]) for label in range(1, next_label)})
    renumber = {root: index + 1 for index, root in enumerate(unique_roots)}
    lookup = np.zeros(next_label, dtype=np.int32)
    for label in range(1, next_label):
        lookup[label] = renumber[int(resolved[label])]

    return lookup[labels], len(unique_roots)


@dataclass
class PlateObject:
    """One thing on the plate, measured in millimetres."""

    object_id: int
    # Position in plate coordinates, relative to the plate centre.
    centre_u_mm: float
    centre_v_mm: float
    bbox_u_mm: float
    bbox_v_mm: float
    bbox_min_u_mm: float
    bbox_min_v_mm: float
    footprint_mm2: float
    height_max_mm: float
    height_mean_mm: float
    volume_mm3: float
    cells: int
    solidity: float
    # Filled in by the tracker across frames.
    track_id: int | None = None
    age_frames: int = 0

    def as_dict(self) -> dict:
        document = {
            "id": self.object_id,
            "centre_mm": [round(self.centre_u_mm, 2), round(self.centre_v_mm, 2)],
            "bbox_mm": [round(self.bbox_u_mm, 2), round(self.bbox_v_mm, 2)],
            "bbox_origin_mm": [round(self.bbox_min_u_mm, 2),
                               round(self.bbox_min_v_mm, 2)],
            "footprint_mm2": round(self.footprint_mm2, 1),
            "height_max_mm": round(self.height_max_mm, 2),
            "height_mean_mm": round(self.height_mean_mm, 2),
            "volume_mm3": round(self.volume_mm3, 1),
            "cells": self.cells,
            "solidity": round(self.solidity, 3),
        }
        if self.track_id is not None:
            document["track"] = self.track_id
            document["age_frames"] = self.age_frames
        return document


def segment_objects(height_map: HeightMap, calibration: PlateCalibration,
                    threshold_mm: float = DEFAULT_HEIGHT_THRESHOLD_MM,
                    min_footprint_mm2: float = DEFAULT_MIN_FOOTPRINT_MM2,
                    max_objects: int = 64,
                    close_gaps: bool = True) -> list[PlateObject]:
    """Find the objects on the plate and measure each one.

    Purely geometric — no model, no training. Anything standing far enough above
    the fitted plate plane is an object, which is exactly the right definition for
    a plate that starts empty.
    """
    raw_occupancy = height_map.occupancy(threshold_mm)
    if not raw_occupancy.any():
        return []

    # Bridge one-cell gaps left by a grid finer than the sampling density. Only
    # the *labelling* uses the closed mask; every measurement below reads the
    # original cells, so closing can rejoin an object but never inflate its
    # footprint, height or volume.
    occupancy = close_mask(raw_occupancy) if close_gaps else raw_occupancy

    labels, count = label_components(occupancy)
    if count == 0:
        return []

    cell_area = calibration.cell_area_mm2
    minimum_cells = max(1, int(round(min_footprint_mm2 / cell_area)))
    rows, columns = occupancy.shape
    half_width = calibration.plate_width_mm / 2.0
    half_height = calibration.plate_height_mm / 2.0

    objects: list[PlateObject] = []
    for label in range(1, count + 1):
        # Measure only genuinely occupied cells: the closed mask may include a
        # bridging cell that was never measured, and counting it would add area
        # and volume that no sensor reading supports.
        selected = (labels == label) & raw_occupancy
        cells = int(selected.sum())
        if cells < minimum_cells:
            continue

        heights = height_map.heights[selected]
        rows_index, columns_index = np.nonzero(selected)

        # Cell centres, so a one-cell object sits at the middle of its cell
        # rather than at its corner.
        u_mm = (columns_index + 0.5) * calibration.cell_mm - half_width
        v_mm = (rows_index + 0.5) * calibration.cell_mm - half_height

        bbox_min_u = float(u_mm.min() - calibration.cell_mm / 2.0)
        bbox_min_v = float(v_mm.min() - calibration.cell_mm / 2.0)
        bbox_u = float(u_mm.max() - u_mm.min() + calibration.cell_mm)
        bbox_v = float(v_mm.max() - v_mm.min() + calibration.cell_mm)
        bbox_cells = max(
            1,
            int(round((bbox_u / calibration.cell_mm) * (bbox_v / calibration.cell_mm))),
        )

        objects.append(
            PlateObject(
                object_id=label,
                centre_u_mm=float(u_mm.mean()),
                centre_v_mm=float(v_mm.mean()),
                bbox_u_mm=bbox_u,
                bbox_v_mm=bbox_v,
                bbox_min_u_mm=bbox_min_u,
                bbox_min_v_mm=bbox_min_v,
                footprint_mm2=cells * cell_area,
                height_max_mm=float(heights.max()),
                height_mean_mm=float(heights.mean()),
                # Each cell contributes its height over one cell of area. This is
                # the volume of the *visible* shape, so an overhang or an
                # undercut the sensor cannot see is not in it.
                volume_mm3=float(heights.sum()) * cell_area,
                cells=cells,
                # How much of its bounding box the object fills — a blob is near
                # 1, a sprawl of spaghetti is much lower.
                solidity=float(cells / bbox_cells),
            )
        )

    # Largest first, so the interesting thing is at the top of the list, and
    # bounded so a frame full of speckle cannot produce a thousand entries.
    objects.sort(key=lambda item: item.footprint_mm2, reverse=True)
    return objects[:max_objects]


# MARK: - Tracking


class ObjectTracker:
    """Gives objects stable identities across frames.

    Nearest-centroid matching within a radius. Deliberately simple: the objects
    being tracked are bolted to a build plate and do not move, so the hard cases
    a real tracker exists for do not arise. What it buys is the ability to say
    "object 3 grew from 4 mm to 41 mm over two hours" rather than having a fresh
    unrelated list of objects every frame.
    """

    def __init__(self, match_radius_mm: float = 15.0, forget_after: int = 30) -> None:
        self.match_radius_mm = match_radius_mm
        self.forget_after = forget_after
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    def update(self, objects: list[PlateObject]) -> list[PlateObject]:
        """Assign track ids, mutating and returning the objects."""
        for track in self._tracks.values():
            track["missing"] += 1

        unmatched = list(objects)
        # Match the biggest objects first: they are the most reliable anchors, and
        # letting a speckle claim a track before the real object gets a chance is
        # how identities end up swapped.
        unmatched.sort(key=lambda item: item.footprint_mm2, reverse=True)

        claimed: set[int] = set()
        for item in unmatched:
            best_id = None
            best_distance = self.match_radius_mm
            for track_id, track in self._tracks.items():
                if track_id in claimed:
                    continue
                distance = float(
                    np.hypot(track["u"] - item.centre_u_mm,
                             track["v"] - item.centre_v_mm)
                )
                if distance <= best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = {
                    "u": item.centre_u_mm, "v": item.centre_v_mm,
                    "age": 0, "missing": 0,
                    "height_max_mm": item.height_max_mm,
                }
            claimed.add(best_id)

            track = self._tracks[best_id]
            track["u"] = item.centre_u_mm
            track["v"] = item.centre_v_mm
            track["age"] += 1
            track["missing"] = 0
            track["height_max_mm"] = max(track["height_max_mm"], item.height_max_mm)

            item.track_id = best_id
            item.age_frames = track["age"]

        for track_id in [
            track_id for track_id, track in self._tracks.items()
            if track["missing"] > self.forget_after
        ]:
            del self._tracks[track_id]

        return objects

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def peak_height(self, track_id: int) -> float | None:
        track = self._tracks.get(track_id)
        return float(track["height_max_mm"]) if track else None


# MARK: - Whole-plate summary


@dataclass
class PlateState:
    """The per-frame answer to "what is on the plate right now?"."""

    objects: list[PlateObject]
    coverage: float
    occupied_mm2: float
    tallest_mm: float
    total_volume_mm3: float
    object_count: int

    def as_dict(self) -> dict:
        return {
            "object_count": self.object_count,
            "occupied_mm2": round(self.occupied_mm2, 1),
            "tallest_mm": round(self.tallest_mm, 2),
            "total_volume_mm3": round(self.total_volume_mm3, 1),
            "map_coverage": round(self.coverage, 4),
            "objects": [item.as_dict() for item in self.objects],
        }


def summarise(height_map: HeightMap, objects: list[PlateObject]) -> PlateState:
    return PlateState(
        objects=objects,
        coverage=height_map.coverage,
        occupied_mm2=float(sum(item.footprint_mm2 for item in objects)),
        tallest_mm=float(max((item.height_max_mm for item in objects), default=0.0)),
        total_volume_mm3=float(sum(item.volume_mm3 for item in objects)),
        object_count=len(objects),
    )


class PlateMapper:
    """Ties it together: depth frame in, measured plate state out."""

    def __init__(self, calibration: PlateCalibration,
                 threshold_mm: float = DEFAULT_HEIGHT_THRESHOLD_MM,
                 min_footprint_mm2: float = DEFAULT_MIN_FOOTPRINT_MM2,
                 track: bool = True) -> None:
        self.calibration = calibration
        self.threshold_mm = threshold_mm
        self.min_footprint_mm2 = min_footprint_mm2
        self.tracker = ObjectTracker() if track else None
        self.last_height_map: HeightMap | None = None

    def process(self, depth_mm: np.ndarray) -> PlateState:
        height_map = build_height_map(depth_mm, self.calibration)
        self.last_height_map = height_map
        objects = segment_objects(
            height_map, self.calibration,
            threshold_mm=self.threshold_mm,
            min_footprint_mm2=self.min_footprint_mm2,
        )
        if self.tracker is not None:
            objects = self.tracker.update(objects)
        return summarise(height_map, objects)
