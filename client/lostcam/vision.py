"""Per-frame measurements for a fixed camera watching a build plate.

Two jobs, and they are the same computation:

1. **Features** a model can train on — how bright the plate is, how sharp, how
   much changed since the last frame, how tall the tallest thing on the plate is.
2. **Data-quality checks** — the same numbers catch the failures that quietly
   ruin a dataset. A blur score that collapses means the lens started hunting. A
   brightness step means auto-exposure was left on, or someone turned the room
   light on. A frame-difference spike on a camera that should be bolted down
   means the rig was knocked.

Everything here is deterministic and array-in/number-out, so it is all testable
and it all produces the same values during training as during capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


class CalibrationError(Exception):
    """The supplied calibration or region is unusable."""


# MARK: - Region of interest


@dataclass(frozen=True)
class ROI:
    """A pixel rectangle, normally the build plate.

    Restricting measurements to the plate is what makes them mean anything: a
    brightness average over the whole frame mostly measures the room, and a
    frame-difference over the whole frame mostly measures someone walking past.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CalibrationError("ROI width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise CalibrationError("ROI origin must not be negative")

    @classmethod
    def parse(cls, text: str) -> ROI:
        """Parse ``x,y,w,h``."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise CalibrationError("ROI must be given as x,y,w,h")
        try:
            values = [int(p) for p in parts]
        except ValueError as exc:
            raise CalibrationError(f"ROI values must be integers: {text!r}") from exc
        return cls(*values)

    def clipped_to(self, shape: tuple[int, ...]) -> ROI:
        """Clip to an image's bounds, so a stale ROI cannot index out of range."""
        height, width = int(shape[0]), int(shape[1])
        x = min(self.x, max(0, width - 1))
        y = min(self.y, max(0, height - 1))
        return ROI(x, y, max(1, min(self.width, width - x)),
                   max(1, min(self.height, height - y)))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        box = self.clipped_to(frame.shape)
        return frame[box.y : box.y + box.height, box.x : box.x + box.width]

    def scaled(self, factor_x: float, factor_y: float) -> ROI:
        """Rescale into another raster — the depth map is smaller than colour."""
        return ROI(int(round(self.x * factor_x)), int(round(self.y * factor_y)),
                   max(1, int(round(self.width * factor_x))),
                   max(1, int(round(self.height * factor_y))))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


# MARK: - Scale calibration


@dataclass(frozen=True)
class Calibration:
    """Millimetres per pixel, so measurements come out metric.

    Derived from something of known size in the frame — the easiest being the
    build plate itself, whose dimensions you already know (220 mm on an Ender 3,
    256 mm on a Prusa MK4, and so on).

    This is only valid for the plane it was measured on. The camera is not
    orthographic, so something 100 mm above the plate covers more pixels per
    millimetre than the plate does. Treating this as a single global scale is an
    approximation, and a fine one for a roughly top-down or shallow-angle view of
    a flat plate — but it is an approximation, and that is why it is a documented
    field rather than a hidden constant.
    """

    mm_per_pixel_x: float
    mm_per_pixel_y: float
    reference: str = "unspecified"

    def __post_init__(self) -> None:
        if self.mm_per_pixel_x <= 0 or self.mm_per_pixel_y <= 0:
            raise CalibrationError("mm-per-pixel must be positive")

    @classmethod
    def from_plate(cls, roi: ROI, plate_width_mm: float,
                   plate_height_mm: float | None = None) -> Calibration:
        """Calibrate from a plate of known size filling a known ROI."""
        if plate_width_mm <= 0:
            raise CalibrationError("plate width must be positive")
        height_mm = plate_height_mm if plate_height_mm else plate_width_mm
        return cls(
            mm_per_pixel_x=plate_width_mm / roi.width,
            mm_per_pixel_y=height_mm / roi.height,
            reference=f"plate {plate_width_mm:g}x{height_mm:g}mm over ROI "
                      f"{roi.width}x{roi.height}px",
        )

    def area_mm2(self, pixel_count: int) -> float:
        return pixel_count * self.mm_per_pixel_x * self.mm_per_pixel_y

    def length_mm(self, pixels: float) -> float:
        """Mean of the two axes; only meaningful when they are close."""
        return pixels * (self.mm_per_pixel_x + self.mm_per_pixel_y) / 2.0

    def as_dict(self) -> dict:
        return {
            "mm_per_pixel_x": round(self.mm_per_pixel_x, 6),
            "mm_per_pixel_y": round(self.mm_per_pixel_y, 6),
            "reference": self.reference,
        }


# MARK: - Frame metrics


def to_grey(frame: np.ndarray) -> np.ndarray:
    """Luma from RGB, using the Rec. 601 weights."""
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("expected an (H, W, 3) RGB frame")
    channels = frame[:, :, :3].astype(np.float32)
    return (0.299 * channels[:, :, 0]
            + 0.587 * channels[:, :, 1]
            + 0.114 * channels[:, :, 2])


def sharpness(grey: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur proxy.

    Higher is sharper. The absolute value depends on the scene, so it is only
    meaningful compared against other frames of *the same* scene, which is
    exactly the situation a fixed rig is in. A sudden drop means the lens hunted
    or something moved during the exposure.
    """
    if grey.size == 0:
        return 0.0
    # A 4-neighbour Laplacian via slicing: no scipy dependency, same result.
    interior = grey[1:-1, 1:-1]
    if interior.size == 0:
        return 0.0
    laplacian = (grey[:-2, 1:-1] + grey[2:, 1:-1]
                 + grey[1:-1, :-2] + grey[1:-1, 2:]
                 - 4.0 * interior)
    return float(laplacian.var())


def exposure_clipping(grey: np.ndarray, low: float = 4.0,
                      high: float = 251.0) -> tuple[float, float]:
    """Fractions of the region that are crushed black or blown white.

    Clipped pixels carry no recoverable detail, so a rising blown fraction means
    the shot is losing exactly the information a model needs.
    """
    if grey.size == 0:
        return (0.0, 0.0)
    total = float(grey.size)
    return (float((grey <= low).sum() / total), float((grey >= high).sum() / total))


@dataclass
class FrameMetrics:
    """Per-frame numbers, written to the manifest for every frame."""

    mean: float = 0.0
    std: float = 0.0
    sharpness: float = 0.0
    clipped_black: float = 0.0
    clipped_white: float = 0.0
    # Change since the previous frame, over the ROI. 0 on the first frame.
    diff_mean: float = 0.0
    diff_fraction: float = 0.0
    # Depth, when a depth frame was aligned to this video frame.
    depth_coverage: float | None = None
    depth_min_mm: float | None = None
    depth_max_mm: float | None = None
    depth_median_mm: float | None = None
    # Height above the plate reference plane, if one was established.
    height_max_mm: float | None = None
    height_mean_mm: float | None = None

    def as_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


class FrameAnalyser:
    """Computes per-frame metrics, holding the previous frame for differencing.

    Stateful by necessity — change detection needs a reference — so it is a class
    rather than a function, and the state it keeps is exactly one greyscale ROI.
    """

    def __init__(self, roi: ROI | None = None, calibration: Calibration | None = None,
                 diff_threshold: float = 12.0) -> None:
        self.roi = roi
        self.calibration = calibration
        self.diff_threshold = diff_threshold
        self._previous: np.ndarray | None = None
        # The plate's depth with nothing on it, used to turn distances into
        # heights. Established explicitly, never guessed.
        self.plate_reference_mm: float | None = None

    def reset(self) -> None:
        self._previous = None

    def region(self, frame: np.ndarray) -> np.ndarray:
        return self.roi.crop(frame) if self.roi else frame

    def analyse(self, frame: np.ndarray,
                depth: np.ndarray | None = None) -> FrameMetrics:
        grey = to_grey(self.region(frame))
        metrics = FrameMetrics(
            mean=float(grey.mean()) if grey.size else 0.0,
            std=float(grey.std()) if grey.size else 0.0,
            sharpness=sharpness(grey),
        )
        metrics.clipped_black, metrics.clipped_white = exposure_clipping(grey)

        if self._previous is not None and self._previous.shape == grey.shape:
            difference = np.abs(grey - self._previous)
            metrics.diff_mean = float(difference.mean())
            metrics.diff_fraction = float(
                (difference > self.diff_threshold).sum() / difference.size
            )
        self._previous = grey

        if depth is not None:
            self._add_depth_metrics(metrics, depth, frame.shape)
        return metrics

    def _add_depth_metrics(self, metrics: FrameMetrics, depth: np.ndarray,
                           colour_shape: tuple[int, ...]) -> None:
        region = depth
        if self.roi is not None:
            # The depth raster is smaller than the colour frame, so the ROI has
            # to be rescaled rather than reused.
            scale_y = depth.shape[0] / colour_shape[0]
            scale_x = depth.shape[1] / colour_shape[1]
            region = self.roi.scaled(scale_x, scale_y).crop(depth)

        valid = region[region > 0]
        metrics.depth_coverage = (
            float(valid.size / region.size) if region.size else 0.0
        )
        if valid.size == 0:
            return

        metrics.depth_min_mm = float(valid.min())
        metrics.depth_max_mm = float(valid.max())
        metrics.depth_median_mm = float(np.median(valid))

        if self.plate_reference_mm is not None:
            # Nearer than the plate means above it. Distances beyond the plate are
            # behind it and are not heights, so they are excluded rather than
            # producing negative "heights".
            heights = self.plate_reference_mm - valid.astype(np.float32)
            above = heights[heights > 0]
            if above.size:
                metrics.height_max_mm = float(above.max())
                metrics.height_mean_mm = float(above.mean())
            else:
                metrics.height_max_mm = 0.0
                metrics.height_mean_mm = 0.0

    def calibrate_plate_reference(self, depth_frames: list[np.ndarray],
                                 colour_shape: tuple[int, ...]) -> float | None:
        """Establish the empty plate's distance, for height measurements.

        Uses the median of medians across several frames: LiDAR is noisy per
        frame, and a median is unbothered by the stray zero-or-far pixels at the
        plate edges. Returns None when there is not enough valid depth to trust,
        rather than inventing a reference that would make every later height
        wrong.
        """
        medians: list[float] = []
        for depth in depth_frames:
            region = depth
            if self.roi is not None:
                scale_y = depth.shape[0] / colour_shape[0]
                scale_x = depth.shape[1] / colour_shape[1]
                region = self.roi.scaled(scale_x, scale_y).crop(depth)
            valid = region[region > 0]
            # Require half the region to have returned something.
            if valid.size and valid.size / max(1, region.size) > 0.5:
                medians.append(float(np.median(valid)))
        if len(medians) < 3:
            return None
        self.plate_reference_mm = float(np.median(medians))
        return self.plate_reference_mm
