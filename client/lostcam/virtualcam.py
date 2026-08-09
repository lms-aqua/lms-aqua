"""Virtual camera sinks.

LostCam does not ship a camera driver. It writes into the virtual cameras that
already exist on the system — ``v4l2loopback`` on Linux, the OBS Virtual Camera
(or Unity Capture) on Windows and macOS — via ``pyvirtualcam``.

``pyvirtualcam`` is imported lazily so that the rest of the package, and the
whole test suite, work on a machine with no virtual camera at all.
"""

from __future__ import annotations

import sys
from typing import Protocol

import numpy as np


class VirtualCameraError(Exception):
    """No usable virtual camera backend, or it rejected the frame size."""


class Sink(Protocol):
    """Anything the pipeline can push RGB frames into."""

    width: int
    height: int

    def send(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


def default_backend() -> str:
    """The backend appropriate to this OS."""
    if sys.platform.startswith("linux"):
        return "v4l2loopback"
    if sys.platform in ("win32", "cygwin", "darwin"):
        return "obs"
    raise VirtualCameraError(f"no known virtual camera backend for {sys.platform}")


def install_hint(backend: str | None = None) -> str:
    """A actionable message for when opening the camera fails."""
    try:
        backend = backend or default_backend()
    except VirtualCameraError:
        backend = ""
    if backend == "v4l2loopback":
        return (
            "Linux needs the v4l2loopback kernel module:\n"
            "  sudo apt install v4l2loopback-dkms\n"
            "  sudo modprobe v4l2loopback devices=1 card_label=LostCam "
            "exclusive_caps=1\n"
            "exclusive_caps=1 makes browsers and Chrome-based apps accept the "
            "device. Check it appeared with: ls /dev/video*"
        )
    if backend in ("obs", "unitycapture"):
        return (
            "Windows/macOS use the OBS Virtual Camera, which ships with OBS "
            "Studio 26.0+.\n"
            "Install OBS, then start it once and click 'Start Virtual Camera' "
            "so the device is registered. LostCam can then write to it with "
            "OBS closed.\n"
            "Note: OBS exposes only one virtual camera instance, so LostCam "
            "and OBS' own output cannot both use it at once — install Unity "
            "Capture and pass --backend unitycapture if you need both."
        )
    return "Install a virtual camera: v4l2loopback on Linux, OBS Studio elsewhere."


class VirtualCamera:
    """Adapter over ``pyvirtualcam.Camera``."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        backend: str | None = None,
        device: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.backend = backend or default_backend()

        try:
            import pyvirtualcam
        except ImportError as exc:  # pragma: no cover - depends on install
            raise VirtualCameraError(
                "pyvirtualcam is not installed. Install the client with "
                "'pip install -e client[vcam]'."
            ) from exc

        kwargs = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "backend": self.backend,
            "fmt": pyvirtualcam.PixelFormat.RGB,
            "print_fps": False,
        }
        if device:
            kwargs["device"] = device
        try:
            self._camera = pyvirtualcam.Camera(**kwargs)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise VirtualCameraError(
                f"could not open a virtual camera ({exc}).\n\n{install_hint(self.backend)}"
            ) from exc

    @property
    def device(self) -> str:
        return getattr(self._camera, "device", "unknown")

    def send(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise VirtualCameraError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, camera expects "
                f"{self.width}x{self.height}"
            )
        self._camera.send(frame)

    def close(self) -> None:
        try:
            self._camera.close()
        except Exception:  # pragma: no cover - best effort on teardown
            pass

    def __enter__(self) -> VirtualCamera:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class NullSink:
    """Counts frames and throws them away.

    Backs ``--no-vcam``, which is how you verify the network and decode path
    on a machine with no virtual camera (CI, headless servers).
    """

    def __init__(self, width: int, height: int, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frames = 0
        self.last_frame: np.ndarray | None = None

    device = "null"

    def send(self, frame: np.ndarray) -> None:
        self.frames += 1
        self.last_frame = frame

    def close(self) -> None:
        return None

    def __enter__(self) -> NullSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_sink(
    width: int,
    height: int,
    fps: int = 30,
    backend: str | None = None,
    device: str | None = None,
    no_vcam: bool = False,
) -> Sink:
    """Open a virtual camera, or a discarding sink when ``no_vcam`` is set."""
    if no_vcam:
        return NullSink(width, height, fps)
    return VirtualCamera(width, height, fps, backend=backend, device=device)
