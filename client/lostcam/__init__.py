"""LostCam — use your phone as a webcam and microphone.

Two transports feed one pipeline:

* ``pull`` — the phone serves MJPEG and the desktop connects out. This is the
  DroidCam-compatible direction, used by the iOS app.
* ``serve`` — the desktop serves a page and the phone's browser pushes frames.
  This is what makes any phone with a browser work, Android included.

Both end at a virtual camera the operating system already provides:
``v4l2loopback`` on Linux, the OBS Virtual Camera on Windows and macOS.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "PROTOCOL_VERSION", "DEFAULT_PULL_PORT", "DEFAULT_PUSH_PORT"]

PROTOCOL_VERSION = 1
DEFAULT_PULL_PORT = 4747  # matches DroidCam, so existing tooling works
DEFAULT_PUSH_PORT = 8443
