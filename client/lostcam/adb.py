"""USB mode via adb port forwarding.

USB is the answer to flaky Wi-Fi, and it is not a different protocol: adb makes
the phone's listening port appear on the desktop's loopback interface, so the
ordinary pull-mode client connects to 127.0.0.1 and nothing else changes. It is
also the private option — the port is never exposed to the network.

Every adb invocation passes an argument list, never a shell string, so device
serials and ports cannot be turned into shell syntax.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

DEFAULT_TIMEOUT = 15.0


class AdbError(Exception):
    """adb is missing, or returned a non-zero exit status."""


@dataclass(frozen=True)
class Device:
    serial: str
    state: str

    @property
    def usable(self) -> bool:
        return self.state == "device"


def adb_path() -> str:
    """Locate the adb binary, or explain how to get it."""
    path = shutil.which("adb")
    if not path:
        raise AdbError(
            "adb was not found on PATH. Install Android platform-tools:\n"
            "  Windows: winget install Google.PlatformTools\n"
            "  Debian/Ubuntu: sudo apt install android-tools-adb\n"
            "  Arch: sudo pacman -S android-tools\n"
            "  macOS: brew install android-platform-tools"
        )
    return path


def _run(args: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    command = [adb_path(), *args]
    try:
        result = subprocess.run(  # noqa: S603 - argument list, never a shell
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdbError(f"adb {' '.join(args)} timed out after {timeout}s") from exc
    except OSError as exc:
        raise AdbError(f"could not run adb: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AdbError(f"adb {' '.join(args)} failed: {detail or result.returncode}")
    return result.stdout


def parse_devices(output: str) -> list[Device]:
    """Parse ``adb devices`` output.

    Split out from the subprocess call so it can be tested against real-world
    output, including the ``unauthorized`` and ``offline`` states that trip
    people up.
    """
    devices: list[Device] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            devices.append(Device(serial=parts[0], state=parts[1]))
    return devices


def list_devices() -> list[Device]:
    return parse_devices(_run(["devices"]))


def _serial_args(serial: str | None) -> list[str]:
    return ["-s", serial] if serial else []


def forward(local_port: int, remote_port: int, serial: str | None = None) -> None:
    """Map ``127.0.0.1:local_port`` on this machine to the phone's port."""
    _validate_port(local_port)
    _validate_port(remote_port)
    _run([*_serial_args(serial), "forward", f"tcp:{local_port}", f"tcp:{remote_port}"])


def remove_forward(local_port: int, serial: str | None = None) -> None:
    _validate_port(local_port)
    try:
        _run([*_serial_args(serial), "forward", "--remove", f"tcp:{local_port}"])
    except AdbError:
        # Tearing down a forward that is already gone is not a failure.
        pass


def pick_serial(devices: list[Device], requested: str | None = None) -> str:
    """Choose which phone to use, with a clear error when it is ambiguous."""
    if requested:
        for device in devices:
            if device.serial == requested:
                if not device.usable:
                    raise AdbError(
                        f"device {requested} is in state '{device.state}' — "
                        "unlock the phone and accept the USB debugging prompt"
                    )
                return requested
        raise AdbError(f"device {requested} is not attached")

    usable = [d for d in devices if d.usable]
    if not usable:
        unusable = ", ".join(f"{d.serial} ({d.state})" for d in devices)
        raise AdbError(
            "no usable device over USB. "
            + (
                f"Attached but not ready: {unusable}. Unlock the phone and accept "
                "the USB debugging prompt."
                if devices
                else "Connect the phone by USB and enable USB debugging in "
                "Developer options."
            )
        )
    if len(usable) > 1:
        serials = ", ".join(d.serial for d in usable)
        raise AdbError(f"several devices attached ({serials}); pass --serial")
    return usable[0].serial


class Forward:
    """Context manager that sets up a forward and always tears it down."""

    def __init__(
        self, local_port: int, remote_port: int, serial: str | None = None
    ) -> None:
        self.local_port = local_port
        self.remote_port = remote_port
        self.serial = serial

    def __enter__(self) -> Forward:
        self.serial = pick_serial(list_devices(), self.serial)
        forward(self.local_port, self.remote_port, self.serial)
        return self

    def __exit__(self, *exc_info: object) -> None:
        remove_forward(self.local_port, self.serial)


def _validate_port(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"invalid TCP port: {port!r}")
