"""UDP discovery, so you do not have to read an IP off the phone's screen.

The desktop broadcasts a probe; senders reply with their ``/info`` payload plus
the port they serve video on. Advisory only — an explicit address always wins,
and discovery failing never blocks a manual connection.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

DISCOVERY_PORT = 4748
PROBE = b"LOSTCAM_DISCOVER_V1"
MAX_REPLY_BYTES = 8192


@dataclass(frozen=True)
class Found:
    host: str
    port: int
    device: str
    platform: str
    info: dict

    @property
    def label(self) -> str:
        return f"{self.device} ({self.platform}) at {self.host}:{self.port}"


def parse_reply(payload: bytes, host: str) -> Found | None:
    """Turn a reply datagram into a ``Found``, or ``None`` if it is not ours.

    Anything on a broadcast port may answer, so this validates rather than
    trusts: unparseable or foreign payloads are ignored, not raised.
    """
    try:
        info = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(info, dict) or info.get("product") != "LostCam":
        return None
    port = info.get("port", 4747)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        return None
    return Found(
        host=host,
        port=port,
        device=str(info.get("device", "unknown device")),
        platform=str(info.get("platform", "unknown")),
        info=info,
    )


def discover(
    timeout: float = 1.5,
    port: int = DISCOVERY_PORT,
    broadcast_addr: str = "255.255.255.255",
) -> list[Found]:
    """Broadcast a probe and collect replies for ``timeout`` seconds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.2)
    found: dict[tuple[str, int], Found] = {}
    try:
        try:
            sock.sendto(PROBE, (broadcast_addr, port))
        except OSError:
            # Some networks/containers refuse broadcast. Not fatal.
            return []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                payload, addr = sock.recvfrom(MAX_REPLY_BYTES)
            except TimeoutError:
                continue
            except OSError:
                break
            entry = parse_reply(payload, addr[0])
            if entry is not None:
                found[(entry.host, entry.port)] = entry
    finally:
        sock.close()
    return sorted(found.values(), key=lambda f: (f.host, f.port))


class Responder:
    """Answers discovery probes on behalf of a sender.

    The iOS app has its own implementation; this one backs the ``mocksender``
    test double and lets the discovery path be tested over loopback.
    """

    def __init__(
        self,
        info: dict,
        port: int = DISCOVERY_PORT,
        bind_host: str = "",
    ) -> None:
        self.info = dict(info)
        self.port = port
        self.bind_host = bind_host
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.2)
        sock.bind((self.bind_host, self.port))
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._sock is not None
        reply = json.dumps(self.info).encode("utf-8")
        while not self._stop.is_set():
            try:
                payload, addr = self._sock.recvfrom(MAX_REPLY_BYTES)
            except TimeoutError:
                continue
            except OSError:
                return
            if payload.strip() == PROBE:
                try:
                    self._sock.sendto(reply, addr)
                except OSError:
                    continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> Responder:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def local_addresses() -> list[str]:
    """Best-effort list of this machine's LAN addresses, for printing URLs."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    # getaddrinfo often only yields loopback; a UDP connect reveals the
    # address the kernel would actually route from, without sending anything.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in addresses if not a.startswith("127."))
