"""The bridge: take the phone's data channel and re-emit it locally.

Connecting straight to the phone's IP works and needs nothing here — `curl -N
http://phone:4747/data` is a valid client. The bridge exists for the consumers
that cannot do that: a Unity or TouchDesigner patch that wants UDP on a local
port, a Blender rig that speaks OSC, a browser dashboard that needs a
WebSocket, or an analysis run that wants a CSV afterwards.

Every sink is optional and independent; one failing must not take the others
down, because a bridge that dies when a UDP listener goes away is not a bridge.
"""

from __future__ import annotations

import csv
import json
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import osc
from .datastream import Sample

# Fields worth turning into individual OSC messages, in the order a receiver
# most often wants them.
OSC_SCALAR_LIMIT = 256


class Sink:
    """Base class: a destination for samples."""

    name = "sink"

    def handle(self, sample: Sample) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class SinkStats:
    delivered: int = 0
    errors: int = 0
    last_error: str = ""


class UDPJSONSink(Sink):
    """Fire each sample as one JSON datagram.

    Datagram-per-sample, deliberately: a Unity/TouchDesigner receiver reading
    one packet gets one complete record with no framing to implement. Samples
    too large for a datagram are dropped with a count rather than fragmented.
    """

    name = "udp-json"
    MAX_DATAGRAM = 65000

    def __init__(self, host: str = "127.0.0.1", port: int = 9001) -> None:
        self.host = host
        self.port = port
        self.stats = SinkStats()
        self.oversized = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def handle(self, sample: Sample) -> None:
        payload = json.dumps(sample.raw, separators=(",", ":")).encode("utf-8")
        if len(payload) > self.MAX_DATAGRAM:
            self.oversized += 1
            return
        try:
            self._sock.sendto(payload, (self.host, self.port))
            self.stats.delivered += 1
        except OSError as exc:
            # Nothing listening on a UDP port is normal and not an error worth
            # stopping for; record it and carry on.
            self.stats.errors += 1
            self.stats.last_error = str(exc)

    def close(self) -> None:
        self._sock.close()


class OSCSink(Sink):
    """Send each sample as an OSC bundle of scalar messages.

    Addresses are ``/lostcam/<channel>/<flattened-key>``, so a face blendshape
    lands at ``/lostcam/ar/face/blend/jawOpen`` — directly bindable in most
    receivers without a mapping table.
    """

    name = "osc"

    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self.host = host
        self.port = port
        self.stats = SinkStats()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def build(self, sample: Sample) -> bytes:
        """Encode a sample. Split out so tests can inspect the bytes."""
        base = f"/lostcam/{sample.channel}"
        messages = []
        for key, value in sample.flatten().items():
            if key in ("t", "seq", "ch") or value is None:
                continue
            if len(messages) >= OSC_SCALAR_LIMIT:
                break
            try:
                messages.append(osc.message(f"{base}/{key}", value))
            except osc.OSCError:
                continue
        messages.append(osc.message(f"{base}/t", int(sample.t)))
        return osc.bundle(messages)

    def handle(self, sample: Sample) -> None:
        try:
            self._sock.sendto(self.build(sample), (self.host, self.port))
            self.stats.delivered += 1
        except (OSError, osc.OSCError) as exc:
            self.stats.errors += 1
            self.stats.last_error = str(exc)

    def close(self) -> None:
        self._sock.close()


class JSONLSink(Sink):
    """Append every sample to a JSONL file, losing nothing."""

    name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.stats = SinkStats()

    def handle(self, sample: Sample) -> None:
        self._handle.write(json.dumps(sample.raw, separators=(",", ":")) + "\n")
        self.stats.delivered += 1

    def close(self) -> None:
        try:
            self._handle.flush()
            self._handle.close()
        except OSError:
            pass


class CSVSink(Sink):
    """One CSV per channel, columns discovered from the first sample.

    Per-channel files rather than one wide file: the channels have almost no
    columns in common, so a single table would be mostly empty cells. Columns
    are fixed by the first sample of each channel, and later samples with new
    keys get those keys ignored — recorded in ``skipped_columns`` so the
    omission is visible rather than silent.
    """

    name = "csv"

    # Blendshapes are sparse — only non-zero coefficients are sent — so the
    # first sample of a channel is a bad guide to its columns. Buffer a warmup
    # window and take the union of keys seen across it before writing a header.
    WARMUP_SAMPLES = 60
    EXTRA_COLUMN = "extra_json"

    def __init__(
        self,
        directory: str | Path,
        prefix: str = "",
        warmup: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.warmup = self.WARMUP_SAMPLES if warmup is None else warmup
        self.stats = SinkStats()
        # Keys that appeared after the header was fixed. Retained in the
        # extra_json column, so this is a note about column layout, not loss.
        self.late_columns: dict[str, set[str]] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._handles: dict[str, object] = {}
        self._pending: dict[str, list[dict]] = {}

    def path_for(self, channel: str) -> Path:
        safe = channel.replace("/", "_").replace(".", "_")
        return self.directory / f"{self.prefix}{safe}.csv"

    def handle(self, sample: Sample) -> None:
        row = sample.flatten()
        row.pop("ch", None)
        channel = sample.channel

        writer = self._writers.get(channel)
        if writer is None:
            buffered = self._pending.setdefault(channel, [])
            buffered.append(row)
            if len(buffered) >= self.warmup:
                self._open(channel)
            self.stats.delivered += 1
            return

        self._write(channel, writer, row)
        self.stats.delivered += 1

    def _open(self, channel: str) -> None:
        """Create the file with a header covering every key seen so far."""
        buffered = self._pending.pop(channel, [])
        columns: list[str] = []
        for row in buffered:
            for key in row:
                if key not in columns:
                    columns.append(key)
        columns.append(self.EXTRA_COLUMN)

        handle = self.path_for(channel).open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        self._writers[channel] = writer
        self._handles[channel] = handle
        for row in buffered:
            self._write(channel, writer, row)

    def _write(self, channel: str, writer: csv.DictWriter, row: dict) -> None:
        known = set(writer.fieldnames)
        unknown = set(row) - known
        if unknown:
            # Keep the values rather than dropping them: a CSV that silently
            # omits a column misrepresents the recording.
            self.late_columns.setdefault(channel, set()).update(unknown)
            extra = {key: row[key] for key in sorted(unknown)}
            row = {key: value for key, value in row.items() if key in known}
            row[self.EXTRA_COLUMN] = json.dumps(extra, separators=(",", ":"))
        writer.writerow(row)

    def close(self) -> None:
        # Flush any channel that never reached the warmup threshold.
        for channel in list(self._pending):
            if self._pending[channel]:
                self._open(channel)
        for handle in self._handles.values():
            try:
                handle.flush()  # type: ignore[attr-defined]
                handle.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        self._handles.clear()
        self._writers.clear()


class CallbackSink(Sink):
    """Adapter for arbitrary Python consumers, and for the WebSocket fan-out."""

    name = "callback"

    def __init__(self, callback: callable, name: str = "callback") -> None:
        self.callback = callback
        self.name = name
        self.stats = SinkStats()

    def handle(self, sample: Sample) -> None:
        try:
            self.callback(sample)
            self.stats.delivered += 1
        except Exception as exc:  # a consumer's bug must not kill the bridge
            self.stats.errors += 1
            self.stats.last_error = str(exc)


@dataclass
class Bridge:
    """Fan one sample stream out to many sinks, isolating their failures."""

    sinks: list[Sink] = field(default_factory=list)
    samples: int = 0
    channels: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, sink: Sink) -> Sink:
        self.sinks.append(sink)
        return sink

    def handle(self, sample: Sample) -> None:
        with self._lock:
            self.samples += 1
            self.channels.add(sample.channel)
        for sink in self.sinks:
            try:
                sink.handle(sample)
            except Exception:
                # Already counted inside most sinks; this is the backstop that
                # guarantees one bad sink cannot starve the others.
                stats = getattr(sink, "stats", None)
                if stats is not None:
                    stats.errors += 1

    def summary(self) -> str:
        parts = [f"{self.samples} samples"]
        if self.channels:
            parts.append("channels: " + ", ".join(sorted(self.channels)))
        for sink in self.sinks:
            stats = getattr(sink, "stats", None)
            if stats is not None:
                detail = f"{sink.name}={stats.delivered}"
                if stats.errors:
                    detail += f" ({stats.errors} error(s))"
                parts.append(detail)
        return "; ".join(parts)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass
