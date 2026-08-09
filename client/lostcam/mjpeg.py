"""Incremental MJPEG stream parsing.

Producers of ``multipart/x-mixed-replace`` are inconsistent in practice: some
omit ``Content-Length``, some pad the boundary differently, some emit preamble
bytes before the first boundary. This parser is deliberately forgiving in the
ways :doc:`../../docs/PROTOCOL.md` §3 requires, while refusing to grow its
buffer without bound.
"""

from __future__ import annotations

SOI = b"\xff\xd8"  # JPEG start of image
EOI = b"\xff\xd9"  # JPEG end of image

DEFAULT_MAX_FRAME_BYTES = 16 * 1024 * 1024


class MJPEGError(Exception):
    """Raised when the stream cannot be parsed, or blows the buffer cap."""


def parse_boundary(content_type: str | None) -> str | None:
    """Pull the multipart boundary out of a Content-Type header value.

    Returns ``None`` when the header is missing or carries no boundary, which
    puts the parser into marker-scanning mode.
    """
    if not content_type:
        return None
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "boundary":
            value = value.strip().strip('"')
            # Some servers already include the leading dashes in the header.
            return value.lstrip("-") or None
    return None


class MJPEGParser:
    """Feed bytes in, get complete JPEG frames out.

    The parser never returns a partial frame. When a boundary is known it uses
    the part headers (and ``Content-Length`` when present); otherwise it falls
    back to scanning for JPEG SOI/EOI markers.
    """

    def __init__(
        self,
        boundary: str | None = None,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.boundary = boundary.lstrip("-") if boundary else None
        self.max_frame_bytes = max_frame_bytes
        self._buf = bytearray()
        self._pending_length: int | None = None
        self._in_part = False

    # -- public API ----------------------------------------------------------

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume a chunk of stream bytes and return any completed frames."""
        if chunk:
            self._buf += chunk
        frames: list[bytes] = []
        while True:
            frame = self._next_frame()
            if frame is None:
                break
            frames.append(frame)
        # Only enforce the cap once no more frames can be extracted, so a read
        # that legitimately contains many frames is not mistaken for a runaway.
        if len(self._buf) > self.max_frame_bytes:
            raise MJPEGError(
                f"buffered {len(self._buf)} bytes without a complete frame "
                f"(cap {self.max_frame_bytes})"
            )
        return frames

    @property
    def buffered(self) -> int:
        """Bytes currently held awaiting a complete frame. Useful in tests."""
        return len(self._buf)

    # -- internals -----------------------------------------------------------

    def _next_frame(self) -> bytes | None:
        if self.boundary is not None:
            return self._next_multipart_frame()
        return self._scan_frame()

    def _next_multipart_frame(self) -> bytes | None:
        if not self._in_part:
            marker = b"--" + self.boundary.encode("ascii", "strict")
            index = self._buf.find(marker)
            if index < 0:
                # Keep a tail in case the boundary straddles two reads.
                self._trim_to_tail(len(marker))
                return None
            header_start = index + len(marker)
            header_end = self._buf.find(b"\r\n\r\n", header_start)
            if header_end < 0:
                if header_start > 0:
                    del self._buf[:index]
                return None
            raw_headers = bytes(self._buf[header_start:header_end])
            del self._buf[: header_end + 4]
            self._pending_length = _content_length(raw_headers)
            self._in_part = True

        if self._pending_length is not None:
            if self._pending_length > self.max_frame_bytes:
                raise MJPEGError(
                    f"part declares {self._pending_length} bytes, "
                    f"over cap {self.max_frame_bytes}"
                )
            if len(self._buf) < self._pending_length:
                return None
            frame = bytes(self._buf[: self._pending_length])
            del self._buf[: self._pending_length]
            self._in_part = False
            self._pending_length = None
            if not frame.startswith(SOI):
                # Declared length was wrong; recover by scanning instead of
                # handing a corrupt buffer to the decoder.
                salvaged = _extract_jpeg(frame)
                if salvaged is None:
                    raise MJPEGError("part payload is not a JPEG frame")
                return salvaged
            return frame

        # No Content-Length on this part: scan for the frame, then expect the
        # next boundary to follow it.
        frame = self._scan_frame()
        if frame is not None:
            self._in_part = False
        return frame

    def _scan_frame(self) -> bytes | None:
        start = self._buf.find(SOI)
        if start < 0:
            self._trim_to_tail(1)
            return None
        end = self._buf.find(EOI, start + 2)
        if end < 0:
            if start > 0:
                del self._buf[:start]
            return None
        frame = bytes(self._buf[start : end + 2])
        del self._buf[: end + 2]
        return frame

    def _trim_to_tail(self, keep: int) -> None:
        """Discard junk we know cannot begin a frame, keeping a small tail."""
        if len(self._buf) > keep:
            del self._buf[: len(self._buf) - keep]


def _content_length(raw_headers: bytes) -> int | None:
    for line in raw_headers.split(b"\r\n"):
        key, sep, value = line.partition(b":")
        if not sep:
            continue
        if key.strip().lower() == b"content-length":
            try:
                length = int(value.strip())
            except ValueError:
                return None
            return length if length >= 0 else None
    return None


def _extract_jpeg(blob: bytes) -> bytes | None:
    start = blob.find(SOI)
    if start < 0:
        return None
    end = blob.find(EOI, start + 2)
    if end < 0:
        return None
    return blob[start : end + 2]
