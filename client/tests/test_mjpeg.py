"""MJPEG parser tests.

These lean on the awkward cases from docs/PROTOCOL.md §3 rather than the happy
path, because the happy path is not what breaks against real senders.
"""

from __future__ import annotations

import pytest

from lostcam.mjpeg import (
    EOI,
    SOI,
    MJPEGError,
    MJPEGParser,
    parse_boundary,
)

BOUNDARY = "lostcamframe"


def fake_jpeg(marker: bytes = b"body") -> bytes:
    return SOI + marker + EOI


def multipart(frames: list[bytes], boundary: str = BOUNDARY, length: bool = True) -> bytes:
    out = b""
    for frame in frames:
        header = f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
        if length:
            header += f"Content-Length: {len(frame)}\r\n"
        header += "\r\n"
        out += header.encode("ascii") + frame + b"\r\n"
    return out


class TestParseBoundary:
    def test_extracts_boundary(self):
        assert parse_boundary("multipart/x-mixed-replace; boundary=abc") == "abc"

    def test_strips_quotes_and_dashes(self):
        assert parse_boundary('multipart/x-mixed-replace; boundary="--abc"') == "abc"

    def test_case_insensitive_key_and_whitespace(self):
        assert parse_boundary("multipart/x-mixed-replace;  BOUNDARY = xy ") == "xy"

    @pytest.mark.parametrize("value", [None, "", "image/jpeg", "multipart/x-mixed-replace"])
    def test_missing_boundary_is_none(self, value):
        assert parse_boundary(value) is None


class TestMultipartParsing:
    def test_single_frame(self):
        parser = MJPEGParser(BOUNDARY)
        frame = fake_jpeg()
        assert parser.feed(multipart([frame])) == [frame]

    def test_several_frames_in_one_read(self):
        parser = MJPEGParser(BOUNDARY)
        frames = [fake_jpeg(b"a"), fake_jpeg(b"bb"), fake_jpeg(b"ccc")]
        assert parser.feed(multipart(frames)) == frames

    def test_frame_split_across_every_possible_boundary(self):
        """A frame arrives in pieces; no split may produce a partial frame."""
        frame = fake_jpeg(b"payload-that-is-a-bit-longer")
        stream = multipart([frame])
        for split in range(1, len(stream)):
            parser = MJPEGParser(BOUNDARY)
            got = parser.feed(stream[:split]) + parser.feed(stream[split:])
            assert got == [frame], f"failed when split at {split}"

    def test_byte_at_a_time(self):
        parser = MJPEGParser(BOUNDARY)
        frame = fake_jpeg(b"drip")
        collected = []
        for index in range(len(multipart([frame]))):
            collected += parser.feed(multipart([frame])[index : index + 1])
        assert collected == [frame]

    def test_tolerates_preamble_before_first_boundary(self):
        parser = MJPEGParser(BOUNDARY)
        frame = fake_jpeg()
        stream = b"junk preamble from a chatty server\r\n" + multipart([frame])
        assert parser.feed(stream) == [frame]

    def test_missing_content_length_falls_back_to_markers(self):
        parser = MJPEGParser(BOUNDARY)
        frames = [fake_jpeg(b"one"), fake_jpeg(b"two")]
        assert parser.feed(multipart(frames, length=False)) == frames

    def test_wrong_content_length_is_salvaged(self):
        """A sender that lies about the length must not corrupt the decoder."""
        frame = fake_jpeg(b"honest")
        header = (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame) + 4}\r\n\r\n"
        ).encode("ascii")
        parser = MJPEGParser(BOUNDARY)
        got = parser.feed(header + b"\x00\x00" + frame + b"\r\n" + b"xx")
        assert got == [frame]

    def test_boundary_with_leading_dashes_in_constructor(self):
        parser = MJPEGParser("--" + BOUNDARY)
        frame = fake_jpeg()
        assert parser.feed(multipart([frame])) == [frame]

    def test_no_frame_yet_returns_empty(self):
        parser = MJPEGParser(BOUNDARY)
        assert parser.feed(b"--" + BOUNDARY.encode()) == []


class TestScanMode:
    """No boundary known: fall back to SOI/EOI scanning."""

    def test_extracts_consecutive_frames(self):
        parser = MJPEGParser(None)
        frames = [fake_jpeg(b"x"), fake_jpeg(b"yy")]
        assert parser.feed(b"".join(frames)) == frames

    def test_ignores_junk_between_frames(self):
        parser = MJPEGParser(None)
        frame = fake_jpeg(b"z")
        assert parser.feed(b"noise" + frame + b"trailing") == [frame]

    def test_incomplete_frame_is_withheld(self):
        parser = MJPEGParser(None)
        assert parser.feed(SOI + b"half") == []
        assert parser.feed(b"rest" + EOI) == [SOI + b"halfrest" + EOI]

    def test_junk_only_does_not_accumulate(self):
        """Bytes that cannot start a frame must be discarded, not buffered."""
        parser = MJPEGParser(None)
        for _ in range(50):
            parser.feed(b"\x00" * 10_000)
        assert parser.buffered <= 2


class TestLimits:
    def test_buffer_cap_raises(self):
        parser = MJPEGParser(None, max_frame_bytes=1024)
        with pytest.raises(MJPEGError, match="without a complete frame"):
            parser.feed(SOI + b"\x11" * 4096)

    def test_declared_length_over_cap_raises(self):
        header = (
            f"--{BOUNDARY}\r\nContent-Length: 999999\r\n\r\n"
        ).encode("ascii")
        parser = MJPEGParser(BOUNDARY, max_frame_bytes=1024)
        with pytest.raises(MJPEGError, match="over cap"):
            parser.feed(header)

    def test_many_frames_in_one_read_is_not_mistaken_for_runaway(self):
        frame = fake_jpeg(b"q" * 200)
        parser = MJPEGParser(BOUNDARY, max_frame_bytes=len(frame) * 3)
        got = parser.feed(multipart([frame] * 20))
        assert len(got) == 20

    def test_zero_cap_rejected(self):
        with pytest.raises(ValueError):
            MJPEGParser(None, max_frame_bytes=0)

    def test_non_jpeg_part_with_length_raises(self):
        payload = b"this is not a jpeg at all"
        header = (
            f"--{BOUNDARY}\r\nContent-Length: {len(payload)}\r\n\r\n"
        ).encode("ascii")
        parser = MJPEGParser(BOUNDARY)
        with pytest.raises(MJPEGError, match="not a JPEG"):
            parser.feed(header + payload)
