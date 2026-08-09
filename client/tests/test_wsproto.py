"""WebSocket codec tests."""

from __future__ import annotations

import base64
import struct

import pytest

from lostcam import wsproto
from lostcam.wsproto import (
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    FrameDecoder,
    WSError,
    accept_key,
    encode_frame,
    mask,
)


def client_frame(payload: bytes, opcode: int = OP_BINARY, fin: bool = True) -> bytes:
    """Build a masked client→server frame, as a real browser would."""
    header = bytearray()
    header.append((0x80 if fin else 0x00) | opcode)
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 1 << 16:
        header.append(0x80 | 126)
        header += struct.pack("!H", length)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", length)
    return bytes(header) + mask(payload, b"\x01\x02\x03\x04")


class TestAcceptKey:
    def test_matches_rfc6455_example(self):
        # The worked example from RFC 6455 §1.3.
        assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    def test_tolerates_surrounding_whitespace(self):
        assert accept_key(" dGhlIHNhbXBsZSBub25jZQ== ") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    def test_output_is_base64_sha1(self):
        assert len(base64.b64decode(accept_key("abc"))) == 20


class TestEncode:
    def test_short_payload_header(self):
        frame = encode_frame(b"hi", OP_TEXT)
        assert frame == b"\x81\x02hi"

    def test_medium_payload_uses_16_bit_length(self):
        frame = encode_frame(b"x" * 200)
        assert frame[0] == 0x82
        assert frame[1] == 126
        assert struct.unpack("!H", frame[2:4])[0] == 200

    def test_large_payload_uses_64_bit_length(self):
        frame = encode_frame(b"x" * 70000)
        assert frame[1] == 127
        assert struct.unpack("!Q", frame[2:10])[0] == 70000

    def test_server_frames_are_not_masked(self):
        assert encode_frame(b"data")[1] & 0x80 == 0

    def test_close_carries_code(self):
        frame = wsproto.encode_close(1002, "nope")
        assert frame[0] == 0x88
        assert struct.unpack("!H", frame[2:4])[0] == 1002


class TestDecode:
    def test_unmasks_binary_frame(self):
        decoder = FrameDecoder()
        messages = decoder.feed(client_frame(b"\xff\xd8jpeg\xff\xd9"))
        assert len(messages) == 1
        assert messages[0].is_binary
        assert messages[0].payload == b"\xff\xd8jpeg\xff\xd9"

    def test_text_frame_decodes_utf8(self):
        decoder = FrameDecoder()
        (message,) = decoder.feed(client_frame(b'{"type":"hello"}', OP_TEXT))
        assert message.is_text
        assert message.text() == '{"type":"hello"}'

    def test_several_frames_in_one_read(self):
        decoder = FrameDecoder()
        stream = client_frame(b"one") + client_frame(b"two") + client_frame(b"three")
        assert [m.payload for m in decoder.feed(stream)] == [b"one", b"two", b"three"]

    def test_frame_split_across_reads(self):
        frame = client_frame(b"payload-of-some-length")
        for split in range(1, len(frame)):
            decoder = FrameDecoder()
            got = decoder.feed(frame[:split]) + decoder.feed(frame[split:])
            assert [m.payload for m in got] == [b"payload-of-some-length"]

    def test_16_bit_length_round_trips(self):
        payload = b"y" * 5000
        decoder = FrameDecoder()
        (message,) = decoder.feed(client_frame(payload))
        assert message.payload == payload

    def test_64_bit_length_round_trips(self):
        payload = b"z" * 70000
        decoder = FrameDecoder()
        (message,) = decoder.feed(client_frame(payload))
        assert message.payload == payload

    def test_fragmented_message_is_reassembled(self):
        decoder = FrameDecoder()
        first = client_frame(b"part-one", OP_BINARY, fin=False)
        rest = client_frame(b"-part-two", wsproto.OP_CONT, fin=True)
        messages = decoder.feed(first + rest)
        assert [m.payload for m in messages] == [b"part-one-part-two"]

    def test_control_frames_pass_through(self):
        decoder = FrameDecoder()
        (message,) = decoder.feed(client_frame(b"", OP_PING))
        assert message.opcode == OP_PING

    def test_close_frame_reported(self):
        decoder = FrameDecoder()
        (message,) = decoder.feed(client_frame(struct.pack("!H", 1000), OP_CLOSE))
        assert message.opcode == OP_CLOSE

    def test_partial_header_returns_nothing(self):
        assert FrameDecoder().feed(b"\x82") == []


class TestDecodeErrors:
    def test_unmasked_client_frame_rejected(self):
        with pytest.raises(WSError, match="not masked"):
            FrameDecoder().feed(encode_frame(b"unmasked"))

    def test_client_side_decoder_accepts_unmasked_server_frames(self):
        """A server must NOT mask, so the client direction must allow it."""
        decoder = FrameDecoder(require_mask=False)
        (message,) = decoder.feed(encode_frame(b"from-server", OP_TEXT))
        assert message.text() == "from-server"

    def test_client_side_decoder_still_accepts_masked_frames(self):
        decoder = FrameDecoder(require_mask=False)
        (message,) = decoder.feed(client_frame(b"masked-anyway"))
        assert message.payload == b"masked-anyway"

    def test_fragmented_control_frame_rejected(self):
        with pytest.raises(WSError, match="must not be fragmented"):
            FrameDecoder().feed(client_frame(b"", OP_PING, fin=False))

    def test_stray_continuation_rejected(self):
        with pytest.raises(WSError, match="nothing to continue"):
            FrameDecoder().feed(client_frame(b"orphan", wsproto.OP_CONT))

    def test_unknown_opcode_rejected(self):
        with pytest.raises(WSError, match="unsupported opcode"):
            FrameDecoder().feed(client_frame(b"", 0x5))

    def test_oversized_frame_rejected(self):
        decoder = FrameDecoder(max_message_bytes=64)
        with pytest.raises(WSError, match="over cap"):
            decoder.feed(client_frame(b"x" * 200))
