"""A minimal RFC 6455 WebSocket server codec.

Only what push mode needs: the handshake response key, unmasked server→client
frames, and a decoder for masked client→server frames including fragmentation
and control frames. No extensions, no permessage-deflate.

Kept free of I/O so it can be unit tested directly.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
from dataclasses import dataclass

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class WSError(Exception):
    """Protocol violation, or a message over the size cap."""


def accept_key(client_key: str) -> str:
    """Compute the ``Sec-WebSocket-Accept`` value for a client key."""
    digest = hashlib.sha1(client_key.strip().encode("ascii") + GUID).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, opcode: int = OP_BINARY, fin: bool = True) -> bytes:
    """Encode one server→client frame. Server frames are never masked."""
    header = bytearray()
    header.append((0x80 if fin else 0x00) | (opcode & 0x0F))
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 1 << 16:
        header.append(126)
        header += struct.pack("!H", length)
    else:
        header.append(127)
        header += struct.pack("!Q", length)
    return bytes(header) + payload


def encode_close(code: int = 1000, reason: str = "") -> bytes:
    return encode_frame(struct.pack("!H", code) + reason.encode("utf-8"), OP_CLOSE)


def mask(payload: bytes, key: bytes | None = None) -> bytes:
    """Produce a masked client→server frame body. Test helper."""
    key = key or os.urandom(4)
    masked = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return key + masked


@dataclass(frozen=True)
class Message:
    opcode: int
    payload: bytes

    @property
    def is_text(self) -> bool:
        return self.opcode == OP_TEXT

    @property
    def is_binary(self) -> bool:
        return self.opcode == OP_BINARY

    def text(self) -> str:
        return self.payload.decode("utf-8")


class FrameDecoder:
    """Incremental decoder for client→server frames.

    Reassembles fragmented data messages; yields control frames as they arrive.
    """

    def __init__(
        self,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        require_mask: bool = True,
    ) -> None:
        """``require_mask`` reflects which direction is being decoded.

        RFC 6455 §5.1 requires clients to mask and forbids servers from doing
        so, so the same decoder cannot enforce one rule for both. Server-side
        decoding (the default) rejects unmasked frames; a client reading server
        frames passes ``require_mask=False``.
        """
        self.max_message_bytes = max_message_bytes
        self.require_mask = require_mask
        self._buf = bytearray()
        self._fragments = bytearray()
        self._fragment_opcode: int | None = None

    def feed(self, chunk: bytes) -> list[Message]:
        if chunk:
            self._buf += chunk
        out: list[Message] = []
        while True:
            message = self._next()
            if message is None:
                break
            out.append(message)
        return out

    def _next(self) -> Message | None:
        buf = self._buf
        if len(buf) < 2:
            return None
        first, second = buf[0], buf[1]
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        offset = 2

        if length == 126:
            if len(buf) < offset + 2:
                return None
            (length,) = struct.unpack_from("!H", buf, offset)
            offset += 2
        elif length == 127:
            if len(buf) < offset + 8:
                return None
            (length,) = struct.unpack_from("!Q", buf, offset)
            offset += 8

        if length > self.max_message_bytes:
            raise WSError(f"frame of {length} bytes over cap {self.max_message_bytes}")

        key = b""
        if masked:
            if len(buf) < offset + 4:
                return None
            key = bytes(buf[offset : offset + 4])
            offset += 4
        elif self.require_mask:
            # RFC 6455 §5.1: a client MUST mask. Refuse rather than guess.
            raise WSError("client frame is not masked")

        if len(buf) < offset + length:
            return None

        payload = bytes(buf[offset : offset + length])
        del buf[: offset + length]
        if key:
            payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))

        if opcode in (OP_CLOSE, OP_PING, OP_PONG):
            if not fin:
                raise WSError("control frames must not be fragmented")
            return Message(opcode, payload)

        if opcode == OP_CONT:
            if self._fragment_opcode is None:
                raise WSError("continuation frame with nothing to continue")
            self._append_fragment(payload)
            if fin:
                message = Message(self._fragment_opcode, bytes(self._fragments))
                self._fragments = bytearray()
                self._fragment_opcode = None
                return message
            return self._next()

        if opcode not in (OP_TEXT, OP_BINARY):
            raise WSError(f"unsupported opcode {opcode:#x}")

        if not fin:
            if self._fragment_opcode is not None:
                raise WSError("new data frame while a fragment was in progress")
            self._fragment_opcode = opcode
            self._append_fragment(payload)
            return self._next()

        return Message(opcode, payload)

    def _append_fragment(self, payload: bytes) -> None:
        if len(self._fragments) + len(payload) > self.max_message_bytes:
            raise WSError("fragmented message over cap")
        self._fragments += payload
