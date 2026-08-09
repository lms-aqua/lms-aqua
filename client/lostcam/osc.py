"""A minimal OSC 1.0 encoder.

OSC is how VJ and motion tooling expects to be fed — TouchDesigner, Blender
add-ons, Unity OSC receivers, Max/MSP, VRChat's tracking input all speak it.
Encoding it is about eighty lines, so it is done here rather than taking a
dependency for one direction of one format.

Only what the bridge sends: messages of floats/ints/strings/bools, and bundles.
No pattern matching, no server side.
"""

from __future__ import annotations

import struct

MAX_ADDRESS_BYTES = 512


class OSCError(Exception):
    """The value could not be represented in OSC."""


def _pad(data: bytes) -> bytes:
    """OSC pads every element to a 4-byte boundary."""
    remainder = len(data) % 4
    return data if remainder == 0 else data + b"\x00" * (4 - remainder)


def _string(value: str) -> bytes:
    # OSC strings are null-terminated *then* padded, so a 4-char string takes 8
    # bytes, not 4 — a classic source of off-by-four bugs.
    return _pad(value.encode("utf-8") + b"\x00")


def sanitize_address(address: str) -> str:
    """Make an arbitrary key safe to use as an OSC address.

    OSC reserves ``# * , ? [ ] { }`` and space for pattern matching, and every
    address must start with ``/``. Blendshape and channel names are already
    tame, but flattened keys like ``blend.jawOpen`` need the dots turned into
    path separators to be useful to a receiver.
    """
    if not address:
        raise OSCError("address must not be empty")
    address = address.replace(".", "/")
    for reserved in "#*,?[]{} ":
        address = address.replace(reserved, "_")
    if not address.startswith("/"):
        address = "/" + address
    while "//" in address:
        address = address.replace("//", "/")
    if len(address.encode("utf-8")) > MAX_ADDRESS_BYTES:
        raise OSCError("address is unreasonably long")
    return address


def message(address: str, *args: object) -> bytes:
    """Encode one OSC message."""
    address = sanitize_address(address)
    tags = ","
    body = b""
    for arg in args:
        # bool is a subclass of int, so it must be tested first.
        if isinstance(arg, bool):
            tags += "T" if arg else "F"  # OSC 1.1 booleans carry no payload
        elif isinstance(arg, int):
            if not -(2**31) <= arg < 2**31:
                # Out of int32 range: send as float rather than truncating.
                tags += "f"
                body += struct.pack(">f", float(arg))
            else:
                tags += "i"
                body += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            body += struct.pack(">f", arg)
        elif isinstance(arg, str):
            tags += "s"
            body += _string(arg)
        elif isinstance(arg, (bytes, bytearray)):
            tags += "b"
            body += struct.pack(">i", len(arg)) + _pad(bytes(arg))
        elif arg is None:
            tags += "N"  # OSC 1.1 nil
        else:
            raise OSCError(f"cannot encode {type(arg).__name__} in OSC")
    return _string(address) + _string(tags) + body


def bundle(messages: list[bytes], timetag: bytes | None = None) -> bytes:
    """Bundle messages so a receiver applies them as one atomic update.

    Sending a pose as sixteen separate messages invites a receiver to render a
    half-updated matrix; a bundle is the fix.
    """
    # The immediate timetag (1) means "apply now".
    tag = timetag if timetag is not None else struct.pack(">Q", 1)
    if len(tag) != 8:
        raise OSCError("an OSC timetag is 8 bytes")
    out = _string("#bundle") + tag
    for element in messages:
        out += struct.pack(">i", len(element)) + element
    return out
