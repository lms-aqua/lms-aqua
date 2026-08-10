"""Shared network helpers.

Small on purpose. It exists because the disconnect test below was written three
times in three servers, and three copies of an errno list is three chances for
one of them to fall out of step with the others.
"""

from __future__ import annotations

import errno

# Errors that mean "the peer went away". Every stream in LostCam ends this way —
# a client stops reading, a phone leaves Wi-Fi, a dashboard's tab closes — so it
# is normal operation rather than a fault worth a traceback.
DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
)

# Some platforms surface the same conditions as a bare OSError with an errno.
# WSAECONNABORTED/WSAECONNRESET are the Windows spellings, which are not in the
# errno module on POSIX, hence the literals.
DISCONNECT_ERRNOS = frozenset(
    {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.EBADF,
        errno.ESHUTDOWN,
        10053,  # WSAECONNABORTED
        10054,  # WSAECONNRESET
    }
)


def is_disconnect(error: BaseException | None) -> bool:
    """Whether an exception means the other end simply hung up."""
    if error is None:
        return False
    if isinstance(error, DISCONNECT_ERRORS):
        return True
    return isinstance(error, OSError) and error.errno in DISCONNECT_ERRNOS


def read_available(response, size: int) -> bytes:
    """Read whatever is available now, up to ``size`` bytes.

    This exists because ``HTTPResponse.read(n)`` is the wrong primitive for a
    never-ending stream: it blocks until it has *all* n bytes. On a stream of
    small frames that means waiting for many frames to accumulate before any is
    delivered — measured at a full second of added latency for 6 KB depth frames
    with a 64 KB read size, and it silently discards intermediate frames because
    only the newest is kept downstream.

    ``read1`` returns as soon as any data has arrived, which is what a live
    stream wants. The fallback keeps this working with any file-like object that
    predates ``read1``.
    """
    reader = getattr(response, "read1", None)
    if reader is not None:
        return reader(size)
    return response.read(size)
