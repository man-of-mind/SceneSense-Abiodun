"""Bounded control messages over a private SOCK_SEQPACKET pair, at most one sealed fd each.

Control headers are small canonical JSON. Bulk data never travels in a header:
it is passed as one sealed memfd (unchanged V4 transport create/verify/mapped).
"""
import array
import hashlib
import json
import os
import socket
import time

from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M

MAX_MESSAGE = 1024 * 1024
BUFFER_BYTES = 4 * MAX_MESSAGE


class ChannelError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ChannelError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pair():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    for sock in (parent, child):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_BYTES)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_BYTES)
    return parent, child


def _timeout(sock, deadline):
    left = deadline - time.monotonic()
    require(left > 0, "channel deadline expired")
    sock.settimeout(left)


def send(sock, value, *, fd=None, deadline):
    payload = canonical(value)
    require(0 < len(payload) <= MAX_MESSAGE, "control message bound exceeded")
    _timeout(sock, deadline)
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    require(sock.sendmsg([payload], ancillary) == len(payload), "partial control message")


def receive(sock, *, deadline):
    _timeout(sock, deadline)
    fds = []
    try:
        payload, ancillary, flags, _ = sock.recvmsg(MAX_MESSAGE + 1, socket.CMSG_SPACE(4 * array.array("i").itemsize),
                                                    socket.MSG_CMSG_CLOEXEC)
        for level, kind, data in ancillary:
            require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS, "unexpected ancillary data")
            values = array.array("i")
            values.frombytes(data[: len(data) - len(data) % values.itemsize])
            fds.extend(values)
        require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) and 0 < len(payload) <= MAX_MESSAGE
                and len(fds) <= 1, "truncated or malformed control message")
        require(payload, "peer closed the channel")
        value = json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ChannelError("nonfinite")))
        require(type(value) is dict and canonical(value) == payload, "canonical control message required")
        fd = fds.pop() if fds else None
        return value, fd
    finally:
        for extra in fds:
            os.close(extra)


def seal_bytes(chunks):
    """Sealed memfd for bulk payload; caller closes the returned fd."""
    fd, size = M.create([memoryview(c).cast("B") for c in chunks])
    return fd, size


def read_sealed(fd, size):
    """Copy a sealed payload into owned bytes (one copy; mapping released before return)."""
    with M.mapped(fd, size) as view:
        return bytes(view)


def pack_files(files):
    names = sorted(files)
    index, chunks, offset = [], [], 0
    for name in names:
        raw = files[name]
        require(type(name) is str and isinstance(raw, (bytes, bytearray)), "named byte files required")
        index.append([name, offset, len(raw), hashlib.sha256(raw).hexdigest()])
        chunks.append(bytes(raw))
        offset += len(raw)
    return dict(files=index, size=offset), chunks


def unpack_files(index, fd):
    raw = read_sealed(fd, index["size"])
    out = {}
    for name, offset, size, digest in index["files"]:
        body = raw[offset:offset + size]
        require(name not in out and hashlib.sha256(body).hexdigest() == digest, "file container digest differs")
        out[name] = body
    return out
