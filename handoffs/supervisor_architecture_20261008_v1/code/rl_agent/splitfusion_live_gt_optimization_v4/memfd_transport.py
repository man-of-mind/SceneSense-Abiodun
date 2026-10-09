"""Immutable, bounded Linux descriptor handoff; no named shared-memory file.

Each request owns one sealed memfd until synchronous completion. The receiver
checks size and kernel seals before mapping read-only. This is not a queue.
"""
import array
from contextlib import contextmanager
import fcntl
import mmap
import os
import socket
import stat
import time
import traceback

MAX_BYTES = 32 * 1024 * 1024
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


class TransferError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise TransferError(message)


def timeout(sock, deadline):
    left = deadline - time.monotonic()
    require(left > 0, "descriptor handoff deadline expired")
    sock.settimeout(left)


def create(chunks):
    """One kernel copy per chunk, no joined image-sized Python byte strings."""
    views = [memoryview(chunk).cast("B") for chunk in chunks]
    size = sum(len(view) for view in views)
    require(0 < size <= MAX_BYTES and 0 < len(views) <= 64, "snapshot transfer bound exceeded")
    descriptor = os.memfd_create("splitfusion_gt_snapshot_v4", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(descriptor, size)
        pending = views
        while pending:
            written = os.writev(descriptor, pending)
            require(written > 0, "snapshot write made no progress")
            while pending and written >= len(pending[0]):
                written -= len(pending[0])
                pending = pending[1:]
            if written:
                pending = [pending[0][written:], *pending[1:]]
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, SEALS)
        verify(descriptor, size)
        return descriptor, size
    except BaseException:
        os.close(descriptor)
        raise


def verify(descriptor, size):
    require(type(size) is int and 0 < size <= MAX_BYTES, "invalid snapshot byte count")
    info = os.fstat(descriptor)
    require(stat.S_ISREG(info.st_mode) and info.st_size == size, "snapshot descriptor size/type mismatch")
    require(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & SEALS == SEALS,
            "snapshot descriptor is not fully immutable")


def send_descriptor(sock, descriptor, deadline):
    timeout(sock, deadline)
    sent = sock.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [descriptor]))])
    require(sent == 1, "partial snapshot descriptor handoff")


def receive_descriptor(sock, deadline):
    timeout(sock, deadline)
    descriptors = []
    try:
        token, ancillary, flags, _ = sock.recvmsg(1, socket.CMSG_SPACE(16 * array.array("i").itemsize),
                                                 socket.MSG_CMSG_CLOEXEC)
        unexpected = False
        for level, kind, payload in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                unexpected = True
                continue
            values = array.array("i")
            require(len(payload) % values.itemsize == 0, "malformed descriptor payload")
            values.frombytes(payload)
            descriptors.extend(values)
        require(token == b"F" and not flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC)
                and not unexpected and len(descriptors) == 1, "invalid snapshot descriptor envelope")
        result = descriptors[0]
        os.set_inheritable(result, False)
        descriptors.pop()
        return result
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


@contextmanager
def mapped(descriptor, size):
    """Caller releases all decoded views before exit; leaked views fail closed."""
    verify(descriptor, size)
    mapping = mmap.mmap(descriptor, size, access=mmap.ACCESS_READ)
    view = memoryview(mapping)
    try:
        yield view
    except BaseException as error:
        # A rejected decoder may retain mmap views in inactive traceback
        # locals. Release those locals while retaining exception type/message,
        # traceback locations and chains; never replace it with BufferError.
        pending, seen = [error], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            traceback.clear_frames(current.__traceback__)
            pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        raise
    finally:
        # mmap exports must be released by the caller, not silently leaked.
        view.release()
        mapping.close()
