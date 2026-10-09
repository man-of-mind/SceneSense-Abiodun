"""Release the FREEZE caller's borrowed source on success AND failure.

The sealed implementation only executes ``del kwargs`` on success. On failure,
its still-executing handle frame keeps the images exported while mmap.close()
runs; traceback.clear_frames cannot clear an executing frame. BufferError then
replaces the real failure. This version changes only that lifetime, not decoding,
GT preparation, checks, transactions, history, or normal-path copies.

An optional, fatal-error-only snapshot preserves the original exception and the
one sealed source needed to replay it. A snapshot error never replaces the GT
error. Existing protocol behavior remains fail-closed; no error is downgraded.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M

ORIGINAL_PREP_CORE = PC.PrepCore
MAX_DIAGNOSTIC_BYTES = 256 * 1024


def _create_file(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        try:
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("failure snapshot write made no progress")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            view.release()
    finally:
        os.close(descriptor)


def save_failure(directory, args, fd, error):
    """At most one <=32 MiB immutable source, plus a bounded exception record.

    READY.json is written last; partial writes are not replay-ready. The caller
    catches any persistence error and re-raises the original exception.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("existing non-symlink diagnostic parent required")
    output = root / "freeze_failure_v1"
    output.mkdir(mode=0o700, exist_ok=False)
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    request = json.dumps(args, sort_keys=True, allow_nan=False).encode()
    if len(request) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("failure request exceeds diagnostic bound")
    # Preserve the exception before attempting source retention.
    failure = dict(schema="splitfusion.gt_freeze_lifetime.failure.v1",
                   exception_type=type(error).__name__, exception=str(error)[:8192],
                   traceback=formatted[-65536:], pid=os.getpid())
    _create_file(output / "ERROR.json", json.dumps(failure, indent=2).encode())
    _create_file(output / "REQUEST.json", request)
    hashes = {"ERROR.json": hashlib.sha256((output / "ERROR.json").read_bytes()).hexdigest(),
              "REQUEST.json": hashlib.sha256(request).hexdigest()}
    if fd is not None:
        size = os.fstat(fd).st_size
        M.verify(fd, size)
        # pread does not change the worker protocol's descriptor offset.
        data = bytearray()
        while len(data) < size:
            piece = os.pread(fd, min(1024 * 1024, size - len(data)), len(data))
            if not piece:
                raise OSError("truncated sealed failure source")
            data.extend(piece)
        _create_file(output / "SOURCE.bin", data)
        hashes["SOURCE.bin"] = hashlib.sha256(data).hexdigest()
    _create_file(output / "READY.json", json.dumps(dict(complete=True, sha256=hashes), indent=2).encode())
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class PrepCoreLifetime(ORIGINAL_PREP_CORE):
    """Unchanged PrepCore for every operation except FREEZE's exception lifetime."""

    failure_directory = None

    def handle(self, op, args, fd):
        if op != "FREEZE":
            return super().handle(op, args, fd)
        e = self.engine
        try:
            PC.C.require(fd is not None, "sealed source fd required")
            offline = args.get("offline")
            PC.C.require((offline is not None) == self.offline, "offline replay inputs only in offline mode")
            with self.gate.admit(args["descriptor"].encode(), fd) as admitted:
                decoded = dict(decoded_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                               decoded_thread_cpu_ns=time.thread_time_ns())
                kwargs = admitted.kwargs
                try:
                    handle = e.prepare_unbound(kwargs, population=kwargs["dynamic_snapshot_evidence"],
                                               **(offline or {}))
                finally:
                    # This MUST execute before FreezeGate / mmap context exit.
                    del kwargs
                sequence, frame = admitted.descriptor["sequence"], admitted.descriptor["frame_id"]
            self.jobs[sequence] = dict(phase="UNBOUND", frame_id=frame, handle=handle)
            return dict(job=sequence, frame_id=frame, **decoded), None
        except BaseException as error:
            # Fatal errors alone touch disk/logging; no per-frame logging or copy.
            try:
                traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
            except BaseException:
                pass
            if self.failure_directory is not None:
                try:
                    save_failure(self.failure_directory, args, fd, error)
                except BaseException as snapshot_error:
                    try:
                        print(f"FREEZE_DIAGNOSTIC_WRITE_FAILED: {type(snapshot_error).__name__}: "
                              f"{str(snapshot_error)[:512]}", file=sys.stderr, flush=True)
                    except BaseException:
                        pass
            raise


@contextmanager
def install(failure_directory=None):
    """Explicit process-local opt-in, before the retained worker constructs a core."""
    if PC.PrepCore is not ORIGINAL_PREP_CORE:
        raise RuntimeError("prep lifetime repair requires the unchanged PrepCore")
    previous_directory = PrepCoreLifetime.failure_directory
    PrepCoreLifetime.failure_directory = failure_directory
    PC.PrepCore = PrepCoreLifetime
    try:
        yield
    finally:
        PC.PrepCore = ORIGINAL_PREP_CORE
        PrepCoreLifetime.failure_directory = previous_directory
