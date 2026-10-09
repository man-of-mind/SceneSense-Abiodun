"""Persistent owner for a bounded CPU Client child.

Linux PR_SET_PDEATHSIG is tied to the thread that forks, not merely its process.
A short-lived frame thread must not create a Client and then exit while the
child is still needed.  This adapter creates and operates the unchanged Client
on one persistent thread, which exits only after owned close/abort.  The child
keeps its original parent-death signal and source binding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading

from .process import Client


class WorkerOwnerError(RuntimeError):
    pass


@dataclass
class _Request:
    method: str
    args: tuple
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None


class OwnedClient:
    """Drop-in Client interface; one request outstanding and one owned child.

    ``ready``, ``pending``, ``events``, ``failed``, ``closed`` and ``process``
    are snapshots updated after every owner-thread operation.  Callers must
    still serialize their ticket lifecycle; no retry or hidden worker queue is
    introduced.  The owner thread is daemonized so an abnormal host exit never
    waits for it, while the child's original parent-death signal remains set.
    """

    def __init__(self, role, config, *, directory, source_sha256, cpu=4,
                 timeout_s=30, writable_attempt_root=None):
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 60:
            raise ValueError("bounded worker-owner timeout required")
        self.timeout = float(timeout_s)
        self.ready = None
        self.pending = None
        self.events = []
        self.process = None
        self.failed = False
        self.closed = False
        self._close_result = None
        self._start_error = None
        self._client = None
        self._start_done = threading.Event()
        self._stop_after_timeout = threading.Event()
        self._requests = queue.Queue(maxsize=1)
        self._call_lock = threading.Lock()
        self._terminal_requested = False
        arguments = dict(role=role, config=config, directory=directory,
            source_sha256=source_sha256, cpu=cpu, timeout_s=timeout_s,
            writable_attempt_root=writable_attempt_root)
        self._thread = threading.Thread(target=self._serve, args=(arguments,),
                                        name="numeric-" + role + "-owner", daemon=True)
        self._thread.start()
        if not self._start_done.wait(self.timeout + 5):
            self._stop_after_timeout.set()
            self.failed = True
            raise WorkerOwnerError("bounded owned-worker startup timed out")
        if self._start_error is not None:
            raise self._start_error

    @property
    def owner_thread_alive(self):
        return self._thread.is_alive()

    @property
    def owner_native_id(self):
        return self._thread.native_id

    def _snapshot(self, client):
        self.ready = getattr(client, "ready", None)
        self.pending = None if client.pending is None else dict(client.pending)
        self.events = list(client.events)
        self.process = client.process
        self.failed = client.failed
        self.closed = client.closed

    def _serve(self, arguments):
        client = None
        try:
            try:
                client = Client(**arguments)
                self._client = client
                self._snapshot(client)
            except BaseException as exc:
                self._start_error = exc
                return
            finally:
                self._start_done.set()
            if self._stop_after_timeout.is_set():
                return
            while True:
                request = self._requests.get()
                try:
                    request.result = getattr(client, request.method)(*request.args)
                except BaseException as exc:
                    request.error = exc
                finally:
                    self._snapshot(client)
                    request.done.set()
                if request.method in ("close", "abort") or self._stop_after_timeout.is_set():
                    return
        finally:
            if client is not None and not client.closed:
                client.abort()
                self._snapshot(client)

    def _request(self, method, *args):
        if not self._call_lock.acquire(timeout=self.timeout + 5):
            raise WorkerOwnerError("owned-worker caller serialization timed out")
        try:
            if self._terminal_requested or not self._thread.is_alive():
                raise WorkerOwnerError("owned-worker thread already terminated")
            request = _Request(method, args)
            if method in ("close", "abort"):
                self._terminal_requested = True
            try:
                self._requests.put_nowait(request)
            except queue.Full as exc:
                raise WorkerOwnerError("owned-worker queue is not bounded to one request") from exc
            limit = (2 if method == "close" else 1) * self.timeout + 5
            if not request.done.wait(limit):
                self.failed = True
                self._stop_after_timeout.set()
                raise WorkerOwnerError("bounded owned-worker operation timed out: " + method)
            if request.error is not None:
                raise request.error
            return request.result
        finally:
            self._call_lock.release()

    def begin(self, operation, value):
        return self._request("begin", operation, value)

    def available(self):
        return self._request("available")

    def finish(self):
        return self._request("finish")

    def call(self, operation, value=None):
        return self._request("call", operation, value)

    def close(self):
        if self.closed and not self.failed:
            return self._close_result
        try:
            result = self._request("close")
            self._close_result = result
            return result
        finally:
            self._thread.join(timeout=self.timeout + 5)

    def abort(self):
        if self.closed:
            return None
        try:
            return self._request("abort")
        finally:
            self._thread.join(timeout=self.timeout + 5)
