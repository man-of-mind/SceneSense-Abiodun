"""One bounded evaluator-owned local scoring handoff, with no disk polling.

Admission means immutable bytes are retained in this process, not persisted.
Consumed identities remain in a bounded ledger. Errors are sticky; overflow,
duplicates and foreign publications cannot silently replace or drop work.
"""
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import socket
import stat
import struct
import threading
import time

from . import scoring_packet as P

MAX_ACK_BYTES = 4096
MAX_JOBS = 1000
MAX_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 1000


class BrokerError(RuntimeError):
    pass


class NotReadyError(BrokerError):
    pass


def _require(value, message):
    if not value:
        raise BrokerError(message)


def _timeout(value, *, allow_zero=False):
    _require(type(value) in (int, float) and math.isfinite(value)
        and (0 <= value <= 30 if allow_zero else 0 < value <= 30), "bounded handoff timeout required")
    return float(value)


def _remaining(sock, deadline):
    left = deadline - time.monotonic()
    _require(left > 0, "handoff communication deadline expired")
    sock.settimeout(left)


def _exact(sock, size, deadline):
    chunks = []
    while size:
        _remaining(sock, deadline)
        block = sock.recv(min(size, 1024 * 1024))
        _require(block, "handoff connection closed before complete message")
        chunks.append(block)
        size -= len(block)
    return b"".join(chunks)


def _send(sock, payload, deadline):
    _remaining(sock, deadline)
    sock.sendall(struct.pack("!Q", len(payload)))
    _remaining(sock, deadline)
    sock.sendall(payload)


def _receive(sock, maximum, deadline):
    size = struct.unpack("!Q", _exact(sock, 8, deadline))[0]
    _require(0 < size <= maximum, "handoff message length exceeds bound")
    return _exact(sock, size, deadline)


def _peer_credentials(connection):
    """Kernel-authenticated Linux credentials, before any application bytes."""
    pid, uid, gid = struct.unpack("=iII", connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("=iII")))
    # A publisher in an ancestor PID namespace is not visible to a container
    # receiver; Linux then reports pid=0. UID/GID are still kernel credentials
    # and must match the registered owner below. Never use this PID for signals.
    _require(pid >= 0, "invalid publisher process identity")
    return dict(pid=pid, uid=uid, gid=gid)


def _ledger_copy(row):
    return dict(row, identity=dict(row["identity"]),
                publisher_credentials=dict(row["publisher_credentials"]))


@contextmanager
def _address(path):
    """Pin an existing real parent directory without the Unix path-length cap.

Both processes resolve their own shared-mount directory FD. The kernel binds
or connects the ordinary filesystem socket inode; no abstract namespace or
new shared mount is introduced. O_NOFOLLOW applies to every directory step.
"""
    path = Path(path).absolute()
    _require(path.name not in ("", ".", "..") and ".." not in path.parts, "unsafe broker endpoint")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in path.parent.parts[1:]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        short = f"/proc/self/fd/{descriptor}/{path.name}"
        _require(len(os.fsencode(short)) < 108, "Unix handoff socket basename too long")
        yield short
    finally:
        os.close(descriptor)


def publish(path, payload, *, binding, timeout_s=3.):
    """Publish one complete packet; return only after bounded admission ACK."""
    timeout = _timeout(timeout_s)
    expected = P.inspect_packet(payload, binding=binding)
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        _remaining(connection, deadline)
        with _address(path) as address:
            connection.connect(address)
        _send(connection, payload, deadline)
        try:
            answer = json.loads(_receive(connection, MAX_ACK_BYTES, deadline))
        except (ValueError, UnicodeError) as exc:
            raise BrokerError("malformed admission acknowledgment") from exc
    _require(isinstance(answer, dict) and answer.get("status") == "ADMITTED",
        "scoring publication rejected: " + str(answer))
    _require(answer.get("binding") == binding and answer.get("identity") == expected["identity"]
        and answer.get("packet_sha256") == expected["packet_sha256"], "foreign admission acknowledgment")
    return answer


class Server:
    def __init__(self, path, *, binding, max_jobs=2, max_bytes=32 * 1024 * 1024,
                 max_frames=100, read_timeout_s=3., publisher_uid=None, publisher_gid=None):
        self.path = Path(path).absolute()
        P.validate_binding(binding)
        for value, maximum, name in ((max_jobs, MAX_JOBS, "jobs"), (max_bytes, MAX_BYTES, "bytes"),
                                    (max_frames, MAX_FRAMES, "frames")):
            _require(type(value) is int and 0 < value <= maximum, "invalid broker " + name + " limit")
        _require(max_jobs <= max_frames, "pending bound exceeds total frame bound")
        explicit = publisher_uid is not None or publisher_gid is not None
        _require((publisher_uid is None) == (publisher_gid is None),
                 "publisher UID and GID must be supplied together")
        creator_uid, creator_gid = os.geteuid(), os.getegid()
        if not explicit:
            publisher_uid, publisher_gid = creator_uid, creator_gid
        _require(all(type(value) is int and 0 <= value < 2**32 - 1
                     for value in (publisher_uid, publisher_gid)), "invalid publisher UID/GID")
        _require((publisher_uid, publisher_gid) == (creator_uid, creator_gid) or creator_uid == 0,
                 "only root may configure foreign publisher ownership")
        self.publisher_uid, self.publisher_gid = publisher_uid, publisher_gid
        self.credential_policy = dict(publisher_uid=publisher_uid, publisher_gid=publisher_gid,
            explicitly_supplied=explicit, creator_uid=creator_uid, creator_gid=creator_gid,
            mode="0600", authentication="SO_PEERCRED_BEFORE_APPLICATION_READ")
        self.binding = binding
        self.max_jobs, self.max_bytes, self.max_frames = max_jobs, max_bytes, max_frames
        self.read_timeout = _timeout(read_timeout_s)
        self.condition = threading.Condition()
        self._pending, self._ledger = {}, {}
        self._stream_identity = None
        self._last_frame = -1
        self._bytes = self._high_jobs = self._high_bytes = self._consumed = 0
        self._failure = None
        self._started = self._closing = False
        self._listener = self._connection = self._thread = None
        self._socket_identity = None
        self._socket_proof = self._rejected_credentials = None

    def _verify_endpoint(self):
        info = self.path.lstat()
        _require(stat.S_ISSOCK(info.st_mode)
            and (info.st_dev, info.st_ino) == self._socket_identity
            and (info.st_uid, info.st_gid) == (self.publisher_uid, self.publisher_gid)
            and stat.S_IMODE(info.st_mode) == 0o600, "broker socket identity/owner/mode changed")
        return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid,
                    gid=info.st_gid, mode="0600")

    def _set_endpoint_owner(self):
        # Pin the newly created socket inode. /proc/self/fd dereferences that
        # inode, not a replaceable pathname; never chown the shared directory.
        descriptor = os.open(self.path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(descriptor)
            _require(stat.S_ISSOCK(info.st_mode)
                and (info.st_dev, info.st_ino) == self._socket_identity,
                "new broker socket replaced before ownership assignment")
            pinned = f"/proc/self/fd/{descriptor}"
            if (info.st_uid, info.st_gid) != (self.publisher_uid, self.publisher_gid):
                os.chown(pinned, self.publisher_uid, self.publisher_gid)
            os.chmod(pinned, 0o600)
            self._socket_proof = self._verify_endpoint()
        finally:
            os.close(descriptor)

    def start(self):
        with self.condition:
            _require(not self._started and not self._closing, "broker start repeated/closed")
            _require(not any(p.is_symlink() for p in (self.path, *self.path.parents)), "symbolic broker endpoint")
            _require(self.path.parent.is_dir() and not os.path.lexists(self.path), "broker endpoint must be create-only")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                with _address(self.path) as address:
                    listener.bind(address)
                info = self.path.lstat()
                self._socket_identity = (info.st_dev, info.st_ino)
                self._set_endpoint_owner()
                listener.listen(1)
                listener.settimeout(.1)
                self._listener = listener
                self._thread = threading.Thread(target=self._serve, name="exact-gt-scoring-handoff", daemon=False)
                self._started = True
                self._thread.start()
            except BaseException:
                listener.close()
                self._remove_endpoint()
                self._closing = True
                raise
        return self

    def _fail(self, exc):
        with self.condition:
            if self._failure is None:
                self._failure = f"{type(exc).__name__}: {exc}"
            self.condition.notify_all()

    def _admit(self, payload, credentials):
        observed = P.inspect_packet(payload, binding=self.binding)
        key = P.identity_key(observed["identity"])
        stream = tuple(observed["identity"][key] for key in ("run_id", "cell_id", "stream_id"))
        with self.condition:
            _require(not self._closing and self._failure is None, "broker closed or failed")
            _require(key not in self._ledger, "duplicate scoring identity, including consumed frame")
            _require(self._stream_identity is None or self._stream_identity == stream,
                "foreign scoring run/cell/stream")
            _require(observed["identity"]["frame_id"] > self._last_frame,
                "duplicate/regressing scoring frame")
            _require(len(self._ledger) < self.max_frames, "total scoring frame bound exceeded")
            _require(len(self._pending) < self.max_jobs and self._bytes + len(payload) <= self.max_bytes,
                "pending scoring capacity exceeded")
            observed.update(admitted_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                            admitted_wall_ns=time.time_ns(), publisher_credentials=dict(credentials))
            self._pending[key] = payload
            self._ledger[key] = dict(observed, consumed=False)
            self._stream_identity, self._last_frame = stream, observed["identity"]["frame_id"]
            self._bytes += len(payload)
            self._high_jobs = max(self._high_jobs, len(self._pending))
            self._high_bytes = max(self._high_bytes, self._bytes)
            self.condition.notify_all()
        return dict(status="ADMITTED", binding=self.binding, identity=observed["identity"],
                    packet_sha256=observed["packet_sha256"])

    def _serve(self):
        try:
            while True:
                with self.condition:
                    if self._closing:
                        return
                try:
                    connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    with self.condition:
                        if self._closing:
                            return
                    raise
                with connection:
                    with self.condition:
                        self._connection = connection
                        closing, failure = self._closing, self._failure
                    if closing:
                        return
                    deadline = time.monotonic() + self.read_timeout
                    try:
                        _require(failure is None, "broker previously failed: " + str(failure))
                        self._verify_endpoint()
                        credentials = _peer_credentials(connection)
                        if (credentials["uid"], credentials["gid"]) != (self.publisher_uid, self.publisher_gid):
                            with self.condition:
                                self._rejected_credentials = dict(credentials)
                            raise BrokerError("publisher SO_PEERCRED does not match registered UID/GID")
                        payload = _receive(connection, P.MAX_PACKET_BYTES, deadline)
                        answer = self._admit(payload, credentials)
                        _send(connection, json.dumps(answer, sort_keys=True, separators=(",", ":")).encode(), deadline)
                    except BaseException as exc:
                        with self.condition:
                            closing = self._closing
                        if not closing:
                            self._fail(exc)
                            try:
                                answer = json.dumps(dict(status="FAILED", error=str(exc)[:1000])).encode()
                                _send(connection, answer, time.monotonic() + min(.2, self.read_timeout))
                            except Exception:
                                pass
                    finally:
                        with self.condition:
                            self._connection = None
        except BaseException as exc:
            self._fail(exc)

    def take(self, expected_identity, timeout_s=0.):
        """Consume exact identity once; a wait timeout leaves all packets intact."""
        timeout = _timeout(timeout_s, allow_zero=True)
        key = P.identity_key(expected_identity)
        deadline = time.monotonic() + timeout
        with self.condition:
            _require(self._started, "broker not started")
            while True:
                _require(self._failure is None, "broker failed: " + str(self._failure))
                if key in self._pending:
                    payload = self._pending.pop(key)
                    self._bytes -= len(payload)
                    self._ledger[key]["consumed"] = True
                    self._consumed += 1
                    return payload
                _require(key not in self._ledger, "scoring identity already consumed")
                _require(not self._closing, "broker closed before requested identity arrived")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise NotReadyError("scoring packet not yet available")
                self.condition.wait(left)

    def stats(self):
        with self.condition:
            return dict(binding=self.binding, started=self._started, closed=self._closing,
                worker_alive=self._thread is not None and self._thread.is_alive(), failure=self._failure,
                admitted=len(self._ledger), consumed=self._consumed, pending_jobs=len(self._pending),
                pending_bytes=self._bytes, high_water_jobs=self._high_jobs, high_water_bytes=self._high_bytes,
                limits=dict(max_jobs=self.max_jobs, max_bytes=self.max_bytes, max_frames=self.max_frames),
                credential_policy=dict(self.credential_policy),
                socket_proof=None if self._socket_proof is None else dict(self._socket_proof),
                rejected_credentials=None if self._rejected_credentials is None else dict(self._rejected_credentials),
                ledger=[_ledger_copy(row) for row in self._ledger.values()],
                all_consumed=not self._pending, payloads_retained_for_unconsumed=len(self._pending))

    def observe(self, expected_identity):
        """Nonconsuming receiver admission evidence; no payload parsing/copy."""
        key = P.identity_key(expected_identity)
        with self.condition:
            _require(self._failure is None, "broker failed: " + str(self._failure))
            row = self._ledger.get(key)
            return None if row is None else _ledger_copy(row)

    def _remove_endpoint(self):
        if self._socket_identity is None:
            return
        try:
            value = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(value.st_mode) and (value.st_dev, value.st_ino) == self._socket_identity:
            self.path.unlink()

    def close(self):
        with self.condition:
            self._closing = True
            connection, listener, thread = self._connection, self._listener, self._thread
            self.condition.notify_all()
        for sock in (connection, listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if thread is not None:
            thread.join(timeout=self.read_timeout + .5)
            _require(not thread.is_alive(), "scoring broker thread failed to stop")
        self._remove_endpoint()
        return self.stats()
