"""Bounded sealed-batch handoff to a dedicated, stdlib-only archive process.

submit() means kernel-owned immutable transfer, never durable publication.
poll()/close() report the child's actual file and directory synchronizations.
No retries, overwrites, unbounded queue, numerical imports, or parent threads.
max_frames is the cumulative submission (job) bound of one client, i.e. of one
registered attempt; jobs, not decision frames (callers size it at 4 x frames).
"""
import os
import sys

ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
if __name__ == "__main__":
    # The retained subprocess hook may strip requested caps. Reassert only
    # inside this dedicated child, before any optional application imports.
    os.environ.update(ENVIRONMENT)

import array
from collections.abc import Mapping
import fcntl
import hashlib
import json
import mmap
from pathlib import Path, PurePosixPath
import select
import socket
import stat
import subprocess
import time

SCHEMA = "splitfusion.gt_archive_handoff.v1"
MAX_HEADER = 128 * 1024
MAX_FILES = 128
MAX_BYTES = 512 * 1024 * 1024
# Capacity repair v1: finite cumulative job bound = 4 x the largest registered
# decision budget (1,000). Outstanding job/byte credits are unchanged and separate.
MAX_CUMULATIVE_JOBS = 4000
# Cumulative client bookkeeping (retained submission metadata + completion
# records), accounted separately from outstanding payload bytes.
MAX_BOOKKEEPING_BYTES = 64 * 1024 * 1024
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


class ArchiveError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ArchiveError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate archive metadata key")
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ArchiveError("nonfinite archive metadata")))


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _configuration(roots, max_jobs, max_bytes, max_frames, binding):
    require(type(max_jobs) is int and 2 <= max_jobs <= 8, "archive max_jobs must be 2..8")
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES, "archive byte bound differs")
    require(type(max_frames) is int and 0 < max_frames <= MAX_CUMULATIVE_JOBS, "archive frame bound differs")
    require(isinstance(binding, dict), "archive binding must be a JSON object")
    require(isinstance(roots, (tuple, list)) and 1 <= len(roots) <= 8, "explicit archive roots required")
    checked = []
    for value in roots:
        path = Path(value)
        require(path.is_absolute() and ".." not in path.parts, "absolute archive root required")
        resolved = path.resolve(strict=True)
        require(str(path) == str(resolved) and path.is_dir() and not path.is_symlink(),
                "archive roots must be existing resolved directories")
        require(resolved != Path("/"), "filesystem root is not an archive root")
        info = resolved.stat()
        checked.append(dict(path=str(resolved), dev=info.st_dev, ino=info.st_ino))
    names = [item["path"] for item in checked]
    require(len(set(names)) == len(names), "duplicate archive roots")
    return dict(schema=SCHEMA, roots=checked, max_jobs=max_jobs, max_bytes=max_bytes,
                max_frames=max_frames, binding=json.loads(canonical(binding)),
                archive_source_sha256=source_sha256())


def _relative(path, roots):
    require(isinstance(path, (str, Path)), "archive file path must be text or Path")
    text = str(path)
    require(text.startswith("/") and not any(part in (".", "..") for part in text.split("/"))
            and "\x00" not in text, "unsafe archive file path")
    value = PurePosixPath(text)
    matches = [(len(PurePosixPath(root["path"]).parts), index, value.relative_to(root["path"]))
               for index, root in enumerate(roots) if value.is_relative_to(root["path"])]
    require(bool(matches), "archive path is outside owned roots")
    _, index, relative = max(matches)
    require(relative.parts and str(relative) != ".", "archive file cannot replace its root")
    # Reject already-present symlinks in the parent too. The worker separately
    # enforces this using directory fds, closing the check/use race.
    candidate = Path(roots[index]["path"])
    for component in relative.parts:
        candidate = candidate / component
        require(not candidate.is_symlink(), "archive path contains a symlink")
    return index, str(relative)


def _encode_files(files, ready_paths, roots, stamp_ready_paths=()):
    require(isinstance(files, Mapping) and 0 < len(files) <= MAX_FILES, "bounded nonempty archive files required")
    require(isinstance(ready_paths, (list, tuple)), "READY paths must be an explicit sequence")
    normalized, chunks, records = {}, [], []
    for path, payload in files.items():
        key = _relative(path, roots)
        require(key not in normalized, "duplicate archive file path")
        require(type(payload) is bytes, "archive payload ownership requires immutable bytes")
        normalized[key] = payload
    ready = [_relative(path, roots) for path in ready_paths]
    require(len(ready) == len(set(ready)) and all(key in normalized for key in ready),
            "READY paths must be a unique subset of files")
    require(isinstance(stamp_ready_paths, (list, tuple)), "stamped READY paths must be an explicit sequence")
    stamped = [_relative(path, roots) for path in stamp_ready_paths]
    require(len(stamped) == len(set(stamped)) and set(stamped).issubset(ready),
            "stamped READY paths must be a unique subset of READY files")
    ready_set, offset = set(ready), 0
    for key in [key for key in normalized if key not in ready_set] + ready:
        payload = normalized[key]
        records.append(dict(root=key[0], relative=key[1], offset=offset, size=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(), ready=key in ready_set,
                            stamp_ready=key in stamped))
        chunks.append(payload)
        offset += len(payload)
    return records, chunks, offset


def _memfd(chunks):
    descriptor = os.memfd_create("splitfusion_archive_batch", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        for chunk in chunks:
            view = memoryview(chunk)
            try:
                position = 0
                while position < len(view):
                    written = os.write(descriptor, view[position:])
                    require(written > 0, "archive memfd write stalled")
                    position += written
            finally:
                view.release()
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, SEALS)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _send(sock, message, timeout, descriptor=None):
    payload = canonical(message)
    require(len(payload) <= MAX_HEADER, "archive metadata bound exceeded")
    sock.settimeout(timeout)
    ancillary = [] if descriptor is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                                array.array("i", [descriptor]))]
    require(sock.sendmsg([payload], ancillary) == len(payload), "partial archive packet transfer")


def _receive(sock, timeout):
    sock.settimeout(timeout)
    descriptors = []
    try:
        payload, ancillary, flags, _ = sock.recvmsg(MAX_HEADER + 1,
            socket.CMSG_SPACE(8 * array.array("i").itemsize), socket.MSG_CMSG_CLOEXEC)
        for level, kind, raw in ancillary:
            require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS, "foreign archive ancillary data")
            values = array.array("i")
            require(len(raw) % values.itemsize == 0, "malformed archive descriptor data")
            values.frombytes(raw)
            descriptors.extend(values)
        require(payload and len(payload) <= MAX_HEADER and not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC),
                "closed or truncated archive packet")
        require(len(descriptors) <= 1, "archive packet has too many descriptors")
        message = _json(payload)
        require(isinstance(message, dict), "archive metadata must be an object")
        descriptor = descriptors.pop() if descriptors else None
        return message, descriptor
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _root_fds(config):
    descriptors = []
    try:
        for root in config["roots"]:
            path = Path(root["path"])
            require(str(path.resolve(strict=True)) == str(path), "archive root contains a symlink")
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            require((info.st_dev, info.st_ino) == (root["dev"], root["ino"]), "archive root identity changed")
        return descriptors
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def _create_file(root_fd, relative, payload, *, atomic=False):
    parts = PurePosixPath(relative).parts
    require(parts and not relative.startswith("/") and all(part not in (".", "..") for part in parts),
            "unsafe relative archive path")
    parent = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, 0o755, dir_fd=parent)
            except FileExistsError:
                pass
            else:
                os.fsync(parent)
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
            os.close(parent)
            parent = child
        name = (f".archive-ready-{os.getpid()}-{time.monotonic_ns()}" if atomic else parts[-1])
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o644, dir_fd=parent)
        try:
            position = 0
            while position < len(payload):
                written = os.write(descriptor, payload[position:])
                require(written > 0, "archive file write stalled")
                position += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if atomic:
            # Link is atomic and fails if READY already exists. Any failed
            # write/link retains its staging file as incomplete evidence.
            os.link(name, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
            os.unlink(name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _write_batch(config, roots, message, descriptor, seen):
    size = message["bytes"]
    require(type(size) is int and 0 <= size <= config["max_bytes"], "archive batch size differs")
    info = os.fstat(descriptor)
    require(stat.S_ISREG(info.st_mode) and info.st_size == size, "archive descriptor size/type differs")
    require(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & SEALS == SEALS, "archive descriptor is mutable")
    records = message["files"]
    require(isinstance(records, list) and 0 < len(records) <= MAX_FILES, "archive file count differs")
    mapping = mmap.mmap(descriptor, size, access=mmap.ACCESS_READ) if size else None
    data = memoryview(mapping) if mapping is not None else memoryview(b"")
    keys, offset, ready_seen, ready_documents = [], 0, False, {}
    try:
        for record in records:
            root, relative, length = record["root"], record["relative"], record["size"]
            require(type(root) is int and 0 <= root < len(roots) and isinstance(relative, str),
                    "archive file root differs")
            require(relative and not relative.startswith("/") and
                    all(part not in ("", ".", "..") for part in relative.split("/")) and "\x00" not in relative,
                    "unsafe archive relative path")
            require(type(length) is int and 0 <= length <= size and
                    type(record["offset"]) is int and record["offset"] == offset,
                    "archive offsets differ")
            require(type(record["ready"]) is bool and not (ready_seen and not record["ready"]),
                    "archive READY ordering differs")
            require(type(record.get("stamp_ready")) is bool and
                    (not record["stamp_ready"] or record["ready"]), "archive READY stamp type differs")
            ready_seen |= record["ready"]
            key = (root, relative)
            require(key not in seen and key not in keys, "duplicate archive publication path")
            keys.append(key)
            piece = data[offset:offset + length]
            try:
                require(len(piece) == length and hashlib.sha256(piece).hexdigest() == record["sha256"],
                        "archive payload checksum differs")
                if record["stamp_ready"]:
                    document = _json(bytes(piece))
                    require(isinstance(document, dict) and
                            document.get("schema") == "splitfusion.prepared_live_v010_transport.v1" and
                            type(document.get("ready_raw_ns")) is int and document["ready_raw_ns"] == 0 and
                            type(document.get("ready_wall_ns")) is int and document["ready_wall_ns"] == 0,
                            "stamped READY requires exact schema and zero clock sentinels")
                    ready_documents[key] = document
            finally:
                piece.release()
            offset += length
        require(offset == size, "archive batch has unbound bytes")
        seen.update(keys)
        written, finalized_ready = [], []
        try:
            for record in records:
                key = (record["root"], record["relative"])
                if key in ready_documents:
                    document = ready_documents[key]
                    document.update(ready_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                                    ready_wall_ns=time.time_ns())
                    piece = memoryview(canonical(document))
                else:
                    piece = data[record["offset"]:record["offset"] + record["size"]]
                try:
                    digest, length = hashlib.sha256(piece).hexdigest(), len(piece)
                    _create_file(roots[record["root"]], record["relative"], piece, atomic=record["ready"])
                finally:
                    piece.release()
                written.append(dict(root=record["root"], relative=record["relative"],
                                    sha256=digest, bytes=length, ready=record["ready"]))
                if key in ready_documents:
                    finalized_ready.append(dict(path=str(Path(config["roots"][key[0]]["path"]) / key[1]),
                        sha256=digest, ready_raw_ns=document["ready_raw_ns"], ready_wall_ns=document["ready_wall_ns"]))
        except BaseException as error:
            error.archive_written = written
            raise
        return written, finalized_ready
    finally:
        data.release()
        if mapping is not None:
            mapping.close()


def _serve(sock):
    roots, config, fault, completed, last_job, seen = [], None, None, 0, 0, set()
    try:
        init, descriptor = _receive(sock, 30.)
        if descriptor is not None:
            os.close(descriptor)
        require(descriptor is None and init.get("command") == "INIT", "archive INIT required")
        config = init["config"]
        expected = _configuration([item["path"] for item in config["roots"]],
            config["max_jobs"], config["max_bytes"], config["max_frames"], config["binding"])
        require(config == expected, "archive startup source/roots/binding differs")
        roots = _root_fds(config)
        context = dict(pid=os.getpid(), parent_pid=os.getppid(), parent_death_signal="SIGTERM",
                       environment={key: os.environ.get(key) for key in ENVIRONMENT},
                       numerical_imports=[name for name in ("numpy", "torch", "cv2") if name in sys.modules])
        require(context["environment"] == ENVIRONMENT and not context["numerical_imports"], "archive child caps/imports differ")
        _send(sock, dict(status="READY", config=config, context=context), 30.)
        while True:
            message, descriptor = _receive(sock, None)
            if message.get("command") == "CLOSE":
                if descriptor is not None:
                    os.close(descriptor)
                require(descriptor is None, "descriptor on archive CLOSE")
                _send(sock, dict(status="CLOSED", jobs=last_job, completed=completed, fault=fault,
                                 context=context), 30.)
                return
            begin = time.monotonic_ns()
            try:
                require(message.get("command") == "SUBMIT" and descriptor is not None, "archive SUBMIT descriptor required")
                require(type(message.get("job_id")) is int and message["job_id"] == last_job + 1 and
                        message["job_id"] <= config["max_frames"], "archive job order/count differs")
                last_job = message["job_id"]
                require(isinstance(message.get("identity"), dict) and
                        type(message["identity"].get("frame_id")) is int and
                        message["identity"]["frame_id"] >= 0, "archive identity missing")
                require(fault is None, "archive stopped after earlier job failure: " + str(fault))
                written, finalized_ready = _write_batch(config, roots, message, descriptor, seen)
                completed += 1
                response = dict(status="COMPLETE", durable=True, written=written, finalized_ready=finalized_ready, error=None)
            except BaseException as error:
                fault = fault or f"{type(error).__name__}: {error}"
                response = dict(status="FAILED", durable=False, written=getattr(error, "archive_written", []), error=fault)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            response.update(job_id=message.get("job_id"), identity=message.get("identity"),
                bytes=message.get("bytes"), start_ns=begin, end_ns=time.monotonic_ns(), pid=os.getpid())
            _send(sock, response, 30.)
    finally:
        for descriptor in roots:
            os.close(descriptor)


class ArchiveClient:
    bookkeeping_bytes = 0     # class default keeps modeled (__new__) clients valid

    def __init__(self, *, roots, max_jobs, max_bytes, max_frames, binding, timeout_s=30.):
        require(type(timeout_s) in (int, float) and 0 < timeout_s <= 120, "bounded archive timeout required")
        self.config = _configuration(roots, max_jobs, max_bytes, max_frames, binding)
        self.timeout = float(timeout_s)
        self.closed = self.failed = False
        self.process = self.socket = None
        self.pending, self.completions, self.submissions = {}, [], []
        self.bookkeeping_bytes = 0
        self._paths, self._identities, self._fault = set(), set(), None
        self.close_result = None
        self.child_stderr = ""
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket = parent
        environment = dict(os.environ, **ENVIRONMENT, PYTHONDONTWRITEBYTECODE="1")
        environment.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", __name__, "--fd", str(child.fileno()),
                "--parent-pid", str(os.getpid())],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, env=environment, cwd=Path(__file__).resolve().parents[2])
            child.close()
            _send(parent, dict(command="INIT", config=self.config), self.timeout)
            response, descriptor = _receive(parent, self.timeout)
            if descriptor is not None:
                os.close(descriptor)
            require(descriptor is None and response.get("status") == "READY" and response.get("config") == self.config,
                    "archive READY source/config binding differs")
            context = response.get("context", {})
            require(context.get("pid") == self.process.pid and context.get("parent_pid") == os.getpid() and
                    context.get("parent_death_signal") == "SIGTERM" and context.get("environment") == ENVIRONMENT and
                    context.get("numerical_imports") == [], "archive READY process/caps differ")
            self.ready = response
        except BaseException as error:
            child.close()
            self._fail(error)
            self.abort()
            raise

    @property
    def outstanding_jobs(self):
        return len(self.pending)

    @property
    def outstanding_bytes(self):
        return sum(row["bytes"] for row in self.pending.values())

    def _fail(self, error):
        self.failed = True
        if self._fault is None:
            self._fault = error if isinstance(error, ArchiveError) else ArchiveError(f"{type(error).__name__}: {error}")
        return self._fault

    def _completion(self, response, descriptor):
        if descriptor is not None:
            os.close(descriptor)
        require(descriptor is None and response.get("status") in ("COMPLETE", "FAILED"), "unexpected archive completion")
        job = response.get("job_id")
        require(type(job) is int and job in self.pending, "unknown/duplicate archive completion")
        expected = self.pending[job]
        require(response.get("identity") == expected["identity"] and response.get("bytes") == expected["bytes"] and
                response.get("pid") == self.process.pid, "archive completion identity differs")
        require(response.get("durable") is (response["status"] == "COMPLETE"), "archive durable status differs")
        if response["status"] == "COMPLETE":
            written, finalized = response.get("written"), response.get("finalized_ready")
            require(isinstance(written, list) and len(written) == len(expected["files"]) and
                    isinstance(finalized, list) and len(finalized) == len(expected["stamp_ready_paths"]),
                    "archive durable file inventory differs")
            finalized_by_path = {row["path"]: row for row in finalized}
            require(set(finalized_by_path) == set(expected["stamp_ready_paths"]), "archive finalized READY paths differ")
            for old, new in zip(expected["files"], written):
                path = str(Path(self.config["roots"][old["root"]]["path"]) / old["relative"])
                if path not in finalized_by_path:
                    require(old == new, "archive durable payload changed")
                else:
                    row = finalized_by_path[path]
                    require(all(old[key] == new.get(key) for key in ("root", "relative", "ready")) and
                            new.get("sha256") == row.get("sha256") and type(new.get("bytes")) is int and new["bytes"] > 0 and
                            all(type(row.get(key)) is int and row[key] > 0 for key in ("ready_raw_ns", "ready_wall_ns")),
                            "archive finalized READY clock/hash binding differs")
        self.pending.pop(job)
        self.completions.append(response)
        self.bookkeeping_bytes += len(canonical(response))
        if response["status"] == "FAILED":
            self._fail(ArchiveError("archive job failed: " + str(response.get("error"))))

    def _poll(self):
        before = len(self.completions)
        while self.pending and select.select([self.socket], [], [], 0)[0]:
            self._completion(*_receive(self.socket, self.timeout))
        return self.completions[before:]

    def poll(self):
        require(not self.closed, "archive client is closed")
        try:
            result = self._poll()
            if self.failed:
                raise self._fault
            return result
        except BaseException as error:
            raise self._fail(error)

    def submit(self, frame_identity, files, ready_paths=(), stamp_ready_paths=()):
        require(not self.closed, "archive client is closed")
        descriptor = None
        try:
            self.poll()
            require(isinstance(frame_identity, dict) and type(frame_identity.get("frame_id")) is int and
                    frame_identity["frame_id"] >= 0, "archive frame identity required")
            identity_bytes = canonical(frame_identity)
            identity = json.loads(identity_bytes)
            require(identity_bytes not in self._identities, "duplicate archive frame identity")
            require(len(self.submissions) < self.config["max_frames"], "archive frame limit exceeded")
            records, chunks, size = _encode_files(files, ready_paths, self.config["roots"], stamp_ready_paths)
            keys = {(record["root"], record["relative"]) for record in records}
            require(not keys & self._paths, "duplicate archive file across jobs")
            require(self.outstanding_jobs < self.config["max_jobs"] and
                    self.outstanding_bytes + size <= self.config["max_bytes"], "archive outstanding job/byte credits exhausted")
            job = len(self.submissions) + 1
            message = dict(command="SUBMIT", job_id=job, identity=identity, files=records, bytes=size)
            metadata = len(canonical(message))
            require(metadata <= MAX_HEADER, "archive metadata bound exceeded")
            require(self.bookkeeping_bytes + metadata <= MAX_BOOKKEEPING_BYTES, "archive bookkeeping bound exceeded")
            descriptor = _memfd(chunks)
            inventory = [dict(root=r["root"], relative=r["relative"], sha256=r["sha256"],
                              bytes=r["size"], ready=r["ready"]) for r in records]
            record = dict(job_id=job, identity=identity, bytes=size, files=inventory,
                          stamp_ready_paths=[str(Path(self.config["roots"][r["root"]]["path"]) / r["relative"])
                                             for r in records if r["stamp_ready"]], ownership="PREPARING_TRANSFER")
            self.pending[job] = record
            self.submissions.append(record)
            self._paths.update(keys)
            self._identities.add(identity_bytes)
            self.bookkeeping_bytes += metadata
            _send(self.socket, message, self.timeout, descriptor)
            record["ownership"] = "SEALED_KERNEL_TRANSFER_NOT_DURABLE"
            return job
        except BaseException as error:
            raise self._fail(error)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _summary(self):
        return dict(schema=SCHEMA, binding=self.config["binding"], submitted=len(self.submissions),
            completed=sum(row["durable"] for row in self.completions), failed=self.failed,
            fault=None if self._fault is None else str(self._fault),
            outstanding_jobs=self.outstanding_jobs, outstanding_bytes=self.outstanding_bytes,
            submissions=list(self.submissions), completions=list(self.completions),
            child_pid=None if self.process is None else self.process.pid,
            child_stderr=self.child_stderr,
            child_returncode=None if self.process is None else self.process.poll())

    def close(self):
        if self.closed:
            if self.failed:
                raise self._fault
            return self.close_result
        deadline = time.monotonic() + self.timeout
        try:
            while self.pending:
                require(time.monotonic() < deadline, "archive drain deadline expired")
                self._completion(*_receive(self.socket, max(.001, deadline - time.monotonic())))
            _send(self.socket, dict(command="CLOSE"), max(.001, deadline - time.monotonic()))
            response, descriptor = _receive(self.socket, max(.001, deadline - time.monotonic()))
            if descriptor is not None:
                os.close(descriptor)
            require(descriptor is None and response.get("status") == "CLOSED" and
                    response.get("jobs") == len(self.submissions), "archive CLOSE reconciliation differs")
            require(response.get("completed") == sum(row["durable"] for row in self.completions),
                    "archive completion denominator differs")
            if response.get("fault") is not None:
                self._fail(ArchiveError(response["fault"]))
            self.process.wait(timeout=max(.001, deadline - time.monotonic()))
            require(self.process.returncode == 0, "archive child exited unsuccessfully")
            self.closed = True
            self.socket.close()
            self.child_stderr = self.process.stderr.read(16384).decode("utf-8", "replace")
            self.process.stderr.close()
        except BaseException as error:
            self._fail(error)
            self.abort()
        self.close_result = self._summary()
        if self.failed:
            raise self._fault
        return self.close_result

    def abort(self):
        if not self.closed:
            self._fail(ArchiveError("archive client aborted"))
        self.closed = True
        if self.socket is not None:
            self.socket.close()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=min(self.timeout, 1.))
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.)
            if self.process.stderr is not None:
                if not self.process.stderr.closed:
                    self.child_stderr = self.process.stderr.read(16384).decode("utf-8", "replace")
                    self.process.stderr.close()
        if self.pending:
            self._fail(ArchiveError("archive aborted with unresolved owned jobs"))
        self.close_result = self._summary()


def main():
    require(len(sys.argv) == 5 and sys.argv[1] == "--fd" and sys.argv[3] == "--parent-pid",
            "archive child requires its inherited socket and owner pid")
    import ctypes
    import signal
    expected_parent = int(sys.argv[4])
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, "archive parent-death signal setup failed")
    require(os.getppid() == expected_parent, "archive parent died during startup")
    sock = socket.socket(fileno=int(sys.argv[2]))
    try:
        _serve(sock)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
