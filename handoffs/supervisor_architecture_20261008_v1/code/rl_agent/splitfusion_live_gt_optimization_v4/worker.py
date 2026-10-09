"""Sealed-memory CPU-process candidate for the unchanged complete-GT body.

Not a live entry point. One caller, one child, one request in flight, no queue.
The child owns sequential tracker/conflict history; it never owns CARLA or CUDA.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
from types import CodeType, FunctionType, SimpleNamespace

MAX_MESSAGE = 32 * 1024 * 1024
MAX_JSON = 2 * 1024 * 1024
FREEVARS = ("attempt_dir", "catalogue", "local", "max_frames", "original")


class WorkerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise WorkerError(message)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def scheduling_context():
    return dict(affinity=sorted(os.sched_getaffinity(0)),
                scheduler=os.sched_getscheduler(0), nice=os.getpriority(os.PRIO_PROCESS, 0))


def verify_child_context(expected):
    import torch
    import cv2
    require(scheduling_context() == expected, "CPU worker inherited scheduling differs")
    threads = {}
    for name in os.listdir("/proc/self/task"):
        tid = int(name)
        try:
            context = dict(affinity=sorted(os.sched_getaffinity(tid)),
                scheduler=os.sched_getscheduler(tid), nice=os.getpriority(os.PRIO_PROCESS, tid))
        except ProcessLookupError:
            continue
        require(context == expected, "CPU worker thread scheduling differs")
        threads[name] = context
    require(not torch.cuda.is_initialized(), "CPU worker initialized CUDA")
    return dict(pid=os.getpid(), scheduling=expected, threads=threads,
        cuda_initialized=False, torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(), opencv_threads=cv2.getNumThreads(),
        environment={name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def sources():
    root = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob("*.py")]
    for package in ("splitfusion_live_gt_optimization_v1", "splitfusion_live_gt_optimization_v2",
                    "splitfusion_live_gt_optimization_v3",
                    "splitfusion_live_parity_repair_v1", "splitfusion_live_parity_repair_v2",
                    "splitfusion_run5_live_avo_v1"):
        paths.extend((root / "rl_agent" / package).glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith("test_")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def frame_bytes(meta, body=b""):
    doc = canonical(meta)
    require(len(doc) <= MAX_JSON and len(doc) + len(body) + 8 <= MAX_MESSAGE, "message bound exceeded")
    return struct.pack("!Q", len(doc)) + doc + body


def unpack_message(payload):
    require(8 <= len(payload) <= MAX_MESSAGE, "invalid message length")
    size = struct.unpack("!Q", payload[:8])[0]
    require(0 < size <= MAX_JSON and size + 8 <= len(payload), "invalid JSON length")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate message key")
            out[key] = value
        return out
    meta = json.loads(payload[8:8+size], object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(WorkerError("nonfinite JSON")))
    require(isinstance(meta, dict), "message metadata must be an object")
    return meta, payload[8+size:]


def remaining(sock, deadline):
    seconds = deadline - time.monotonic()
    require(seconds > 0, "CPU worker communication deadline expired")
    sock.settimeout(seconds)


def send(sock, payload, deadline):
    require(len(payload) <= MAX_MESSAGE, "message bound exceeded")
    remaining(sock, deadline)
    sock.sendall(struct.pack("!Q", len(payload)))
    remaining(sock, deadline)
    sock.sendall(payload)


def receive(sock, deadline):
    def exact(count):
        result = bytearray()
        while len(result) < count:
            remaining(sock, deadline)
            block = sock.recv(min(count-len(result), 1024*1024))
            require(block, "CPU worker connection closed")
            result.extend(block)
        return bytes(result)
    size = struct.unpack("!Q", exact(8))[0]
    require(8 <= size <= MAX_MESSAGE, "incoming message bound exceeded")
    return unpack_message(exact(size))


def cell(value):
    return (lambda: value).__closure__[0]


class Engine:
    """Single CPU-owner proof engine executing the exact pinned closure body."""
    def __init__(self, *, attempt_dir, catalogue_dir, max_frames, expected_sources,
                 allow_replay_clock=False, source_attempt_name=None, semantic_attempt_dir=None):
        from rl_agent.splitfusion_live_gt_optimization_v2 import runtime as V2
        from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
        from rl_agent.splitfusion_live_parity_repair_v2.gt_timing import StageLog
        require(type(max_frames) is int and 1 <= max_frames <= 1000, "bounded frame count required")
        require(expected_sources == sources(), "CPU worker source binding differs")
        self.bindings, self.max_frames = expected_sources, max_frames
        self.attempt = Path(attempt_dir).resolve()
        self.attempt.mkdir(parents=True, exist_ok=False)
        self.catalogue = Path(catalogue_dir).resolve(strict=True)
        source_attempt_name = self.catalogue.parent.name if source_attempt_name is None else source_attempt_name
        require(isinstance(source_attempt_name, str) and source_attempt_name not in ("", ".", "..")
                and Path(source_attempt_name).name == source_attempt_name, "invalid source attempt name")
        semantic_attempt = self.attempt / source_attempt_name if semantic_attempt_dir is None else Path(semantic_attempt_dir).resolve(strict=True)
        require(semantic_attempt.name == source_attempt_name, "semantic attempt identity differs")
        self.stack = ExitStack()
        self.closed = self.failed = False
        self.count = 0
        self.last_frame = -1
        self.sync = None
        self.allow_replay_clock = bool(allow_replay_clock)
        self.log = StageLog(max_frames)
        self.owner = SimpleNamespace(ego=None, live=SimpleNamespace(last_datagram={}), _complete_producer=None)
        self.local = SimpleNamespace(collector=self.owner, kwargs={})
        try:
            self.stack.enter_context(V2.install_cpu_gt_optimizations(expected_sources=V2.source_inventory()))
            from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, evaluation as E, target_transport as T
            from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
            bindings = ((LG, "write_avo_bundle", "avo_scoring_and_sidecar"),
                (S.LiveCompleteTargetProducer, "build", "complete_source_build"),
                (E, "prepare_target", "prepare_complete_target"),
                (T, "write_bundle", "publish_complete_target"),
                (G, "write_source_archive", "archive_auxiliary_sources"))
            for module, name, stage in bindings:
                original = getattr(module, name)
                self.stack.callback(setattr, module, name, original)
                setattr(module, name, self.log.wrap(stage, original))
            code = [v for v in G.install_complete_gt.__wrapped__.__code__.co_consts
                    if isinstance(v, CodeType) and v.co_name == "write"]
            require(len(code) == 1 and code[0].co_freevars == FREEVARS, "original complete-GT closure changed")
            values = dict(attempt_dir=semantic_attempt, catalogue=SimpleNamespace(directory=self.catalogue),
                local=self.local, max_frames=max_frames, original=(None, None, LG.write_avo_bundle, None))
            self.write = FunctionType(code[0], G.__dict__, "write",
                closure=tuple(cell(values[name]) for name in FREEVARS))
        except BaseException:
            self.stack.close()
            raise

    def process(self, *, directory, kwargs, radar, sent_raw_ns, sync_config=None,
                replay_observed_raw_ns=None):
        from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY
        require(not self.closed and not self.failed and self.count < self.max_frames, "CPU worker not admissible")
        frame = kwargs["identity"]["frame_id"]
        require(type(frame) is int and frame > self.last_frame, "CPU worker frame order differs")
        require(type(sent_raw_ns) is int and 0 < sent_raw_ns <= raw(), "GT request precedes last send")
        population = kwargs["frozen_world"].population_evidence
        ego_id = population["population"]["ego_id"]
        if self.owner.ego is None:
            self.owner.ego = SimpleNamespace(id=ego_id)
        require(self.owner.ego.id == ego_id, "CPU worker ego changed")
        self.owner.live.last_datagram = {frame: sent_raw_ns}
        self.local.kwargs = {"radar_points": {"world_xyz": radar}}
        if sync_config is not None:
            if self.sync is None:
                self.sync = SY.LiveSyncEvidence(Path(sync_config["directory"]),
                    raw_bundle_limit=sync_config["raw_bundle_limit"], raw_byte_limit=sync_config["raw_byte_limit"])
            require(str(self.sync.directory) == sync_config["directory"] and
                self.sync.raw_bundle_limit == sync_config["raw_bundle_limit"] and
                self.sync.raw_byte_limit == sync_config["raw_byte_limit"], "source synchronization owner changed")
        elif self.sync is not None:
            raise WorkerError("source synchronization evidence disappeared")
        require(replay_observed_raw_ns is None or self.allow_replay_clock, "offline timestamp supplied to production worker")
        before, begin, cpu = len(self.log.rows), raw(), time.thread_time_ns()
        original_alignment = SY.source_alignment_record
        if replay_observed_raw_ns is not None:
            def alignment(*args, **kw):
                result = original_alignment(*args, **kw)
                result["observed_raw_ns"] = replay_observed_raw_ns
                return result
            # Only source.build imports this name at call time; existing AVO
            # retains its real clock. Offline exact provenance replay only.
            SY.source_alignment_record = alignment
        try:
            record = self.write(Path(directory), **kwargs, sync_evidence=self.sync)
            status, error = "OK", None
            self.count += 1
            self.last_frame = frame
        except BaseException as exc:
            self.failed = True
            record = None
            status, error = "FAILED", dict(type=type(exc).__name__, message=str(exc),
                avo_evidence_record=getattr(exc, "avo_evidence_record", None))
        finally:
            SY.source_alignment_record = original_alignment
        rows = [{**row, "process_pid": os.getpid(), "process_role": "dedicated_cpu_gt"}
                for row in self.log.rows[before:]]
        return dict(status=status, identity=kwargs["identity"], record=record, error=error,
            rows=rows, sync=None if self.sync is None else self.sync.summary(),
            worker=dict(pid=os.getpid(), start_raw_ns=begin, end_raw_ns=raw(),
                        thread_cpu_ns=time.thread_time_ns()-cpu), processed=self.count)

    def close(self):
        if self.closed:
            return self.close_result
        self.closed = True
        self.close_result = None
        try:
            self.stack.close()
            require(self.bindings == sources(), "CPU worker sources changed")
            import torch
            require(not torch.cuda.is_initialized(), "CPU worker initialized CUDA")
        finally:
            # A handled failure is a stopped synchronous job, not an active
            # request. Preserve its real partial stage log before replying.
            self.close_result = self.log.flush(self.attempt / "complete_gt_stage_timing", worker_drained=True)
        return self.close_result


class Client:
    def __init__(self, *, attempt_dir, catalogue_dir, max_frames, expected_sources,
                 timeout_s=30., allow_replay_clock=False, source_attempt_name=None, semantic_attempt_dir=None):
        require(0 < timeout_s <= 120, "bounded worker timeout required")
        self.timeout = float(timeout_s)
        self.failed = self.closed = False
        self.count = 0
        self.lock = threading.Lock()
        self.events = []
        root = Path(attempt_dir)
        root.mkdir(parents=True, exist_ok=False)
        parent, child = socket.socketpair()
        self.socket = parent
        try:
            self.console = (root / "worker.log").open("xb")
        except BaseException:
            parent.close()
            child.close()
            raise
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", __name__, "--fd", str(child.fileno())],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=self.console,
                stderr=subprocess.STDOUT, env=env, cwd=Path(__file__).resolve().parents[2])
        except BaseException:
            parent.close()
            child.close()
            self.console.close()
            raise
        child.close()
        try:
            self.expected_context = scheduling_context()
            self.ready = self._exchange(dict(command="INIT", attempt_dir=str(root / "engine"), catalogue_dir=str(catalogue_dir),
                max_frames=max_frames, expected_sources=expected_sources, allow_replay_clock=allow_replay_clock,
                source_attempt_name=source_attempt_name,
                semantic_attempt_dir=None if semantic_attempt_dir is None else str(semantic_attempt_dir),
                expected_process_context=self.expected_context))
        except BaseException:
            self.abort()
            raise

    def _exchange(self, meta, body=b"", descriptor=None):
        require(not self.closed and not self.failed, "CPU worker unavailable")
        deadline = time.monotonic() + self.timeout
        try:
            send(self.socket, frame_bytes(meta, body), deadline)
            if descriptor is not None:
                from .memfd_transport import send_descriptor
                send_descriptor(self.socket, descriptor, deadline)
            result, extra = receive(self.socket, deadline)
            require(not extra, "unexpected response binary data")
            require(result.get("status") == "OK", f"CPU worker rejected request: {result}")
            return result
        except BaseException:
            self.failed = True
            self.abort()
            raise

    def run(self, *, directory, kwargs, radar, sent_raw_ns, sync_config=None,
            replay_observed_raw_ns=None):
        from .snapshot import pack_chunks
        from .memfd_transport import create
        require(self.lock.acquire(blocking=False), "concurrent CPU worker requests forbidden")
        descriptor = None
        try:
            begin, cpu = raw(), time.thread_time_ns()
            chunks = pack_chunks(kwargs, radar)
            packed = raw()
            descriptor, request_bytes = create(chunks)
            sealed = raw()
            response = self._exchange(dict(command="RUN", directory=str(directory), sent_raw_ns=sent_raw_ns,
                sync_config=sync_config, replay_observed_raw_ns=replay_observed_raw_ns,
                expected_identity=kwargs["identity"], snapshot_bytes=request_bytes), descriptor=descriptor)
            result = response["result"]
            require(result["identity"] == kwargs["identity"] and result["worker"]["pid"] == self.process.pid,
                    "foreign CPU worker response identity")
            if result["status"] != "OK":
                self.failed = True
                error = WorkerError(str(result["error"]))
                error.worker_result = result
                if result["error"].get("avo_evidence_record") is not None:
                    error.avo_evidence_record = result["error"]["avo_evidence_record"]
                self.abort()
                raise error
            self.count += 1
            require(result["processed"] == self.count, "CPU worker completion count differs")
            event = dict(frame_id=kwargs["identity"]["frame_id"], start_raw_ns=begin,
                packed_raw_ns=packed, completed_raw_ns=raw(), parent_thread_cpu_ns=time.thread_time_ns()-cpu,
                sealed_raw_ns=sealed, request_bytes=request_bytes, child_pid=self.process.pid)
            self.events.append(event)
            return {**result, "ipc": event}
        except BaseException:
            self.failed = True
            self.abort()
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.lock.release()

    def abort(self):
        self.failed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.socket.close()
        self.console.close()
        self.closed = True

    def close(self):
        if self.closed:
            require(not self.failed, "CPU worker previously failed")
            return
        require(self.lock.acquire(blocking=False), "CPU worker request still in flight")
        try:
            result = self._exchange({"command": "CLOSE"})
            require(result["processed"] == self.count, "CPU worker close accounting differs")
            require(self.process.wait(timeout=5) == 0, "CPU worker exit failed")
            self.socket.close()
            self.console.close()
            self.closed = True
            return result
        except BaseException:
            self.abort()
            raise
        finally:
            self.lock.release()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    engine = None
    expected_context = None
    try:
        while True:
            meta, body = receive(sock, time.monotonic()+120)
            command = meta.pop("command")
            if command == "INIT":
                require(engine is None and not body, "duplicate/invalid worker initialization")
                expected_context = meta.pop("expected_process_context")
                engine = Engine(**meta)
                context = verify_child_context(expected_context)
                result = dict(status="OK", pid=os.getpid(), cuda_initialized=False, process_context=context)
            elif command == "RUN":
                from .snapshot import unpack_snapshot
                from .memfd_transport import receive_descriptor, mapped
                require(engine is not None and not body, "uninitialized/invalid CPU worker")
                descriptor = receive_descriptor(sock, time.monotonic()+30)
                try:
                    size = meta.pop("snapshot_bytes")
                    with mapped(descriptor, size) as payload:
                        kwargs = radar = None
                        try:
                            kwargs, radar = unpack_snapshot(payload)
                            require(kwargs["identity"] == meta.pop("expected_identity"), "request identity differs")
                            result = dict(status="OK", result=engine.process(kwargs=kwargs, radar=radar, **meta))
                        finally:
                            kwargs = radar = None
                finally:
                    os.close(descriptor)
                if result["result"]["status"] != "OK":
                    result["result"]["failure_close"] = engine.close()
                result["result"]["process_context"] = verify_child_context(expected_context)
            elif command == "CLOSE":
                require(engine is not None and not body and not meta, "invalid worker close")
                timing = engine.close()
                result = dict(status="OK", processed=engine.count, timing=timing,
                              process_context=verify_child_context(expected_context))
            else:
                raise WorkerError("unknown CPU worker command")
            send(sock, frame_bytes(result), time.monotonic()+30)
            if command == "CLOSE":
                return 0
            if command == "RUN" and engine.failed:
                return 1
    except BaseException as exc:
        close_error = None
        if engine is not None and not engine.closed:
            try:
                engine.close()
            except Exception as closing:
                close_error = f"{type(closing).__name__}: {closing}"
        try:
            send(sock, frame_bytes(dict(status="FAILED", type=type(exc).__name__, error=str(exc),
                close_error=close_error)), time.monotonic()+2)
        except Exception:
            pass
        raise
    finally:
        if engine is not None and not engine.closed:
            engine.close()
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
