"""Bounded, additive CPU-GT seam; not a live runner or launch authorization.

The retained GT queue still owns one synchronous job at a time. Only its
complete-target write body runs in an exec'd CPU process. This scope belongs
outside the unchanged composition/source-ownership scopes, and replaces (not
wraps) their parent-process complete-GT timing scope.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading

from . import worker as W


class RuntimeErrorGT(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeErrorGT(message)


def _write(path, value):
    payload = W.canonical(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _timed(function):
    from rl_agent.splitfusion_live_parity_repair_v2.gt_timing import StageLog
    seen = set()
    for _ in range(12):
        if function is None or id(function) in seen:
            return False
        seen.add(id(function))
        for item in getattr(function, "__closure__", ()) or ():
            try:
                if isinstance(item.cell_contents, StageLog):
                    return True
            except ValueError:
                pass
        function = getattr(function, "__wrapped__", None)
    raise RuntimeErrorGT("unbounded parent GT timing wrapper")


def _require_no_parent_timing():
    from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G, source_geometry as S, evaluation as E, target_transport as T
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    require(not any(_timed(value) for value in (LG.write_avo_bundle,
        S.LiveCompleteTargetProducer.build, E.prepare_target, T.write_bundle,
        G.write_source_archive)), "parent complete-GT timing scope must be replaced by child timing")


def _merge_sync(owner, summary, *, frame, max_frames, allow_partial=False):
    """Mirror child counters only: its original durable files are authoritative."""
    from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import LiveSyncEvidence, SYNC_SCHEMA
    sync = owner._avo_sync
    require(isinstance(sync, LiveSyncEvidence) and isinstance(summary, dict), "real synchronization evidence required")
    expected = dict(schema=SYNC_SCHEMA, directory=str(sync.directory),
        raw_bundle_limit=sync.raw_bundle_limit, raw_byte_limit=sync.raw_byte_limit)
    require(all(summary.get(key) == value for key, value in expected.items()), "child synchronization binding differs")
    keys = ("accepted", "rejected", "raw_bundles", "raw_bytes")
    require(all(type(summary.get(key)) is int and summary[key] >= 0 for key in keys), "invalid synchronization counters")
    records = summary.get("records")
    total = summary["accepted"] + summary["rejected"]
    # The original collector increments a counter before the final durable
    # write. Preserve that exact partial state on a handled write failure.
    require(isinstance(records, list) and total <= max_frames
        and 0 <= total-len(records) <= int(allow_partial), "synchronization record count differs")
    require(summary["raw_bundles"] <= summary["raw_bundle_limit"]
        and summary["raw_bytes"] <= summary["raw_byte_limit"], "synchronization raw bound exceeded")
    require(all(isinstance(record, dict) and type(record.get("frame_id")) is int for record in records),
        "invalid synchronization record")
    frames = [record["frame_id"] for record in records]
    require(frames == sorted(set(frames)) and (not frames or frames[-1] <= frame), "foreign synchronization frame")
    with sync._lock:
        require(records[:len(sync.records)] == sync.records and len(records)-len(sync.records) in (0, 1),
            "synchronization history changed or skipped")
        require(all(summary[key] >= getattr(sync, key) for key in keys), "synchronization counter regressed")
        for key in keys:
            setattr(sync, key, summary[key])
        sync.records = json.loads(json.dumps(records, allow_nan=False))


class _Session:
    def __init__(self, attempt, max_frames, expected_sources, timeout_s, client_factory):
        self.attempt = attempt
        self.max_frames, self.sources, self.timeout = max_frames, dict(expected_sources), timeout_s
        self.client_factory = client_factory
        self.owner = self.client = None
        self.local = threading.local()
        self.errors, self.ipc = [], []
        self.closed = self.joint_drained = self.preserved = False
        self.preservation_started = self.close_succeeded = False
        self.timing = None
        self.started_raw_ns = self.ready_raw_ns = None
        self.state = dict(schema="splitfusion.cpu_gt_worker.runtime_scope.v4", live_qualified=False,
            sources=self.sources, max_frames=max_frames, new_python_threads=0,
            new_cpu_processes=0, new_queues=0, no_cuda_work=True, complete=False)

    def fail(self, exc):
        if len(self.errors) < self.max_frames + 16:
            self.errors.append(dict(type=type(exc).__name__, error=str(exc)))

    def start(self, owner, catalogue):
        require(self.owner is owner and self.client is None, "CPU worker duplicate/foreign startup")
        catalogue.require_world(owner.world)
        require(owner._avo_sync is not None and not owner._avo_sync.records, "GT synchronization already started")
        _require_no_parent_timing()
        self.started_raw_ns = W.raw()
        self.client = self.client_factory(attempt_dir=self.attempt / "cpu_gt_worker",
            catalogue_dir=catalogue.directory, max_frames=self.max_frames,
            expected_sources=self.sources, timeout_s=self.timeout,
            source_attempt_name=self.attempt.name, semantic_attempt_dir=self.attempt)
        self.ready_raw_ns = W.raw()
        require(self.client.ready.get("cuda_initialized") is False, "CPU worker CUDA neutrality missing")
        self.state["new_cpu_processes"] = 1

    def write(self, directory, **kwargs):
        owner = getattr(self.local, "owner", None)
        require(owner is self.owner and owner is not None and self.client is not None,
            "CPU complete GT outside retained worker")
        require(not self.closed and not self.errors, "CPU GT session already failed/closed")
        sync = kwargs.pop("sync_evidence", None)
        require(sync is owner._avo_sync and sync is not None, "foreign parent synchronization owner")
        require(Path(directory).resolve() == Path(owner.edge_evidence_dir).resolve(), "foreign target destination")
        identity = kwargs["identity"]
        frame = identity["frame_id"]
        require(identity == owner.live.identity_for_frame(frame), "GT scientific identity changed")
        radar = self.local.kwargs.get("radar_points", {}).get("world_xyz")
        require(radar is not None, "owned radar source missing")
        sent = owner.live.last_datagram.get(frame)
        require(type(sent) is int and 0 < sent <= W.raw(), "complete GT precedes last send")
        config = dict(directory=str(sync.directory), raw_bundle_limit=sync.raw_bundle_limit,
            raw_byte_limit=sync.raw_byte_limit)
        try:
            result = self.client.run(directory=directory, kwargs=kwargs, radar=radar,
                sent_raw_ns=sent, sync_config=config)
            _merge_sync(owner, result["sync"], frame=frame, max_frames=self.max_frames)
            require(len(self.ipc) < self.max_frames, "IPC evidence bound exceeded")
            self.ipc.append(dict(stage="parent_snapshot_and_ipc", **result["ipc"]))
            return result["record"]
        except BaseException as exc:
            result = getattr(exc, "worker_result", None)
            if isinstance(result, dict) and result.get("identity") == identity and result.get("sync") is not None:
                try:
                    _merge_sync(owner, result["sync"], frame=frame, max_frames=self.max_frames, allow_partial=True)
                except Exception as merge_error:
                    self.fail(merge_error)
            self.fail(exc)
            raise

    def _copy_timing(self):
        source = self.attempt / "cpu_gt_worker/engine/complete_gt_stage_timing"
        destination = self.attempt / "complete_gt_stage_timing"
        require(source.is_dir() and not source.is_symlink(), "child GT timing not preserved")
        require({path.name for path in source.iterdir()} == {"MANIFEST.json", "stages.jsonl"},
            "child GT timing inventory differs")
        require(not any(path.is_symlink() for path in source.iterdir()), "unsafe child GT timing")
        manifest = json.loads((source / "MANIFEST.json").read_bytes())
        digest = hashlib.sha256((source / "stages.jsonl").read_bytes()).hexdigest()
        require(manifest.get("sha256") == {"stages.jsonl": digest}
            and manifest.get("flush_after_worker_drain") is True and manifest.get("dropped") == 0,
            "invalid child GT stage evidence")
        shutil.copytree(source, destination)
        require(all((source / name).read_bytes() == (destination / name).read_bytes()
            for name in ("MANIFEST.json", "stages.jsonl")), "child timing preservation differs")
        self.timing = dict(source=str(source), destination=str(destination),
            process_pid=self.client.process.pid, process_role="dedicated_cpu_gt",
            original_manifest_sha256=hashlib.sha256((source / "MANIFEST.json").read_bytes()).hexdigest(),
            records=manifest["records"], copied_only_after_joint_drain=True)

    def before_preserve(self):
        require(not self.preservation_started and self.owner is not None, "duplicate/foreign GT preservation")
        self.preservation_started = True
        worker = getattr(self.owner, "evaluation_worker", None)
        require(worker is not None and worker.is_alive() is False, "retained GT thread has not drained")
        require(getattr(self.owner.evaluation_queue, "unfinished_tasks", None) == 0,
            "retained GT queue has not drained")
        require(self.client is not None and not self.client.lock.locked(), "CPU GT request still in flight")
        if not self.client.closed:
            try:
                self.client.close()
                self.close_succeeded = True
            except Exception as exc:
                # Client.close normally terminates and joins on failure. It
                # must not discard previous complete targets if every writer
                # is already stopped; the failed close still fails the scope.
                self.fail(exc)
        require(self.client.process.poll() is not None, "CPU GT child has not exited")
        if self.client.process.poll() != 0 and not self.errors:
            self.fail(RuntimeErrorGT("CPU GT child exited unsuccessfully"))
        self.closed = self.joint_drained = True
        try:
            self._copy_timing()
        except Exception as exc:
            # A dead child cannot race the retained preservation. Keep prior
            # complete targets even if a hard stop lost its timing footer.
            # Missing/partial timing is explicit failure, never synthesized.
            self.fail(exc)

    def cleanup(self):
        if self.client is not None and not self.client.closed:
            try:
                self.client.abort()
            except Exception as exc:
                self.fail(exc)
        if self.owner is not None:
            thread = getattr(self.owner, "evaluation_worker", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.)
                if thread.is_alive():
                    self.fail(RuntimeErrorGT("retained GT worker still alive after bounded CPU child abort"))
        if self.client is not None and self.client.process.poll() is None:
            self.fail(RuntimeErrorGT("CPU child still alive at scope exit"))
        self.closed = True
        self.state.update(complete=self.joint_drained and self.preserved and not self.errors,
            joint_drained=self.joint_drained, preserved_before_teardown=self.preserved,
            preservation_started=self.preservation_started, worker_close_succeeded=self.close_succeeded,
            errors=list(self.errors), child_pid=None if self.client is None else self.client.process.pid,
            child_ready=None if self.client is None else self.client.ready,
            started_raw_ns=self.started_raw_ns, ready_raw_ns=self.ready_raw_ns,
            parent_ipc_records=list(self.ipc), child_timing=self.timing)
        _write(self.attempt / "GT_WORKER_SCOPE.json", self.state)


@contextmanager
def install_cpu_gt_worker(*, attempt_dir, max_frames, expected_sources, timeout_s=30., _client_factory=None):
    """Install before composition; preserved sealed V6 files are never edited.

    ``_client_factory`` is a test seam only. A future live registration must
    bind these sources, the worker and the caller; this function grants none.
    """
    from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    require(type(max_frames) is int and 1 <= max_frames <= 1000, "registered frame bound required")
    require(0 < timeout_s <= 30., "bounded 0..30s worker timeout required")
    require(expected_sources == W.sources(), "CPU GT runtime source binding differs")
    require(not LG._INSTALLED, "CPU GT worker requires fresh retained composition")
    require(not getattr(G.install_complete_gt, "_cpu_gt_runtime", False), "nested CPU GT runtime forbidden")
    _require_no_parent_timing()
    attempt = Path(attempt_dir).resolve(strict=True)
    require(not (attempt / "GT_WORKER_SCOPE.json").exists()
        and not (attempt / "complete_gt_stage_timing").exists(), "GT runtime evidence is create-only")
    session = _Session(attempt, max_frames, expected_sources, timeout_s, _client_factory or W.Client)
    original_install = G.install_complete_gt

    @contextmanager
    def install(*, attempt_dir, max_frames):
        require(Path(attempt_dir).resolve() == attempt and max_frames == session.max_frames,
            "complete GT scope binding differs")
        with original_install(attempt_dir=attempt_dir, max_frames=max_frames) as state:
            _require_no_parent_timing()
            factory, write, preserve = LG.build_live_gt_collector, LG.write_avo_bundle, LG.preserve_avo_evidence
            catalogue = state["catalogue"]

            def remote_factory(base):
                inherited = factory(base)
                class RemoteCollector(inherited):
                    def __init__(self, **kwargs):
                        require(session.owner is None, "one collector per CPU GT scope required")
                        session.owner = self
                        try:
                            super().__init__(**kwargs)
                        except BaseException as exc:
                            session.fail(exc)
                            if session.client is not None:
                                session.client.abort()
                            raise

                    def _spawn_sensors(self):
                        session.start(self, catalogue)
                        return super()._spawn_sensors()

                    def _ground_truth(self, **kwargs):
                        require(getattr(session.local, "owner", None) is None, "nested CPU GT ownership")
                        session.local.owner, session.local.kwargs = self, kwargs
                        try:
                            return super()._ground_truth(**kwargs)
                        finally:
                            session.local.owner = session.local.kwargs = None
                return RemoteCollector

            def preserving(*args, **kwargs):
                session.before_preserve()
                result = preserve(*args, **kwargs)
                session.preserved = True
                return result

            LG.build_live_gt_collector, LG.write_avo_bundle, LG.preserve_avo_evidence = remote_factory, session.write, preserving
            try:
                yield state
            finally:
                LG.build_live_gt_collector, LG.write_avo_bundle, LG.preserve_avo_evidence = factory, write, preserve

    install._cpu_gt_runtime = True
    G.install_complete_gt = install
    failed = False
    try:
        yield session.state
    except BaseException as exc:
        failed = True
        session.fail(exc)
        raise
    finally:
        G.install_complete_gt = original_install
        try:
            if W.sources() != session.sources:
                session.fail(RuntimeErrorGT("CPU GT runtime sources changed"))
            session.cleanup()
            if not failed and not session.state["complete"]:
                raise RuntimeErrorGT("CPU GT runtime did not complete with joint drain and preserved evidence")
        except Exception as evidence_error:
            if not failed:
                raise
            print(f"CPU GT runtime evidence preservation failed: {evidence_error!r}",
                  file=sys.stderr, flush=True)
