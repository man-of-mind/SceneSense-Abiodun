"""Measure the complete GT stages without changing their work or readiness.

Tiny bounded in-memory records only while the worker is active; serialization
and fsync run on scope exit after the retained collector has drained. No new
thread, queue, cache, source conversion, or deadline semantics are introduced.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

SCHEMA = "splitfusion.live_parity_repair.gt_stage_timing.v2"


def _frame(args, kwargs):
    identity = kwargs.get("identity")
    if isinstance(identity, dict):
        return identity.get("frame_id")
    inputs = kwargs.get("inputs")
    if inputs is None:
        inputs = next((arg for arg in args if isinstance(getattr(arg, "frame", None), dict)), None)
    return None if inputs is None else inputs.frame.get("frame_id")


class StageLog:
    def __init__(self, max_frames):
        if type(max_frames) is not int or not 1 <= max_frames <= 1000:
            raise ValueError("explicit 1..1000 stage frame budget required")
        self.limit = max_frames * 8
        self.rows = []
        self.dropped = 0
        self.lock = threading.Lock()
        self.closed = False

    def wrap(self, stage, original):
        @wraps(original)
        def timed(*args, **kwargs):
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            cpu = time.thread_time_ns()
            success = False
            try:
                result = original(*args, **kwargs)
                success = True
                return result
            finally:
                end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                cpu_end = time.thread_time_ns()
                # Measurement errors must never replace a real GT exception.
                try:
                    with self.lock:
                        if self.closed or len(self.rows) >= self.limit:
                            self.dropped += 1
                        else:
                            self.rows.append(dict(stage=stage, frame_id=_frame(args, kwargs),
                                start_raw_ns=start, end_raw_ns=end, wall_ns=end-start,
                                thread_cpu_ns=cpu_end-cpu, completed=success))
                except Exception:
                    self.dropped += 1
        return timed

    def flush(self, directory, *, worker_drained):
        from rl_agent.splitfusion_live_parity_runtime_v2.startup import write_x
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        if type(worker_drained) is not bool:
            raise ValueError("actual worker-drain observation required")
        with self.lock:
            if self.closed:
                raise ValueError("GT stage evidence already flushed")
            self.closed = True
            rows, dropped = list(self.rows), self.dropped
        raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()+b"\n"
                       for row in rows)
        write_x(directory / "stages.jsonl", raw)
        result = dict(schema=SCHEMA, records=len(rows), dropped=dropped,
            bound=self.limit, clock="CLOCK_MONOTONIC_RAW", cpu_clock="thread_time_ns",
            flush_after_worker_drain=worker_drained, work_and_readiness_unchanged=True,
            sha256={"stages.jsonl": hashlib.sha256(raw).hexdigest()})
        write_x(directory / "MANIFEST.json", result)
        return result


@contextmanager
def install_gt_timing(*, attempt_dir, max_frames, drain_check):
    from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge, source_geometry, evaluation, target_transport
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt
    log = StageLog(max_frames)
    directory = Path(attempt_dir) / "complete_gt_stage_timing"
    if directory.exists():
        raise FileExistsError(directory)
    bindings = [
        (live_gt, "write_avo_bundle", "avo_scoring_and_sidecar"),
        (source_geometry.LiveCompleteTargetProducer, "build", "complete_source_build"),
        (evaluation, "prepare_target", "prepare_complete_target"),
        (target_transport, "write_bundle", "publish_complete_target"),
        (gt_bridge, "write_source_archive", "archive_auxiliary_sources"),
    ]
    originals = [(owner, name, getattr(owner, name), stage) for owner, name, stage in bindings]
    for owner, name, original, stage in originals:
        setattr(owner, name, log.wrap(stage, original))
    failed = False
    try:
        yield log
    except BaseException:
        failed = True
        raise
    finally:
        for owner, name, original, _stage in reversed(originals):
            setattr(owner, name, original)
        try:
            log.flush(directory, worker_drained=drain_check())
        except Exception as exc:
            if not failed:
                raise
            print(f"GT stage timing preservation failed: {exc!r}", file=sys.stderr, flush=True)
