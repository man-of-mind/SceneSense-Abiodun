"""Source-bound, reversible input observer; no launch authorization implied."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import threading

from .capture import InputArchive, sha

PINS = {
    "rl_agent/splitfusion_live_parity_repair_v1/perception.py":
        "e8eb43a89f90809731a3af233239f61492650cdfc23288ea0357768a7158c792",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_ue_runtime_v2.py":
        "c97103645d4e55dea3a35a7be8b942ec15d1320223d83996c04bcb5d5ebc3e66",
}
_LOCK = threading.Lock()
_RETURN = "return fused.unsqueeze(0).to(device=device, dtype=torch.float32)"
_CAPTURE = "return _paired_input_observer(fused.unsqueeze(0), device).to(device=device, dtype=torch.float32)"


def retained_packing(original, observer):
    """Only insert an identity observer before the existing CPU->device copy."""
    source = inspect.getsource(original)
    if source.count(_RETURN) != 1 or "_paired_input_observer" in source:
        raise ValueError("unsupported or already instrumented input preparation")
    namespace = dict(original.__globals__, _paired_input_observer=observer)
    exec(compile(source.replace(_RETURN, _CAPTURE), "<bound-paired-input-retention>", "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def install_capture(archive: InputArchive):
    """Install outside the corrected UE perception scope, restore after drain.

    This module deliberately has no CLI. A new prospective live registration
    must bind it, preserve the existing actor/GT/170ms rules, and verify both
    worker drains before calling archive.flush(). Never inject it into v5's
    already sealed capture configuration.
    """
    if not _LOCK.acquire(blocking=False):
        raise ValueError("nested input retention scope")
    try:
        from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        root = Path(__file__).resolve().parents[2]
        if any(sha(root/name) != digest for name, digest in PINS.items()):
            raise ValueError("source-bound retention prerequisite drift")
        packing, materialize = P.prepare_live_input, U.Run4FramePipelineV2.materialize
        replacement = retained_packing(packing, archive.observe_input)

        def observed(self, planned, *args, **kwargs):
            record = prepared = None
            try:
                try:
                    record = archive.begin(planned)
                except Exception as exc:
                    archive.fault(exc)
                prepared = materialize(self, planned, *args, **kwargs)
                return prepared
            finally:
                # Never turn observer failure into a transport/agent outcome.
                try:
                    archive.end(record, prepared)
                except Exception as exc:
                    archive.fault(exc)

        P.prepare_live_input = replacement
        U.Run4FramePipelineV2.materialize = observed
        try:
            yield {"sources": dict(PINS), "max_frames": archive.max_frames,
                   "max_bytes": archive.max_bytes, "new_threads": 0,
                   "extra_gpu_copies": 0, "live_file_writes": 0,
                   "retention_overhead_not_yet_measured": True}
        finally:
            P.prepare_live_input = packing
            U.Run4FramePipelineV2.materialize = materialize
    finally:
        _LOCK.release()
