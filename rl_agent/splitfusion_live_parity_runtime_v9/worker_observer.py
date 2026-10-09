"""Matched observation, preserving full and native CLOSE receipts separately."""
from pathlib import Path
from rl_agent.splitfusion_live_parity_runtime_v8 import worker_observer as V8
from rl_agent.splitfusion_live_gt_placement_trial_v1.client import Client
from . import contract as C
from .bindings import function, namespace

FACTORY = "rl_agent.splitfusion_live_parity_runtime_v9.worker_observer.ObservedClient"
archive_startup = V8.archive_startup


class ObservedClient(Client):
    def __init__(self, *, _observation, mode, **kwargs):
        self.observation = _observation
        self.close_record = self.archive_startup = None
        self.observation["clients"].append(self)
        try:
            super().__init__(mode=mode, placement_cpu=C.trial(mode)["placement_cpu"], **kwargs)
            self.archive_startup = archive_startup(self.process.pid)
        except BaseException:
            if getattr(self, "process", None) is not None and self.process.poll() is None:
                self.abort()
            raise

    _observe = V8.ObservedClient._observe

    def run(self, **kwargs):
        try:
            result = super().run(**kwargs)
        except BaseException as error:
            self._observe(kwargs.get("kwargs", {}).get("identity", {}),
                getattr(error, "worker_result", {}) or {}, f"{type(error).__name__}: {error}")
            raise
        self._observe(result["identity"], result)
        return result

    def close(self):
        result = super().close()
        if result is not None:
            self.close_record = dict(self.handoff_close,
                extended_close_receipt=dict(self.extended_close_record))
        return result


def observe_cpu_worker(*, attempt_dir, **kwargs):
    from .host_capture import verify_prospective
    record = verify_prospective(Path(attempt_dir).resolve().parents[2])
    mode = record["trial"]["mode"]
    class Bound(ObservedClient):
        def __init__(self, **options):
            super().__init__(mode=mode, **options)
    local = namespace(V8, H=C.WORKER, ObservedClient=Bound, FACTORY=FACTORY)
    return local["observe_cpu_worker"](attempt_dir=attempt_dir, **kwargs)
