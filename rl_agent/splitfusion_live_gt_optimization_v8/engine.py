"""All numerical GT work remains the sealed handoff Engine's own methods."""
from rl_agent.splitfusion_live_gt_handoff_v1.engine import Engine as HandoffEngine
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from .gc_policy import GCPolicy


class Engine(HandoffEngine):
    gc_mode = "freeze"

    def __init__(self, **kwargs):
        self.gc_policy = None
        self.gc_scope_record = None
        super().__init__(**kwargs)
        try:
            if (self.archiver is None or self.archiver.process.poll() is not None
                    or self.count != 0 or self.owner._complete_producer is not None):
                raise ValueError("full startup and archive child READY required before GC policy")
            self.gc_policy = GCPolicy(self.gc_mode)
            self.gc_policy.set_phase("idle_before_first_frame", 0)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def process(self, **kwargs):
        self.gc_policy.set_phase("GT_PROCESS", self.count)
        try:
            # Includes the exact original geometry, history, validation, packet
            # creation, publication, archive enqueue and failure semantics.
            return super().process(**kwargs)
        finally:
            if not self.gc_policy.closed:
                self.gc_policy.set_phase("IDLE_OR_IPC", self.count)

    def close(self):
        result = None
        native_error = cleanup_error = None
        try:
            result = super().close()
        except BaseException as error:
            native_error = error
        policy = self.gc_policy
        if policy is not None and not policy.closed:
            # Preserve native drain errors even if restoration itself detects
            # a fault. Both outcomes are captured; neither is a success.
            try:
                policy.close()
            except BaseException as error:
                cleanup_error = error
            try:
                snapshot = policy.snapshot(include_events=True)
                snapshot.update(native_close_succeeded=native_error is None,
                    native_close_error=None if native_error is None else repr(native_error),
                    cleanup_error=None if cleanup_error is None else repr(cleanup_error))
                payload = W.canonical(snapshot)
                path = self.attempt/"GC_SCOPE.json"
                T._write_x(path, payload)
                import hashlib
                self.gc_scope_record = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                    settings_restored=snapshot["settings_restored"],
                    native_close_succeeded=native_error is None,
                    exact_gc_generations_restored=False)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if native_error is not None:
            if cleanup_error is not None:
                raise native_error from cleanup_error
            raise native_error
        if cleanup_error is not None:
            raise cleanup_error
        if result is not None and self.gc_scope_record is not None:
            result["runtime_gc_scope"] = self.gc_scope_record
        return result


class ControlEngine(Engine):
    gc_mode = "control"
