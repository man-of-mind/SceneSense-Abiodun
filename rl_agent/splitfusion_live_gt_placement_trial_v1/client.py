"""Mode-bound INIT and native drain projection with a full authenticated receipt."""
import hashlib
import json
from pathlib import Path
from rl_agent.splitfusion_live_gt_optimization_v8.control_client import Client as ControlClient
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from . import contract as C


class Client(ControlClient):
    contract = C

    def __init__(self, *, mode, placement_cpu=None, **kwargs):
        W.require(type(mode) is str and mode in C.MODES, "registered trial mode required")
        W.require((mode == "control" and placement_cpu is None) or
                  (mode == "p4" and (placement_cpu is None or type(placement_cpu) is int
                                      and placement_cpu == C.TREATMENT_CPU)),
                  "trial placement CPU differs from registered mode")
        self.trial_mode = mode
        self.placement_cpu = C.TREATMENT_CPU if mode == "p4" else None
        self.original_parent_context = W.scheduling_context()
        W.require(mode == "control" or C.TREATMENT_CPU in self.original_parent_context["affinity"],
                  "P4 outside original allowed mask")
        self.worker_root = Path(kwargs["attempt_dir"])
        self.extended_handoff_close = self.extended_close_record = None
        super().__init__(**kwargs)

    def _exchange(self, meta, body=b"", descriptor=None):
        command = meta.get("command")
        if command == "INIT":
            W.require(meta["expected_process_context"] == self.original_parent_context,
                      "parent scheduling changed before trial initialization")
            meta = dict(meta, handoff_config=self.handoff_config, placement_mode=self.trial_mode,
                        placement_original_context=self.original_parent_context)
        try:
            result = W.Client._exchange(self, meta, body=body, descriptor=descriptor)
            context = (result.get("result", {}).get("process_context") if command == "RUN"
                       else result.get("process_context"))
            C.validate_context(context, self.bootstrap_sources)
            placement = context.get("runtime_placement", {})
            W.require(context.get("pid") == self.process.pid
                      and placement.get("worker_pid") == self.process.pid
                      and placement.get("main_tid") == self.process.pid
                      and placement.get("schema") == C.SCHEMA
                      and placement.get("trial_mode") == self.trial_mode
                      and placement.get("requested_cpu") == self.placement_cpu,
                      "foreign trial worker identity or mode")
            W.require(placement.get("original_scheduling") == self.original_parent_context
                      and context.get("original_parent_scheduling") == self.original_parent_context,
                      "trial original scheduling provenance differs")
            self.expected_context = (dict(self.original_parent_context, affinity=[C.TREATMENT_CPU])
                if self.trial_mode == "p4" and not placement.get("closed") else self.original_parent_context)
            W.require(context.get("scheduling") == self.expected_context,
                      "trial effective main scheduling differs")
            W.require(W.scheduling_context() == self.original_parent_context,
                      "parent scheduling changed during trial request")
            if command == "INIT":
                W.require(result.get("pid") == self.process.pid and result.get("cuda_initialized") is False
                          and not placement.get("closed"), "trial READY differs")
            if command == "CLOSE":
                W.require(placement.get("closed") and placement.get("restored"),
                          "trial main scheduling not restored before CLOSE")
            if command == "RUN":
                failed = result.get("result", {})
                if failed.get("status") == "FAILED" and isinstance(failed.get("failure_close"), dict):
                    self._preserve_close(dict(status="FAILED", timing=failed["failure_close"],
                        processed=failed.get("processed"), handled_failure=True,
                        process_context=context))
            return result
        except BaseException:
            self.abort()
            raise

    def _preserve_close(self, full):
        timing = dict(full["timing"])
        extensions = {name: timing.pop(name) for name in ("runtime_gc_scope", "runtime_placement_scope")}
        drain_path = self.worker_root / "engine/HANDOFF_DRAIN.json"
        drain = drain_path.read_bytes()
        W.require(timing == json.loads(drain), "native drain differs after exact extension projection")
        from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
        payload = W.canonical(dict(schema=C.SCHEMA, trial_mode=self.trial_mode, full_close=full,
            native_drain=dict(path=str(drain_path), sha256=hashlib.sha256(drain).hexdigest()),
            removed_extension_keys=sorted(extensions), extensions=extensions))
        path = self.worker_root / "TRIAL_WORKER_CLOSE.json"
        T._write_x(path, payload)
        self.extended_handoff_close = full
        self.extended_close_record = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
        self.handoff_close = dict(full, timing=timing)

    def close(self):
        result = super().close()
        if result is not None:
            self._preserve_close(result)
        return result
