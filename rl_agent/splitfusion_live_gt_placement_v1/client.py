"""v8 CONTROL client with explicit child-only INIT placement and attestation."""
from rl_agent.splitfusion_live_gt_optimization_v8.control_client import Client as ControlClient
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from . import contract as C


class Client(ControlClient):
    contract = C

    def __init__(self, *, placement_cpu, **kwargs):
        W.require(type(placement_cpu) is int and placement_cpu >= 0, "integer placement CPU required")
        self.placement_cpu = placement_cpu
        self.original_parent_context = W.scheduling_context()
        W.require(placement_cpu in self.original_parent_context["affinity"],
                  "placement CPU outside original allowed mask")
        super().__init__(**kwargs)

    def _exchange(self, meta, body=b"", descriptor=None):
        command = meta.get("command")
        if command == "INIT":
            W.require(meta["expected_process_context"] == self.original_parent_context,
                      "parent scheduling changed before worker initialization")
            meta = dict(meta, handoff_config=self.handoff_config, placement_cpu=self.placement_cpu,
                        placement_original_context=self.original_parent_context)
        try:
            # This is the unchanged V4 wire exchange used by V5/v8. The V5
            # inherited-mask validator is replaced only by explicit placement
            # lifecycle validation; it must not label changed affinity inherited.
            result = W.Client._exchange(self, meta, body=body, descriptor=descriptor)
            context = (result.get("result", {}).get("process_context") if command == "RUN"
                       else result.get("process_context"))
            C.validate_context(context, self.bootstrap_sources)
            W.require(context.get("pid") == self.process.pid, "foreign GT placement process")
            placement = context.get("runtime_placement", {})
            W.require(placement.get("schema") == C.SCHEMA
                      and placement.get("worker_pid") == self.process.pid
                      and placement.get("main_tid") == self.process.pid
                      and placement.get("requested_cpu") == self.placement_cpu,
                      "GT placement identity differs")
            W.require(placement.get("original_scheduling") == self.original_parent_context
                      and context.get("original_parent_scheduling") == self.original_parent_context,
                      "original scheduling provenance differs")
            self.expected_context = (self.original_parent_context if placement.get("closed") else
                                     dict(self.original_parent_context, affinity=[self.placement_cpu]))
            W.require(context.get("scheduling") == self.expected_context,
                      "GT effective placement scheduling differs")
            W.require(W.scheduling_context() == self.original_parent_context,
                      "parent scheduling changed during GT request")
            if command == "INIT":
                W.require(result.get("pid") == self.process.pid and result.get("cuda_initialized") is False
                          and not placement.get("closed"), "GT placement READY differs")
            if command == "CLOSE":
                W.require(placement.get("closed") and placement.get("restored"),
                          "GT placement not restored before CLOSE")
            if command == "RUN":
                failed = result.get("result", {})
                if failed.get("status") == "FAILED" and isinstance(failed.get("failure_close"), dict):
                    self.handoff_close = dict(status="FAILED", timing=failed["failure_close"],
                        processed=failed.get("processed"), handled_failure=True)
            return result
        except BaseException:
            self.abort()
            raise
