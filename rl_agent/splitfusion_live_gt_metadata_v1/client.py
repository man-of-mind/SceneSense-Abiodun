"""Preserve the real P4 worker, sealed IPC, ownership and native close protocol."""
from types import FunctionType
from rl_agent.splitfusion_live_gt_placement_trial_v1.client import Client as TrialClient
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from . import contract as C


class Client(TrialClient):
    contract = C

    def __init__(self, *, metadata_mode, **kwargs):
        W.require(type(metadata_mode) is str and metadata_mode in C.METADATA_MODES,
                  "explicit offline metadata mode required")
        W.require(kwargs.get("mode") == "p4", "both metadata arms require retained P4 scope")
        self.metadata_mode = metadata_mode
        super().__init__(**kwargs)

    def _exchange(self, meta, body=b"", descriptor=None):
        if meta.get("command") == "INIT":
            meta = dict(meta, metadata_mode=self.metadata_mode)
        function = TrialClient._exchange
        private = FunctionType(function.__code__, dict(function.__globals__, C=C),
                               function.__name__, function.__defaults__, function.__closure__)
        result = private(self, meta, body=body, descriptor=descriptor)
        context = (result["result"]["process_context"] if meta["command"] == "RUN"
                   else result["process_context"])
        if context["runtime_metadata"]["mode"] != self.metadata_mode:
            self.abort()
            raise W.WorkerError("foreign metadata intervention mode")
        return result
