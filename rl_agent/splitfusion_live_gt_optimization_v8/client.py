"""Reuse the sealed IPC and drain protocols; vary only registered bootstrap."""
from types import FunctionType
from rl_agent.splitfusion_live_gt_optimization_v5.client import Client as V5Client
from . import contract as C


def bound(function, contract):
    result = FunctionType(function.__code__, dict(function.__globals__, C=contract),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Client(V5Client):
    contract = C

    def __init__(self, *, handoff_config=None, **kwargs):
        if handoff_config is None:
            raise ValueError("GC experiment requires archive READY before initial collection")
        self.handoff_config = handoff_config
        self.handoff_close = None
        bound(V5Client.__init__, self.contract)(self, **kwargs)

    def _exchange(self, meta, body=b"", descriptor=None):
        if meta.get("command") == "INIT":
            meta = dict(meta, handoff_config=self.handoff_config)
        result = bound(V5Client._exchange, self.contract)(self, meta, body=body, descriptor=descriptor)
        if meta.get("command") == "RUN":
            failed = result.get("result", {})
            if failed.get("status") == "FAILED" and isinstance(failed.get("failure_close"), dict):
                self.handoff_close = dict(status="FAILED", timing=failed["failure_close"],
                    processed=failed.get("processed"), handled_failure=True)
        return result

    def close(self):
        result = super().close()
        if result is not None:
            self.handoff_close = result
        return result
