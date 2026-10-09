"""Preserve verified CPU startup/IPC; expose archival drain receipt at close."""
from types import FunctionType
from rl_agent.splitfusion_live_gt_optimization_v5.client import Client as V5Client
from . import contract as C


def _bound(function):
    result = FunctionType(function.__code__, dict(function.__globals__, C=C),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Client(V5Client):
    def __init__(self, *, handoff_config=None, **kwargs):
        self.handoff_config = handoff_config
        self.handoff_close = None
        _bound(V5Client.__init__)(self, **kwargs)

    def _exchange(self, meta, body=b"", descriptor=None):
        if meta.get("command") == "INIT":
            meta = dict(meta, handoff_config=self.handoff_config)
        result = _bound(V5Client._exchange)(self, meta, body=body, descriptor=descriptor)
        # The retained worker closes/drains before returning a handled RUN
        # failure. Client.run then aborts and cannot send a second CLOSE. Keep
        # that authentic drain receipt for preservation without changing the
        # FAILED status or pretending the request succeeded.
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
