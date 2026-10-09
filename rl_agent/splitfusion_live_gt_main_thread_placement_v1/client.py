"""Retain sealed IPC and explicit placement-lifecycle checks under new binding."""
from rl_agent.splitfusion_live_gt_placement_v1.client import Client as PlacementClient
from rl_agent.splitfusion_live_gt_optimization_v8.client import bound
from . import contract as C


class Client(PlacementClient):
    contract = C

    def _exchange(self, meta, body=b"", descriptor=None):
        return bound(PlacementClient._exchange, C)(self, meta, body=body, descriptor=descriptor)
