"""Production Route-B seam over the GT-free v3 raw-spool bridge.

V4 adds the exact current-sweep radar-window handoff required by the policy
state.  It deliberately reuses the v3 queue, budget, raw-spool and post-route
materialization implementation; no GT transformation is introduced here.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from . import b_route_bridge_v3 as V3

BRouteBridgeError = V3.BRouteBridgeError
OpportunitySuperseded = V3.OpportunitySuperseded
RouteStopped = V3.RouteStopped
RouteFailed = V3.RouteFailed
RouteBudgetReached = V3.RouteBudgetReached
RouteOpportunityV4 = V3.RouteOpportunityV3
B_COLLECTOR_ROWS_NAME = "b_collector_rows.jsonl"
BRouteBridgeV4 = V3.BRouteBridgeV3
RawGroundTruthSpoolV4 = V3.RawGroundTruthSpoolV3
PrimitiveSceneSnapshotSourceV4 = V3.PrimitiveSceneSnapshotSourceV3


class _Counters:
    def snapshot(self) -> dict[str, int]:
        return {}


class BridgeLiveRuntimeV4:
    def __init__(self, bridge: BRouteBridgeV4, **_kwargs: Any) -> None:
        self.bridge, self.counters, self._metrics = bridge, _Counters(), {}
        self.scene_hook = None

    def submit(self, **kwargs: Any) -> Mapping[str, Any]:
        kwargs.pop("on_commit", None)
        if self.scene_hook is None:
            raise BRouteBridgeError("exact radar-window hook is not installed")
        kwargs["b_window_meta"] = self.scene_hook(
            float(kwargs["carla_timestamp"]))
        frame = int(kwargs["frame_id"])
        sent = self.bridge.offer_prepared(RouteOpportunityV4(
            sequence=self.bridge.transmitted, frame_id=frame,
            capture_timestamp_ns=int(kwargs["capture_timestamp_ns"]),
            action_open_monotonic_raw_ns=time.clock_gettime_ns(
                time.CLOCK_MONOTONIC_RAW), submit_kwargs=dict(kwargs)))
        self._metrics[frame] = {
            "payload_bytes": sent.payload_bytes,
            "identity_sha256": sent.identity.exact_sha256(),
        }
        return {"sent": True, "front_ms": "",
                "payload_bytes": sent.payload_bytes,
                "payload_bytes_uncompressed": "", "payload_chunks": ""}

    def take_metric(self, frame: int) -> Optional[Mapping[str, Any]]:
        row = self._metrics.get(int(frame))
        return None if row is None else dict(row)

    def close(self) -> Mapping[str, Any]:
        return {"errors": [], "transport_counters": {}}


def build_b_collector_class(base: type, bridge: BRouteBridgeV4) -> type:
    base_v3 = V3.build_b_collector_class(base, bridge)

    class BCollectorV4(base_v3):
        def __init__(self, **kwargs: Any) -> None:
            self._b_window_lock = threading.Lock()
            self._b_rows_lock = threading.Lock()
            self._b_windows: dict[float, Mapping[str, Any]] = {}
            super().__init__(**kwargs)
            original = self.aggregator.window_detections

            def recorded(*args: Any, **keywords: Any):
                detections, meta = original(*args, **keywords)
                key = float(keywords["reference_timestamp_s"])
                with self._b_window_lock:
                    self._b_windows[key] = meta
                    while len(self._b_windows) > 64:
                        del self._b_windows[next(iter(self._b_windows))]
                return detections, meta

            self.aggregator.window_detections = recorded
            self.live.scene_hook = self._b_take_window

        def _append_row(self, row: Mapping[str, Any]) -> None:
            # The pinned collector keeps per-frame status rows (including
            # drop reasons and worker failures) in memory until cleanup.
            # Stream each one durably so the evidence survives an abort.
            super()._append_row(row)
            path = Path(self.attempt_dir) / B_COLLECTOR_ROWS_NAME
            line = json.dumps(dict(row), sort_keys=True, default=str)
            with self._b_rows_lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()

        def _b_take_window(self, timestamp: float) -> Mapping[str, Any]:
            with self._b_window_lock:
                value = self._b_windows.pop(float(timestamp), None)
            if value is None:
                raise BRouteBridgeError("exact radar window is absent")
            return value

    return BCollectorV4


_SEAM_LOCK = threading.Lock()


@contextlib.contextmanager
def installed_b_route_seams(bridge: BRouteBridgeV4):
    from rl_agent import ue_map_install_feedback_v1 as feedback
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    if not _SEAM_LOCK.acquire(blocking=False):
        raise BRouteBridgeError("Route-B seams already installed")
    prior = (pinned.LivePilotCellRuntime, pinned.PassiveSplitCollector,
             pinned.SceneSnapshotSource, feedback.InstallFeedbackLedger)
    try:
        pinned.LivePilotCellRuntime = lambda **kw: BridgeLiveRuntimeV4(
            bridge, **kw)
        pinned.PassiveSplitCollector = build_b_collector_class(prior[1], bridge)
        pinned.SceneSnapshotSource = PrimitiveSceneSnapshotSourceV4
        feedback.InstallFeedbackLedger = V3.NoLegacyFeedbackLedgerV3
        yield pinned
    finally:
        (pinned.LivePilotCellRuntime, pinned.PassiveSplitCollector,
         pinned.SceneSnapshotSource, feedback.InstallFeedbackLedger) = prior
        _SEAM_LOCK.release()


def pinned_route_driver(route_kwargs: Mapping[str, Any]):
    frozen = dict(route_kwargs)
    def run(bridge: BRouteBridgeV4):
        with installed_b_route_seams(bridge) as pinned:
            return pinned.run_route_b(**frozen)
    return run


execute_300_with_postrun_gt = V3.execute_300_with_postrun_gt


__all__ = [
    "BRouteBridgeError", "OpportunitySuperseded", "RouteStopped",
    "RouteFailed", "RouteBudgetReached", "RouteOpportunityV4",
    "BRouteBridgeV4", "RawGroundTruthSpoolV4",
    "PrimitiveSceneSnapshotSourceV4", "BridgeLiveRuntimeV4",
    "build_b_collector_class", "installed_b_route_seams",
    "pinned_route_driver", "execute_300_with_postrun_gt",
]
