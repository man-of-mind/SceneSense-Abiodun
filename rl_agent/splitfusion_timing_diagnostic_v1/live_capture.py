#!/usr/bin/env python3
"""Live Route-B capture wrappers for the timing diagnostic.

The qualified Route-B split-cell adapter already owns the live contract this
amendment requires: the Town10HD_Opt route, seed 31 / traffic-manager seed 31,
the 50-vehicle / 50-pedestrian density, the RGB + semantic + radar sensor rig,
the every-other-tick 10 Hz preparation gating, the four-callback radar-window
contract and the preparation/stale drop taxonomy. That adapter is SHA-256
pinned by the campaign config, so nothing here edits it. Everything is an
additive wrapper installed at run time:

* :class:`TimingSocket` wraps only the UE's feature-uplink socket and stamps
  the wall clock immediately before the first datagram of a frame leaves and
  immediately after the last one does. Sensor preparation and CARLA waiting
  are therefore structurally outside every uplink interval.
* :class:`DiagnosticLivePilotCellRuntime` subclasses the deployed UE runtime
  to install that socket and retain the boundaries per frame.
* :class:`BudgetedDiagnosticCollector` subclasses the deployed collector to
  stop at exactly the configured number of successfully transmitted frames,
  to enforce the safety timeout, and to route the evaluation-only queues to a
  discarding sink so segmentation-quality scoring, object ground truth and
  exact-record retrieval never run. Map install and install feedback -- real
  deployment functions -- keep running, so the deployed tail service span
  still includes deployment contention and downstream post-processing.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from typing import Any, Mapping

from . import diagnostic_common as common


class RouteBudgetReached(RuntimeError):
    """The transmitted-frame budget or the safety timeout ended the route."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


class TimingSocket:
    """Delegating UDP socket that stamps a frame's first and last send.

    Only the feature-uplink socket is wrapped. ``sendto`` is the exact
    boundary the amendment names: application-level uplink timing starts
    immediately before the first datagram is sent, so nothing upstream of the
    socket -- sensor assembly, radar rasterisation, CARLA waiting, front or AE
    compute -- can enter an uplink interval.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._first_ns = 0
        self._last_ns = 0
        self._count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def begin_frame(self) -> None:
        self._first_ns = 0
        self._last_ns = 0
        self._count = 0

    def sendto(self, payload: Any, address: Any) -> Any:
        if self._first_ns == 0:
            self._first_ns = time.time_ns()
        result = self._delegate.sendto(payload, address)
        self._last_ns = time.time_ns()
        self._count += 1
        return result

    def snapshot(self) -> dict[str, int]:
        return {
            "ue_first_send_wall_ns": int(self._first_ns),
            "ue_final_send_wall_ns": int(self._last_ns),
            "ue_datagrams_observed": int(self._count),
        }


def build_runtime_class(base: type) -> type:
    """Subclass the deployed UE runtime, adding only send-boundary stamps."""

    class DiagnosticLivePilotCellRuntime(base):  # type: ignore[misc, valid-type]
        def __init__(self, **keywords: Any) -> None:
            super().__init__(**keywords)
            self.timing_socket = TimingSocket(self.sender)
            self.sender = self.timing_socket
            self.send_boundaries: dict[int, dict[str, int]] = {}
            self._boundary_lock = threading.Lock()

        def submit(self, **keywords: Any) -> dict[str, Any]:
            self.timing_socket.begin_frame()
            result = super().submit(**keywords)
            snapshot = self.timing_socket.snapshot()
            frame_id = int(keywords["frame_id"])
            if result.get("sent") and snapshot["ue_first_send_wall_ns"]:
                common.require(
                    snapshot["ue_final_send_wall_ns"]
                    >= snapshot["ue_first_send_wall_ns"],
                    "negative UE send-loop interval",
                )
                with self._boundary_lock:
                    self.send_boundaries[frame_id] = snapshot
                with self.lock:
                    if frame_id in self.metrics:
                        self.metrics[frame_id].update(snapshot)
            return result

    return DiagnosticLivePilotCellRuntime


class _DiscardingQueue(queue.Queue):
    """Pass shutdown sentinels through; discard evaluation-only work.

    The deployed collector enqueues segmentation-quality, object-ground-truth
    and exact-installed-record tickets. Those are offline scoring, not
    deployment, and their CPU/GPU cost would inflate the deployed tail service
    span this diagnostic reports. Discarding the tickets leaves the deployed
    code path and its shutdown handshake untouched: ``None`` sentinels still
    reach the workers, and ``unfinished_tasks`` stays at zero so every drain
    in ``finish()`` returns immediately.
    """

    def __init__(self) -> None:
        super().__init__(maxsize=8)
        self.discarded = 0

    def put_nowait(self, item: Any) -> None:
        if item is None:
            super().put_nowait(item)
            return
        self.discarded += 1

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        if item is None:
            super().put(item, block, timeout)
            return
        self.discarded += 1


class GpuSampler:
    """Bounded GPU utilization/contention sampling for the live cell."""

    QUERY = (
        "utilization.gpu", "utilization.memory", "memory.used",
        "temperature.gpu", "clocks_throttle_reasons.active",
    )

    def __init__(self, *, interval_s: float = 0.5) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.foreign_process_samples: list[int] = []
        self.error = ""

    def _sample_once(self) -> None:
        completed = subprocess.run(
            (
                "nvidia-smi", f"--query-gpu={','.join(self.QUERY)}",
                "--format=csv,noheader,nounits",
            ),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False, timeout=10.0,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return
        fields = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        if len(fields) < 4:
            return
        row: dict[str, Any] = {"wall_ns": time.time_ns()}
        for name, value in zip(self.QUERY, fields):
            if name == "clocks_throttle_reasons.active":
                row[name] = value
                continue
            try:
                row[name] = float(value)
            except ValueError:
                row[name] = None
        self.samples.append(row)
        apps = subprocess.run(
            ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False, timeout=10.0,
        )
        if apps.returncode == 0:
            self.foreign_process_samples.append(
                len([line for line in apps.stdout.splitlines() if line.strip()])
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="diag-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "interval_s": self.interval_s,
            "error": self.error,
            "utilization_gpu_percent": common.summarize(
                [row["utilization.gpu"] for row in self.samples
                 if row.get("utilization.gpu") is not None]
            ),
            "utilization_memory_percent": common.summarize(
                [row["utilization.memory"] for row in self.samples
                 if row.get("utilization.memory") is not None]
            ),
            "memory_used_mib": common.summarize(
                [row["memory.used"] for row in self.samples
                 if row.get("memory.used") is not None]
            ),
            "temperature_c": common.summarize(
                [row["temperature.gpu"] for row in self.samples
                 if row.get("temperature.gpu") is not None]
            ),
            "concurrent_compute_processes": common.summarize(self.foreign_process_samples),
            "throttle_reasons_observed": sorted(
                {
                    str(row.get("clocks_throttle_reasons.active", ""))
                    for row in self.samples
                    if str(row.get("clocks_throttle_reasons.active", "")).strip()
                }
            ),
        }


def build_collector_class(base: type) -> type:
    """Subclass the deployed collector: bounded budget, evaluation off."""

    class BudgetedDiagnosticCollector(base):  # type: ignore[misc, valid-type]
        transmitted_budget = 300
        safety_timeout_s = 90.0

        def __init__(self, **keywords: Any) -> None:
            super().__init__(**keywords)
            # Deployment functions (map install, install feedback) keep
            # running; evaluation-only scoring is discarded at the queue.
            self.segmentation_queue = _DiscardingQueue()
            self.evaluation_queue = _DiscardingQueue()
            self.exact_retrieval_queue = _DiscardingQueue()
            self.diagnostic_stop_reason = ""
            self.route_started_wall_ns = 0
            self.budget_reached_wall_ns = 0
            self.route_stopped_wall_ns = 0
            self.first_tick_monotonic: float | None = None
            self._deadline_monotonic: float | None = None
            self._stop_requested = False
            self.ticks_observed = 0

        # -- budget control -------------------------------------------------

        def _request_stop(self, reason: str) -> None:
            if not self._stop_requested:
                self._stop_requested = True
                self.diagnostic_stop_reason = str(reason)

        def on_world_tick(self, frame_id: int, route_tick: int) -> None:
            self.ticks_observed += 1
            if self.first_tick_monotonic is None:
                self.first_tick_monotonic = time.monotonic()
                self._deadline_monotonic = (
                    self.first_tick_monotonic + self.safety_timeout_s
                )
                self.route_started_wall_ns = time.time_ns()
            if self._stop_requested:
                self.route_stopped_wall_ns = time.time_ns()
                raise RouteBudgetReached(self.diagnostic_stop_reason)
            if self.sent >= self.transmitted_budget:
                self._request_stop("TRANSMITTED_BUDGET_REACHED")
                self.route_stopped_wall_ns = time.time_ns()
                raise RouteBudgetReached(self.diagnostic_stop_reason)
            if (
                self._deadline_monotonic is not None
                and time.monotonic() >= self._deadline_monotonic
            ):
                self._request_stop("SAFETY_TIMEOUT_EXPIRED")
                self.route_stopped_wall_ns = time.time_ns()
                raise RouteBudgetReached(self.diagnostic_stop_reason)
            super().on_world_tick(frame_id, route_tick)

        def _process_token(self, token: Mapping[str, Any]) -> None:
            if self.sent >= self.transmitted_budget:
                self.transport_counters.bump("preparation_skipped_after_budget")
                return
            before = int(self.sent)
            super()._process_token(token)
            if int(self.sent) > before and int(self.sent) >= self.transmitted_budget:
                self.budget_reached_wall_ns = time.time_ns()
                self._request_stop("TRANSMITTED_BUDGET_REACHED")

        # -- reporting ------------------------------------------------------

        def diagnostic_summary(self) -> dict[str, Any]:
            return {
                "transmitted_budget": int(self.transmitted_budget),
                "safety_timeout_s": float(self.safety_timeout_s),
                "transmitted_frames": int(self.sent),
                "reached_budget": bool(self.sent >= self.transmitted_budget),
                "stop_reason": self.diagnostic_stop_reason,
                "preparation_opportunities_dropped": int(self.dropped),
                "route_ticks_observed": int(self.ticks_observed),
                "route_started_wall_ns": int(self.route_started_wall_ns),
                "budget_reached_wall_ns": int(self.budget_reached_wall_ns),
                "route_stopped_wall_ns": int(self.route_stopped_wall_ns),
                "route_wall_seconds": (
                    (self.route_stopped_wall_ns - self.route_started_wall_ns) / 1e9
                    if self.route_started_wall_ns and self.route_stopped_wall_ns
                    else None
                ),
                "transport_counters": self.transport_counters.snapshot(),
                "evaluation_scaffolding": {
                    "segmentation_quality_scoring": "DISABLED_FOR_DIAGNOSTIC",
                    "object_ground_truth": "DISABLED_FOR_DIAGNOSTIC",
                    "exact_installed_record_retrieval": "DISABLED_FOR_DIAGNOSTIC",
                    "map_install": "ENABLED_DEPLOYMENT_FUNCTION",
                    "install_feedback": "ENABLED_DEPLOYMENT_FUNCTION",
                    "discarded_segmentation_tickets": int(
                        getattr(self.segmentation_queue, "discarded", 0)
                    ),
                    "discarded_object_gt_tickets": int(
                        getattr(self.evaluation_queue, "discarded", 0)
                    ),
                    "discarded_exact_record_tickets": int(
                        getattr(self.exact_retrieval_queue, "discarded", 0)
                    ),
                },
                "send_boundaries": dict(getattr(self.live, "send_boundaries", {})),
                "failures": list(self.failures),
                "cleanup_ok": bool(self.cleanup_ok),
            }

    return BudgetedDiagnosticCollector
