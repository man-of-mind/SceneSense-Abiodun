"""UE-side live wrappers for explicit edge-scheduler terminal feedback."""

from __future__ import annotations

import json
import csv
import socket
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    DeadlineExpired,
    EDGE_RESULT_SCHEMA,
    EDGE_TERMINAL_ACK_SCHEMA,
    OBJECT_MAP_UPDATE_SCHEMA,
    UE_STAGE_BEFORE_MAP_PUBLICATION,
    _require,
    check_deadline,
)
from rl_agent.splitfusion_timing_diagnostic_v1 import live_capture as diagnostic

from .scheduler import SCHEMA as SCHEDULER_FEEDBACK_SCHEMA


def build_scheduler_runtime_class(base: type) -> type:
    """Accept normal results plus explicit non-install scheduler terminals."""

    class SchedulerAwareRuntime(base):  # type: ignore[misc, valid-type]
        def __init__(self, **keywords: Any) -> None:
            self.scheduler_feedback_callback: Any = None
            self.scheduler_feedback_records: list[dict[str, Any]] = []
            self.scheduler_feedback_lock = threading.Lock()
            super().__init__(**keywords)

        def _result_loop(self) -> None:
            expired_seen = 0
            while not self.stop_event.is_set():
                try:
                    datagram, address = self.receiver.recvfrom(65535)
                except socket.timeout:
                    self.reassembler.expire(time.monotonic())
                    if self.reassembler.expired_messages != expired_seen:
                        self.counters.bump(
                            "result_incomplete_reassemblies_expired",
                            self.reassembler.expired_messages - expired_seen,
                        )
                        expired_seen = self.reassembler.expired_messages
                    continue
                except OSError:
                    return
                received_ns = time.perf_counter_ns()
                received_wall = time.time()
                self.counters.bump("result_datagrams_received")
                try:
                    complete = self.reassembler.ingest(
                        str(address), datagram, received_at_s=time.monotonic()
                    )
                    if complete is None:
                        continue
                    self.counters.bump("result_messages_reassembled")
                    value = json.loads(complete.payload.decode("utf-8"))
                    if value.get("schema") == SCHEDULER_FEEDBACK_SCHEMA:
                        self._accept_scheduler_terminal(
                            value,
                            message_id=int(complete.message_id),
                            received_ns=received_ns,
                            received_wall=received_wall,
                        )
                        continue
                    _require(
                        value.get("schema") == EDGE_RESULT_SCHEMA,
                        "edge result schema drift",
                    )
                    _require(
                        int(value["frame_id"]) == complete.message_id,
                        "result chunk/frame identity drift",
                    )
                    frame_id = int(value["frame_id"])
                    metric = self.metrics.get(frame_id)
                    _require(metric is not None, "edge result has no transmitted UE frame")
                    _require(
                        int(value["action_id"]) == int(metric["action_id"]),
                        "edge action identity drift",
                    )
                    _require(
                        str(value["profile_id"]) == str(metric["profile_id"]),
                        "edge profile identity drift",
                    )
                    _require(
                        str(value["stream_id"]) == str(metric["stream_id"]),
                        "edge stream identity drift",
                    )
                    _require(
                        "semantic_labels_b64" not in value,
                        "dense evaluation label map must not ride the radio return path",
                    )
                    update = value["object_map_update"]
                    _require(
                        update.get("schema") == OBJECT_MAP_UPDATE_SCHEMA
                        and int(update["frame_id"]) == frame_id
                        and str(update["stream_id"]) == str(metric["stream_id"]),
                        "object map update schema/identity drift",
                    )
                    terminal = value["edge_terminal_ack"]
                    _require(
                        terminal.get("schema") == EDGE_TERMINAL_ACK_SCHEMA
                        and int(terminal["frame_id"]) == frame_id,
                        "edge terminal ACK schema/identity drift",
                    )
                    with self.lock:
                        duplicate = frame_id in self._published_frames
                    if duplicate:
                        self.counters.bump("duplicate_result_messages")
                        continue
                    capture_timestamp_ns = int(value["capture_timestamp_ns"])
                    try:
                        check_deadline(
                            UE_STAGE_BEFORE_MAP_PUBLICATION,
                            capture_timestamp_ns,
                            self.deadline_s,
                            now_s=received_wall,
                        )
                    except DeadlineExpired as expired:
                        self.counters.bump(f"deadline_drop_{expired.stage}")
                        self.counters.bump("results_expired_before_map_publication")
                        with self.lock:
                            metric.update(
                                self._receipt_fields(
                                    value,
                                    complete,
                                    received_ns,
                                    received_wall,
                                    terminal,
                                )
                            )
                            metric["map_publication_status"] = (
                                "EXPIRED_BEFORE_PUBLICATION"
                            )
                            metric["deadline_expiry_stage"] = expired.stage
                            metric["deadline_expiry_age_ms"] = expired.age_ms
                            self.completed += 1
                        continue
                    published = {
                        "schema": "fusion_object_spatial_map.v1",
                        "stream_id": value["stream_id"],
                        "frame_id": frame_id,
                        "capture_id": metric["capture_id"],
                        "capture_timestamp": capture_timestamp_ns / 1e9,
                        "action_id": str(value["action_id"]),
                        "carla_timestamp": metric["carla_timestamp"],
                        "objects": update["records"],
                        "segmentation": {
                            "available": True,
                            "evidence": dict(terminal.get("evidence") or {}),
                            "installation_status": str(
                                terminal.get("installation_status") or ""
                            ),
                        },
                        "timing": {
                            "t_edge_recv_perf": float(value["edge_received_ns"]) / 1e9,
                            "t_tail_done_perf": float(value["tail_finished_ns"]) / 1e9,
                            "t_map_publish_perf": time.perf_counter(),
                        },
                    }
                    self.map_socket.sendto(
                        zlib.compress(
                            json.dumps(
                                published,
                                allow_nan=False,
                                separators=(",", ":"),
                            ).encode("utf-8"),
                            level=1,
                        ),
                        self.map_remote,
                    )
                    self.counters.bump("results_published_to_map")
                    with self.lock:
                        self._published_frames.add(frame_id)
                        metric.update(
                            self._receipt_fields(
                                value,
                                complete,
                                received_ns,
                                received_wall,
                                terminal,
                            )
                        )
                        metric["map_publication_status"] = "PUBLISHED"
                        self.completed += 1
                except Exception as exc:
                    with self.lock:
                        self.errors.append(f"{type(exc).__name__}: {exc}")
                    return

        def _accept_scheduler_terminal(
            self,
            value: Mapping[str, Any],
            *,
            message_id: int,
            received_ns: int,
            received_wall: float,
        ) -> None:
            frame_id = int(value["frame_id"])
            _require(frame_id == message_id, "scheduler chunk/frame identity drift")
            metric = self.metrics.get(frame_id)
            _require(metric is not None, "scheduler terminal has no UE frame")
            _require(
                int(value["action_id"]) == int(metric["action_id"])
                and str(value["profile_id"]) == str(metric["profile_id"])
                and str(value["stream_id"]) == str(metric["stream_id"]),
                "scheduler terminal identity drift",
            )
            _require(
                str(value["terminal_reason"]) != "RESULT_PUBLISHED",
                "result publication must use the normal map-install path",
            )
            callback = self.scheduler_feedback_callback
            _require(callable(callback), "scheduler feedback callback is unavailable")
            callback(value, metric["capture_id"])
            with self.scheduler_feedback_lock:
                self.scheduler_feedback_records.append(dict(value))
            with self.lock:
                metric.update(
                    {
                        "edge_result_received_ns": int(received_ns),
                        "feature_received_at": float(received_wall),
                        "edge_scheduler_terminal_reason": str(
                            value["terminal_reason"]
                        ),
                        "edge_scheduler_outcome_class": str(
                            value["outcome_class"]
                        ),
                        "edge_scheduler_stage": str(value["stage"]),
                        "edge_scheduler_agent_credit": dict(
                            value.get("agent_credit") or {}
                        ),
                        "map_publication_status": (
                            "EDGE_TERMINATED_WITHOUT_MAP_PUBLICATION"
                        ),
                    }
                )
                self.completed += 1
            self.counters.bump("scheduler_terminal_feedback_received")

    return SchedulerAwareRuntime


def build_scheduler_collector_class(base: type) -> type:
    """Forward an edge terminal into the existing authoritative UE ledger."""

    class SchedulerAwareCollector(base):  # type: ignore[misc, valid-type]
        def __init__(self, **keywords: Any) -> None:
            super().__init__(**keywords)
            self._scheduler_feedback_socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            )
            self.live.scheduler_feedback_callback = self._scheduler_terminal

        def _scheduler_terminal(
            self, value: Mapping[str, Any], capture_id: str
        ) -> None:
            reason = str(value["terminal_reason"])
            outcome_class = str(value["outcome_class"])
            message = {
                "schema": "scenesense.map_install_feedback.v1",
                "experiment_id": str(self.campaign["campaign_id"]),
                "cell_id": str(self.cell["cell_id"]),
                "stream_id": str(value["stream_id"]),
                "capture_id": str(capture_id),
                "frame_id": int(value["frame_id"]),
                "capture_timestamp": int(value["capture_timestamp_ns"]) / 1e9,
                "action_id": str(value["action_id"]),
                "install_timestamp": "",
                "feedback_emit_at": time.time(),
                "result_status": "EDGE_SCHEDULER_TERMINATED_WITHOUT_INSTALL",
                "status": "NACK_REJECTED",
                "rejection_reason": (
                    f"EDGE_SCHEDULER:{reason}:{outcome_class}"
                ),
            }
            packet = json.dumps(
                message, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self._scheduler_feedback_socket.sendto(
                packet, ("127.0.0.1", int(self.feedback_port))
            )
            self.transport_counters.bump("scheduler_terminals_forwarded_to_agent")

        def finish(self) -> bool:
            try:
                return super().finish()
            finally:
                self.live.scheduler_feedback_callback = None
                self._scheduler_feedback_socket.close()

        def diagnostic_summary(self) -> dict[str, Any]:
            document = dict(super().diagnostic_summary())
            with self.live.scheduler_feedback_lock:
                scheduler = [
                    dict(item) for item in self.live.scheduler_feedback_records
                ]
            reasons = Counter(str(item["terminal_reason"]) for item in scheduler)
            classes = Counter(str(item["outcome_class"]) for item in scheduler)
            scheduler_frames = {int(item["frame_id"]) for item in scheduler}
            with self.rows_lock:
                sent_rows = [
                    dict(row)
                    for row in self.rows
                    if str(row.get("prepare_status")) == "SENT"
                ]
            captures = {
                int(row["frame_id"]): float(row["capture_wall_s"])
                for row in sent_rows
                if row.get("capture_wall_s") not in (None, "")
            }
            with self.gt_lock:
                installed = dict(self.installed_at)
            aoi_ms = sorted(
                (float(installed[frame]) - captures[frame]) * 1000.0
                for frame in set(installed) & set(captures)
            )
            status_by_frame: dict[int, set[str]] = {}
            feedback_path = Path(self.attempt_dir) / "map_feedback.csv"
            if feedback_path.is_file():
                with feedback_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        try:
                            frame = int(row["frame_id"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        status_by_frame.setdefault(frame, set()).add(
                            str(row.get("status") or "")
                        )
            true_timeout_frames = {
                frame
                for frame, statuses in status_by_frame.items()
                if "TIMEOUT_NO_ACK" in statuses and frame not in scheduler_frames
                and "ACK_INSTALLED" not in statuses
            }
            map_nack_without_scheduler = {
                frame
                for frame, statuses in status_by_frame.items()
                if frame not in scheduler_frames
                and statuses.intersection({"NACK_REJECTED", "NACK_REASSEMBLY_TIMEOUT"})
            }

            # Time-weighted AoI is defined only from the first installed map.
            # Each newer capture resets the age; an out-of-order installation
            # is recorded but cannot make the map younger.
            useful = 0
            out_of_order = 0
            integral_ms_s = 0.0
            weighted_duration_s = 0.0
            current_capture: float | None = None
            previous_install: float | None = None
            for frame, install_at in sorted(installed.items(), key=lambda item: item[1]):
                capture_at = captures.get(int(frame))
                if capture_at is None:
                    continue
                installed_at = float(install_at)
                if current_capture is not None and previous_install is not None:
                    duration = max(0.0, installed_at - previous_install)
                    start_age_ms = max(0.0, previous_install - current_capture) * 1000.0
                    end_age_ms = max(0.0, installed_at - current_capture) * 1000.0
                    integral_ms_s += 0.5 * (start_age_ms + end_age_ms) * duration
                    weighted_duration_s += duration
                if current_capture is None or capture_at > current_capture:
                    current_capture = capture_at
                    useful += 1
                else:
                    out_of_order += 1
                previous_install = installed_at

            def percentile(values: list[float], fraction: float) -> float | None:
                if not values:
                    return None
                index = min(
                    len(values) - 1,
                    max(0, int(round(fraction * (len(values) - 1)))),
                )
                return float(values[index])

            document["freshness_scheduler"] = {
                "terminal_feedback_records": len(scheduler),
                "terminal_reason_counts": dict(sorted(reasons.items())),
                "outcome_class_counts": dict(sorted(classes.items())),
                "intentional_freshness_drop_frames": sum(
                    str(item["outcome_class"]) == "INTENTIONAL_FRESHNESS_DROP"
                    for item in scheduler
                ),
                "expired_work_frames": sum(
                    str(item["outcome_class"]) == "EXPIRED_WORK"
                    for item in scheduler
                ),
                "feature_bytes_charged": sum(
                    int((item.get("agent_credit") or {}).get("charge_feature_bytes", 0))
                    for item in scheduler
                ),
                "wasted_feature_bytes": sum(
                    int((item.get("agent_credit") or {}).get("wasted_feature_bytes", 0))
                    for item in scheduler
                ),
                "compute_ns_charged": sum(
                    int((item.get("agent_credit") or {}).get("charge_compute_ns", 0))
                    for item in scheduler
                ),
            }
            document["map_utility"] = {
                "sent_frames": len(sent_rows),
                "ack_installed_frames": len(installed),
                "useful_newer_map_installations": useful,
                "out_of_order_installations": out_of_order,
                "explicit_scheduler_non_install_terminals": len(scheduler_frames),
                "true_timeout_without_scheduler_or_install": len(true_timeout_frames),
                "map_nack_without_scheduler": len(map_nack_without_scheduler),
                "map_feedback_status_counts": dict(
                    sorted(
                        Counter(
                            status
                            for statuses in status_by_frame.values()
                            for status in statuses
                        ).items()
                    )
                ),
                "installed_within_100ms": sum(value <= 100.0 for value in aoi_ms),
                "installed_within_500ms": sum(value <= 500.0 for value in aoi_ms),
                "install_aoi_ms_median": percentile(aoi_ms, 0.5),
                "install_aoi_ms_p95": percentile(aoi_ms, 0.95),
                "time_weighted_map_aoi_ms_after_first_install": (
                    None
                    if weighted_duration_s <= 0.0
                    else integral_ms_s / weighted_duration_s
                ),
                "time_weighted_duration_s": weighted_duration_s,
            }
            # Retain the authoritative per-frame install boundary for the
            # bounded payoff experiment. This is compact timing/accounting
            # evidence, not a prediction, payload, image, mask or map record.
            document["map_install_records"] = [
                {
                    "frame_id": int(frame),
                    "capture_wall_s": float(captures[frame]),
                    "install_timestamp_s": float(installed[frame]),
                    "install_aoi_ms": (
                        float(installed[frame]) - float(captures[frame])
                    )
                    * 1000.0,
                }
                for frame in sorted(set(installed) & set(captures))
            ]
            return document

    return SchedulerAwareCollector


def install_live_wrappers(
    adapter: Any,
    *,
    transmitted_budget: int,
    safety_timeout_s: float,
    artifacts_dir: Any = None,
) -> dict[str, Any]:
    """Install timing wrappers, then add scheduler-aware feedback handling."""

    original = diagnostic.install_live_wrappers(
        adapter,
        transmitted_budget=transmitted_budget,
        safety_timeout_s=safety_timeout_s,
        artifacts_dir=artifacts_dir,
    )
    adapter.LivePilotCellRuntime = build_scheduler_runtime_class(
        adapter.LivePilotCellRuntime
    )
    adapter.PassiveSplitCollector = build_scheduler_collector_class(
        adapter.PassiveSplitCollector
    )
    return original
