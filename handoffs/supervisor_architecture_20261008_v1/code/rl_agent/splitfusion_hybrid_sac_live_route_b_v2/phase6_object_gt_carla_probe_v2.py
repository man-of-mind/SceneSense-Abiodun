#!/usr/bin/env python3
"""Addendum 9, Phase C: CARLA-only object-GT timing probe (about 30 Route-B frames).

The launcher starts only a CARLA server (the runner's lifecycle helper, same
flags and port) on a verified-idle host. It runs this module's child and then
stops CARLA. It starts no OAI, edge container, map service, Docker, CUDA or
Phase-6 runtime.

The child drives the real Route-B collector chain,
``build_run4_collector_class(QualityPassiveSplitCollector)`` over the pinned
``PassiveSplitCollector``: the real sensors, radar window, rasterization,
scene freezing, the pinned evaluation worker, the pinned ``_ground_truth``,
the repaired object-row hook, the reward gate and LOW skip, and the
real object/semantic GT writers. Only the UE runtime is replaced, by
:class:`ProbeRuntimeV2`. That stub alternates reward-requested and hold
frames, opens the decision (the ``reward_planned_hook``), and then sleeps the
v8-measured action-open -> last-datagram time (26.9 ms) in place of
front/codec/send. It never runs a model and never touches the radio.

Timing is recorded on the optimized path only. Shadow comparisons (the pinned
140-m builder, run on each ticket's exact frozen inputs and a copy of the
tracker state taken just before the optimized build) run after the route
ends, so they cannot contend with the measured path.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXECUTE_TOKEN = "GO_RUN4_PHASE6_OBJECT_GT_CARLA_PROBE_V9"
SCHEMA = "scenesense.run4_live_v2.phase6_object_gt_carla_probe.v1"
FRAMES = 30
V8_ACTION_OPEN_TO_LAST_SEND_MS = 26.923322        # frame 1255 decision stages
V8_EDGE_READY_AFTER_ACTION_OPEN_MS = 123.5        # policy frame map-installed
V8_GT_DETECT_TO_EMIT_MS = 5.747944                # evaluator detect -> emit (edge)
V8_OBJECTS_WRITE_TO_DETECT_MS = 2.443659          # objects write end -> detected
V8_EMIT_TO_UE_RECEIPT_MS = 2.1                    # capture-anchored reconciliation
TARGET_GT_READY_MS = 150.0
DEADLINE_MS = 170.0


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return {"n": len(vals), "p50": statistics.median(vals), "p95": p95, "max": vals[-1],
            "min": vals[0], "method": "nearest-rank on sorted samples"}


# ---------------------------------------------------------------------------
# Host idleness (Phase C is inadmissible under concurrent load)
# ---------------------------------------------------------------------------


def host_idle_report(config: Optional[Mapping[str, Any]] = None,
                     max_load1: float = 1.5) -> dict[str, Any]:
    """Load, CARLA processes, running containers and the supervisor cold check."""
    load1, load5, load15 = (float(v) for v in Path("/proc/loadavg").read_text().split()[:3])
    carla = subprocess.run(["pgrep", "-f", "CarlaUnreal"], capture_output=True, text=True)
    docker = subprocess.run(["sudo", "-n", "docker", "ps", "--format", "{{.Names}}"],
                            capture_output=True, text=True)
    busy = subprocess.run(["ps", "-eo", "pid,pcpu,etime,args", "--sort=-pcpu"],
                          capture_output=True, text=True).stdout.splitlines()[1:8]
    report = {"load1": load1, "load5": load5, "load15": load15,
              "carla_pids": carla.stdout.split(),
              "docker_ps_returncode": docker.returncode,
              "docker_containers": docker.stdout.split(), "top_cpu": busy,
              "max_load1": max_load1, "supervisor_cold": None}
    if config is not None:
        from rl_agent import ue_288_campaign_supervisor as supervisor

        try:
            supervisor._require_phase15_application_cold(config)
            report["supervisor_cold"] = True
        except Exception as exc:  # noqa: BLE001
            report["supervisor_cold"] = f"{type(exc).__name__}: {exc}"[:300]
    report["idle"] = (load1 < max_load1 and not report["carla_pids"]
                      and docker.returncode == 0 and not report["docker_containers"]
                      and report["supervisor_cold"] in (None, True))
    return report


# ---------------------------------------------------------------------------
# Stub UE runtime (no model, no radio)
# ---------------------------------------------------------------------------


class _NoCounters:
    def snapshot(self) -> dict[str, Any]:
        return {}


class ProbeRuntimeV2:
    """Exactly the ``self.live`` surface the collector chain touches."""

    def __init__(self, *, campaign: Mapping[str, Any], cell: Mapping[str, Any],
                 attempt_dir: Path, evidence_out: Path, sleep_ms: float) -> None:
        from . import phase6_object_gt_v2 as OG

        self.campaign, self.cell = campaign, cell
        self.attempt_dir = Path(attempt_dir)
        self.evidence_out = Path(evidence_out)
        self.sleep_s = float(sleep_ms) / 1000.0
        self._run4_identity: dict[int, dict[str, Any]] = {}
        self._gt_identity: dict[int, dict[str, Any]] = {}
        self.sensor_stages: "collections.OrderedDict[float, dict]" = collections.OrderedDict()
        self.scene_hooks = None
        self.reward_planned_hook = None
        self.cycle_boundary_reached = False
        self.infrastructure_fault = None
        self.counters = _NoCounters()
        self.gt_log = OG.GtTicketLogV3()
        self.last_datagram = OG.LastDatagramMarksV2()
        self.records: dict[int, dict[str, Any]] = {}
        self.sent = 0
        self._lock = threading.Lock()
        self.shadow_inputs: list[dict[str, Any]] = []

    def register_sensor_ready(self, frame_id: int, *, observed_wall_ns: int) -> None:
        return None

    def identity_for_frame(self, frame_id: int) -> dict[str, Any]:
        return dict(self._gt_identity[int(frame_id)])

    def take_metric(self, frame_id: int) -> None:
        return None

    def submit(self, *, frame_bgr, radar_tensor, frame_id, capture_timestamp_ns, ego_pose,
               stream_id, carla_timestamp, capture_id, action_id=None,
               on_commit=None) -> dict[str, Any]:
        from . import run4_map_protocol_v2 as MP
        from . import ue_telemetry_provider_v2 as T

        del frame_bgr, radar_tensor, ego_pose, capture_id, action_id
        self.sensor_stages.pop(float(carla_timestamp), None)
        window_meta, rgb_raw = self.scene_hooks(int(frame_id), float(carla_timestamp))
        del window_meta
        with self._lock:
            seq = self.sent
        reward = seq % 2 == 0                     # reward, hold, reward, hold, ...
        open_raw, open_wall = T.raw_now_ns(), time.time_ns()
        if self.reward_planned_hook is not None:
            self.reward_planned_hook(int(frame_id), float(carla_timestamp), reward)
        hook_raw = T.raw_now_ns()
        time.sleep(self.sleep_s)                  # stands in for front/codec/send only
        self._run4_identity[int(frame_id)] = {
            "reward_requested": reward, "frame_id": int(frame_id), "tensor_seq": seq,
            "decision_seq": seq // 2 if reward else None,
            "ticket_seq": seq // 2 if reward else None, "session_uuid": "probe"}
        self._gt_identity[int(frame_id)] = MP.gt_identity(
            run_id=str(self.campaign["campaign_id"]), cell_id=str(self.cell["cell_id"]),
            stream_id=str(stream_id), frame_id=int(frame_id), anchor_action_id=None,
            anchor_profile_id=None, capture_timestamp_ns=int(capture_timestamp_ns))
        if on_commit is not None:
            on_commit()
        self.last_datagram.mark(int(frame_id), T.raw_now_ns())   # emulated last datagram
        done_raw, done_wall = T.raw_now_ns(), time.time_ns()
        with self._lock:
            self.records[int(frame_id)] = {
                "frame_id": int(frame_id), "reward_requested": reward, "tensor_seq": seq,
                "capture_wall_ns": int(capture_timestamp_ns), "rgb_receipt_raw_ns": rgb_raw,
                "action_open_raw_ns": open_raw, "action_open_wall_ns": open_wall,
                "hook_return_raw_ns": hook_raw, "submit_return_raw_ns": done_raw,
                "submit_return_wall_ns": done_wall}
            self.sent += 1
        return {"sent": True, "front_ms": (done_raw - open_raw) / 1e6, "payload_bytes": 0,
                "payload_bytes_uncompressed": "", "payload_chunks": 0}

    def close(self) -> dict[str, Any]:
        from . import phase6_object_gt_v2 as OG

        gate = getattr(self, "gt_gate", None)
        evidence = {"records": [self.records[k] for k in sorted(self.records)],
                    "gt_objects": self.gt_log.snapshot(),
                    "gt_reward_gate": None if gate is None else {
                        "events": list(gate.events), "pending_at_close": gate.pending()},
                    "gt_last_datagram": self.last_datagram.snapshot(),
                    "gt_queue": None if getattr(self, "gt_queue", None) is None else {
                        "counters": dict(self.gt_queue.counters),
                        "unfinished_tasks": self.gt_queue.unfinished_tasks},
                    "raw_clock": "CLOCK_MONOTONIC_RAW", "low_skipped_status":
                        OG.LOW_SKIPPED_STATUS}
        self.evidence_out.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_out.write_text(json.dumps(evidence, sort_keys=True, default=str),
                                     encoding="utf-8")
        return {"probe_runtime": "closed"}


def build_probe_collector_class(base: type, runtime_holder: dict) -> type:
    """Keep each exact-scene ticket's frozen inputs for the post-run shadow."""

    class ProbeCollector(base):  # type: ignore[misc, valid-type]
        def _run4_object_rows(self, real_build, **kwargs):
            frozen = kwargs.get("world") is not self.world
            before = None
            if frozen:
                tracker = kwargs["stationary_tracker"]
                before = (dict(tracker._ages), dict(tracker._last_time))
            try:
                rows = super()._run4_object_rows(real_build, **kwargs)
            except Exception as exc:
                if frozen:
                    runtime_holder["runtime"].shadow_inputs.append(
                        {"kwargs": kwargs, "tracker_before": before, "rows": None,
                         "error": type(exc).__name__})
                raise
            if frozen:
                tracker = kwargs["stationary_tracker"]
                runtime_holder["runtime"].shadow_inputs.append(
                    {"kwargs": kwargs, "tracker_before": before, "rows": rows,
                     "tracker_after": (dict(tracker._ages), dict(tracker._last_time))})
            return rows

    return ProbeCollector


# ---------------------------------------------------------------------------
# Post-run shadow parity and timing summary
# ---------------------------------------------------------------------------


def exact(value: Any) -> str:
    """Bit-exact canonical serialization (float.hex for every float)."""
    def walk(v):
        if isinstance(v, float):
            return v.hex()
        if isinstance(v, dict):
            return {str(k): walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        return v
    return json.dumps(walk(value), sort_keys=True, default=repr, separators=(",", ":"))


def shadow_parity(parked: Any, shadow_inputs: Sequence[Mapping[str, Any]], *,
                  max_gt_distance_m: float, min_gt_area_px: float,
                  valid_localization_objects: Any) -> dict[str, Any]:
    """Pinned 140-m builder on the identical frozen inputs vs the optimized rows."""
    per_ticket, failures = [], []
    for item in shadow_inputs:
        kwargs = dict(item["kwargs"])
        frame = int(kwargs["sample_base"]["frame_id"])
        if item.get("rows") is None:
            per_ticket.append({"frame_id": frame, "skipped": item.get("error")})
            continue
        tracker = parked.ActorStationaryTracker(0.35, 5.0)
        tracker._ages, tracker._last_time = (dict(item["tracker_before"][0]),
                                             dict(item["tracker_before"][1]))
        kwargs["stationary_tracker"] = tracker
        started = time.perf_counter_ns()
        reference = parked.build_object_rows(**kwargs)
        pinned_ms = (time.perf_counter_ns() - started) / 1e6
        width, height = int(kwargs["width"]), int(kwargs["height"])

        def eligible(rows):
            return valid_localization_objects(rows, image_width=width, image_height=height,
                                              min_area_px=min_area, max_distance_m=limit)
        limit, min_area = float(max_gt_distance_m), float(min_gt_area_px)
        ref_targets, new_targets = eligible(reference), eligible(item["rows"])
        checks = {
            "rows_identical_within_eligibility": exact(
                [r for r in reference if r["gt_distance_m"] <= limit])
                == exact(item["rows"]),
            "eligible_targets_identical": exact(ref_targets)
                == exact(new_targets),
            "class_world_xy_identical": [(o["class_name"], o["world_x"], o["world_y"])
                                         for o in ref_targets]
                == [(o["class_name"], o["world_x"], o["world_y"]) for o in new_targets],
            "tracker_state_identical": exact(
                {"a": dict(sorted(tracker._ages.items())),
                 "l": dict(sorted(tracker._last_time.items()))})
                == exact({"a": dict(sorted(item["tracker_after"][0].items())),
                                  "l": dict(sorted(item["tracker_after"][1].items()))}),
        }
        row = {"frame_id": frame, "pinned_140m_builder_ms_uncontended": pinned_ms,
               "pinned_rows": len(reference), "optimized_rows": len(item["rows"]),
               "eligible_targets": len(ref_targets), **checks}
        per_ticket.append(row)
        if not all(checks.values()):
            failures.append(frame)
    return {"tickets": per_ticket, "failures": failures,
            "compared": sum(1 for t in per_ticket if "skipped" not in t),
            "passed": not failures and any("skipped" not in t for t in per_ticket)}


def timing_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    records = {int(r["frame_id"]): r for r in evidence["records"]}
    tickets = {int(t["frame_id"]): t for t in evidence["gt_objects"]["tickets"]}
    gate_events = evidence.get("gt_reward_gate", {}).get("events", []) or []
    opens = {int(e["frame_id"]): e["raw_ns"] for e in gate_events if e["event"] == "open"}
    closes = {int(e["frame_id"]): e["raw_ns"] for e in gate_events if e["event"] == "close"}
    high, low = [], []
    for frame, rec in sorted(records.items()):
        t = tickets.get(frame, {})
        prof = t.get("object_builder") or {}
        row = {"frame_id": frame, "reward_requested": rec["reward_requested"],
               "queue_class": t.get("queue_class"), "queue_wait_ms": t.get("queue_wait_ms"),
               "low_skipped": t.get("low_skipped"), "outcome": prof.get("outcome"),
               "builder_wall_ms": prof.get("wall_ms"),
               "builder_thread_cpu_ms": prof.get("thread_cpu_ms"),
               "voluntary_ctx_switches": prof.get("voluntary_ctx_switches"),
               "involuntary_ctx_switches": prof.get("involuntary_ctx_switches"),
               "stage_ms": prof.get("stage_ms"), "actor_counts": prof.get("actor_counts")}
        ao_wall = rec["action_open_wall_ns"]
        for key, name in (("enqueue_wall_ns", "enqueue"), ("worker_start_wall_ns", "dequeue"),
                          ("object_rows_start_wall_ns", "gt_call_start"),
                          ("objects_write_start_wall_ns", "write_start"),
                          ("objects_write_end_wall_ns", "objects_ready")):
            if t.get(key) is not None:
                row[f"{name}_after_action_open_ms"] = (int(t[key]) - ao_wall) / 1e6
        if prof.get("start_raw_ns") is not None:
            row["builder_start_after_action_open_ms"] = (
                prof["start_raw_ns"] - rec["action_open_raw_ns"]) / 1e6
            row["builder_end_after_action_open_ms"] = (
                prof["end_raw_ns"] - rec["action_open_raw_ns"]) / 1e6
        if t.get("objects_write_start_wall_ns") and t.get("objects_write_end_wall_ns"):
            row["write_ms"] = (t["objects_write_end_wall_ns"]
                               - t["objects_write_start_wall_ns"]) / 1e6
        if prof.get("end_wall_ns") and t.get("objects_write_start_wall_ns"):
            row["eligibility_filter_ms"] = (t["objects_write_start_wall_ns"]
                                            - prof["end_wall_ns"]) / 1e6
        if rec["reward_requested"]:
            ready = row.get("objects_ready_after_action_open_ms")
            if ready is not None:
                gt_detect = ready + V8_OBJECTS_WRITE_TO_DETECT_MS
                row["predicted_feedback_after_action_open_ms"] = (
                    max(V8_EDGE_READY_AFTER_ACTION_OPEN_MS, gt_detect)
                    + V8_GT_DETECT_TO_EMIT_MS + V8_EMIT_TO_UE_RECEIPT_MS)
            # any LOW builder running inside this reward ticket's gate window
            o, c = opens.get(frame), closes.get(frame)
            intrusions = []
            for other, ot in tickets.items():
                op = ot.get("object_builder") or {}
                if ot.get("queue_class") != "LOW" or op.get("start_raw_ns") is None or o is None:
                    continue
                end = c if c is not None else op["end_raw_ns"]
                overlap = min(op["end_raw_ns"], end) - max(op["start_raw_ns"], o)
                if overlap > 0:
                    intrusions.append({"frame_id": other, "overlap_ms": overlap / 1e6,
                                       "outcome": op.get("outcome")})
            row["low_inside_reward_window"] = intrusions
            high.append(row)
        else:
            low.append(row)
    ready = [r["objects_ready_after_action_open_ms"] for r in high
             if r.get("objects_ready_after_action_open_ms") is not None]
    predicted = [r["predicted_feedback_after_action_open_ms"] for r in high
                 if r.get("predicted_feedback_after_action_open_ms") is not None]
    builder = [r["builder_wall_ms"] for r in high if r.get("builder_wall_ms") is not None]
    residual = [i["overlap_ms"] for r in high for i in r["low_inside_reward_window"]]
    summary = {
        "high": high, "low": low,
        "high_builder_wall_ms": _stats(builder),
        "high_builder_thread_cpu_ms": _stats([r["builder_thread_cpu_ms"] for r in high
                                              if r.get("builder_thread_cpu_ms") is not None]),
        "high_queue_wait_ms": _stats([r["queue_wait_ms"] for r in high
                                      if r.get("queue_wait_ms") is not None]),
        "high_objects_ready_after_action_open_ms": _stats(ready),
        "high_predicted_feedback_after_action_open_ms": _stats(predicted),
        "low_builder_wall_ms": _stats([r["builder_wall_ms"] for r in low
                                       if r.get("builder_wall_ms") is not None]),
        "low_skipped": sum(1 for r in low if r.get("low_skipped")),
        "low_preempted": sum(1 for r in low if r.get("outcome") == "LOW_PREEMPTED"),
        "low_residual_inside_reward_window_ms": _stats(residual),
        "prediction_model": {
            "formula": ("max(v8 edge-ready 123.5, objects_ready + v8 write->detect 2.44)"
                        " + v8 detect->emit 5.75 + v8 emit->UE receipt 2.1"),
            "v8_action_open_to_last_send_ms_emulated_by_sleep": V8_ACTION_OPEN_TO_LAST_SEND_MS},
    }
    all_high_ready = len(ready) == len(high) and len(high) > 0
    summary["gates"] = {
        "every_high_ticket_produced_objects": all_high_ready,
        "objects_ready_max_le_150ms": bool(ready) and max(ready) <= TARGET_GT_READY_MS,
        "predicted_feedback_max_lt_170ms": bool(predicted) and max(predicted) < DEADLINE_MS,
        "predicted_feedback_margin_ms": (DEADLINE_MS - max(predicted)) if predicted else None,
        "no_low_delaying_high": all(
            (r.get("queue_wait_ms") or 0.0) < 5.0 for r in high)
            and all(i.get("outcome") == "LOW_PREEMPTED" for r in high
                    for i in r["low_inside_reward_window"])
            and (max(residual) if residual else 0.0) < 5.0,
    }
    summary["gates"]["passed"] = all(v for k, v in summary["gates"].items()
                                     if k != "predicted_feedback_margin_ms")
    return summary


# ---------------------------------------------------------------------------
# Child (CARLA client only)
# ---------------------------------------------------------------------------


def child(args: argparse.Namespace) -> int:  # pragma: no cover - live CARLA only
    from rl_agent import ue_288_campaign_supervisor as supervisor
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_cell_child as LC
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP

    from . import phase6_gt_handoff_v2 as GH
    from . import phase6_live_runner_v2 as RUN
    from . import phase6_ue_runtime_v2 as U

    out = Path(args.output_dir).resolve(strict=True)
    config, cells, _report = LP.offline_preflight(
        RUN.DEFAULT_CONFIG, action_ids=(RUN.FALLBACK_ACTION,), profile_ids=(RUN.PROFILE,),
        transmitted_budget=int(args.frames), safety_timeout_s=120.0, output_root=None)
    registered = cells[0]
    campaign = LP._probe_campaign(config, run_id=out.name)
    campaign["campaign_id"] = f"splitfusion_run4_phase6_object_gt_probe_v9/{out.name}"
    cell = supervisor.cell_to_dict(supervisor.Cell(
        cell_id=f"objgt_probe_{out.name}", action_index=registered.action_index,
        action_id=registered.action_id, profile_id=registered.profile_id,
        model_family=registered.model_family,
        network_profile_id=registered.network_profile_id,
        trace_id=registered.trace_id, seed=registered.seed))
    attempt = out / "attempt"
    artifacts = attempt / "phase6_artifacts"
    artifacts.mkdir(parents=True)
    evidence_dir = Path(tempfile.mkdtemp(prefix="objgt_probe_edge_evidence_"))
    campaign["_target_start_file"] = str(out / "target_start")
    holder: dict[str, Any] = {}

    def runtime_factory(*, campaign: Mapping[str, Any], cell: Mapping[str, Any],
                        attempt_dir: Path, map_host: str, map_port: int,
                        evidence_dir: Path) -> ProbeRuntimeV2:
        del map_host, map_port, evidence_dir
        runtime = ProbeRuntimeV2(campaign=campaign, cell=cell, attempt_dir=attempt_dir,
                                 evidence_out=out / "probe_evidence.json",
                                 sleep_ms=V8_ACTION_OPEN_TO_LAST_SEND_MS)
        holder["runtime"] = runtime
        holder["recorder"].ticket_log = runtime.gt_log       # object-write timing
        return runtime

    pinned.LivePilotCellRuntime = runtime_factory
    pinned.SceneSnapshotSource = Q.QualitySceneSnapshotSource
    pinned.PassiveSplitCollector = LC.build_bounded_collector_class(
        build_probe_collector_class(U.build_run4_collector_class(
            Q.QualityPassiveSplitCollector), holder),
        transmitted_budget=int(args.frames), safety_timeout_s=120.0,
        artifacts_dir=artifacts)
    recorder = GH.GtWriteRecorderV2(out / "gt_writes.jsonl")   # ticket_log bound below
    holder["recorder"] = recorder
    Q.write_object_ground_truth, Q.write_semantic_ground_truth = recorder.wrap(
        Q.write_object_ground_truth, Q.write_semantic_ground_truth)
    row = pinned.action_row(campaign, int(cell["action_id"]))
    result: dict[str, Any] = {"schema": SCHEMA + ".child", "error": ""}
    try:
        ok, detail, collector = pinned.run_route_b(
            campaign=campaign, cell=cell, row=row,
            binding={"dispatcher": "run4_phase6_object_gt_probe"}, attempt_dir=attempt,
            carla_host="127.0.0.1", carla_port=int(args.carla_port),
            map_api_port=int(args.map_api_port), feedback_port=int(args.feedback_port),
            edge_evidence_dir=evidence_dir, maximum_loop_sim_s=120.0)
        result.update(route_accepted=bool(ok), route_detail=detail,
                      sent=None if collector is None else int(collector.sent),
                      stop_reason=None if collector is None else collector.probe_stop_reason,
                      collector_failures=None if collector is None
                      else list(collector.failures)[:8])
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    runtime = holder.get("runtime")
    if runtime is not None and (out / "probe_evidence.json").is_file():
        from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import (
            valid_localization_objects,
        )
        import carla_collect_parked_ego_fusion_training_data as parked

        evidence = json.loads((out / "probe_evidence.json").read_text(encoding="utf-8"))
        contract = campaign["measurement_contract"]
        result["timing"] = timing_summary(evidence)
        result["shadow_parity"] = shadow_parity(
            parked, runtime.shadow_inputs, max_gt_distance_m=float(contract["max_gt_distance_m"]),
            min_gt_area_px=float(contract["min_gt_area_px"]),
            valid_localization_objects=valid_localization_objects)
    (out / "child_result.json").write_text(json.dumps(result, sort_keys=True, default=str,
                                                      indent=1), encoding="utf-8")
    return 0 if not result["error"] else 2


# ---------------------------------------------------------------------------
# Launcher (CARLA server only)
# ---------------------------------------------------------------------------


def launch(args: argparse.Namespace) -> int:  # pragma: no cover - live CARLA only
    from rl_agent import ue_288_campaign_supervisor as supervisor
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP

    from . import phase6_live_runner_v2 as RUN

    require(args.execute == EXECUTE_TOKEN, f"requires --execute {EXECUTE_TOKEN}")
    # The CARLA server inherits this environment (lifecycle.child_env); it must
    # see the GPU exactly as under the live runner. The probe child alone is
    # denied CUDA below.
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "launch with CUDA_VISIBLE_DEVICES=0 (CARLA server GPU, as the live runner)")
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config, _cells, _ = LP.offline_preflight(
        RUN.DEFAULT_CONFIG, action_ids=(RUN.FALLBACK_ACTION,), profile_ids=(RUN.PROFILE,),
        transmitted_budget=int(args.frames), safety_timeout_s=120.0, output_root=None)
    idle = host_idle_report(config)
    (out / "host_idle_before.json").write_text(json.dumps(idle, indent=1), encoding="utf-8")
    require(idle["idle"], f"host is not idle: {idle}")
    lifecycle = supervisor.import_lifecycle_helper(config)
    service = Path(tempfile.mkdtemp(prefix="objgt_probe_carla_"))
    report: dict[str, Any] = {"schema": SCHEMA, "status": "FAILED", "host_idle_before": idle}
    server = pgid = None
    try:
        server, pgid = lifecycle.start_carla(int(args.carla_port), service / "carla_server.log")
        require(lifecycle.wait_for_rpc(int(args.carla_port), 180.0) is not None,
                "CARLA not ready")
        argv = [sys.executable, "-m", __spec__.name if __spec__ else
                "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_object_gt_carla_probe_v2",
                "--child", "--output-dir", str(out), "--frames", str(int(args.frames)),
                "--carla-port", str(int(args.carla_port)),
                "--map-api-port", str(int(args.map_api_port)),
                "--feedback-port", str(int(args.feedback_port))]
        with (out / "child.log").open("xb") as stream:
            rc = subprocess.run(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT,
                                env={**lifecycle.child_env(), "CUDA_VISIBLE_DEVICES": ""},
                                timeout=float(args.child_timeout_s)).returncode
        report["child_returncode"] = rc
        child_result = json.loads((out / "child_result.json").read_text(encoding="utf-8"))
        report["child_error"] = child_result.get("error")
        report["status"] = "COLLECTED" if rc == 0 else "FAILED"
    except BaseException as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None and pgid is not None:
            try:
                report["carla_stopped"] = lifecycle.stop_carla(server, pgid,
                                                               int(args.carla_port))
            except BaseException as exc:  # noqa: BLE001
                report["carla_stop_error"] = f"{type(exc).__name__}: {exc}"
        log = service / "carla_server.log"
        if log.is_file():
            (out / "carla_server.log").write_bytes(log.read_bytes())
        report["host_idle_after"] = host_idle_report(config)
        (out / "PROBE_REPORT.json").write_text(json.dumps(report, indent=1, default=str),
                                               encoding="utf-8")
    return 0 if report["status"] == "COLLECTED" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--execute", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--map-api-port", type=int, default=39420)
    parser.add_argument("--feedback-port", type=int, default=39421)
    parser.add_argument("--child-timeout-s", type=float, default=600.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return child(args) if args.child else launch(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
