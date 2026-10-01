#!/usr/bin/env python3
"""Offline validation and coverage of one profiled attempt (read-only).

Reads the profiler evidence (UE, edge, map, sampler) and the unchanged Phase-6
evidence of one cell and reports, without changing anything:

* **validity**: header/footer per file; unique sequence numbers; per-thread
  timestamp monotonicity; no negative interval; every span inside its parent,
  with the registered parent stage; no crossed sibling intervals; derived
  spans inside their parent; every identity field equal to the transmitted
  identity of its frame;
* **coverage**: per registered stage, frames covered against the registered
  denominator, with missing counts and examples;
* **sampler**: interval P50/P95/max, dropped samples, per-sample cost, CPU;
* **overhead**: profiler hot-path time per call and in total, writer CPU,
  sampler CPU, dropped records, internal errors, file sizes;
* **clock domains**: per-process (realtime - raw) offsets from clock probes and
  their cross-process spread;
* **semantics**: (a) every live actor decision replayed offline through the
  configured frozen actor gives the same mode, q_e4, state and head digests;
  (b) every reward resolution obeys the configured inclusive deadline and
  reward formula exactly; (c) every feedback-receipt mark equals the
  authoritative receipt stamp.

No result here is a timing claim; stage durations are descriptive only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import recorder_v1 as RV
from . import stages_v1 as SV

SCHEMA = "scenesense.single_gpu_profiler.analysis.v1"
IDENTITY_JOIN_KEYS = ("tensor_seq", "session_uuid", "decision_seq", "ticket_seq",
                      "reward_requested", "mode_id", "q_e4", "frame_kind")
CROSS_PROCESS_CLOCK_TOLERANCE_NS = 2_000_000


def nearest_rank(values: Sequence[float], p: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, -(-int(p * len(ordered)) // 100)) - 1]


def dist(values: Iterable[Optional[float]]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {"n": len(present), "p50": nearest_rank(present, 50),
            "p95": nearest_rank(present, 95), "p99": nearest_rank(present, 99),
            "max": max(present) if present else None,
            "mean": statistics.fmean(present) if present else None}


def load(path: Path) -> list[dict]:
    return list(RV.iter_records(path)) if Path(path).is_file() else []


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


def validate_stream(records: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, Any]:
    """Structural checks on one recorder file (pure)."""
    problems: "collections.Counter[str]" = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    def bad(kind: str, detail: Any) -> None:
        problems[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(detail)

    kinds = collections.Counter(r.get("r") for r in records)
    if kinds.get("header") != 1:
        bad("header_count", kinds.get("header"))
    if kinds.get("footer") != 1:
        bad("footer_count", kinds.get("footer"))
    seqs = [r["seq"] for r in records if r.get("r") in ("span", "mark", "note") and "seq" in r]
    if len(seqs) != len(set(seqs)):
        bad("duplicate_seq", len(seqs) - len(set(seqs)))
    spans = {r["seq"]: r for r in records if r.get("r") == "span"}
    for seq, span in spans.items():
        if span["end_raw_ns"] < span["start_raw_ns"]:
            bad("negative_interval", seq)
        stage = span.get("stage")
        registered = SV.STAGES.get(stage)
        if registered is None:
            bad("unregistered_stage", stage)
            continue
        parent_seq = span.get("parent")
        parent = spans.get(parent_seq) if parent_seq is not None else None
        if parent_seq is not None and parent is None:
            bad("missing_parent", seq)
            continue
        parent_stage = None if parent is None else parent.get("stage")
        allowed = registered["parents"]
        if allowed is None:
            if parent_stage is not None:
                bad("nesting_registry", f"{stage} in {parent_stage}")
        elif parent_stage not in allowed:
            bad("nesting_registry", f"{stage} in {parent_stage}")
        if parent is not None:
            if parent.get("tid") != span.get("tid"):
                bad("parent_other_thread", seq)
            if not (parent["start_raw_ns"] <= span["start_raw_ns"]
                    and span["end_raw_ns"] <= parent["end_raw_ns"]):
                bad("outside_parent", f"{stage}#{seq}")
    # crossed siblings and per-thread monotonicity of non-derived span starts
    siblings: dict[tuple, list] = collections.defaultdict(list)
    for span in spans.values():
        siblings[(span.get("tid"), span.get("parent"))].append(span)
    for group in siblings.values():
        group.sort(key=lambda s: (s["start_raw_ns"], s["seq"]))
        for previous, current in zip(group, group[1:]):
            if current["start_raw_ns"] < previous["end_raw_ns"]:
                bad("crossed_siblings", f"{previous['stage']}#{previous['seq']}"
                                        f"x{current['stage']}#{current['seq']}")
    by_tid: dict[Any, list] = collections.defaultdict(list)
    for span in spans.values():
        if not span.get("derived"):
            by_tid[span.get("tid")].append(span)
    for group in by_tid.values():
        group.sort(key=lambda s: s["seq"])
        for previous, current in zip(group, group[1:]):
            if current["start_raw_ns"] < previous["start_raw_ns"]:
                bad("non_monotonic_span_start", current["seq"])
    marks: dict[Any, list] = collections.defaultdict(list)
    for record in records:
        if record.get("r") == "mark":
            marks[record.get("tid")].append(record)
    for group in marks.values():
        group.sort(key=lambda m: m["seq"])
        for previous, current in zip(group, group[1:]):
            if current["t_raw_ns"] < previous["t_raw_ns"]:
                bad("non_monotonic_mark", current["seq"])
    clocks = [r["raw_ns"] for r in records if r.get("r") == "clock"]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        bad("non_monotonic_clock_probe", source)
    cuda_spans = {s["seq"] for s in spans.values() if s.get("cuda") == "PENDING"}
    cuda_records = collections.Counter(r["span_seq"] for r in records if r.get("r") == "cuda")
    missing_cuda = cuda_spans - set(cuda_records)
    if missing_cuda:
        bad("cuda_without_resolution", len(missing_cuda))
    if any(n > 1 for n in cuda_records.values()):
        bad("cuda_duplicate_resolution", sum(1 for n in cuda_records.values() if n > 1))
    return {"source": source, "records": dict(kinds), "problems": dict(problems),
            "examples": dict(examples), "passed": not problems}


def identity_conflicts(records: Iterable[Mapping[str, Any]],
                       transmitted: Mapping[int, Mapping[str, Any]], *,
                       pre_send_stages: frozenset) -> dict[str, Any]:
    """Every live record naming a frame must agree with that frame's identity."""
    conflicts, foreign, checked = [], [], 0
    for record in records:
        if record.get("r") not in ("span", "mark") or record.get("ph") not in ("live", None):
            continue
        frame = record.get("frame_id")
        if frame is None:
            continue
        identity = transmitted.get(int(frame))
        if identity is None:
            if record.get("stage") not in pre_send_stages:
                foreign.append((record.get("stage"), int(frame)))
            continue
        checked += 1
        for key in IDENTITY_JOIN_KEYS:
            if key in record and key in identity and record[key] != identity[key]:
                conflicts.append({"stage": record.get("stage"), "frame_id": int(frame),
                                  "key": key, "record": record[key],
                                  "transmitted": identity[key]})
    return {"checked": checked, "conflicts": len(conflicts), "conflict_examples": conflicts[:10],
            "foreign_frames": len(foreign), "foreign_examples": foreign[:10]}


PRE_SEND_STAGES = frozenset({
    "ue.worker_frame", "ue.sensor_wait", "ue.radar_window", "ue.radar_rasterize",
    "ue.rgb_convert", "ue.scene_snapshot", "ue.submit", "ue.si_p40", "ue.plan",
    "agent.telemetry_snapshot", "agent.state_build", "agent.actor", "ue.input_7ch",
    "ue.sfd4_pack", "ue.gt_build"} | {f"ue.codec.{s}" for s in SV.UE_CODEC})


# ---------------------------------------------------------------------------
# coverage and durations
# ---------------------------------------------------------------------------


def coverage(records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
             denominators: Mapping[str, set]) -> dict[str, Any]:
    seen: dict[str, set] = collections.defaultdict(set)
    counts: "collections.Counter[str]" = collections.Counter()
    for records in records_by_role.values():
        for record in records:
            if record.get("r") not in ("span", "mark") or record.get("ph") not in ("live", None):
                continue
            stage = record.get("stage")
            counts[stage] += 1
            if record.get("frame_id") is not None:
                seen[stage].add(int(record["frame_id"]))
    out = {}
    for stage, spec in SV.STAGES.items():
        scope = spec["scope"]
        row: dict[str, Any] = {"scope": scope, "records": counts.get(stage, 0)}
        population = denominators.get(scope)
        if population is not None:
            covered = seen.get(stage, set()) & population
            missing = sorted(population - covered)
            row.update({"denominator": len(population), "covered": len(covered),
                        "missing": len(missing), "missing_examples": missing[:10],
                        "coverage": (len(covered) / len(population)) if population else None})
        out[stage] = row
    return out


def durations(records_by_role: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    wall: dict[str, list] = collections.defaultdict(list)
    gpu: dict[str, list] = collections.defaultdict(list)
    status: dict[str, "collections.Counter[str]"] = collections.defaultdict(collections.Counter)
    for records in records_by_role.values():
        by_seq = {}
        for record in records:
            if record.get("r") == "span" and record.get("ph") in ("live", None):
                wall[record["stage"]].append((record["end_raw_ns"] - record["start_raw_ns"]) / 1e6)
                by_seq[record["seq"]] = record["stage"]
        for record in records:
            if record.get("r") == "cuda" and record.get("span_seq") in by_seq:
                stage = by_seq[record["span_seq"]]
                status[stage][record.get("status")] += 1
                if record.get("gpu_ms") is not None:
                    gpu[stage].append(record["gpu_ms"])
    return {stage: {"wall_ms": dist(values),
                    "cuda_ms": dist(gpu.get(stage, [])) if stage in gpu or stage in status
                    else None,
                    "cuda_status": dict(status.get(stage, {}))}
            for stage, values in sorted(wall.items())}


# ---------------------------------------------------------------------------
# sampler, overhead, clocks
# ---------------------------------------------------------------------------


def sampler_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = [r for r in records if r.get("r") == "sample"]
    footer = next((r for r in records if r.get("r") == "footer"), None)
    header = next((r for r in records if r.get("r") == "header"), None)
    times = [s["t_raw_ns"] for s in samples]
    intervals = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    roles: dict[str, set] = collections.defaultdict(set)
    unsupported_gpu_fields = collections.Counter()
    for sample in samples:
        for row in sample.get("procs") or ():
            roles[row["role"]].add(row["pid"])
        for key, value in (sample.get("gpu") or {}).items():
            if value is None:
                unsupported_gpu_fields[key] += 1
    peak = {}
    for key in ("util_gpu_pct", "util_mem_pct", "mem_used_bytes", "power_mw", "temp_c",
                "clock_sm_mhz"):
        values = [s["gpu"][key] for s in samples
                  if s.get("gpu") and s["gpu"].get(key) is not None]
        peak[key] = dist(values)
    run_queue = dist([s["host"].get("procs_running") for s in samples if s.get("host")])
    return {"present": header is not None, "footer_present": footer is not None,
            "samples": len(samples), "interval_ms": dist(intervals),
            "dropped_samples": None if footer is None else footer.get("dropped_samples"),
            "sample_cost": None if footer is None else footer.get("sample_cost"),
            "sampler_cpu_pct_of_one_core": None if footer is None
            else footer.get("sampler_cpu_pct_of_one_core"),
            "counters": None if footer is None else footer.get("counters"),
            "nvml": None if header is None else header.get("nvml"),
            "nvml_error": None if header is None else header.get("nvml_error"),
            "nvml_unsupported": None if footer is None else footer.get("nvml_unsupported"),
            "roles_observed": {role: sorted(pids) for role, pids in roles.items()},
            "gpu": peak, "gpu_fields_missing": dict(unsupported_gpu_fields),
            "procs_running": run_queue}


def overhead_report(records_by_role: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    out = {}
    for role, records in records_by_role.items():
        footer = next((r for r in records if r.get("r") == "footer"), None)
        header = next((r for r in records if r.get("r") == "header"), None)
        if footer is None or header is None:
            out[role] = {"footer_present": footer is not None, "header_present": header is not None}
            continue
        duration_ns = footer["t_raw_ns"] - header["clock"]["raw_ns"]
        hot = int(footer.get("hot_path_ns_total") or 0)
        calls = int(footer.get("hot_path_calls") or 0)
        writer = int(footer.get("writer_cpu_ns") or 0)
        out[role] = {"duration_s": duration_ns / 1e9, "hot_path_ns_total": hot,
                     "hot_path_calls": calls,
                     "hot_path_ns_per_call": (hot / calls) if calls else None,
                     "writer_cpu_ns": writer,
                     "profiler_cpu_fraction_of_one_core": ((hot + writer) / duration_ns
                                                           if duration_ns > 0 else None),
                     "counters": footer.get("counters"), "gauge_errors": footer.get("gauge_errors"),
                     "bytes_written": footer.get("bytes_written"),
                     "cuda_state": footer.get("cuda_state")}
    return out


def clock_report(streams: Mapping[str, Sequence[Mapping[str, Any]]],
                 sampler: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per = {}
    for role, records in streams.items():
        offsets = [r["wall_ns"] - r["raw_ns"] for r in records if r.get("r") == "clock"]
        header = next((r for r in records if r.get("r") == "header"), None)
        if header is not None:
            offsets.append(header["clock"]["wall_ns"] - header["clock"]["raw_ns"])
        if offsets:
            per[role] = {"n": len(offsets), "median_offset_ns": int(statistics.median(offsets)),
                         "spread_ns": max(offsets) - min(offsets),
                         "boot_id": None if header is None else header.get("boot_id"),
                         "clock_domain": None if header is None else header.get("clock_domain")}
    offsets = [s["t_wall_ns"] - s["t_raw_ns"] for s in sampler if s.get("r") == "sample"]
    if offsets:
        per["sampler"] = {"n": len(offsets), "median_offset_ns": int(statistics.median(offsets)),
                          "spread_ns": max(offsets) - min(offsets), "clock_domain":
                          "CLOCK_MONOTONIC_RAW"}
    medians = [v["median_offset_ns"] for v in per.values()]
    spread = (max(medians) - min(medians)) if medians else None
    boots = {v.get("boot_id") for v in per.values() if v.get("boot_id")}
    return {"per_process": per, "cross_process_median_spread_ns": spread,
            "tolerance_ns": CROSS_PROCESS_CLOCK_TOLERANCE_NS,
            "single_raw_domain": (spread is not None and spread <= CROSS_PROCESS_CLOCK_TOLERANCE_NS
                                  and len(boots) <= 1),
            "note": "realtime-raw offset per process; equal medians mean the processes "
                    "(host and container) read the same CLOCK_MONOTONIC_RAW"}


# ---------------------------------------------------------------------------
# semantics
# ---------------------------------------------------------------------------


def _import(path: str) -> Any:
    module, _, attr = path.partition(":")
    value = importlib.import_module(module)
    return getattr(value, attr) if attr else value


def actor_replay(ue_records: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]],
                 *, loader: str, method: str) -> dict[str, Any]:
    """Replay every live actor call offline; compare with the live decision."""
    actor = _import(loader)()
    act = getattr(actor, method)
    by_frame = {int(d["frame_id"]): d for d in decisions}
    mismatches, replayed = [], 0
    for record in ue_records:
        if record.get("r") != "span" or record.get("stage") != "agent.actor":
            continue
        features = record.get("features")
        if features is None:
            mismatches.append({"frame_id": record.get("frame_id"), "reason": "no features"})
            continue
        decision = act(tuple(float(v) for v in features))
        replayed += 1
        live = by_frame.get(int(record.get("frame_id", -1)), {})
        got = (decision.mode_id, decision.q_e4, decision.state_sha256, decision.heads_sha256)
        recorded = (record.get("decided_mode_id"), record.get("decided_q_e4"),
                    record.get("state_sha256"), record.get("heads_sha256"))
        planned = (live.get("mode_id"), live.get("q_e4"))
        if got != recorded or planned != got[:2]:
            mismatches.append({"frame_id": record.get("frame_id"), "replayed": got,
                               "recorded": recorded, "planned": planned})
    decision_frames = [d for d in decisions if d.get("kind") == "POLICY_DECISION"]
    return {"replayed": replayed, "policy_decisions": len(decision_frames),
            "mismatches": len(mismatches), "mismatch_examples": mismatches[:10],
            "passed": replayed == len(decision_frames) and not mismatches}


def reward_semantics(ue: Mapping[str, Any], *, contract_module: str,
                     controller_module: str) -> dict[str, Any]:
    contract = _import(contract_module)
    controller = _import(controller_module)
    deadline = int(controller.REWARD_DEADLINE_NS)
    timeout_elapsed = int(controller.TIMEOUT_RESOLUTION_ELAPSED_NS)
    weight = float(contract.REWARD_LATENCY_WEIGHT)
    deadline_ms = float(contract.REWARD_DEADLINE_MS)
    failure = float(contract.REGISTERED_FAILURE_REWARD)
    errors = []
    terminals = collections.Counter()
    for resolution in ue.get("resolutions") or ():
        terminal = resolution.get("terminal")
        terminals[terminal] += 1
        elapsed = int(resolution["resolution_timestamp_ns"]) - int(
            resolution["action_open_timestamp_ns"])
        reward = resolution.get("reward")
        if terminal == "SUCCESS":
            expected = float(resolution["q_perc"]) - weight * ((elapsed / 1_000_000.0)
                                                               / deadline_ms)
            if elapsed > deadline or reward is None or reward != expected:
                errors.append({"terminal": terminal, "elapsed_ns": elapsed, "reward": reward,
                               "expected": expected})
        elif terminal == "TIMEOUT":
            if reward != failure or elapsed != timeout_elapsed:
                errors.append({"terminal": terminal, "elapsed_ns": elapsed, "reward": reward})
        elif terminal in ("REGISTERED_SERVICE_FAILURE", "REGISTERED_DELIVERY_FAILURE"):
            if reward != failure:
                errors.append({"terminal": terminal, "reward": reward})
        elif terminal in ("EVALUATOR_FAULT", "INFRASTRUCTURE_FAULT"):
            if reward is not None or resolution.get("learning_included") is not False:
                errors.append({"terminal": terminal, "reward": reward})
    # feedback classes against the inclusive deadline
    action_open = {}
    for decision in ue.get("decisions") or ():
        if decision.get("kind") == "POLICY_DECISION" and decision.get("action_open"):
            action_open[int(decision["frame_id"])] = int(decision["action_open"]["ns"])
    class_errors = []
    for row in ue.get("feedback_rows") or ():
        opened = action_open.get(int(row["frame_id"]))
        if opened is None:
            continue
        elapsed = int(row["receipt"]["ns"]) - opened
        if row["class"] == "ACCEPTED" and elapsed > deadline:
            class_errors.append({"frame_id": row["frame_id"], "class": row["class"],
                                 "elapsed_ns": elapsed})
        if row["class"] == "LATE_ORPHAN" and elapsed <= deadline:
            class_errors.append({"frame_id": row["frame_id"], "class": row["class"],
                                 "elapsed_ns": elapsed})
    return {"deadline_ns": deadline, "timeout_resolution_elapsed_ns": timeout_elapsed,
            "inclusive": True, "terminals": dict(terminals),
            "resolution_errors": len(errors), "resolution_error_examples": errors[:10],
            "feedback_class_errors": len(class_errors),
            "feedback_class_error_examples": class_errors[:10],
            "passed": not errors and not class_errors}


def receipt_mark_agreement(ue_records: Sequence[Mapping[str, Any]],
                           ue: Mapping[str, Any]) -> dict[str, Any]:
    stamps = collections.defaultdict(list)
    for row in ue.get("feedback_rows") or ():
        stamps[int(row["frame_id"])].append(int(row["receipt"]["ns"]))
    marks = [r for r in ue_records if r.get("r") == "mark"
             and r.get("stage") == "ue.feedback_receipt"]
    disagreements = [m["frame_id"] for m in marks
                     if int(m["t_raw_ns"]) not in stamps.get(int(m["frame_id"]), [])]
    return {"marks": len(marks), "feedback_rows": sum(len(v) for v in stamps.values()),
            "disagreements": len(disagreements), "examples": disagreements[:10],
            "passed": len(marks) == sum(len(v) for v in stamps.values()) and not disagreements}


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def _csv(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(output_root: Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(output_root)
    cells = sorted((output_root / "cells").iterdir()) if (output_root / "cells").is_dir() else []
    if len(cells) != 1:
        raise RuntimeError(f"expected exactly one cell under {output_root}, found {len(cells)}")
    cell = cells[0]
    files = {"ue": cell / "profiler" / "ue.profile.jsonl",
             "edge": cell / "profiler" / "edge" / "edge.profile.jsonl",
             "map": cell / "profiler" / "map.profile.jsonl"}
    sampler_path = output_root / "profiler" / "sampler.jsonl"
    streams = {role: load(path) for role, path in files.items()}
    sampler = load(sampler_path)
    ue = json.loads((cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json").read_text())
    transmitted = {int(x["frame_id"]): dict(x["run4_identity"])
                   for x in ue.get("transmitted_identities") or ()}
    decisions = list(ue.get("decisions") or ())
    terminals = _csv(cell / "map_feedback.csv")
    installed = {int(r["frame_id"]) for r in terminals if r.get("outcome") == "RESULT_INSTALLED"}

    def frames(role: str, stage: str) -> set:
        return {int(r["frame_id"]) for r in streams.get(role, ())
                if r.get("stage") == stage and r.get("ph") in ("live", None)
                and r.get("frame_id") is not None}

    ticks = {r.get("seq") for r in streams["ue"] if r.get("stage") == "carla.world_tick"}
    denominators = {
        "sent": set(transmitted),
        "decision": {int(d["frame_id"]) for d in decisions if d.get("kind") == "POLICY_DECISION"},
        "planned": {int(d["frame_id"]) for d in decisions} & set(transmitted),
        "edge": frames("edge", "edge.reassembly_complete") & set(transmitted),
        "processed": {int(r["frame_id"]) for r in streams["edge"]
                      if r.get("stage") == "edge.process" and r.get("ok")
                      and r.get("ph") == "live" and r.get("frame_id") is not None}
        & set(transmitted),
        "reward": frames("edge", "edge.evaluator_submit") & set(transmitted),
        "installed": installed,
    }
    validity = {role: validate_stream(records, source=str(files[role]))
                for role, records in streams.items()}
    identity = {role: identity_conflicts(records, transmitted, pre_send_stages=PRE_SEND_STAGES)
                for role, records in streams.items()}
    report: dict[str, Any] = {
        "schema": SCHEMA, "output_root": str(output_root), "cell": str(cell),
        "claim": "INSTRUMENTATION_QUALIFICATION_NOT_A_PROFILING_BASELINE",
        "files": {role: {"path": str(path), "exists": Path(path).is_file(),
                         "bytes": Path(path).stat().st_size if Path(path).is_file() else None}
                  for role, path in {**files, "sampler": sampler_path}.items()},
        "counts": {"transmitted": len(transmitted), "decisions_recorded": len(decisions),
                   "world_ticks": len(ticks),
                   "denominators": {k: len(v) for k, v in denominators.items()}},
        "validity": validity, "identity": identity,
        "coverage": coverage(streams, denominators), "durations": durations(streams),
        "sampler": sampler_report(sampler), "overhead": overhead_report(streams),
        "clocks": clock_report(streams, sampler),
    }
    semantics: dict[str, Any] = {}
    policy = dict(config.get("semantics") or {})
    if policy.get("actor_loader"):
        try:
            semantics["actor_replay"] = actor_replay(
                streams["ue"], decisions, loader=policy["actor_loader"],
                method=policy.get("act_on_vector", "act_on_vector"))
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            semantics["actor_replay"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    if policy.get("contract_module") and policy.get("controller_module"):
        semantics["reward"] = reward_semantics(ue, contract_module=policy["contract_module"],
                                               controller_module=policy["controller_module"])
    semantics["receipt_marks"] = receipt_mark_agreement(streams["ue"], ue)
    report["semantics"] = semantics
    report["gauges"] = gauge_report(streams)
    return report


def gauge_report(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    values: dict[str, list] = collections.defaultdict(list)
    for records in streams.values():
        for record in records:
            if record.get("r") == "gauge":
                for name, value in (record.get("values") or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) \
                            and math.isfinite(float(value)):
                        values[name].append(float(value))
    return {name: dist(series) for name, series in sorted(values.items())}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = json.loads((args.output_root / "PROFILER_CONFIG.json").read_text())
    report = analyze(args.output_root, config=config)
    with args.write.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
