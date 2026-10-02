#!/usr/bin/env python3
"""Reproducible A/B comparison of two profiled captures (read-only inputs).

Both arms are read from their output roots: the frozen ``export_v1`` tables
(verified against their manifests first), the Phase-6 timing characterization,
the profiler analysis, the UE evidence and the thread-cap evidence.  Resource
series are restricted to the identical **live window** of each arm (first
``ue.worker_frame`` start to last end), so setup and teardown are excluded the
same way in both.

Primary: feedback within the inclusive 170 ms over ALL policy decisions;
action-open -> feedback P50/P95/P99 over received feedback, with missing
feedback counted separately.  Secondary: outcome accounting, fallbacks and
decision/hold counts, UE CPU and threads, host busy and run queue, stage
timings, GPU utilization/memory/power, freshness, profiler overhead and
workload differences.  Differences are descriptive; one pair is engineering
evidence, not causal proof.  Output is create-only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import export_v1 as EX
from . import threadcap_v1 as TC

SCHEMA = "scenesense.single_gpu_profiler.threadcap_ab_comparison.v1"
STAGES = (
    ("carla.world_tick", "CARLA world tick"),
    ("ue.worker_frame", "UE frame worker (whole)"), ("ue.sensor_wait", "sensor wait"),
    ("ue.radar_window", "radar window"), ("ue.radar_rasterize", "radar rasterize"),
    ("ue.rgb_convert", "RGB convert"), ("ue.scene_snapshot", "scene snapshot"),
    ("ue.si_p40", "scene SI/P40"), ("agent.telemetry_snapshot", "telemetry snapshot"),
    ("agent.state_build", "state build"), ("agent.actor", "actor"),
    ("ue.input_7ch", "7-channel input"), ("ue.codec.total_ue_preparation", "UE codec total"),
    ("ue.codec.front_backbone", "front backbone"), ("ue.codec.ranker_selection", "ranker"),
    ("ue.codec.ae_encode", "AE encode"), ("ue.codec.quantize_pack", "quantize/pack"),
    ("ue.codec.zstd_compression", "zstd"), ("ue.sfd4_pack", "SFD4 pack"),
    ("ue.udp_send", "datagram burst"), ("edge.compute", "edge compute"),
    ("edge.codec.frozen_tail", "edge frozen tail"), ("edge.publish_cpu", "edge publish CPU"),
    ("edge.map_publish", "edge map publish"), ("edge.qperc_evaluate", "Q_perc evaluate"),
    ("edge.feedback_emit", "feedback emit"), ("map.install", "map install"),
    ("ue.gt_build", "UE GT build"),
)
TIMING = (
    ("action_open_to_feedback", "action-open -> feedback (received)"),
    ("capture_to_feedback", "capture -> feedback (received)"),
    ("capture_to_anchor", "capture -> action-open/plan"),
    ("anchor_to_first_datagram", "action-open -> first datagram"),
    ("anchor_to_last_datagram", "action-open -> last datagram"),
    ("last_datagram_to_uplink_complete", "last datagram -> edge reassembly"),
    ("edge_queue", "edge queue"), ("edge_compute", "edge compute (wall stamps)"),
    ("reward_gt_ready_after_action_open", "reward GT ready after action-open"),
    ("evaluator", "evaluator"), ("feedback_emit_after_action_open", "feedback emit after action-open"),
    ("capture_to_map_install", "capture -> map install"),
    ("anchor_to_map_install", "action-open -> map install"),
    ("action_open_to_next_eligible", "action-open -> next eligible decision"),
)


def pct(values: Sequence[float], p: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, -(-int(p * len(ordered)) // 100)) - 1]


def dist(values: Iterable[Optional[float]]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None and v != ""]
    return {"n": len(present), "mean": statistics.fmean(present) if present else None,
            "p50": pct(present, 50), "p95": pct(present, 95), "p99": pct(present, 99),
            "max": max(present) if present else None}


def _csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> Optional[float]:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


class Arm:
    def __init__(self, root: Path, label: str) -> None:
        self.root, self.label = Path(root), label
        self.export = self.root / "export_v1"
        self.verify = EX.verify(self.export)
        TC.require(self.verify["verified"], f"{label}: export manifest mismatch {self.verify}")
        self.manifest = json.loads((self.export / "MANIFEST.json").read_text())
        self.result = json.loads((self.root / "QUALIFICATION_RESULT.json").read_text())
        self.timing = json.loads((self.root / "PHASE6_TIMING_CHARACTERIZATION.json").read_text())
        self.analysis = json.loads((self.root / "PROFILER_ANALYSIS.json").read_text())
        cells = sorted((self.root / "cells").iterdir())
        self.cell = cells[0]
        self.ue = json.loads((self.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text())
        cell_result = json.loads((self.cell / "CELL_RESULT.json").read_text())
        self.collector = dict(((cell_result.get("child") or {}).get("collector")) or {})
        self.spans = _csv(self.export / "spans.csv")
        self.frames = _csv(self.export / "frames.csv")
        self.gpu = _csv(self.export / "gpu.csv")
        self.procs = _csv(self.export / "procs.csv")
        self.threads = _csv(self.export / "threads.csv")
        self.window = self._live_window()

    def _live_window(self) -> tuple[int, int]:
        starts = [int(r["start_raw_ns"]) for r in self.spans
                  if r["stage"] == "ue.worker_frame" and r["phase"] == "live"]
        ends = [int(r["end_raw_ns"]) for r in self.spans
                if r["stage"] == "ue.worker_frame" and r["phase"] == "live"]
        return (min(starts), max(ends))

    def _in_window(self, row: Mapping[str, Any]) -> bool:
        t = int(row["t_raw_ns"])
        return self.window[0] <= t <= self.window[1]

    def metrics(self) -> dict[str, Any]:
        t, a = self.timing, self.analysis
        kinds = dict(t["counts"]["by_kind"])
        decisions = int(kinds.get("POLICY_DECISION", 0))
        terminals = dict(t["outcomes"]["resolution_terminals"])
        timely = t["policy_deadline_action_open_to_feedback_le_170"]
        ao = t["distributions_ms"]["action_open_to_feedback"]
        reasons: "collections.Counter[str]" = collections.Counter()
        for decision in self.ue.get("decisions") or ():
            if decision.get("kind") == "FALLBACK":
                for reason in decision.get("fallback_reasons") or ("UNSPECIFIED",):
                    reasons[":".join(str(reason).split(":")[:2])] += 1
        ue_rows = [r for r in self.procs if r["role"] == "ue" and self._in_window(r)]
        busy_by_tick: dict[str, int] = collections.defaultdict(int)
        tids_with_cpu = set()
        for r in self.threads:
            if r["role"] == "ue" and self._in_window(r) and int(r["delta_ticks"] or 0) > 0:
                busy_by_tick[r["t_raw_ns"]] += 1
                tids_with_cpu.add(r["tid"])
        gpu_rows = [r for r in self.gpu if self._in_window(r)]
        threadcap = dict(self.result.get("threadcap") or {})
        effective = dict(threadcap.get("effective") or {})
        overhead = dict(a["overhead"].get("ue") or {})
        sent = int(t["counts"]["sent_frames"])
        return {
            "identity": {"run_id": self.result.get("run_id"), "root": str(self.root),
                         "arm": (self.result.get("arm") or {}).get("arm"),
                         "verdict": self.result.get("verdict"), "gates": self.result.get("gates"),
                         "g1_branch": (self.result.get("g1_detail") or {}).get("branch"),
                         "manifest_sha256": EX._sha(self.export / "MANIFEST.json"),
                         "export_sha256": self.manifest["export_sha256"],
                         "threadcap_checks": threadcap.get("checks"),
                         "live_window_s": (self.window[1] - self.window[0]) / 1e9},
            "primary": {
                "policy_decisions": decisions,
                "timely_feedback_le_170ms": {"count": timely["count"], "n": timely["n"],
                                             "fraction": timely["fraction"]},
                "action_open_to_feedback_ms": {k: ao.get(k) for k in ("n", "p50", "p90", "p95", "p99", "max")},
                "missing_feedback": ao.get("missing"),
                "feedback_classes": dict(t["outcomes"]["feedback_classes"]),
            },
            "outcomes": {
                "terminals": terminals,
                "defined_quality_success": terminals.get("SUCCESS", 0),
                "exclusions_undefined_quality": terminals.get("EVALUATOR_FAULT", 0)
                + terminals.get("INFRASTRUCTURE_FAULT", 0),
                "timeouts": terminals.get("TIMEOUT", 0),
                "service_failures": terminals.get("REGISTERED_SERVICE_FAILURE", 0)
                + terminals.get("REGISTERED_DELIVERY_FAILURE", 0),
                "evaluator": dict(t["outcomes"]["evaluator_counters"]),
                "map_outcomes": dict(t["outcomes"]["map_outcomes"]),
                "system_kpi_capture_to_feedback_le_200": t["system_kpi_capture_to_feedback_le_200"],
                "feedback_before_next_eligible": t["feedback_before_next_eligible_decision"],
            },
            "workload": {
                "frames_sent": sent, "kinds": kinds, "fallback_reasons": dict(reasons),
                "world_ticks": a["counts"].get("world_ticks"),
                "dropped_preparation_opportunities": self.collector.get("dropped_preparation_opportunities"),
                "payload_bytes": dist(_f(r["payload_bytes"]) for r in self.frames),
                "datagrams": dist(_f(r["datagrams"]) for r in self.frames),
                "ticket_tensors": {"mean": (sent - int(kinds.get("FALLBACK", 0))) / decisions
                                   if decisions else None},
            },
            "ue_process": {
                "cpu_pct": dist(_f(r["cpu_pct"]) for r in ue_rows),
                "threads": dist(_f(r["threads"]) for r in ue_rows),
                "rss_bytes": dist(_f(r["rss_bytes"]) for r in ue_rows),
                "busy_threads_per_sample": dist(busy_by_tick.values()),
                "distinct_tids_with_cpu": len(tids_with_cpu),
                "samples": len(ue_rows),
                "effective": {k: effective.get(k) for k in (
                    "torch_intra_op_threads", "torch_inter_op_threads", "openmp_max_threads",
                    "cap_effective", "cv2", "affinity_cpus")},
                "ue_environment": effective.get("ue_environment"),
                "launch_restore": effective.get("launch_restore"),
                "scope": {k: (threadcap.get("inventory") or {}).get(k)
                          for k in ("scope_ok", "scope_violations", "roles", "unreadable")},
            },
            "host": {
                "cpu_busy_pct": dist(_f(r["cpu_busy_pct"]) for r in gpu_rows),
                "procs_running": dist(_f(r["procs_running"]) for r in gpu_rows),
                "load1": dist(_f(r["load1"]) for r in gpu_rows),
                "samples": len(gpu_rows),
                "sampler": {k: a["sampler"].get(k) for k in ("samples", "dropped_samples",
                                                             "interval_ms", "sampler_cpu_pct_of_one_core")},
                "edge_rows_in_procs_csv": sum(1 for r in self.procs if r["role"] == "edge"),
            },
            "gpu": {
                "util_gpu_pct": dist(_f(r["util_gpu_pct"]) for r in gpu_rows),
                "util_mem_pct": dist(_f(r["util_mem_pct"]) for r in gpu_rows),
                "mem_used_gib": dist((_f(r["mem_used_bytes"]) or 0) / 2**30 for r in gpu_rows
                                     if _f(r["mem_used_bytes"]) is not None),
                "power_w": dist((_f(r["power_mw"]) or 0) / 1e3 for r in gpu_rows
                                if _f(r["power_mw"]) is not None),
                "temp_c": dist(_f(r["temp_c"]) for r in gpu_rows),
            },
            "timing_ms": {key: {k: t["distributions_ms"][key].get(k)
                                for k in ("n", "missing", "p50", "p95", "p99", "max")}
                          for key, _label in TIMING if key in t["distributions_ms"]},
            "stages_ms": {key: {"wall": {k: (a["durations"].get(key) or {}).get("wall_ms", {}).get(k)
                                         for k in ("n", "p50", "p95", "p99", "max")},
                                "cuda": {k: ((a["durations"].get(key) or {}).get("cuda_ms") or {}).get(k)
                                         for k in ("n", "p50", "p95", "p99")}}
                          for key, _label in STAGES if key in a["durations"]},
            "profiler": {
                "ue_hot_path_ms_per_frame": (overhead.get("hot_path_ns_total") or 0) / 1e6 / sent,
                "ue_hot_path_calls": overhead.get("hot_path_calls"),
                "profiler_cpu_pct_one_core": {r: round(100 * (o.get("profiler_cpu_fraction_of_one_core") or 0), 3)
                                              for r, o in a["overhead"].items()},
                "cuda_unresolved": {r: (o.get("counters") or {}).get("cuda_unresolved", 0)
                                    for r, o in a["overhead"].items()},
            },
        }


def _delta(a: Any, b: Any) -> Optional[float]:
    return None if a is None or b is None else float(b) - float(a)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    lines = ["# UE numerical-thread cap: A (baseline) vs B (OMP/MKL = 1)", "",
             f"A: `{a['identity']['run_id']}` ({a['identity']['verdict']}, G1 {a['identity']['g1_branch']}); "
             f"B: `{b['identity']['run_id']}` ({b['identity']['verdict']}, G1 {b['identity']['g1_branch']}).",
             "Resource series are restricted to each arm's live window "
             f"(A {a['identity']['live_window_s']:.1f} s, B {b['identity']['live_window_s']:.1f} s). "
             "Profiler cost is included in every span. Differences are descriptive.", ""]
    rows: list[tuple[str, Any, Any, int]] = []
    pa, pb = a["primary"], b["primary"]
    rows += [("Policy decisions", pa["policy_decisions"], pb["policy_decisions"], 0),
             ("Feedback <= 170 ms / all decisions",
              f"{pa['timely_feedback_le_170ms']['count']}/{pa['timely_feedback_le_170ms']['n']} "
              f"({100 * pa['timely_feedback_le_170ms']['fraction']:.1f}%)",
              f"{pb['timely_feedback_le_170ms']['count']}/{pb['timely_feedback_le_170ms']['n']} "
              f"({100 * pb['timely_feedback_le_170ms']['fraction']:.1f}%)", 0),
             ("Missing feedback (decisions without reply)", pa["missing_feedback"], pb["missing_feedback"], 0)]
    for k in ("p50", "p95", "p99", "max"):
        rows.append((f"Action-open -> feedback {k} (ms, received)",
                     pa["action_open_to_feedback_ms"][k], pb["action_open_to_feedback_ms"][k], 1))
    oa, ob = a["outcomes"], b["outcomes"]
    for key, label in (("defined_quality_success", "SUCCESS (defined quality, timely)"),
                       ("exclusions_undefined_quality", "Excluded (undefined quality)"),
                       ("timeouts", "TIMEOUT"), ("service_failures", "Registered service failures")):
        rows.append((label, oa[key], ob[key], 0))
    wa, wb = a["workload"], b["workload"]
    for kind in ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK"):
        rows.append((f"Frames: {kind}", wa["kinds"].get(kind, 0), wb["kinds"].get(kind, 0), 0))
    rows.append(("Dropped preparation opportunities", wa["dropped_preparation_opportunities"],
                 wb["dropped_preparation_opportunities"], 0))
    ua, ub = a["ue_process"], b["ue_process"]
    rows += [("UE torch intra-op threads (effective)", ua["effective"]["torch_intra_op_threads"],
              ub["effective"]["torch_intra_op_threads"], 0),
             ("UE torch inter-op threads (effective)", ua["effective"]["torch_inter_op_threads"],
              ub["effective"]["torch_inter_op_threads"], 0),
             ("UE CPU % of one core, mean", ua["cpu_pct"]["mean"], ub["cpu_pct"]["mean"], 1),
             ("UE CPU % p95", ua["cpu_pct"]["p95"], ub["cpu_pct"]["p95"], 1),
             ("UE busy threads per sample, mean", ua["busy_threads_per_sample"]["mean"],
              ub["busy_threads_per_sample"]["mean"], 1),
             ("UE busy threads per sample, p95", ua["busy_threads_per_sample"]["p95"],
              ub["busy_threads_per_sample"]["p95"], 1),
             ("UE thread count, p50", ua["threads"]["p50"], ub["threads"]["p50"], 0)]
    ha, hb = a["host"], b["host"]
    rows += [("Host CPU busy %, mean", ha["cpu_busy_pct"]["mean"], hb["cpu_busy_pct"]["mean"], 1),
             ("Host CPU busy %, p95", ha["cpu_busy_pct"]["p95"], hb["cpu_busy_pct"]["p95"], 1),
             ("Run queue (procs_running) p50", ha["procs_running"]["p50"], hb["procs_running"]["p50"], 0),
             ("Run queue p95", ha["procs_running"]["p95"], hb["procs_running"]["p95"], 0)]
    ga, gb = a["gpu"], b["gpu"]
    rows += [("GPU util % mean", ga["util_gpu_pct"]["mean"], gb["util_gpu_pct"]["mean"], 1),
             ("GPU util % p95", ga["util_gpu_pct"]["p95"], gb["util_gpu_pct"]["p95"], 1),
             ("GPU memory used GiB p50", ga["mem_used_gib"]["p50"], gb["mem_used_gib"]["p50"], 2),
             ("GPU power W mean", ga["power_w"]["mean"], gb["power_w"]["mean"], 1)]
    for key, label in TIMING:
        if key in a["timing_ms"] and key in b["timing_ms"]:
            for k in ("p50", "p95", "p99"):
                rows.append((f"{label} {k}", a["timing_ms"][key][k], b["timing_ms"][key][k], 1))
    for key, label in STAGES:
        if key in a["stages_ms"] and key in b["stages_ms"]:
            for k in ("p50", "p95", "p99"):
                rows.append((f"{label} {k} (wall)", a["stages_ms"][key]["wall"][k],
                             b["stages_ms"][key]["wall"][k], 2))
    rows += [("Profiler hot path ms/frame (UE)", a["profiler"]["ue_hot_path_ms_per_frame"],
              b["profiler"]["ue_hot_path_ms_per_frame"], 3)]
    lines += ["| Metric | A | B | B − A |", "|---|---|---|---|"]
    for label, va, vb, digits in rows:
        delta = _delta(va, vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        lines.append(f"| {label} | {_fmt(va, digits)} | {_fmt(vb, digits)} | {_fmt(delta, digits)} |")
    lines += ["", "## Fallback reasons", "", "| Reason | A | B |", "|---|---|---|"]
    for reason in sorted(set(wa["fallback_reasons"]) | set(wb["fallback_reasons"])):
        lines.append(f"| {reason} | {wa['fallback_reasons'].get(reason, 0)} | "
                     f"{wb['fallback_reasons'].get(reason, 0)} |")
    lines += ["", "## Scope and effective settings", "",
              f"- A UE env: {ua['ue_environment']}", f"- B UE env: {ub['ue_environment']}",
              f"- A OpenMP max threads: {ua['effective']['openmp_max_threads']}; "
              f"B: {ub['effective']['openmp_max_threads']}",
              f"- A scope_ok: {ua['scope']['scope_ok']} (violations {len(ua['scope']['scope_violations'] or [])}); "
              f"B scope_ok: {ub['scope']['scope_ok']} (violations {len(ub['scope']['scope_violations'] or [])})",
              f"- Edge rows in procs.csv: A {ha['edge_rows_in_procs_csv']}, B {hb['edge_rows_in_procs_csv']} "
              "(sampler edge-process coverage gap retained in both arms, by design)", ""]
    return "\n".join(lines) + "\n"


def compare(a_root: Path, b_root: Path) -> dict[str, Any]:
    a, b = Arm(a_root, "A").metrics(), Arm(b_root, "B").metrics()
    deltas = {
        "timely_fraction_points": 100 * _delta(a["primary"]["timely_feedback_le_170ms"]["fraction"],
                                               b["primary"]["timely_feedback_le_170ms"]["fraction"]),
        "action_open_to_feedback_ms": {k: _delta(a["primary"]["action_open_to_feedback_ms"][k],
                                                 b["primary"]["action_open_to_feedback_ms"][k])
                                       for k in ("p50", "p95", "p99")},
        "ue_cpu_pct_mean": _delta(a["ue_process"]["cpu_pct"]["mean"], b["ue_process"]["cpu_pct"]["mean"]),
        "host_cpu_busy_pct_mean": _delta(a["host"]["cpu_busy_pct"]["mean"], b["host"]["cpu_busy_pct"]["mean"]),
        "procs_running_p50": _delta(a["host"]["procs_running"]["p50"], b["host"]["procs_running"]["p50"]),
    }
    return {"schema": SCHEMA, "A": a, "B": b, "deltas_b_minus_a": deltas,
            "caveats": ["one A/B pair; different live trajectories, action mixes and feedback histories",
                        "resource series over each arm's live window; NVML utilization is a sampled window",
                        "profiler cost (~1 ms/frame on the UE worker) included in all spans, not subtracted",
                        "sampler never matches the edge container process in either arm (known gap, retained)",
                        "the busy-thread/OpenMP explanation remains a hypothesis; this compares outcomes"]}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = compare(args.a, args.b)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "COMPARISON.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    markdown = render_markdown(report["A"], report["B"])
    with (args.out / "COMPARISON.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
