#!/usr/bin/env python3
"""Addendum-5 engineering gates for the bounded 120-frame qualification.

Pure evaluation of one completed Phase-6 attempt from durable evidence. The
gates are the registered engineering conditions, and the P0-P8 values are
taken unchanged from ``CELL_RESULT.json``.

    python3 -m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_engineering_gates_v2 \
        <attempt cell dir> [--write <create-only json>]

Importing this module performs no I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

READY_WAIT_BOUND_MS = 250.0      # pre-repair head-of-line stall was ~2,000 ms
SERVICE_OUTCOMES = ("SUPERSEDED_PENDING", "STALE_BEFORE_EDGE", "STALE_BEFORE_MAP")


def _ns(stamp: Any) -> Optional[int]:
    if isinstance(stamp, Mapping):
        return int(stamp["ns"])
    return None if stamp in (None, "") else int(stamp)


def ordering_violations(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Addendum 6: frames violating the RUN4_CONTRACT latency boundary.

    Sensor preparation (radar raster, RGB conversion, SI/P40) and state commit
    precede action-open; 7-channel/front/send follow it.
    """
    from . import phase6_ue_runtime_v2 as U

    bad = []
    for record in decisions:
        stages = record.get("stages") or {}
        if stages.get("radar_tensor_end_raw_ns") is None or stages.get(
                "si_p40_end_raw_ns") is None:
            bad.append({"frame_id": record.get("frame_id"), "why": "NO_SENSOR_STAGES"})
            continue
        violations = U.boundary_violations(record)
        if violations:
            bad.append({"frame_id": record.get("frame_id"), "why": violations})
    return bad


def e5_status(edge: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Non-vacuous evaluator gate: PASS only with real, prompt GT reads."""
    records = list(edge.get("evaluations") or ())
    evaluator = edge.get("evaluator") or {}
    waits = ready_wait_ms(records)
    ready = [r for r in records if (r.get("timing") or {}).get("gt_ready_detected_wall_ns")]
    exceptions = [r for r in records if r.get("reason") == "EVALUATOR_EXCEPTION"]
    detail = {"ready_wait_n": len(waits), "gt_bundles_read": len(ready),
              "max_wait_ms": max(waits) if waits else None,
              "reader_exceptions": len(exceptions),
              "duplicate_emission_refused": int(evaluator.get("duplicate_emission_refused", 0)),
              "queue_overflow": int(evaluator.get("queue_overflow", 0))}
    if not ready or not waits:
        return "INCONCLUSIVE_NO_GT_READY", detail
    ok = (all(w <= READY_WAIT_BOUND_MS for w in waits) and not exceptions
          and detail["duplicate_emission_refused"] == 0 and detail["queue_overflow"] == 0)
    return ("PASS" if ok else "FAIL"), detail


def ready_wait_ms(records: Sequence[Mapping[str, Any]]) -> list[float]:
    out = []
    for record in records:
        timing = record.get("timing") or {}
        ready = timing.get("gt_ready_detected_wall_ns")
        start = timing.get("evaluator_start_wall_ns")
        if ready is not None and start is not None:
            out.append((int(start) - int(ready)) / 1e6)
    return out


def evaluate(cell: Path) -> dict[str, Any]:
    cell = Path(cell)
    result = json.loads((cell / "CELL_RESULT.json").read_text(encoding="utf-8"))
    ue = json.loads((cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json").read_text())
    edge = json.loads((cell / "run4_phase6" / "edge_report.json").read_text())
    with (cell / "map_feedback.csv").open(newline="") as handle:
        terminals = list(csv.DictReader(handle))
    gates6 = (result.get("phase6") or {}).get("gates") or {}
    summary = (result.get("phase6") or {}).get("result_summary") or {}
    decisions = list(ue.get("decisions") or ())
    reward_frames = {int(f["frame_id"]) for f in ue.get("frames") or ()
                     if f.get("reward_requested")}
    terminal_frames = {int(r["frame_id"]) for r in ue.get("terminal_rows") or ()}
    missing_terminal = sorted(
        int(r["frame_id"]) for r in terminals
        if r.get("terminal_source") == "EDGE_INFERENCE_SERVICE"
        and r.get("outcome") in SERVICE_OUTCOMES and int(r["frame_id"]) in reward_frames
        and int(r["frame_id"]) not in terminal_frames)
    classes = [r.get("class") for c in ue.get("feedback_ledgers") or () for r in c]
    sent = int(((result.get("child") or {}).get("collector") or {}).get(
        "transmitted_frames", -1))
    unique_terminal_frames = len({r["frame_id"] for r in terminals})
    waits = ready_wait_ms(edge.get("evaluations") or ())
    order_bad = ordering_violations([d for d in decisions if d.get("kind")])
    camera_age = [f for f in ue.get("fallback_log") or ()
                  if any("camera_si is stale" in r for r in f.get("reasons") or ())]
    e5, e5_detail = e5_status(edge)
    gates = {
        "E1_P0_POLICY_ADMISSION": bool(gates6.get("P0_POLICY_COVERAGE")),
        "E2_TIMING_BOUNDARY_EVERY_FRAME": not order_bad,
        "E3_EXACT_TERMINAL_ACCOUNTING": (
            bool(gates6.get("P4_ACCOUNTING")) and summary.get("open_at_stop") == 0
            and unique_terminal_frames == sent
            and ue.get("unresolved_tickets_at_close") == 0),
        "E4_SUPERSEDED_REWARD_TERMINALS_DELIVERED": not missing_terminal,
        "E5_NON_VACUOUS_EVALUATOR": e5 == "PASS",
        "E6_NO_CONFLICT_ORPHAN_OR_FAULT": (
            ue.get("faulted") is None and "UNKNOWN_ORPHAN" not in classes
            and "CONFLICT" not in classes and bool(gates6.get("P8_NO_INFRASTRUCTURE_FAULT"))
            and int(edge.get("evaluator", {}).get("duplicate_emission_refused", 0)) == 0),
        "E7_ACTOR_ONLY_ADMITTED": bool(gates6.get("P5_NO_ACTOR_AFTER_REFUSAL")),
        "E9_HOST_AND_CHANNEL_COLD": bool(gates6.get("P6_RESTORE_COLD")),
    }
    return {
        "schema": "scenesense.run4_live_v2.phase6_engineering_gates.v1",
        "addendum": "PHASE6_REPAIR_DIAGNOSTIC_ADDENDUM_6",
        "gates": gates,
        "evidence": {
            "p0": (result.get("phase6") or {}).get("reported_not_gated", {}).get(
                "fallback_fraction"),
            "ordering_violations": order_bad[:20],
            "camera_age_fallbacks": len(camera_age),
            "sent": sent, "unique_terminal_frames": unique_terminal_frames,
            "open_at_stop": summary.get("open_at_stop"),
            "reward_terminals_missing": missing_terminal,
            "evaluator_ready_wait_ms_max": max(waits) if waits else None,
            "evaluator_ready_wait_n": len(waits),
            "evaluator_counters": edge.get("evaluator"),
            "feedback_classes": {c: classes.count(c) for c in sorted(set(map(str, classes)))},
            "e5_status": e5, "e5_detail": e5_detail,
            "stop_reason": (result.get("child") or {}).get("stop_reason"),
            "unresolved_tickets_at_close": ue.get("unresolved_tickets_at_close"),
        },
        "note": "E8 (actor/checkpoint digest) is verified separately by the "
                "frozen-actor audit before and after the run.",
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate(args.cell)
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    if args.write is not None:
        with args.write.open("x", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
