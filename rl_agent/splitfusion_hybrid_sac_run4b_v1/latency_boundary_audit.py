#!/usr/bin/env python3
"""Run-4B pre-training gate: does the modeled training latency exclude GT?

Run-4B defines ``L_ms`` as action-open to UE receipt of the immediate
``TAIL_OUTPUT_READY`` operational ACK, excluding GT evaluation and map
installation.  Training may only proceed if the existing modeled latency used
by the Run-4 collector already represents that operational path.

The Run-4 collector (``ue_production_transport_model_v2/collector_v1.py``)
composes every successful latency as::

    retained_residual + send_span + modeled_transport + actor_reserve
    retained_residual = model_prepare_start->ue_receive
                        - (first_feature_send->ue_receive
                           - evaluation_enqueued->ue_receive)

i.e. ``[prepare_start -> first_send] + [evaluation_enqueued -> ue_receive]``.
This audit re-derives that residual from the exact retained rows and splits
the second term into its recorded stages.  It is read-only: no CARLA, OAI,
Docker, CUDA, network or RNG use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ("experiments/splitfusion_quality_feedback_probe_v1/"
         "20260916_action50_favorable_adverse_retry4/cells")
CELLS = ("a50__favorable_stable", "a50__adverse_stable")
COLLECTOR = "rl_agent/ue_production_transport_model_v2/collector_v1.py"
SEAM_CONTRACT = "rl_agent/ue_production_queue_capture_v1/contract.py"
OUT = Path(__file__).resolve().parent / "LATENCY_BOUNDARY_AUDIT.json"

TOTAL = "model_prepare_start_to_ue_receive_ms"
FIRST = "first_feature_datagram_send_to_ue_receive_ms"
READY = "final_prediction_ready_to_ue_receive_ms"
ENQ = "evaluation_enqueued_to_ue_receive_ms"
STARTED = "evaluation_started_to_ue_receive_ms"
DONE = "evaluation_completed_to_ue_receive_ms"
EMIT = "ack_emit_start_to_ue_receive_ms"
SEND = "edge_socket_send_call_to_ue_receive_ms"
COLUMNS = (TOTAL, FIRST, READY, ENQ, STARTED, DONE, EMIT, SEND)

# Thresholds that the GT stages must satisfy for the residual to be a
# GT-free operational path.  Any nonzero GT wait/scoring in the pool fails.
GT_FREE_MAX_MS = 0.0


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))]


def _summary(values):
    return {"n": len(values), "p50_ms": _pct(values, 0.50),
            "p95_ms": _pct(values, 0.95), "p99_ms": _pct(values, 0.99),
            "min_ms": min(values), "max_ms": max(values),
            "mean_ms": sum(values) / len(values)}


def audit(root: Path = ROOT) -> dict:
    cells = {}
    pooled = {"gt_wait": [], "gt_scoring": [], "gt_total": [],
              "residual": [], "residual_gt_free": []}
    sources = {}
    for cell in CELLS:
        path = root / PROBE / cell / "quality_feedback_timing_join.csv"
        sources[str(path.relative_to(root))] = _sha_file(path)
        stages = {k: [] for k in (
            "residual_as_used_by_collector", "prepare_start_to_first_send",
            "evaluation_enqueued_to_ue_receive",
            "prediction_ready_to_evaluation_enqueued_not_in_residual",
            "gt_wait_evaluation_enqueued_to_started",
            "gt_scoring_evaluation_started_to_completed",
            "evaluation_completed_to_ack_emit", "ack_emit_to_socket_send",
            "socket_send_to_ue_receive_downlink",
            "residual_minus_gt_wait_and_scoring")}
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            try:
                v = {c: float(row[c]) for c in COLUMNS}
            except (KeyError, TypeError, ValueError):
                continue
            # Identical row filter to collector_v1.load_retained_residuals.
            residual = v[TOTAL] - (v[FIRST] - v[ENQ])
            if not (math.isfinite(residual) and residual > 0):
                continue
            wait = v[ENQ] - v[STARTED]
            score = v[STARTED] - v[DONE]
            stages["residual_as_used_by_collector"].append(residual)
            stages["prepare_start_to_first_send"].append(v[TOTAL] - v[FIRST])
            stages["evaluation_enqueued_to_ue_receive"].append(v[ENQ])
            stages["prediction_ready_to_evaluation_enqueued_not_in_residual"
                   ].append(v[READY] - v[ENQ])
            stages["gt_wait_evaluation_enqueued_to_started"].append(wait)
            stages["gt_scoring_evaluation_started_to_completed"].append(score)
            stages["evaluation_completed_to_ack_emit"].append(v[DONE] - v[EMIT])
            stages["ack_emit_to_socket_send"].append(v[EMIT] - v[SEND])
            stages["socket_send_to_ue_receive_downlink"].append(v[SEND])
            stages["residual_minus_gt_wait_and_scoring"].append(
                residual - wait - score)
            pooled["gt_wait"].append(wait)
            pooled["gt_scoring"].append(score)
            pooled["gt_total"].append(wait + score)
            pooled["residual"].append(residual)
            pooled["residual_gt_free"].append(residual - wait - score)
        cells[cell] = {k: _summary(x) for k, x in stages.items()}

    gt_total = pooled["gt_total"]
    share = sum(gt_total) / sum(pooled["residual"])
    over = lambda xs, t: sum(1 for x in xs if x > t)
    gt_free = max(gt_total) <= GT_FREE_MAX_MS
    collector_src = (root / COLLECTOR).read_text(encoding="utf-8")
    code_evidence = {
        "collector_residual_expression_present": (
            "residual = total - (first - enqueued)" in collector_src),
        "collector_reads_evaluation_enqueued_column": (
            '"evaluation_enqueued_to_ue_receive_ms"' in collector_src),
        "collector_composition_expression_present": (
            "composed_ns = (context.retained_residual_ns + send_span_ns"
            in collector_src),
    }
    return {
        "schema": "scenesense.run4b.latency_boundary_audit.v1",
        "question": ("Does the existing Run-4 modeled successful latency "
                     "represent action-open -> UE receipt of an immediate "
                     "TAIL_OUTPUT_READY ACK, excluding GT evaluation and map "
                     "installation?"),
        "verdict": ("GT_FREE_OPERATIONAL_PATH_ESTABLISHED" if gt_free
                    else "STOP_MODELED_LATENCY_INCLUDES_GT_EVALUATOR_DELAY"),
        "training_permitted": gt_free,
        "composition": {
            "collector": COLLECTOR,
            "successful_total": ("retained_residual + send_span + "
                                 "modeled_transport + actor_reserve"),
            "retained_residual": ("[model_prepare_start -> first_feature_send]"
                                  " + [evaluation_enqueued -> ue_receive]"),
            "evaluation_enqueued_to_ue_receive_contains": [
                "GT wait (evaluation_enqueued -> evaluation_started)",
                "GT scoring (evaluation_started -> evaluation_completed)",
                "evaluation_completed -> quality-ACK emit",
                "quality-ACK emit -> edge socket send",
                "edge socket send -> UE receive (quality-ACK downlink)"],
            "modeled_transport_boundary": (
                "LAST_UDP_SOCKET_HANDOFF__TO__COMPLETE_RECEIVER_REASSEMBLY "
                f"({SEAM_CONTRACT})"),
            "secondary_observation_unquantified": (
                "The residual removes first_feature_send -> "
                "evaluation_enqueued, but the replacement (send_span + "
                "transport) ends at complete edge reassembly. Edge "
                "pre-model, model tail and prediction-ready -> enqueue are "
                "therefore not represented by any composed term. The "
                "retained probe has no populated reassembly timestamp, so "
                "this gap is not quantified here."),
            "ack_semantics": (
                "Every downlink stage in the residual is the post-GT quality "
                "ACK, not an immediate TAIL_OUTPUT_READY ACK. No "
                "TAIL_OUTPUT_READY ACK exists in the repository source or "
                "retained evidence."),
        },
        "code_evidence": code_evidence,
        "per_cell": cells,
        "pooled": {
            "rows": len(gt_total),
            "gt_wait": _summary(pooled["gt_wait"]),
            "gt_scoring": _summary(pooled["gt_scoring"]),
            "gt_wait_plus_scoring": _summary(gt_total),
            "residual_as_used": _summary(pooled["residual"]),
            "residual_minus_gt_wait_and_scoring": _summary(
                pooled["residual_gt_free"]),
            "gt_share_of_residual_sum": share,
            "rows_with_gt_over_10ms": over(gt_total, 10.0),
            "rows_with_gt_over_50ms": over(gt_total, 50.0),
            "rows_with_gt_over_170ms": over(gt_total, 170.0),
            "rows_with_residual_over_170ms": over(pooled["residual"], 170.0),
            "rows_with_gt_free_residual_over_170ms": over(
                pooled["residual_gt_free"], 170.0),
        },
        "source_sha256": {**sources,
                          COLLECTOR: _sha_file(root / COLLECTOR),
                          SEAM_CONTRACT: _sha_file(root / SEAM_CONTRACT)},
        "not_done": ("No alternative latency model was constructed; removing "
                     "the GT stages or adding an edge-compute term would be "
                     "a new model and requires an explicit Abiodun decision."),
    }


def main() -> int:
    report = audit()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(report["verdict"])
    return 0 if report["training_permitted"] else 2


if __name__ == "__main__":
    sys.exit(main())
