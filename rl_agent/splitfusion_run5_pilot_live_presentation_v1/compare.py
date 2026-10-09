"""Descriptive stage comparison of two pilot 1,000-frame sessions (per-decision rows; P50/P95/P99; halves).
Usage: python -m rl_agent.splitfusion_run5_pilot_live_presentation_v1.compare <baseline.jsonl> <run.jsonl> <out.json>
"""
import collections
import json
import sys

import numpy as np

STAGES = {
    "ue_input_preparation": lambda r: r["ue_input_end"] - r["ue_input_start"],
    "ue_encode": lambda r: r["ue_encode_end"] - r["ue_encode_start"],
    "prediction_ready_edge": lambda r: r["edge_compute_end"],
    "prediction_at_service": lambda r: r["service_prediction_received"],
    "gt_published": lambda r: r["gt_published"],
    "gt_publication_to_admission_start": lambda r: r["target_worker_begin"] - r["gt_published"],
    "target_admission_only_request": lambda r: (r["target_worker_end"] - r["target_worker_begin"]
                                                if r["scoring_request_op"] == "PREDICTION" else None),
    "final_scoring_request": lambda r: (r["prediction_worker_end"] - r["prediction_worker_begin"]
                                        if r["scoring_request_op"] == "PREDICTION" else r["target_worker_end"] - r["target_worker_begin"]),
    "scoring_end": lambda r: r["scoring_worker_end"],
    "ue_receipt": lambda r: r["ue_receipt"],
}


def pct(values):
    v = np.array([x for x in values if x is not None], float)
    return dict(n=int(len(v))) if not len(v) else dict(n=int(len(v)), p50=round(float(np.percentile(v, 50)), 2),
        p95=round(float(np.percentile(v, 95)), 2), p99=round(float(np.percentile(v, 99)), 2))


def summary(rows):
    def safe(f, r):
        try:
            return f(r)
        except TypeError:
            return None
    out = dict(decisions=len(rows), classes=dict(collections.Counter(r["outcome_class"] for r in rows)),
               stages={k: pct([safe(f, r) for r in rows]) for k, f in STAGES.items()})
    half = len(rows) // 2
    out["halves"] = {}
    for name, sub in (("first", rows[:half]), ("second", rows[half:])):
        out["halves"][name] = dict(n=len(sub), timely=sum(r["outcome_class"].startswith("TIMELY") for r in sub),
            classes=dict(collections.Counter(r["outcome_class"] for r in sub)),
            input_prep=pct([r["ue_input_end"] - r["ue_input_start"] for r in sub]),
            receipt=pct([r["ue_receipt"] for r in sub]), gt_published=pct([r["gt_published"] for r in sub]),
            q_e4_p50=float(np.median([r["executed_action"]["q_e4"] for r in sub])),
            payload_p50=float(np.median([r["payload_bytes"] for r in sub])))
    return out


if __name__ == "__main__":
    base, run = ([json.loads(line) for line in open(p)] for p in sys.argv[1:3])
    result = dict(baseline=summary(base), run=summary(run), note="different sessions: descriptive, not causal")
    open(sys.argv[3], "x").write(json.dumps(result, indent=1, sort_keys=True))
    for k in STAGES:
        print(f"{k:36s} base {result['baseline']['stages'][k]} | run {result['run']['stages'][k]}")
