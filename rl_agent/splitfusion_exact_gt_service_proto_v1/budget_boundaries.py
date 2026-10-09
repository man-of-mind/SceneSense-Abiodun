"""Corrected per-frame budget arithmetic for the prototype (NOT a measurement or forecast).

Two prediction boundaries, per recorded live100 decision (n=48; this capture
already contains the corrected perception-input function):
* ``evaluator_submit``  - current edge mark, after map update build and map publication;
* ``compute_end``       - end of the edge ``compute`` call, the earliest instant at
                          which records and the native label map exist on CPU. Using it
                          requires an edge change that hands the prediction to evaluation
                          before (in parallel with) map build/publication. That ordering
                          change is NOT implemented or measured here.
Post-prediction path = A (unmeasured prediction admission: ticket, digest, sealed
reference, transfer, copy, ``admit_prediction``) + recorded scorer join/score/quality
+ recorded result return/emission + recorded downlink. GT readiness uses the review's
scenarios (recorded per-frame prepare shifted earlier). Map time is never subtracted.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "experiments/splitfusion_numeric_gt_runtime_v1/critical_path_decision_01/BOUNDARY_TABLE.csv"
TIMINGS = ROOT / "experiments/splitfusion_numeric_gt_runtime_v1/completion_live100_timing_01/FRAME_TIMINGS.csv"
REVIEW = ROOT / "experiments/splitfusion_exact_gt_architecture_review_v1/SCENARIO_FRAMES.csv"
OUTPUT = ROOT / "experiments/splitfusion_exact_gt_service_proto_v1"
ADMISSION_MS = (1.0, 3.0, 5.0)
GT = ("freeze_current__handoff10__pred3", "freeze_current__handoff20__pred3", "freeze_early__handoff10__pred3")


def nearest(v, p):
    v = sorted(v)
    return v[max(0, math.ceil(p * len(v)) - 1)]


def summary(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return dict(n=n, median=round(med, 3), p95=round(nearest(v, .95), 3), min=round(v[0], 3), max=round(v[-1], 3))


def rows(path, key="frame_id"):
    with open(path, newline="") as stream:
        return {int(r[key]): r for r in csv.DictReader(stream)}


def run(name):
    out = OUTPUT / name
    out.mkdir(parents=False, exist_ok=False)
    b, t, r = rows(BOUNDARY), rows(TIMINGS), rows(REVIEW)
    assert set(b) == set(t) == set(r) and len(b) == 48
    frames, results = [], {}
    for f in sorted(b):
        post = float(t[f]["scorer_prediction_to_score_start_ms"]) + float(t[f]["score_start_to_score_end_ms"]) \
            + float(t[f]["score_end_to_emitted_ms"]) + float(t[f]["emitted_to_ue_receipt_ms"])
        row = dict(frame_id=f, compute_end=float(b[f]["compute_end"]), predictions_cpu=float(b[f]["predictions_cpu"]),
                   map_publish_end=float(b[f]["map_publish_end"]), evaluator_submit=float(b[f]["evaluator_submit"]),
                   recorded_post_without_admission=post,
                   world_tick_end=float(r[f]["world_tick_end"]), sensor_wait_end=float(r[f]["sensor_wait_end"]),
                   source_owned=float(r[f]["source_owned"]))
        for g in GT:
            row["gt_ready__" + g] = float(r[f][g + "__gt_ready"])
        frames.append(row)
    for boundary in ("evaluator_submit", "compute_end"):
        for a in ADMISSION_MS:
            for g in (None, *GT):
                receipts, late_gt = [], 0
                for row in frames:
                    ready = row[boundary] + a
                    if g is not None:
                        late_gt += row["gt_ready__" + g] > ready
                        ready = max(ready, row["gt_ready__" + g])
                    receipts.append(ready + row["recorded_post_without_admission"])
                key = f"{boundary}__admission{int(a)}__{'gt_never_late' if g is None else g}"
                results[key] = dict(receipt_ms=summary(receipts), timely=sum(x <= 170 for x in receipts),
                                    gt_later_than_admitted_prediction=None if g is None else late_gt)
    boundaries = {k: summary([row[k] for row in frames]) for k in
                  ("compute_end", "predictions_cpu", "map_publish_end", "evaluator_submit", "recorded_post_without_admission",
                   "world_tick_end", "sensor_wait_end", "source_owned")}
    boundaries["compute_end_to_evaluator_submit"] = summary([row["evaluator_submit"] - row["compute_end"] for row in frames])
    result = dict(schema="splitfusion.exact_gt_service_proto.budget.v1", arithmetic_not_measurement=True,
        input_contract="recorded live100 capture with the corrected perception-input function (432-row content, "
                       "PIL resampling, normalization, 16-row padding); legacy input timing is NOT assumed",
        boundaries_ms_from_action_open=boundaries, scenarios=results,
        unmeasured=["prediction admission A", "edge reordering of evaluation before map build/publish",
                    "freeze copy + per-buffer SHA-256 on the UE before action-open", "prepare duration in new placement"],
        capture_to_action_open_note="world tick end and sensor-ready precede action-open; any freeze work added "
                                    "before action-open lengthens capture-to-action-open latency and is not hidden by the 170-ms clock")
    raw = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
    (out / "BUDGET.json").write_bytes(raw)
    cols = list(frames[0])
    lines = [",".join(cols)] + [",".join(str(round(row[c], 4)) for c in cols) for row in frames]
    (out / "BUDGET_FRAMES.csv").write_text("\n".join(lines) + "\n")
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (BOUNDARY, TIMINGS, REVIEW)}
    (out / "INPUT_SHA256.json").write_text(json.dumps(inputs, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    print(json.dumps(run(parser.parse_args().name)["boundaries_ms_from_action_open"], indent=1))
