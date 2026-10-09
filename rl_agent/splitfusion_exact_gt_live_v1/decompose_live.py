"""Per-decision decomposition of action-open -> UE feedback for the single early-GT live attempt.

Reads the sealed live_result FRAMES.jsonl plus the attempt's edge profiler marks; writes
DECOMPOSITION.json (create-only) next to the result. Measured instants only, no reconstruction.
"""
import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summary(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    n = len(v)
    p50 = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return dict(n=n, p50=round(p50, 2), p95=round(v[max(0, math.ceil(.95 * n) - 1)], 2), max=round(v[-1], 2))


def run(attempt, result):
    attempt, result = Path(attempt).resolve(strict=True), Path(result).resolve(strict=True)
    cell = next((attempt / "capture/cells").iterdir())
    rows = [json.loads(line) for line in open(result / "FRAMES.jsonl")]
    opens = {json.loads(line)["frame_id"]: json.loads(line)["action_open_raw_ns"]
             for line in open(cell / "numeric_gt/service/FRAMES.jsonl")}
    spans, marks = collections.defaultdict(dict), collections.defaultdict(dict)
    for line in open(cell / "profiler/edge/edge.profile.jsonl"):
        d = json.loads(line)
        if d.get("ph") != "live" or d.get("frame_id") is None:
            continue
        if d["r"] == "span":
            spans[d["frame_id"]].setdefault(d["stage"], (d["start_raw_ns"], d["end_raw_ns"]))
        elif d["r"] == "mark":
            marks[d["frame_id"]].setdefault(d["stage"], d["t_raw_ns"])
    instants, segments = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        fid, ao = r["frame_id"], opens[r["frame_id"]]
        at = lambda ns: (ns - ao) / 1e6
        m, s = marks[fid], spans[fid]
        instants["ue_last_datagram_sent"].append(r["last_send_ms"])
        instants["edge_first_datagram"].append(at(m["edge.datagram_first"]) if "edge.datagram_first" in m else None)
        instants["edge_reassembly_complete"].append(at(m["edge.reassembly_complete"]) if "edge.reassembly_complete" in m else None)
        instants["edge_compute_start"].append(at(s["edge.compute"][0]) if "edge.compute" in s else None)
        instants["prediction_ready_edge_compute_end"].append(r["edge_compute_end_ms"])
        instants["gt_ready_target_admitted"].append(r["gt_ready_ms"])
        instants["prediction_admitted"].append(r["prediction_admitted_ms"])
        instants["feedback_emitted"].append(r["feedback_emitted_ms"])
        instants["ue_receipt"].append(r["ue_receipt_ms"])
        segments["uplink_last_send_to_reassembly"].append(
            at(m["edge.reassembly_complete"]) - r["last_send_ms"] if "edge.reassembly_complete" in m else None)
        segments["edge_compute"].append(r["edge_compute_ms"])
        segments["materialized_to_service_receive"].append(r["prediction_received_ms"] - r["prediction_materialized_ms"])
        segments["service_receive_to_prediction_admitted"].append(r["prediction_admitted_ms"] - r["prediction_received_ms"])
        segments["gt_wait_after_prediction_admitted"].append(max(0.0, r["gt_ready_ms"] - r["prediction_admitted_ms"]))
        segments["join_score_to_result_sent"].append(r["result_sent_ms"] - max(r["gt_ready_ms"], r["prediction_admitted_ms"]))
        segments["result_sent_to_edge_emit"].append(r["feedback_emitted_ms"] - r["result_sent_ms"])
        segments["edge_emit_to_ue_receipt_downlink"].append(r["ue_receipt_ms"] - r["feedback_emitted_ms"])
        segments["prediction_ready_to_ue_receipt"].append(r["ue_receipt_ms"] - r["prediction_materialized_ms"])
    counts = dict(decisions=len(rows),
        prediction_ready_le_170=sum(r["edge_compute_end_ms"] <= 170 for r in rows),
        gt_ready_le_170=sum(r["gt_ready_ms"] <= 170 for r in rows),
        feedback_emitted_le_170=sum(r["feedback_emitted_ms"] <= 170 for r in rows),
        ue_receipt_le_170=sum(r["ue_receipt_ms"] is not None and r["ue_receipt_ms"] <= 170 for r in rows),
        ue_receipt_missing=sum(r["ue_receipt_ms"] is None for r in rows),
        gt_admitted_no_later_than_prediction=sum(r["gt_ready_ms"] <= r["prediction_admitted_ms"] for r in rows),
        controller_resolution=dict(collections.Counter(r["controller_resolution"] for r in rows)))
    doc = dict(schema="splitfusion.exact_gt_live.decomposition.v1", unit="ms after retained action-open",
               instants={k: summary(v) for k, v in instants.items()},
               segments={k: summary(v) for k, v in segments.items()}, counts=counts,
               ue_receipts_sorted=sorted(round(r["ue_receipt_ms"], 1) for r in rows if r["ue_receipt_ms"] is not None))
    with open(result / "DECOMPOSITION.json", "x") as stream:
        json.dump(doc, stream, indent=1, sort_keys=True)
    print(json.dumps(dict(segments=doc["segments"], counts=counts), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run(args.attempt, args.result)
