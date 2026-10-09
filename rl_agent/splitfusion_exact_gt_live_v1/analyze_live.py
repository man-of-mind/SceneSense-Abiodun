"""Post-run evaluation of the single early-GT live attempt against the registered acceptance criteria.

Read-only on the attempt; writes create-only RESULT/FRAMES/OUTPUT_SHA256 into an evidence dir.
All latencies are measured live instants (CLOCK_MONOTONIC_RAW); nothing is reconstructed here.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_exact_gt_live_v1"
COLD = 2


def stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    n = len(v)
    return dict(n=n, p50=round(v[(n - 1) // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2, 3),
                p95=round(v[max(0, math.ceil(.95 * n) - 1)], 3), max=round(v[-1], 3), min=round(v[0], 3))


def load_profile(path):
    spans, marks = collections.defaultdict(dict), collections.defaultdict(dict)
    if not path.exists():
        return spans, marks
    for line in open(path):
        d = json.loads(line)
        f = d.get("frame_id", d.get("carla_frame"))
        if f is None or d.get("ph") not in (None, "live"):
            continue
        if d.get("r") == "span":
            spans[int(f)].setdefault(d["stage"], (d["start_raw_ns"], d["end_raw_ns"]))
        elif d.get("r") == "mark":
            marks[int(f)].setdefault(d["stage"], d["t_raw_ns"])
    return spans, marks


def rescore(directory):
    """Exact offline re-score from durable archived inputs (pinned admission + scorer + formula)."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
    from rl_agent.splitfusion_exact_gt_service_proto_v1.scorer_core import expected_from_identity
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
    spec = W.load_run4_quality_spec(ROOT)
    metadata_bytes = (directory / "target_metadata.json").read_bytes()
    score = json.loads((directory / "numerical_score.json").read_bytes())
    metadata = C._parse(metadata_bytes)
    identity = C.canonical({k: metadata[k] for k in ("capture_identity", "action_identity", "plan_sha256", "run_binding",
                                                    "producer_binding", "frame_context", "prediction_contract")})
    _, expected = expected_from_identity(identity)
    target = R._admit(metadata, (directory / "segmentation.u8").read_bytes(), (directory / "object_ignore.u8").read_bytes(),
                      **expected)
    labels = np.frombuffer((directory / "predicted_labels.u8").read_bytes(), np.uint8).reshape(720, 1280)
    scorer = Q.NumericalScorer()
    prediction = Q.admit_prediction(scorer, identity=score["prediction_identity"], labels=labels,
                                    records=score["prediction_records"], prediction_contract=metadata["prediction_contract"])
    result = scorer.score(Q.join_admitted_inputs(scorer, target=target, prediction=prediction))
    value = quality.evaluate_exact_quality(spec, result.measurement)
    q = None if value is None else float(value.q_perc)
    return dict(q_perc=q, live_q_perc=score["q_perc"], exact=q == score["q_perc"]
                and C.canonical(result.measurement) == C.canonical(score["measurement"])
                and target.semantic_sha256 == score["ready_semantic_sha256"])


def run(attempt, name):
    attempt = Path(attempt).resolve(strict=True)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    cell = next((attempt / "capture/cells").iterdir())
    qualification = json.loads((attempt / "capture/QUALIFICATION_RESULT.json").read_bytes())
    service_dir = cell / "numeric_gt/service"
    frames = [json.loads(l) for l in open(service_dir / "FRAMES.jsonl")]
    service = json.loads((service_dir / "SERVICE_RESULT.json").read_bytes())
    samples = json.loads((service_dir / "QUEUE_SAMPLES.json").read_bytes())
    memory = json.loads((service_dir / "MEMORY_SAMPLES.json").read_bytes())
    ue_result = json.loads((cell / "numeric_gt/EXACT_GT_UE_RESULT.json").read_bytes())
    edge_doc = json.loads((cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json").read_bytes())
    evidence = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_bytes())
    ue_spans, ue_marks = load_profile(cell / "profiler/ue.profile.jsonl")
    edge_spans, edge_marks = load_profile(cell / "profiler/edge/edge.profile.jsonl")
    decisions = {d["frame_id"]: d for d in evidence["decisions"] if d.get("action_open")}
    by_open = {int(r["action_open_timestamp_ns"]): r for r in evidence.get("resolutions", [])}
    resolution = {f["frame_id"]: by_open.get(f["action_open_raw_ns"]) for f in frames if f.get("decision")}
    feedback = collections.defaultdict(list)
    for row in evidence["feedback_rows"]:
        feedback[int(row["frame_id"])].append(row)
    state = edge_doc.get("state") or {}
    edge_frames = state.get("frames", {})
    edge_records = state.get("records", [])
    rows = []
    decision_frames = [f for f in frames if f.get("decision")]
    for k, f in enumerate(decision_frames):
        fid = f["frame_id"]
        ev = {}
        for e in f["events"]:
            ev.setdefault(e["role"] + ":" + e["name"], e["raw_ns"])
        ao = f["action_open_raw_ns"]
        ms = lambda key: None if key not in ev else (ev[key] - ao) / 1e6
        receipts = [r for r in feedback.get(fid, []) if r.get("receipt")]
        receipt = min((r["receipt"]["ns"] for r in receipts), default=None)
        feedback_class = sorted({r.get("class") for r in receipts}) or None
        d = decisions.get(fid, {})
        stages = d.get("stages", {})
        us, es = ue_spans.get(fid, {}), edge_spans.get(fid, {})
        row = dict(frame_id=fid, cold=k < COLD, outcome=f.get("outcome"),
            capture_to_action_open_ms=None if "carla.world_tick" not in us else (ao - us["carla.world_tick"][1]) / 1e6,
            sensor_ready_to_action_open_ms=None if "ue.sensor_wait" not in us else (ao - us["ue.sensor_wait"][1]) / 1e6,
            ue_input_ms=None if "ue.input_7ch" not in us else (us["ue.input_7ch"][1] - us["ue.input_7ch"][0]) / 1e6,
            ue_encode_ms=None if "ue.codec.total_ue_preparation" not in us else
                (us["ue.codec.total_ue_preparation"][1] - us["ue.codec.total_ue_preparation"][0]) / 1e6,
            last_send_ms=ms("ue:last_datagram_sent"),
            edge_compute_ms=None if "edge.compute" not in es else (es["edge.compute"][1] - es["edge.compute"][0]) / 1e6,
            edge_compute_end_ms=None if "edge.compute" not in es else (es["edge.compute"][1] - ao) / 1e6,
            freeze_enqueue_ms=ms("ue:freeze_enqueued_on_ue_thread"), freeze_sent_ms=ms("ue_freeze_thread:freeze_sealed"),
            source_decoded_ms=ms("prep:source_decoded"), unbound_ready_ms=ms("prep:unbound_ready"), bound_ms=ms("prep:bound"),
            committed_ms=ms("prep:committed"), gt_ready_ms=ms("score:target_admitted"),
            prediction_materialized_ms=ms("edge:prediction_materialized"), prediction_received_ms=ms("service:prediction_received"),
            prediction_admitted_ms=ms("score:prediction_admitted"), result_sent_ms=ms("service:result_sent"),
            feedback_emitted_ms=ms("edge:feedback_emitted"),
            ue_receipt_ms=None if receipt is None else (receipt - ao) / 1e6, ue_terminal=f.get("ue_terminal"),
            prediction_before_gt=f.get("prediction_before_gt"), joined=f.get("joined_on_target", False),
            q_perc=(f.get("result") or {}).get("q_perc"), feedback_class=feedback_class,
            controller_resolution=(resolution.get(fid) or {}).get("terminal"),
            controller_resolution_ms=None if resolution.get(fid) is None else
                (resolution[fid]["resolution_timestamp_ns"] - resolution[fid]["action_open_timestamp_ns"]) / 1e6,
            controller_action_open_matches=resolution.get(fid) is not None,
            controller_q_perc=(resolution.get(fid) or {}).get("q_perc"), committed=bool(f.get("committed")), commit=f.get("commit"))
        row["timely"] = row["ue_receipt_ms"] is not None and row["ue_receipt_ms"] <= 170.0
        rows.append(row)
    scored = [r for r in rows if r["outcome"] == "SCORED"]
    rescored = {}
    for r in scored:
        rescored[r["frame_id"]] = rescore(service_dir / "archive" / f"frame_{r['frame_id']}" / "scorer")
    commits = [r["commit"] for r in rows if r["committed"]]
    chain = all(a["history_after_sha256"] == b["history_before_sha256"] for a, b in zip(commits, commits[1:]))
    edge_emissions = collections.Counter(int(rec["frame_id"]) for rec in edge_records)
    decision_ids = {r["frame_id"] for r in rows}
    reached = {r["frame_id"] for r in rows if r["prediction_received_ms"] is not None}
    limits = service["limits"]
    gt_first = [r for r in scored if r["gt_ready_ms"] is not None and r["prediction_admitted_ms"] is not None
                and r["gt_ready_ms"] <= r["prediction_admitted_ms"]]
    restoration = json.loads((attempt / "RESTORATION_LATEST.json").read_bytes()) if (attempt / "RESTORATION_LATEST.json").exists() else {}
    affinity = json.loads((attempt / "AFFINITY_APPLIED.json").read_bytes()) if (attempt / "AFFINITY_APPLIED.json").exists() else {}
    gates = qualification.get("gates", qualification)
    criteria = dict(
        A1_capture_valid=qualification.get("verdict") == "CAPTURE_VALID",
        A2_terminal_accounting=all(r["outcome"] is not None for r in rows)
            and all(edge_emissions.get(fid, 0) == (1 if fid in reached else edge_emissions.get(fid, 0)) for fid in decision_ids)
            and set(edge_emissions) <= decision_ids and all(v == 1 for v in edge_emissions.values())
            and all(rec["q_perc"] is None for rec in edge_records if rec["kind"] != "DELIVERED_SUCCESS"),
        A3_exact_quality_offline=all(v["exact"] for v in rescored.values()) and len(rescored) == len(scored),
        A4_history=chain,
        A5_no_pre_action_gt_work=all(r["freeze_enqueue_ms"] is not None and r["freeze_enqueue_ms"] >= 0 for r in rows
                                     if r["outcome"] != "NO_GT_UE_FREEZE_QUEUE_FULL"),
        A6_bounded_refusals=len(service["refusals"]) <= 5 and all(
            s["prep_retained"] <= limits["prep_jobs"] and s["scorer_slots"] <= limits["scorer_jobs"]
            and s["freeze_queue"] <= limits["freeze_queue"] for s in samples)
            and (state.get("evaluator_counters") or {}).get("queue_overflow", 0) == 0,
        A7_gt_alongside_prediction=len(scored) > 0 and len(gt_first) >= 0.8 * len(scored),
        A8_cleanup=not service["errors"] and not ue_result["errors"] and ue_result["service_returncode"] == 0
            and restoration.get("passed") is True and edge_doc.get("error") is None,
        A9_retained_settings=affinity.get("passed") is True)
    perf_keys = ["ue_receipt_ms", "capture_to_action_open_ms", "sensor_ready_to_action_open_ms", "ue_input_ms", "ue_encode_ms",
                 "last_send_ms", "edge_compute_ms", "edge_compute_end_ms", "freeze_enqueue_ms", "freeze_sent_ms",
                 "source_decoded_ms", "unbound_ready_ms", "bound_ms", "committed_ms", "gt_ready_ms",
                 "prediction_materialized_ms", "prediction_received_ms", "prediction_admitted_ms", "result_sent_ms",
                 "feedback_emitted_ms"]
    performance = {}
    for label, subset in (("all", rows), ("warm", [r for r in rows if not r["cold"]]), ("cold", [r for r in rows if r["cold"]])):
        performance[label] = {k: stats([r[k] for r in subset]) for k in perf_keys}
        performance[label]["timely_le_170"] = sum(r["timely"] for r in subset)
        performance[label]["missing_ue_receipt"] = sum(r["ue_receipt_ms"] is None for r in subset)
        performance[label]["decisions"] = len(subset)
    result = dict(schema="splitfusion.exact_gt_live.live_result.v1", attempt=str(attempt),
        decision="GT_ARCHITECTURE_WORKING" if all(criteria.values()) else "INTEGRATION_OR_ARCHITECTURE_BLOCKED",
        criteria=criteria, qualification=dict(verdict=qualification.get("verdict"), gates=gates),
        outcomes=service["outcomes"], refusals=service["refusals"], decisions=len(rows), scored=len(scored),
        gt_first=len(gt_first), rescored_exact=sum(v["exact"] for v in rescored.values()), history_chain=chain,
        edge_counters=state.get("evaluator_counters"), queue_max={k: max((s[k] or 0) for s in samples) for k in
            ("prep_queue", "score_queue", "freeze_queue", "prep_retained", "scorer_slots", "waiting", "archive_outstanding")},
        memory_kib_max={k: max(m[k] for m in memory if m.get(k) is not None) for k in (memory[0] if memory else {}) if k not in ("frame_id", "t")},
        performance=performance, restoration=restoration.get("passed"), affinity=affinity.get("passed"),
        deadline_note="Exceeding 170 ms alone does not mean the GT architecture failed.")
    with open(out / "RESULT.json", "x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, default=str)
    with open(out / "FRAMES.jsonl", "x") as stream:
        for r in rows:
            stream.write(json.dumps(dict(r, rescore=rescored.get(r["frame_id"])), sort_keys=True, default=str) + "\n")
    inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(inventory, indent=1, sort_keys=True))
    print(json.dumps({k: result[k] for k in ("decision", "criteria", "outcomes", "decisions", "scored", "gt_first",
                                             "rescored_exact")}, indent=1, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.attempt, args.name)
