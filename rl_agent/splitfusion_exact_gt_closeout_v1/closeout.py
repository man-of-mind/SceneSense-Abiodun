"""Close-out of the single early-GT live attempt from existing evidence only.

Read-only on the sealed attempt and on live_result_01. Writes create-only outputs:
BOUNDARIES.jsonl (per decision, ms after the retained action-open), CLOSEOUT.json (tables).
No modelled latency is produced: every value is a measured instant difference or a quoted
recorded statistic. Intervals are reported per frame; combined statistics are never formed
by adding component medians.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLD = 2
DEADLINE = 170.0


def q(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return dict(n=n, min=round(v[0], 3), p50=round(med, 3), p95=round(v[max(0, math.ceil(.95 * n) - 1)], 3),
                max=round(v[-1], 3))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    for r, x in ((ra, a), (rb, b)):            # average ranks for ties
        for value in np.unique(x):
            idx = np.where(x == value)[0]
            r[idx] = r[idx].mean()
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def profile(path):
    spans, marks = collections.defaultdict(dict), collections.defaultdict(dict)
    for line in open(path):
        d = json.loads(line)
        fid = d.get("frame_id", d.get("carla_frame"))
        if fid is None or d.get("ph") != "live":
            continue
        if d["r"] == "span":
            spans[int(fid)].setdefault(d["stage"], (d["start_raw_ns"], d["end_raw_ns"]))
        elif d["r"] == "mark":
            marks[int(fid)].setdefault(d["stage"], d)
    return spans, marks


def run(attempt, prior, name):
    attempt, prior = Path(attempt).resolve(strict=True), Path(prior).resolve(strict=True)
    out = ROOT / "experiments/splitfusion_exact_gt_live_v1" / name
    out.mkdir(parents=True, exist_ok=False)
    cell = next((attempt / "capture/cells").iterdir())
    frames = [json.loads(l) for l in open(cell / "numeric_gt/service/FRAMES.jsonl")]
    frames = [f for f in frames if f.get("decision")]
    prior_rows = {json.loads(l)["frame_id"]: json.loads(l) for l in open(prior / "FRAMES.jsonl")}
    edge_state = json.loads((cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json").read_bytes())["state"]
    phase6 = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_bytes())
    actions = {x["frame_id"]: x["action"] for x in phase6["frames"]}
    receipts = collections.defaultdict(list)
    for row in phase6["feedback_rows"]:
        receipts[int(row["frame_id"])].append(row)
    by_open = {int(r["action_open_timestamp_ns"]): r for r in phase6["resolutions"]}
    ue_spans, _ = profile(cell / "profiler/ue.profile.jsonl")
    edge_spans, edge_marks = profile(cell / "profiler/edge/edge.profile.jsonl")

    # global score-worker request timeline (worker-side instants) for busy-at-receipt attribution
    score_requests = []
    for f in frames:
        for r in f.get("requests", []):
            if r["worker"] == "score":
                score_requests.append((r["timing"]["begin_raw_ns"], r["timing"]["end_raw_ns"], r["op"], f["frame_id"]))
    score_requests.sort()

    rows = []
    for k, f in enumerate(frames):
        fid, ao = f["frame_id"], f["action_open_raw_ns"]
        ms = lambda ns: None if ns is None else (ns - ao) / 1e6
        ev = {}
        for e in f["events"]:
            ev.setdefault(e["role"] + ":" + e["name"], e["raw_ns"])
        req = {(r["worker"], r["op"]): r for r in f["requests"]}
        pred, target = req[("score", "PREDICTION")], req[("score", "TARGET")]
        producer = json.loads((cell / f"numeric_gt/service/archive/frame_{fid}/producer/producer_events.json").read_bytes())
        pev = {e["name"]: e["raw_ns"] for e in producer["events"]}
        us, es, em = ue_spans[fid], edge_spans[fid], edge_marks[fid]
        edge_frame = edge_state["frames"][str(fid)]
        receipt = min((r["receipt"]["ns"] for r in receipts[fid] if r.get("receipt")), default=None)
        resolution = by_open.get(ao)
        gt_first = not f.get("joined_on_target")          # scored inside the PREDICTION request
        scoring_request = pred if gt_first else target
        busy = [s for s in score_requests if s[0] <= ev["service:prediction_received"] < s[1] and s[3] != fid]
        busy_own = [s for s in score_requests if s[0] <= ev["service:prediction_received"] < s[1] and s[3] == fid]
        a = actions[fid]
        row = dict(frame_id=fid, cold=k < COLD, gt_first=gt_first, outcome=f["outcome"],
            action=dict(family=a["family"], quantizer=a["quantizer"], mode_id=a["mode_id"], q_e4=a["q_e4"],
                        keep_count=a["keep_count"]),
            datagrams=em["edge.reassembly_complete"]["chunks"],
            controller_resolution=None if resolution is None else resolution["terminal"],
            feedback_class=sorted({r["class"] for r in receipts[fid]}),
            q_perc=(f.get("result") or {}).get("q_perc"),
            # ---- before action-open (negative) and UE work after it
            capture_world_tick_end=ms(us["carla.world_tick"][1]), sensor_ready=ms(us["ue.sensor_wait"][1]),
            actor_end=ms(us["agent.actor"][1]),
            ue_input_start=ms(us["ue.input_7ch"][0]), ue_input_end=ms(us["ue.input_7ch"][1]),
            ue_encode_start=ms(us["ue.codec.total_ue_preparation"][0]),
            ue_encode_end=ms(us["ue.codec.total_ue_preparation"][1]),
            ue_first_send=ms(us["ue.udp_send"][0]), ue_last_send=ms(ev["ue:last_datagram_sent"]),
            # ---- edge prediction path
            edge_first_datagram=ms(em["edge.datagram_first"]["t_raw_ns"]),
            edge_reassembly_complete=ms(em["edge.reassembly_complete"]["t_raw_ns"]),
            edge_compute_start=ms(es["edge.compute"][0]), edge_compute_end=ms(es["edge.compute"][1]),
            prediction_materialized=ms(ev["edge:prediction_materialized"]),
            prediction_dequeued=ms(ev["edge:prediction_dequeued"]),
            prediction_message_built=ms(ev["edge:prediction_sent"]),
            service_prediction_received=ms(ev["service:prediction_received"]),
            prediction_worker_begin=ms(pred["timing"]["begin_raw_ns"]),
            prediction_worker_end=ms(pred["timing"]["end_raw_ns"]),
            prediction_reply_at_service=ms(pred["end"]),
            prediction_worker_thread_cpu_ms=pred["timing"]["thread_cpu_ns"] / 1e6,
            # ---- GT path
            gt_numeric_ready_producer=ms(pev.get("numeric_ready")), gt_published=ms(ev["prep:published"]),
            target_worker_begin=ms(target["timing"]["begin_raw_ns"]),
            target_worker_end=ms(target["timing"]["end_raw_ns"]),
            target_reply_at_service=ms(target["end"]),
            target_worker_thread_cpu_ms=target["timing"]["thread_cpu_ns"] / 1e6,
            # ---- result path
            scoring_request_op=pred["op"] if gt_first else target["op"],
            scoring_worker_end=ms(scoring_request["timing"]["end_raw_ns"]),
            scoring_reply_at_service=ms(scoring_request["end"]),
            result_sent=ms(ev["service:result_sent"]), edge_result_received=ms(edge_frame["result_received"]),
            feedback_emitted=ms(edge_frame["emitted"]), ue_receipt=ms(receipt),
            score_worker_busy_at_receipt=[dict(op=s[2], frame_id=s[3]) for s in busy],
            own_target_in_flight_at_receipt=bool(busy_own))
        assert abs(row["edge_compute_end"] - prior_rows[fid]["edge_compute_end_ms"]) < 1e-6
        rows.append(row)

    # ---- non-overlapping post-prediction chain, telescoping exactly to materialized -> UE receipt
    chain_gt_first = [
        ("edge_queue_wait", "prediction_materialized", "prediction_dequeued",
         "compute thread hands the ticket to the sender thread (bounded queue)"),
        ("edge_digest_copy_seal", "prediction_dequeued", "prediction_message_built",
         "labels tobytes copy, concat copy + SHA-256 digest, records copy, sealed memfd copy"),
        ("edge_to_service_ipc", "prediction_message_built", "service_prediction_received",
         "JSON encode + sendmsg + container-to-host socket + service loop wake/busy + recvmsg + parse/canonical check"),
        ("service_to_worker_start", "service_prediction_received", "prediction_worker_begin",
         "service housekeeping in the same loop pass + score-worker queue wait + request send + worker wake"),
        ("worker_prediction_request", "prediction_worker_begin", "prediction_worker_end",
         "sealed-read copy + prediction validation (incl. second digest) + admission + join + score + Q_perc"),
        ("worker_reply_to_service", "prediction_worker_end", "prediction_reply_at_service",
         "reply send + service wake"),
        ("service_deliver", "scoring_reply_at_service", "result_sent", "service bookkeeping before RESULT send"),
        ("service_to_edge_ipc", "result_sent", "edge_result_received",
         "JSON encode + host-to-container socket + edge receiver thread wake/parse"),
        ("edge_feedback_emit", "edge_result_received", "feedback_emitted",
         "unchanged quality_feedback recomputation + feedback encode + UDP send (stamp after send returns)"),
        ("downlink_and_ue_receive", "feedback_emitted", "ue_receipt", "GTP/RFsim downlink + UE receive thread"),
    ]
    chain_pred_first = chain_gt_first[:6] + [
        ("gt_wait_after_prediction_stored", "prediction_reply_at_service", "target_worker_begin",
         "waiting for GT publication + score-queue dispatch (prediction already stored)"),
        ("worker_target_request", "target_worker_begin", "target_worker_end",
         "sealed-read copy + target admission/validation + join + score + Q_perc"),
        ("worker_reply_to_service", "target_worker_end", "target_reply_at_service", "reply send + service wake"),
    ] + chain_gt_first[6:]
    chains = {}
    for label, chain, subset in (("gt_first", chain_gt_first, [r for r in rows if r["gt_first"]]),
                                 ("prediction_first", chain_pred_first, [r for r in rows if not r["gt_first"]])):
        table, negatives = [], 0
        for r in subset:
            total = r["ue_receipt"] - r["prediction_materialized"]
            parts = [r[b] - r[a] for _, a, b, _ in chain]
            negatives += sum(p < 0 for p in parts)
            assert abs(sum(parts) - total) < 1e-6
        for key, a, b, meaning in chain:
            table.append(dict(interval=key, start=a, end=b, contains=meaning,
                              ms=q([r[b] - r[a] for r in subset])))
        chains[label] = dict(decisions=len(subset), intervals=table, negative_intervals=negatives,
                             total_materialized_to_ue_receipt=q([r["ue_receipt"] - r["prediction_materialized"] for r in subset]))
    # combined intervals measured per frame (never sums of medians)
    combined = {
        "materialized_to_service_receipt": q([r["service_prediction_received"] - r["prediction_materialized"] for r in rows]),
        "service_receipt_to_prediction_worker_end": q([r["prediction_worker_end"] - r["service_prediction_received"] for r in rows]),
        "service_receipt_to_prediction_worker_end__gt_first": q([r["prediction_worker_end"] - r["service_prediction_received"] for r in rows if r["gt_first"]]),
        "service_receipt_to_prediction_worker_end__prediction_first": q([r["prediction_worker_end"] - r["service_prediction_received"] for r in rows if not r["gt_first"]]),
        "scoring_worker_end_to_result_sent": q([r["result_sent"] - r["scoring_worker_end"] for r in rows]),
        "result_sent_to_feedback_emitted": q([r["feedback_emitted"] - r["result_sent"] for r in rows]),
        "scoring_worker_end_to_ue_receipt": q([r["ue_receipt"] - r["scoring_worker_end"] for r in rows]),
        "materialized_to_ue_receipt": q([r["ue_receipt"] - r["prediction_materialized"] for r in rows]),
        "compute_end_to_materialized": q([r["prediction_materialized"] - r["edge_compute_end"] for r in rows]),
        "prediction_worker_thread_cpu__gt_first": q([r["prediction_worker_thread_cpu_ms"] for r in rows if r["gt_first"]]),
        "prediction_worker_thread_cpu__prediction_first": q([r["prediction_worker_thread_cpu_ms"] for r in rows if not r["gt_first"]]),
        "target_worker_thread_cpu__gt_first": q([r["target_worker_thread_cpu_ms"] for r in rows if r["gt_first"]]),
        "target_worker_thread_cpu__prediction_first": q([r["target_worker_thread_cpu_ms"] for r in rows if not r["gt_first"]]),
    }
    busy = collections.Counter(b["op"] for r in rows for b in r["score_worker_busy_at_receipt"])
    busy_frames = [r for r in rows if r["score_worker_busy_at_receipt"] or r["own_target_in_flight_at_receipt"]]
    contention = dict(
        decisions_with_score_worker_busy_on_another_frame=sum(1 for r in rows if r["score_worker_busy_at_receipt"]),
        decisions_with_own_target_in_flight=sum(1 for r in rows if r["own_target_in_flight_at_receipt"]),
        busy_ops=dict(busy),
        service_to_worker_start_when_busy=q([r["prediction_worker_begin"] - r["service_prediction_received"] for r in busy_frames]),
        service_to_worker_start_when_idle=q([r["prediction_worker_begin"] - r["service_prediction_received"]
                                             for r in rows if r not in busy_frames]))

    # ---- GT readiness against explicit boundaries
    def count(gt, pred):
        return sum(1 for r in rows if r[gt] <= r[pred])
    gt_boundaries = {}
    for gt in ("gt_numeric_ready_producer", "gt_published", "target_worker_begin", "target_worker_end"):
        gt_boundaries[gt] = dict(ms=q([r[gt] for r in rows]), le_170=sum(r[gt] <= DEADLINE for r in rows), **{
            "no_later_than_" + p: count(gt, p) for p in ("edge_compute_end", "prediction_materialized",
                                                          "service_prediction_received", "prediction_worker_begin",
                                                          "prediction_worker_end")})
    gt_boundaries["scoring_path"] = dict(gt_first_scored_in_prediction_request=sum(r["gt_first"] for r in rows),
                                         prediction_first_scored_in_target_request=sum(not r["gt_first"] for r in rows))

    # ---- deadline-relevant instants (all after action-open unless negative)
    instants = [k for k in rows[0] if isinstance(rows[0][k], float) and not k.endswith("_cpu_ms")]
    timing = {label: {k: q([r[k] for r in subset]) for k in instants}
              for label, subset in (("all", rows), ("warm", [r for r in rows if not r["cold"]]),
                                    ("cold", [r for r in rows if r["cold"]]),
                                    ("gt_first", [r for r in rows if r["gt_first"]]),
                                    ("prediction_first", [r for r in rows if not r["gt_first"]]))}
    counts = dict(decisions=len(rows), gt_first=sum(r["gt_first"] for r in rows),
        prediction_ready_le_170=sum(r["edge_compute_end"] <= DEADLINE for r in rows),
        scoring_complete_le_170=sum(r["scoring_worker_end"] <= DEADLINE for r in rows),
        feedback_emitted_le_170=sum(r["feedback_emitted"] <= DEADLINE for r in rows),
        ue_receipt_le_170=sum(r["ue_receipt"] is not None and r["ue_receipt"] <= DEADLINE for r in rows),
        ue_receipt_missing=sum(r["ue_receipt"] is None for r in rows),
        controller_resolution=dict(collections.Counter(r["controller_resolution"] for r in rows)),
        timely_by_scoring_path={k: sum(1 for r in rows if r["gt_first"] == (k == "gt_first") and r["ue_receipt"] <= DEADLINE)
                                for k in ("gt_first", "prediction_first")})

    # ---- per-action (policy selection) table
    groups = collections.defaultdict(list)
    for r in rows:
        a = r["action"]
        groups[f"{a['family']}/{a['quantizer']}/mode{a['mode_id']}"].append(r)
    per_action = {}
    for key, subset in sorted(groups.items()):
        per_action[key] = dict(n=len(subset), datagrams=q([r["datagrams"] for r in subset]),
            action_open_to_first_send=q([r["ue_first_send"] for r in subset]),
            uplink_last_send_to_reassembly=q([r["edge_reassembly_complete"] - r["ue_last_send"] for r in subset]),
            edge_compute=q([r["edge_compute_end"] - r["edge_compute_start"] for r in subset]),
            prediction_ready=q([r["edge_compute_end"] for r in subset]),
            ue_receipt=q([r["ue_receipt"] for r in subset]),
            timely=sum(r["ue_receipt"] <= DEADLINE for r in subset))
    uplink = [r["edge_reassembly_complete"] - r["ue_last_send"] for r in rows]
    first_arrival = [r["edge_first_datagram"] - r["ue_first_send"] for r in rows]
    spread = [r["edge_reassembly_complete"] - r["edge_first_datagram"] for r in rows]
    payload = dict(
        datagrams=q([r["datagrams"] for r in rows]),
        first_send_to_first_datagram_at_edge=q(first_arrival),
        first_to_last_datagram_at_edge=q(spread),
        last_send_to_reassembly=q(uplink),
        spearman_datagrams_vs_last_send_to_reassembly=spearman([r["datagrams"] for r in rows], uplink),
        spearman_datagrams_vs_first_to_last_datagram_at_edge=spearman([r["datagrams"] for r in rows], spread),
        spearman_datagrams_vs_ue_receipt=spearman([r["datagrams"] for r in rows], [r["ue_receipt"] for r in rows]),
        note="Only the selected actions are observed; other actions' live latency is unmeasured here.")

    # ---- training vs live boundaries (recorded training statistics quoted, nothing modelled)
    audit = json.loads((ROOT / "experiments/splitfusion_run5_diagnosis_v1/20261003_v1/environment_audit.json").read_bytes())
    stats = audit["latency"]["statistics"]
    live_residual_like = [r["ue_first_send"] + (r["ue_receipt"] - r["prediction_materialized"]) for r in rows]
    training = dict(
        actor_trained_under="original Run-5 collector composition (Run-5 diagnosis REPORT section 3)",
        formula=audit["latency"]["formula"], residual_identity=audit["latency"]["residual_identity"],
        recorded_training_residual_ms={k: stats["residual_ms"][k] for k in ("n", "median", "p95", "p99")},
        recorded_training_gt_wait_ms={k: stats["gt_wait_ms"][k] for k in ("n", "median", "p95")},
        recorded_training_gt_scoring_ms={k: stats["gt_scoring_ms"][k] for k in ("n", "median", "p95")},
        live_nearest_boundaries=dict(
            action_open_to_first_send=q([r["ue_first_send"] for r in rows]),
            prediction_materialized_to_ue_receipt=q([r["ue_receipt"] - r["prediction_materialized"] for r in rows]),
            per_frame_sum_of_the_two=q(live_residual_like),
            last_send_to_reassembly__training_term_is_modelled_T_v2b=q(uplink),
            reassembly_to_prediction_materialized__absent_from_training=q(
                [r["prediction_materialized"] - r["edge_reassembly_complete"] for r in rows])),
        caveats=["'prepare' in the training residual is the retained collector's boundary; action-open is the "
                 "nearest live stamp and is used without claiming identity.",
                 "T_v2b was not evaluated for these frames; no modelled value is reported.",
                 "The shared L_op provider (Run-4B) is a different composition and did not train this actor."])
    # state-distribution evidence for the actor's previous-outcome features
    prior_success = sum(1 for r in rows if r["controller_resolution"] == "SUCCESS")
    state_shift = dict(live_previous_success_fraction=round(prior_success / len(rows), 4),
                       recorded_training_previous_success_fraction=0.8486,
                       source="training value from Run-5 diagnosis REPORT section 4 (seed-43, 40,288 decisions)")

    doc = dict(schema="splitfusion.exact_gt_live.closeout.v1", attempt=str(attempt), prior_result=str(prior),
               unit="ms after the retained action-open (CLOCK_MONOTONIC_RAW)", counts=counts,
               post_prediction_chains=chains, combined_intervals=combined, score_worker_contention=contention,
               gt_readiness=gt_boundaries, timing=timing, per_action=per_action, payload=payload,
               training_vs_live=training, state_shift=state_shift)
    with open(out / "CLOSEOUT.json", "x") as stream:
        json.dump(doc, stream, indent=1, sort_keys=True)
    with open(out / "BOUNDARIES.jsonl", "x") as stream:
        for r in rows:
            stream.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps(dict(counts=counts, chains={k: [(i["interval"], i["ms"].get("p50"), i["ms"].get("p95"))
          for i in v["intervals"]] + [("TOTAL", v["total_materialized_to_ue_receipt"]["p50"], v["negative_intervals"])]
          for k, v in chains.items()}, contention=contention, gt=gt_boundaries, payload=payload, combined=combined,
          per_action=per_action, training=training["live_nearest_boundaries"], state=state_shift), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.attempt, args.prior, args.name)
