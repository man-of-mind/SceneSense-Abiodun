"""Per-decision rows and the action-by-action quality/latency table for the fixed-action screen.

Read-only on the arms; create-only outputs under experiments/splitfusion_fixed_action_screen_v1/<name>:
DECISIONS.jsonl (one row per decision, every arm), ARMS.json (per-arm metrics + integration result),
TABLE.md, PIPELINE_IDENTITY.json, UNAVAILABLE_FIELDS.json, OUTPUT_SHA256.json.
All instants: ms after the retained action-open (CLOCK_MONOTONIC_RAW). No sums of medians.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from .analyze import Arm, COLD, DEADLINE, MAIN, WT, integration, load_jsonl, profile, q, sha, wilson

UNAVAILABLE = {
    "prior_ul_mcs, pre_action_rlc_backlog_bytes, camera_si": "not stored raw per decision; decoded exactly from the "
        "recorded 22-D state (run4_contract scaling: MCS*28, expm1(x*log1p(5e7)), SI*scale+center); null on fallbacks",
    "payload bytes": "UE per_frame_metrics.csv payload_bytes (zstd application payload) and payload_chunks; IP/GTP wire "
        "overhead not recorded per frame",
    "scorer-internal validation vs scoring split": "not archived; worker request reported as one interval",
}


def decode_state(features):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.live_state_v2 import TRAINING_SCALING as S
    if not features:
        return dict(camera_si=None, radar_p40=None, prior_ul_mcs=None, pre_action_rlc_backlog_bytes=None, snr_feature=None)
    return dict(camera_si=features[0] * S.camera_si_scale + S.camera_si_center, radar_p40=features[1],
                prior_ul_mcs=int(round(features[2] * 28)),
                pre_action_rlc_backlog_bytes=int(round(math.expm1(features[3] * S.backlog_log1p_scale))),
                snr_feature=features[21])


def quality_fields(measurement):
    m = measurement or {}
    out = {}
    for cls in ("person", "vehicle"):
        out[f"{cls}_tp"], out[f"{cls}_fn"], out[f"{cls}_fp"] = (m.get(f"loc_{cls}_{k}") for k in ("tp", "fn", "fp"))
        out[f"{cls}_eligible_gt"] = m.get(f"loc_{cls}_eligible_gt")
        out[f"{cls}_matched_xy_errors_m"] = m.get(f"loc_{cls}_matched_xy_errors_m")
        for k in ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels"):
            out[f"seg_{cls}_{k}"] = m.get(f"seg_{cls}_{k}")
    return out


def rows_for(arm):
    cell = arm.cell
    frames = [f for f in load_jsonl(cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")]
    edge_state = (arm.j("profiler/edge/numeric_runtime/EDGE_EARLY_GT.json") or {}).get("state") or {}
    records = {int(r["frame_id"]): r for r in edge_state.get("records") or []}
    phase6 = arm.j("run4_phase6/PHASE6_UE_EVIDENCE.json")
    receipts = collections.defaultdict(list)
    for row in phase6["feedback_rows"]:
        receipts[int(row["frame_id"])].append(row)
    by_open = {int(r["action_open_timestamp_ns"]): r for r in phase6["resolutions"]}
    decisions = {d["frame_id"]: d for d in phase6["decisions"]}
    snr = {int(d["frame_id"]): d for d in (arm.j("run5_policy/SNR_EVIDENCE.json") or {}).get("decision_snr", [])}
    per_frame = {int(r["frame_id"]): r for r in csv.DictReader(open(cell / "per_frame_metrics.csv"))}
    forced = {r["frame_id"]: r for r in load_jsonl(cell / "forced_action/FORCED_ACTIONS.jsonl")}
    us, _, _ = profile(cell / "profiler/ue.profile.jsonl")
    es, em, _ = profile(cell / "profiler/edge/edge.profile.jsonl")
    out = []
    for k, f in enumerate(frames):
        fid, ao = f["frame_id"], f["action_open_raw_ns"]
        ms = lambda ns: None if ns is None else (ns - ao) / 1e6
        ev = {}
        for e in f["events"]:
            ev.setdefault(e["role"] + ":" + e["name"], e["raw_ns"])
        req = {(r["worker"], r["op"]): r for r in f.get("requests", [])}
        pred, target = req.get(("score", "PREDICTION")), req.get(("score", "TARGET"))
        gt_first = not f.get("joined_on_target")
        scoring = pred if gt_first else target
        edge_frame = (edge_state.get("frames") or {}).get(str(fid)) or {}
        receipt = min((r["receipt"]["ns"] for r in receipts[fid] if r.get("receipt")), default=None)
        res = by_open.get(ao)
        u, e_, m = us.get(fid, {}), es.get(fid, {}), em.get(fid, {})
        span = lambda d, key, i: ms(d[key][i]) if key in d else None
        producer = cell / f"numeric_gt/service/archive/frame_{fid}/producer/producer_events.json"
        pev = {x["name"]: x["raw_ns"] for x in json.loads(producer.read_bytes())["events"]} if producer.exists() else {}
        dec = decisions.get(fid, {})
        state = decode_state(dec.get("features"))
        pf = per_frame.get(fid, {})
        result = f.get("result") or {}
        record = records.get(fid) or {}
        ident = json.loads(f["identity"]) if f.get("identity") else {}
        context = ident.get("frame_context") or {}
        fr = forced.get(fid) or {}
        row = dict(arm=arm.dir.name, cell_id=cell.name, network_profile_id=arm.reg.get("network_profile"),
            frame_id=fid, cold=k < COLD, decision_index=k,
            executed_action=fr.get("executed"), proposed_action=fr.get("proposed"), forced_action=arm.forced,
            # boundaries (closeout_01 names)
            capture_world_tick_end=span(u, "carla.world_tick", 1), sensor_ready=span(u, "ue.sensor_wait", 1),
            actor_end=span(u, "agent.actor", 1), ue_input_start=span(u, "ue.input_7ch", 0), ue_input_end=span(u, "ue.input_7ch", 1),
            ue_encode_start=span(u, "ue.codec.total_ue_preparation", 0), ue_encode_end=span(u, "ue.codec.total_ue_preparation", 1),
            ue_first_send=span(u, "ue.udp_send", 0), ue_last_send=ms(ev.get("ue:last_datagram_sent")),
            edge_first_datagram=ms((m.get("edge.datagram_first") or {}).get("t_raw_ns")),
            edge_reassembly_complete=ms((m.get("edge.reassembly_complete") or {}).get("t_raw_ns")),
            edge_compute_start=span(e_, "edge.compute", 0), edge_compute_end=span(e_, "edge.compute", 1),
            prediction_materialized=ms(ev.get("edge:prediction_materialized")),
            prediction_dequeued=ms(ev.get("edge:prediction_dequeued")), prediction_message_built=ms(ev.get("edge:prediction_sent")),
            service_prediction_received=ms(ev.get("service:prediction_received")),
            prediction_worker_begin=None if pred is None else ms(pred["timing"]["begin_raw_ns"]),
            prediction_worker_end=None if pred is None else ms(pred["timing"]["end_raw_ns"]),
            gt_numeric_ready_producer=ms(pev.get("numeric_ready")), gt_published=ms(ev.get("prep:published")),
            target_worker_begin=None if target is None else ms(target["timing"]["begin_raw_ns"]),
            target_worker_end=None if target is None else ms(target["timing"]["end_raw_ns"]),
            gt_first=gt_first, scoring_request_op=None if scoring is None else scoring["op"],
            scoring_worker_end=None if scoring is None else ms(scoring["timing"]["end_raw_ns"]),
            result_sent=ms(ev.get("service:result_sent")), edge_result_received=ms(edge_frame.get("result_received")),
            feedback_emitted=ms(edge_frame.get("emitted")), ue_receipt=ms(receipt),
            # outcomes
            service_outcome=f.get("outcome"), controller_resolution=None if res is None else res["terminal"],
            feedback_kind=record.get("kind"), feedback_reason=record.get("reason"),
            feedback_class=sorted({r.get("class") for r in receipts[fid]}),
            q_perc=result.get("q_perc"), quality_defined=result.get("quality_defined"),
            timely=receipt is not None and (receipt - ao) / 1e6 <= DEADLINE,
            # payload / channel / scene
            payload_bytes=int(pf["payload_bytes"]) if pf.get("payload_bytes") else None,
            payload_bytes_uncompressed=int(pf["payload_bytes_uncompressed"]) if pf.get("payload_bytes_uncompressed") else None,
            payload_chunks=int(pf["payload_chunks"]) if pf.get("payload_chunks") else None,
            datagrams_at_edge=(m.get("edge.reassembly_complete") or {}).get("chunks"),
            snr_db=((snr.get(fid) or {}).get("snr") or {}).get("value_db"), **state,
            raw_radar_return_count=int(pf["raw_radar_return_count"]) if pf.get("raw_radar_return_count") else None,
            carla_timestamp=dec.get("carla_timestamp"), capture_timestamp_ns=context.get("capture_timestamp_ns"),
            ego_world=context.get("ego_world"), route_tick=pf.get("route_tick"),
            **quality_fields(result.get("measurement")))
        out.append(row)
    return out


def pooled(rows, cls):
    tp = sum(r[f"{cls}_tp"] or 0 for r in rows)
    fn = sum(r[f"{cls}_fn"] or 0 for r in rows)
    fp = sum(r[f"{cls}_fp"] or 0 for r in rows)
    errors = [e for r in rows for e in (r[f"{cls}_matched_xy_errors_m"] or [])]
    inter = sum(r[f"seg_{cls}_intersection_pixels"] or 0 for r in rows)
    union = sum(r[f"seg_{cls}_union_pixels"] or 0 for r in rows)
    return dict(tp=tp, fn=fn, fp=fp, eligible=tp + fn, recall=None if tp + fn == 0 else round(tp / (tp + fn), 4),
                misses=fn, matched_error_m=q(errors), seg_iou_pooled=None if union == 0 else round(inter / union, 4),
                frames_with_eligible=sum(1 for r in rows if (r[f"{cls}_eligible_gt"] or 0) > 0))


def interval(rows, a, b):
    return q([r[b] - r[a] for r in rows if r.get(a) is not None and r.get(b) is not None])


def arm_metrics(rows):
    n = len(rows)
    timely = sum(r["timely"] for r in rows)
    success = sum(r["controller_resolution"] == "SUCCESS" for r in rows)
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    defined = [r for r in scored if r["quality_defined"]]
    timely_defined = [r for r in defined if r["timely"]]
    ci = wilson(timely, n)
    feas = None if ci is None else ("LATENCY_FEASIBLE" if ci[0] >= 0.5 else "LATENCY_INFEASIBLE" if ci[1] < 0.5 else "INCONCLUSIVE")
    def quality(subset):
        return dict(decisions=len(subset), q_perc=q([r["q_perc"] for r in subset]),
                    person=pooled(subset, "person"), vehicle=pooled(subset, "vehicle"))
    latency = {lab: q([r["ue_receipt"] for r in sub]) for lab, sub in
               (("all", rows), ("warm", [r for r in rows if not r["cold"]]), ("cold", [r for r in rows if r["cold"]]))}
    stages = dict(action_open_to_first_send=q([r["ue_first_send"] for r in rows]),
        ue_input=interval(rows, "ue_input_start", "ue_input_end"), ue_encode=interval(rows, "ue_encode_start", "ue_encode_end"),
        first_send_to_first_datagram=interval(rows, "ue_first_send", "edge_first_datagram"),
        last_send_to_reassembly=interval(rows, "ue_last_send", "edge_reassembly_complete"),
        reassembly_to_compute_start=interval(rows, "edge_reassembly_complete", "edge_compute_start"),
        edge_compute=interval(rows, "edge_compute_start", "edge_compute_end"),
        prediction_ready=q([r["edge_compute_end"] for r in rows]), gt_ready_target_admitted=q([r["target_worker_end"] for r in rows]),
        gt_numeric_ready=q([r["gt_numeric_ready_producer"] for r in rows]),
        materialized_to_service=interval(rows, "prediction_materialized", "service_prediction_received"),
        service_to_scoring_complete=interval(rows, "service_prediction_received", "scoring_worker_end"),
        scoring_complete_to_ue_receipt=interval(rows, "scoring_worker_end", "ue_receipt"),
        materialized_to_ue_receipt=interval(rows, "prediction_materialized", "ue_receipt"),
        materialized_to_ue_receipt_gt_first=interval([r for r in rows if r["gt_first"]], "prediction_materialized", "ue_receipt"),
        materialized_to_ue_receipt_prediction_first=interval([r for r in rows if not r["gt_first"]], "prediction_materialized", "ue_receipt"))
    return dict(decisions=n, timely=timely, timely_fraction=None if n == 0 else round(timely / n, 4), timely_wilson95=ci,
        latency_feasibility_provisional=feas, success_defined_quality=success,
        success_fraction=None if n == 0 else round(success / n, 4), success_wilson95=wilson(success, n),
        missing_replies=sum(r["ue_receipt"] is None for r in rows),
        timeouts=sum(r["controller_resolution"] == "TIMEOUT" for r in rows),
        undefined_quality=sum(r["feedback_reason"] == "QUALITY_UNDEFINED_NO_ELIGIBLE_GT" for r in rows),
        undefined_quality_timely=sum(r["feedback_reason"] == "QUALITY_UNDEFINED_NO_ELIGIBLE_GT" and r["timely"] for r in rows),
        other_faults=sum((r["feedback_kind"] or "DELIVERED_SUCCESS") != "DELIVERED_SUCCESS"
                         and r["feedback_reason"] != "QUALITY_UNDEFINED_NO_ELIGIBLE_GT" for r in rows),
        gt_first=sum(r["gt_first"] for r in rows), latency_ms=latency, stages_ms=stages,
        warm=dict(decisions=sum(not r["cold"] for r in rows), timely=sum(r["timely"] for r in rows if not r["cold"]),
                  success=sum(r["controller_resolution"] == "SUCCESS" for r in rows if not r["cold"])),
        payload_bytes=q([r["payload_bytes"] for r in rows]), datagrams=q([r["datagrams_at_edge"] for r in rows]),
        snr_db=q([r["snr_db"] for r in rows]), prior_ul_mcs=q([r["prior_ul_mcs"] for r in rows]),
        backlog_bytes=q([r["pre_action_rlc_backlog_bytes"] for r in rows]), camera_si=q([r["camera_si"] for r in rows]),
        radar_p40=q([r["radar_p40"] for r in rows]),
        quality_all_scored_defined=quality(defined), quality_timely_defined=quality(timely_defined),
        scene=dict(first_frame=rows[0]["frame_id"] if rows else None, last_frame=rows[-1]["frame_id"] if rows else None,
                   first_carla_timestamp=rows[0]["carla_timestamp"] if rows else None,
                   last_carla_timestamp=rows[-1]["carla_timestamp"] if rows else None))


def pipeline_identity(arm):
    reg = arm.reg
    image = arm.j("edge_image_prelaunch.json") or {}
    return dict(arm=arm.dir.name, registration_file_sha256=sha(arm.dir / "registration.json"),
        registration_sha256=reg["registration_sha256"], controller_lineage_sha256=reg["controller_lineage_sha256"],
        exact_sources_digest=hashlib.sha256(json.dumps(reg["exact_gt_sources"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        edge_image=image.get("id"), cpu_partition=json.loads((arm.dir / "arm.json").read_bytes()),
        exact_gt_limits=reg["exact_gt_limits"], forced_action=reg["forced_action"], actor_weights_sha256=reg["policy"]["expected_weights_sha256"],
        acceptance_sha256=reg["acceptance_sha256"], selection_sha256=reg["selection_sha256"],
        prospective_file_sha256=sha(arm.dir / "PROSPECTIVE.json"))


def fmt(x, digits=1):
    return "–" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def run(screen, campaign, name):
    screen, campaign = Path(screen).resolve(strict=True), Path(campaign).resolve(strict=True)
    out = MAIN / "experiments/splitfusion_fixed_action_screen_v1" / name
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((MAIN / "experiments/splitfusion_fixed_action_screen_v1/selection_01/SELECTION.json").read_bytes())
    arms, identities, all_rows = {}, {}, []
    for index, action in enumerate(selection["arm_order"]):
        directory = screen / f"arm{index}_a{action}"
        arm = Arm(directory)
        if arm.cell is None:
            arms[directory.name] = dict(action=action, launched=False)
            continue
        check_path = campaign / f"check_{directory.name}.json"
        check = json.loads(check_path.read_bytes()) if check_path.exists() and check_path.stat().st_size else None
        rows = rows_for(arm)
        all_rows.extend(rows)
        arms[directory.name] = dict(action=action, roles=arm.reg["arm_roles"], forced_action=arm.forced,
                                    integration=check, metrics=arm_metrics(rows))
        identities[directory.name] = pipeline_identity(arm)
    with open(out / "DECISIONS.jsonl", "x") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    with open(out / "ARMS.json", "x") as stream:
        json.dump(arms, stream, indent=1, sort_keys=True, default=str)
    with open(out / "PIPELINE_IDENTITY.json", "x") as stream:
        json.dump(identities, stream, indent=1, sort_keys=True)
    with open(out / "UNAVAILABLE_FIELDS.json", "x") as stream:
        json.dump(UNAVAILABLE, stream, indent=1, sort_keys=True)
    lines = ["| Arm | Action | Role | Payload P50 B | Timely / n | Success / n | Receipt P50 / P95 ms | Pred ready P50 | "
             "Q_perc mean (all scored) | Person recall | Person err P50 m | Vehicle recall | Veh err P50 m | Seg IoU P/V | Q_perc mean (timely) | Latency class |",
             "|" + "---|" * 16]
    for key, a in arms.items():
        if not a.get("metrics"):
            lines.append(f"| {key} | {a['action']} | not launched |" + " |" * 13)
            continue
        m, qa, qt = a["metrics"], a["metrics"]["quality_all_scored_defined"], a["metrics"]["quality_timely_defined"]
        f = a["forced_action"]
        lines.append(" | ".join(["", key, f"{f['action_id']} {f['family']}/{f['quantizer']} q{f['q_e4']}", ", ".join(a["roles"]),
            fmt(m["payload_bytes"].get("p50"), 0), f"{m['timely']}/{m['decisions']}", f"{m['success_defined_quality']}/{m['decisions']}",
            f"{fmt(m['latency_ms']['all'].get('p50'))} / {fmt(m['latency_ms']['all'].get('p95'))}",
            fmt(m["stages_ms"]["prediction_ready"].get("p50")), fmt(qa["q_perc"].get("mean"), 3),
            fmt(qa["person"]["recall"], 3), fmt(qa["person"]["matched_error_m"].get("p50"), 2),
            fmt(qa["vehicle"]["recall"], 3), fmt(qa["vehicle"]["matched_error_m"].get("p50"), 2),
            f"{fmt(qa['person']['seg_iou_pooled'], 3)} / {fmt(qa['vehicle']['seg_iou_pooled'], 3)}",
            fmt(qt["q_perc"].get("mean"), 3), str(m["latency_feasibility_provisional"]), ""]).strip())
    (out / "TABLE.md").write_text("\n".join(lines) + "\n")
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest["campaign"] = {p.name: sha(p) for p in sorted(campaign.iterdir()) if p.is_file()}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.screen, args.campaign, args.name)
