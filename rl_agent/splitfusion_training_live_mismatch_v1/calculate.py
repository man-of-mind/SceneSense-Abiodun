"""Training-versus-live latency/state mapping for the original quality-aware Run-4 and Run-5 actors.

Read-only on every input; writes create-only CALCULATIONS.json and INPUT_SHA256.json. CPU only, no inference,
no services. Every statistic is computed per row or per frame; nothing is reconstructed from pooled medians.
Quantiles: numpy linear. Run from the abiodun/ root:
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
        python3 -m rl_agent.splitfusion_training_live_mismatch_v1.calculate --output <new dir>
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as ss

ROOT = Path(__file__).resolve().parents[2]
NEU = ROOT.parent
W5 = NEU / "abiodun_run5_wt"
PROF = NEU / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
DEADLINE_MS = 170.0
SEND_SPAN_NS_PER_BYTE = 0.513047          # collector_v1.py:87 (training constant)
ACTOR_RESERVE_MS = 1.0                    # ACTOR_TIMING_QUALIFICATION.json via collector_v1.load_actor_reserve_ns

PROBE = ROOT / "experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse_retry4/cells"
PROBE_CELLS = ("a50__favorable_stable", "a50__adverse_stable")
LIVE100 = PROF / "20261005_exact_gt_early_service_live100_v1/capture/cells/a71__favorable_stable"
RUN5CAP = PROF / "20261002T224700Z_run5_live_avo_capture_1000_v1/capture/cells/a71__favorable_stable"
INPUTS = {
    "collector_v1": ROOT / "rl_agent/ue_production_transport_model_v2/collector_v1.py",
    "artifact_v2": ROOT / "rl_agent/ue_production_transport_model_v2/artifact_v2.py",
    "model_v2": ROOT / "rl_agent/ue_production_transport_model_v2/model_v2.py",
    "transport_model_v2": ROOT / "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json",
    "run4_seed43_decisions": ROOT / ("rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_campaign/"
                                     "20260928_2a92201_three_seed_10000_v1/seed_43/DECISIONS.jsonl"),
    "run5_seed43_decisions": W5 / "rl_agent/splitfusion_hybrid_sac_run5_v1/campaign_runs/run5_three_seed_10000_v1/seed_43/decisions.jsonl",
    "live100_boundaries": ROOT / "experiments/splitfusion_exact_gt_live_v1/closeout_01/BOUNDARIES.jsonl",
    "live100_evidence": LIVE100 / "run4_phase6/PHASE6_UE_EVIDENCE.json",
    "live100_per_frame": LIVE100 / "per_frame_metrics.csv",
    "run5cap_legacy_chain": ROOT / "experiments/splitfusion_numeric_gt_runtime_v1/critical_path_decision_01/LEGACY_CHAIN.csv",
    "run5cap_evidence": RUN5CAP / "run4_phase6/PHASE6_UE_EVIDENCE.json",
    "run5cap_per_frame": RUN5CAP / "per_frame_metrics.csv",
}
for cell in PROBE_CELLS:
    INPUTS[f"probe_join_{cell}"] = PROBE / cell / "quality_feedback_timing_join.csv"
    INPUTS[f"probe_policy_timing_{cell}"] = PROBE / cell / "quality_policy_timing.csv"
    INPUTS[f"probe_edge_publication_{cell}"] = PROBE / cell / "direct_edge_map/direct_edge_publication.csv"
EXPECTED_SHA = {  # training pins recorded by the Run-4 campaign / Run-5 preregistration
    "collector_v1": "6ff7766923d16c03ed7c4e31ebe2c4c7198acc0b4f06dc3a194e4657ad173385",
    "transport_model_v2": "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c",
    "probe_join_a50__favorable_stable": "a2740342de391e8888d6a517c9571e9c0548020070d6d8080806c9863995f1d6",
    "probe_join_a50__adverse_stable": "f9113fda612083dfb8d3f905935651798474e2e9fd7b2ae9cc01978688a17367",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), mean=round(float(v.mean()), 3), p50=round(float(np.percentile(v, 50)), 3),
                p95=round(float(np.percentile(v, 95)), 3), p99=round(float(np.percentile(v, 99)), 3),
                min=round(float(v.min()), 3), max=round(float(v.max()), 3),
                le_170=int((v <= DEADLINE_MS).sum()))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- training composition (as trained)
def training_side():
    out = {}
    rows = [json.loads(l) for l in open(INPUTS["run4_seed43_decisions"])]
    succ = [r for r in rows if r["terminal"] == "SUCCESS"]
    send_span = [(r["composed_ns"] - r["retained_residual_ns"] - r["transport_ns"]) / 1e6 - ACTOR_RESERVE_MS for r in rows]
    out["run4_seed43"] = dict(
        decisions=len(rows), success=len(succ), timeout=len(rows) - len(succ),
        success_fraction=round(len(succ) / len(rows), 6),
        timeout_success_draw_failed=sum(1 for r in rows if r["terminal"] != "SUCCESS" and not r["success_drawn"]),
        timeout_composed_over_170=sum(1 for r in rows if r["terminal"] != "SUCCESS" and r["success_drawn"]),
        timeout_residual_draw_alone_over_170=sum(1 for r in rows if r["terminal"] != "SUCCESS" and r["success_drawn"]
                                                 and r["retained_residual_ns"] / 1e6 > DEADLINE_MS),
        timeout_composed_over_170_with_residual_le_170=sum(
            1 for r in rows if r["terminal"] != "SUCCESS" and r["success_drawn"]
            and r["retained_residual_ns"] / 1e6 <= DEADLINE_MS),
        composed_latency_all_ms=q([r["composed_ns"] / 1e6 for r in rows]),
        composed_latency_success_ms=q([r["composed_ns"] / 1e6 for r in succ]),
        term_retained_residual_ms=q([r["retained_residual_ns"] / 1e6 for r in rows]),
        term_transport_T_v2b_ms=q([r["transport_ns"] / 1e6 for r in rows]),
        term_send_span_ms=q(send_span),
        residual_draw_over_170=sum(1 for r in rows if r["retained_residual_ns"] / 1e6 > DEADLINE_MS),
        reward_frame_wire_bytes=q([r["reward_frame_wire_bytes"] for r in rows]),
        prev_success_fraction=round(sum(1 for r in rows if r["previous_success"]) / len(rows), 6),
        genesis_states=sum(1 for r in rows if not r["previous_present"]))
    rows5 = [json.loads(l) for l in open(INPUTS["run5_seed43_decisions"])]
    rows5.sort(key=lambda r: r["index"])
    s5 = [r for r in rows5 if r["terminal"] == "SUCCESS"]
    prev_success = sum(1 for a in rows5[:-1] if a["terminal"] == "SUCCESS")
    out["run5_seed43"] = dict(
        decisions=len(rows5), success=len(s5), success_fraction=round(len(s5) / len(rows5), 6),
        latency_success_ms=q([r["latency_ms"] for r in s5]),
        prev_success_fraction=round(prev_success / len(rows5), 6),
        note="per-term ns are not retained in Run-5 decisions.jsonl; same collector_v1 composition (verified pin)")
    return out


# ---------------------------------------------------------------- training source probe re-decomposition
def probe_side():
    res, per_row = {}, []
    for cell in PROBE_CELLS:
        join = {int(r["frame_id"]): r for r in csv.DictReader(open(INPUTS[f"probe_join_{cell}"]))}
        pol = {int(r["frame_id"]): r for r in csv.DictReader(open(INPUTS[f"probe_policy_timing_{cell}"]))}
        pub = {int(r["frame_id"]): r for r in csv.DictReader(open(INPUTS[f"probe_edge_publication_{cell}"]))}
        for fid, r in join.items():
            total = fnum(r["model_prepare_start_to_ue_receive_ms"])
            first = fnum(r["first_feature_datagram_send_to_ue_receive_ms"])
            enq = fnum(r["evaluation_enqueued_to_ue_receive_ms"])
            if None in (total, first, enq):
                continue
            residual = total - (first - enq)
            if not (math.isfinite(residual) and residual > 0):       # identical filter to load_retained_residuals
                continue
            act = fnum(r["model_action_start_to_ue_receive_ms"])
            pred = fnum(r["final_prediction_ready_to_ue_receive_ms"])
            started = fnum(r["evaluation_started_to_ue_receive_ms"])
            done = fnum(r["evaluation_completed_to_ue_receive_ms"])
            row = dict(cell=cell, frame_id=fid, residual=residual,
                       seven_channel_input=None if act is None else act - total,
                       prepare_to_first_send=total - first, first_send_to_enqueue=first - enq,
                       first_send_to_prediction_ready=None if pred is None else first - pred,
                       prediction_ready_to_enqueue=None if pred is None else pred - enq,
                       enqueue_to_receipt=enq,
                       gt_wait=None if started is None else enq - started,
                       scoring=None if None in (started, done) else started - done,
                       prepare_to_receipt=total, action_start_to_receipt=act,
                       first_send_to_reassembly=None, reassembly_to_enqueue=None)
            p, e = pol.get(fid), pub.get(fid)
            if p is not None and e is not None and fnum(e.get("reassembly_complete_wall_s")) is not None:
                reas_ns = float(e["reassembly_complete_wall_s"]) * 1e9   # edge time.time(); same kernel CLOCK_REALTIME
                fs, en = fnum(p["first_feature_datagram_send_wall_ns"]), fnum(p["evaluation_enqueued_wall_ns"])
                if fs is not None and en is not None:
                    row["first_send_to_reassembly"] = (reas_ns - fs) / 1e6
                    row["reassembly_to_enqueue"] = (en - reas_ns) / 1e6
            per_row.append(row)
    keys = [k for k in per_row[0] if k not in ("cell", "frame_id")]
    res["rows"] = len(per_row)
    res["intervals_ms"] = {k: q([r[k] for r in per_row]) for k in keys}
    joined = [r for r in per_row if r["reassembly_to_enqueue"] is not None]
    res["reassembly_join"] = dict(joined=len(joined), negative_intervals=sum(
        1 for r in joined if r["reassembly_to_enqueue"] < 0 or r["first_send_to_reassembly"] < 0))
    res["residual_without_gt_wait_and_scoring_ms"] = q([r["residual"] - r["gt_wait"] - r["scoring"] for r in per_row
                                                       if r["gt_wait"] is not None and r["scoring"] is not None])
    return res, per_row


# ---------------------------------------------------------------- transport model evaluated on live frames
def load_transport():
    from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
    return A2.ProductionTransportModelV2.load(INPUTS["transport_model_v2"])


def decision_inputs(evidence_path, per_frame_path):
    ev = json.load(open(evidence_path))
    pf = {int(r["frame_id"]): r for r in csv.DictReader(open(per_frame_path))}
    out = {}
    for d in ev["decisions"]:
        if d.get("kind") != "POLICY_DECISION" or d.get("features") is None:
            continue
        f, r = d["features"], pf.get(int(d["frame_id"]))
        wire = None
        if r is not None:
            wire = fnum(r.get("udp_application_bytes")) or fnum(r.get("payload_bytes"))
        out[int(d["frame_id"])] = dict(
            mcs=int(round(f[2] * 28.0)),                              # (mcs - 0)/(28 - 0), run4_contract.py:1602
            backlog=float(round(math.expm1(f[3] * math.log1p(50_000_000)))),  # collector_v1 BACKLOG_LOG1P_SCALE
            wire=wire, wire_source="udp_application_bytes" if r is not None and fnum(r.get("udp_application_bytes"))
            else "payload_bytes", features=f, mode_id=d.get("mode_id"), q_e4=d.get("q_e4"),
            session=d.get("decision_session_uuid"))
    return ev, out


def transport_eval(model, inputs, measured_uplink, residual_pool):
    rows = []
    pool = np.asarray(residual_pool)
    for fid, up in measured_uplink.items():
        x = inputs.get(fid)
        if x is None or x["wire"] is None or up is None:
            continue
        p = model.predict(pre_enqueue_backlog_bytes=x["backlog"], wire_bytes=int(x["wire"]), prior_ul_mcs=x["mcs"],
                          allow_out_of_support=True)
        fixed = SEND_SPAN_NS_PER_BYTE * x["wire"] / 1e6 + p.conditional_latency_ms + ACTOR_RESERVE_MS
        # Training probability that this exact decision resolves SUCCESS: success draw x P_pool(R + fixed <= 170).
        p_timely = p.on_time_probability * float(((pool + fixed) <= DEADLINE_MS).mean())
        rows.append(dict(frame_id=fid, wire=x["wire"], mcs=x["mcs"], backlog=x["backlog"],
                         modeled_T_v2b=p.conditional_latency_ms, on_time_probability=p.on_time_probability,
                         measured_last_send_to_reassembly=up, error=up - p.conditional_latency_ms,
                         training_success_probability=p_timely))
    if not rows:
        return dict(n=0), rows
    rho = ss.spearmanr([r["modeled_T_v2b"] for r in rows], [r["measured_last_send_to_reassembly"] for r in rows])
    return dict(
        n=len(rows), modeled_T_v2b_ms=q([r["modeled_T_v2b"] for r in rows]),
        measured_last_send_to_reassembly_ms=q([r["measured_last_send_to_reassembly"] for r in rows]),
        measured_minus_modeled_ms=q([r["error"] for r in rows]),
        spearman_modeled_vs_measured=dict(rho=round(float(rho.correlation), 4), p=float(rho.pvalue)),
        on_time_probability=q([r["on_time_probability"] for r in rows]),
        expected_training_success_count=round(sum(r["training_success_probability"] for r in rows), 2),
        mcs=q([r["mcs"] for r in rows]), wire_bytes=q([r["wire"] for r in rows]),
        backlog_bytes=q([r["backlog"] for r in rows])), rows


def state_summary(ev, inputs):
    feats = [x["features"] for x in inputs.values()]
    succ = [f for f in feats if f[20] == 1.0]
    res = {r["terminal"] for r in ev.get("resolutions", [])}
    return dict(decisions=len(feats), genesis_prev_present_0=sum(1 for f in feats if f[19] == 0.0),
                prev_success=len(succ), prev_success_fraction=round(len(succ) / len(feats), 6),
                prev_timeout_state=sum(1 for f in feats if f[19] == 1.0 and f[20] == 0.0),
                prev_latency_ms_on_success=q([f[18] * DEADLINE_MS for f in succ]),
                prev_qperc_on_success=q([f[17] for f in succ]),
                sessions=len({x["session"] for x in inputs.values()}),
                resolution_terminals={t: sum(1 for r in ev.get("resolutions", []) if r["terminal"] == t) for t in res})


# ---------------------------------------------------------------- live100 mapping onto training terms
def live100(model, pool):
    rows = [json.loads(l) for l in open(INPUTS["live100_boundaries"])]
    ev, inputs = decision_inputs(INPUTS["live100_evidence"], INPUTS["live100_per_frame"])
    seg = {k: [] for k in ("A_action_open_to_input_start", "B_seven_channel_input", "C_input_end_to_first_send",
                           "D_first_to_last_send", "E_last_send_to_reassembly", "F_reassembly_to_prediction_materialized",
                           "G_prediction_materialized_to_ue_receipt")}
    on_training_clock, telescopes, payload, uplink, receipt = [], True, [], {}, []
    for r in rows:
        parts = dict(A_action_open_to_input_start=r["ue_input_start"],
                     B_seven_channel_input=r["ue_input_end"] - r["ue_input_start"],
                     C_input_end_to_first_send=r["ue_first_send"] - r["ue_input_end"],
                     D_first_to_last_send=r["ue_last_send"] - r["ue_first_send"],
                     E_last_send_to_reassembly=r["edge_reassembly_complete"] - r["ue_last_send"],
                     F_reassembly_to_prediction_materialized=r["prediction_materialized"] - r["edge_reassembly_complete"],
                     G_prediction_materialized_to_ue_receipt=r["ue_receipt"] - r["prediction_materialized"])
        telescopes &= abs(sum(parts.values()) - r["ue_receipt"]) < 1e-6
        for k, v in parts.items():
            seg[k].append(v)
        # Same live reply measured on the training composition's boundaries: actor reserve instead of A,
        # no seven-channel input (B) and no reassembly->prediction (F).
        on_training_clock.append(r["ue_receipt"] - parts["A_action_open_to_input_start"] + ACTOR_RESERVE_MS
                                 - parts["B_seven_channel_input"] - parts["F_reassembly_to_prediction_materialized"])
        uplink[r["frame_id"]] = parts["E_last_send_to_reassembly"]
        payload.append(inputs[r["frame_id"]]["wire"])
        receipt.append(r["ue_receipt"])
    tv, tv_rows = transport_eval(model, inputs, uplink, pool)
    families = {}
    for r in rows:
        fam = r["action"]["family"] + "/" + r["action"]["quantizer"]
        families.setdefault(fam, []).append(r)
    by_family = {fam: dict(n=len(v), ue_receipt_p50=q([x["ue_receipt"] for x in v]).get("p50"),
                           ue_last_send_p50=q([x["ue_last_send"] for x in v]).get("p50"),
                           reassembly_to_prediction_p50=q([x["prediction_materialized"] - x["edge_reassembly_complete"]
                                                           for x in v]).get("p50"),
                           timely=sum(1 for x in v if x["ue_receipt"] <= DEADLINE_MS))
                 for fam, v in sorted(families.items())}
    scenario = {}
    min_wire = int(model.fit_support["min_wire_bytes"])
    for label, wire in (("fit_support_min_wire", min_wire), ("smallest_live100_payload", 96_627)):
        vals = []
        for r in rows:
            x = inputs[r["frame_id"]]
            t = model.predict(pre_enqueue_backlog_bytes=x["backlog"], wire_bytes=wire, prior_ul_mcs=x["mcs"],
                              allow_out_of_support=True).conditional_latency_ms
            vals.append(r["ue_receipt"] - uplink[r["frame_id"]] + t)
        scenario[label] = dict(wire_bytes=wire, receipt_ms=q(vals))
    scenario["assumption"] = ("SCENARIO, not a prediction: only the last-send->reassembly term is replaced by the "
                              "transport model at the given wire size; every other stage keeps its measured per-frame "
                              "value. Action invariance of codec, edge compute and post-prediction stages is NOT "
                              "verified, and the quality of such actions is not considered.")
    omitted = [b + f for b, f in zip(seg["B_seven_channel_input"], seg["F_reassembly_to_prediction_materialized"])]
    rho_up = ss.spearmanr(payload, [uplink[r["frame_id"]] for r in rows])
    rho_rx = ss.spearmanr(payload, receipt)
    return dict(decisions=len(rows), telescopes_exactly=bool(telescopes),
                segments_ms={k: q(v) for k, v in seg.items()}, ue_receipt_ms=q(receipt),
                omitted_terms_B_plus_F_ms=q(omitted),
                omitted_terms_A_minus_reserve_plus_B_plus_F_ms=q(
                    [a - ACTOR_RESERVE_MS + b for a, b in zip(seg["A_action_open_to_input_start"], omitted)]),
                receipt_minus_uplink_ms=q([r["ue_receipt"] - uplink[r["frame_id"]] for r in rows]),
                by_family_descriptive=by_family, uplink_only_action_scenario=scenario,
                live_reply_on_training_boundaries_ms=q(on_training_clock),
                transport_model_on_live=tv, state=state_summary(ev, inputs),
                payload_vs_uplink_spearman=dict(rho=round(float(rho_up.correlation), 4), p=float(rho_up.pvalue)),
                payload_vs_receipt_spearman=dict(rho=round(float(rho_rx.correlation), 4), p=float(rho_rx.pvalue)),
                wire_source=sorted({x["wire_source"] for x in inputs.values()})), tv_rows


# ---------------------------------------------------------------- legacy Run-5 capture (2026-10-02), measured chain
def run5_capture(model, pool):
    chain = {int(r["frame_id"]): {k: fnum(v) for k, v in r.items()} for r in csv.DictReader(open(INPUTS["run5cap_legacy_chain"]))}
    ev, inputs = decision_inputs(INPUTS["run5cap_evidence"], INPUTS["run5cap_per_frame"])
    seg = {k: [] for k in ("UE_action_open_to_last_send", "E_last_send_to_reassembly",
                           "F_reassembly_to_evaluator_submit", "G_evaluator_submit_to_ue_receipt",
                           "F2_reassembly_to_predictions_cpu", "G2_predictions_cpu_to_ue_receipt")}
    uplink, on_training_clock, receipt = {}, [], []
    for fid, c in chain.items():
        if None in (c["last_send"], c["reassembly_complete"], c["evaluator_submit"], c["ue_receipt"]):
            continue
        seg["UE_action_open_to_last_send"].append(c["last_send"])
        seg["E_last_send_to_reassembly"].append(c["reassembly_complete"] - c["last_send"])
        seg["F_reassembly_to_evaluator_submit"].append(c["evaluator_submit"] - c["reassembly_complete"])
        seg["G_evaluator_submit_to_ue_receipt"].append(c["ue_receipt"] - c["evaluator_submit"])
        if c.get("predictions_cpu") is not None:
            seg["F2_reassembly_to_predictions_cpu"].append(c["predictions_cpu"] - c["reassembly_complete"])
            seg["G2_predictions_cpu_to_ue_receipt"].append(c["ue_receipt"] - c["predictions_cpu"])
        uplink[fid] = c["reassembly_complete"] - c["last_send"]
        receipt.append(c["ue_receipt"])
        if c.get("ue_input_7ch_ms") is not None:
            on_training_clock.append(c["ue_receipt"] - c["ue_input_7ch_ms"] - (c["evaluator_submit"] - c["reassembly_complete"]))
    tv, _ = transport_eval(model, inputs, uplink, pool)
    return dict(decisions_joined=len(receipt), segments_ms={k: q(v) for k, v in seg.items()}, ue_receipt_ms=q(receipt),
                seven_channel_input_ms=q([c.get("ue_input_7ch_ms") for c in chain.values()]),
                live_reply_on_training_boundaries_ms=q(on_training_clock),
                live_reply_on_training_boundaries_note="ue_receipt - ue.input_7ch span - (reassembly->evaluator_submit); "
                                                       "action-open->input start not removed (not in LEGACY_CHAIN)",
                transport_model_on_live=tv, state=state_summary(ev, inputs))


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {k: sha(v) for k, v in INPUTS.items()}
    pins = {k: hashes[k] == v for k, v in EXPECTED_SHA.items()}
    if not all(pins.values()):
        raise SystemExit(f"training pin mismatch: {pins}")
    from rl_agent.ue_production_transport_model_v2 import collector_v1 as CV
    pool = np.asarray(CV.load_retained_residuals(ROOT), float) / 1e6
    model = load_transport()
    probe, probe_rows = probe_side()
    if probe["rows"] != pool.size:
        raise SystemExit(f"probe row count {probe['rows']} != residual pool {pool.size}")
    if not np.allclose(np.sort([r["residual"] for r in probe_rows]), np.sort(pool), atol=1e-6):
        raise SystemExit("probe residual reconstruction differs from collector_v1 pool")
    l100, l100_rows = live100(model, pool)
    support = model.fit_support
    leverage = {}
    for wire in (int(support["min_wire_bytes"]), 96_627, 312_630, 406_912, int(support["max_wire_bytes"])):
        for mcs in (26, 20, 12):
            p = model.predict(pre_enqueue_backlog_bytes=0.0, wire_bytes=wire, prior_ul_mcs=mcs)
            leverage[f"wire{wire}_mcs{mcs}"] = dict(T_v2b_ms=round(p.conditional_latency_ms, 3),
                                                     on_time_probability=round(p.on_time_probability, 6))
    doc = dict(schema="splitfusion.training_live_mismatch.calculations.v1", deadline_ms=DEADLINE_MS,
               quantiles="numpy linear; le_170 counts inclusive", training_pins_verified=pins,
               residual_pool=dict(n=int(pool.size), stats=q(pool), over_170=int((pool > DEADLINE_MS).sum()),
                                  model_transport_artifact_sha256=model.document_sha256),
               transport_model_uplink_leverage_backlog0=leverage,
               training=training_side(), probe=probe, live100=l100, run5_capture=run5_capture(model, pool))
    (out / "CALCULATIONS.json").open("x").write(json.dumps(doc, indent=1, sort_keys=True))
    with (out / "LIVE100_TRANSPORT_MODEL_ROWS.csv").open("x", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(l100_rows[0]))
        w.writeheader()
        w.writerows(l100_rows)
    (out / "INPUT_SHA256.json").open("x").write(json.dumps(
        {k: dict(path=str(INPUTS[k]), sha256=h) for k, h in hashes.items()}, indent=1, sort_keys=True))
    print(json.dumps(doc, indent=1, sort_keys=True))


def verify(sealed):
    """Recompute into a fresh temporary directory and byte-compare every output with the sealed directory."""
    import tempfile
    sealed = Path(sealed)
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "out"
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            run(fresh)
        names = ("CALCULATIONS.json", "LIVE100_TRANSPORT_MODEL_ROWS.csv", "INPUT_SHA256.json")
        result = {n: (fresh / n).read_bytes() == (sealed / n).read_bytes() for n in names}
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if all(result.values()) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--verify")
    args = parser.parse_args()
    verify(args.verify) if args.verify else run(args.output)
