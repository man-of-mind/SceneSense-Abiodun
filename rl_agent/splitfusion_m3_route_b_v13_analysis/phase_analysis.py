"""Reproducible whole-route and phase analysis for the sole M3.3 V13 capture.

This is presentation/postflight analysis only; it is not part of the registered runtime
inventory and cannot change the registered verdict.  Decision phases follow the causal,
ACKed commanded-SNR value observed by the actor.  Frame phases follow the recorded
route-crossing frame.  Radio phases split at the first ACKed BAD command.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
from . import analyze as A
from . import telemetry as T

PHASES = ("GOOD", "BAD")
TARGET = {"GOOD": 23.5, "BAD": 8.5}
UL = re.compile(r"ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+), ulsch_DTX (\d+)")


def load(path):
    return json.loads(Path(path).read_bytes())


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value):
    return str(value).lower() in ("1", "true")


def f(value):
    try:
        return float(value) if value not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def stats(values, ps=(50, 95, 99)):
    a = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(a):
        return dict(n=0, **{f"p{x}": None for x in ps})
    return dict(n=int(len(a)), **{f"p{x}": float(np.percentile(a, x)) for x in ps},
                min=float(a.min()), max=float(a.max()), mean=float(a.mean()))


def ratio(a, b):
    return None if not b else a / b


def decision_phase(row):
    value = float(row["snr_proxy_db"])
    if value == TARGET["GOOD"]:
        return "GOOD"
    if value == TARGET["BAD"]:
        return "BAD"
    raise ValueError(f"unregistered actor SNR value {value}")


def quality(raw_rows, timeline):
    scored = [r for r in raw_rows if r["service_outcome"] == "SCORED"]
    defined = [r for r in scored if r["quality_defined"]]
    timely = [r for r in raw_rows if r["quality_defined"] and r["timely"]]
    result = dict(scored=len(scored), defined=len(defined), undefined=len(scored) - len(defined), classes={})
    for cls in ("vehicle", "person"):
        pooled = A.pooled(defined, cls)
        timely_pooled = A.pooled(timely, cls)
        result["classes"][cls] = {k: v for k, v in pooled.items() if k != "errors"}
        result["classes"][cls]["deadline_aware_within"] = timely_pooled["within_tolerance"]
        result["classes"][cls]["deadline_aware_recall"] = ratio(
            timely_pooled["within_tolerance"], pooled["eligible_gt"])
        result["classes"][cls]["deadline_misses_eligible_gt"] = (
            pooled["eligible_gt"] - timely_pooled["eligible_gt"])
    result["gt_unavailable_decisions"] = sum(not truth(r["quality_defined"]) for r in timeline)
    return result


def phase_decisions(timeline, raw_rows, phase):
    indices = [i for i, row in enumerate(timeline) if decision_phase(row) == phase]
    selected_t = [timeline[i] for i in indices]
    selected_r = [raw_rows[i] for i in indices]
    prediction = [f(r["prediction_completion_ms"]) for r in selected_t]
    feedback = [f(r["feedback_receipt_ms"]) for r in selected_t]
    return dict(
        decisions=len(indices), prediction_completed=sum(x is not None for x in prediction),
        prediction_completion_ms=stats(prediction), prediction_by_170=sum(x is not None and x <= 170 for x in prediction),
        feedback_received=sum(x is not None for x in feedback), feedback_receipt_ms=stats(feedback),
        feedback_by_170=sum(x is not None and x <= 170 for x in feedback),
        feedback_by_170_fraction=ratio(sum(x is not None and x <= 170 for x in feedback), len(indices)),
        outcome_classes=dict(collections.Counter(r["outcome_class"] for r in selected_t)),
        gt_published_ms=stats([f(r["gt_published_ms"]) for r in selected_t]),
        scoring_end_ms=stats([f(r["scoring_end_ms"]) for r in selected_t]),
        payload_decision_bytes=stats([f(r["payload_l2_bytes"]) for r in selected_t]),
        modes={str(k): v for k, v in sorted(collections.Counter(int(r["mode_id"]) for r in selected_t).items())},
        q_e4=stats([f(r["q_e4"]) for r in selected_t]),
        pre_action_rlc_backlog_bytes=stats([f(r["pre_action_rlc_backlog_bytes"]) for r in selected_t]),
        backlog_nonzero=sum((f(r["pre_action_rlc_backlog_bytes"]) or 0) > 0 for r in selected_t),
        quality=quality(selected_r, selected_t), indices=indices)


def frame_metrics(cell, analysis, crossing_frame, bad_ack_wall_ns):
    per_frame = csv_rows(cell / "per_frame_metrics.csv")
    frames = csv_rows(analysis / "FRAME_MAP_PAYLOAD.csv")
    phase6 = load(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json")
    aoi_rows = csv_rows(analysis / "MAP_AOI_TIMELINE.csv")
    result = {}
    for phase in PHASES:
        predicate = (lambda fid: fid < crossing_frame) if phase == "GOOD" else (lambda fid: fid >= crossing_frame)
        offers = [r for r in per_frame if predicate(int(r["frame_id"]))]
        frows = [r for r in frames if predicate(int(r["frame_id"]))]
        actions = [r for r in phase6["decisions"] if predicate(int(r["frame_id"]))]
        sent = [r for r in frows if r["prepare_status"] == "SENT"]
        if phase == "GOOD":
            aoi = [f(r["aoi_ms"]) for r in aoi_rows if int(float(r["wall_s"]) * 1e9) < bad_ack_wall_ns]
        else:
            aoi = [f(r["aoi_ms"]) for r in aoi_rows if int(float(r["wall_s"]) * 1e9) >= bad_ack_wall_ns]
        result[phase] = dict(
            offered=len(offers), prepare_status=dict(collections.Counter(r["prepare_status"] for r in offers)),
            sent=len(sent), map_installed=sum(r["map_outcome"] == "RESULT_INSTALLED" for r in sent),
            holds=sum(r["kind"] == "POLICY_HOLD" for r in actions),
            fallbacks=sum(r["kind"] == "FALLBACK" for r in actions),
            fallback_reasons=dict(collections.Counter(reason.split(":")[0] for r in actions
                                                       if r["kind"] == "FALLBACK" for reason in r.get("fallback_reasons", []))),
            payload_l2_bytes=stats([f(r["payload_l2_bytes"]) for r in sent]),
            payload_l2_total_bytes=int(sum(f(r["payload_l2_bytes"]) or 0 for r in sent)),
            map_age_at_install_ms=stats([f(r["map_age_at_install_ms"]) for r in sent]),
            map_aoi_ms=stats(aoi),
            map_update_gap_ms=stats(np.diff(sorted(f(r["install_wall_s"]) for r in sent
                                                   if f(r["install_wall_s"]) is not None)) * 1000))
    return result


def radio_metrics(capture, analysis, bad_ack_wall_ns):
    timeline = csv_rows(analysis / "DECISION_TIMELINE.csv")
    start = int(min(float(r["capture_wall_s"]) for r in timeline) * 1e9)
    end = int(max(float(r["capture_wall_s"]) for r in timeline) * 1e9)
    streams = T.load(capture / "capture/m3_radio/ttracer")
    mac = []
    for line in (capture / "capture/m3_radio/gnb_stats_observed.jsonl").read_text().splitlines():
        row = json.loads(line)
        match = UL.search(row["line"])
        if match and start <= int(row["observed_wall_ns"]) <= end:
            mac.append((int(row["observed_wall_ns"]), [int(match.group(i)) for i in range(1, 7)]))
    out = {}
    for phase in PHASES:
        lo, hi = (start, bad_ack_wall_ns) if phase == "GOOD" else (bad_ack_wall_ns, end + 1)
        pusch = [r for r in streams["pusch"]["rows"] if lo <= r["wall_ns"] < hi]
        mcs = [r for r in streams["mcs"]["rows"] if lo <= r["wall_ns"] < hi]
        counters = [v for when, v in mac if lo <= when < hi]
        delta = [b - a for a, b in zip(counters[0], counters[-1])] if len(counters) >= 2 else None
        out[phase] = dict(
            measured_pusch_snr_db=dict(stats([r["pusch_snr_db"] for r in pusch]),
                                        p05=(None if not pusch else float(np.percentile([r["pusch_snr_db"] for r in pusch], 5)))),
            selected_mcs=stats([r["selected_mcs"] for r in mcs]), final_mcs=stats([r["final_mcs"] for r in mcs]),
            grants=len(mcs), grants_with_reported_data=sum(r["estimated_ul_buffer"] > 0 for r in mcs),
            final_prb=stats([r["rb_size_final"] for r in mcs]),
            mac_log=dict(samples=len(counters), rounds_delta=None if delta is None else delta[:4],
                         retransmissions=None if delta is None else sum(delta[1:4]),
                         errors=None if delta is None else delta[4], dtx=None if delta is None else delta[5]))
    return out


def accounting(capture, first_bad_decision):
    doc = load(capture / "gt_gc_trace/storage_accounting_timing.json")
    width = len(doc["caller_names"])
    assert len(doc["cpu_ns"]) % width == 0
    cpu = np.asarray(doc["cpu_ns"], dtype=float).reshape((-1, width)) / 1e6
    assert cpu.shape[0] >= first_bad_decision
    return dict(record={k: doc[k] for k in ("calls", "recorded", "dropped", "errors", "foreign_frames", "capacity", "caller_names")},
                GOOD=stats(cpu[:first_bad_decision].sum(axis=1)), BAD=stats(cpu[first_bad_decision:].sum(axis=1)))


def scene(layers, phases):
    rows = csv_rows(layers / "ROUTE_LAYERS_JOINED.csv")
    out = {}
    for phase in PHASES:
        selected = [r for r in rows if (float(r["snr_proxy_db"]) == TARGET[phase])]
        out[phase] = dict(camera_si=stats([f(r["camera_si"]) for r in selected]),
                          radar_p40=stats([f(r["radar_p40"]) for r in selected]))
    return out


def corresponding_halves(path):
    rows = csv_rows(Path(path) / "DECISION_TIMELINE.csv")
    def decile(row):
        return int(str(row["decile"]).rsplit("_", 1)[-1])
    out = {}
    for label, keep in (("FIRST_HALF", lambda r: decile(r) < 5),
                        ("SECOND_HALF", lambda r: decile(r) >= 5)):
        selected = [r for r in rows if r.get("decile") not in ("", None) and keep(r)]
        out[label] = dict(decisions=len(selected), timely_fraction=ratio(sum(truth(r["timely_170"]) for r in selected), len(selected)),
                          prediction_completion_ms=stats([f(r["prediction_completion_ms"]) for r in selected]),
                          feedback_receipt_ms=stats([f(r["feedback_receipt_ms"]) for r in selected]),
                          payload_l2_bytes=stats([f(r["payload_l2_bytes"]) for r in selected]),
                          vehicle_eligible=sum(int(float(r["vehicle_eligible"] or 0)) for r in selected),
                          vehicle_within=sum(int(float(r["vehicle_within_1p0"] or 0)) for r in selected),
                          person_eligible=sum(int(float(r["person_eligible"] or 0)) for r in selected),
                          person_within=sum(int(float(r["person_within_1p2"] or 0)) for r in selected))
        for cls in ("vehicle", "person"):
            out[label][f"{cls}_recall"] = ratio(out[label][f"{cls}_within"], out[label][f"{cls}_eligible"])
    return out


def plots(out, phases, comparison, transition, timeline, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"GOOD": "#147ABB", "BAD": "#D86624"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    values = [
        ("Feedback ≤170 ms", [phases[p]["decision"]["feedback_by_170_fraction"] for p in PHASES], "fraction"),
        ("Prediction P50", [phases[p]["decision"]["prediction_completion_ms"]["p50"] for p in PHASES], "ms"),
        ("Feedback P50", [phases[p]["decision"]["feedback_receipt_ms"]["p50"] for p in PHASES], "ms"),
        ("Payload P50", [phases[p]["frame"]["payload_l2_bytes"]["p50"] / 1000 for p in PHASES], "kB"),
        ("Vehicle recall ≤1.0 m", [phases[p]["decision"]["quality"]["classes"]["vehicle"]["recall_within_tolerance"] for p in PHASES], "fraction"),
        ("Pedestrian recall ≤1.2 m", [phases[p]["decision"]["quality"]["classes"]["person"]["recall_within_tolerance"] for p in PHASES], "fraction")]
    for ax, (title, vals, unit) in zip(axes.flat, values):
        bars = ax.bar(PHASES, vals, color=[colors[p] for p in PHASES])
        ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="y", alpha=.25)
        if unit == "fraction": ax.set_ylim(0, 1)
        for bar, value in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.3g}", ha="center", va="bottom")
    fig.suptitle("M3.3 GOOD_TO_BAD — full Town10 map phase summary\nDifferent route sections; descriptive, not a paired channel-only comparison")
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out / "01_phase_summary.png", dpi=180); plt.close(fig)

    joined = csv_rows(layers / "ROUTE_LAYERS_JOINED.csv")
    t = np.asarray([float(r["t_since_first_capture_s"]) for r in joined])
    centre = float(joined[transition["first_actor_bad_index"]]["t_since_first_capture_s"])
    mask = (t >= centre - 25) & (t <= centre + 25)
    fig, axes = plt.subplots(5, 1, figsize=(15, 10), sharex=True)
    fields = (("snr_proxy_db", "Commanded SNR (dB)"), ("measured_pusch_snr_db", "Measured PUSCH (dB)"),
              ("mode_id", "Mode"), ("q_e4", "q ×10⁴"), ("application_payload_bytes", "Payload (B)"))
    for ax, (key, label) in zip(axes, fields):
        y = np.asarray([float(r[key]) for r in joined]); ax.plot(t[mask], y[mask], lw=1); ax.set_ylabel(label); ax.grid(alpha=.25)
        ax.axvline(centre, color="#B78213", ls="--", label="first actor BAD")
    axes[0].legend(); axes[-1].set_xlabel("seconds since first decision capture")
    fig.suptitle("M3.3 actual transition — commanded target distinguished from measured PUSCH SNR")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out / "02_transition_timeline.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sessions = ("V11 GOOD_ALL", "V12 BAD_ALL", "V13 GOOD_TO_BAD")
    for ax, half in zip(axes[:2], ("FIRST_HALF", "SECOND_HALF")):
        vals = [comparison[s][half]["timely_fraction"] for s in sessions]
        ax.bar(range(3), vals, color=["#147ABB", "#D86624", "#078778"]); ax.set_xticks(range(3), ["V11", "V12", "V13"])
        ax.set_ylim(0, 1); ax.set_title(half.replace("_", " ") + " feedback ≤170 ms"); ax.grid(axis="y", alpha=.25)
    vals = [comparison[s]["SECOND_HALF"]["payload_l2_bytes"]["p50"] / 1000 for s in sessions]
    axes[2].bar(range(3), vals, color=["#147ABB", "#D86624", "#078778"]); axes[2].set_xticks(range(3), ["V11", "V12", "V13"])
    axes[2].set_title("Second-half payload P50"); axes[2].set_ylabel("kB"); axes[2].grid(axis="y", alpha=.25)
    fig.suptitle("Corresponding route halves across different sessions/scenes — descriptive only")
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "03_session_half_comparison.png", dpi=180); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--layers", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--v11-analysis", type=Path, required=True)
    parser.add_argument("--v12-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    cells = sorted((args.capture / "capture/cells").glob("*")); assert len(cells) == 1
    cell = cells[0]
    timeline = csv_rows(args.analysis / "DECISION_TIMELINE.csv")
    _, raw_rows, _ = LIVE.rows(args.capture)
    assert len(timeline) == len(raw_rows)
    assert [int(r["frame_id"]) for r in timeline] == [int(r["frame_id"]) for r in raw_rows]
    transition_rows = [json.loads(x) for x in (cell / "run5_policy/ROUTE_CHANNEL_TRANSITION.jsonl").read_text().splitlines() if x.strip()]
    crossings = [r for r in transition_rows if r.get("threshold_crossed_now") is True]; assert len(crossings) == 1
    crossing = crossings[0]; crossing_frame = int(crossing["source_frame_id"])
    radio = csv_rows(cell / "radio_trace.csv")
    first_bad_radio_index = next(i for i, r in enumerate(radio) if float(r["target_snr_db"]) == TARGET["BAD"])
    first_bad = radio[first_bad_radio_index]; bad_ack_wall_ns = int(first_bad["command_ack_wall_ns"])
    first_bad_decision = next(i for i, r in enumerate(timeline) if decision_phase(r) == "BAD")
    decision = {p: phase_decisions(timeline, raw_rows, p) for p in PHASES}
    frame = frame_metrics(cell, args.analysis, crossing_frame, bad_ack_wall_ns)
    radio_by_phase = radio_metrics(args.capture, args.analysis, bad_ack_wall_ns)
    scenes = scene(args.layers, decision)
    phases = {p: dict(decision={k: v for k, v in decision[p].items() if k != "indices"},
                              frame=frame[p], radio=radio_by_phase[p], scene=scenes[p]) for p in PHASES}
    assessment = load(args.assessment)
    progress = load(cell / "m3_route/ROUTE_PROGRESS_SUMMARY.json")
    trace = [json.loads(x) for x in (cell / "m3_route/ROUTE_PROGRESS_TRACE.jsonl").read_text().splitlines() if x.strip()]
    actor_bad = timeline[first_bad_decision]
    transition = dict(crossing=crossing, first_bad_radio_index=first_bad_radio_index,
                      command_issue_delay_ms=(int(first_bad["command_send_monotonic_ns"]) - int(crossing["crossing_monotonic_ns"])) / 1e6,
                      command_ack_delay_ms=(int(first_bad["command_ack_monotonic_ns"]) - int(crossing["crossing_monotonic_ns"])) / 1e6,
                      command_ack_latency_ms=float(first_bad["command_latency_ms"]), first_bad_radio_row=first_bad,
                      first_actor_bad_index=first_bad_decision, first_actor_bad_frame=int(actor_bad["frame_id"]),
                      first_actor_bad_elapsed_s=float(actor_bad["t_since_first_capture_s"]),
                      actor_values_exactly_one_switch=[decision_phase(r) for r in timeline] == ["GOOD"] * first_bad_decision + ["BAD"] * (len(timeline) - first_bad_decision))
    audit_checks = dict(summary_has_crossing_object=progress.get("crossing") is not None,
                        summary_crossing_at_or_above_threshold=float(progress["crossing"]["cumulative_distance_m"]) >= float(progress["threshold_distance_m"]),
                        terminal_trace_threshold_crossed=trace[-1].get("threshold_crossed") is True,
                        transition_trace_exactly_once=len(crossings) == 1,
                        no_bad_to_good_reversion=transition["actor_values_exactly_one_switch"],
                        command_issue_within_registered_delay=0 <= transition["command_issue_delay_ms"] <= 1100,
                        summary_omits_checker_redundant_field="threshold_crossed" not in progress)
    assessment_reasons = assessment.get("reasons", assessment["complete_assessment"].get("reasons", []))
    supplemental = dict(status="NON_AUTHORITATIVE_TRACE_AUDIT_PASS" if all(audit_checks.values()) else "NON_AUTHORITATIVE_TRACE_AUDIT_FAIL",
                        registered_verdict=assessment["verdict"], registered_reasons=assessment_reasons,
                        changes_registered_verdict=False, checks=audit_checks,
                        defect="sealed writer records crossing object but omits the redundant summary.threshold_crossed key required by the sealed checker; terminal trace records threshold_crossed=true")
    comparison = {"V11 GOOD_ALL": corresponding_halves(args.v11_analysis),
                  "V12 BAD_ALL": corresponding_halves(args.v12_analysis),
                  "V13 GOOD_TO_BAD": corresponding_halves(args.analysis)}
    document = dict(schema="splitfusion.m3_route_b_v13.phase_analysis.v1",
                    definitions=dict(decision_phase="actor-observed ACKed commanded SNR", frame_phase="actual route crossing frame",
                                     radio_phase="first ACKed BAD command wall time", cross_session="different sessions/scenes; descriptive only"),
                    registered_verdict=dict(verdict=assessment["verdict"], reasons=assessment_reasons, checks=assessment["complete_assessment"]["checks"]),
                    supplemental_route_progress_audit=supplemental, transition=transition, route_progress_summary=progress,
                    whole_route=load(args.analysis / "METRICS.json"), phases=phases,
                    accounting=accounting(args.capture, first_bad_decision),
                    gc_safety=load(args.capture / "gt_gc_trace/prep_gc_trace.json").get("safety_stop"),
                    corresponding_route_halves=comparison,
                    inputs_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                   (args.analysis / "DECISION_TIMELINE.csv", args.analysis / "METRICS.json", args.layers / "ROUTE_LAYERS_JOINED.csv",
                                    args.assessment, cell / "m3_route/ROUTE_PROGRESS_SUMMARY.json", cell / "m3_route/ROUTE_PROGRESS_TRACE.jsonl",
                                    cell / "run5_policy/ROUTE_CHANNEL_TRANSITION.jsonl", cell / "radio_trace.csv")})
    (args.out / "PHASE_METRICS.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    flat = []
    for phase in PHASES:
        d, fr, rr = phases[phase]["decision"], phases[phase]["frame"], phases[phase]["radio"]
        flat.append(dict(phase=phase, decisions=d["decisions"], holds=fr["holds"], fallbacks=fr["fallbacks"],
                         offered=fr["offered"], sent=fr["sent"], completed=d["prediction_completed"], map_installs=fr["map_installed"],
                         prediction_p50_ms=d["prediction_completion_ms"]["p50"], prediction_p99_ms=d["prediction_completion_ms"]["p99"],
                         prediction_by_170=d["prediction_by_170"], feedback_p50_ms=d["feedback_receipt_ms"]["p50"],
                         feedback_p99_ms=d["feedback_receipt_ms"]["p99"], feedback_by_170=d["feedback_by_170"],
                         feedback_by_170_fraction=d["feedback_by_170_fraction"], payload_p50_bytes=fr["payload_l2_bytes"]["p50"],
                         payload_p99_bytes=fr["payload_l2_bytes"]["p99"], pusch_p50_db=rr["measured_pusch_snr_db"]["p50"],
                         pusch_p05_db=rr["measured_pusch_snr_db"]["p05"], selected_mcs_p50=rr["selected_mcs"]["p50"],
                         q_e4_p50=d["q_e4"]["p50"], map_age_p50_ms=fr["map_age_at_install_ms"]["p50"],
                         map_age_p99_ms=fr["map_age_at_install_ms"]["p99"], map_aoi_p50_ms=fr["map_aoi_ms"]["p50"],
                         map_aoi_p99_ms=fr["map_aoi_ms"]["p99"], map_gap_p99_ms=fr["map_update_gap_ms"]["p99"],
                         vehicle_eligible=d["quality"]["classes"]["vehicle"]["eligible_gt"],
                         vehicle_recall=d["quality"]["classes"]["vehicle"]["recall_within_tolerance"],
                         vehicle_deadline_recall=d["quality"]["classes"]["vehicle"]["deadline_aware_recall"],
                         vehicle_precision=d["quality"]["classes"]["vehicle"]["precision_3m_gate"],
                         vehicle_seg_iou=d["quality"]["classes"]["vehicle"]["seg_iou_pooled"],
                         person_eligible=d["quality"]["classes"]["person"]["eligible_gt"],
                         person_recall=d["quality"]["classes"]["person"]["recall_within_tolerance"],
                         person_deadline_recall=d["quality"]["classes"]["person"]["deadline_aware_recall"],
                         person_precision=d["quality"]["classes"]["person"]["precision_3m_gate"],
                         person_seg_iou=d["quality"]["classes"]["person"]["seg_iou_pooled"],
                         gt_published_p50_ms=d["gt_published_ms"]["p50"],
                         accounting_p50_ms=document["accounting"][phase]["p50"],
                         gt_defined=d["quality"]["defined"], gt_undefined=d["quality"]["undefined"]))
    with (args.out / "PHASE_METRICS.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    plots(args.out, phases, comparison, transition, timeline, args.layers)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status="PASS", registered_verdict=assessment["verdict"], supplemental=supplemental["status"], phases=flat), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
