"""Offline per-decision diagnosis of single-UE policy-feedback latency (CPU only, read-only on captures).

For each capture, reuses the retained loader (splitfusion_run5_full_live_analysis_v1.report.rows -> fixed-action
screen rows_for) and adds what it does not join: the edge pre-send emit span, the kernel arrival of the exact
feedback datagram on oaitun_ue1 (pcap, joined by SHA-256 of the UDP payload = edge feedback_sha256), the UE-side DL
DCI that carried it (UE T-tracer), the UE GT-freeze events, GT worker GC events, GT queue depth and overlapping UE
spans. All instants are ms after the UE action-open (CLOCK_MONOTONIC_RAW), the origin of the 170-ms deadline.

The stage partition telescopes exactly to the UE receipt of each decision (verified); means are additive across
stages, medians/percentiles are not and are reported only per stage. Nothing is clamped or dropped.

Usage:
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -B -m \
    rl_agent.splitfusion_single_ue_feedback_latency_v1.diagnose \
    --capture V11_GOOD_ALL=<root>=<METRICS.json> --capture V20_DEMO_03=<root>=<METRICS.json> --out <new dir>
"""
import argparse
import bisect
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
from . import sources as SRC

DEADLINE_MS = 170.0
EXTREME_MS = 200.0
SPIKE_EXCESS_MS = 20.0          # a single stage exceeding its on-time median by >= this is called the spike stage
UE_FREEZE_STALL_MS = 60.0
GROUPS = ("ON_TIME", "LATE_170_200", "EXTREME_GT200", "MISSING")

# (column, label, start instant, end instant) -- consecutive, so the sum telescopes to ue_receipt
STAGES = (
    ("st_ue_actor", "UE state+actor", None, "actor_end"),
    ("st_ue_pre_input", "UE wait before 7-ch input", "actor_end", "ue_input_start"),
    ("st_ue_input", "UE 7-ch input build", "ue_input_start", "ue_input_end"),
    ("st_ue_encode_send", "UE encode+send", "ue_input_end", "ue_last_send"),
    ("st_uplink", "uplink (UE stack, UL radio, core, edge rx)", "ue_last_send", "edge_reassembly_complete"),
    ("st_edge", "edge queue+compute", "edge_reassembly_complete", "edge_compute_end"),
    ("st_pred_to_service", "prediction edge->GT service", "edge_compute_end", "service_prediction_received"),
    ("st_gt_extra_wait", "extra wait for GT publication", "service_prediction_received", "inputs_both_ready"),
    ("st_score_wait", "score-channel wait after both inputs", "inputs_both_ready", "scoring_op_begin"),
    ("st_score_exec", "scoring op", "scoring_op_begin", "scoring_worker_end"),
    ("st_result_return", "result service->edge", "scoring_worker_end", "edge_result_received"),
    ("st_feedback_prep", "edge feedback build (pre-send)", "edge_result_received", "edge_emit_span_start"),
    ("st_send_to_tun", "edge send -> UE tunnel (core+gNB DL+air+UE stack)", "edge_emit_span_start", "ue_tun_arrival"),
    ("st_tun_to_app", "UE tunnel -> app recv (kernel+thread)", "ue_tun_arrival", "ue_receipt"),
)


def pct(values, ps=(50, 95, 99)):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if not len(v):
        return dict(n=0)
    out = dict(n=int(len(v)), mean=round(float(v.mean()), 3))
    for p in ps:
        if p == 99 and len(v) < 100:
            continue                                   # P99 only on populations of at least 100
        out[f"p{p}"] = round(float(np.percentile(v, p)), 3)
    out["max"] = round(float(v.max()), 3)
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10:
        return None
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def group_of(receipt):
    if receipt is None:
        return "MISSING"
    return "ON_TIME" if receipt <= DEADLINE_MS else ("LATE_170_200" if receipt <= EXTREME_MS else "EXTREME_GT200")


# --------------------------------------------------------------------------- one capture
def build(label, root, metrics_path):
    root = Path(root).resolve(strict=True)
    arm, rows, _ = LIVE.rows(root)
    cell = Path(arm.cell)
    expect = json.loads(Path(metrics_path).read_bytes())["latency_ms"]
    reproduced = dict(decisions=len(rows), timely_170=sum(bool(r["timely"]) for r in rows),
                      prediction_completed_by_170=sum(r["edge_compute_end"] is not None and r["edge_compute_end"] <= DEADLINE_MS
                                                      for r in rows))
    recorded = dict(decisions=expect["denominator_decisions"], timely_170=expect["timely_170"],
                    prediction_completed_by_170=expect["prediction_completed_by_170"])
    if reproduced != recorded:
        raise SystemExit(f"{label}: recorded counts not reproduced: {reproduced} != {recorded}")

    frames = {f["frame_id"]: f for f in SRC.load_jsonl(cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")}
    edge_state = json.loads((cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json").read_bytes())["state"]
    edge_frames = {int(k): v for k, v in edge_state["frames"].items()}
    edge_records = {int(r["frame_id"]): r for r in edge_state["records"]}
    emit_spans, _ = SRC.profile_spans(cell / "profiler/edge/edge.profile.jsonl", {"edge.feedback_emit"})
    emit_span = {}
    for s, e, _, fid, _ in emit_spans:
        emit_span.setdefault(int(fid), (s, e))
    pcap, pcap_info = SRC.pcap_payloads(cell / "quality_ack_oaitun_ue1.pcap")
    clock = SRC.WallToRaw([root / "capture/m3_radio/ttracer/live_mcs_with_ingest.csv",
                           root / "capture/m3_radio/ttracer/live_pusch_with_ingest.csv"])
    dl, t_counts = SRC.ue_dl_dcis(cell / "ttracer/ue/ue.raw")
    dl_raw = clock(dl[:, 0])
    order = np.argsort(dl_raw)
    dl, dl_raw = dl[order], dl_raw[order]
    ue_spans, _ = SRC.profile_spans(cell / "profiler/ue.profile.jsonl")
    ticks = SRC.Intervals([x for x in ue_spans if x[2] == "carla.world_tick"])
    ue_all = SRC.Intervals([x for x in ue_spans if x[2] not in ("ue.worker_frame", "ue.submit")])
    gc_prep = SRC.gc_events(root / "gt_gc_trace/prep_gc_trace.json") or []
    gc_score = SRC.gc_events(root / "gt_gc_trace/score_gc_trace.json") or []
    gc_prep_s = [g[0] for g in gc_prep]
    gc_score_s = [g[0] for g in gc_score]
    queue = SRC.queue_samples(cell / "numeric_gt/service/QUEUE_SAMPLES.json")
    queue_t = [q["t"] for q in queue]

    F = lambda i: DCI_INDEX[i]
    first_ao = min(f["action_open_raw_ns"] for f in frames.values())
    out, checks = [], collections.Counter()
    for r in rows:
        fid = int(r["frame_id"])
        f = frames[fid]
        ao = f["action_open_raw_ns"]
        ms = lambda ns: None if ns is None else (float(ns) - ao) / 1e6
        ev = {}
        for e in f["events"]:
            ev.setdefault(e["role"] + ":" + e["name"], e["raw_ns"])
        req = {}
        for q in f.get("requests", []):
            req.setdefault((q["worker"], q["op"]), q)
        rec, ef = edge_records.get(fid), edge_frames.get(fid, {})
        span = emit_span.get(fid)
        tun_wall = (pcap.get(rec["feedback_sha256"]) or [None]) if rec else [None]
        if rec and len(pcap.get(rec["feedback_sha256"]) or []) > 1:
            checks["pcap_duplicate_payload"] += 1
        tun = None if tun_wall[0] is None else float(clock(tun_wall[0]))
        row = dict(capture=label, decision_index=r["decision_index"], frame_id=fid,
                   session_t_s=round((ao - first_ao) / 1e9, 4), action_open_raw_ns=ao,
                   mode_id=r["executed_action"]["mode_id"], q_e4=r["executed_action"]["q_e4"],
                   decision_kind=r["executed_action"]["kind"], payload_bytes=r["payload_bytes"],
                   payload_chunks=r["payload_chunks"], prior_ul_mcs_state=r["prior_ul_mcs"],
                   snr_proxy_db=r["snr_db"], service_outcome=r["service_outcome"],
                   controller_resolution=r["controller_resolution"], feedback_kind=r["feedback_kind"],
                   feedback_reason=r["feedback_reason"], outcome_class=r["outcome_class"],
                   timely_170=bool(r["timely"]), gt_published_before_prediction_received=bool(r["gt_first"]),
                   scoring_op=r["scoring_request_op"])
        # ---- instants (ms after action open)
        inst = dict(
            actor_end=r["actor_end"], ue_freeze_start=ms(ev.get("ue_freeze_thread:freeze_start")),
            ue_freeze_frozen=ms(ev.get("ue_freeze_thread:freeze_frozen")),
            ue_input_start=r["ue_input_start"], ue_input_end=r["ue_input_end"], ue_last_send=r["ue_last_send"],
            edge_reassembly_complete=r["edge_reassembly_complete"], edge_compute_start=r["edge_compute_start"],
            edge_compute_end=r["edge_compute_end"], service_prediction_received=r["service_prediction_received"],
            service_freeze_received=ms(ev.get("service:freeze_received")),
            prep_freeze_begin=ms((req.get(("prep", "FREEZE")) or {}).get("timing", {}).get("begin_raw_ns")),
            prep_freeze_end=ms((req.get(("prep", "FREEZE")) or {}).get("timing", {}).get("end_raw_ns")),
            gt_published=r["gt_published"], target_worker_begin=r["target_worker_begin"],
            target_worker_end=r["target_worker_end"], prediction_worker_begin=r["prediction_worker_begin"],
            prediction_worker_end=r["prediction_worker_end"],
            scoring_op_begin=r["prediction_worker_begin"] if r["scoring_request_op"] == "PREDICTION" else r["target_worker_begin"],
            scoring_worker_end=r["scoring_worker_end"], result_sent=r["result_sent"],
            edge_result_received=r["edge_result_received"],
            edge_emit_span_start=None if span is None else ms(span[0]),
            edge_emit_span_end=None if span is None else ms(span[1]),
            feedback_emitted_post_send=r["feedback_emitted"], ue_tun_arrival=None if tun is None else (tun - ao) / 1e6,
            ue_receipt=r["ue_receipt"])
        both = None
        if inst["service_prediction_received"] is not None and inst["gt_published"] is not None:
            both = max(inst["service_prediction_received"], inst["gt_published"])
        inst["inputs_both_ready"] = both
        # ---- DL DCI that carried the datagram: last DL DCI in [pre-send, tunnel arrival]
        dci = dict(dl_dci=None, dl_dci_in_window=None, dl_dci_round=None, dl_dci_ndi=None, dl_dci_harq_pid=None,
                   dl_dci_tbs=None, dl_dci_mcs=None, dl_dci_rb_size=None, dl_window_any_retx=None)
        if tun is not None and span is not None:
            i0 = bisect.bisect_left(dl_raw, span[0])
            i1 = bisect.bisect_right(dl_raw, tun)
            dci["dl_dci_in_window"] = int(i1 - i0)
            if i1 > i0:
                v = dl[i1 - 1]
                dci.update(dl_dci=(dl_raw[i1 - 1] - ao) / 1e6, dl_dci_round=int(v[1 + F("round")]),
                           dl_dci_ndi=int(v[1 + F("ndi")]), dl_dci_harq_pid=int(v[1 + F("harq_pid")]),
                           dl_dci_tbs=int(v[1 + F("tbs")]), dl_dci_mcs=int(v[1 + F("mcs")]),
                           dl_dci_rb_size=int(v[1 + F("rb_size")]),
                           dl_window_any_retx=bool(any(int(dl[k][1 + F("round")]) > 0 for k in range(i0, i1))))
        row.update(inst)
        row.update(dci)
        # ---- partition
        total = 0.0
        ok = True
        for col, _, a, b in STAGES:
            ta = 0.0 if a is None else inst.get(a)
            tb = inst.get(b)
            v = None if ta is None or tb is None else tb - ta
            row[col] = v
            if v is None:
                ok = False
            else:
                total += v
        row["partition_complete"] = ok
        row["partition_residual_ms"] = None if not ok or inst["ue_receipt"] is None else total - inst["ue_receipt"]
        # ---- sub-intervals / bounds
        sub = lambda a, b: None if a is None or b is None else b - a
        row.update(
            net_send_to_dl_dci=sub(inst["edge_emit_span_start"], dci["dl_dci"]),
            net_dl_dci_to_tun=sub(dci["dl_dci"], inst["ue_tun_arrival"]),
            net_post_send_stamp_to_tun=sub(inst["feedback_emitted_post_send"], inst["ue_tun_arrival"]),
            edge_emit_span_ms=sub(inst["edge_emit_span_start"], inst["edge_emit_span_end"]),
            ue_freeze_ms=sub(inst["ue_freeze_start"], inst["ue_freeze_frozen"]),
            prediction_ready_to_receipt=sub(inst["edge_compute_end"], inst["ue_receipt"]),
            emit_post_send_to_receipt=sub(inst["feedback_emitted_post_send"], inst["ue_receipt"]),
            prep_freeze_wall_ms=sub(inst["prep_freeze_begin"], inst["prep_freeze_end"]),
            prep_freeze_cpu_ms=None if ("prep", "FREEZE") not in req else req[("prep", "FREEZE")]["timing"]["thread_cpu_ns"] / 1e6,
            score_exec_cpu_ms=None if (("score", r["scoring_request_op"]) not in req) else
            req[("score", r["scoring_request_op"])]["timing"]["thread_cpu_ns"] / 1e6)
        # ---- associations: GC, queue, overlapping UE spans
        end = ao + (inst["ue_receipt"] or 400.0) * 1e6
        def gc_in(starts, events):
            i0, i1 = bisect.bisect_left(starts, ao), bisect.bisect_right(starts, end)
            ev_ = events[i0:i1]
            return len(ev_), round(sum(g[1] for g in ev_) / 1e6, 3), sum(1 for g in ev_ if g[2] is not None)
        row["prep_gen2_n"], row["prep_gen2_wall_ms"], row["prep_gen2_in_request_n"] = gc_in(gc_prep_s, gc_prep)
        row["score_gen2_n"], row["score_gen2_wall_ms"], row["score_gen2_in_request_n"] = gc_in(gc_score_s, gc_score)
        t_pred = ao + (inst["service_prediction_received"] or 0) * 1e6
        k = bisect.bisect_right(queue_t, t_pred) - 1
        qs = queue[k] if k >= 0 else {}
        row.update(queue_score_at_pred=qs.get("score_queue"), queue_prep_at_pred=qs.get("prep_queue"),
                   queue_freeze_at_pred=qs.get("freeze_queue"), queue_waiting_at_pred=qs.get("waiting"))
        def longest(index, a, b):
            if a is None or b is None or b <= a:
                return None
            hits = index.overlapping(ao + a * 1e6, ao + b * 1e6)
            return None if not hits else round(max((x[1] - x[0]) for x in hits) / 1e6, 3)
        row["carla_tick_longest_during_tun_to_app_ms"] = longest(ticks, inst["ue_tun_arrival"], inst["ue_receipt"])
        row["carla_tick_longest_during_ue_freeze_ms"] = longest(ticks, inst["ue_freeze_start"], inst["ue_freeze_frozen"])
        row["ue_other_longest_during_ue_freeze_ms"] = longest(ue_all, inst["ue_freeze_start"], inst["ue_freeze_frozen"])
        row["group"] = group_of(inst["ue_receipt"])
        # ---- clock / causality checks (counted, never repaired)
        if rec is not None and ef.get("emitted") is not None:
            checks["edge_wall_vs_raw_n"] += 1
        if tun is None:
            checks["tun_unjoined"] += 1
        else:
            if inst["ue_receipt"] is not None and inst["ue_tun_arrival"] > inst["ue_receipt"]:
                checks["tun_after_app_receipt"] += 1
            if span is not None and inst["ue_tun_arrival"] < inst["edge_emit_span_start"]:
                checks["tun_before_pre_send"] += 1
        out.append(row)

    # edge same-instant validation of the clock bridge (wall stamped just before raw, both post-send)
    val = np.array([float(clock(edge_records[f]["emit_wall_ns"])) - edge_frames[f]["emitted"]
                    for f in edge_records if f in edge_frames]) / 1e6
    info = dict(label=label, capture_root=str(root), cell=str(cell), recorded=recorded, reproduced=reproduced,
                pcap=pcap_info, ue_trace_event_counts=t_counts, dl_dci_total=int(len(dl)),
                dl_dci_round_counts=dict(collections.Counter(int(x) for x in dl[:, 1 + F("round")])),
                clock_bridge=dict(clock.summary,
                                  edge_validation_mapped_wall_minus_raw_ms=pct(val, (1, 50, 99)) | dict(
                                      min=round(float(val.min()), 4), positive=int((val > 0).sum()))),
                checks=dict(checks), gc=dict(prep_gen2=len(gc_prep), prep_gen2_in_request=sum(g[2] is not None for g in gc_prep),
                                             score_gen2=len(gc_score), score_gen2_in_request=sum(g[2] is not None for g in gc_score),
                                             score_gen2_events=[dict(start_ms_after_first_open=round((g[0] - first_ao) / 1e6, 1),
                                                                     wall_ms=round(g[1] / 1e6, 2), op=g[2]) for g in gc_score]),
                inputs={k: SRC.sha256(p) for k, p in dict(
                    frames=cell / "numeric_gt/service/FRAMES.jsonl",
                    edge_state=cell / "profiler/edge/numeric_runtime/EDGE_EARLY_GT.json",
                    edge_profile=cell / "profiler/edge/edge.profile.jsonl", ue_profile=cell / "profiler/ue.profile.jsonl",
                    ue_evidence=cell / "run4_phase6/PHASE6_UE_EVIDENCE.json", pcap=cell / "quality_ack_oaitun_ue1.pcap",
                    ue_ttrace=cell / "ttracer/ue/ue.raw", queue=cell / "numeric_gt/service/QUEUE_SAMPLES.json",
                    ingest_mcs=root / "capture/m3_radio/ttracer/live_mcs_with_ingest.csv",
                    ingest_pusch=root / "capture/m3_radio/ttracer/live_pusch_with_ingest.csv",
                    prep_gc=root / "gt_gc_trace/prep_gc_trace.json", score_gc=root / "gt_gc_trace/score_gc_trace.json",
                    t_messages=SRC.T_MESSAGES).items()})
    return out, info


DCI_INDEX = {k: i for i, k in enumerate(SRC.DCI_FIELDS)}


# --------------------------------------------------------------------------- summaries
def summarize(rows, info):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    on = by["ON_TIME"]
    med_on = {c: float(np.median([r[c] for r in on if r[c] is not None])) for c, *_ in STAGES}
    late = [r for r in rows if r["group"] in ("LATE_170_200", "EXTREME_GT200")]
    for r in rows:
        ex = {c: r[c] - med_on[c] for c, *_ in STAGES if r[c] is not None}
        dom = max(ex, key=ex.get)
        r["dominant_excess_stage"] = dom
        r["dominant_excess_ms"] = round(ex[dom], 3)
        r["spike_class"] = ("ON_TIME" if r["group"] == "ON_TIME" else
                            (f"SINGLE_STAGE:{dom}" if ex[dom] >= SPIKE_EXCESS_MS else "DIFFUSE_ACCUMULATION"))
        r["ue_freeze_stall"] = r["ue_freeze_ms"] is not None and r["ue_freeze_ms"] >= UE_FREEZE_STALL_MS
    stage_stats = {g: {c: pct([r[c] for r in by[g]]) for c, *_ in STAGES} | dict(
        ue_receipt=pct([r["ue_receipt"] for r in by[g]]), n=len(by[g])) for g in GROUPS}
    subs = ("net_send_to_dl_dci", "net_dl_dci_to_tun", "net_post_send_stamp_to_tun", "edge_emit_span_ms",
            "ue_freeze_ms", "prediction_ready_to_receipt", "prep_freeze_wall_ms", "prep_freeze_cpu_ms", "payload_bytes")
    sub_stats = {g: {c: pct([r[c] for r in by[g]]) for c in subs} for g in GROUPS}
    allp = [r["prediction_ready_to_receipt"] for r in rows if r["prediction_ready_to_receipt"] is not None]
    post_p50 = float(np.median(allp))
    classes = collections.Counter(r["spike_class"] for r in late)
    classes_extreme = collections.Counter(r["spike_class"] for r in by["EXTREME_GT200"])
    late_series = np.array([r["group"] != "ON_TIME" for r in rows], float)
    x = late_series - late_series.mean()
    acf = [round(float(np.dot(x[:-k], x[k:]) / np.dot(x, x)), 4) for k in range(1, 31)]
    band = 1.96 / math.sqrt(len(x))
    stall_t = [r["session_t_s"] for r in rows if r["ue_freeze_stall"]]
    bins = collections.defaultdict(list)
    for r in rows:
        bins[int(r["session_t_s"] // 60)].append(r)
    dl_late_net = [r for r in late if r["st_send_to_tun"] is not None]
    return dict(
        late_rescued_if_stage_at_on_time_median=rescue_counts(late, med_on),
        population=dict(decisions=len(rows), groups={g: len(by[g]) for g in GROUPS},
                        missing_receipt=len(by["MISSING"]),
                        service_outcomes=dict(collections.Counter(r["service_outcome"] for r in rows)),
                        controller_resolutions=dict(collections.Counter(r["controller_resolution"] for r in rows)),
                        outcome_classes=dict(collections.Counter(r["outcome_class"] for r in rows)),
                        late_within_10ms=sum(1 for r in late if r["ue_receipt"] <= DEADLINE_MS + 10)),
        partition=dict(complete=sum(r["partition_complete"] for r in rows),
                       max_abs_residual_ms=max(abs(r["partition_residual_ms"]) for r in rows if r["partition_complete"]),
                       negative_stage_counts={c: sum(1 for r in rows if r[c] is not None and r[c] < 0) for c, *_ in STAGES}),
        on_time_stage_medians_ms={c: round(v, 3) for c, v in med_on.items()},
        stage_stats_ms=stage_stats, sub_interval_stats_ms=sub_stats,
        post_prediction_chain=dict(all=pct(allp), prediction_ready_cutoff_ms=round(DEADLINE_MS - post_p50, 2),
                                   late_with_prediction_after_cutoff=sum(1 for r in late if r["edge_compute_end"] > DEADLINE_MS - post_p50),
                                   on_time_with_prediction_after_cutoff=sum(1 for r in on if r["edge_compute_end"] > DEADLINE_MS - post_p50)),
        spike_classes_late=dict(classes.most_common()), spike_classes_extreme=dict(classes_extreme.most_common()),
        ue_freeze_stall=dict(n=len(stall_t), in_late=sum(r["ue_freeze_stall"] for r in late),
                             in_extreme=sum(r["ue_freeze_stall"] for r in by["EXTREME_GT200"]),
                             on_time=sum(r["ue_freeze_stall"] for r in on),
                             inter_arrival_s=pct(np.diff(stall_t)) if len(stall_t) > 2 else None,
                             carla_tick_longest_during_stall_ms=pct([r["carla_tick_longest_during_ue_freeze_ms"] for r in rows if r["ue_freeze_stall"]]),
                             other_ue_span_longest_during_stall_ms=pct([r["ue_other_longest_during_ue_freeze_ms"] for r in rows if r["ue_freeze_stall"]])),
        ue_app_receive_stalls=[dict(decision_index=r["decision_index"], tun_to_app_ms=round(r["st_tun_to_app"], 2),
                                    carla_tick_longest_ms=r["carla_tick_longest_during_tun_to_app_ms"], receipt_ms=round(r["ue_receipt"], 1))
                               for r in rows if r["st_tun_to_app"] is not None and r["st_tun_to_app"] >= 20],
        downlink=dict(send_to_tun_late=pct([r["st_send_to_tun"] for r in dl_late_net]),
                      send_to_tun_on_time=pct([r["st_send_to_tun"] for r in on]),
                      dci_in_window=dict(collections.Counter(r["dl_dci_in_window"] for r in rows)),
                      dci_round_of_carrier=dict(collections.Counter(r["dl_dci_round"] for r in rows)),
                      windows_with_any_retx=sum(1 for r in rows if r["dl_window_any_retx"]),
                      over_10ms=[dict(decision_index=r["decision_index"], send_to_tun=round(r["st_send_to_tun"], 2),
                                      send_to_dci=r["net_send_to_dl_dci"], dci_to_tun=r["net_dl_dci_to_tun"],
                                      dci_in_window=r["dl_dci_in_window"], any_retx=r["dl_window_any_retx"],
                                      receipt=round(r["ue_receipt"], 1))
                                 for r in rows if r["st_send_to_tun"] is not None and r["st_send_to_tun"] > 10]),
        gt=dict(gt_extra_wait_positive=dict(on_time=sum(r["st_gt_extra_wait"] > 0 for r in on),
                                            late=sum(r["st_gt_extra_wait"] > 0 for r in late)),
                score_gen2_overlapping_late=sum(r["score_gen2_n"] > 0 for r in late),
                prep_gen2_in_request_overlapping_late=sum(r["prep_gen2_in_request_n"] > 0 for r in late)),
        associations_spearman_with_receipt={k: spearman([r[k] if r[k] is not None else np.nan for r in rows],
                                                        [r["ue_receipt"] for r in rows])
                                            for k in ("payload_bytes", "payload_chunks", "prep_freeze_cpu_ms", "ue_freeze_ms",
                                                      "session_t_s", "queue_score_at_pred", "st_send_to_tun")},
        periodicity=dict(late_indicator_acf_lags_1_30=acf, white_noise_band=round(band, 4),
                         lags_outside_band=[k + 1 for k, a in enumerate(acf) if abs(a) > band],
                         late_fraction_by_decision_mod_12=[round(float(np.mean([r["group"] != "ON_TIME" for r in rows if r["decision_index"] % 12 == m])), 3)
                                                           for m in range(12)]),
        by_minute=[dict(minute=k, n=len(v), on_time_fraction=round(sum(r["group"] == "ON_TIME" for r in v) / len(v), 3),
                        receipt_p50=round(float(np.median([r["ue_receipt"] for r in v])), 1),
                        prediction_p50=round(float(np.median([r["edge_compute_end"] for r in v])), 1),
                        gt_published_p50=round(float(np.median([r["gt_published"] for r in v])), 1),
                        ue_freeze_stalls=sum(r["ue_freeze_stall"] for r in v))
                   for k, v in sorted(bins.items())],
    )


PREDICTION_BRANCH = ("st_ue_actor", "st_ue_pre_input", "st_ue_input", "st_ue_encode_send", "st_uplink", "st_edge",
                     "st_pred_to_service")


def rescue_counts(late, med_on):
    """Arithmetic attribution on recorded intervals (not causal): late decisions whose receipt would be <= 170 ms if
    one stage were reduced to its on-time median, all other recorded intervals unchanged. Prediction-branch
    reductions respect the join (GT publication may then become the later input); the network rows set the
    interval to zero, which bounds the most the downlink/UE-receive path could explain."""
    out = {}
    for c, *_ in STAGES:
        n = 0
        for r in late:
            red = max(0.0, r[c] - med_on[c])
            if c in PREDICTION_BRANCH:
                old = r["inputs_both_ready"]
                delta = old - max(r["service_prediction_received"] - red, r["gt_published"])
            else:
                delta = red
            n += r["ue_receipt"] - delta <= DEADLINE_MS
        out[c] = n
    out["network_send_to_tun_zero"] = sum(r["ue_receipt"] - r["st_send_to_tun"] <= DEADLINE_MS for r in late)
    out["network_and_ue_receive_zero"] = sum(r["ue_receipt"] - r["st_send_to_tun"] - r["st_tun_to_app"] <= DEADLINE_MS
                                             for r in late)
    out["denominator_late"] = len(late)
    return out


CSV_EXCLUDE = {"action_open_raw_ns"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="append", required=True, help="LABEL=<capture root>=<recorded METRICS.json>")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    all_rows, report = [], dict(schema="splitfusion.single_ue_feedback_latency.v1", deadline_ms=DEADLINE_MS,
                                extreme_ms=EXTREME_MS, spike_excess_ms=SPIKE_EXCESS_MS,
                                ue_freeze_stall_ms=UE_FREEZE_STALL_MS,
                                stages=[dict(column=c, label=l, start=s or "action_open", end=e) for c, l, s, e in STAGES],
                                captures={})
    for spec in a.capture:
        label, root, metrics = spec.split("=", 2)
        rows, info = build(label, root, metrics)
        info["summary"] = summarize(rows, info)
        report["captures"][label] = info
        all_rows.extend(rows)
        print(label, info["reproduced"], info["summary"]["population"]["groups"], flush=True)
    fields = [k for k in all_rows[0] if k not in CSV_EXCLUDE]
    with open(a.out / "DECISIONS_JOINED.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in fields})
    (a.out / "SUMMARY.json").write_text(json.dumps(report, indent=1, default=str))
    from . import plots
    plots.render(all_rows, report, a.out)
    outputs = sorted(p for p in a.out.iterdir() if p.is_file() and p.name != "OUTPUT_SHA256.json")
    (a.out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: SRC.sha256(p) for p in outputs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
