"""Channel-adaptation, paired-quality and byte-layer analysis of the ONE FADE_RECOVERY capture (read-only, create-only).

All rules come from ``definitions.py`` (digest bound in ACCEPTANCE.json before launch). Three evidence classes are kept
apart in every output: LIVE (measured in the capture), OFFLINE_PAIRED (post-teardown replay of retained inputs) and
DERIVED/INTERPRETATION.

  python3 -m rl_agent.splitfusion_run5_channel_response_v1_analysis.channel_analysis \
      --decisions <DECISIONS jsonl> --replay <REPLAY.json> --out <dir>
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import definitions as D

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOT = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261006_run5_channel_response_v1/validation"
CONFIG = MAIN / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
COMPARATOR = dict(
    root=WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261006_run5_full_live_retention_v1/validation",
    decisions=MAIN / "experiments/splitfusion_run5_full_live_retention_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
    replay=MAIN / "experiments/splitfusion_run5_quality_reference_v1/replay_01/REPLAY.json")
PHASE_COLORS = {"P0_INITIAL_MIXED": "#E6E6E7", "P1_GOOD_PRE": "#CDEFEF", "P2_MIXED": "#EEF0F2",
                "P3_DEGRADED_SUSTAINED": "#F8D9C0", "P4_RECOVERY_GOOD": "#CDEFEF", "P5_LATE": "#EEF0F2"}
STATE_COLOR = dict(FAVORABLE="#00A3A3", INTERMEDIATE="#E39C2D", ADVERSE="#C2364F")
BRANCH = dict(full_local=("A full-local (FP32, no AE)", "#1f3a93"), noae_uint8_q0=("B noAE/UINT8/q0", "#7f7f7f"),
              agent_selected=("C agent (recorded action)", "#d35400"))
SPARSE = "SPARSE_TAIL_ESTIMATE"
SHORT = {"P0_INITIAL_MIXED": "P0\ninitial mixed", "P1_GOOD_PRE": "P1\ngood", "P2_MIXED": "P2\nmixed",
         "P3_DEGRADED_SUSTAINED": "P3\ndegraded", "P4_RECOVERY_GOOD": "P4\nrecovery", "P5_LATE": "P5\nlate"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def q(values, ps=(50, 95, 99)):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    out = dict(n=int(v.size))
    for p in ps:
        out[f"p{p}"] = float(np.percentile(v, p)) if v.size else None
    if v.size:
        out.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    if 99 in ps:
        out["p99_label"] = SPARSE if v.size < 1000 else "EMPIRICAL"
    return out


# ---------------------------------------------------------------------------- channel evidence (LIVE)
def design_states():
    config = json.loads(CONFIG.read_bytes())
    with open(MAIN / config["network"]["traces_csv"], newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["profile_id"] == "FADE_RECOVERY"]
    return [r["state"] for r in rows], [float(r["target_snr_db"]) for r in rows]


def channel_evidence(cell, states, targets):
    """Applied impairment (radio_trace.csv, CLOCK_MONOTONIC) and causal observation (ACK datagrams, CLOCK_MONOTONIC_RAW)."""
    with open(cell / "radio_trace.csv", newline="") as stream:
        trace = list(csv.DictReader(stream))
    acks = jl(cell / "run5_policy/CONTROLLER_ACKS.jsonl")
    snr = json.loads((cell / "run5_policy/SNR_EVIDENCE.json").read_bytes())
    identity = {(r["profile_id"], r["trace_id"], r["seed"]) for r in trace}
    steps = [int(r["step_index"]) for r in trace]
    statuses = {}
    for r in trace:
        statuses[r["command_timing_status"]] = statuses.get(r["command_timing_status"], 0) + 1
    target_err = max(abs(float(r["target_snr_db"]) - targets[int(r["step_index"])]) for r in trace if int(r["step_index"]) < len(targets))
    acked = [r for r in trace if r["command_timing_status"] in ("ACK_ON_TIME", "ACK_LATE")]
    valid_acks = [a for a in acks if a.get("valid") and a.get("command_id")]
    ack_step, ack_match = {}, 0
    for a in valid_acks:          # ack-N <-> N-th acknowledged radio_trace row (skipped steps send no command)
        n = int(a["command_id"].split("-")[1])
        if 1 <= n <= len(acked):
            row = acked[n - 1]
            ack_step[a["command_id"]] = int(row["step_index"])
            ack_match += abs(float(row["target_snr_db"]) - float(a["target_snr_db"])) < 1e-9
    clock_offsets = []
    for a in valid_acks[:2000]:
        n = int(a["command_id"].split("-")[1])
        if 1 <= n <= len(acked) and acked[n - 1].get("command_ack_monotonic_ns"):
            clock_offsets.append((a["effective_raw_ns"] - int(acked[n - 1]["command_ack_monotonic_ns"])) / 1e6)
    lat = [float(r["command_latency_ms"]) for r in acked if r.get("command_latency_ms")]
    dec = snr["decision_snr"]
    ages, invalid, refused, refusals = [], 0, 0, []
    for row in dec:
        s = row["snr"]
        if not s.get("valid"):
            invalid += 1
        if not row.get("admitted"):
            refused += 1
            refusals.append(dict(frame_id=row["frame_id"], reason=row.get("refusal"), snr_valid=s.get("valid")))
        if s.get("effective_since_ns") is not None:
            ages.append((row["boundary"]["state_commit_timestamp_ns"] - s["effective_since_ns"]) / 1e6)
    checks = dict(
        single_registered_identity=identity == {("FADE_RECOVERY", "GM_V2_FADE_RECOVERY_SEED_2026082104", "2026082104")},
        contiguous_steps_from_zero=steps == list(range(len(steps))),
        applied_targets_equal_pinned_trace=target_err <= 5.1e-7,
        every_valid_ack_maps_to_its_applied_step=ack_match == len(valid_acks),
        every_state_build_had_valid_causal_snr=invalid == 0,
        state_refusals_are_non_snr_registered_fallbacks=all(r["snr_valid"] and "snr" not in (r["reason"] or "").lower()
                                                            and "ExternalFallbackRequired" in (r["reason"] or "") for r in refusals),
        observation_kind_is_simulator_effective_target_not_gnb=all(
            r["snr"].get("kind") == "SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB" for r in dec))
    return dict(
        steps_applied=len(steps), max_step=max(steps), statuses=statuses, max_abs_target_error_db=target_err,
        command_latency_ms=q(lat), acks_valid=len(valid_acks), acks_total=len(acks),
        ack_raw_minus_monotonic_ms=q(clock_offsets) | dict(note="two clock domains reported side by side; no conversion used "
                                                               "in analysis (decision timing stays in CLOCK_MONOTONIC_RAW)"),
        state_builds=len(dec), invalid_snr=invalid, refused_state_builds=refused, refusals=refusals, snapshot_age_at_state_commit_ms=q(ages),
        clocks=dict(radio_trace="CLOCK_MONOTONIC (time.monotonic_ns) + CLOCK_REALTIME wall", acks_and_decisions="CLOCK_MONOTONIC_RAW"),
        gnb_snr_used=False, checks=checks, passed=all(checks.values())), ack_step, acks


# ---------------------------------------------------------------------------- per-decision join (LIVE)
def join(rows, snr_doc, ack_step, acks, states):
    by_frame = {int(r["frame_id"]): r for r in snr_doc["decision_snr"]}
    t0 = min(a["effective_raw_ns"] for a in acks if a.get("valid") and a.get("command_id") == "ack-1")
    out = []
    for r in sorted(rows, key=lambda r: r["decision_index"]):
        s = by_frame.get(int(r["frame_id"]))
        cmd = s["snr"].get("active_command_id") if s else None
        step = ack_step.get(cmd)
        x = dict(r)
        x.update(step=step, phase=D.phase_of(step) if step is not None else None,
                 state=states[step] if step is not None and step < len(states) else None,
                 action_open_raw_ns=s["boundary"]["action_open_timestamp_ns"] if s else None,
                 state_commit_raw_ns=s["boundary"]["state_commit_timestamp_ns"] if s else None,
                 snr_effective_raw_ns=s["snr"].get("effective_since_ns") if s else None,
                 t_s=(s["boundary"]["action_open_timestamp_ns"] - t0) / 1e9 if s else None,
                 on_time=r.get("ue_receipt") is not None and r["ue_receipt"] <= D.DEADLINE_MS,
                 mode=(r.get("executed_action") or {}).get("mode_id"), q_e4=(r.get("executed_action") or {}).get("q_e4"),
                 adv_core=step is not None and D.ADV_CORE["start"] <= step < D.ADV_CORE["end"])
        x["uplink_ms"] = (r["edge_reassembly_complete"] - r["ue_first_send"]) if r.get("edge_reassembly_complete") is not None \
            and r.get("ue_first_send") is not None else None
        x["non_uplink_ms"] = (r["ue_receipt"] - x["uplink_ms"]) if x["uplink_ms"] is not None and r.get("ue_receipt") is not None else None
        x["return_ms"] = (r["ue_receipt"] - r["result_sent"]) if r.get("result_sent") is not None and r.get("ue_receipt") is not None else None
        out.append(x)
    return out


def rolling(flags, w):
    flags = np.asarray(flags, float)
    out = np.full(flags.size, np.nan)
    for i in range(flags.size):
        if i + 1 >= w:
            out[i] = flags[i + 1 - w:i + 1].mean()
    return out


def transition_response(d, step, direction, next_step):
    """Frozen rule: pre-window band, first and sustained response, counted from the first admitting decision."""
    idx = [i for i, x in enumerate(d) if x["step"] is not None]
    pre = [d[i] for i in idx if d[i]["step"] < step][-D.PRE_WINDOW_DECISIONS:]
    post = [i for i in idx if d[i]["step"] >= step and (next_step is None or d[i]["step"] < next_step)]
    res = dict(step=step, direction=direction, pre_n=len(pre), post_n=len(post))
    if len(pre) < 5 or not post:
        res["class"] = "NOT_EVALUABLE"
        return res
    pay = np.asarray([x["payload_bytes"] for x in pre], float)
    lo, hi = np.percentile(pay, D.BAND[0]), np.percentile(pay, D.BAND[1])
    outside = (lambda x: x["payload_bytes"] < lo) if direction == "deterioration" else (lambda x: x["payload_bytes"] > hi)
    first_adm = post[0]
    window = post[:D.SEARCH_LIMIT_DECISIONS]
    first = next((k for k, i in enumerate(window) if outside(d[i])), None)
    sust = next((k for k, i in enumerate(window)
                 if sum(outside(d[j]) for j in window[k:k + D.SUSTAIN["window"]]) >= D.SUSTAIN["required"]), None)
    cls = "NONE" if sust is None else ("QUICK" if sust <= D.ACTION_RESPONSE_CLASS["QUICK_MAX_DECISIONS"] else
                                        "SLOW" if sust >= D.ACTION_RESPONSE_CLASS["SLOW_MIN_DECISIONS"] else "MODERATE")
    f = d[first_adm]
    res.update(band_bytes=[float(lo), float(hi)], pre_payload_p50=float(np.median(pay)),
               pre_q_p50=float(np.median([x["q_e4"] for x in pre])), pre_snr_p50=float(np.median([x["snr_db"] for x in pre])),
               first_admitting_decision=f["decision_index"], first_admitting_snr_db=f["snr_db"],
               observation_delay_ms=(f["state_commit_raw_ns"] - f["snr_effective_raw_ns"]) / 1e6
               if f.get("snr_effective_raw_ns") else None,
               first_response_decisions=first, sustained_response_decisions=sust, **{"class": cls})
    for key, k in (("first", first), ("sustained", sust)):
        if k is not None:
            x = d[window[k]]
            res[f"{key}_response"] = dict(decision_index=x["decision_index"], payload_bytes=x["payload_bytes"], q_e4=x["q_e4"],
                                          mode=x["mode"], snr_db=x["snr_db"],
                                          ms_after_first_admitting_action_open=(x["action_open_raw_ns"] - f["action_open_raw_ns"]) / 1e6)
    post_w = [d[i] for i in window]
    res["post_window"] = dict(payload_p50=float(np.median([x["payload_bytes"] for x in post_w])),
                              q_p50=float(np.median([x["q_e4"] for x in post_w])),
                              modes=sorted({x["mode"] for x in post_w}),
                              on_time=float(np.mean([x["on_time"] for x in post_w])), n=len(post_w))
    return res


def kpi_recovery(d, p1_on_time):
    roll = rolling([x["on_time"] for x in d], D.ROLLING_WINDOW_DECISIONS)
    t_rec = next((i for i, x in enumerate(d) if x["step"] is not None and x["step"] >= 972), None)
    p4_end = max((i for i, x in enumerate(d) if x["phase"] == "P4_RECOVERY_GOOD"), default=None)
    out = {}
    for name, thr in (("relative", p1_on_time - D.KPI_RECOVERY["margin"]), ("absolute_0.80", D.KPI_RECOVERY["secondary_absolute"])):
        hit = None
        if t_rec is not None:
            for i in range(t_rec, len(d) - D.KPI_RECOVERY["hold_decisions"]):
                seg = roll[i:i + D.KPI_RECOVERY["hold_decisions"] + 1]
                if not np.any(np.isnan(seg)) and np.all(seg >= thr):
                    hit = i
                    break
        k = None if hit is None else hit - t_rec
        within = hit is not None and p4_end is not None and hit <= p4_end
        cls = ("NOT_RECOVERED" if not within else "QUICK" if k <= D.KPI_RECOVERY_CLASS["QUICK_MAX_DECISIONS"]
               else "SLOW" if k >= D.KPI_RECOVERY_CLASS["SLOW_MIN_DECISIONS"] else "MODERATE")
        out[name] = dict(threshold=float(thr), decisions_after_T_REC=k, decision_index=None if hit is None else d[hit]["decision_index"],
                         seconds_after_T_REC=None if hit is None else (d[hit]["action_open_raw_ns"] - d[t_rec]["action_open_raw_ns"]) / 1e9,
                         **{"class": cls})
    # trailing window at T_REC itself (was KPI already above threshold?)
    out["trailing20_at_T_REC"] = None if t_rec is None or np.isnan(roll[t_rec]) else float(roll[t_rec])
    return out, roll


def phase_table(d, actors):
    out = {}
    names = [p["name"] for p in D.PHASES] + [D.ADV_CORE["name"]]
    for name in names:
        rs = [x for x in d if (x["adv_core"] if name == D.ADV_CORE["name"] else x["phase"] == name)]
        if not rs:
            out[name] = dict(n=0)
            continue
        k = sum(x["on_time"] for x in rs)
        late = [x for x in rs if not x["on_time"]]
        up = [x["uplink_ms"] for x in rs if x["uplink_ms"] is not None]
        rate = [8 * x["payload_bytes"] / x["uplink_ms"] / 1e3 for x in rs if x["uplink_ms"] and x["uplink_ms"] > 0]   # Mbit/s
        nonup = [x["non_uplink_ms"] for x in rs if x["non_uplink_ms"] is not None]
        budget = (np.median(rate) * 1e6 / 8) * max(0.0, D.DEADLINE_MS - np.median(nonup)) / 1e3 if rate and nonup else None
        uplink_bound = sum(1 for x in late if x["uplink_ms"] is not None and x["non_uplink_ms"] is not None
                           and x["uplink_ms"] > D.DEADLINE_MS - float(np.median(nonup)))
        pays = np.asarray([x["payload_bytes"] for x in rs], float)
        small = [x for x in rs if x["payload_bytes"] <= np.percentile(pays, 25)]
        sc = [x for x in rs if x["vehicle_eligible_gt"] is not None]
        out[name] = dict(
            n=len(rs), steps=[min(x["step"] for x in rs), max(x["step"] for x in rs)],
            snr_db=q([x["snr_db"] for x in rs], (5, 50, 95)), states={s: sum(x["state"] == s for x in rs) for s in D.CONDITIONS},
            on_time=k / len(rs), on_time_k=k, on_time_wilson=list(D.wilson(k, len(rs))),
            missing=sum(x.get("ue_receipt") is None for x in rs),
            outcomes={c: sum(x["outcome_class"] == c for x in rs) for c in sorted({x["outcome_class"] for x in rs})},
            mode_share={str(m): sum(x["mode"] == m for x in rs) / len(rs) for m in sorted({x["mode"] for x in rs})},
            q_e4=q([x["q_e4"] for x in rs], (10, 50, 90)), payload_L2_bytes=q([x["payload_bytes"] for x in rs]),
            prior_ul_mcs=q([x.get("prior_ul_mcs") for x in rs], (10, 50, 90)),
            rlc_backlog_bytes=q([x.get("pre_action_rlc_backlog_bytes") for x in rs], (50, 95)),
            receipt_ms=q([x.get("ue_receipt") for x in rs]),
            stages_ms=dict(ue_input_end=q([x.get("ue_input_end") for x in rs], (50, 95)),
                           ue_encode_end=q([x.get("ue_encode_end") for x in rs], (50, 95)),
                           uplink=q(up, (50, 95)), prediction_at_service=q([x.get("service_prediction_received") for x in rs], (50, 95)),
                           gt_published=q([x.get("gt_published") for x in rs], (50, 95)),
                           scoring_end=q([x.get("scoring_worker_end") for x in rs], (50, 95)),
                           result_sent=q([x.get("result_sent") for x in rs], (50, 95)),
                           return_to_ue=q([x["return_ms"] for x in rs], (50, 95))),
            gt_first_share=float(np.mean([bool(x.get("gt_first")) for x in rs])),
            feasibility_DERIVED=dict(effective_uplink_mbps=q(rate, (10, 50, 90)), non_uplink_ms=q(nonup, (50, 95)),
                                     implied_L2_budget_bytes_at_median=budget, late=len(late), late_uplink_exceeds_slack=uplink_bound,
                                     smallest_quartile_on_time=float(np.mean([x["on_time"] for x in small])) if small else None,
                                     smallest_quartile_payload_max=float(np.percentile(pays, 25))),
            scene=dict(actors=q([actors.get(x["frame_id"]) for x in rs], (50, 95)),
                       vehicle_eligible_mean=float(np.mean([x["vehicle_eligible_gt"] for x in sc])) if sc else None,
                       person_eligible_mean=float(np.mean([x["person_eligible_gt"] for x in sc])) if sc else None,
                       camera_si=q([x.get("camera_si") for x in rs], (50,)), radar_p40=q([x.get("radar_p40") for x in rs], (50,))),
            freshness_capture_to_feedback_ms=q([x.get("freshness_capture_to_feedback_ms") for x in rs]),
            late_causes=dict(gt_later_than_prediction=sum(1 for x in late if x.get("gt_published") and x.get(
                "service_prediction_received") and x["gt_published"] > x["service_prediction_received"]),
                prediction_incl_uplink_later=sum(1 for x in late if not (x.get("gt_published") and x.get(
                    "service_prediction_received") and x["gt_published"] > x["service_prediction_received"])),
                late_uplink_ms=q([x["uplink_ms"] for x in late], (50,)), late_payload_L2=q([x["payload_bytes"] for x in late], (50,))),
            on_time_by_payload_tercile_DERIVED=terciles(rs),
            live_agent_quality=live_quality(rs), live_agent_quality_on_time=live_quality([x for x in rs if x["on_time"]]))
    return out


def terciles(rs):
    pays = np.asarray([x["payload_bytes"] for x in rs], float)
    if pays.size < 9:
        return None
    cut = np.percentile(pays, [100 / 3, 200 / 3])
    out = []
    for lo, hi in ((-1, cut[0]), (cut[0], cut[1]), (cut[1], np.inf)):
        g = [x for x in rs if lo < x["payload_bytes"] <= hi]
        out.append(dict(upper_bytes=None if np.isinf(hi) else float(hi), n=len(g), on_time=float(np.mean([x["on_time"] for x in g]))
                        if g else None, uplink_ms_p50=float(np.median([x["uplink_ms"] for x in g if x["uplink_ms"] is not None]))
                        if g else None))
    return dict(note="same phase, different scenes: descriptive feasibility evidence, not a fixed-action measurement", bins=out)


def live_quality(rs):
    """Agent quality on ALL scored decisions of a set (LIVE archive scores; misses counted)."""
    out = {}
    for cls, tol in D.TOLERANCE_M.items():
        elig = sum(x.get(f"{cls}_eligible_gt") or 0 for x in rs)
        hits = sum(sum(e <= tol for e in (x.get(f"{cls}_matched_xy_errors_m") or [])) for x in rs)
        tp = sum(x.get(f"{cls}_tp") or 0 for x in rs)
        fp = sum(x.get(f"{cls}_fp") or 0 for x in rs)
        inter = sum(x.get(f"seg_{cls}_intersection_pixels") or 0 for x in rs)
        union = sum(x.get(f"seg_{cls}_union_pixels") or 0 for x in rs)
        out[cls] = dict(eligible=elig, within_tol=hits, recall=hits / elig if elig else None,
                        precision=tp / (tp + fp) if tp + fp else None, seg_iou=inter / union if union else None)
    return out


# ---------------------------------------------------------------------------- paired quality (OFFLINE_PAIRED)
def m_counts(meas, cls):
    tol = D.TOLERANCE_M[cls]
    errs = meas[f"loc_{cls}_matched_xy_errors_m"]
    return dict(elig=meas[f"loc_{cls}_eligible_gt"], hits=sum(e <= tol for e in errs), tp=meas[f"loc_{cls}_tp"],
                fp=meas[f"loc_{cls}_fp"], inter=meas[f"seg_{cls}_intersection_pixels"], union=meas[f"seg_{cls}_union_pixels"],
                errs=list(errs), fn=meas[f"loc_{cls}_fn"])


def pooled(samples, branch, cls):
    c = [m_counts(s[branch]["measurement"], cls) for s in samples]
    elig, hits = sum(x["elig"] for x in c), sum(x["hits"] for x in c)
    tp, fp = sum(x["tp"] for x in c), sum(x["fp"] for x in c)
    i, u = sum(x["inter"] for x in c), sum(x["union"] for x in c)
    return dict(eligible=elig, hits=hits, recall=hits / elig if elig else None, tp=tp, fp=fp,
                precision=tp / (tp + fp) if tp + fp else None, seg_iou=i / u if u else None, misses=sum(x["fn"] for x in c))


def block_boot(samples, fn, rng):
    n = len(samples)
    if n == 0:
        return [None, None]
    b = D.BOOTSTRAP["block"]
    starts = np.arange(n)
    vals = []
    for _ in range(D.BOOTSTRAP["resamples"]):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, min(s + b, n)))
        v = fn([samples[k] for k in idx[:n]])
        if v is not None:
            vals.append(v)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_quality(samples, label):
    rng = np.random.default_rng(D.BOOTSTRAP["seed"])
    out = dict(view=label, samples=len(samples))
    for cls in D.TOLERANCE_M:
        a, b, c = (pooled(samples, br, cls) for br in ("full_local", "noae_uint8_q0", "agent_selected"))
        frames = [s for s in samples if s["full_local"]["measurement"][f"loc_{cls}_eligible_gt"] > 0]

        def diff(key):
            def f(ss):
                x, y = pooled(ss, "agent_selected", cls)[key], pooled(ss, "full_local", cls)[key]
                return None if x is None or y is None else x - y
            return f

        def ratio(key):
            def f(ss):
                x, y = pooled(ss, "agent_selected", cls)[key], pooled(ss, "full_local", cls)[key]
                return None if x is None or not y else x / y
            return f
        per = []
        zero_ref = []
        for s in frames:
            ca, cc = m_counts(s["full_local"]["measurement"], cls), m_counts(s["agent_selected"]["measurement"], cls)
            ra, rc = ca["hits"] / ca["elig"], cc["hits"] / cc["elig"]
            row = dict(frame_id=s["frame_id"], eligible=ca["elig"], full_local=ra, agent=rc, diff=rc - ra,
                       ratio=rc / ra if ra > 0 else None)
            (per if ra > 0 else zero_ref).append(row)
        shares = {f"{t:.2f}": (sum(r["ratio"] >= t for r in per) / len(per) if per else None) for t in D.RETENTION_SHARES}
        # short windows of consecutive retained samples (pooled within the window)
        wins = []
        W = D.QUALITY_WINDOW_RETAINED
        for k in range(0, max(0, len(samples) - W + 1)):
            seg = samples[k:k + W]
            pa, pc = pooled(seg, "full_local", cls), pooled(seg, "agent_selected", cls)
            if pa["hits"]:
                wins.append(pc["hits"] / pa["hits"])
        out[cls] = dict(
            full_local=a, noae_uint8=b, agent=c,
            recall_retention=(c["hits"] / a["hits"]) if a["hits"] else None,
            recall_diff=(c["recall"] - a["recall"]) if a["recall"] is not None else None,
            recall_diff_ci=block_boot(samples, diff("recall"), rng), recall_retention_ci=block_boot(samples, ratio("hits"), rng),
            precision_retention=(c["precision"] / a["precision"]) if a["precision"] else None,
            precision_diff_ci=block_boot(samples, diff("precision"), rng),
            seg_retention=(c["seg_iou"] / a["seg_iou"]) if a["seg_iou"] else None,
            seg_diff_ci=block_boot(samples, diff("seg_iou"), rng),
            uint8_minus_full_recall=(b["recall"] - a["recall"]) if a["recall"] is not None else None,
            frames_with_eligible=len(frames), frames_full_local_zero=dict(n=len(zero_ref), agent_hits_on_them=sum(
                r["agent"] > 0 for r in zero_ref), rows=zero_ref),
            per_frame_diff=q([r["diff"] for r in per + zero_ref], (5, 50, 95)), per_frame_rows=per,
            descriptive_share_frames_retaining=shares | dict(n_frames=len(per), note="descriptive, not gates"),
            descriptive_share_windows_retaining={f"{t:.2f}": (float(np.mean([w >= t for w in wins])) if wins else None)
                                                 for t in D.RETENTION_SHARES} | dict(n_windows=len(wins), window=W),
            uncertainty_note="person n small: intervals only, no pass/fail" if cls == "person" else None)
    return out


# ---------------------------------------------------------------------------- bytes
def byte_layers(sample, wire, chunk_bytes, header_size, W, X):
    """L1/L2/L3 for agent (transmitted) and baseline (offline re-encoded) on the same frame."""
    from phase2_map_sharing.transport import chunk_payload
    env, ctx = W.unpack_sfd4(wire)
    l1_agent = len(env.inner_payload)
    l2_agent = len(wire)
    overhead = l2_agent - l1_agent
    method = "ARITHMETIC_SAME_FRAME_OVERHEAD"
    l2_base = sample["baseline"]["payload_bytes"] + overhead
    try:                                   # preferred: actually re-pack the baseline inner payload into this frame's SFD4
        import dataclasses
        base_inner = None
        if "_baseline_inner" in sample:
            base_inner = sample["_baseline_inner"]
        if base_inner is not None:
            env_b = dataclasses.replace(env, inner_payload=base_inner, mode_id=0, q_e4=0)
            l2_base = len(W.pack_sfd4(env_b, ctx))
            method = "REPACKED_SFD4"
    except Exception as exc:  # noqa: BLE001 - recorded, arithmetic fallback kept
        method = f"ARITHMETIC_SAME_FRAME_OVERHEAD (repack refused: {type(exc).__name__})"

    def l3(n):
        chunks = -(-n // (chunk_bytes - header_size))
        return dict(datagrams=chunks, udp_payload=n + chunks * header_size, ip_udp=n + chunks * (header_size + 28))
    a3, b3 = l3(l2_agent), l3(l2_base)
    real = chunk_payload(wire, message_id=1, chunk_bytes=chunk_bytes)
    check = len(real) == a3["datagrams"] and sum(len(c) for c in real) == a3["udp_payload"]
    return dict(agent=dict(L1=l1_agent, L2=l2_agent, L3_udp=a3["udp_payload"], L3_ip=a3["ip_udp"], datagrams=a3["datagrams"]),
                baseline=dict(L1=sample["baseline"]["payload_bytes"], L2=l2_base, L3_udp=b3["udp_payload"], L3_ip=b3["ip_udp"],
                              datagrams=b3["datagrams"]),
                envelope_overhead_bytes=overhead, l2_method=method, l3_formula_matches_chunker=check)


def byte_summary(rows, key_fn):
    groups = {}
    for r in rows:
        for k in ("ALL", key_fn(r)):
            groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in groups.items():
        g = dict(n=len(rs))
        for layer in ("L1", "L2", "L3_udp", "L3_ip"):
            a = [r["bytes"]["agent"][layer] for r in rs]
            b = [r["bytes"]["baseline"][layer] for r in rs]
            g[layer] = dict(agent=q(a), baseline_offline_reencoded=q(b),
                            aggregate_agent=int(sum(a)), aggregate_baseline=int(sum(b)),
                            aggregate_saving_bytes=int(sum(b) - sum(a)), aggregate_saving_fraction=1 - sum(a) / sum(b),
                            paired_ratio_agent_over_baseline=q([x / y for x, y in zip(a, b)]))
        g["datagrams"] = dict(agent=q([r["bytes"]["agent"]["datagrams"] for r in rs], (50, 95)),
                              baseline=q([r["bytes"]["baseline"]["datagrams"] for r in rs], (50, 95)))
        out[k] = g
    return out


# ---------------------------------------------------------------------------- figures
def shade(ax, d, xkey="t_s"):
    for p in D.PHASES:
        xs = [x[xkey] for x in d if x["phase"] == p["name"] and x[xkey] is not None]
        if xs:
            ax.axvspan(min(xs), max(xs), color=PHASE_COLORS[p["name"]], alpha=0.6, lw=0, zorder=0)


def mark_transitions(ax, d, label=False):
    for t in D.PRIMARY_TRANSITIONS:
        x = next((r["t_s"] for r in d if r["step"] is not None and r["step"] >= t["step"]), None)
        if x is not None:
            ax.axvline(x, color="#010644", ls="--", lw=1)
            if label:
                ax.text(x, 1.02, t["name"], transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#010644")


def fig_timeline(d, roll, radio, out):
    fig, ax = plt.subplots(5, 1, figsize=(15, 13), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.3, 1, 1, 0.8]))
    for a in ax:
        shade(a, d)
        mark_transitions(a, d, label=a is ax[0])
    t = [x["t_s"] for x in d]
    ax[0].plot(radio["t"], radio["snr"], color="#9aa0a6", lw=0.8, label="applied target SNR (radio_trace, every 100-ms step)")
    ax[0].scatter(t, [x["snr_db"] for x in d], s=7, c=[STATE_COLOR.get(x["state"], "k") for x in d], zorder=3,
                  label="SNR admitted by each decision (causal ACK)")
    ax[0].set_ylabel("target UL SNR (dB)")
    ax[0].legend(loc="lower left", fontsize=8)
    rec = [x.get("ue_receipt") for x in d]
    ax[1].scatter([a for a, b in zip(t, rec) if b is not None], [b for b in rec if b is not None], s=7,
                  c=["#2e7d32" if b <= 170 else "#c62828" for b in rec if b is not None], zorder=3)
    miss = [a for a, b in zip(t, rec) if b is None]
    if miss:
        ax[1].scatter(miss, [ax[1].get_ylim()[1]] * len(miss), marker="x", color="k", label="missing reply")
    ax[1].axhline(170, color="k", ls=":", lw=1)
    ax[1].set_ylabel("feedback receipt (ms)")
    ax2 = ax[1].twinx()
    ax2.plot(t, roll, color="#010644", lw=1.4, label="trailing-20 on-time fraction")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("on-time (trailing 20)")
    ax2.legend(loc="lower left", fontsize=8)
    ax[2].scatter(t, [x["payload_bytes"] / 1e3 for x in d], s=7, color="#d35400")
    ax[2].set_ylabel("L2 SFD4 payload (KB)")
    ax[3].scatter(t, [x["q_e4"] for x in d], s=7, color="#5e35b1")
    ax[3].set_ylabel("q_e4")
    ax[4].scatter(t, [x["mode"] for x in d], s=7, color="#455a64")
    ax[4].set_ylabel("mode")
    ax[4].set_xlabel("time since trace step 0 (s, CLOCK_MONOTONIC_RAW)")
    fig.suptitle("LIVE: FADE_RECOVERY channel, feedback latency and selected q / payload (raw, unsmoothed except the labelled rolling line)")
    fig.tight_layout()
    fig.savefig(out / "fig01_timeline_snr_latency_q_payload.png", dpi=130)
    plt.close(fig)


def fig_transitions(d, roll, responses, out):
    prim = [r for r in responses if r.get("primary")]
    fig, ax = plt.subplots(4, len(prim), figsize=(5.5 * len(prim), 10), sharey="row", squeeze=False)
    for j, r in enumerate(prim):
        k = next((i for i, x in enumerate(d) if x["step"] is not None and x["step"] >= r["step"]), None)
        if k is None:
            continue
        lo, hi = max(0, k - 40), min(len(d), k + 40)
        rel = list(range(lo - k, hi - k))
        seg = d[lo:hi]
        ax[0, j].plot(rel, [x["snr_db"] for x in seg], ".-", color="#555")
        ax[1, j].plot(rel, [x["payload_bytes"] / 1e3 for x in seg], ".", color="#d35400")
        if r.get("band_bytes"):
            ax[1, j].axhspan(r["band_bytes"][0] / 1e3, r["band_bytes"][1] / 1e3, color="#d35400", alpha=0.12, label="pre-window P10-P90")
        ax[2, j].plot(rel, [x["q_e4"] for x in seg], ".", color="#5e35b1")
        ax[3, j].plot(rel, roll[lo:hi], color="#010644")
        ax[3, j].scatter(rel, [1.02 if x["on_time"] else -0.02 for x in seg], s=6, c=["#2e7d32" if x["on_time"] else "#c62828" for x in seg])
        for a in ax[:, j]:
            a.axvline(0, color="#010644", ls="--", lw=1)
            for key, c in (("first_response", "#ff9800"), ("sustained_response", "#e65100")):
                if r.get(key):
                    a.axvline(r[key]["decision_index"] - d[k]["decision_index"], color=c, lw=1)
        ax[0, j].set_title(f"{r['name']} (step {r['step']}, {r['direction']}): {r.get('class')}")
        ax[3, j].set_xlabel("decisions relative to first admitting decision")
    ax[0, 0].set_ylabel("admitted SNR (dB)")
    ax[1, 0].set_ylabel("L2 payload (KB)")
    ax[2, 0].set_ylabel("q_e4")
    ax[3, 0].set_ylabel("trailing-20 on-time")
    fig.suptitle("LIVE: response around the primary transitions (orange = first response, dark orange = sustained)")
    fig.tight_layout()
    fig.savefig(out / "fig02_transition_response.png", dpi=130)
    plt.close(fig)


def cdf_errors(samples, branch, cls):
    tol_vals = []
    elig = 0
    for s in samples:
        m = s[branch]["measurement"]
        elig += m[f"loc_{cls}_eligible_gt"]
        tol_vals.extend(m[f"loc_{cls}_matched_xy_errors_m"])
    v = np.sort(tol_vals)
    return v, (np.arange(1, v.size + 1) / elig) if elig else np.array([]), elig


def fig_cdf(samples, on_time, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for i, (view, ss) in enumerate((("all scored", samples), ("on-time only", on_time))):
        for j, cls in enumerate(D.TOLERANCE_M):
            a = ax[i, j]
            for br, (lab, col) in BRANCH.items():
                v, y, elig = cdf_errors(ss, br, cls)
                if v.size:
                    a.step(v, y, where="post", color=col, label=f"{lab}: R@{D.TOLERANCE_M[cls]}m="
                           f"{np.sum(v <= D.TOLERANCE_M[cls]) / elig:.3f}")
            a.axvline(D.TOLERANCE_M[cls], color="k", ls=":", lw=1)
            a.set_xlim(0, 3.0)
            a.set_ylim(0, 1)
            a.set_title(f"{cls} ({view}; {len(ss)} decisions, {cdf_errors(ss, 'full_local', cls)[2]} eligible GT)")
            a.set_xlabel("matched localization error (m); misses never reach the curve")
            a.set_ylabel("fraction of ALL eligible GT")
            a.grid(alpha=.3)
            a.legend(fontsize=8, loc="lower right")
    fig.suptitle("OFFLINE PAIRED: conservative localization CDFs on identical retained frames")
    fig.tight_layout()
    fig.savefig(out / "fig03_conservative_localization_cdf.png", dpi=130)
    plt.close(fig)


def fig_route_quality(d, samples, out):
    by_frame = {x["frame_id"]: x for x in d}
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for a in ax:
        shade(a, d)
        mark_transitions(a, d)
    for j, cls in enumerate(D.TOLERANCE_M):
        xs, ra, rc, el = [], [], [], []
        for s in samples:
            x = by_frame.get(s["frame_id"])
            ma, mc = m_counts(s["full_local"]["measurement"], cls), m_counts(s["agent_selected"]["measurement"], cls)
            if x is None or ma["elig"] == 0:
                continue
            xs.append(x["t_s"]); el.append(ma["elig"])
            ra.append(ma["hits"] / ma["elig"]); rc.append(mc["hits"] / mc["elig"])
        a = ax[j]
        a.plot(xs, ra, "o", ms=4, color=BRANCH["full_local"][1], label="full-local recall")
        a.plot(xs, rc, "x", ms=5, color=BRANCH["agent_selected"][1], label="agent recall")
        if len(xs) >= D.QUALITY_WINDOW_RETAINED:
            w = D.QUALITY_WINDOW_RETAINED
            hits_a = np.convolve(np.asarray(ra) * el, np.ones(w), "valid")
            hits_c = np.convolve(np.asarray(rc) * el, np.ones(w), "valid")
            ret = np.where(hits_a > 0, hits_c / np.where(hits_a > 0, hits_a, 1), np.nan)
            a.plot(xs[w - 1:], ret, color="#010644", lw=1.2, label=f"pooled retention, trailing {w} samples (smoothed)")
        b = a.twinx()
        b.bar(xs, el, width=0.6, color="#9e9e9e", alpha=0.35)
        b.set_ylabel("eligible GT (bars)")
        a.set_ylabel(f"{cls} recall @{D.TOLERANCE_M[cls]} m")
        a.set_ylim(-0.05, 1.3)
        a.legend(fontsize=8, loc="upper left", ncol=3)
    ax[2].scatter([by_frame[s["frame_id"]]["t_s"] for s in samples if s["frame_id"] in by_frame],
                  [s["agent_selected"]["q_perc"] if s["agent_selected"]["quality_defined"] else np.nan for s in samples
                   if s["frame_id"] in by_frame], s=10, color=BRANCH["agent_selected"][1], label="agent Q_perc")
    ax[2].scatter([by_frame[s["frame_id"]]["t_s"] for s in samples if s["frame_id"] in by_frame],
                  [s["full_local"]["q_perc"] if s["full_local"]["quality_defined"] else np.nan for s in samples
                   if s["frame_id"] in by_frame], s=10, color=BRANCH["full_local"][1], label="full-local Q_perc")
    ax[2].set_ylabel("Q_perc")
    ax[2].legend(fontsize=8)
    ax[2].set_xlabel("time since trace step 0 (s)")
    fig.suptitle("OFFLINE PAIRED: baseline-relative quality over the route (per retained decision; eligible counts as bars)")
    fig.tight_layout()
    fig.savefig(out / "fig04_baseline_relative_quality_route.png", dpi=130)
    plt.close(fig)


def fig_phase_summary(phases, pq_phase, bytes_phase, out):
    names = [p["name"] for p in D.PHASES if phases.get(p["name"], {}).get("n")]
    x = np.arange(len(names))
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ot = [phases[n]["on_time"] for n in names]
    ci = np.asarray([phases[n]["on_time_wilson"] for n in names])
    ax[0, 0].bar(x, ot, color="#00A3A3", yerr=[np.asarray(ot) - ci[:, 0], ci[:, 1] - np.asarray(ot)], capsize=4)
    for i, n in enumerate(names):
        ax[0, 0].text(i, 0.03, f"n={phases[n]['n']}\nSNR P50 {phases[n]['snr_db']['p50']:.1f}", ha="center", fontsize=8)
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].set_ylabel("on-time fraction (Wilson 95%)")
    ax[0, 0].set_title("LIVE timeliness by phase")
    for k, (cls, col) in enumerate((("vehicle", "#1f3a93"), ("person", "#8e24aa"))):
        vals = [(pq_phase.get(n) or {}).get(cls, {}).get("recall_retention") for n in names]
        lab = [(pq_phase.get(n) or {}).get(cls, {}).get("full_local", {}).get("eligible") for n in names]
        ax[0, 1].bar(x + (k - 0.5) * 0.35, [v if v is not None else 0 for v in vals], 0.35, color=col, label=f"{cls} recall retention")
        for i, (v, e) in enumerate(zip(vals, lab)):
            ax[0, 1].text(i + (k - 0.5) * 0.35, (v or 0) + 0.02, f"n={e}", ha="center", fontsize=7)
    ax[0, 1].axhline(1, color="k", lw=0.8)
    ax[0, 1].set_ylim(0, 1.3)
    ax[0, 1].set_title("OFFLINE PAIRED recall retention (agent / full-local hits), eligible GT shown")
    ax[0, 1].legend(fontsize=8)
    pa = [phases[n]["payload_L2_bytes"]["p50"] / 1e3 for n in names]
    pb = [((bytes_phase.get(n) or {}).get("L2") or {}).get("baseline_offline_reencoded", {}).get("p50") for n in names]
    ax[1, 0].bar(x - 0.2, pa, 0.4, color="#d35400", label="agent L2 P50, all decisions (LIVE)")
    ax[1, 0].bar(x + 0.2, [(v or 0) / 1e3 for v in pb], 0.4, color="#7f7f7f", label="noAE/UINT8/q0 L2 P50, retained (OFFLINE re-encode)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel("KB (log)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("payload by phase (same L2 layer)")
    stages = [("ue_encode_end", "UE input+encode"), ("uplink", "uplink"), ("prediction_at_service", "prediction at service"),
              ("gt_published", "GT ready"), ("result_sent", "result sent"), ("return_to_ue", "return to UE")]
    for k, (key, lab) in enumerate(stages):
        ax[1, 1].plot(x, [phases[n]["stages_ms"][key]["p50"] for n in names], "o-", label=lab)
    ax[1, 1].axhline(170, color="k", ls=":")
    ax[1, 1].set_ylabel("ms after action open (P50); 'uplink'/'return' are durations")
    ax[1, 1].set_title("LIVE runtime stages by phase")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.set_xticks(x)
        a.set_xticklabels([SHORT[n] for n in names], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig05_phase_timeliness_quality_payload_stages.png", dpi=130)
    plt.close(fig)


def fig_heatmap_freshness(d, out):
    names = [p["name"] for p in D.PHASES]
    qbins = np.arange(0, 10001, 1000)
    modes = sorted({x["mode"] for x in d})
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    snr_bins = np.arange(4, 26, 2)
    H = np.zeros((len(qbins) - 1, len(snr_bins) - 1))
    for x in d:
        if x["mode"] == 11:
            i = min(np.searchsorted(qbins, x["q_e4"], side="right") - 1, len(qbins) - 2)
            j = min(max(np.searchsorted(snr_bins, x["snr_db"], side="right") - 1, 0), len(snr_bins) - 2)
            H[i, j] += 1
    im = ax[0].imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[snr_bins[0], snr_bins[-1], 0, 10000])
    fig.colorbar(im, ax=ax[0], label="decisions")
    ax[0].set_xlabel("admitted SNR (dB)")
    ax[0].set_ylabel("q_e4 (mode 11 only)")
    ax[0].set_title(f"LIVE action heatmap, mode 11 ({sum(x['mode'] == 11 for x in d)}/{len(d)} decisions)")
    M = np.zeros((len(modes), len(names)))
    for x in d:
        if x["phase"] in names:
            M[modes.index(x["mode"]), names.index(x["phase"])] += 1
    M = M / np.maximum(M.sum(0, keepdims=True), 1)
    im = ax[1].imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax[1].set_yticks(range(len(modes)))
    ax[1].set_yticklabels([f"mode {m}" for m in modes])
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([n.split("_", 1)[0] for n in names])
    fig.colorbar(im, ax=ax[1], label="share of phase")
    ax[1].set_title("LIVE mode share by phase")
    for n in names:
        v = np.sort([x["freshness_capture_to_feedback_ms"] for x in d if x["phase"] == n and x.get("freshness_capture_to_feedback_ms")])
        if v.size:
            ax[2].step(v, np.arange(1, v.size + 1) / v.size, where="post", label=f"{n.split('_', 1)[0]} (n={v.size})")
    ax[2].set_xlabel("capture -> UE feedback (ms)")
    ax[2].set_ylabel("cumulative fraction")
    ax[2].set_title("LIVE freshness (capture to feedback) by phase")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig06_action_heatmap_freshness.png", dpi=130)
    plt.close(fig)


def fig_bytes(byte_rows, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, layer in enumerate(("L1", "L2")):
        for cond, col in STATE_COLOR.items():
            rs = [r for r in byte_rows if r["state"] == cond]
            ax[0].scatter([r["bytes"]["baseline"][layer] / 1e3 for r in rs], [r["bytes"]["agent"][layer] / 1e3 for r in rs],
                          s=12, color=col, marker="o" if layer == "L2" else ".", label=f"{cond} ({layer}, n={len(rs)})")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("noAE/UINT8/q0, OFFLINE re-encode (KB)")
    ax[0].set_ylabel("agent, transmitted (KB)")
    ax[0].set_title("paired bytes on identical retained frames")
    ax[0].legend(fontsize=7)
    for layer, ls in (("L1", ":"), ("L2", "-"), ("L3_udp", "--")):
        for br, col in (("agent", "#d35400"), ("baseline", "#7f7f7f")):
            v = np.sort([r["bytes"][br][layer] / 1e3 for r in byte_rows])
            ax[1].step(v, np.arange(1, v.size + 1) / v.size, where="post", color=col, ls=ls, label=f"{br} {layer}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("KB (log)")
    ax[1].set_ylabel("cumulative fraction of retained frames")
    ax[1].set_title("byte layers: L1 codec, L2 SFD4 frame, L3 UDP datagrams")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig07_paired_bytes.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def run(decisions, replay, out, comparator_too=True):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from phase2_map_sharing.transport import CHUNK_HEADER
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cell = next((ROOT / "capture/cells").iterdir())
    states, targets = design_states()
    chan, ack_step, acks = channel_evidence(cell, states, targets)
    snr_doc = json.loads((cell / "run5_policy/SNR_EVIDENCE.json").read_bytes())
    d = join(jl(decisions), snr_doc, ack_step, acks, states)
    actors = {}
    for x in d:
        meta = cell / f"numeric_gt/service/archive/frame_{x['frame_id']}/producer/input_metadata.json"
        if meta.exists():
            actors[x["frame_id"]] = len(json.loads(meta.read_bytes()).get("actor_boxes", []))
    with open(cell / "radio_trace.csv", newline="") as stream:
        tr = [r for r in csv.DictReader(stream) if r.get("command_ack_monotonic_ns")]
    t0_mono = int(tr[0]["command_ack_monotonic_ns"])
    radio = dict(t=[(int(r["command_ack_monotonic_ns"]) - t0_mono) / 1e9 for r in tr], snr=[float(r["target_snr_db"]) for r in tr])
    phases = phase_table(d, actors)
    p1 = phases["P1_GOOD_PRE"]["on_time"] if phases["P1_GOOD_PRE"].get("n") else float("nan")
    kpi, roll = kpi_recovery(d, p1)
    for w in (10,):
        roll10 = rolling([x["on_time"] for x in d], w)
    # transitions: primary + every recurring design-state change reached by the run
    reached = max(x["step"] for x in d if x["step"] is not None)
    changes = [t for t in D.transition_steps(states[:reached + 1])]
    responses = []
    for i, t in enumerate(changes):
        nxt = changes[i + 1]["step"] if i + 1 < len(changes) else None
        r = transition_response(d, t["step"], t["direction"], nxt)
        prim = next((p for p in D.PRIMARY_TRANSITIONS if p["step"] == t["step"]), None)
        r.update(from_state=t["from_state"], to_state=t["to_state"], primary=prim is not None,
                 name=prim["name"] if prim else f"T{t['step']}")
        responses.append(r)
    # primary transitions with the 'next transition' bound replaced by the next PRIMARY boundary (phase-level response)
    phase_level = []
    for j, p in enumerate(D.PRIMARY_TRANSITIONS):
        nxt = D.PRIMARY_TRANSITIONS[j + 1]["step"] if j + 1 < len(D.PRIMARY_TRANSITIONS) else 1114
        r = transition_response(d, p["step"], p["direction"], nxt)
        r.update(name=p["name"], primary=True, scope="phase-level (search bounded by next primary boundary)")
        phase_level.append(r)
    # ------------------------------------------------------------ paired quality + bytes
    rep = json.loads(Path(replay).read_bytes())
    adm = [r for r in rep["rows"] if r.get("admission") == "ADMITTED"]
    by_frame = {x["frame_id"]: x for x in d}
    for s in adm:
        x = by_frame.get(s["frame_id"], {})
        s["phase"], s["state"], s["on_time"] = x.get("phase"), x.get("state"), x.get("on_time")
    pq = dict(all_scored=paired_quality(adm, "ALL_SCORED"),
              on_time_only=paired_quality([s for s in adm if s["on_time"]], "ON_TIME_ONLY"))
    pq_phase = {}
    for p in [p["name"] for p in D.PHASES]:
        ss = [s for s in adm if s["phase"] == p]
        if ss:
            pq_phase[p] = paired_quality(ss, p)
    pq_state = {c: paired_quality([s for s in adm if s["state"] == c], c) for c in D.CONDITIONS if any(s["state"] == c for s in adm)}
    runtime = json.loads(CONFIG.read_bytes())["runtime"]
    chunk_bytes = int(runtime["udp_chunk_bytes"])
    retention_dir = ROOT / "paired_retention_v2"
    manifest = json.loads((retention_dir / "MANIFEST.json").read_bytes())
    entries = {int(e["join_key"]["frame_id"]): e for e in manifest["records"]} if isinstance(manifest.get("records"), list) else {}
    byte_rows = []
    for s in adm:
        e = entries.get(int(s["frame_id"]))
        wire = (retention_dir / e["wire_file"]).read_bytes() if e else None
        if wire is None or len(wire) != s["agent_payload_bytes"]:
            continue
        byte_rows.append(dict(frame_id=s["frame_id"], phase=s["phase"], state=s["state"], on_time=s["on_time"],
                              bytes=byte_layers(s, wire, chunk_bytes, CHUNK_HEADER.size, W, X)))
    overheads = {}
    for r in byte_rows:
        overheads[r["bytes"]["envelope_overhead_bytes"]] = overheads.get(r["bytes"]["envelope_overhead_bytes"], 0) + 1
    bytes_all = byte_summary(byte_rows, lambda r: r["phase"])
    bytes_state = byte_summary(byte_rows, lambda r: r["state"])
    # whole-run agent traffic (every transmitted frame: decisions, holds, fallbacks)
    whole = whole_run_traffic(cell, chunk_bytes, CHUNK_HEADER.size)
    # comparator (descriptive): FAVORABLE_STABLE retention capture over the same step ranges
    comp = comparator_by_phase() if comparator_too else None
    summary = dict(
        schema="splitfusion.run5_channel_response_v1.channel_analysis.v1",
        definitions_sha256=sha(Path(D.__file__)), decisions_file=str(decisions), decisions_sha256=sha(decisions),
        replay_file=str(replay), replay_sha256=sha(replay),
        evidence_classes=dict(LIVE="measured in this capture", OFFLINE_PAIRED="post-teardown replay of retained inputs",
                              DERIVED="computed from live measurements under stated assumptions; not measured"),
        LIVE=dict(channel_verification=chan, decisions=len(d), last_step_reached=reached,
                  phases=phases, kpi_recovery=kpi, transitions_recurring=responses, transitions_phase_level=phase_level,
                  overall=dict(on_time=float(np.mean([x["on_time"] for x in d])), receipt_ms=q([x.get("ue_receipt") for x in d]),
                               missing=sum(x.get("ue_receipt") is None for x in d)),
                  q_vs_snr_spearman=spearman([x["snr_db"] for x in d if x["mode"] == 11], [x["q_e4"] for x in d if x["mode"] == 11]),
                  payload_vs_snr_spearman=spearman([x["snr_db"] for x in d], [x["payload_bytes"] for x in d]),
                  whole_run_agent_traffic=whole),
        OFFLINE_PAIRED=dict(admitted=len(adm), admission_counts=rep["admission_counts"], quality=strip_rows(pq),
                            quality_by_phase=strip_rows(pq_phase), quality_by_condition=strip_rows(pq_state),
                            bytes_by_phase=bytes_all, bytes_by_condition=bytes_state, byte_rows=len(byte_rows),
                            byte_methods=sorted({r["bytes"]["l2_method"] for r in byte_rows}),
                            sfd4_overhead_bytes_observed=overheads,
                            l2_method_note="REPLAY.json stores only the baseline inner payload length/digest, so baseline L2 = "
                                           "baseline L1 + this frame's SFD4 overhead; the overhead was identical on every "
                                           "retained wire across all modes/q (fixed-width envelope), hence exact",
                            l3_formula_matches_chunker=all(r["bytes"]["l3_formula_matches_chunker"] for r in byte_rows),
                            labels=dict(baseline_sizes="OFFLINE_REENCODED", baseline_network_latency="UNMEASURED",
                                        full_local="NO_FEATURE_OFFLOAD_TRAFFIC (FP32 tensor never reported as transmitted)")),
        DESCRIPTIVE_COMPARATOR=comp)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float))
    with open(out / "DECISIONS_JOINED.jsonl", "w") as stream:
        for x in d:
            stream.write(json.dumps({k: v for k, v in x.items()}, sort_keys=True, default=float) + "\n")
    with open(out / "PAIRED_FRAMES.csv", "w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["frame_id", "phase", "state", "on_time", "cls", "eligible", "full_local_hits", "agent_hits", "uint8_hits",
                    "full_local_fp", "agent_fp", "agent_L1", "baseline_L1", "agent_L2", "baseline_L2", "agent_L3_udp", "baseline_L3_udp"])
        brows = {r["frame_id"]: r["bytes"] for r in byte_rows}
        for s in adm:
            for cls in D.TOLERANCE_M:
                a, b, c = (m_counts(s[br]["measurement"], cls) for br in ("full_local", "noae_uint8_q0", "agent_selected"))
                bb = brows.get(s["frame_id"], {})
                w.writerow([s["frame_id"], s["phase"], s["state"], s["on_time"], cls, a["elig"], a["hits"], c["hits"], b["hits"],
                            a["fp"], c["fp"], bb.get("agent", {}).get("L1"), bb.get("baseline", {}).get("L1"),
                            bb.get("agent", {}).get("L2"), bb.get("baseline", {}).get("L2"),
                            bb.get("agent", {}).get("L3_udp"), bb.get("baseline", {}).get("L3_udp")])
    fig_timeline(d, roll, radio, out)
    fig_transitions(d, roll, phase_level, out)
    fig_cdf(adm, [s for s in adm if s["on_time"]], out)
    fig_route_quality(d, adm, out)
    fig_phase_summary(phases, {k: v for k, v in pq_phase.items()}, bytes_all, out)
    fig_heatmap_freshness(d, out)
    if byte_rows:
        fig_bytes(byte_rows, out)
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir())
                                                         if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    return summary


def strip_rows(value):
    if isinstance(value, dict):
        return {k: strip_rows(v) for k, v in value.items() if k not in ("per_frame_rows",)}
    if isinstance(value, list):
        return [strip_rows(v) for v in value]
    return value


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return None
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def whole_run_traffic(cell, chunk_bytes, header):
    """Every transmitted frame of the run (decisions, holds, fallbacks). L2 = per_frame_metrics.payload_bytes (the
    transmitted SFD4 frame, verified equal to the retained wires); datagrams as recorded; L3 by the verified chunk
    formula. L1 is not recorded per frame in this capture (columns empty) and is therefore not reported whole-run."""
    ue = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_bytes())
    kind = {int(r["frame_id"]): r["run4_identity"].get("frame_kind", "UNKNOWN") for r in ue.get("transmitted_identities", [])}
    with open(cell / "per_frame_metrics.csv", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("payload_bytes")]
    kinds = {}
    mismatched_datagrams = 0
    for r in rows:
        k = kind.get(int(r["frame_id"]), "NOT_IN_TRANSMITTED_IDENTITIES")
        n = int(r["payload_bytes"])
        c = -(-n // (chunk_bytes - header))
        mismatched_datagrams += r.get("datagrams") not in ("", None) and int(r["datagrams"]) != c
        g = kinds.setdefault(k, dict(frames=0, L2_bytes=0, L3_udp_bytes=0, L3_ip_bytes=0, datagrams=0, _sizes=[]))
        g["frames"] += 1
        g["L2_bytes"] += n
        g["L3_udp_bytes"] += n + c * header
        g["L3_ip_bytes"] += n + c * (header + 28)
        g["datagrams"] += c
        g["_sizes"].append(n)
    for g in kinds.values():
        g["L2_per_frame"] = q(g.pop("_sizes"))
    tot = {key: sum(g[key] for g in kinds.values()) for key in ("frames", "L2_bytes", "L3_udp_bytes", "L3_ip_bytes", "datagrams")}
    return dict(source="per_frame_metrics.csv (payload_bytes, datagrams) joined to PHASE6_UE_EVIDENCE transmitted_identities",
                transmitted_identities=len(kind), frames_with_size=len(rows), by_kind=kinds, total=tot,
                datagram_formula_mismatches=int(mismatched_datagrams),
                L1_whole_run="NOT_RECORDED_PER_FRAME (only for the 98 retained wires)")


def comparator_by_phase():
    root, dec = COMPARATOR["root"], COMPARATOR["decisions"]
    if not dec.exists():
        return None
    cell = next((root / "capture/cells").iterdir())
    snr = json.loads((cell / "run5_policy/SNR_EVIDENCE.json").read_bytes())
    acks = jl(cell / "run5_policy/CONTROLLER_ACKS.jsonl")
    with open(cell / "radio_trace.csv", newline="") as stream:
        acked = [r for r in csv.DictReader(stream) if r["command_timing_status"] in ("ACK_ON_TIME", "ACK_LATE")]
    ack_step = {a["command_id"]: int(acked[int(a["command_id"].split("-")[1]) - 1]["step_index"])
                for a in acks if a.get("valid") and a.get("command_id") and int(a["command_id"].split("-")[1]) <= len(acked)}
    fs = [None] * 0
    d = join(jl(dec), snr, ack_step, acks, ["FAVORABLE"] * 100000)
    out = dict(note="FAVORABLE_STABLE retention capture, SAME trace-step ranges (same route time index); different session, "
                    "traffic and channel; DESCRIPTIVE ONLY", source=str(dec))
    for p in list(D.PHASES) + [D.ADV_CORE]:
        rs = [x for x in d if p["start"] <= (x["step"] if x["step"] is not None else -1) < p["end"]]
        if rs:
            out[p["name"]] = dict(n=len(rs), on_time=float(np.mean([x["on_time"] for x in rs])),
                                  snr_p50=float(np.median([x["snr_db"] for x in rs])),
                                  payload_L2_p50=float(np.median([x["payload_bytes"] for x in rs])),
                                  q_p50=float(np.median([x["q_e4"] for x in rs])),
                                  receipt_p50=float(np.median([x["ue_receipt"] for x in rs if x.get("ue_receipt") is not None])),
                                  uplink_p50=float(np.median([x["uplink_ms"] for x in rs if x["uplink_ms"] is not None])),
                                  vehicle_eligible_mean=float(np.mean([x["vehicle_eligible_gt"] or 0 for x in rs])),
                                  person_eligible_mean=float(np.mean([x["person_eligible_gt"] or 0 for x in rs])),
                                  live_agent_quality=live_quality(rs))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    s = run(a.decisions, a.replay, a.out)
    print(json.dumps(dict(decisions=s["LIVE"]["decisions"], channel=s["LIVE"]["channel_verification"]["passed"],
                          admitted=s["OFFLINE_PAIRED"]["admitted"]), default=float))
