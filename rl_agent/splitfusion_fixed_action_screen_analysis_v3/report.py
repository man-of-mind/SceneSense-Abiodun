"""Three-arm frontier table, OTR and comparisons for the v3 D10 measurement (read-only on attempts, create-only out).

Per-decision rows: the v2 row builder (``report.rows_for``) unchanged. Per-arm metrics: v2 ``arm_metrics`` and
``report_combined.admission`` unchanged, plus:
  * OTR (decision memo section 2.1): 1[receipt <= 170 ms] x class-weighted within-tolerance recall; undefined without
    eligible GT; missing receipt counts as late; per opportunity, fallbacks count 0;
  * separated intervals from action-open: prediction at service, GT ready (publish), prediction-request and
    target-request worker waits, scoring end -> UE receipt (feedback return);
  * every quality metric for all scored and for timely scored decisions.
Comparisons: primary against the NEW action-67 arm (unpaired decision bootstrap and ego-pose route-tick pairing with a
moving-block bootstrap; actor/eligibility differences at shared ticks are counted). Historical seven-arm a67 secondary.
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import Arm, MAIN, WT, q, wilson
from rl_agent.splitfusion_fixed_action_screen_analysis_v2.report import arm_metrics, pooled, rows_for, pipeline_identity
from rl_agent.splitfusion_fixed_action_screen_analysis_v2.report_combined import admission
from rl_agent.splitfusion_quality_pilot_design_v1 import analysis as QA

SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_m11_frontier_v3"
OUT = MAIN / "experiments/splitfusion_fixed_action_screen_v3"
HISTORICAL = MAIN / "experiments/splitfusion_fixed_action_screen_v2/results_combined_01/DECISIONS.jsonl"
DEADLINE = 170.0
BOOT = 4000


def y(row):
    return QA.screen_y(row)


def otr(row):
    v = y(row)
    if v is None:
        return None
    return v if row["ue_receipt"] is not None and row["ue_receipt"] <= DEADLINE else 0.0


def ci(values, seed):
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (BOOT, v.size))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def block_ci(diffs, seed, block=3):
    """Moving-block bootstrap over the tick-ordered paired differences (temporal dependence)."""
    d = np.asarray(diffs, float)
    n = d.size
    if n == 0:
        return None
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = math.ceil(n / block)
    means = []
    for _ in range(BOOT):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:n]
        means.append(d[idx].mean())
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def gap(rows, a, b):
    return q([r[b] - r[a] for r in rows if r.get(a) is not None and r.get(b) is not None])


def arm_report(arm, rows):
    m = arm_metrics(rows)
    adm = admission(arm, rows)
    warm = [r for r in rows if not r["cold"]]
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    otr_adm = [otr(r) for r in rows]
    defined_otr = [v for v in otr_adm if v is not None]
    opps = adm["opportunities"]
    fallbacks = adm["fallbacks"]["count"]
    timely = sum(r["timely"] for r in rows)
    sub = lambda s: dict(n=len(s), vehicle=pooled(s, "vehicle"), person=pooled(s, "person"),
                         q_perc=q([r["q_perc"] for r in s if r["quality_defined"]]))
    out = dict(
        arm=arm.dir.name, forced_action=arm.forced, pipeline=pipeline_identity(arm),
        admission=adm, opportunities=opps, admitted=len(rows), fallbacks=adm["fallbacks"],
        timely_over_admitted=[timely, len(rows)], timely_over_admitted_wilson=wilson(timely, len(rows)),
        timely_over_opportunities=[timely, opps], timely_over_opportunities_wilson=wilson(timely, opps),
        warm_timely=[sum(r["timely"] for r in warm), len(warm)], warm_timely_wilson=wilson(sum(r["timely"] for r in warm), len(warm)),
        receipt_ms=dict(cold_inclusive=q([r["ue_receipt"] for r in rows]), warm=q([r["ue_receipt"] for r in warm]),
                        cold=q([r["ue_receipt"] for r in rows if r["cold"]])),
        intervals_from_action_open_ms=dict(
            prediction_ready_edge=q([r["edge_compute_end"] for r in rows]),
            prediction_at_service=q([r["service_prediction_received"] for r in rows]),
            gt_ready_published=q([r["gt_published"] for r in rows]),
            gt_numeric_ready_producer=q([r["gt_numeric_ready_producer"] for r in rows]),
            scoring_worker_end=q([r["scoring_worker_end"] for r in rows]), ue_receipt=q([r["ue_receipt"] for r in rows])),
        waits_ms=dict(prediction_request_wait=gap(rows, "service_prediction_received", "prediction_worker_begin"),
                      target_request_wait=gap(rows, "gt_published", "target_worker_begin"),
                      prediction_request_service=gap(rows, "prediction_worker_begin", "prediction_worker_end"),
                      target_request_service=gap(rows, "target_worker_begin", "target_worker_end"),
                      feedback_return_scoring_end_to_ue_receipt=gap(rows, "scoring_worker_end", "ue_receipt"),
                      edge_compute=gap(rows, "edge_compute_start", "edge_compute_end"),
                      uplink_first_send_to_reassembly=gap(rows, "ue_first_send", "edge_reassembly_complete")),
        gt_first=[sum(r["gt_first"] for r in rows), len(rows)],
        payload_bytes=q([r["payload_bytes"] for r in rows]), datagrams=q([r["datagrams_at_edge"] for r in rows]),
        prior_ul_mcs=q([r["prior_ul_mcs"] for r in rows]), snr_db=q([r["snr_db"] for r in rows]),
        backlog_bytes=q([r["pre_action_rlc_backlog_bytes"] for r in rows]), camera_si=q([r["camera_si"] for r in rows]),
        radar_p40=q([r["radar_p40"] for r in rows]),
        eligible_gt=dict(vehicle=sum(r["vehicle_eligible_gt"] or 0 for r in rows), person=sum(r["person_eligible_gt"] or 0 for r in rows)),
        quality_counts=dict(defined=sum(bool(r["quality_defined"]) for r in scored),
                            undefined=sum(r["service_outcome"] == "SCORED" and not r["quality_defined"] for r in rows),
                            missing_not_scored=sum(r["service_outcome"] != "SCORED" for r in rows),
                            outcomes=dict(collections.Counter(r["service_outcome"] for r in rows))),
        otr=dict(defined=len(defined_otr), undefined=len(rows) - len(defined_otr),
                 mean_over_defined_admitted=None if not defined_otr else round(float(np.mean(defined_otr)), 4),
                 ci_over_defined_admitted=ci(defined_otr, 1),
                 per_opportunity_approx=None if not defined_otr else round(float(np.mean(defined_otr)) * len(rows) / (len(rows) + fallbacks), 4),
                 warm_mean=None if not [otr(r) for r in warm if otr(r) is not None] else
                 round(float(np.mean([otr(r) for r in warm if otr(r) is not None])), 4)),
        quality_all_scored=sub(scored), quality_timely_scored=sub([r for r in scored if r["timely"]]),
        box_footprint_iou="NOT_RECORDED_BY_THE_LIVE_EVALUATOR (segmentation IoU reported instead)",
        retained_metrics=m)
    return out


def paired(rows_a, rows_b, f, seed):
    """b minus a at shared route ticks (ego pose identical; other actors may differ)."""
    A = {int(r["route_tick"]): r for r in rows_a if r["route_tick"] is not None}
    B = {int(r["route_tick"]): r for r in rows_b if r["route_tick"] is not None}
    ticks = sorted(set(A) & set(B))
    diffs, same_eligible, same_pose = [], 0, 0
    for t in ticks:
        a, b = f(A[t]), f(B[t])
        if a is None or b is None:
            continue
        diffs.append(b - a)
        same_eligible += (A[t]["vehicle_eligible_gt"], A[t]["person_eligible_gt"]) == (B[t]["vehicle_eligible_gt"], B[t]["person_eligible_gt"])
        pa, pb = A[t].get("ego_world") or [None, None], B[t].get("ego_world") or [None, None]
        same_pose += None not in pa[:2] + pb[:2] and math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= 0.01
    return dict(shared_ticks=len(ticks), pairs=len(diffs), mean_difference=None if not diffs else round(float(np.mean(diffs)), 4),
                block_bootstrap_ci=block_ci(diffs, seed), eligible_counts_identical=same_eligible, ego_pose_identical=same_pose)


def run(name):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=False)
    keys = json.loads((OUT / "registration_01/SELECTION.json").read_bytes())["arm_order"]
    arms = {k: Arm(SCREEN / k) for k in keys if (SCREEN / k / "capture").exists()}
    rows = {k: rows_for(a) for k, a in arms.items()}
    with open(out / "DECISIONS.jsonl", "x") as stream:
        for k in rows:
            for r in rows[k]:
                stream.write(json.dumps(dict(r, otr=otr(r)), sort_keys=True, default=str) + "\n")
    reports = {k: arm_report(arms[k], rows[k]) for k in rows}
    ref = "arm0_m11_q3000_a67"
    comparisons = {}
    if ref in rows:
        for k in rows:
            if k == ref:
                continue
            cur = lambda r: None if r["ue_receipt"] is None or r["ue_receipt"] > DEADLINE else (
                None if not r["quality_defined"] else r["q_perc"] - 0.25 * r["ue_receipt"] / DEADLINE)
            cur_full = lambda r: -1.0 if (r["ue_receipt"] is None or r["ue_receipt"] > DEADLINE) else cur(r)
            timely = lambda r: float(r["timely"])
            comparisons[f"{k}_minus_{ref}"] = dict(
                unpaired=dict(
                    otr=dict(difference=round(float(np.mean([v for v in map(otr, rows[k]) if v is not None])
                                                   - np.mean([v for v in map(otr, rows[ref]) if v is not None])), 4),
                             ci=_diff_ci([v for v in map(otr, rows[k]) if v is not None], [v for v in map(otr, rows[ref]) if v is not None], 11)),
                    timely=dict(difference=round(float(np.mean([r["timely"] for r in rows[k]]) - np.mean([r["timely"] for r in rows[ref]])), 4),
                                ci=_diff_ci([timely(r) for r in rows[k]], [timely(r) for r in rows[ref]], 12)),
                    current_reward=dict(difference=round(float(np.mean([v for v in map(cur_full, rows[k]) if v is not None])
                                                             - np.mean([v for v in map(cur_full, rows[ref]) if v is not None])), 4),
                                        ci=_diff_ci([v for v in map(cur_full, rows[k]) if v is not None],
                                                    [v for v in map(cur_full, rows[ref]) if v is not None], 13))),
                route_tick_paired=dict(otr=paired(rows[ref], rows[k], otr, 21), timely=paired(rows[ref], rows[k], timely, 22),
                                       current_reward=paired(rows[ref], rows[k], cur_full, 23)))
    historical = [json.loads(l) for l in open(HISTORICAL)]
    hist67 = [r for r in historical if r["arm"] == "arm4_a67"]
    secondary = dict(source=str(HISTORICAL), seven_arm_a67=dict(
        decisions=len(hist67), timely=sum(r["timely"] for r in hist67),
        receipt_ms=q([r["ue_receipt"] for r in hist67]),
        otr_mean=round(float(np.mean([v for v in map(otr, hist67) if v is not None])), 4),
        vehicle=pooled(hist67, "vehicle"), person=pooled(hist67, "person")))
    result = dict(schema="splitfusion.fixed_action_screen_v3.results.v1", arms=reports, comparisons_vs_new_a67=comparisons,
                  historical_secondary=secondary, definitions=dict(otr=QA.__doc__.split("Quantities")[1][:600],
                  deadline_ms_inclusive=DEADLINE, cold_decisions=2, bootstrap=BOOT,
                  route_tick_pairing="ego pose identical at shared ticks; other actors may differ; moving-block bootstrap, block 3"))
    (out / "ANALYSIS.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    return result


def _diff_ci(a, b, seed):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, a.size, (BOOT, a.size))].mean(1) - b[rng.integers(0, b.size, (BOOT, b.size))].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
