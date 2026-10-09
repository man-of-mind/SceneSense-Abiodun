"""Per-decision rows, metrics and both reward definitions for a pilot live run (read-only, create-only outputs).

Rows: the retained boundary/quality row builder (``splitfusion_fixed_action_screen_analysis_v2.report.rows_for``)
plus, from the run's own Phase-6 evidence, the executed action and fallback reasons. Q_perc and Q_seg are recomputed
from each archived measurement with the pinned approved spec (must equal the live Q_perc exactly).
Outcome classes (exclusive): TIMELY_DEFINED (receipt <= 170 ms, quality defined), TIMELY_UNDEFINED (receipt <= 170 ms,
no eligible GT -> excluded EVALUATOR_FAULT), LATE (receipt > 170 ms), MISSING (no receipt).
Rewards (post-processing only; live semantics unchanged): current = Q_perc - 0.25 L/170 (TIMELY_DEFINED), -1 (LATE,
MISSING); candidate = Y x (0.7 + 0.3 Q_seg) (TIMELY_DEFINED), 0 (LATE, MISSING); TIMELY_UNDEFINED excluded from both.
Freshness: capture-to-feedback age = UE receipt - CARLA capture world-tick end (ms).
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import q as quant, wilson
from rl_agent.splitfusion_fixed_action_screen_analysis_v2.report import pooled, rows_for
from rl_agent.splitfusion_run5_restricted_pilot_v1 import support as SUP

TOL = {"vehicle": 1.0, "person": 1.2}
WEIGHT = {"person": 0.6, "vehicle": 0.4}
DEADLINE = 170.0


class LiveArm:
    def __init__(self, directory):
        self.dir = Path(directory).resolve(strict=True)
        self.reg = json.loads((self.dir / "registration.json").read_bytes())
        self.forced = None
        cells = list((self.dir / "capture/cells").iterdir())
        self.cell = cells[0] if len(cells) == 1 else None

    def j(self, relative, base=None):
        path = (base or self.cell) / relative
        return json.loads(path.read_bytes()) if path.exists() else None


def _spec():
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import scientific_basis as SB
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
    root = Path(__file__).resolve().parents[2]
    return Q, Q.load_reward_spec(root / SB.QUALITY_SOURCE_RELATIVE_PATH, SB.verify_quality_source(root))


def y_value(r):
    num = den = 0.0
    for c in TOL:
        el = r[f"{c}_eligible_gt"] or 0
        if el:
            num += WEIGHT[c] * sum(1 for e in (r[f"{c}_matched_xy_errors_m"] or []) if e <= TOL[c]) / el
            den += WEIGHT[c]
    return None if den == 0 else num / den


def rows(arm_dir):
    arm = LiveArm(arm_dir)
    out = rows_for(arm)
    evidence = arm.j("run4_phase6/PHASE6_UE_EVIDENCE.json")
    decisions = {int(d["frame_id"]): d for d in evidence["decisions"]}
    Q, spec = _spec()
    for r in out:
        d = decisions.get(int(r["frame_id"])) or {}
        r["executed_action"] = dict(mode_id=d.get("mode_id"), q_e4=d.get("q_e4"), anchor_action_id=d.get("anchor_action_id"),
                                    kind=d.get("kind"))
        if r["quality_defined"]:
            m = {"frame_id": r["frame_id"]}
            for c in TOL:
                for k in ("eligible_gt", "tp", "fn", "matched_xy_errors_m"):
                    m[f"loc_{c}_{k}"] = r[f"{c}_{k}"]
                for k in ("gt", "pred", "intersection", "union"):
                    m[f"seg_{c}_{k}_pixels"] = r[f"seg_{c}_{k}_pixels"]
            value = Q.evaluate_exact_quality(spec, m)
            r["q_seg"] = None if value.q_seg is None else float(value.q_seg)
            r["q_perc_recomputed"] = float(value.q_perc)
        else:
            r["q_seg"], r["q_perc_recomputed"] = None, None
        receipt = r["ue_receipt"]
        if receipt is None:
            r["outcome_class"] = "MISSING"
        elif receipt > DEADLINE:
            r["outcome_class"] = "LATE"
        else:
            r["outcome_class"] = "TIMELY_DEFINED" if r["quality_defined"] else "TIMELY_UNDEFINED"
        y = y_value(r) if r["quality_defined"] else None
        r["y_tolerance_recall"] = y
        if r["outcome_class"] == "TIMELY_DEFINED":
            r["current_reward"] = r["q_perc"] - 0.25 * receipt / DEADLINE
            r["candidate_reward"] = y * (1.0 if r["q_seg"] is None else 0.7 + 0.3 * r["q_seg"])
        elif r["outcome_class"] in ("LATE", "MISSING"):
            r["current_reward"], r["candidate_reward"] = -1.0, 0.0
        else:
            r["current_reward"] = r["candidate_reward"] = None
        r["freshness_capture_to_feedback_ms"] = (None if receipt is None or r["capture_world_tick_end"] is None
                                                 else receipt - r["capture_world_tick_end"])
    return arm, out, evidence


def summarize(arm, rs, evidence):
    n = len(rs)
    warm = [r for r in rs if not r["cold"]]
    classes = collections.Counter(r["outcome_class"] for r in rs)
    timely = [r for r in rs if r["outcome_class"] in ("TIMELY_DEFINED", "TIMELY_UNDEFINED")]
    scored = [r for r in rs if r["service_outcome"] == "SCORED"]
    timely_scored = [r for r in scored if r["outcome_class"] in ("TIMELY_DEFINED", "TIMELY_UNDEFINED")]
    included = [r for r in rs if r["current_reward"] is not None]
    opportunities = evidence["opportunities"]
    fallbacks = [d for d in evidence["decisions"] if d["kind"] == "FALLBACK"]
    recompute_exact = all(r["q_perc_recomputed"] == r["q_perc"] for r in rs if r["quality_defined"])

    def stage(name):
        return quant([r[name] for r in rs])

    def quality(sub):
        return dict(n=len(sub), defined=sum(bool(r["quality_defined"]) for r in sub),
                    q_perc=quant([r["q_perc"] for r in sub if r["quality_defined"]]),
                    q_seg=quant([r["q_seg"] for r in sub if r["q_seg"] is not None]),
                    y_tolerance_recall=quant([r["y_tolerance_recall"] for r in sub if r["y_tolerance_recall"] is not None]),
                    vehicle=pooled(sub, "vehicle"), person=pooled(sub, "person"))
    modes = collections.Counter(r["executed_action"]["mode_id"] for r in rs)
    return dict(
        budget=arm.reg["transmitted_budget"], decisions=n, warm_decisions=len(warm), opportunities=len(opportunities),
        admitted=sum(bool(o["admitted"]) for o in opportunities), fallbacks=len(fallbacks),
        fallback_reasons=dict(collections.Counter(x.split(":")[0] for d in fallbacks for x in d.get("fallback_reasons", []))),
        fallback_executed=sorted({(d["mode_id"], d["q_e4"], d.get("anchor_action_id")) for d in fallbacks}),
        outcome_classes=dict(classes), timely=len(timely), timely_wilson=wilson(len(timely), n),
        timely_over_opportunities=[len(timely), len(opportunities)],
        warm_timely=[sum(1 for r in warm if r["outcome_class"] in ("TIMELY_DEFINED", "TIMELY_UNDEFINED")), len(warm)],
        current_reward_mean=None if not included else float(np.mean([r["current_reward"] for r in included])),
        candidate_reward_mean=None if not included else float(np.mean([r["candidate_reward"] for r in included])),
        reward_rows=len(included), q_perc_recomputed_exact=recompute_exact,
        receipt_ms=dict(all=stage("ue_receipt"), warm=quant([r["ue_receipt"] for r in warm]),
                        cold=quant([r["ue_receipt"] for r in rs if r["cold"]])),
        boundaries_ms=dict(prediction_ready_edge=stage("edge_compute_end"), prediction_at_service=stage("service_prediction_received"),
                           gt_numeric_ready=stage("gt_numeric_ready_producer"), gt_ready_published=stage("gt_published"),
                           scoring_end=stage("scoring_worker_end"), ue_receipt=stage("ue_receipt"),
                           edge_compute=quant([r["edge_compute_end"] - r["edge_compute_start"] for r in rs
                                               if r["edge_compute_end"] is not None and r["edge_compute_start"] is not None]),
                           feedback_return=quant([r["ue_receipt"] - r["scoring_worker_end"] for r in rs
                                                  if r["ue_receipt"] is not None and r["scoring_worker_end"] is not None])),
        gt_first=sum(bool(r["gt_first"]) for r in rs),
        freshness_capture_to_feedback_ms=dict(all=quant([r["freshness_capture_to_feedback_ms"] for r in rs]),
                                              timely=quant([r["freshness_capture_to_feedback_ms"] for r in timely])),
        payload_bytes=quant([r["payload_bytes"] for r in rs]), snr_db=quant([r["snr_db"] for r in rs]),
        prior_ul_mcs=quant([r["prior_ul_mcs"] for r in rs]), camera_si=quant([r["camera_si"] for r in rs]),
        radar_p40=quant([r["radar_p40"] for r in rs]), backlog_bytes=quant([r["pre_action_rlc_backlog_bytes"] for r in rs]),
        mode_share={m: modes.get(m, 0) / n for m in SUP.ALLOWED_MODES},
        q_by_mode={m: quant([r["executed_action"]["q_e4"] for r in rs if r["executed_action"]["mode_id"] == m])
                   for m in SUP.ALLOWED_MODES},
        quality_all_scored=quality(scored), quality_timely_scored=quality(timely_scored))
