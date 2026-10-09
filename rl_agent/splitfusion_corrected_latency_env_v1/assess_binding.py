"""Read-only assessment of one calibration binding (e.g. the fixed-action screen once sealed and bound).

Reports, from measured chains only:
  * exact recomposition of every chain (boundary accounting);
  * per-stratum (= per fixed action) timely count at the inclusive 170 ms, receipt distribution, ordering mix and
    per-frame stage distributions (numpy linear quantiles), plus the measured q_perc definedness;
  * the fraction of FIT training scenes inside each stratum's measured SI/P40 support (scene coverage);
  * the launch blockers.
Writes create-only ASSESSMENT.json. CPU only; do not run during a live window.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -m \
        rl_agent.splitfusion_corrected_latency_env_v1.assess_binding --binding <json> --output <new dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import calibration as K
from . import chains as CH
from . import outcome as O
from .scene_catalog import FitSceneCatalogWithUndefinedV1

STAGES = ("A", "B", "C", "D", "E", "F", "H", "G", "R")
# Current localisation-tolerance reference (pending quality-design review; not a reward or acceptance threshold).
TOLERANCE_M = {"vehicle": 1.0, "person": 1.2}
DESCRIPTIVE_Q = 0.4      # descriptive count only; not an agreed quality-acceptance threshold


def _class_quality(chains) -> dict:
    """Pooled per-class detection/localisation/segmentation over the given decisions (sum over frames)."""
    out = {}
    for cls, tol in TOLERANCE_M.items():
        eligible = tp = fp = 0
        errors = []
        for c in chains:
            q = c.quality
            if f"{cls}_eligible_gt" not in q:
                return {"available": False}
            eligible += int(q[f"{cls}_eligible_gt"])
            tp += int(q[f"{cls}_tp"])
            fp += int(q[f"{cls}_fp"])
            errors += [float(e) for e in q[f"{cls}_matched_xy_errors_m"]]
        within = sum(1 for e in errors if e <= tol)
        inter = sum(int(c.quality.get(f"seg_{cls}_intersection_pixels", 0)) for c in chains)
        union = sum(int(c.quality.get(f"seg_{cls}_union_pixels", 0)) for c in chains)
        out[cls] = dict(tolerance_m=tol, eligible_gt=eligible,
                        recall_within_tolerance=None if eligible == 0 else round(within / eligible, 4),
                        recall_any_match_3m_gate=None if eligible == 0 else round(tp / eligible, 4),
                        precision=None if tp + fp == 0 else round(tp / (tp + fp), 4),
                        matched_error_m=_q(errors), seg_iou_pooled=None if union == 0 else round(inter / union, 4))
    out["available"] = True
    return out


def _q(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), p50=round(float(np.percentile(v, 50)), 3), p95=round(float(np.percentile(v, 95)), 3),
                min=round(float(v.min()), 3), max=round(float(v.max()), 3))


def assess(binding_path: Path) -> dict:
    cal = K.load_binding(binding_path)
    worst = 0.0
    for cid, chain in cal.chains.items():
        comp = CH.compose(cal.vectors[cid])
        for name, value in comp.instants.items():
            if chain.instants.get(name) is not None:
                worst = max(worst, abs(value - chain.instants[name]))
    catalog = FitSceneCatalogWithUndefinedV1()
    descriptors = [catalog.scene_descriptors(k) for k in catalog.keys]
    strata = {}
    for key in sorted({(c.context.network_profile, c.context.mode_id, c.context.q_e4) for c in cal.chains.values()}):
        profile, mode_id, q_e4 = key
        members = cal.stratum(mode_id, q_e4, profile)
        vectors = [cal.vectors[c.chain_id] for c in members]
        receipts = [c.instants["ue_receipt"] for c in members]
        outcomes = [O.classify(O.receipt_elapsed_ns(r), c.q_perc) for r, c in zip(receipts, members)]
        si = [c.context.camera_si for c in members]
        p40 = [c.context.radar_p40 for c in members]
        pad_si = cal.support_margin * (max(si) - min(si))
        pad_p = cal.support_margin * (max(p40) - min(p40))
        inside = sum(1 for s, p in descriptors if min(si) - pad_si <= s <= max(si) + pad_si
                     and min(p40) - pad_p <= p <= max(p40) + pad_p)
        strata[f"{profile}|mode{mode_id}|q{q_e4}"] = dict(
            n=len(members), usable=len(members) >= cal.min_chains,
            family=members[0].context.family, quantizer=members[0].context.quantizer,
            wire_bytes=_q([c.context.wire_bytes for c in members]),
            timely_le_170=sum(1 for r in receipts if r is not None and O.receipt_elapsed_ns(r) <= O.DEADLINE_NS),
            missing_receipt=sum(1 for r in receipts if r is None),
            outcomes={k: sum(1 for o in outcomes if o.kind.value == k)
                      for k in ("DELIVERED_SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
            receipt_ms=_q(receipts),
            prediction_at_service_ms=_q([CH.compose(v).prediction_at_service for v in vectors]),
            gt_first=sum(1 for v in vectors if v.realized_ordering is CH.Ordering.GT_FIRST),
            gt_ready_ms=_q([CH.compose(v).gt_ready for v in vectors]),
            gt_segments_ms={name: _q([dict(v.gt_segments)[name] for v in vectors])
                            for name, _ in vectors[0].gt_segments},
            stages_ms={s: _q([getattr(v, s) for v in vectors]) for s in STAGES},
            score_requests_ms={role: {part: _q([getattr(getattr(v.costs, role), part) for v in vectors
                                                if getattr(v.costs, role) is not None])
                                      for part in ("dispatch", "send", "execute", "reply")}
                               for role in ("target_admit", "prediction_score", "prediction_store", "target_score")},
            mcs=_q([c.context.prior_ul_mcs for c in members]),
            joint_latency_quality=dict(
                timely_defined=sum(1 for r, c in zip(receipts, members) if r is not None
                                   and O.receipt_elapsed_ns(r) <= O.DEADLINE_NS and c.q_perc is not None),
                timely_q_perc=_q([c.q_perc for r, c in zip(receipts, members) if r is not None
                                  and O.receipt_elapsed_ns(r) <= O.DEADLINE_NS and c.q_perc is not None]),
                timely_q_perc_ge_0_4_descriptive=sum(
                    1 for r, c in zip(receipts, members) if r is not None and O.receipt_elapsed_ns(r) <= O.DEADLINE_NS
                    and c.q_perc is not None and c.q_perc >= DESCRIPTIVE_Q),
                all_scored_q_perc=_q([c.q_perc for c in members]),
                classes_all_scored=_class_quality(members),
                classes_timely=_class_quality([c for r, c in zip(receipts, members) if r is not None
                                               and O.receipt_elapsed_ns(r) <= O.DEADLINE_NS])),
            score_channel_carry_in_binding=sum(
                1 for v in vectors if v.carry_in is not None
                and v.carry_in > min(CH.compose(v).prediction_at_service, CH.compose(v).gt_ready)),
            fit_scenes_inside_scene_support=f"{inside}/{len(descriptors)}")
    return dict(schema="splitfusion.corrected_latency_env.binding_assessment.v1",
                binding_path=str(binding_path), binding_sha256=cal.binding_sha256,
                binding_file_sha256=K.sha256_file(binding_path), chains=len(cal.chains),
                composition_model=CH.COMPOSITION_MODEL, tolerance_reference_m=TOLERANCE_M,
                descriptive_q_threshold_note="Q_perc >= 0.4 is a descriptive count, not an agreed acceptance threshold",
                worst_recomposition_error_ms=worst, coverage=cal.coverage(), strata=strata,
                launch_blockers=cal.launch_blockers(), candidate_source_digest=K.candidate_source_digest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    document = assess(Path(args.binding))
    (out / "ASSESSMENT.json").open("x").write(json.dumps(document, indent=1, sort_keys=True))
    print(json.dumps({k: document[k] for k in ("chains", "worst_recomposition_error_ms", "launch_blockers")}, indent=1))


if __name__ == "__main__":
    main()
