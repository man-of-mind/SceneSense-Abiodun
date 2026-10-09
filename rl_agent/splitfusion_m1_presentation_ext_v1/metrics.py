"""Deadline-aware localisation/segmentation for the four retained M1 live sessions (analysis only).

Primary boundary: action-open -> successful live ``edge.compute`` end with the CPU prediction materialized
(the final_03 definition, reused through ``analyze_v3.load_session``). Feedback receipt is reported separately.

Per policy decision (an *opportunity*):
  prediction state  TIMELY   completed and latency <= 170 ms
                    LATE     completed and latency > 170 ms
                    NONE     no completed prediction (STALE/timeout/no result are terminal subcategories)
  GT state          MEASURED      live scorer measurement (objects, matched errors, segmentation counts)
                    OBJECTS_ONLY  no measurement; eligible-object counts re-derived from the archived ready target
                                  with the scorer's own frozen ``load_gt`` + AVO>=0.65 rule (validated exactly against
                                  every MEASURED frame); no GT segmentation raster was retained
                    MISSING       no GT target at all (unknown; never zero, never silently dropped)

Deadline-aware recall  R_c(e,170) = sum_i [TIMELY_i] * within_e_i / sum_i eligible_i  over GT-known opportunities.
Deadline-aware IoU: a LATE/NONE prediction is an empty foreground mask -> intersection 0, union = GT pixels.
If GT is unknown on any failed opportunity, the reported value is an upper bound on the whole-session value
(unknown objects/pixels can only enlarge the denominator) and is labelled as such.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from rl_agent.splitfusion_m1_baselines_v1 import analyze_v3 as A3

MAIN = Path(__file__).resolve().parents[2]
PROFILER = MAIN.parent / ("abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/"
                          "splitfusion_single_gpu_profiler_v1")
ROOTS = {
    "agent": PROFILER / "20261007_m1_baselines_v1/agent",
    "action0": PROFILER / "20261007_m1_baselines_v1/static_attempt_02",
    "arm_a": PROFILER / "20261007_m1_baselines_v3/fixed_ae32_uint8_a54",
    "arm_b": PROFILER / "20261007_m1_baselines_v3/fixed_mode11_q6000",
}
FINAL_03 = MAIN / "experiments/splitfusion_m1_baselines_v1/final_03"
DEADLINE_MS = 170.0
TOL = A3.TOL                                   # vehicle 1.0 m, person 1.2 m (registered)
CLASSES = ("vehicle", "person")
AVO_THRESHOLD = 0.65                           # quality_adapter.AVO_THRESHOLD / scoring.join_admitted_inputs
THRESHOLDS = np.round(np.arange(0, 3.0001, 0.01), 2)
BOOT = 2000
SEED = 20261009


def sha(path):
    return A3.sha(path)


class EligibilityFromTarget:
    """Scorer-identical eligible-GT counts from an archived ready target (no prediction needed)."""

    def __init__(self):
        from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
        self._scorer = S.NumericalScorer()        # verifies the pinned evaluator sources

    def __call__(self, meta):
        namespace = dict(self._scorer._detection, read_csv=lambda _: meta["object_boxes"])
        grouped, _ = FunctionType(namespace["load_gt"].__code__, namespace)(Path("/offline"), "v010")
        gt = grouped.get(meta["sample_id"], [])
        return dict(vehicle=sum(g["class_name"] == "vehicle" for g in gt),
                    person=sum(float(r["avo"]) >= AVO_THRESHOLD for r in meta["avo"]["qualified"]))


def opportunities(key, session, eligibility):
    cell = A3.find_one(Path(session["root"]) / "capture", "CELL_RESULT.json").parent
    archive = cell / "numeric_gt/service/archive"
    out, validated = [], 0
    for row in session["rows"]:
        fid = int(row["frame_id"])
        completed = row["prediction_materialized"] is not None
        latency = row["edge_compute_end"] if completed else None
        state = "NONE" if not completed else ("TIMELY" if latency <= DEADLINE_MS else "LATE")
        target = archive / f"frame_{fid}/scorer/target_metadata.json"
        derived = eligibility(json.loads(target.read_bytes())) if target.exists() else None
        measured = row["vehicle_eligible_gt"] is not None
        if measured:
            if derived is None or derived != {c: row[f"{c}_eligible_gt"] for c in CLASSES}:
                raise ValueError(f"{key} frame {fid}: target-derived eligibility differs from live scorer")
            validated += 1
            gt_state = "MEASURED"
        else:
            gt_state = "OBJECTS_ONLY" if derived is not None else "MISSING"
        if measured and not completed:
            raise ValueError(f"{key} frame {fid}: measurement without completed prediction")
        rec = dict(session=key, decision_index=row["decision_index"], frame_id=fid,
                   prediction_state=state, prediction_latency_ms=latency, service_outcome=row["service_outcome"],
                   gt_state=gt_state, feedback_receipt_ms=row["ue_receipt"],
                   feedback_by_170=row["ue_receipt"] is not None and row["ue_receipt"] <= DEADLINE_MS,
                   payload_bytes=row["payload_bytes"])
        for c in CLASSES:
            errors = [float(e) for e in (row[f"{c}_matched_xy_errors_m"] or [])]
            rec[f"{c}_eligible"] = (row[f"{c}_eligible_gt"] if measured
                                    else None if derived is None else derived[c])
            rec[f"{c}_errors"] = errors
            rec[f"{c}_within_tol"] = sum(e <= TOL[c] for e in errors) if measured else None
            for k in ("tp", "fp"):
                rec[f"{c}_{k}"] = row[f"{c}_{k}"]
            for k in ("gt", "intersection", "union"):
                rec[f"seg_{c}_{k}"] = row[f"seg_{c}_{k}_pixels"]
        out.append(rec)
    return out, validated


def _sum(rows, field):
    return int(sum(r[field] or 0 for r in rows))


def localisation(opps, cls):
    """Object-instance accounting; every count is an object count."""
    known = [r for r in opps if r["gt_state"] != "MISSING"]
    completed = [r for r in opps if r["prediction_state"] != "NONE"]
    timely = [r for r in opps if r["prediction_state"] == "TIMELY"]
    late = [r for r in opps if r["prediction_state"] == "LATE"]
    tol = TOL[cls]
    den = _sum(known, f"{cls}_eligible")
    den_completed = _sum(completed, f"{cls}_eligible")
    within_timely = sum(e <= tol for r in timely for e in r[f"{cls}_errors"])
    within_late = sum(e <= tol for r in late for e in r[f"{cls}_errors"])
    within_completed = within_timely + within_late
    failed_known = [r for r in opps if r["prediction_state"] == "NONE" and r["gt_state"] != "MISSING"]
    missing = [r for r in opps if r["gt_state"] == "MISSING"]
    tp_c, fp_c = _sum(completed, f"{cls}_tp"), _sum(completed, f"{cls}_fp")
    tp_t, fp_t = _sum(timely, f"{cls}_tp"), _sum(timely, f"{cls}_fp")
    if not missing:
        bound = "EXACT"
    elif any(r["prediction_state"] != "NONE" for r in missing):
        bound = "UNBOUNDED_UNKNOWN_GT_ON_COMPLETED_OPPORTUNITY"
    elif within_timely == 0 and den > 0:
        bound = "EXACT_ZERO_NO_TIMELY_PREDICTION"
    else:
        bound = "UPPER_BOUND_UNKNOWN_GT_ON_FAILED_OPPORTUNITIES"

    def curve(rows, denominator):
        errors = np.sort([e for r in rows for e in r[f"{cls}_errors"]])
        return [None if not denominator else float(np.searchsorted(errors, t, side="right") / denominator)
                for t in THRESHOLDS]

    return dict(
        tolerance_m=tol,
        all_completion=dict(
            definition="within-tolerance matches on every completed prediction / eligible GT objects on completed frames",
            numerator=within_completed, denominator=den_completed,
            recall=A3.ratio(within_completed, den_completed),
            completed_frames=len(completed), precision=A3.ratio(tp_c, tp_c + fp_c), tp=tp_c, fp=fp_c),
        deadline_aware=dict(
            definition="within-tolerance matches on predictions completed <=170 ms / eligible GT objects on every "
                       "GT-known opportunity (late or absent predictions contribute zero)",
            numerator=within_timely, denominator=den, recall=A3.ratio(within_timely, den),
            bound=bound,
            gt_known_opportunities=len(known), opportunities=len(opps),
            precision_timely=A3.ratio(tp_t, tp_t + fp_t), tp_timely=tp_t, fp_timely=fp_t),
        object_accounting=dict(
            eligible_gt_known=den,
            within_tol_by_timely_prediction=within_timely,
            within_tol_only_late_prediction=within_late,
            eligible_on_completed_frames_not_within_tol=den_completed - within_completed,
            eligible_on_failed_opportunities=_sum(failed_known, f"{cls}_eligible"),
            failed_opportunities_with_known_objects=len(failed_known),
            opportunities_without_gt=len(missing),
            opportunities_without_gt_prediction_states=dict(collections.Counter(r["prediction_state"] for r in missing))),
        curve_all_completion=curve(completed, den_completed),
        curve_deadline_aware=curve(timely, den))


def segmentation(opps, cls):
    measured = [r for r in opps if r["gt_state"] == "MEASURED"]
    completed = [r for r in opps if r["prediction_state"] != "NONE"]
    timely = [r for r in measured if r["prediction_state"] == "TIMELY"]
    not_timely = [r for r in measured if r["prediction_state"] != "TIMELY"]
    unmeasured = [r for r in opps if r["gt_state"] != "MEASURED"]
    inter_c, union_c = _sum(completed, f"seg_{cls}_intersection"), _sum(completed, f"seg_{cls}_union")
    inter = _sum(timely, f"seg_{cls}_intersection")
    union = _sum(timely, f"seg_{cls}_union") + _sum(not_timely, f"seg_{cls}_gt")
    return dict(
        all_completion=dict(intersection=inter_c, union=union_c, iou=A3.ratio(inter_c, union_c),
                            frames=len(completed)),
        deadline_aware=dict(
            definition="late/absent prediction = empty predicted foreground: intersection 0, union = GT pixels; "
                       "pooled over opportunities with a retained GT segmentation count",
            intersection=inter, union=union, iou=A3.ratio(inter, union),
            evaluable_opportunities=len(measured), opportunities=len(opps),
            unevaluable_opportunities=len(unmeasured),
            unevaluable_prediction_states=dict(collections.Counter(r["prediction_state"] for r in unmeasured)),
            bound=("EXACT" if not unmeasured else
                   "ZERO_INTERSECTION_UNION_UNMEASURED" if not completed else
                   "UPPER_BOUND_GT_PIXELS_UNKNOWN_ON_FAILED_OPPORTUNITIES")))


def bootstrap(opps, rng):
    """Frame-cluster bootstrap of deadline-aware and all-completion statistics (descriptive, per session)."""
    known = [r for r in opps if r["gt_state"] == "MEASURED"]
    if len(known) != len(opps):
        return None
    out = collections.defaultdict(list)
    idx = np.arange(len(opps))
    for _ in range(BOOT):
        sample = [opps[i] for i in rng.choice(idx, len(idx), replace=True)]
        for cls in CLASSES:
            loc = localisation(sample, cls)
            seg = segmentation(sample, cls)
            out[f"{cls}_recall_deadline"].append(loc["deadline_aware"]["recall"])
            out[f"{cls}_recall_all"].append(loc["all_completion"]["recall"])
            out[f"{cls}_iou_deadline"].append(seg["deadline_aware"]["iou"])
            out[f"{cls}_iou_all"].append(seg["all_completion"]["iou"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def session_summary(key, session, opps, validated):
    states = collections.Counter(r["prediction_state"] for r in opps)
    gt = collections.Counter((r["prediction_state"], r["gt_state"]) for r in opps)
    terminal = collections.Counter(r["service_outcome"] for r in opps if r["prediction_state"] == "NONE")
    lat = [r["prediction_latency_ms"] for r in opps if r["prediction_latency_ms"] is not None]
    return dict(
        label=A3.LABELS[key], decisions=len(opps), prediction_states=dict(states),
        completed=states["TIMELY"] + states["LATE"], completed_by_170=states["TIMELY"],
        completion_latency_ms=A3.q(lat),
        no_completion_service_terminals=dict(terminal),
        feedback_by_170=sum(r["feedback_by_170"] for r in opps),
        gt_coverage={f"{p}|{g}": n for (p, g), n in sorted(gt.items())},
        eligibility_rederivation_validated_frames=validated,
        localisation={c: localisation(opps, c) for c in CLASSES},
        segmentation={c: segmentation(opps, c) for c in CLASSES},
        payload_bytes_all_transmitted=dict(**A3.q(session["payload_bytes"], (50, 99)), n=len(session["payload_bytes"])))


def write_csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out / "metrics"
    out.mkdir(parents=True, exist_ok=False)
    final_manifest = json.loads((FINAL_03 / "OUTPUT_SHA256.json").read_text())
    for name, digest in final_manifest.items():
        if sha(FINAL_03 / name) != digest:
            raise ValueError(f"final_03 {name} changed")
    eligibility = EligibilityFromTarget()
    sessions = {key: A3.load_session(key, root) for key, root in ROOTS.items()}
    final = json.loads((FINAL_03 / "SUMMARY.json").read_text())
    summary, all_opps, rng = {}, [], np.random.default_rng(SEED)
    for key, session in sessions.items():
        opps, validated = opportunities(key, session, eligibility)
        ref = final["sessions"][key]["prediction_completion"]
        assert len(opps) == ref["denominator"], key
        s = session_summary(key, session, opps, validated)
        assert (s["completed"], s["completed_by_170"]) == (ref["completed"], ref["completed_by_170"]), key
        if key != "action0":   # all-completion equals the final_03 live quality exactly
            for c in CLASSES:
                live = final["sessions"][key]["live_quality"]["classes"][c]
                if live:
                    assert s["localisation"][c]["all_completion"]["numerator"] == live["within_tol"], (key, c)
                    assert s["localisation"][c]["all_completion"]["denominator"] == live["eligible"], (key, c)
        s["bootstrap_95"] = bootstrap(opps, rng)
        summary[key] = s
        all_opps += opps
    fields = ["session", "decision_index", "frame_id", "prediction_state", "prediction_latency_ms", "service_outcome",
              "gt_state", "feedback_receipt_ms", "feedback_by_170", "payload_bytes",
              *[f"{c}_{k}" for c in CLASSES for k in ("eligible", "within_tol", "tp", "fp")],
              *[f"seg_{c}_{k}" for c in CLASSES for k in ("gt", "intersection", "union")]]
    write_csv(out / "OPPORTUNITIES.csv", all_opps, fields)
    curve_rows = []
    for key, s in summary.items():
        for c in CLASSES:
            loc = s["localisation"][c]
            for kind in ("all_completion", "deadline_aware"):
                for t, y in zip(THRESHOLDS, loc[f"curve_{kind}"]):
                    curve_rows.append(dict(session=key, cls=c, convention=kind, threshold_m=float(t), fraction=y,
                                           denominator_objects=loc[kind]["denominator"]))
    write_csv(out / "LOCALISATION_CURVES.csv", curve_rows,
              ["session", "cls", "convention", "threshold_m", "fraction", "denominator_objects"])
    full_local = {k: final["paired_quality"][k]["classes"] for k in ("agent", "arm_a", "arm_b")}
    document = dict(
        schema="splitfusion.m1_presentation_ext_v1.deadline_aware_quality.v1",
        boundary=dict(primary="action-open -> successful live edge.compute end with CPU prediction materialized",
                      deadline_ms=DEADLINE_MS, deadline_label="original 170-ms feedback deadline used as the reference",
                      feedback="UE feedback receipt reported separately; never substituted"),
        tolerances_m=TOL, avo_threshold=AVO_THRESHOLD, match_radius_m=3.0,
        sessions_are_separate=True, sessions={k: {kk: vv for kk, vv in v.items()} for k, v in summary.items()},
        full_local_offline_reference=dict(
            note="Offline same-input reference on retained frames only (agent 99, a54 94, m11 100); no measured runtime, "
                 "so no timely status; not a whole-session metric", values=full_local),
        bootstrap=dict(resamples=BOOT, seed=SEED, unit="policy-decision frame (cluster)",
                       note="computed only when every opportunity has measured GT"),
        inputs={str(p): sha(p) for p in [FINAL_03 / "SUMMARY.json", FINAL_03 / "OUTPUT_SHA256.json",
                                         *[A3.find_one(r / "capture", "FRAMES.jsonl") for r in ROOTS.values()]]},
        source_sha256={"metrics.py": sha(__file__), "analyze_v3.py": sha(A3.__file__)})
    for s in document["sessions"].values():
        for c in CLASSES:
            s["localisation"][c].pop("curve_all_completion")
            s["localisation"][c].pop("curve_deadline_aware")
    (out / "DEADLINE_AWARE_QUALITY.json").write_text(json.dumps(document, indent=1, sort_keys=True,
                                                                allow_nan=False) + "\n")
    print(json.dumps({k: dict(states=v["prediction_states"], gt=v["gt_coverage"],
                              loc={c: [v["localisation"][c]["all_completion"]["recall"],
                                       v["localisation"][c]["deadline_aware"]["recall"],
                                       v["localisation"][c]["deadline_aware"]["bound"]] for c in CLASSES},
                              seg={c: [v["segmentation"][c]["all_completion"]["iou"],
                                       v["segmentation"][c]["deadline_aware"]["iou"],
                                       v["segmentation"][c]["deadline_aware"]["bound"]] for c in CLASSES})
                      for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
