"""Why was quality undefined? Read-only classification over a numeric-GT service archive (no rescoring change).

Quality is undefined exactly when the admitted target has zero eligible vehicle GT (POSITIVE vehicle rows of the
complete v010 target) AND zero eligible persons (geometry-qualified AVO rows with avo >= 0.65)
(``offline_quality_grid.quality.evaluate_exact_quality``). For every such decision this module reads the archived
producer inputs (``actor_visibility``, ``static_candidates``) and the admitted target (``object_boxes``,
``object_ignore_regions``, ``avo``) and assigns the most-specific cause that the records support:

  INPUT_OR_EVALUATOR_FAULT          service outcome not SCORED, provenance incomplete, or exact rescore failed
  NO_RELEVANT_OBJECT_IN_IMAGE       no vehicle/person actor (dynamic or static vehicle candidate) projects into the image
  IN_IMAGE_ONLY_BEYOND_40M          every in-image relevant object is farther than the registered 40 m geometry gate
  IN_IMAGE_WITHIN_40M_BELOW_AREA    within 40 m but projected area < 12 px (registered geometry gate)
  WITHIN_GATE_BUT_OCCLUDED          passes geometry, but depth visibility < 0.10 (vehicle/person v010) or AVO < 0.65 (person)
  OTHER_ELIGIBILITY_RULE            a vehicle passes geometry and visibility, yet the target builder refused it (registered
                                    static quarantine / identity conflict, or zero semantic support); reasons recorded
  UNKNOWN                           records do not support any of the above

Eligibility, scoring, session behaviour and rewards are not touched. Rule thresholds are those of
``data_collection/route_b_perception_v3/visibility_v1.py`` (40 m, 12 px, 0.10) and the AVO 0.65 threshold.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MAX_DISTANCE_M, MIN_AREA_PX, VISIBLE_THRESHOLD, AVO_THRESHOLD = 40.0, 12.0, 0.10, 0.65
ORDER = ("INPUT_OR_EVALUATOR_FAULT", "OTHER_ELIGIBILITY_RULE", "WITHIN_GATE_BUT_OCCLUDED",
         "IN_IMAGE_WITHIN_40M_BELOW_AREA", "IN_IMAGE_ONLY_BEYOND_40M", "NO_RELEVANT_OBJECT_IN_IMAGE", "UNKNOWN")


def _f(value):
    return float(value)


def distance(row: dict) -> float:
    """Dynamic visibility rows carry ``gt_distance_m``; static vehicle candidates carry ``range_m``."""
    return _f(row["gt_distance_m"] if "gt_distance_m" in row else row["range_m"])


def object_stage(row: dict, forward: bool) -> str:
    """Furthest registered gate an object reaches (dynamic actor or static vehicle candidate visibility row)."""
    if not forward or _f(row.get("clipped_projected_area_px", 0.0)) <= 0.0:
        return "NOT_IN_IMAGE"
    if distance(row) > MAX_DISTANCE_M:
        return "BEYOND_40M"
    if _f(row["clipped_projected_area_px"]) < MIN_AREA_PX:
        return "BELOW_AREA"
    if not bool(row.get("eligible_visible_v010")):
        return "OCCLUDED"
    return "VISIBLE_WITHIN_GATE"


def classify_frame(frame_dir: Path, *, scored: bool, rescore_exact: bool | None) -> dict:
    meta = json.loads((frame_dir / "scorer/target_metadata.json").read_bytes())
    inputs = json.loads((frame_dir / "producer/input_metadata.json").read_bytes())
    score = json.loads((frame_dir / "scorer/numerical_score.json").read_bytes())
    m = score["measurement"]
    positives = collections.Counter(r["label"] for r in meta["object_boxes"] if r.get("contract_state") == "POSITIVE")
    eligible_person = sum(float(r["avo"]) >= AVO_THRESHOLD for r in meta["avo"]["qualified"])
    # the target builder's own decision per source (POSITIVE row or IGNORE reason)
    decision = {r["source_identity"]: "POSITIVE" for r in meta["object_boxes"]}
    decision.update({r["source_identity"]: r["reason"] for r in meta["object_ignore_regions"]})
    boxes = {r["gt_actor_id"]: r for r in inputs["actor_boxes"]}
    objects = []
    for row in inputs["actor_visibility"]:
        if row["label"] in ("vehicle", "person"):
            box = boxes[row["gt_actor_id"]]
            objects.append(dict(kind="dynamic", label=row["label"], actor=row["gt_actor_id"],
                                stage=object_stage(row, _f(box["object_sensor_x"]) > 0.0),
                                distance_m=distance(row), visible_fraction=_f(row["visible_fraction"]),
                                target_decision=decision.get(f'{box["experiment_id"]}:actor:{row["gt_actor_id"]}')))
    for row in inputs["static_candidates"]:
        objects.append(dict(kind="static", label="vehicle", actor=row["environment_object_id"],
                            stage=object_stage(row, _f(row["object_sensor_x_m"]) > 0.0), distance_m=distance(row),
                            visible_fraction=_f(row.get("visible_fraction", 0.0)),
                            target_decision=decision.get(row["environment_object_id"], "SKIPPED_CONFIRMED_SECONDARY")))
    persons_geom = [dict(actor=r["gt_actor_id"], avo=float(r["avo"]), no_support=bool(r.get("no_support")))
                    for r in meta["avo"]["qualified"]]
    stages = collections.Counter(o["stage"] for o in objects)
    visible = [o for o in objects if o["stage"] == "VISIBLE_WITHIN_GATE"]
    complete = all(meta["provenance"]["complete"].values())
    undefined = score["q_perc"] is None
    other_reasons = sorted({o["target_decision"] for o in visible if o["label"] == "vehicle"})
    if not scored or not complete or rescore_exact is False:
        cause = "INPUT_OR_EVALUATOR_FAULT"
    elif any(o["label"] == "vehicle" for o in visible):
        cause = "OTHER_ELIGIBILITY_RULE"        # visible vehicle refused by quarantine / semantic-support rule
    elif persons_geom and all(p["avo"] < AVO_THRESHOLD for p in persons_geom):
        cause = "WITHIN_GATE_BUT_OCCLUDED"      # geometry-qualified persons with AVO < 0.65
    elif stages["OCCLUDED"] or visible:
        cause = "WITHIN_GATE_BUT_OCCLUDED"
    elif stages["BELOW_AREA"]:
        cause = "IN_IMAGE_WITHIN_40M_BELOW_AREA"
    elif stages["BEYOND_40M"]:
        cause = "IN_IMAGE_ONLY_BEYOND_40M"
    elif set(stages) <= {"NOT_IN_IMAGE"}:
        cause = "NO_RELEVANT_OBJECT_IN_IMAGE"
    else:
        cause = "UNKNOWN"
    return dict(frame_id=meta["capture_identity"]["frame_id"], q_perc=score["q_perc"], quality_undefined=undefined,
                eligible_vehicle=int(m["loc_vehicle_eligible_gt"]), eligible_person=int(m["loc_person_eligible_gt"]),
                positive_rows=dict(positives), avo_eligible_person=eligible_person, provenance_complete=complete,
                scored=scored, rescore_exact=rescore_exact, cause=cause, object_stages=dict(stages),
                visible_within_gate=[{k: o[k] for k in ("kind", "label", "distance_m", "visible_fraction", "target_decision")}
                                     for o in visible],
                other_rule_reasons=other_reasons if cause == "OTHER_ELIGIBILITY_RULE" else [],
                n_relevant_objects=len(objects), person_geometry_qualified=persons_geom,
                seg_gt_pixels=dict(vehicle=int(m["seg_vehicle_gt_pixels"]), person=int(m["seg_person_gt_pixels"])),
                seg_pred_pixels=dict(vehicle=int(m["seg_vehicle_pred_pixels"]), person=int(m["seg_person_pred_pixels"])),
                ignore_reasons=dict(collections.Counter(f'{r["class_name"]}:{r["reason"]}' for r in meta["object_ignore_regions"])),
                nearest_in_image_m=min((o["distance_m"] for o in objects if o["stage"] != "NOT_IN_IMAGE"), default=None))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True, help="analysis DECISIONS jsonl (service_outcome, timely)")
    p.add_argument("--rescore", type=Path, help="gt_adapter.check_archive output (optional, for exact-rescore status)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    decisions = {int(json.loads(l)["frame_id"]): json.loads(l) for l in a.decisions.read_text().splitlines() if l}
    exact = None
    if a.rescore:
        doc = json.loads(a.rescore.read_bytes())
        exact = {int(r["frame_id"]): bool(r["exact"]) for r in doc["rows"] if "frame_id" in r}
    rows = []
    for frame_id, d in sorted(decisions.items()):
        frame_dir = a.archive / f"frame_{frame_id:04d}"
        if not frame_dir.is_dir():
            rows.append(dict(frame_id=frame_id, cause="INPUT_OR_EVALUATOR_FAULT", detail="NO_ARCHIVE_FRAME",
                             quality_undefined=True))
            continue
        row = classify_frame(frame_dir, scored=d.get("service_outcome") == "SCORED",
                             rescore_exact=None if exact is None else exact.get(frame_id))
        row.update(timely=bool(d.get("timely")), decision_index=d.get("decision_index"), route_tick=d.get("route_tick"),
                   ego_world=d.get("ego_world"))
        if row["quality_undefined"] != (not d.get("quality_defined")):
            row["cause"], row["detail"] = "INPUT_OR_EVALUATOR_FAULT", "UNDEFINED_FLAG_DISAGREES_WITH_ANALYSIS"
        rows.append(row)
    undefined = [r for r in rows if r["quality_undefined"]]
    summary = dict(schema="splitfusion.run5_quality_reference.undefined_quality.v1", decisions=len(rows),
                   undefined=len(undefined), timely_undefined=sum(r.get("timely", False) for r in undefined),
                   causes={c: sum(r["cause"] == c for r in undefined) for c in ORDER},
                   undefined_with_seg_gt_pixels=sum(sum(r["seg_gt_pixels"].values()) > 0 for r in undefined if "seg_gt_pixels" in r),
                   undefined_with_predicted_object_pixels=sum(sum(r["seg_pred_pixels"].values()) > 0 for r in undefined if "seg_pred_pixels" in r),
                   defined_zero_eligible_check=sum((not r["quality_undefined"]) and r.get("eligible_vehicle", 1) + r.get("eligible_person", 1) == 0 for r in rows),
                   rescore_status_available=exact is not None,
                   rules=dict(max_distance_m=MAX_DISTANCE_M, min_area_px=MIN_AREA_PX, visible_threshold=VISIBLE_THRESHOLD,
                              avo_threshold=AVO_THRESHOLD), rows=undefined)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as s:
        json.dump(summary, s, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
