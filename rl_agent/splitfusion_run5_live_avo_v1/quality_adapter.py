"""GT/measurement primitives for explicitly declared LIVE metrics plus AVO.

The raw-label fix is deliberately separate from the v010 training-mask contract.
Raw semantic labels alone do not recreate depth-derived person pixels, static
vehicle reconciliation, or IGNORE pixels.  Launch callers must use
``require_live_gt_contract``; none of these helpers grants live readiness.

AVO is the existing original, unnormalized actor-volume observability >= 0.65,
not the failed normalized/band estimators. No model predictions enter AVO.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core, scoring

VEHICLE_TAGS = frozenset((14, 15, 16))  # exact FCOS v010 vehicle-class grouping
PERSON_TAGS = frozenset((12,))
AVO_THRESHOLD = 0.65
PERSON_SCORE_THRESHOLD = 0.25
REQUIRED_LIVE_INPUTS = (
    "same_frame_rgb_semantic_depth",
    "same_timestamp_rgb_semantic_depth",
    "frozen_actor_rows_before_localization_filter",
    "all_competing_pedestrian_volumes",
    "camera_matrices_and_native_intrinsics",
    "corrected_raw_semantic_live_mask",
    "live_vehicle_localization_unchanged",
    "qualified_live_gt_producer",
)


class LiveGtContractError(ValueError):
    pass


def live_metrics_contract() -> dict[str, Any]:
    """Approved prospective semantics; not a claim of live qualification."""
    return {
        "schema": "splitfusion.run5.live_metrics_avo_contract.v1",
        "mask_contract": "LIVE_CORRECTED_RAW_CARLA_3CLASS_NOT_OFFLINE_V010",
        "vehicle_semantic_tags": [14, 15, 16], "person_semantic_tags": [12],
        "segmentation_grid": "GT_NEAREST_RESIZED_TO_PREDICTION_GRID_AS_EXISTING_LIVE_ENGINE",
        "person_localization_contract": "ORIGINAL_UNNORMALIZED_AVO_GE_0_65",
        "avo_threshold": AVO_THRESHOLD, "person_score_threshold": PERSON_SCORE_THRESHOLD,
        "person_match_radius_m": 3.0,
        "person_matching_order": ["AVO_ELIGIBLE", "AVO_IGNORED", "STRUCTURAL_IGNORED"],
        "vehicle_localization": "UNCHANGED_EXISTING_LIVE_GT_AND_MATCHER",
        "source": "SYNCHRONIZED_LIVE_FRAME",
        "depth_sensor": {"width": 1280, "height": 720, "fov_degrees": 120,
                         "pose": "EXACT_RGB_POSE", "frame_timestamp": "EXACT_EQUAL"},
        "full_offline_fcos_v010_equivalence_claim": False,
        "reward_formula": "UNCHANGED", "quality_inside_inclusive_170_ms": True,
        "new_action_path_gt_threads": 0,
        "qualification_claim": "NONE__PROSPECTIVE_CONTRACT_NOT_LIVE_PASS",
    }


def require_live_gt_contract(binding: Mapping[str, Any]) -> None:
    """Refuse incomplete wiring; raw LUT or an old corpus AVO table is not parity.

The caller must independently verify the qualified producer digest/evidence.
This is a necessary schema guard, not a self-issued qualification certificate.
"""
    if binding.get("mask_contract") != "LIVE_CORRECTED_RAW_CARLA_3CLASS_NOT_OFFLINE_V010":
        raise LiveGtContractError("live mask source is not the approved live contract")
    if binding.get("person_localization_contract") != "ORIGINAL_UNNORMALIZED_AVO_GE_0_65":
        raise LiveGtContractError("live AVO contract differs from FCOS")
    if binding.get("source") != "SYNCHRONIZED_LIVE_FRAME":
        raise LiveGtContractError("retained corpus table cannot label a new live frame")
    missing = [key for key in REQUIRED_LIVE_INPUTS if binding.get(key) is not True]
    if missing:
        raise LiveGtContractError("LIVE_GT_INPUTS_NOT_QUALIFIED: " + ", ".join(missing))
    digest = binding.get("producer_evidence_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise LiveGtContractError("qualified producer evidence digest required")


def corrected_raw_semantic_3class(tags: np.ndarray) -> np.ndarray:
    """Correct CARLA raw-tag grouping, NOT training-equivalent v010 GT.

Tags 1/2/4/6/7/19 are not FCOS v010 vehicles; tag 15 is truck, not person.
"""
    tags = np.asarray(tags)
    if tags.ndim != 2 or tags.dtype != np.uint8:
        raise LiveGtContractError("raw semantic tags must be 2-D uint8")
    lut = np.zeros(256, dtype=np.uint8)
    lut[list(VEHICLE_TAGS)] = 1
    lut[list(PERSON_TAGS)] = 2
    return lut[tags]


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LiveGtContractError(f"nonfinite {name}")
    return result


def _structural_reasons(row, width, height):
    reasons = []
    if _finite(row["gt_distance_m"], "distance") > 40:
        reasons.append("distance_gt_40m")
    if not (0 <= _finite(row["gt_center_x"], "center x") < width and 0 <= _finite(row["gt_center_y"], "center y") < height):
        reasons.append("projected_box_center_outside_image")
    if _finite(row["gt_bbox_area_px"], "projected area") < 12:
        reasons.append("projected_area_lt_12px")
    if _finite(row["object_sensor_x"], "forward centre") <= 0:
        reasons.append("camera_forward_center_nonpositive")
    return reasons


def _geometry_relevance(people, boxes, reasons):
    """Prove that an unsupported tilted structural ignore cannot affect AVO.

    The sphere enclosing extent+.05 includes every original accepted point
    under any rigid rotation. Strict disjointness preserves ownership, ties,
    and support exactly. Projection-only exclusions cannot justify omitting a
    tilted actor because their yaw-only projection itself is unsupported.
    """
    tilted = []
    for row in people:
        if "object_pitch_deg" not in row or "object_roll_deg" not in row:
            raise LiveGtContractError("live yaw-only AVO needs explicit frozen pitch/roll evidence")
        pitch, roll = (_finite(row[f"object_{name}_deg"], f"actor {name}") for name in ("pitch", "roll"))
        tilted.append(abs(pitch) > 1e-6 or abs(roll) > 1e-6)
    targets = [i for i, why in enumerate(reasons) if not why]
    omissions = []
    for index, is_tilted in enumerate(tilted):
        if not is_tilted:
            continue
        row, box = people[index], boxes[index]
        evidence = {"gt_actor_id": box["key"], "pitch_deg": float(row["object_pitch_deg"]),
                    "roll_deg": float(row["object_roll_deg"]), "structural_reasons": reasons[index],
                    "target_separation": []}
        if not reasons[index]:
            reason = "TILTED_SCOREABLE_TARGET"
        elif not {"distance_gt_40m", "camera_forward_center_nonpositive"}.intersection(reasons[index]):
            reason = "TILTED_PROJECTION_ONLY_IGNORE_UNPROVEN"
        else:
            reason = None
            radius = _finite(np.linalg.norm(np.asarray(box["extent"]) + core.ACTOR_VOLUME_TOLERANCE_M),
                             "actor bounding sphere radius")
            evidence["bounding_sphere_radius_m"] = radius
            for target in targets:
                target_box = boxes[target]
                target_radius = float(np.linalg.norm(np.asarray(target_box["extent"]) + core.ACTOR_VOLUME_TOLERANCE_M))
                distance = float(np.linalg.norm(np.asarray(box["centre"]) - np.asarray(target_box["centre"])))
                # Outward rounded plus a small conservative arithmetic reserve,
                # never an expansion of the frozen scoring tolerance.
                scale = float(np.sum(np.abs(box["centre"])) + np.sum(np.abs(target_box["centre"])) + radius + target_radius)
                roundoff = max(1e-9, 32 * np.finfo(np.float64).eps * scale)
                required = float(np.nextafter(radius + target_radius, np.inf)) + roundoff
                proof = {"target_id": target_box["key"], "distance_m": distance,
                         "target_radius_m": target_radius, "required_separation_m": required,
                         "roundoff_reserve_m": roundoff, "separation_margin_m": distance - required}
                evidence["target_separation"].append(proof)
                if not all(math.isfinite(value) for value in (radius, target_radius, distance, required, scale)):
                    reason = "TILTED_COMPETITOR_SEPARATION_NONFINITE"
                elif not distance > required:
                    reason = "TILTED_POTENTIALLY_OVERLAPPING_COMPETITOR"
            if reason is None:
                omissions.append(evidence)
        if reason is not None:
            error = LiveGtContractError(f"yaw-only AVO does not support tilted actor {box['key']}: "
                f"{reason}; pitch={evidence['pitch_deg']!r}, roll={evidence['roll_deg']!r}")
            error.geometry_diagnostics = {"status": reason, "offender": evidence}
            raise error
    omitted_ids = {item["gt_actor_id"] for item in omissions}
    return [box for box in boxes if box["key"] not in omitted_ids], {
        "contract": "UPRIGHT_AVO_WITH_PROVEN_NONCOMPETING_TILTED_STRUCTURAL_IGNORES_V1",
        "sphere_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
        "omitted_tilted_structural_actors": omissions}


def score_live_avo_people(*, frame_id: int, timestamp: float, depth_frame_id: int,
                          depth_timestamp: float, depth_m: np.ndarray | None,
                          camera_matrix: np.ndarray, camera_inverse: np.ndarray,
                          intrinsics: np.ndarray, raw_actor_rows: Sequence[Mapping[str, Any]],
                          width: int, height: int) -> dict[str, Any]:
    """Use the frozen geometry on this frame's raw actor rows and depth only.

All pedestrian rows, including those subsequently structurally ignored, are
provided to competing-volume assignment. The caller must retain actor rows
before ``valid_localization_objects`` discards IDs, geometry, and ignored GT.
"""
    if depth_m is None:
        raise LiveGtContractError("LIVE_AVO_REQUIRES_SYNCHRONIZED_DEPTH")
    if int(frame_id) != int(depth_frame_id) or _finite(timestamp, "timestamp") != _finite(depth_timestamp, "depth timestamp"):
        raise LiveGtContractError("AVO source frame/timestamp mismatch")
    depth = np.asarray(depth_m)
    # The frozen scorer ignores 0 and the 1000-m far plane as invalid-support
    # pixels. They are legal depth encodings, not a missing sensor/frame.
    if depth.shape != (height, width) or not np.all(np.isfinite(depth)) or np.any(depth < 0) or np.any(depth > 1000):
        raise LiveGtContractError("invalid synchronized depth")
    camera, inverse, k = (np.asarray(x, dtype=np.float64) for x in (camera_matrix, camera_inverse, intrinsics))
    if camera.shape != (4, 4) or inverse.shape != (4, 4) or k.shape != (3, 3):
        raise LiveGtContractError("camera calibration shape mismatch")
    if not all(np.all(np.isfinite(x)) for x in (camera, inverse, k)) or np.max(np.abs(camera @ inverse - np.eye(4))) >= 1e-4:
        raise LiveGtContractError("invalid camera calibration inverse")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise LiveGtContractError("invalid focal length")
    people = [dict(row) for row in raw_actor_rows if row.get("label") == "person"]
    keys: set[str] = set()
    boxes = []
    for row in people:
        if int(row["frame_id"]) != frame_id or _finite(row["timestamp"], "actor timestamp") != timestamp:
            raise LiveGtContractError("actor/depth frame identity mismatch")
        key = str(row["gt_actor_id"])
        if key in keys:
            raise LiveGtContractError("duplicate actor ID")
        keys.add(key)
        centre = tuple(_finite(row[f"object_world_{axis}"], "actor centre") for axis in "xyz")
        extent = tuple(_finite(row[f"gt_extent_{axis}_m"], "actor extent") for axis in "xyz")
        if min(extent) <= 0:
            raise LiveGtContractError("nonpositive actor extent")
        boxes.append({"key": key, "centre": centre, "extent": extent,
                      "yaw_deg": _finite(row["object_yaw_deg"], "actor yaw")})
    reasons_by_row = [_structural_reasons(row, width, height) for row in people]
    competitors, relevance = _geometry_relevance(people, boxes, reasons_by_row)
    qualified, structural = [], []
    for row, box, reasons in zip(people, boxes, reasons_by_row):
        basic = {"gt_actor_id": box["key"], "world_x": box["centre"][0],
                 "world_y": box["centre"][1], "frame_id": frame_id}
        if reasons:
            structural.append({**basic, "qualified": False, "reasons": reasons})
            continue
        result = scoring.score_actor_frame(
            depth_m=depth, camera_matrix=camera, camera_inverse=inverse, intrinsics=k,
            width=width, height=height, target_key=box["key"],
            target_centre=box["centre"], target_extent=box["extent"],
            target_yaw_deg=box["yaw_deg"], pedestrian_boxes=competitors)
        if result["clipped_projected_area_px"] < 12:
            structural.append({**basic, "qualified": False, "reasons": ["projected_area_lt_12px"]})
            continue
        qualified.append({**basic, "qualified": True, "avo": float(result["visibility"]),
                          "no_support": bool(result["no_support"]),
                          "retained_actor_point_count": int(result["retained_actor_point_count"]),
                          "algorithm_version": core.ALGORITHM_VERSION})
    result = {"frame_id": frame_id, "timestamp": timestamp, "qualified": qualified,
              "structural_ignored": structural, "threshold": AVO_THRESHOLD}
    if relevance["omitted_tilted_structural_actors"]:
        result["geometry_relevance"] = relevance
    return result


def greedy_match(predictions, targets, available_prediction_indices=None):
    """Literal original 3-m matcher, without importing its pandas CLI in edge.

    Parity with run_comparison.greedy_match is tested including distance ties.
    """
    available = (set(range(len(predictions))) if available_prediction_indices is None
                 else set(available_prediction_indices))
    candidates = []
    for pred_index in sorted(available):
        prediction = predictions[pred_index]
        for gt_index, target in enumerate(targets):
            distance = math.hypot(float(prediction["world_x"]) - float(target["world_x"]),
                                  float(prediction["world_y"]) - float(target["world_y"]))
            if distance <= 3.0:
                candidates.append((distance, pred_index, gt_index))
    used_predictions, used_targets, matched = set(), set(), {}
    for _distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_predictions or gt_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(gt_index)
        matched[pred_index] = gt_index
    return matched, used_targets


def person_avo_measurement(predictions: Sequence[Mapping[str, Any]], avo: Mapping[str, Any]) -> dict[str, Any]:
    """Exact FCOS matching order: eligible -> AVO ignored -> structural ignored."""

    pred = [dict(p) for p in predictions if p.get("class_name") == "person" and _finite(p["score"], "score") >= PERSON_SCORE_THRESHOLD]
    for row in [*pred, *avo["qualified"], *avo["structural_ignored"]]:
        _finite(row["world_x"], "localization world_x")
        _finite(row["world_y"], "localization world_y")
    for row in avo["qualified"]:
        if not 0 <= _finite(row["avo"], "actor volume observability") <= 1:
            raise LiveGtContractError("actor volume observability outside [0,1]")
    eligible = [r for r in avo["qualified"] if float(r["avo"]) >= AVO_THRESHOLD]
    ignored = [r for r in avo["qualified"] if float(r["avo"]) < AVO_THRESHOLD]
    matched, used_gt = greedy_match(pred, eligible)
    remaining = set(range(len(pred))) - set(matched)
    ignored_avo, _ = greedy_match(pred, ignored, remaining)
    remaining -= set(ignored_avo)
    ignored_structural, _ = greedy_match(pred, avo["structural_ignored"], remaining)
    remaining -= set(ignored_structural)
    errors = [math.hypot(float(pred[p]["world_x"]) - float(eligible[g]["world_x"]),
                         float(pred[p]["world_y"]) - float(eligible[g]["world_y"]))
              for p, g in sorted(matched.items())]
    return {"loc_person_eligible_gt": len(eligible), "loc_person_tp": len(used_gt),
            "loc_person_fp": len(remaining), "loc_person_fn": len(eligible) - len(used_gt),
            "loc_person_ignored_predictions": len(ignored_avo) + len(ignored_structural),
            "loc_person_avo_ignored_predictions": len(ignored_avo),
            "loc_person_structural_ignored_predictions": len(ignored_structural),
            "loc_person_matched_xy_errors_m": errors}


def segmentation_sufficient_statistics(predicted: np.ndarray, v010_truth: np.ndarray) -> dict[str, Any]:
    """FCOS v010 confusion semantics, including label255 IGNORE.

    Caller supplies an actual v010 mask; this function does not construct one.
    Prediction is resized to the target grid, matching the frozen FCOS scorer.
    """
    import cv2

    pred, truth = np.asarray(predicted), np.asarray(v010_truth)
    if pred.ndim != 2 or truth.ndim != 2 or pred.dtype != np.uint8 or truth.dtype != np.uint8:
        raise LiveGtContractError("segmentation must be 2-D uint8")
    if set(np.unique(pred)) - {0, 1, 2} or set(np.unique(truth)) - {0, 1, 2, 255}:
        raise LiveGtContractError("illegal segmentation label")
    if pred.shape != truth.shape:
        pred = cv2.resize(pred, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = truth != 255
    confusion = np.bincount(truth[valid].astype(np.int64) * 3 + pred[valid].astype(np.int64), minlength=9).reshape(3, 3)
    result = {"seg_ignored_pixels": int(np.count_nonzero(~valid)), "seg_confusion_matrix": confusion.tolist()}
    for name, c in (("vehicle", 1), ("person", 2)):
        intersection = int(confusion[c, c])
        ngt, npred = int(confusion[c].sum()), int(confusion[:, c].sum())
        result.update({f"seg_{name}_gt_pixels": ngt, f"seg_{name}_pred_pixels": npred,
                       f"seg_{name}_intersection_pixels": intersection,
                       f"seg_{name}_union_pixels": ngt + npred - intersection})
    return result
