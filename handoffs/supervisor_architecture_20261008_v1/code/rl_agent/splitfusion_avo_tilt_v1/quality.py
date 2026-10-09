"""Versioned copies of the live pedestrian-AVO row builder, scorer and numeric AVO step, using ``geometry``.

Copies of ``splitfusion_run5_live_avo_v1.live_gt.frozen_people_rows``,
``splitfusion_run5_live_avo_v1.quality_adapter.score_live_avo_people`` and
``splitfusion_numeric_gt_producer_v1.avo.numeric_avo``.  Differences, all geometric or labelling:

* rows carry the frozen actor rotation matrix, the bbox rotation and the geometry version, and their projected box
  (``gt_center_*``, ``gt_bbox_area_px`` -> structural eligibility) uses the world-from-box rotation;
* scoring uses the same rotation for target containment/ground rejection and for every competitor;
* the yaw-only tilted-actor refusal is replaced by the explicit ground-rule envelope (``geometry`` docstring);
* qualified rows carry ``algorithm_version`` = the new geometry version and every result carries ``avo_geometry``.

Structural reasons, the 12-px area rule, the AVO threshold, matching and the result schema read by the target
validators are unchanged.  The historical modules are imported, not edited.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as QA
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import geometry as G

ROW_FIELDS = ("object_actor_rotation_matrix", "object_bbox_rotation_deg", "geometry_version")


def frozen_people_rows(world: Any, *, frame_id: int, timestamp: float, camera_matrix: np.ndarray,
                       camera_inverse: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> list[dict]:
    """``live_gt.frozen_people_rows`` with the frozen rotation evidence and a tilt-correct projected box."""
    if int(world.frozen_frame_id) != frame_id:
        raise LiveGtContractError("frozen world frame differs from depth")
    out = []
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        transform, bbox = actor.get_transform(), actor.bounding_box
        matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
        centre = (matrix @ np.array([bbox.location.x, bbox.location.y, bbox.location.z, 1.]))[:3]
        extent = (float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z))
        sensor = (camera_inverse @ np.r_[centre, 1.])[:3]
        row = dict(label="person", gt_actor_id=str(actor.id), frame_id=frame_id,
                   timestamp=timestamp, object_yaw_deg=float(transform.rotation.yaw),
                   object_pitch_deg=float(transform.rotation.pitch),
                   object_roll_deg=float(transform.rotation.roll),
                   gt_distance_m=float(np.linalg.norm(matrix[:3, 3] - camera_matrix[:3, 3])),
                   object_sensor_x=float(sensor[0]), gt_center_x=-1., gt_center_y=-1.,
                   gt_bbox_area_px=0.)
        row.update({f"object_world_{axis}": float(centre[i]) for i, axis in enumerate("xyz")})
        row.update({f"gt_extent_{axis}_m": extent[i] for i, axis in enumerate("xyz")})
        row.update(object_actor_rotation_matrix=matrix[:3, :3].reshape(9).tolist(),
                   object_bbox_rotation_deg=[float(bbox.rotation.pitch), float(bbox.rotation.yaw),
                                             float(bbox.rotation.roll)],
                   geometry_version=G.GEOMETRY_VERSION)
        rotation, _path = _row_rotation(row)
        try:
            box = core.projected_boxes(G.oriented_box_corners(centre, extent, rotation),
                                       camera_inverse, intrinsics, width=width, height=height)
        except ValueError:
            box = None
        if box is not None:
            row.update(gt_center_x=box["clipped_bbox_x"] + box["clipped_bbox_w"] / 2,
                       gt_center_y=box["clipped_bbox_y"] + box["clipped_bbox_h"] / 2,
                       gt_bbox_area_px=box["clipped_projected_area_px"])
        out.append(row)
    return out


def _row_rotation(row: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    missing = [k for k in ("object_pitch_deg", "object_roll_deg", "object_yaw_deg", *ROW_FIELDS) if k not in row]
    if missing:
        raise LiveGtContractError("tilt-aware AVO needs frozen rotation evidence: " + ", ".join(missing))
    if row["geometry_version"] != G.GEOMETRY_VERSION:
        raise LiveGtContractError("pedestrian row built by a different AVO geometry version")
    return G.box_rotation(yaw_deg=row["object_yaw_deg"], pitch_deg=row["object_pitch_deg"],
                          roll_deg=row["object_roll_deg"], actor_rotation_matrix=row["object_actor_rotation_matrix"],
                          bbox_rotation_deg=row["object_bbox_rotation_deg"])


def _separation_proof(box, targets, boxes):
    """The previous adapter's exact bounding-sphere disjointness proof (copied unchanged)."""
    radius = G._finite(np.linalg.norm(np.asarray(box["extent"]) + core.ACTOR_VOLUME_TOLERANCE_M),
                       "actor bounding sphere radius")
    proofs, separated = [], True
    for target in targets:
        target_box = boxes[target]
        target_radius = float(np.linalg.norm(np.asarray(target_box["extent"]) + core.ACTOR_VOLUME_TOLERANCE_M))
        distance = float(np.linalg.norm(np.asarray(box["centre"]) - np.asarray(target_box["centre"])))
        scale = float(np.sum(np.abs(box["centre"])) + np.sum(np.abs(target_box["centre"])) + radius + target_radius)
        roundoff = max(1e-9, 32 * np.finfo(np.float64).eps * scale)
        required = float(np.nextafter(radius + target_radius, np.inf)) + roundoff
        finite = all(math.isfinite(v) for v in (radius, target_radius, distance, required, scale))
        separated = separated and finite and distance > required
        proofs.append({"target_id": target_box["key"], "distance_m": distance, "target_radius_m": target_radius,
                       "required_separation_m": required, "roundoff_reserve_m": roundoff,
                       "separation_margin_m": distance - required})
    return separated, radius, proofs


def _envelope(people, boxes, reasons):
    """Every tilted actor inside the ground-rule envelope competes with its true geometry; outside it, a target or
    possible competitor is refused, and only a structural actor proven disjoint from every target is omitted."""
    targets = [i for i, why in enumerate(reasons) if not why]
    tilted, omitted = [], []
    for index, box in enumerate(boxes):
        if box["geometry_path"] != G.PATH_TILTED:
            continue
        row = people[index]
        entry = {"gt_actor_id": box["key"], "pitch_deg": float(row["object_pitch_deg"]),
                 "roll_deg": float(row["object_roll_deg"]), "footprint_tilt_m": box["footprint_tilt_m"],
                 "role": "TARGET" if not reasons[index] else "STRUCTURAL_IGNORE",
                 "structural_reasons": list(reasons[index])}
        if box["footprint_tilt_m"] <= core.GROUND_REJECT_MARGIN_M:
            tilted.append(dict(entry, handling="TILT_AWARE_GEOMETRY"))
            continue
        status = "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET"
        if reasons[index]:
            separated, radius, proofs = _separation_proof(box, targets, boxes)
            if separated:
                omitted.append(dict(entry, handling="OMITTED_PROVEN_DISJOINT", bounding_sphere_radius_m=radius,
                                    target_separation=proofs))
                continue
            status = "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_POSSIBLE_COMPETITOR"
        error = LiveGtContractError(f"tilt-aware AVO v1 does not support actor {box['key']}: {status}; "
                                    f"pitch={entry['pitch_deg']!r}, roll={entry['roll_deg']!r}, "
                                    f"footprint_tilt_m={entry['footprint_tilt_m']!r}")
        error.geometry_diagnostics = {"status": status, "offender": entry}
        raise error
    omitted_ids = {item["gt_actor_id"] for item in omitted}
    return [box for box in boxes if box["key"] not in omitted_ids], {
        "version": G.GEOMETRY_VERSION, "upright_boundary_deg": G.UPRIGHT_TOLERANCE_DEG,
        "ground_rule_envelope_m": core.GROUND_REJECT_MARGIN_M, "tilted_people": tilted,
        "omitted_out_of_envelope_structural": omitted}


def score_live_avo_people(*, frame_id: int, timestamp: float, depth_frame_id: int, depth_timestamp: float,
                          depth_m: np.ndarray | None, camera_matrix: np.ndarray, camera_inverse: np.ndarray,
                          intrinsics: np.ndarray, raw_actor_rows: Sequence[Mapping[str, Any]],
                          width: int, height: int) -> dict[str, Any]:
    """``quality_adapter.score_live_avo_people`` with the tilt-aware geometry (all validators kept)."""
    if depth_m is None:
        raise LiveGtContractError("LIVE_AVO_REQUIRES_SYNCHRONIZED_DEPTH")
    if int(frame_id) != int(depth_frame_id) or G._finite(timestamp, "timestamp") != G._finite(depth_timestamp, "depth timestamp"):
        raise LiveGtContractError("AVO source frame/timestamp mismatch")
    depth = np.asarray(depth_m)
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
        if int(row["frame_id"]) != frame_id or G._finite(row["timestamp"], "actor timestamp") != timestamp:
            raise LiveGtContractError("actor/depth frame identity mismatch")
        key = str(row["gt_actor_id"])
        if key in keys:
            raise LiveGtContractError("duplicate actor ID")
        keys.add(key)
        centre = tuple(G._finite(row[f"object_world_{axis}"], "actor centre") for axis in "xyz")
        extent = tuple(G._finite(row[f"gt_extent_{axis}_m"], "actor extent") for axis in "xyz")
        if min(extent) <= 0:
            raise LiveGtContractError("nonpositive actor extent")
        rotation, path = _row_rotation(row)
        boxes.append({"key": key, "centre": centre, "extent": extent, "rotation": rotation, "geometry_path": path,
                      "footprint_tilt_m": G.footprint_tilt_m(rotation, extent)})
    reasons_by_row = [QA._structural_reasons(row, width, height) for row in people]
    competitors, geometry = _envelope(people, boxes, reasons_by_row)
    qualified, structural = [], []
    for row, box, reasons in zip(people, boxes, reasons_by_row):
        basic = {"gt_actor_id": box["key"], "world_x": box["centre"][0], "world_y": box["centre"][1],
                 "frame_id": frame_id}
        if reasons:
            structural.append({**basic, "qualified": False, "reasons": reasons})
            continue
        result = G.score_actor_frame(
            depth_m=depth, camera_matrix=camera, camera_inverse=inverse, intrinsics=k, width=width, height=height,
            target_key=box["key"], target_centre=box["centre"], target_extent=box["extent"],
            target_rotation=box["rotation"], pedestrian_boxes=competitors)
        if result["clipped_projected_area_px"] < 12:
            structural.append({**basic, "qualified": False, "reasons": ["projected_area_lt_12px"]})
            continue
        qualified.append({**basic, "qualified": True, "avo": float(result["visibility"]),
                          "no_support": bool(result["no_support"]),
                          "retained_actor_point_count": int(result["retained_actor_point_count"]),
                          "algorithm_version": G.GEOMETRY_VERSION})
    return {"frame_id": frame_id, "timestamp": timestamp, "qualified": qualified,
            "structural_ignored": structural, "threshold": QA.AVO_THRESHOLD, "avo_geometry": geometry}


def numeric_avo(*, identity: Mapping[str, Any], rgb: Mapping[str, Any], semantic_image: Any, depth_image: Any,
                frozen_world: Any, camera_matrix: np.ndarray, camera_inverse: np.ndarray, depth: np.ndarray,
                sync_validated: bool):
    """``producer_v1.avo.numeric_avo`` (validators verbatim) with the tilt-aware rows and scorer."""
    from rl_agent.splitfusion_numeric_gt_producer_v1.avo import NumericAvo
    if not sync_validated:
        raise LiveGtContractError("source synchronization must be validated before AVO")
    frame_id = int(identity["frame_id"])
    for image in (semantic_image, depth_image):
        if int(image.frame) != frame_id or float(image.timestamp) != float(rgb["timestamp"]):
            raise LiveGtContractError("RGB/semantic/depth frame or timestamp mismatch")
        if (int(image.width), int(image.height)) != (int(rgb["width"]), int(rgb["height"])):
            raise LiveGtContractError("RGB/semantic/depth image geometry mismatch")
        if not np.allclose(np.asarray(image.transform.get_matrix()), camera_matrix, rtol=0, atol=1e-5):
            raise LiveGtContractError("RGB/semantic/depth pose mismatch")
    if int(rgb["frame_id"]) != frame_id or not np.allclose(rgb["camera_matrix"], camera_matrix, rtol=0, atol=1e-5):
        raise LiveGtContractError("RGB/ticket identity or pose mismatch")
    width, height = int(depth_image.width), int(depth_image.height)
    if depth.shape != (height, width) or depth.dtype != np.float32:
        raise LiveGtContractError("shared decoded depth does not match the depth image")
    k = LG.native_intrinsics(width, height)
    rows = frozen_people_rows(frozen_world, frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                              camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                              intrinsics=k, width=width, height=height)
    avo = score_live_avo_people(frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                                depth_frame_id=int(depth_image.frame), depth_timestamp=float(depth_image.timestamp),
                                depth_m=depth, camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                                intrinsics=k, raw_actor_rows=rows, width=width, height=height)
    return NumericAvo(frame_id=frame_id, timestamp=float(rgb["timestamp"]), intrinsics=k, rows=tuple(rows), avo=avo)
