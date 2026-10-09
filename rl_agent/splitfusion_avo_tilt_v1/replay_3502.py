"""AVO-step replay of the saved V7 failure (frame 3502) -- a GEOMETRY replay, not a full producer replay.

Inputs are exact: the sealed FREEZE source in ``freeze_failure_v1/SOURCE.bin`` is re-admitted through a fresh
``FreezeGate`` with its own REQUEST descriptor (every buffer digest re-verified by the gate), depth is decoded with the
producer's bound ``visibility.decode_depth_bgra``, and camera matrices are the owned source matrices (what the
producer stages).  ``numeric_avo`` takes nothing else: it uses no producer history, static catalogue or owner state.
The producer's ``compute_unbound`` (target, conflicts, history) is NOT re-executed and no historical producer state is
claimed to be restored.

  OLD: ``splitfusion_numeric_gt_producer_v1.avo.numeric_avo`` (yaw-only)   -> must raise the live exception verbatim
  NEW: ``splitfusion_avo_tilt_v1.quality.numeric_avo``                     -> must produce a valid AVO result
Plus, for actor 117: corners vs CARLA's ``BoundingBox.get_world_vertices`` and vs the producer target path
(``M @ (bbox.location + offsets)``), and a labelled counterfactual (old geometry with 117's pitch/roll zeroed).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_avo_tilt_v1.replay_3502 \\
    --snapshot <capture root>/gt_gc_trace/freeze_failure_v1 --name <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_exact_gt_service_proto_v2 import freeze_codec as FZ
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA, originals as O
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as QA
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import geometry as G, quality as Q, switch

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_avo_tilt_v1"
TARGET = "117"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def run(snapshot, name):
    snapshot = Path(snapshot).resolve(strict=True)
    ready = json.loads((snapshot / "READY.json").read_bytes())
    for filename, digest in ready["sha256"].items():
        if sha((snapshot / filename).read_bytes()) != digest:
            raise ValueError("snapshot hash mismatch: " + filename)
    error_record = json.loads((snapshot / "ERROR.json").read_bytes())
    request = json.loads((snapshot / "REQUEST.json").read_bytes())
    descriptor = json.loads(request["descriptor"])
    source = (snapshot / "SOURCE.bin").read_bytes()
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    if switch.installed():
        raise RuntimeError("replay compares both geometries; the process switch must be off")
    fd, _ = FZ.seal([memoryview(source)])
    gate = FZ.FreezeGate(descriptor["session_id"])
    gate.last_sequence, gate.last_frame = descriptor["sequence"] - 1, descriptor["frame_id"] - 1
    result = dict(schema="splitfusion.avo_tilt_v1.replay_3502.v1", replay_kind="AVO_GEOMETRY_STEP_REPLAY",
                  full_producer_replay=False, historical_producer_state_restored=False,
                  snapshot=str(snapshot), snapshot_ready_sha256=ready["sha256"],
                  frame_id=descriptor["frame_id"], sequence=descriptor["sequence"],
                  live_exception=dict(type=error_record["exception_type"], message=error_record["exception"]))
    with gate.admit(request["descriptor"].encode(), fd) as admitted:
        kw = admitted.kwargs
        depth_raw = np.frombuffer(kw["depth_image"].raw_data, np.uint8).reshape(720, 1280, 4).copy()
        depth = O.bound().visibility.decode_depth_bgra(depth_raw)
        result["depth_decode_equals_live_gt_decode"] = bool(np.array_equal(depth, LG.decode_depth(kw["depth_image"])))
        args = dict(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
                    depth_image=kw["depth_image"], frozen_world=kw["frozen_world"],
                    camera_matrix=np.asarray(kw["camera_matrix"]), camera_inverse=np.asarray(kw["camera_inverse"]),
                    depth=depth, sync_validated=True)
        # ---- OLD geometry: the live rejection, reproduced
        try:
            NA.numeric_avo(**args)
            old = dict(raised=False)
        except LiveGtContractError as exc:
            old = dict(raised=True, type=type(exc).__name__, message=str(exc),
                       diagnostics=_jsonable(getattr(exc, "geometry_diagnostics", None)))
        old["identical_to_live_exception"] = (old.get("type") == error_record["exception_type"]
                                              and old.get("message") == error_record["exception"])
        result["old_geometry"] = old
        # ---- NEW geometry
        new = Q.numeric_avo(**args)
        rows = {r["gt_actor_id"]: r for r in new.rows}
        qualified = {r["gt_actor_id"]: r for r in new.avo["qualified"]}
        structural = {r["gt_actor_id"]: r for r in new.avo["structural_ignored"]}
        result["new_geometry"] = dict(
            raised=False, people=len(new.rows), qualified=len(qualified), structural_ignored=len(structural),
            target_actor=dict(row={k: rows[TARGET][k] for k in ("object_pitch_deg", "object_yaw_deg",
                                   "object_roll_deg", "gt_distance_m", "gt_bbox_area_px", "gt_center_x",
                                   "gt_center_y", "object_actor_rotation_matrix")},
                              qualified=qualified.get(TARGET), structural=structural.get(TARGET)),
            avo_geometry=new.avo["avo_geometry"], qualified_rows=list(new.avo["qualified"]))
        # ---- independent corner references for actor 117
        import carla
        actor = next(a for a in kw["frozen_world"].get_actors() if str(a.id) == TARGET)
        transform, bbox = actor.get_transform(), actor.bounding_box
        row = rows[TARGET]
        rotation, path = Q._row_rotation(row)
        centre = np.asarray([row[f"object_world_{a}"] for a in "xyz"])
        extent = [row[f"gt_extent_{a}_m"] for a in "xyz"]
        mine = G.oriented_box_corners(centre, extent, rotation)
        target_path = (np.asarray(transform.get_matrix()) @ np.c_[
            np.asarray([bbox.location.x, bbox.location.y, bbox.location.z]) + G.core.bbox_corner_offsets(extent),
            np.ones(8)].T).T[:, :3]
        carla_vertices = np.asarray([[v.x, v.y, v.z] for v in bbox.get_world_vertices(
            carla.Transform(transform.location, transform.rotation))])
        matched = [float(np.min(np.linalg.norm(carla_vertices - corner, axis=1))) for corner in mine]
        yaw_only = G.core.oriented_box_corners(centre, extent, row["object_yaw_deg"])
        result["independent_reference_actor_117"] = dict(
            geometry_path=path,
            max_abs_vs_producer_target_path_m=float(np.max(np.abs(mine - target_path))),
            max_nearest_vs_carla_get_world_vertices_m=max(matched),
            max_abs_rotation_vs_carla_euler=float(np.max(np.abs(rotation - G.carla_rotation_matrix(
                row["object_pitch_deg"], row["object_yaw_deg"], row["object_roll_deg"])))),
            max_abs_yaw_only_corner_error_m=float(np.max(np.abs(yaw_only - mine))),
            footprint_tilt_m=G.footprint_tilt_m(rotation, extent))
        # ---- labelled counterfactual: old geometry with only 117 forced upright (NOT a valid evaluation)
        old_rows = LG.frozen_people_rows(kw["frozen_world"], frame_id=int(kw["identity"]["frame_id"]),
            timestamp=float(kw["rgb"]["timestamp"]), camera_matrix=args["camera_matrix"],
            camera_inverse=args["camera_inverse"], intrinsics=LG.native_intrinsics(1280, 720), width=1280, height=720)
        forced = [dict(r, object_pitch_deg=0.0, object_roll_deg=0.0) if r["gt_actor_id"] == TARGET else r
                  for r in old_rows]
        counterfactual = QA.score_live_avo_people(frame_id=int(kw["identity"]["frame_id"]),
            timestamp=float(kw["rgb"]["timestamp"]), depth_frame_id=int(kw["depth_image"].frame),
            depth_timestamp=float(kw["depth_image"].timestamp), depth_m=depth, camera_matrix=args["camera_matrix"],
            camera_inverse=args["camera_inverse"], intrinsics=LG.native_intrinsics(1280, 720),
            raw_actor_rows=forced, width=1280, height=720)
        cf = {r["gt_actor_id"]: r for r in counterfactual["qualified"]}
        result["counterfactual_yaw_only_117_forced_upright"] = dict(
            label="COUNTERFACTUAL_NOT_AN_EVALUATION", qualified_ids_equal=sorted(cf) == sorted(qualified),
            structural_ids_equal=sorted(r["gt_actor_id"] for r in counterfactual["structural_ignored"]) == sorted(structural),
            avo_by_actor={k: dict(yaw_only=cf[k]["avo"], tilt_aware=qualified[k]["avo"]) for k in sorted(cf) if k in qualified},
            other_qualified_identical=all(cf[k]["avo"] == qualified[k]["avo"] and
                cf[k]["retained_actor_point_count"] == qualified[k]["retained_actor_point_count"]
                for k in cf if k in qualified and k != TARGET))
        del kw, args
    import os
    os.close(fd)
    checks = dict(
        old_geometry_reproduces_live_exception=old["identical_to_live_exception"],
        new_geometry_valid_result=result["new_geometry"]["raised"] is False,
        target_scored_or_explicitly_structural=(result["new_geometry"]["target_actor"]["qualified"] is not None
                                                or result["new_geometry"]["target_actor"]["structural"] is not None),
        target_handled_tilt_aware=any(t["gt_actor_id"] == TARGET and t["handling"] == "TILT_AWARE_GEOMETRY"
                                      for t in new.avo["avo_geometry"]["tilted_people"]),
        corners_match_producer_target_path=result["independent_reference_actor_117"]["max_abs_vs_producer_target_path_m"] <= 1e-9,
        corners_match_carla_world_vertices=result["independent_reference_actor_117"]["max_nearest_vs_carla_get_world_vertices_m"] <= 1e-4,
        depth_decode_consistent=result["depth_decode_equals_live_gt_decode"])
    result.update(checks=checks, status="PASS" if all(checks.values()) else "FAIL", live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(_jsonable(result), indent=1, sort_keys=True) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha((out / "RESULT.json").read_bytes())},
                                                       indent=1) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks, old=old.get("message"),
                          target=result["new_geometry"]["target_actor"]["qualified"]
                          or result["new_geometry"]["target_actor"]["structural"],
                          reference=result["independent_reference_actor_117"],
                          counterfactual=result["counterfactual_yaw_only_117_forced_upright"]), indent=1, default=str))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.name)
