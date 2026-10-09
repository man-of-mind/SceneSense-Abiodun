"""Object-level failure breakdown for the 49-frame full-local reference (diagnostic, offline).

Inputs: the reproduced full-local evidence (dump_full_local.py), the v11 complete-target bundles
(object_boxes.csv rows) and the AVO input sidecars (raw pedestrian rows). Nothing is rescored with
changed rules: eligibility, thresholds and matches are read from the evaluator evidence itself.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CELL = Path("experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11/treatment/capture/cells/a71__favorable_stable")
SX = 768 / 1280.0
PERSON_TOL, VEHICLE_TOL = 1.2, 1.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(cam, gt, pred):
    gx, gy = gt[0] - cam[0], gt[1] - cam[1]
    r = math.hypot(gx, gy)
    ux, uy = gx / r, gy / r
    dx, dy = pred[0] - gt[0], pred[1] - gt[1]
    return dx * ux + dy * uy, -dx * uy + dy * ux  # radial (+ = farther), lateral


def inside(box, x, y, pad=0.0):
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    targets = {}
    for t in glob.glob(str(CELL / "parity_complete_targets/target_*/target.json")):
        d = json.load(open(t))
        rows = d["tables"]["object_boxes.csv"]
        fid = int(json.load(open(Path(t).parent / "inputs.json"))["frame"]["frame_id"])
        targets[fid] = (rows, Path(t).parent.name)
    objects = []
    inputs_sha = {}
    for f in sorted(glob.glob(str(a.evidence / "frames/frame_*.json"))):
        ev = json.load(open(f))
        inputs_sha[Path(f).name] = sha(f)
        assert ev["reproduced_sealed_full_local"]
        fid = ev["measurement"]["frame_id"]
        side = json.load(open(glob.glob(str(CELL / f"run5_live_avo_inputs/quality_gt_*_{fid}.avo_inputs.json"))[0]))
        cam = (side["camera_matrix"][0][3], side["camera_matrix"][1][3])
        raw_people = {str(r["gt_actor_id"]): r for r in side["raw_pedestrian_rows"]}
        recs = ev["prediction_records"]
        pre = ev["pre_p025_post_nms_candidates"]
        # ---- persons ----
        eligible = [r for r in ev["person_avo"]["qualified"] if float(r["avo"]) >= .65]
        p_idx = [i for i, p in enumerate(recs) if p["class_name"] == "person" and p["score"] >= .25]
        matched = {m["target_index"]: m for m in ev["matched_pairs"]["person"]}
        assert len(matched) == ev["measurement"]["loc_person_tp"]
        matched_pred = {m["prediction_record_index"] for m in matched.values()}
        for gi, g in enumerate(eligible):
            raw = raw_people[str(g["gt_actor_id"])]
            gxy = (float(g["world_x"]), float(g["world_y"]))
            cx, cy = raw["gt_center_x"] * SX, raw["gt_center_y"] * SX
            o = dict(cls="person", frame_id=fid, actor=str(g["gt_actor_id"]), avo=float(g["avo"]),
                     distance_m=float(raw["gt_distance_m"]), forward_m=float(raw["object_sensor_x"]),
                     bbox_area_px_native=float(raw["gt_bbox_area_px"]), center_model_px=[cx, cy])
            if gi in matched:
                m = matched[gi]
                p = recs[m["prediction_record_index"]]
                rad, lat = decompose(cam, gxy, (p["world_x"], p["world_y"]))
                o.update(status="TP_WITHIN" if m["error_m"] <= PERSON_TOL else "TP_BEYOND", error_m=m["error_m"],
                         radial_err_m=rad, lateral_err_m=lat, score=p["score"],
                         pred_forward_depth_m=p.get("actor_forward_depth_m"), radar_support=p.get("radar_support_score"))
            else:
                o["status"] = "FN"
                # nearest final person record in world and image
                best = None
                for i in p_idx:
                    p = recs[i]
                    d = math.hypot(p["world_x"] - gxy[0], p["world_y"] - gxy[1])
                    img = math.hypot(p["center_x_px"] - cx, p["center_y_px"] - cy)
                    box = (p["bbox_x0"], p["bbox_y0"], p["bbox_x1"], p["bbox_y1"])
                    cand = dict(rec=i, world_dist_m=d, image_dist_px=img, score=p["score"],
                                box_contains_gt_center=inside(box, cx, cy), used_by_other=i in matched_pred,
                                radial_lateral=decompose(cam, gxy, (p["world_x"], p["world_y"])))
                    if best is None or img < best["image_dist_px"]:
                        best = cand
                o["nearest_final_person_in_image"] = best
                # pre-p025 person candidates whose 2D box contains the GT centre
                pc = []
                for c in pre:
                    if c["label_internal"] != 1:
                        continue
                    if inside(c["box"], cx, cy, pad=4.0):
                        w = c.get("world_xyz")
                        pc.append(dict(score=c["score"], world_dist_m=None if w is None else math.hypot(w[0] - gxy[0], w[1] - gxy[1])))
                o["pre_p025_person_candidates_on_gt"] = sorted(pc, key=lambda x: -x["score"])[:5]
                vc = [dict(score=c["score"]) for c in pre if c["label_internal"] != 1 and inside(c["box"], cx, cy)]
                o["pre_p025_vehicle_boxes_on_gt_center"] = len(vc)
            objects.append(o)
        # ---- vehicles ----
        rows, tdir = targets[fid]
        frame_gt = ev["vehicle_targets"]
        by_id = {r["gt_actor_id"]: r for r in rows if r["label"] == "vehicle"}
        vm = {m["target_index"]: m for m in ev["matched_pairs"]["vehicle"]}
        scored = sorted([p for p in recs if p["score"] >= .20], key=lambda p: (-p["score"], p["class_name"]))
        for gi, g in enumerate(frame_gt):
            if g["class_name"] != "vehicle":
                continue
            aid = g["source_identity"].split(":")[-1]
            r = by_id.get(aid, {})
            gxy = (g["world_x"], g["world_y"])
            box = g["box"]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            o = dict(cls="vehicle", frame_id=fid, actor=aid, source_identity=g["source_identity"],
                     gt_source=r.get("gt_source"), parked=r.get("parked_label"), contract_reason=r.get("contract_reason"),
                     distance_m=float(r["gt_distance_m"]) if r else math.hypot(gxy[0] - cam[0], gxy[1] - cam[1]),
                     radar_support_points=r.get("radar_support_points"), bbox_area_px_model=g["area_px"],
                     box_model=box, type=r.get("gt_actor_type_id"))
            if gi in vm:
                m = vm[gi]
                p = recs[m["prediction_record_index"]]
                rad, lat = decompose(cam, gxy, (p["world_x"], p["world_y"]))
                o.update(status="TP_WITHIN" if m["error_m"] <= VEHICLE_TOL else "TP_BEYOND", error_m=m["error_m"],
                         radial_err_m=rad, lateral_err_m=lat, score=p["score"],
                         pred_forward_depth_m=p.get("actor_forward_depth_m"), radar_support=p.get("radar_support_score"),
                         parked_score=p.get("parked_score"))
            else:
                o["status"] = "FN"
                best = None
                for i, p in enumerate(recs):
                    if p["class_name"] != "vehicle":
                        continue
                    d = math.hypot(p["world_x"] - gxy[0], p["world_y"] - gxy[1])
                    pbox = (p["bbox_x0"], p["bbox_y0"], p["bbox_x1"], p["bbox_y1"])
                    ix = max(0, min(pbox[2], box[2]) - max(pbox[0], box[0])) * max(0, min(pbox[3], box[3]) - max(pbox[1], box[1]))
                    iou = ix / ((pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) + (box[2] - box[0]) * (box[3] - box[1]) - ix)
                    cand = dict(world_dist_m=d, box_iou=iou, score=p["score"], above_020=p["score"] >= .20,
                                radial_lateral=decompose(cam, gxy, (p["world_x"], p["world_y"])))
                    if best is None or iou > best["box_iou"] or (iou == best["box_iou"] == 0 and d < best["world_dist_m"]):
                        best = cand
                o["best_vehicle_record"] = best
                o["pre_vehicle_candidates_on_gt"] = sorted(
                    [c["score"] for c in pre if c["label_internal"] != 1 and inside(c["box"], cx, cy)], reverse=True)[:5]
            objects.append(o)
    # ---- summaries ----
    out = {}
    for cls in ("person", "vehicle"):
        rows = [o for o in objects if o["cls"] == cls]
        st = Counter(o["status"] for o in rows)
        actors = defaultdict(list)
        for o in rows:
            actors[o["actor"]].append(o["status"])
        out[cls] = dict(object_frames=len(rows), status=dict(st), unique_actors=len(actors),
                        actors={k: dict(Counter(v)) for k, v in sorted(actors.items(), key=lambda kv: -len(kv[1]))})
    (a.output / "OBJECTS.json").write_text(json.dumps(objects, indent=1, sort_keys=True))
    with (a.output / "OBJECTS.csv").open("w", newline="") as s:
        keys = ["cls", "frame_id", "actor", "status", "distance_m", "avo", "bbox_area_px_native", "bbox_area_px_model",
                "error_m", "radial_err_m", "lateral_err_m", "score", "pred_forward_depth_m", "radar_support",
                "radar_support_points", "gt_source", "parked", "type"]
        w = csv.DictWriter(s, keys, extrasaction="ignore")
        w.writeheader()
        for o in objects:
            w.writerow(o)
    out["inputs_sha256"] = inputs_sha
    out["script_sha256"] = sha(__file__)
    (a.output / "FAILURE_SUMMARY.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k != "inputs_sha256"}, indent=1))


if __name__ == "__main__":
    main()
