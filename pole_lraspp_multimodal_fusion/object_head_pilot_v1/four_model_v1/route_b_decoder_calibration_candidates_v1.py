#!/usr/bin/env python3
"""Conditional person-world-NMS extension and per-candidate deep dive.

Runs entirely offline on the already-saved threshold-0.02 validation
predictions. The person world-NMS arm is a *conditional* extension: it is
evaluated only because the retained predictions demonstrate duplicate person
predictions independently of ground truth (29-41% of person predictions have a
higher-scoring same-class prediction within 3.0 m).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from route_b_decoder_calibration_v1 import (  # noqa: E402
    PERSON_THRESHOLDS, VEHICLE_THRESHOLDS, VEHICLE_WORLD_NMS_M,
    load_gt, load_predictions, score_setting,
)
from visibility_audit_v1 import load_csv  # noqa: E402

PERSON_WORLD_NMS_EXT = (0.0, 0.5, 1.0, 2.0)
RECALL_RADII = (1.0, 2.0, 3.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True, type=Path)
    ap.add_argument("--detections", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    exp = args.experiment_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    split_ids, gts = load_gt(exp / "dataset" / "object_boxes.csv",
                             exp / "dataset" / "manifest.csv", args.split)
    preds = load_predictions(args.detections)
    collision_ids = {
        str(r["sample_id"]) for r in load_csv(exp / "provenance" / "collision_window_samples.csv")
        if r.get("split") == args.split and str(r.get("retained_in_dataset")) == "1"
    }
    non_collision = [s for s in split_ids if s not in collision_ids]

    ext = []
    for vt in VEHICLE_THRESHOLDS:
        for pt in PERSON_THRESHOLDS:
            for vn in VEHICLE_WORLD_NMS_M:
                for pn in PERSON_WORLD_NMS_EXT:
                    ext.append(score_setting(preds, gts, split_ids,
                                             {"vehicle": vt, "person": pt},
                                             {"vehicle": vn, "person": pn}))
    keys = [k for k in ext[0] if k != "distance_bands"]
    with (out / "operating_point_grid_person_nms_extension.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in ext:
            wr.writerow({k: r[k] for k in keys})
    print(f"extension settings: {len(ext)}", flush=True)

    # ---- candidate selection: Pareto-style extremes on the extended grid ----
    def best(metric, pool=ext):
        return max(pool, key=lambda r: r[metric])

    cands = {
        "max_vehicle_recall": best("vehicle_recall"),
        "max_person_recall": best("person_recall"),
        "max_vehicle_precision": best("vehicle_precision"),
        "max_person_precision": best("person_precision"),
        "max_mean_f1": best("mean_f1"),
        "max_overall_f1": best("overall_f1"),
        "min_mean_target_gap": min(
            ext, key=lambda r: (max(0.0, 0.90 - r["vehicle_recall"])
                                + max(0.0, 0.85 - r["person_recall"])
                                + max(0.0, 0.80 - r["vehicle_precision"])
                                + max(0.0, 0.80 - r["person_precision"]))),
    }

    deep = {}
    for name, c in cands.items():
        thr = {"vehicle": c["vehicle_score_threshold"], "person": c["person_score_threshold"]}
        nms = {"vehicle": c["vehicle_world_nms_m"], "person": c["person_world_nms_m"]}
        entry = {
            "setting": {**thr, "vehicle_world_nms_m": nms["vehicle"],
                        "person_world_nms_m": nms["person"]},
            "primary": score_setting(preds, gts, split_ids, thr, nms, with_bands=True),
            "collision_window_excluded": score_setting(preds, gts, non_collision, thr, nms),
            "recall_vs_match_radius": {
                f"{r:g}m": {
                    "vehicle": score_setting(preds, gts, split_ids, thr, nms, match_m=r)["vehicle_recall"],
                    "person": score_setting(preds, gts, split_ids, thr, nms, match_m=r)["person_recall"],
                } for r in RECALL_RADII
            },
        }
        deep[name] = entry
        p = entry["primary"]
        print(f"{name}: v_thr={thr['vehicle']} p_thr={thr['person']} "
              f"v_nms={nms['vehicle']} p_nms={nms['person']} "
              f"vR={p['vehicle_recall']:.4f} vP={p['vehicle_precision']:.4f} "
              f"pR={p['person_recall']:.4f} pP={p['person_precision']:.4f} "
              f"vXY={p['vehicle_xy_mae_m']:.4f} pXY={p['person_xy_mae_m']:.4f}", flush=True)

    (out / "candidate_deep_dive.json").write_text(
        json.dumps({
            "frames_primary": len(split_ids),
            "frames_collision_window_excluded": len(non_collision),
            "collision_window_frames_removed": len(split_ids) - len(non_collision),
            "person_world_nms_extension_justification": (
                "conditional arm: 29.3-40.5% of retained person predictions have a "
                "higher-scoring same-class person prediction within 3.0 m in predicted "
                "world XY (ground truth never enters the duplicate test)"),
            "candidates": deep,
        }, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
