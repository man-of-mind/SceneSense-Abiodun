"""Feasibility check: are there same-frame radar returns at the true object position?

For each eligible object observation, count retained radar world points (target .sources/radar_world_xyz.npy,
the window the UE rasterised) within 1.0 m in XY of the GT actor origin, and within 1.0 m of the evaluated
prediction. Descriptive only; no localisation rule is changed or tested here.
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np

CELL = Path("experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11/treatment/capture/cells/a71__favorable_stable")


def main():
    objs = json.load(open(sys.argv[1]))
    ev_dir = Path(sys.argv[2])
    radar = {}
    for t in glob.glob(str(CELL / "parity_complete_targets/target_*/inputs.json")):
        fid = int(json.load(open(t))["frame"]["frame_id"])
        radar[fid] = np.load(Path(t).parent.with_suffix(".sources") / "radar_world_xyz.npy")
    rows = []
    for o in objs:
        pts = radar[o["frame_id"]]
        ev = json.load(open(ev_dir / f"frames/frame_{o['frame_id']}.json"))
        if o["cls"] == "person":
            g = [r for r in ev["person_avo"]["qualified"] if r["gt_actor_id"] == o["actor"]][0]
            gxy = (g["world_x"], g["world_y"])
        else:
            g = [t for t in ev["vehicle_targets"] if t["source_identity"] == o["source_identity"]][0]
            gxy = (g["world_x"], g["world_y"])
        d = np.hypot(pts[:, 0] - gxy[0], pts[:, 1] - gxy[1])
        rows.append(dict(cls=o["cls"], frame_id=o["frame_id"], actor=o["actor"], status=o["status"],
                         distance_m=o["distance_m"], radar_pts_within_1m=int((d <= 1.0).sum()),
                         radar_pts_within_2m=int((d <= 2.0).sum())))
    out = {"n_radar_points_per_frame_median": float(np.median([len(v) for v in radar.values()])), "rows": rows}
    for cls in ("person", "vehicle"):
        sub = [r for r in rows if r["cls"] == cls]
        out[cls] = {"observations": len(sub), "with_radar_within_1m": sum(r["radar_pts_within_1m"] > 0 for r in sub),
                    "median_pts_within_1m": float(np.median([r["radar_pts_within_1m"] for r in sub]))}
        for st in ("TP_WITHIN", "TP_BEYOND", "FN"):
            s2 = [r for r in sub if r["status"] == st]
            out[cls][st] = {"n": len(s2), "with_radar_within_1m": sum(r["radar_pts_within_1m"] > 0 for r in s2)}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        if r["cls"] == "person": print(r)


if __name__ == "__main__":
    main()
