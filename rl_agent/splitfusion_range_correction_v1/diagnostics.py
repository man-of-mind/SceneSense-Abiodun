"""Train-split (in-sample for the detector) radial/lateral decomposition with the same greedy
matcher and the same statistics as rl_agent/splitfusion_perception_limit_v1/original_radial_check.py,
for contrast with the established validation numbers.  Train GT only; validation not opened."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import score as S
from .apply import frame_origin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    det = S.read_detections(a.detections)
    gts = {"vehicle": S.load_vehicle_gt("train"), "person": S.load_v010_person_gt("train")}
    rows = {"vehicle": [], "person": []}
    for sid, recs in det.items():
        o = frame_origin(recs)
        for c in rows:
            g = gts[c].get(sid, [])
            p = sorted([x for x in recs if x["class_name"] == c and x["score"] >= S.THRESH[c]], key=lambda it: -it["score"])
            _, _, p2g = S.audit_v1.match_frame(p, g)
            for pi, gi in p2g.items():
                gg, pp = g[gi], p[pi]
                u = np.array([gg["world_x"] - o[0], gg["world_y"] - o[1]]); u /= np.linalg.norm(u)
                e = np.array([pp["world_x"] - gg["world_x"], pp["world_y"] - gg["world_y"]])
                rows[c].append((gg["distance_m"], float(np.hypot(*e)), float(e @ u), float(abs(e[0] * u[1] - e[1] * u[0]))))
    out = {"note": "TRAIN split = detector training data (in-sample); persons vs v010 POSITIVE (not AVO)"}
    for c, rs in rows.items():
        x = np.array(rs)
        def summ(z):
            return {"n": int(len(z)), "median_abs_radial_m": float(np.median(np.abs(z[:, 2]))),
                    "median_signed_radial_m": float(np.median(z[:, 2])), "median_lateral_m": float(np.median(z[:, 3])),
                    "within_tol_share_of_matches": float(np.mean(z[:, 1] <= S.TOL[c])), "p95_err_m": float(np.percentile(z[:, 1], 95))}
        out[c] = {"all_matches": summ(x), "bins": {b: summ(x[(x[:, 0] >= lo) & (x[:, 0] < hi)]) for b, lo, hi in S.BINS
                                                     if np.any((x[:, 0] >= lo) & (x[:, 0] < hi))}}
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
