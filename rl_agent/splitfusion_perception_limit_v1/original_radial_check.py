"""Radial-vs-lateral decomposition of matched errors on the ORIGINAL 3,345-frame validation (CPU, read-only).

Re-uses the correspondence audit's exact original re-scoring (like_for_like.original semantics: vehicles v010
POSITIVE GT with score >= 0.20, persons AVO >= 0.65 with p025 score >= 0.25, greedy 3 m). The sensor origin of
each frame is recovered from that frame's own prediction records by a 2-D rigid fit local(x,y,z) -> world(x,y,z) (3-D Kabsch; the camera is pitched);
frames whose fit residual exceeds 1 cm are excluded and counted. Radial = error component along the
sensor->GT ray (positive = predicted farther).
"""
import csv, hashlib, importlib.util, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(".").resolve()
spec = importlib.util.spec_from_file_location("lfl", ROOT / "experiments/splitfusion_perception_latency_retention_v1/20261006_v1/correspondence/like_for_like.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)

def main(out_path):
    assert L.sha(L.DET) == L.DET_SHA and L.sha(L.AVO) == L.AVO_SHA
    val = [r["sample_id"] for r in csv.DictReader((L.DS / "dataset/manifest.csv").open()) if r["split"] == "val"]
    veh_p, per_p, fit = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in csv.DictReader(L.DET.open()):
        s = float(r["score"]); w = (float(r["world_x"]), float(r["world_y"])); l = (float(r["local_x"]), float(r["local_y"]), float(r["local_z"])); w3 = (*w, float(r["world_z"]))
        fit[r["sample_id"]].append((l, w3))
        if r["class_name"] == "vehicle" and s >= 0.20: veh_p[r["sample_id"]].append(w)
        elif r["class_name"] == "person" and s >= 0.25: per_p[r["sample_id"]].append(w)
    origin, bad = {}, 0
    for sid, pairs in fit.items():
        if len(pairs) < 3: continue
        A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
        ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
        t = cb - R @ ca
        if np.abs(A @ R.T + t - B).max() > 0.01: bad += 1; continue
        origin[sid] = t[:2]
    veh_g, per_g = defaultdict(list), defaultdict(list)
    for r in csv.DictReader((L.DS / "contracts/v010/val/object_boxes.csv").open()):
        if r["label"] == "vehicle": veh_g[r["sample_id"]].append((float(r["object_world_x"]), float(r["object_world_y"]), float(r["gt_distance_m"])))
    for r in csv.DictReader(L.AVO.open()):
        if float(r["actor_volume_observability"]) >= 0.65: per_g[r["sample_id"]].append((float(r["object_world_x"]), float(r["object_world_y"]), float(r["distance_m"])))
    rows = {"vehicle": [], "person": []}; skipped = 0
    for sid in val:
        for cls, P, G in (("vehicle", veh_p, veh_g), ("person", per_p, per_g)):
            preds, gts = P.get(sid, []), G.get(sid, [])
            if not gts: continue
            cand = sorted((math.hypot(p[0]-g[0], p[1]-g[1]), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts) if math.hypot(p[0]-g[0], p[1]-g[1]) <= 3.0)
            up, ug = set(), set()
            for d, i, j in cand:
                if i in up or j in ug: continue
                up.add(i); ug.add(j)
                if sid not in origin: skipped += 1; continue
                o = origin[sid]; g = gts[j]; p = preds[i]
                u = np.array([g[0]-o[0], g[1]-o[1]]); u /= np.linalg.norm(u)
                e = np.array([p[0]-g[0], p[1]-g[1]])
                rows[cls].append((g[2], d, float(e @ u), float(abs(e[0]*u[1]-e[1]*u[0]))))
    tol = {"vehicle": 1.0, "person": 1.2}; out = {"frames_fit_excluded": bad, "matches_without_origin": skipped}
    for cls, rs in rows.items():
        a = np.array(rs); beyond = a[a[:, 1] > tol[cls]]
        def summ(x):
            return {"n": int(len(x)), "median_abs_radial_m": float(np.median(np.abs(x[:, 2]))) if len(x) else None,
                    "median_lateral_m": float(np.median(x[:, 3])) if len(x) else None,
                    "share_radial_dominant": float(np.mean(np.abs(x[:, 2]) > x[:, 3])) if len(x) else None,
                    "median_signed_radial_m": float(np.median(x[:, 2])) if len(x) else None,
                    "share_beyond_tol_if_radial_removed": float(np.mean(x[:, 3] > tol[cls])) if len(x) else None}
        out[cls] = {"all_matches": summ(a), "matches_beyond_tol": summ(beyond),
                    "bins": {f"{lo}-{hi}": summ(a[(a[:, 0] >= lo) & (a[:, 0] < hi)]) for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40))}}
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(out_path).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
