"""One-shot validation of the FROZEN range correction on the full 3,345-frame validation.

Original vs corrected with the unchanged matchers (score.py).  Writes VALIDATION_RESULT.json,
gt_outcomes.csv, record_scales.csv into --output.  Refuses to run without the frozen sha.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import score as S
from .apply import correct_frame, frame_origin, load_frozen, log_scale
from .features import edge_touching

SEG_MANIFEST = S.ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions/segmentation_manifest.csv"
SEG_MANIFEST_SHA = "52d6131e674adbfc75d8ea449a4f3be54a51fd993d3cb818cca23e8d759078f7"
EXPECTED = {"vehicle": {"eligible": 9691, "tp": 8416, "fn": 1275, "within": 7231, "fp": 618, "neutral": 2411},
            "person": {"eligible": 2877, "tp": 2052, "fn": 825, "within": 1533, "fp": 862, "neutral": 289}}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(x, q):
    return float(np.percentile(np.asarray(x), q)) if len(x) else None


def summarize(rows, fp, neutral, cls, arm):
    errs = [r[f"err_{arm}"] for r in rows if r[f"err_{arm}"] is not None]
    n = len(rows); tp = len(errs); w = sum(e <= S.TOL[cls] for e in errs)
    out = {"eligible": n, "tp": tp, "fn": n - tp, "recall_3m": tp / n if n else None,
           "within_tol": w, "recall_within_tol": w / n if n else None,
           "xy_mae_m": float(np.mean(errs)) if errs else None,
           "p50": pct(errs, 50), "p90": pct(errs, 90), "p95": pct(errs, 95), "p99": pct(errs, 99)}
    if fp is not None:
        out.update({"fp": fp, "neutral_or_ignored": neutral, "precision": tp / max(1, tp + fp)})
    return out


def transitions(rows, cls):
    t = S.TOL[cls]
    def w(e): return e is not None and e <= t
    return {"recovered_within_tol": sum((not w(r["err_orig"])) and w(r["err_corr"]) for r in rows),
            "lost_within_tol": sum(w(r["err_orig"]) and not w(r["err_corr"]) for r in rows),
            "new_matches_3m": sum(r["err_orig"] is None and r["err_corr"] is not None for r in rows),
            "lost_matches_3m": sum(r["err_orig"] is not None and r["err_corr"] is None for r in rows)}


def bootstrap(rows, cls, key, reps=2000, seed=20261006):
    """Cluster bootstrap of the within-tol recall delta; key(row) gives the cluster id."""
    t = S.TOL[cls]
    clusters = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        c = clusters[key(r)]
        c[0] += 1
        c[1] += int(r["err_orig"] is not None and r["err_orig"] <= t)
        c[2] += int(r["err_corr"] is not None and r["err_corr"] <= t)
    arr = np.array(list(clusters.values()), dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        deltas.append((s[2] - s[1]) / s[0])
    point = (arr[:, 2].sum() - arr[:, 1].sum()) / arr[:, 0].sum()
    return {"delta": float(point), "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "clusters": int(len(arr)), "reps": reps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--params-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    params = load_frozen(a.params, a.params_sha256)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if sha(S.DET) != S.DET_SHA or sha(S.AVO) != S.AVO_SHA:
        raise RuntimeError("input drift")
    seg_before = sha(SEG_MANIFEST)
    manifest = {r["sample_id"]: r for r in csv.DictReader((S.DS / "dataset/manifest.csv").open()) if r["split"] == "val"}
    val = list(manifest)
    assert len(val) == 3345
    det = S.read_detections(S.DET)
    vg = S.load_vehicle_gt("val"); avo = S.load_avo_table(S.AVO); st = S.load_val_structural()
    avo_trunc = {}
    for r in csv.DictReader(S.AVO.open()):
        avo_trunc[(r["sample_id"], r["gt_actor_id"])] = float(r["truncation"]) > 0.0

    # --- apply (timed) ---
    corrected, origins, no_origin, origin_dev = {}, {}, 0, []
    t0 = time.perf_counter()
    for sid in val:
        recs = det.get(sid, [])
        o = frame_origin(recs)
        origins[sid] = o
        corrected[sid] = correct_frame(recs, params, o)
    apply_s = time.perf_counter() - t0
    n_records = sum(len(det.get(s, [])) for s in val)
    for sid in val:
        if origins[sid] is None:
            no_origin += 1
        else:
            cam = np.array(json.loads(manifest[sid]["camera_matrix_json"]))[:3, 3]
            origin_dev.append(float(np.abs(origins[sid] - cam).max()))
    # microbenchmark: scale computation alone + full per-record correction (fixed origin)
    sample = [r for s in val[:400] for r in det.get(s, [])]
    t0 = time.perf_counter()
    for r in sample:
        log_scale(r, params)
    t_scale = (time.perf_counter() - t0) / len(sample)
    t0 = time.perf_counter()
    for s in val[:400]:
        frame_origin(det.get(s, []))
    t_origin = (time.perf_counter() - t0) / 400

    # --- invariants: only geometry fields changed; boxes/scores/order unchanged ---
    changed_fields = set()
    for sid in val:
        for r0, r1 in zip(det.get(sid, []), corrected[sid]):
            for k in r0:
                if r0[k] != r1[k]:
                    changed_fields.add(k)
    allowed = {"world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "actor_forward_depth_m"}
    if not changed_fields <= allowed:
        raise RuntimeError(f"non-geometry field changed: {changed_fields - allowed}")

    # --- score both arms ---
    rows = {"vehicle": [], "person": []}
    fp = {c: {"orig": 0, "corr": 0} for c in rows}; neu = {c: {"orig": 0, "corr": 0} for c in rows}
    fp_by_frame = {c: {"orig": defaultdict(set), "corr": defaultdict(set)} for c in rows}
    fp_trunc = {c: {arm: {"edge": 0, "interior": 0} for arm in ("orig", "corr")} for c in rows}
    rec_by_id = {}
    for sid in val:
        ignore = S.audit_v1.load_ignore(S.DS, "v010", sid)
        ep = S.episode_of(sid)
        res = {}
        for arm, preds in (("orig", det.get(sid, [])), ("corr", corrected[sid])):
            res[arm] = {"vehicle": S.vehicle_frame(preds, vg.get(sid, []), ignore),
                        "person": S.person_frame(preds, avo.get(sid, []), st.get(sid, []))}
            for c in rows:
                fp[c][arm] += res[arm][c]["fp"]; neu[c][arm] += res[arm][c]["neutral"]
        for r in det.get(sid, []):
            rec_by_id[(sid, r["prediction_index"])] = r
        # vehicle FP ids per arm (person FP ids are not exposed by the unchanged adapter -> counts only)
        for arm in ("orig", "corr"):
            ids = set(res[arm]["vehicle"]["fp_ids"])
            fp_by_frame["vehicle"][arm][sid] = ids
            for pid in ids:
                fp_trunc["vehicle"][arm]["edge" if edge_touching(rec_by_id[(sid, pid)]) else "interior"] += 1
        for j, g in enumerate(vg.get(sid, [])):
            b = g["box"]
            rows["vehicle"].append({"sample_id": sid, "episode": ep, "gt": g["source_identity"], "distance_m": g["distance_m"],
                                    "bin": S.bin_of(g["distance_m"]),
                                    "truncated": b[0] <= 2 or b[1] <= 2 or b[2] >= 766 or b[3] >= 430,
                                    "err_orig": res["orig"]["vehicle"]["err"][j], "err_corr": res["corr"]["vehicle"]["err"][j]})
        for j, g in enumerate(res["orig"]["person"]["eligible"]):
            rows["person"].append({"sample_id": sid, "episode": ep, "gt": g["gt_actor_id"], "distance_m": g["distance_m"],
                                   "bin": g["distance_bin"], "truncated": avo_trunc[(sid, g["gt_actor_id"])],
                                   "err_orig": res["orig"]["person"]["err"][j], "err_corr": res["corr"]["person"]["err"][j]})
    result: dict[str, Any] = {
        "schema": "splitfusion_range_correction_v1.validation", "frozen_params_sha256": params["_sha256"],
        "validation_frames": len(val), "records": n_records, "frames_without_origin": no_origin,
        "origin_vs_camera_matrix_max_abs_m": max(origin_dev), "changed_fields": sorted(changed_fields),
        "classes": {}, "cost": {"apply_total_s": apply_s, "apply_us_per_record_incl_origin_fit": 1e6 * apply_s / n_records,
                                "log_scale_us_per_record": 1e6 * t_scale, "origin_fit_us_per_frame": 1e6 * t_origin,
                                "note": "pure-Python/NumPy single-thread, CPU; includes all records (score floor 0.02)"},
    }
    for c, rs in rows.items():
        o = summarize(rs, fp[c]["orig"], neu[c]["orig"], c, "orig")
        k = summarize(rs, fp[c]["corr"], neu[c]["corr"], c, "corr")
        repro = {key: o[{"eligible": "eligible", "tp": "tp", "fn": "fn", "within": "within_tol", "fp": "fp",
                         "neutral": "neutral_or_ignored"}[key]] == val_ for key, val_ in EXPECTED[c].items()}
        entry = {"original": o, "corrected": k, "baseline_reproduced": all(repro.values()), "baseline_checks": repro,
                 "transitions": transitions(rs, c),
                 "bootstrap_frame": bootstrap(rs, c, lambda r: r["sample_id"]),
                 "bootstrap_block50": bootstrap(rs, c, lambda r: (r["episode"], int(r["sample_id"].rsplit("_", 2)[1]) // 50)),
                 "by_bin": {}, "by_truncation": {}, "by_episode": {}}
        if c == "vehicle":
            new_fp = sum(len(fp_by_frame[c]["corr"][s] - fp_by_frame[c]["orig"][s]) for s in val)
            gone_fp = sum(len(fp_by_frame[c]["orig"][s] - fp_by_frame[c]["corr"][s]) for s in val)
            entry["fp_identity_changes"] = {"new_fp_records": new_fp, "removed_fp_records": gone_fp,
                                            "fp_by_pred_box_truncation": fp_trunc[c]}
        else:
            entry["fp_identity_changes"] = {"net_fp_delta": fp[c]["corr"] - fp[c]["orig"],
                                            "note": "person FP identities are not exposed by the unchanged person_avo_measurement; net count only"}
        for name, sel in ([(b, lambda r, b=b: r["bin"] == b) for b, _, _ in S.BINS]):
            sub = [r for r in rs if sel(r)]
            entry["by_bin"][name] = {"original": summarize(sub, None, None, c, "orig"),
                                     "corrected": summarize(sub, None, None, c, "corr"), "transitions": transitions(sub, c)}
        for name, flag in (("edge_touching_gt_box", True), ("interior_gt_box", False)):
            sub = [r for r in rs if r["truncated"] == flag]
            entry["by_truncation"][name] = {"original": summarize(sub, None, None, c, "orig"),
                                            "corrected": summarize(sub, None, None, c, "corr"), "transitions": transitions(sub, c)}
        for ep in sorted({r["episode"] for r in rs}):
            sub = [r for r in rs if r["episode"] == ep]
            entry["by_episode"][ep] = {"original": summarize(sub, None, None, c, "orig"),
                                       "corrected": summarize(sub, None, None, c, "corr"), "transitions": transitions(sub, c)}
        result["classes"][c] = entry
    seg_after = sha(SEG_MANIFEST)
    result["segmentation"] = {"manifest_sha256_before": seg_before, "manifest_sha256_after": seg_after,
                              "matches_inference_manifest": seg_before == seg_after == SEG_MANIFEST_SHA,
                              "correction_reads_or_writes_segmentation": False,
                              "evaluation_v010_segmentation_metrics_apply_unchanged": True}
    (out / "VALIDATION_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    with (out / "gt_outcomes.csv").open("w", newline="") as s:
        w = csv.writer(s); w.writerow(["class", "sample_id", "episode", "gt", "distance_m", "bin", "truncated", "err_orig", "err_corr"])
        for c, rs in rows.items():
            for r in rs:
                w.writerow([c, r["sample_id"], r["episode"], r["gt"], f"{r['distance_m']:.4f}", r["bin"], int(r["truncated"]),
                            "" if r["err_orig"] is None else f"{r['err_orig']:.5f}", "" if r["err_corr"] is None else f"{r['err_corr']:.5f}"])
    with (out / "record_scales.csv").open("w", newline="") as s:
        w = csv.writer(s); w.writerow(["sample_id", "prediction_index", "class_name", "score", "local_x_orig", "scale"])
        for sid in val:
            for r0, r1 in zip(det.get(sid, []), corrected[sid]):
                if r0["score"] >= S.THRESH.get(r0["class_name"], 9):
                    w.writerow([sid, r0["prediction_index"], r0["class_name"], f"{r0['score']:.6f}", f"{r0['local_x']:.4f}",
                                f"{r1['local_x'] / r0['local_x']:.6f}" if r0["local_x"] else "1"])
    brief = {c: {"orig": {k: result["classes"][c]["original"][k] for k in ("within_tol", "recall_within_tol", "tp", "fp", "p50", "p95")},
                 "corr": {k: result["classes"][c]["corrected"][k] for k in ("within_tol", "recall_within_tol", "tp", "fp", "p50", "p95")},
                 "repro": result["classes"][c]["baseline_reproduced"], "trans": result["classes"][c]["transitions"],
                 "ci_block": result["classes"][c]["bootstrap_block50"]["ci95"]} for c in rows}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
