"""Part 1: favourable-channel Pareto alternatives, three explicitly separate layers.

  A_OLD  original 288 live campaign, FAVORABLE_STABLE cells (measured; consolidation 20260909): payload, map install per
         sent frame, capture->install AoI (500-ms Phase-15 boundary) and the OLD per-action validation quality
         (3-m-gated precision/recall, combined_quality presentation coordinate). No 170-ms clock, no within-tolerance hits.
  B_CUR  current objective in the current corrected latency environment: the EXISTING headroom fixed-action runs on
         TRAINING evidence (FIT scenes, TRAIN_GCV2, CORRECTED, seeds 8017/8029; 71 in-support cells), FAVORABLE decisions
         only (profile column borrowed by index from this audit's own controller-A runs on the same seeds; streams asserted
         identical). P = timely within-tolerance localisation (v 1.0 m / p 1.2 m, misses count), registered retentions.
  C_SCN  per-scene: exact FIT quality grid, same scene, every in-support non-11 cell vs the best mode-11 in-support cell at
         <= its payload ON THAT SCENE (no latency; quality only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m rl_agent.splitfusion_run5_mode_value_audit_v1.part1 --out <ABS dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_constrained_headroom_v1 import collect as K
from rl_agent.splitfusion_run5_constrained_headroom_v1 import metrics as MT
from rl_agent.splitfusion_run5_full_v1.support import in_support
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as REG

ROOT = REG.ROOT
CONS = ROOT / "experiments/splitfusion_288_offline_rl_dataset_v1/20260909_offline_consolidation_v1"
SUP = ROOT / "experiments/splitfusion_supervisor_analysis_v1/20260914_latency_quality_and_sensor_v1/action_profile_quality_latency.csv"
HEAD = ROOT / "experiments/splitfusion_run5_constrained_headroom_v1/runs_01"
DB = ROOT / "experiments/splitfusion_hybrid_sac_quality_grid_v1/20260918_exact_continuous_q_grid_a1b_full/quality_rows.sqlite3"
TOL = {"vehicle": 1.0, "person": 1.2}


def pareto(points, keys):
    """keys: list of (name, +1 maximise / -1 minimise). Returns indices of non-dominated points."""
    nd = []
    for i, p in enumerate(points):
        dom = False
        for j, o in enumerate(points):
            if i == j:
                continue
            ge = all(s * o[k] >= s * p[k] for k, s in keys)
            gt = any(s * o[k] > s * p[k] for k, s in keys)
            if ge and gt:
                dom = True
                break
        if not dom:
            nd.append(i)
    return nd


def layer_a():
    live = {r["cell_id"]: r for r in csv.DictReader(open(CONS / "action_72x4_summary.csv"))}
    qual = {r["cell_id"]: r for r in csv.DictReader(open(SUP))}
    rows = []
    for cid, r in live.items():
        if r["network_profile"] != "FAVORABLE_STABLE":
            continue
        s = qual[cid]
        f = lambda x: float(x) if x not in ("", None) else float("nan")
        rows.append(dict(action_id=int(r["action_id"]), family=r["family"], quantizer=r["quantizer"], q_e4=int(float(r["q_e4"])),
                         payload_kb=f(r["live_estimated_wire_bytes_median"]) / 1e3,
                         installed_per_sent=f(r["rate_installed_per_sent"]),
                         aoi_install_p50_ms=f(r["s13_capture_to_install_aoi_ms__median"]),
                         old_combined_quality=f(s["combined_quality"]),
                         old_vehicle_recall_3m=f(s["val_vehicle_recall"]), old_person_recall_3m=f(s["val_canonical_person_recall"]),
                         old_vehicle_precision=f(s["val_vehicle_precision"]), old_person_precision=f(s["val_canonical_person_precision"]),
                         old_seg_miou=f(s["val_segmentation_miou"]), old_vehicle_iou=f(s["val_vehicle_iou"])))
    rows.sort(key=lambda r: r["action_id"])
    ok = [r for r in rows if np.isfinite(r["payload_kb"]) and np.isfinite(r["old_combined_quality"])]
    for r in rows:
        r["pareto_payload_quality"] = False
        r["pareto_delivery_quality"] = False
    for i in pareto(ok, [("old_combined_quality", 1), ("payload_kb", -1)]):
        ok[i]["pareto_payload_quality"] = True
    dl = [r for r in ok if np.isfinite(r["installed_per_sent"])]
    for i in pareto(dl, [("old_combined_quality", 1), ("installed_per_sent", 1)]):
        dl[i]["pareto_delivery_quality"] = True
    return rows


def load_fixed(m, q, seeds=(8017, 8029)):
    parts = [np.load(HEAD / f"A_m{m}_q{q}__TRAIN__CORRECTED__{s}.npz") for s in seeds]
    return MT.as_dict(np.concatenate([p["data"] for p in parts]), parts[0]["cols"])


def layer_b(audit_runs: Path):
    prof = []
    mcs_ref = []
    for s in (8017, 8029):
        z = np.load(audit_runs / f"A__TRAIN__CORRECTED__{s}.npz")
        cols = list(z["cols"])
        prof.append(z["data"][:, cols.index("profile")])
        mcs_ref.append(z["data"][:, cols.index("mcs")])
    prof, mcs_ref = np.concatenate(prof), np.concatenate(mcs_ref)
    out = {}
    for m, q in K.ACTIONS:
        a = load_fixed(m, q)
        assert np.array_equal(a["mcs"], mcs_ref), "MCS stream differs: profile cannot be borrowed"
        res = {}
        for pi, pname in enumerate(("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY", "ALL")):
            sub = a if pname == "ALL" else MT.subset(a, prof == pi)
            mt = MT.metrics(sub)
            res[pname] = dict(n=mt["n"], P=mt["P"], timely=mt["timely"], wire_p50_kb=mt["wire_p50_kb"], wire_p99_kb=mt["wire_p99_kb"],
                              R_v=mt["vehicle"]["recall_within_tol"], R_p=mt["person"]["recall_within_tol"],
                              ret={f"{k}_{c}": mt[c]["retention"][k] for k, c in MT.TIERS},
                              seg_v=mt["vehicle"]["seg_iou"], seg_p=mt["person"]["seg_iou"],
                              prec_v=mt["vehicle"]["precision"], prec_p=mt["person"]["precision"])
        out[f"m{m}_q{q}"] = dict(mode=m, q_e4=q, **res)
    fav = [dict(key=k, **v["FAVORABLE_STABLE"]) for k, v in out.items()]
    for i in pareto(fav, [("P", 1), ("wire_p50_kb", -1)]):
        out[fav[i]["key"]]["pareto_fav_P_payload"] = True
    flat = [dict(key=k, P=v["FAVORABLE_STABLE"]["P"], sv=v["FAVORABLE_STABLE"]["ret"]["seg_vehicle"],
                 pp=v["FAVORABLE_STABLE"]["ret"]["precision_person"]) for k, v in out.items()]
    for i in pareto(flat, [("P", 1), ("sv", 1), ("pp", 1)]):
        out[flat[i]["key"]]["pareto_fav_P_segv_precp"] = True
    return out


def layer_c():
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    scenes = defaultdict(dict)
    for (blob,) in con.execute("SELECT row_json FROM quality_rows WHERE grid_split='fit'"):
        r = json.loads(blob)
        m, q = int(r["mode_id"]), int(r["q_e4"])
        if not in_support(m, q):
            continue
        h = {c: sum(1 for x in r[f"loc_{c}_matched_xy_errors_m"] if float(x) <= TOL[c]) for c in TOL}
        e = {c: int(r[f"loc_{c}_eligible_gt"]) for c in TOL}
        scenes[r["sample_id"]][(m, q)] = dict(w=int(r["udp_application_bytes"]), hv=h["vehicle"], hp=h["person"],
                                              ev=e["vehicle"], ep=e["person"],
                                              iv=int(r["seg_vehicle_intersection_pixels"] or 0), uv=int(r["seg_vehicle_union_pixels"] or 0),
                                              tpp=int(r["loc_person_tp"]), fpp=int(r["loc_person_fp"]),
                                              tpv=int(r["loc_vehicle_tp"]), fpv=int(r["loc_vehicle_fp"]))
    # For each scene and each non-11 cell X: does some mode-11 cell with payload <= X's payload (same scene) match or beat
    # X on vehicle hits, person hits, vehicle seg intersection-over-union and person FP? If none does, X is a per-scene
    # non-11 advantage at its payload.
    per_mode = defaultdict(lambda: dict(cells_scenes=0, not_covered=0, strictly_better_loc=0))
    examples = []
    for sid, cells in scenes.items():
        m11 = [(k, v) for k, v in cells.items() if k[0] == 11]
        for (m, q), x in cells.items():
            if m == 11:
                continue
            pm = per_mode[m]
            pm["cells_scenes"] += 1
            cand = [v for k, v in m11 if v["w"] <= x["w"]]
            iou = lambda v: v["iv"] / v["uv"] if v["uv"] else 1.0
            covered = any(v["hv"] >= x["hv"] and v["hp"] >= x["hp"] and iou(v) >= iou(x) - 1e-9 and v["fpp"] <= x["fpp"]
                          for v in cand)
            if not covered:
                pm["not_covered"] += 1
                best_loc = max((0.4 * v["hv"] + 0.6 * v["hp"] for v in cand), default=-1)
                if 0.4 * x["hv"] + 0.6 * x["hp"] > best_loc:
                    pm["strictly_better_loc"] += 1
                    if len(examples) < 40:
                        examples.append(dict(sample_id=sid, mode=m, q=q, wire_kb=x["w"] / 1e3, hv=x["hv"], hp=x["hp"],
                                             ev=x["ev"], ep=x["ep"], n_m11_cheaper=len(cand)))
    n_scenes = len(scenes)
    return dict(n_scenes=n_scenes, per_mode={str(k): v for k, v in sorted(per_mode.items())}, examples=examples,
                rule="non-11 cell X on scene s is COVERED if some in-support mode-11 cell on s with payload <= X has >= "
                     "vehicle and person within-tolerance hits, >= vehicle seg IoU and <= person FP (quality only, no latency)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    res = dict(layer_A_old_288_favorable=layer_a(), layer_B_current_env_train_evidence=layer_b(a.out / "runs_01"),
               layer_C_per_scene_grid=layer_c())
    with open(a.out / "PART1_PARETO.json", "x") as f:
        json.dump(res, f, indent=1, default=float)
    print("written")
