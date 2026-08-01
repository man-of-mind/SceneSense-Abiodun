#!/usr/bin/env python3
"""GATES for the density-knob analysis (DENSITY_ADAPTIVE_KNOB_PLAN.md guardrail 7:
"validate + demote, don't rescue"). Nothing is written to DENSITY_KNOB_RESULTS.md until the gates
that its claims depend on pass; a failing gate demotes the dependent quantity, it is not rescued.

G1  GT-CONVENTION. The plan says GT = actor ORIGIN and hard-fail if origin_x/y is absent. That rule
    was written for the LIVE capture CSVs, where world_x (bbox centre) and origin_x (actor origin)
    coexist and got mixed. This is the OFFLINE eval, and there is exactly one GT column here:
    object_world_x/y. G1 therefore does three things instead of a blind assert:
      (a) hard-fail if object_world_x/y is missing/empty on any scored GT row;
      (b) prove the eval GT column is the SAME column train_fusion.py regresses (self-consistency --
          scoring against anything else would *inject* a convention offset, which is the actual bug
          the guardrail protects against);
      (c) MEASURE the origin-vs-bbox-centre XY delta from the live GT CSVs (which carry both
          columns) so the residual is a number, not an assumption.
G2  MATRIX REPRODUCTION. The published PERMODEL_KNOB_MATRIX_ZSTD.md rows must be reproduced by this
    driver (payload / obj recall / ped recall / loc MAE). This is the real anti-bug check: a wrong GT
    column, a loose matcher, or a broken ROI gate all break it.
G3  FLOOR ANCHOR. loc MAE at no-AE u8 roi0.0 must sit at the offline anchor ~0.95 m (never a loose
    live number ~3 m, never implausibly low).
G4  BIN SAMPLE SIZE. every density bin used for a policy needs enough frames; thin bins are demoted.
G5  ROI MONOTONICITY. payload must fall as the ROI drop fraction rises, for every quant.
G6  DENSITY LABEL AGREEMENT. the density label and the per-frame accuracy denominator must agree.
G7  SEG REPRODUCTION. the NEW seg columns (mIoU + vehicle IoU) must reproduce the published matrix on
    roi 0/0.3/0.5, the anti-bug check for the seg head exactly like G2 is for detection.
"""
from __future__ import annotations

import collections
import csv
import glob
import math
import re
import sys
from pathlib import Path

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
RAW = AB / "rl_agent" / "density_knob" / "raw"
MATRIX = AB / "rl_agent" / "PERMODEL_KNOB_MATRIX_ZSTD.md"
DS = AB / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"
LIVE_GT = sorted(glob.glob(str(AB / "staleness/uplink_only_latency_budget/fresh_run_20260730_000257"
                              / "*/front_metrics/streams/*object_ground_truth.csv")))
ANCHOR_LOC_M = 0.95
MIN_BIN_FRAMES = 100
results: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# ---------------------------------------------------------------- G1 GT convention
def g1_convention() -> dict:
    print("\nG1  GT convention")
    # (a) column present + populated on every scored GT row
    n_rows = n_missing = 0
    with (DS / "object_boxes.csv").open() as fh:
        for b in csv.DictReader(fh):
            if b.get("label") not in ("vehicle", "person") or b.get("gt_source") != "actor":
                continue
            try:
                if float(b.get("gt_distance_m") or 1e9) > 40.0:
                    continue
                if float(b.get("gt_bbox_area_px") or 0.0) < 12.0:
                    continue
            except ValueError:
                continue
            n_rows += 1
            if b.get("object_world_x", "") == "" or b.get("object_world_y", "") == "":
                n_missing += 1
    gate("G1a GT column populated", n_missing == 0 and n_rows > 0,
         f"object_world_x/y present on {n_rows - n_missing}/{n_rows} scored GT rows")

    # (b) the eval GT column == the column the model was TRAINED to regress
    tf = (AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/train_fusion.py").read_text()
    ot = (AB / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/object_targets.py").read_text()
    trains_on_same = ("valid_localization_objects" in tf
                      and '"world_x": _float(row, "object_world_x")' in ot)
    gate("G1b train/eval GT self-consistent", trains_on_same,
         "train_fusion.py -> valid_localization_objects -> world_x = object_world_x "
         "(same column this eval scores against)")

    # (c) measure the origin-vs-bbox-centre XY delta on the live GT (has BOTH columns)
    d_all, per_type = [], collections.defaultdict(list)
    for p in LIVE_GT:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    dx = float(r["world_x"]) - float(r["origin_x"])
                    dy = float(r["world_y"]) - float(r["origin_y"])
                except (ValueError, KeyError, TypeError):
                    continue
                d = math.hypot(dx, dy)
                d_all.append(d)
                per_type[r.get("type_id", "?")].append(d)
    d_all.sort()
    delta = {
        "n": len(d_all),
        "mean": sum(d_all) / len(d_all) if d_all else float("nan"),
        "p50": d_all[len(d_all) // 2] if d_all else float("nan"),
        "p95": d_all[int(0.95 * len(d_all))] if d_all else float("nan"),
        "max": d_all[-1] if d_all else float("nan"),
        "worst_type": max(((t, sum(v) / len(v)) for t, v in per_type.items()), key=lambda kv: kv[1])
        if per_type else ("?", float("nan")),
    }
    gate("G1c convention delta measured", len(d_all) > 1000,
         f"origin-vs-bbox-centre XY delta over {delta['n']} live GT rows: "
         f"mean {delta['mean']:.3f} m, p50 {delta['p50']:.3f} m, p95 {delta['p95']:.3f} m, "
         f"max {delta['max']:.3f} m (worst asset {delta['worst_type'][0]} {delta['worst_type'][1]:.2f} m)")
    return delta


# ---------------------------------------------------------------- load per-frame rows
def load_perframe():
    agg = collections.defaultdict(lambda: collections.Counter())
    frames = collections.Counter()
    for p in sorted(RAW.glob("perframe_*.csv")):
        with p.open() as fh:
            for r in csv.DictReader(fh):
                k = (r["model"], r["quant"], float(r["roi"]))
                a = agg[k]
                for f in ("payload_bytes", "tp", "fp", "fn", "tp_veh", "fp_veh", "fn_veh",
                          "tp_ped", "fp_ped", "fn_ped", "n_pred", "n_inview"):
                    a[f] += int(r[f])
                a["loc_err_sum"] += float(r["loc_err_sum"])
                for ci in range(3):  # seg confusion (row=GT, col=pred); absent in detection-only CSVs
                    for cj in range(3):
                        key = f"conf_{ci}{cj}"
                        if key in r and r[key] != "":
                            a[key] += int(r[key])
                frames[k] += 1
    return agg, frames


def seg_iou(a) -> dict:
    """mIoU + per-class IoU from the summed 3x3 confusion, identical to class_iou_from_confusion."""
    ious = []
    for c in range(3):
        tp = a[f"conf_{c}{c}"]
        fp = sum(a[f"conf_{r}{c}"] for r in range(3)) - tp   # predicted c, GT != c
        fn = sum(a[f"conf_{c}{cj}"] for cj in range(3)) - tp  # GT c, predicted != c
        d = tp + fp + fn
        ious.append(tp / d if d > 0 else float("nan"))
    valid = [v for v in ious if not math.isnan(v)]
    return {"miou": (sum(valid) / len(valid)) if valid else float("nan"),
            "iou_bg": ious[0], "veh_iou": ious[1], "person_iou": ious[2]}


def prof_metrics(a, nframes):
    tp, fp, fn = a["tp"], a["fp"], a["fn"]
    return {
        "payload_kb": a["payload_bytes"] / max(1, nframes) / 1024.0,
        "obj_recall": tp / max(1, tp + fn),
        "obj_precision": tp / max(1, tp + fp),
        "ped_recall": a["tp_ped"] / max(1, a["tp_ped"] + a["fn_ped"]),
        "loc_m": a["loc_err_sum"] / max(1, tp),
        "frames": nframes,
    }


# ---------------------------------------------------------------- G2 matrix reproduction
def parse_matrix_rows() -> dict:
    out = {}
    if not MATRIX.exists():
        return out
    for line in MATRIX.read_text().splitlines():
        if not line.startswith("| ") or "__" not in line:
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 13:
            continue
        try:
            out[c[0]] = {"payload_kb": float(c[5]), "obj_recall": float(c[10]),
                         "ped_recall": float(c[9]), "loc_m": float(c[11]),
                         "miou": float(c[7]), "veh_iou": float(c[8])}
        except ValueError:
            continue
    return out


def g2_reproduce(agg, frames):
    print("\nG2  published-matrix reproduction (this driver vs PERMODEL_KNOB_MATRIX_ZSTD.md)")
    pub = parse_matrix_rows()
    tol = {"payload_kb": 0.03, "obj_recall": 0.015, "ped_recall": 0.015, "loc_m": 0.04}  # rel, abs, abs, abs
    checked = 0
    all_ok = True
    print(f"    {'profile':<24}{'metric':<12}{'mine':>9}{'published':>11}{'delta':>9}")
    for (model, quant, roi), a in sorted(agg.items()):
        if roi not in (0.0, 0.3, 0.5):
            continue  # only these three ROIs exist in the published matrix
        name = f"{model}__{quant.replace('per_channel_', '')}__roi{roi}"
        if name not in pub:
            continue
        m = prof_metrics(a, frames[(model, quant, roi)])
        checked += 1
        for k in ("payload_kb", "obj_recall", "ped_recall", "loc_m"):
            mine, ref = m[k], pub[name][k]
            d = mine - ref
            ok = abs(d) <= (tol[k] * ref if k == "payload_kb" else tol[k])
            all_ok &= ok
            if not ok:
                print(f"    {name:<24}{k:<12}{mine:>9.3f}{ref:>11.3f}{d:>+9.3f}  <-- OUT OF TOL")
    # show the anchor row explicitly
    key = ("noae", "per_channel_uint8", 0.0)
    if key in agg:
        m = prof_metrics(agg[key], frames[key])
        ref = pub.get("noae__uint8__roi0.0", {})
        for k in ("payload_kb", "obj_recall", "ped_recall", "loc_m"):
            print(f"    {'noae__uint8__roi0.0':<24}{k:<12}{m[k]:>9.3f}{ref.get(k, float('nan')):>11.3f}"
                  f"{m[k]-ref.get(k, float('nan')):>+9.3f}")
    gate("G2 matrix reproduction", all_ok and checked >= 24,
         f"{checked} published profiles cross-checked on 4 metrics, all within tolerance"
         if all_ok else f"{checked} checked, at least one metric out of tolerance (see above)")
    return all_ok


# ---------------------------------------------------------------- G7 seg reproduction
def g7_seg_reproduce(agg, frames):
    """The seg columns are NEW in this driver (the first density run was detection-only). They must
    reproduce the published matrix mIoU + vehicle IoU on the roi 0/0.3/0.5 profiles, exactly like G2
    does for detection -- otherwise the seg head is wired wrong and the seg-aware policy is invalid."""
    print("\nG7  seg reproduction (mIoU + vehicle IoU vs PERMODEL_KNOB_MATRIX_ZSTD.md)")
    pub = parse_matrix_rows()
    have_seg = any(agg[k].get("conf_11", 0) for k in agg)
    if not have_seg:
        return gate("G7 seg reproduction", False,
                    "no seg confusion in per-frame CSVs -- re-run density_knob_eval.py with seg")
    tol = {"miou": 0.02, "veh_iou": 0.02}  # absolute IoU tolerance (matrix accept tol was 2%)
    checked, all_ok = 0, True
    print(f"    {'profile':<24}{'metric':<10}{'mine':>9}{'published':>11}{'delta':>9}")
    for (model, quant, roi), a in sorted(agg.items()):
        if roi not in (0.0, 0.3, 0.5):
            continue
        name = f"{model}__{quant.replace('per_channel_', '')}__roi{roi}"
        if name not in pub:
            continue
        s = seg_iou(a)
        checked += 1
        for k in ("miou", "veh_iou"):
            mine, ref = s[k], pub[name][k]
            d = mine - ref
            ok = abs(d) <= tol[k]
            all_ok &= ok
            flag = "  <-- OUT OF TOL" if not ok else ""
            if not ok or name == "noae__uint8__roi0.0":
                print(f"    {name:<24}{k:<10}{mine:>9.3f}{ref:>11.3f}{d:>+9.3f}{flag}")
    gate("G7 seg reproduction", all_ok and checked >= 24,
         f"{checked} profiles cross-checked on mIoU+vehIoU, all within {tol['miou']} IoU"
         if all_ok else f"{checked} checked, at least one seg metric out of tolerance (see above)")
    return all_ok


# ---------------------------------------------------------------- G3 floor anchor
def g3_floor(agg, frames):
    print("\nG3  model floor anchor")
    key = ("noae", "per_channel_uint8", 0.0)
    if key not in agg:
        return gate("G3 floor anchor", False, "no-AE u8 roi0.0 profile absent")
    loc = prof_metrics(agg[key], frames[key])["loc_m"]
    return gate("G3 floor anchor", abs(loc - ANCHOR_LOC_M) <= 0.10,
                f"no-AE u8 roi0.0 loc MAE = {loc:.3f} m vs offline knob-matrix anchor "
                f"{ANCHOR_LOC_M} m (NOT a live loose-matcher number)")


# ---------------------------------------------------------------- G4 bin sample size
def g4_bins():
    print("\nG4  density-bin sample size")
    fd = list(csv.DictReader((RAW / "frame_density.csv").open()))
    c = collections.Counter(r["density_bin"] for r in fd)
    o = collections.Counter()
    for r in fd:
        o[r["density_bin"]] += int(r["n_inview"])
    ok = True
    usable = {}
    for b in ("0", "1-2", "3-4", "5+"):
        good = c[b] >= MIN_BIN_FRAMES
        usable[b] = good
        ok &= good
        print(f"    bin {b:<4} frames={c[b]:>5}  in-view GT objects={o[b]:>5}  "
              f"{'usable' if good else 'THIN -> DEMOTE'}")
    gate("G4 bin sample size", ok, f"all four bins >= {MIN_BIN_FRAMES} frames "
         f"(min {min(c.values())})" if ok else "at least one bin thin -- demote it")
    return usable, c, o


# ---------------------------------------------------------------- G5 ROI monotonicity
def g5_monotonic(agg, frames):
    print("\nG5  ROI payload monotonicity (physics sanity)")
    bad = []
    for model in sorted({k[0] for k in agg}):
        for quant in sorted({k[1] for k in agg if k[0] == model}):
            ks = sorted([k for k in agg if k[0] == model and k[1] == quant], key=lambda k: k[2])
            kb = [prof_metrics(agg[k], frames[k])["payload_kb"] for k in ks]
            if any(kb[i + 1] >= kb[i] for i in range(len(kb) - 1)):
                bad.append(f"{model}/{quant}: {[round(x, 1) for x in kb]}")
    return gate("G5 ROI monotonicity", not bad,
                "payload falls strictly with the ROI drop fraction for every model x quant"
                if not bad else "; ".join(bad))


# ---------------------------------------------------------------- G6 density label agreement
def g6_label_agreement():
    """The density label (frame_density.csv) and the accuracy denominator (per-frame eval rows) are
    computed by two independent code paths. They must agree frame-by-frame, or a frame could be
    binned as 'empty' while being scored against objects (or vice versa)."""
    print("\nG6  density label == accuracy denominator")
    want = {r["sample_id"]: (int(r["n_inview"]), int(r["n_inview_veh"]), int(r["n_inview_ped"]))
            for r in csv.DictReader((RAW / "frame_density.csv").open())}
    n = bad = 0
    for p in sorted(RAW.glob("perframe_*.csv")):
        for r in csv.DictReader(p.open()):
            got = (int(r["n_inview"]), int(r["n_inview_veh"]), int(r["n_inview_ped"]))
            n += 1
            if want.get(r["sample_id"]) != got:
                bad += 1
    return gate("G6 density label agreement", bad == 0 and n > 0,
                f"{n - bad}/{n} profile-frame rows agree with the independently built density label")


def main() -> int:
    print("=" * 92)
    print("DENSITY-KNOB GATES")
    print("=" * 92)
    delta = g1_convention()
    agg, frames = load_perframe()
    if not agg:
        print("\nno per-frame CSVs yet -- run density_knob_eval.py first")
        return 2
    print(f"\nloaded {len(agg)} profiles x {max(frames.values())} frames")
    g2_reproduce(agg, frames)
    g7_seg_reproduce(agg, frames)
    g3_floor(agg, frames)
    g4_bins()
    g5_monotonic(agg, frames)
    g6_label_agreement()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 92)
    print(f"{len(results) - n_fail}/{len(results)} gates PASS" + ("" if not n_fail else f"  ({n_fail} FAIL)"))
    print("=" * 92)
    (RAW / "gate_report.txt").write_text(
        "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}" for n, ok, d in results) + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
