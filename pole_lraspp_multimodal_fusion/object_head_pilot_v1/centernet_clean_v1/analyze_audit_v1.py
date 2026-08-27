#!/usr/bin/env python3
"""Aggregate the CenterNet evaluation-contract audit into report tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True, type=Path)
    args = ap.parse_args()
    A = args.audit_dir
    summary = json.loads((A / "audit_summary.json").read_text(encoding="utf-8"))
    frames = load(A / "per_frame_decoder_audit.csv")
    grid = load(A / "target_grid_audit.csv")
    objs = load(A / "gt_object_audit.csv")
    n = len(frames)
    out = {}

    # ---------- Phase 2: budget accounting ----------
    p2 = {}
    for tag, thr in (("020", 0.20), ("002", 0.02)):
        e = {}
        raw = dcell = disc = surv = scell = nlm = 0
        for c in ("vehicle", "person"):
            r = sum(f(x, f"cur_{c}_raw_above_thr_{tag}") for x in frames)
            d = sum(f(x, f"cur_{c}_distinct_native_cells_{tag}") for x in frames)
            k = sum(f(x, f"cur_{c}_discarded_neighbour_dup_{tag}") for x in frames)
            s = sum(f(x, f"cur_{c}_survivors_{tag}") for x in frames)
            sc = sum(f(x, f"cur_{c}_survivor_native_cells_{tag}") for x in frames)
            m = sum(f(x, f"native_localmax_{c}_{tag}") for x in frames)
            e[c] = {
                "raw_topk_entries_above_thr": int(r),
                "distinct_native_cells": int(d),
                "discarded_as_neighbour_dup": int(k),
                "survivors": int(s),
                "survivor_distinct_native_cells": int(sc),
                "native_localmax_above_thr": int(m),
                "per_frame_raw": r / n,
                "per_frame_distinct_cells": d / n,
                "per_frame_survivors": s / n,
                "per_frame_native_localmax": m / n,
                "budget_share_duplicate_pixels": (r - d) / max(1.0, r),
                "native_peaks_lost_by_current": (m - sc) / max(1.0, m),
            }
            raw += r; dcell += d; disc += k; surv += s; scell += sc; nlm += m
        e["both_classes"] = {
            "raw_topk_entries_above_thr": int(raw),
            "distinct_native_cells": int(dcell),
            "discarded_as_neighbour_dup": int(disc),
            "survivors": int(surv),
            "survivor_distinct_native_cells": int(scell),
            "native_localmax_above_thr": int(nlm),
            "per_frame_raw": raw / n,
            "per_frame_distinct_cells": dcell / n,
            "per_frame_survivors": surv / n,
            "per_frame_native_localmax": nlm / n,
            "budget_share_duplicate_pixels": (raw - dcell) / max(1.0, raw),
        }
        e["frames_current_topk_saturated"] = sum(int(f(x, f"cur_topk_saturated_{tag}")) for x in frames)
        e["frames_native_topk_saturated"] = sum(int(f(x, f"native_topk_saturated_{tag}")) for x in frames)
        e["frames_hybrid_topk_saturated"] = sum(int(f(x, f"hybrid_topk_saturated_{tag}")) for x in frames)
        e["frames_native_cap512_saturated"] = sum(
            int(f(x, f"native_cap_topk_saturated_{tag}")) for x in frames
        )
        p2[f"{thr:.2f}"] = e
    out["phase2_budget"] = p2

    # ---------- Phase 3 ----------
    out["phase3_matching"] = summary["greedy_vs_bipartite"]

    # ---------- Phase 4 ----------
    tot_elig = sum(f(x, "eligible_gt_total") for x in grid)
    p4 = {
        "frames": len(grid),
        "eligible_gt_total": int(tot_elig),
        "eligible_gt_vehicle": int(sum(f(x, "eligible_gt_vehicle") for x in grid)),
        "eligible_gt_person": int(sum(f(x, "eligible_gt_person") for x in grid)),
        "targets_placed_total": int(sum(f(x, "targets_placed") for x in grid)),
        "targets_dropped_out_of_input": int(sum(f(x, "targets_dropped_out_of_input") for x in grid)),
        "gt_aux_array_overflow_objects": int(sum(f(x, "gt_array_capped") for x in grid)),
        "frames_over_max_objects_64": sum(1 for x in grid if f(x, "eligible_gt_total") > 64),
        "fullres_peak_collisions_same_class": int(
            sum(f(x, "fullres_peak_collisions_same_class") for x in grid)
        ),
        "native_cell_collisions_same_class_vehicle": int(
            sum(f(x, "native_cell_collisions_same_class_vehicle") for x in grid)
        ),
        "native_cell_collisions_same_class_person": int(
            sum(f(x, "native_cell_collisions_same_class_person") for x in grid)
        ),
        "native_cells_with_two_classes": int(sum(f(x, "native_cells_with_two_classes") for x in grid)),
        "reg_target_overwrites": int(sum(f(x, "reg_target_overwrites") for x in grid)),
    }
    for c in ("vehicle", "person"):
        p4[f"subcell_w_{c}"] = int(sum(f(x, f"subcell_w_{c}") for x in grid))
        p4[f"subcell_h_{c}"] = int(sum(f(x, f"subcell_h_{c}") for x in grid))
        p4[f"subcell_either_{c}"] = int(sum(f(x, f"subcell_either_{c}") for x in grid))
    out["phase4_target_parity"] = p4

    # recall by distance / size band
    bands = defaultdict(lambda: defaultdict(int))
    for o in objs:
        for keyname, val in (("dist", o["distance_band"]), ("size", o["input_size_band"])):
            k = (o["class_name"], keyname, val)
            bands[k]["n"] += 1
            for v in ("current_greedy_002", "current_greedy_020", "hybrid_greedy_002", "hybrid_greedy_020",
                      "native_greedy_002", "native_greedy_020"):
                bands[k][v] += int(f(o, f"tp_{v}"))
    band_rows = []
    for (cls, kind, val), d in sorted(bands.items()):
        r = {"class_name": cls, "band_kind": kind, "band": val, "n_gt": d["n"]}
        for v in ("current_greedy_020", "hybrid_greedy_020", "current_greedy_002", "hybrid_greedy_002",
                  "native_greedy_020", "native_greedy_002"):
            r[f"recall_{v}"] = round(d[v] / max(1, d["n"]), 4)
            r[f"tp_{v}"] = d[v]
        band_rows.append(r)
    with (A / "recall_by_band.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(band_rows[0].keys()))
        w.writeheader()
        for r in band_rows:
            w.writerow(r)
    out["phase4_bands_csv"] = "recall_by_band.csv"

    # ---------- Phase 5 ----------
    veh = [o for o in objs if o["class_name"] == "vehicle"]
    zero = [o for o in veh if f(o, "vehicle_semantic_px_in_box") == 0]
    pos = [o for o in veh if f(o, "vehicle_semantic_px_in_box") > 0]
    def rec(rows, key):
        return sum(int(f(o, key)) for o in rows) / max(1, len(rows))
    out["phase5_vehicle_semantic_support"] = {
        "vehicle_gt_total": len(veh),
        "zero_support_boxes": len(zero),
        "positive_support_boxes": len(pos),
        "zero_support_fraction": len(zero) / max(1, len(veh)),
        "recall_zero_support_020": round(rec(zero, "tp_current_greedy_020"), 4),
        "recall_positive_support_020": round(rec(pos, "tp_current_greedy_020"), 4),
        "recall_zero_support_002": round(rec(zero, "tp_current_greedy_002"), 4),
        "recall_positive_support_002": round(rec(pos, "tp_current_greedy_002"), 4),
        "recall_zero_support_020_corrected": round(rec(zero, "tp_hybrid_greedy_020"), 4),
        "recall_positive_support_020_corrected": round(rec(pos, "tp_hybrid_greedy_020"), 4),
        "median_support_px_positive": float(
            np.median([f(o, "vehicle_semantic_px_in_box") for o in pos]) if pos else 0.0
        ),
    }
    per = [o for o in objs if o["class_name"] == "person"]
    out["phase5_person"] = {
        "person_gt_total": len(per),
        "person_fn_at_002_current_and_corrected": sum(
            1 for o in per if not int(f(o, "tp_current_greedy_002")) and not int(f(o, "tp_hybrid_greedy_002"))
        ),
        "frames_with_any_walker_semantic_pixel": sum(
            1 for x in grid if f(x, "person_semantic_tag_px") > 0
        ),
        "note": "no automatic pedestrian visibility gate exists in this corpus",
    }

    # ---------- decoder_metric_comparison.csv ----------
    rows = []
    for key in sorted(summary["variants"]):
        v = summary["variants"][key]
        name, thr = key.split("@")
        r = {"variant": name, "score_threshold": thr}
        for k in ("tp", "fp", "fn", "n_pred", "precision", "recall", "f1", "xy_mae_m"):
            r[k] = v[k]
        for c in ("vehicle", "person"):
            for k in ("tp", "fp", "fn", "precision", "recall", "f1", "xy_mae_m", "dimension_mae_m"):
                r[f"{c}_{k}"] = v.get(f"{c}_{k}")
        rows.append(r)
    with (A / "decoder_metric_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (A / "analysis_tables.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
