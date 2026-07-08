#!/usr/bin/env python3
"""CEP (Circular Error Probability) of model localization vs distance, per radar-pps model.

For each of the 5 pps models we reuse the EXISTING per-detection eval CSV (test_learned_object_metrics.csv,
which carries global_xy_error_m + gt_world_x/y per matched detection) and join each detection to its
gt_distance_m from the model's dataset object_boxes.csv. CEP50 = median radial error in a distance bin;
CEP95 = 95th percentile. Pure aggregation — no re-inference, no CARLA.

Outputs (to abiodun/cooperative_fusion/pps_study_figs/):
  cep_heatmap_<class>.{png,pdf}  — x=distance bin, y=pps, color=CEP50 (the "at 15m & 250k -> CEP" lookup)
  cep_lines_<class>.{png,pdf}    — x=distance, y=CEP, one line per pps (error-growth curve)
  ../../CEP_SUMMARY.md           — tables for the slides
"""
from __future__ import annotations
import csv, os, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
OUT = AB / "cooperative_fusion" / "pps_study_figs"; OUT.mkdir(parents=True, exist_ok=True)

EVAL_DIR = {
    100000: AB / "experiments/autonomous_arch_runs_20260625/det_stage2c_centerw4/eval_best_thr010",
    150000: AB / "experiments/eval_pps150000_v2_nms6",
    200000: AB / "experiments/eval_pps200000_v2_nms6",
    250000: AB / "experiments/eval_pps250000_v2_nms6",
    300000: AB / "experiments/eval_pps300000_v2_nms6",
}
PPS = sorted(EVAL_DIR)
BINS = list(range(0, 45, 5))  # 0-5,...,40-45 m (detection gated <=40 m)
CLASSES = ["person", "vehicle"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 12, "axes.titleweight": "bold"})


def _dataset_object_boxes(eval_dir: Path):
    ds = (eval_dir / "dataset")
    ds = Path(os.path.realpath(ds)) if ds.exists() else None
    if ds and (ds / "object_boxes.csv").exists():
        return ds / "object_boxes.csv"
    return None


def load_errors(pps):
    """Return list of (gt_class, distance_m, radial_error_m) for matched (tp) detections."""
    ed = EVAL_DIR[pps]
    csvf = ed / "metrics" / "test_learned_object_metrics.csv"
    obf = _dataset_object_boxes(ed)
    if not csvf.exists() or obf is None:
        print(f"  !! pps={pps}: missing eval csv or object_boxes ({csvf.exists()}, {obf})")
        return []
    idx = {}
    for r in csv.DictReader(open(obf)):
        try:
            idx.setdefault(r["sample_id"], []).append(
                (float(r["object_world_x"]), float(r["object_world_y"]), float(r["gt_distance_m"])))
        except (KeyError, ValueError):
            pass
    out = []
    for r in csv.DictReader(open(csvf)):
        if r.get("match_status") != "tp":
            continue
        try:
            gx, gy, err = float(r["gt_world_x"]), float(r["gt_world_y"]), float(r["global_xy_error_m"])
        except (KeyError, ValueError):
            continue
        cand = idx.get(r["sample_id"])
        if not cand:
            continue
        wx, wy, dist = min(cand, key=lambda c: (c[0] - gx) ** 2 + (c[1] - gy) ** 2)
        if (wx - gx) ** 2 + (wy - gy) ** 2 > 1.0:
            continue
        out.append((str(r.get("gt_class_name", "")).lower(), dist, err))
    return out


def cep_table(data, cls):
    """data: list of (class,dist,err). Returns {pps: {bin: (n, cep50, cep95)}} for one class."""
    table = {}
    for pps in PPS:
        errs_by_bin = {}
        for c, dist, err in data[pps]:
            if c != cls:
                continue
            b = int(dist // 5) * 5
            if b < BINS[0] or b > BINS[-1]:
                continue
            errs_by_bin.setdefault(b, []).append(err)
        row = {}
        for b, e in errs_by_bin.items():
            e = sorted(e)
            if len(e) < 5:
                continue
            row[b] = (len(e), e[int(0.5 * len(e))], e[min(len(e) - 1, int(0.95 * len(e)))])
        table[pps] = row
    return table


def fig_heatmap(table, cls):
    bins = BINS[:-1]
    M = np.full((len(PPS), len(bins)), np.nan)
    for i, pps in enumerate(PPS):
        for j, b in enumerate(bins):
            if b in table[pps]:
                M[i, j] = table[pps][b][1]  # CEP50
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([f"{b}-{b+5}" for b in bins])
    ax.set_yticks(range(len(PPS))); ax.set_yticklabels([f"{p//1000}k" for p in PPS])
    ax.set_xlabel("distance to object (m)"); ax.set_ylabel("radar pps")
    ax.set_title(f"{cls.capitalize()} localization CEP50 (m) by distance & radar pps")
    for i in range(len(PPS)):
        for j in range(len(bins)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] > np.nanmean(M) else "black", fontsize=10)
    cb = fig.colorbar(im, ax=ax); cb.set_label("CEP50 (m)  — 50% of detections fall within this radius")
    fig.tight_layout()
    fig.savefig(OUT / f"cep_heatmap_{cls}.png", bbox_inches="tight")
    fig.savefig(OUT / f"cep_heatmap_{cls}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_lines(table, cls):
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, pps in enumerate(PPS):
        xs = [b + 2.5 for b in BINS[:-1] if b in table[pps]]
        ys = [table[pps][b][1] for b in BINS[:-1] if b in table[pps]]
        if xs:
            ax.plot(xs, ys, "-o", color=cmap(i / (len(PPS) - 1)), lw=2, ms=6, label=f"{pps//1000}k")
    ax.set_xlabel("distance to object (m)"); ax.set_ylabel("CEP50 (m)")
    ax.set_title(f"{cls.capitalize()} localization error grows with distance")
    ax.grid(alpha=0.25); ax.legend(title="radar pps", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"cep_lines_{cls}.png", bbox_inches="tight")
    fig.savefig(OUT / f"cep_lines_{cls}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    global PPS
    data = {}
    for pps in PPS:
        data[pps] = load_errors(pps)
        print(f"pps {pps//1000}k: {len(data[pps])} matched detections with distance")
    # keep only models with data (100k's prior-collection dataset is gone -> no distance join)
    dropped = [p for p in PPS if not data[p]]
    if dropped:
        print(f"(omitting {[p//1000 for p in dropped]}k — no dataset for distance join)")
    PPS = [p for p in PPS if data[p]]
    md = ["# CEP (Circular Error Probability) — localization accuracy vs distance\n",
          "CEP50 = radius (m) within which 50% of matched detections land (median radial error); "
          "CEP95 = 95th percentile. Reused existing per-detection eval (global_xy_error_m) + gt_distance_m.\n"]
    for cls in CLASSES:
        table = cep_table(data, cls)
        fig_heatmap(table, cls); fig_lines(table, cls)
        md.append(f"\n## {cls.capitalize()} — CEP50 (m) by distance & pps")
        header = "| pps | " + " | ".join(f"{b}-{b+5} m" for b in BINS[:-1]) + " |"
        md.append(header); md.append("|" + "---|" * (len(BINS)))
        for pps in PPS:
            cells = [f"{table[pps][b][1]:.2f}" if b in table[pps] else "—" for b in BINS[:-1]]
            md.append(f"| {pps//1000}k | " + " | ".join(cells) + " |")
    (AB / "CEP_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
