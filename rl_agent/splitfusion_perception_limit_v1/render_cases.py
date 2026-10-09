"""Render representative full-local failure cases (RGB from the retained prepared input; GT and predictions).

Green: eligible GT (vehicle box / person projected centre, labelled with actor and status).
Red: final records (score >= threshold used by the evaluator); orange: pre-p025 person candidates
within 3 m of a failing person. Right panel: radar channel 3 (raw as stored in the prepared input).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

CELL = Path("experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11/treatment/capture/cells/a71__favorable_stable")
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
CASES = [(1527, "person 101 FN: 40 m, no candidate within 3 m"),
         (1563, "person 101 FN: detected in image, 4.4 m radial error"),
         (1619, "person 112 FN (next to vehicles) + person 139 2.9 m radial"),
         (1643, "person 139 TP beyond 1.2 m: +2.9 m radial"),
         (1479, "vehicle 80 FN: box IoU 0.85, +8.8 m radial"),
         (1571, "vehicle 68 FN: box IoU 0.81, +3.4 m radial")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    objs = json.loads(a.objects.read_text())
    fig, axes = plt.subplots(len(CASES), 2, figsize=(18, 4.6 * len(CASES)), gridspec_kw={"width_ratios": [3, 1.3]})
    for row, (fid, title) in enumerate(CASES):
        ev = json.loads((a.evidence / f"frames/frame_{fid}.json").read_text())
        x = np.load(CELL / "paired_input_archive" / ev["source_input_file"])[0]
        rgb = np.clip(x[:3, :432].transpose(1, 2, 0) * STD + MEAN, 0, 1)
        ax = axes[row, 0]
        ax.imshow(rgb)
        for o in [o for o in objs if o["frame_id"] == fid]:
            if o["cls"] == "vehicle":
                b = o["box_model"]
                ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.5))
                ax.text(b[0], b[1] - 3, f"GT v{o['actor'][:4]} {o['status']} d={o['distance_m']:.0f}m", color="lime", fontsize=9)
            else:
                cx, cy = o["center_model_px"]
                ax.plot(cx, cy, "o", mfc="none", mec="lime", ms=14, mew=2)
                ax.text(cx + 8, cy + 14, f"GT p{o['actor']} {o['status']} d={o['distance_m']:.0f}m", color="lime", fontsize=9)
        for p in ev["prediction_records"]:
            thr = .25 if p["class_name"] == "person" else .20
            if p["score"] < thr:
                continue
            ax.add_patch(mp.Rectangle((p["bbox_x0"], p["bbox_y0"]), p["bbox_x1"] - p["bbox_x0"], p["bbox_y1"] - p["bbox_y0"],
                                      fill=False, ec="red", lw=1))
            ax.text(p["bbox_x0"], p["bbox_y1"] + 9, f"{p['class_name'][0]} {p['score']:.2f} r={p['actor_forward_depth_m']:.0f}m",
                    color="red", fontsize=8)
        ax.set_title(f"frame {fid}: {title}", fontsize=11)
        ax.axis("off")
        ax2 = axes[row, 1]
        ax2.imshow(x[3, :432], cmap="magma")
        ax2.set_title("prepared input channel 3 (radar raster)", fontsize=10)
        ax2.axis("off")
    fig.tight_layout()
    fig.savefig(a.output, dpi=110)


if __name__ == "__main__":
    main()
