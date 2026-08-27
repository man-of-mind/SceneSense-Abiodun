#!/usr/bin/env python3
"""Contact sheet for score-0.02 person false negatives (MANUAL REVIEW ONLY).

CARLA 0.10 does not render walker semantics in this corpus, so there is no automatic
pedestrian visibility gate and none is invented here.  This sheet exists so a human can
look at the pedestrians the detector misses.  Nothing in it classifies a pedestrian as
visible or occluded, and no recall denominator is changed anywhere on the basis of it.

Sampling is deterministic: person FNs (missed by BOTH the current and the corrected
decoder at score 0.02) are sorted by (sample_id, gt_index) and drawn round-robin across
the four distance bands, so the sheet is reproducible from the same audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

BANDS = ("0-10", "10-20", "20-30", "30-40")


def overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return float(inter / area)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panels", type=int, default=32)
    args = parser.parse_args()

    records = json.loads((args.audit_dir / "person_fn_records.json").read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("no person false negatives recorded")

    boxes_by_sample: Dict[str, List[dict]] = defaultdict(list)
    with (args.dataset_dir / "object_boxes.csv").open(newline="", encoding="utf-8") as fh:
        for b in csv.DictReader(fh):
            boxes_by_sample[str(b.get("sample_id", ""))].append(b)

    by_band: Dict[str, List[dict]] = {b: [] for b in BANDS}
    for rec in sorted(records, key=lambda r: (str(r["sample_id"]), int(r["gt_index"]))):
        by_band.setdefault(str(rec["distance_band"]), []).append(rec)
    chosen: List[dict] = []
    cursor = {b: 0 for b in by_band}
    while len(chosen) < int(args.panels):
        progressed = False
        for band in BANDS:
            pool = by_band.get(band, [])
            if cursor.get(band, 0) < len(pool) and len(chosen) < int(args.panels):
                chosen.append(pool[cursor[band]])
                cursor[band] += 1
                progressed = True
        if not progressed:
            break

    cols = 8
    rows = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.35, rows * 2.85))
    axes = np.atleast_2d(axes)
    table_rows: List[dict] = []
    for i, ax in enumerate(axes.reshape(-1)):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(chosen):
            ax.axis("off")
            continue
        rec = chosen[i]
        img = Image.open(args.dataset_dir / rec["rgb_path"]).convert("RGB")
        W, H = img.size
        bx = [float(v) for v in rec["box_orig"]]
        cx, cy = [float(v) for v in rec["center_orig"]]
        pad = max(28.0, 1.6 * max(bx[2] - bx[0], bx[3] - bx[1]))
        cx0, cy0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        cx1, cy1 = min(W, int(cx + pad)), min(H, int(cy + pad))
        ax.imshow(np.asarray(img.crop((cx0, cy0, cx1, cy1))))
        ax.add_patch(
            mpatches.Rectangle(
                (bx[0] - cx0, bx[1] - cy0),
                bx[2] - bx[0],
                bx[3] - bx[1],
                fill=False,
                edgecolor="#ff2d55",
                linewidth=1.4,
            )
        )
        ax.plot([cx - cx0], [cy - cy0], marker="+", color="#00e5ff", markersize=7, mew=1.4)

        # Nearer-box overlap: any GT box (either class, gate-independent) whose distance is
        # smaller than this pedestrian's. Reported as a raw geometric quantity, not a verdict.
        best_overlap, best_label = 0.0, ""
        for b in boxes_by_sample.get(str(rec["sample_id"]), []):
            if b.get("gt_source") != "actor":
                continue
            try:
                d = float(b.get("gt_distance_m") or 1e9)
                bw, bh = float(b.get("gt_bbox_w") or 0), float(b.get("gt_bbox_h") or 0)
                bcx, bcy = float(b.get("gt_center_x") or 0), float(b.get("gt_center_y") or 0)
            except ValueError:
                continue
            if d >= float(rec["distance_m"]) - 1e-6 or bw <= 0 or bh <= 0:
                continue
            ov = overlap_fraction(
                bx, (bcx - bw / 2.0, bcy - bh / 2.0, bcx + bw / 2.0, bcy + bh / 2.0)
            )
            if ov > best_overlap:
                best_overlap, best_label = ov, str(b.get("label", ""))
        ax.set_title(
            f"{rec['distance_m']:.1f} m  {rec['gt_bbox_w_orig']:.0f}x{rec['gt_bbox_h_orig']:.0f}px\n"
            f"radar={rec['radar_support_points']}  nearer-ov={best_overlap:.2f}"
            f"{(' ' + best_label) if best_label else ''}",
            fontsize=6.2,
        )
        table_rows.append(
            {
                "panel": i,
                "sample_id": rec["sample_id"],
                "gt_index": rec["gt_index"],
                "distance_m": rec["distance_m"],
                "distance_band": rec["distance_band"],
                "gt_bbox_w_orig": rec["gt_bbox_w_orig"],
                "gt_bbox_h_orig": rec["gt_bbox_h_orig"],
                "gt_bbox_w_input": rec["gt_bbox_w_input"],
                "gt_bbox_h_input": rec["gt_bbox_h_input"],
                "radar_support_points": rec["radar_support_points"],
                "nearer_box_overlap_fraction": round(best_overlap, 4),
                "nearer_box_label": best_label,
                "person_semantic_tag_px_in_frame": rec["person_tag_px_frame"],
                "manual_review_verdict": "",
            }
        )
    fig.suptitle(
        "Score-0.02 person false negatives (missed by BOTH current and corrected decoders)\n"
        "LABELLED FOR MANUAL REVIEW - not an automatic visibility classification. "
        "CARLA 0.10 renders no walker semantics in this corpus, so pedestrian visibility is UNRESOLVED.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    csv_path = args.out.with_name(args.out.stem + "_manual_review.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        for r in table_rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(table_rows)} panels) and {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
