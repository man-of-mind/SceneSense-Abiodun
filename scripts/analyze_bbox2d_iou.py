"""Offline 2D-box IoU from an eval's test_learned_object_metrics.csv.

The eval writes, for each TP match, the predicted box (input-image px) and the GT box
(original-image px) plus both frame sizes. We normalize both to [0,1] and compute IoU,
overall and per class, plus the fraction of boxes with IoU >= 0.5 / 0.7.

Usage: python3 scripts/analyze_bbox2d_iou.py <test_learned_object_metrics.csv>
"""
import csv
import sys


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def main(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("match_status") != "tp" or not r.get("pred_bbox_x0"):
                continue
            try:
                iw, ih = float(r["input_w"]), float(r["input_h"])
                ow, oh = float(r["orig_w"]), float(r["orig_h"])
                pred = (float(r["pred_bbox_x0"]) / iw, float(r["pred_bbox_y0"]) / ih,
                        float(r["pred_bbox_x1"]) / iw, float(r["pred_bbox_y1"]) / ih)
                cx, cy = float(r["gt_center_x"]), float(r["gt_center_y"])
                bw, bh = float(r["gt_bbox_w"]), float(r["gt_bbox_h"])
                gt = ((cx - bw / 2) / ow, (cy - bh / 2) / oh, (cx + bw / 2) / ow, (cy + bh / 2) / oh)
            except (KeyError, ValueError):
                continue
            rows.append((r.get("gt_class_name", "?"), iou(pred, gt)))
    if not rows:
        print("No TP rows with bbox fields found. (Was the model trained with predict_bbox2d?)")
        return

    def rep(name, sub):
        if not sub:
            return
        ious = [x[1] for x in sub]
        n = len(ious)
        mean = sum(ious) / n
        p50 = 100 * sum(1 for v in ious if v >= 0.5) / n
        p70 = 100 * sum(1 for v in ious if v >= 0.7) / n
        print(f"  {name:8s} n={n:4d}  meanIoU={mean:.3f}  IoU>=0.5={p50:4.0f}%  IoU>=0.7={p70:4.0f}%")

    print(f"2D-box IoU on TP matches  ({path.split('/')[-3] if '/' in path else path})")
    rep("all", rows)
    for c in ("vehicle", "person"):
        rep(c, [x for x in rows if x[0] == c])


if __name__ == "__main__":
    main(sys.argv[1])
