#!/usr/bin/env python3
"""Create-only fixed-contract Route B validation evaluation for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_ROOT = HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for path in (HERE, PKG_ROOT, PKG_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_head_pilot_v1.evaluate_route_b_checkpoint_v1 import (  # noqa: E402
    FIXED_DECODER,
    load_rows,
    summarize,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--object-score-threshold", type=float, choices=(0.02, 0.20), required=True)
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("only the validation split is authorized; the test split stays locked")

    exp_dir = args.experiment_dir.resolve()
    out_dir = exp_dir / "eval" / args.tag
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {out_dir}")
    checkpoint = args.checkpoint.resolve(strict=True)
    command = [
        args.python, str(HERE / "eval_entry_v1.py"),
        "--config", str(Path(args.config).resolve(strict=True)),
        "--experiment-dir", str(exp_dir),
        "--checkpoint", str(checkpoint),
        "--split", "val",
        "--require-cuda",
        "--object-score-threshold", str(float(args.object_score_threshold)),
        "--topk-objects", str(FIXED_DECODER["topk_objects"]),
        "--object-nms-radius-px", str(FIXED_DECODER["object_nms_radius_px"]),
        "--match-distance-m", str(FIXED_DECODER["match_distance_m"]),
        "--max-gt-distance-m", str(FIXED_DECODER["max_gt_distance_m"]),
        "--feature-drop-fraction", "0.0",
    ]
    result = subprocess.run(command, cwd=PKG_ROOT)
    if result.returncode:
        return int(result.returncode)

    out_dir.mkdir(parents=True)
    metrics_dir = exp_dir / "metrics"
    for destination, source in (
        ("evaluator_metrics.json", metrics_dir / "val_fusion_evaluation_metrics.json"),
        ("detections.csv", metrics_dir / "val_learned_object_metrics.csv"),
    ):
        shutil.move(str(source), str(out_dir / destination))
    figure = exp_dir / "figures" / "val_fusion_confusion.png"
    if figure.exists():
        shutil.move(str(figure), str(out_dir / "confusion.png"))

    manifest_rows = load_rows(exp_dir / "dataset" / "manifest.csv")
    split_ids = {row["sample_id"] for row in manifest_rows if row.get("split") == "val"}
    detections = load_rows(out_dir / "detections.csv")
    evaluator = json.loads((out_dir / "evaluator_metrics.json").read_text(encoding="utf-8"))
    derived = {
        "tag": args.tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split": "val",
        "fixed_decoder": {
            **FIXED_DECODER,
            "object_score_threshold": float(args.object_score_threshold),
            "min_gt_area_px": 12.0,
            "class_aware": True,
        },
        "segmentation": {
            key: evaluator.get(key)
            for key in ("miou", "vehicle_iou", "person_iou", "background_iou", "pixel_accuracy")
        },
        "primary": summarize(
            detections,
            frame_ids=split_ids,
            duplicate_radius_m=3.0,
            label="all_val_frames",
        ),
    }
    (out_dir / "derived_metrics.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(derived, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

