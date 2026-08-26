#!/usr/bin/env python3
"""Route B validation decode for hybrid (and baseline) checkpoints.

Identical decoder contract and identical derived-metric code as the pilot's
``evaluate_route_b_checkpoint_v1.py`` - ``FIXED_DECODER``, ``summarize`` and
``load_rows`` are *imported* from it rather than copied, so the two cannot drift.
The only difference is the evaluator entry point: this one goes through
``eval_entry_v1``, which installs the ``centerfusion_hybrid_v1`` builder before
handing over to the untouched production evaluator. Checkpoints with any other
``object_head_arch`` (the frozen baseline, for instance) fall through to the
production builder unchanged, so both models are decoded by the same code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
PILOT_ROOT = _HERE.parent
PKG_ROOT = PILOT_ROOT.parent
for _p in (str(_HERE), str(PKG_ROOT), str(PKG_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from object_head_pilot_v1.evaluate_route_b_checkpoint_v1 import (  # noqa: E402
    FIXED_DECODER,
    load_rows,
    summarize,
)

ENTRY = _HERE / "eval_entry_v1.py"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--duplicate-radius-m", type=float, default=3.0)
    parser.add_argument("--feature-drop-fraction", type=float, default=0.0)
    parser.add_argument("--object-score-threshold", type=float, choices=(0.02, 0.20), default=0.20)
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args(argv)

    if args.split == "test":
        raise SystemExit("the Route B test split is locked for this task")

    config_path = str(Path(args.config).expanduser().resolve(strict=True))
    exp_dir = args.experiment_dir.resolve()
    out_dir = exp_dir / "eval" / args.tag
    if out_dir.exists():
        print(f"refusing to overwrite an existing evaluation: {out_dir}", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint.resolve(strict=True)

    command = [
        args.python, str(ENTRY),
        "--config", config_path,
        "--experiment-dir", str(exp_dir),
        "--checkpoint", str(checkpoint),
        "--split", args.split,
        "--require-cuda",
        "--object-score-threshold", str(float(args.object_score_threshold)),
        "--topk-objects", str(FIXED_DECODER["topk_objects"]),
        "--object-nms-radius-px", str(FIXED_DECODER["object_nms_radius_px"]),
        "--match-distance-m", str(FIXED_DECODER["match_distance_m"]),
        "--max-gt-distance-m", str(FIXED_DECODER["max_gt_distance_m"]),
        "--feature-drop-fraction", str(float(args.feature_drop_fraction)),
    ]
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(PKG_ROOT))
    if result.returncode != 0:
        return result.returncode

    out_dir.mkdir(parents=True)
    metrics_dir = exp_dir / "metrics"
    for name, path in {
        "evaluator_metrics.json": metrics_dir / f"{args.split}_fusion_evaluation_metrics.json",
        "detections.csv": metrics_dir / f"{args.split}_learned_object_metrics.csv",
    }.items():
        shutil.move(str(path), str(out_dir / name))
    figure = exp_dir / "figures" / f"{args.split}_fusion_confusion.png"
    if figure.exists():
        shutil.move(str(figure), str(out_dir / "confusion.png"))

    manifest_rows = load_rows(exp_dir / "dataset" / "manifest.csv")
    split_ids = {str(row["sample_id"]) for row in manifest_rows if row.get("split") == args.split}
    collision_ids = {
        str(row["sample_id"])
        for row in load_rows(exp_dir / "provenance" / "collision_window_samples.csv")
        if row.get("split") == args.split and str(row.get("retained_in_dataset")) == "1"
    }
    detections = load_rows(out_dir / "detections.csv")
    evaluator_metrics = json.loads((out_dir / "evaluator_metrics.json").read_text(encoding="utf-8"))

    derived = {
        "tag": args.tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "split": args.split,
        "fixed_decoder": {**FIXED_DECODER,
                          "object_score_threshold": float(args.object_score_threshold)},
        "feature_drop_fraction": float(args.feature_drop_fraction),
        "segmentation": {key: evaluator_metrics.get(key)
                         for key in ("miou", "vehicle_iou", "person_iou",
                                     "background_iou", "pixel_accuracy")},
        "primary": summarize(detections, frame_ids=split_ids,
                             duplicate_radius_m=args.duplicate_radius_m,
                             label="all_val_frames"),
        "collision_window_excluded": summarize(detections, frame_ids=split_ids - collision_ids,
                                               duplicate_radius_m=args.duplicate_radius_m,
                                               label="collision_window_excluded"),
        "collision_window_frames_excluded": len(collision_ids & split_ids),
    }
    (out_dir / "derived_metrics.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "segmentation": derived["segmentation"],
                      **{k: v for k, v in derived["primary"].items()
                         if "precision" in k or "recall" in k or "_f1" in k or "xy_mae" in k}},
                     indent=2, sort_keys=True), flush=True)
    return 0


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
