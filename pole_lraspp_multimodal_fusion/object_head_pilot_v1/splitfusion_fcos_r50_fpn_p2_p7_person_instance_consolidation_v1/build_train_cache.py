from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.labeling import (
    contract_world_targets,
)

from .core import (
    GROUP_IOU_THRESHOLDS,
    SEMANTIC_SUPPORT_THRESHOLDS,
    build_frame_record,
    partition_experiment_ids,
)
from .runtime import load_frozen_runtime, require_device

EXPECTED_TRAIN_FRAMES = 16_827
SHARD_FRAMES = 256


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the train-only person consolidation cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    runtime = load_frozen_runtime(device)
    dataset = runtime.base.data.RouteBDataset(runtime.dataset_root, "train", seed=20260830, augment=False)
    if dataset.split != "train" or len(dataset) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("consolidation cache requires exactly 16,827 training frames")
    fit_ids, holdout_ids = partition_experiment_ids([row["experiment_id"] for row in dataset.rows])
    fit_set, holdout_set = set(fit_ids), set(holdout_ids)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "shards").mkdir()
    frames: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    counts = {
        "fit": {"frames": 0, "person_candidates": 0, "eligible_person_gt": 0},
        "holdout": {"frames": 0, "person_candidates": 0, "eligible_person_gt": 0},
    }

    def flush() -> None:
        if not frames:
            return
        relative = Path("shards") / f"shard_{len(shards):05d}.pt"
        torch.save({"frames": list(frames)}, output / relative)
        shards.append({
            "path": str(relative),
            "frames": len(frames),
            "person_candidates": sum(frame["scores"].numel() for frame in frames),
        })
        frames.clear()

    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            row, target = item["row"], item["target"]
            experiment_id = row["experiment_id"]
            partition = "fit" if experiment_id in fit_set else "holdout" if experiment_id in holdout_set else None
            if partition is None:
                raise RuntimeError(f"unregistered train episode: {experiment_id}")
            calibration = {name: target[name].to(device) for name in ("intrinsic", "extrinsic")}
            outputs = runtime.model(item["input"].unsqueeze(0).to(device), dense=False)
            detections = runtime.model.postprocess(outputs, [calibration])[0]
            gt_world_xy, gt_classes = contract_world_targets(dataset.objects.get(target["sample_id"], ()))
            gt_person_world_xy = gt_world_xy[gt_classes == 1]
            frame = build_frame_record(
                outputs=outputs,
                detections=detections,
                ignore_mask=target["ignore_mask"],
                gt_person_world_xy=gt_person_world_xy,
                sample_id=target["sample_id"],
                experiment_id=experiment_id,
            )
            frames.append(frame)
            counts[partition]["frames"] += 1
            counts[partition]["person_candidates"] += frame["scores"].numel()
            counts[partition]["eligible_person_gt"] += frame["gt_world_xy"].shape[0]
            if (index + 1) % SHARD_FRAMES == 0:
                flush()
            if (index + 1) % 500 == 0:
                print(json.dumps({"train_frames": index + 1}), flush=True)
    flush()
    if sum(value["frames"] for value in counts.values()) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("train cache pass did not visit every frame exactly once")

    manifest = {
        "schema": "splitfusion_fcos_person_instance_consolidation_cache_v1",
        "split": "train",
        "pass_count": 1,
        "frames": EXPECTED_TRAIN_FRAMES,
        "stored_candidate_class": "person_only",
        "stored_fields": [
            "original_indices", "scores", "boxes", "world_xy", "component_ids", "semantic_support",
            "ignore_flags", "gt_world_xy", "sample_id", "experiment_id", "semantic_component_count",
        ],
        "original_candidate_labels_stored": False,
        "semantic_components": {"connectivity": 8, "morphology": False, "person_channel": 2},
        "canonical_person_threshold": 0.20,
        "canonical_world_match_radius_m": 3.0,
        "grid": {
            "semantic_support_thresholds": list(SEMANTIC_SUPPORT_THRESHOLDS),
            "group_box_iou_thresholds": list(GROUP_IOU_THRESHOLDS),
        },
        "episode_split": {"fit": list(fit_ids), "holdout": list(holdout_ids)},
        "partition_counts": counts,
        "base_checkpoint": str(runtime.checkpoint_path),
        "base_checkpoint_sha256": runtime.checkpoint_sha256,
        "shards": shards,
        "validation_or_test_accessed": False,
    }
    (output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"cache": str(output), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
