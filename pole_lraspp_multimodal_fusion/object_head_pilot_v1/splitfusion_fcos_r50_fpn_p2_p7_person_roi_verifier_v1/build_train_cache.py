from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.labeling import (
    contract_world_targets,
    label_candidates,
)

from .runtime import load_frozen_runtime, require_device
from .verifier import (
    FEATURE_DIM,
    ROI_DESCRIPTOR_DIM,
    SCALAR_FEATURE_NAMES,
    PersonRoIDescriptor,
    fp16_round_trip_roi_descriptors,
    partition_experiment_ids,
)

EXPECTED_TRAIN_FRAMES = 16_827
SHARD_FRAMES = 256


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one train-only frozen person ROI cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    runtime = load_frozen_runtime(device)
    dataset = runtime.base.data.RouteBDataset(runtime.dataset_root, "train", seed=20260830, augment=False)
    if dataset.split != "train" or len(dataset) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("person ROI cache requires exactly 16,827 training frames")
    fit_ids, holdout_ids = partition_experiment_ids([row["experiment_id"] for row in dataset.rows])
    fit_set, holdout_set = set(fit_ids), set(holdout_ids)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shard_dir = output / "shards"
    shard_dir.mkdir()
    extractor = PersonRoIDescriptor().to(device).eval()
    buffers: dict[str, list[Any]] = {name: [] for name in (
        "roi_descriptors", "scalar_features", "base_scores", "labels", "candidate_identities",
        "sample_ids", "experiment_ids", "partitions",
    )}
    shards: list[dict[str, Any]] = []
    counts = {
        "fit": {"frames": 0, "candidates": 0, "eligible_person_gt": 0, "labels": Counter()},
        "holdout": {"frames": 0, "candidates": 0, "eligible_person_gt": 0, "labels": Counter()},
    }
    frame_count = 0

    def flush() -> None:
        if not buffers["roi_descriptors"]:
            return
        relative = Path("shards") / f"shard_{len(shards):05d}.pt"
        payload = {
            "roi_descriptors": torch.cat(buffers["roi_descriptors"]),
            "scalar_features": torch.cat(buffers["scalar_features"]),
            "base_scores": torch.cat(buffers["base_scores"]),
            "labels": torch.cat(buffers["labels"]),
            "candidate_identities": torch.cat(buffers["candidate_identities"]),
            "sample_ids": [value for values in buffers["sample_ids"] for value in values],
            "experiment_ids": [value for values in buffers["experiment_ids"] for value in values],
            "partitions": torch.cat(buffers["partitions"]),
        }
        torch.save(payload, output / relative)
        shards.append({"path": str(relative), "person_candidates": int(payload["labels"].numel())})
        for values in buffers.values():
            values.clear()

    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            row, target = item["row"], item["target"]
            experiment_id = row["experiment_id"]
            partition = "fit" if experiment_id in fit_set else "holdout" if experiment_id in holdout_set else None
            if partition is None:
                raise RuntimeError(f"unregistered training episode: {experiment_id}")
            fused = item["input"].unsqueeze(0).to(device)
            calibration = {name: target[name].to(device) for name in ("intrinsic", "extrinsic")}
            outputs = runtime.model(fused, dense=False)
            detections = runtime.model.postprocess(outputs, [calibration])[0]
            roi_descriptors, scalar_features, person_indices = extractor(outputs, detections)

            gt_world_xy, gt_classes = contract_world_targets(dataset.objects.get(target["sample_id"], ()))
            if not torch.equal(gt_classes, target["labels"].cpu()):
                raise RuntimeError(f"v0.10 eligible GT order drift for {target['sample_id']}")
            all_labels, summary = label_candidates(
                candidate_world_xy=detections["world_xyz"][:, :2],
                candidate_classes=detections["labels_internal"],
                candidate_boxes=detections["boxes"],
                gt_world_xy=gt_world_xy,
                gt_classes=gt_classes,
                ignore_mask=target["ignore_mask"],
            )
            labels = all_labels.index_select(0, person_indices.cpu())
            person_count = person_indices.numel()
            if (not summary["tp_plus_fn_reconciles"] or roi_descriptors.shape[0] != person_count
                    or scalar_features.shape != (person_count, len(SCALAR_FEATURE_NAMES))):
                raise RuntimeError(f"person candidate reconciliation failed for {target['sample_id']}")

            rounded_roi = fp16_round_trip_roi_descriptors(roi_descriptors.detach())
            buffers["roi_descriptors"].append(rounded_roi.to(torch.float16).cpu())
            buffers["scalar_features"].append(scalar_features.detach().float().cpu())
            buffers["base_scores"].append(detections["scores"].index_select(0, person_indices).detach().float().cpu())
            buffers["labels"].append(labels)
            buffers["candidate_identities"].append(
                detections["candidate_identity"].index_select(0, person_indices).detach().to(torch.int32).cpu(),
            )
            buffers["sample_ids"].append([target["sample_id"]] * person_count)
            buffers["experiment_ids"].append([experiment_id] * person_count)
            partition_value = 0 if partition == "fit" else 1
            buffers["partitions"].append(torch.full((person_count,), partition_value, dtype=torch.int8))
            counts[partition]["frames"] += 1
            counts[partition]["candidates"] += person_count
            counts[partition]["eligible_person_gt"] += int((gt_classes == 1).sum())
            counts[partition]["labels"].update(int(value) for value in labels.tolist())
            frame_count += 1
            if frame_count % SHARD_FRAMES == 0:
                flush()
            if frame_count % 500 == 0:
                total_candidates = sum(counts[name]["candidates"] for name in counts)
                print(json.dumps({"train_frames": frame_count, "person_candidates": total_candidates}), flush=True)
    flush()
    if frame_count != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("train-only cache pass did not visit every frame exactly once")

    partition_counts = {
        name: {
            "episodes": len(fit_ids if name == "fit" else holdout_ids),
            "frames": value["frames"],
            "person_candidates": value["candidates"],
            "eligible_person_gt": value["eligible_person_gt"],
            "label_counts": {str(label): value["labels"][label] for label in (-1, 0, 1)},
        }
        for name, value in counts.items()
    }
    manifest = {
        "schema": "splitfusion_fcos_person_roi_cache_v1",
        "split": "train",
        "pass_count": 1,
        "frames": frame_count,
        "person_candidates": sum(value["candidates"] for value in counts.values()),
        "roi_descriptor_dim": ROI_DESCRIPTOR_DIM,
        "roi_descriptor_dtype": "float16",
        "raw_roi_cached": False,
        "scalar_feature_dim": len(SCALAR_FEATURE_NAMES),
        "feature_dim": FEATURE_DIM,
        "scalar_feature_order": list(SCALAR_FEATURE_NAMES),
        "stored_fields": list(buffers),
        "labels": {"-1": "ignored", "0": "negative", "1": "positive"},
        "episode_split": {"fit": list(fit_ids), "holdout": list(holdout_ids)},
        "partition_counts": partition_counts,
        "partition_encoding": {"0": "fit", "1": "holdout"},
        "candidate_identity": ["image", "fpn_level", "flattened_point", "class"],
        "base_checkpoint": str(runtime.checkpoint_path),
        "base_checkpoint_sha256": runtime.checkpoint_sha256,
        "shards": shards,
        "validation_or_test_accessed": False,
    }
    (output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"cache": str(output), "frames": frame_count, "partitions": partition_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
