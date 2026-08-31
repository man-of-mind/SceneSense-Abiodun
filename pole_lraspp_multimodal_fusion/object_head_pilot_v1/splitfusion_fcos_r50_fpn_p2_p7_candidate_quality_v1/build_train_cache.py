from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .labeling import contract_world_targets, label_candidates
from .quality import EXTRA_FEATURE_NAMES, FEATURE_DIM, extract_candidate_features
from .runtime import load_frozen_runtime, require_device

SHARD_FRAMES = 256


def _cache_float16(features: torch.Tensor) -> torch.Tensor:
    value = features.detach().float().cpu()
    if not bool(torch.isfinite(value).all()) or (value.numel() and float(value.abs().amax()) > torch.finfo(torch.float16).max):
        raise FloatingPointError("candidate features are not safe for float16 cache storage")
    return value.to(torch.float16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the train-only frozen candidate cache")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    runtime = load_frozen_runtime(device)
    dataset = runtime.base.data.RouteBDataset(runtime.dataset_root, "train", seed=20260829, augment=False)
    if dataset.split != "train":
        raise RuntimeError("candidate cache builder is train-only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shard_dir = output / "shards"
    shard_dir.mkdir()

    buffers: dict[str, list[Any]] = {name: [] for name in (
        "features", "classes", "base_scores", "labels", "candidate_identities", "sample_ids",
    )}
    shards: list[dict[str, Any]] = []
    label_counts: Counter[int] = Counter()
    frame_count = 0

    def flush() -> None:
        if not buffers["features"]:
            return
        shard_index = len(shards)
        relative = Path("shards") / f"shard_{shard_index:05d}.pt"
        payload = {
            "features": torch.cat(buffers["features"]),
            "classes": torch.cat(buffers["classes"]),
            "base_scores": torch.cat(buffers["base_scores"]),
            "labels": torch.cat(buffers["labels"]),
            "candidate_identities": torch.cat(buffers["candidate_identities"]),
            "sample_ids": [sample_id for values in buffers["sample_ids"] for sample_id in values],
        }
        torch.save(payload, output / relative)
        shards.append({"path": str(relative), "candidates": int(payload["labels"].numel())})
        for values in buffers.values():
            values.clear()

    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            fused = item["input"].unsqueeze(0).to(device)
            target = item["target"]
            calibration = {name: target[name].to(device) for name in ("intrinsic", "extrinsic")}
            outputs = runtime.model(fused, dense=False)
            detections = runtime.model.postprocess(outputs, [calibration])[0]
            features = extract_candidate_features(outputs, detections)
            gt_world_xy, gt_classes = contract_world_targets(dataset.objects.get(target["sample_id"], ()))
            if not torch.equal(gt_classes, target["labels"].cpu()):
                raise RuntimeError(f"v0.10 eligible GT order drift for {target['sample_id']}")
            labels, summary = label_candidates(
                candidate_world_xy=detections["world_xyz"][:, :2],
                candidate_classes=detections["labels_internal"],
                candidate_boxes=detections["boxes"],
                gt_world_xy=gt_world_xy,
                gt_classes=gt_classes,
                ignore_mask=target["ignore_mask"],
            )
            if not summary["tp_plus_fn_reconciles"] or features.shape[0] != labels.numel():
                raise RuntimeError(f"candidate label reconciliation failed for {target['sample_id']}")
            count = labels.numel()
            buffers["features"].append(_cache_float16(features))
            buffers["classes"].append(detections["labels_internal"].detach().to(torch.int8).cpu())
            buffers["base_scores"].append(detections["scores"].detach().float().cpu())
            buffers["labels"].append(labels)
            buffers["candidate_identities"].append(detections["candidate_identity"].detach().to(torch.int32).cpu())
            buffers["sample_ids"].append([target["sample_id"]] * count)
            label_counts.update(int(value) for value in labels.tolist())
            frame_count += 1
            if frame_count % SHARD_FRAMES == 0:
                flush()
            if frame_count % 500 == 0:
                print(json.dumps({"train_frames": frame_count, "candidates": sum(label_counts.values())}), flush=True)
    flush()
    if frame_count != len(dataset):
        raise RuntimeError("train-only cache pass did not visit every frame exactly once")
    manifest = {
        "schema": "splitfusion_fcos_candidate_cache_v1",
        "split": "train",
        "pass_count": 1,
        "frames": frame_count,
        "candidates": sum(label_counts.values()),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float16",
        "feature_order": ["frozen_fpn_256", *EXTRA_FEATURE_NAMES],
        "stored_fields": ["features", "classes", "base_scores", "labels", "candidate_identities", "sample_ids"],
        "labels": {"-1": "ignored", "0": "negative", "1": "positive"},
        "label_counts": {str(key): label_counts[key] for key in (-1, 0, 1)},
        "candidate_identity": ["image", "fpn_level", "flattened_point", "class"],
        "base_checkpoint": str(runtime.checkpoint_path),
        "base_checkpoint_sha256": runtime.checkpoint_sha256,
        "shards": shards,
        "validation_or_test_accessed": False,
    }
    (output / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cache": str(output), "frames": frame_count, "candidates": manifest["candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
