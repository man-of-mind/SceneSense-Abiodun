#!/usr/bin/env python3
"""Exactly the prescribed launch checks for the final Faster R-CNN pilot."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, HERE.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes

from dataset_v1 import RouteBFasterRCNNDataset, detection_collate
from model_v1 import boundary_manifest, build_model, freeze_batch_norm
from split_runtime_adapter_v1 import (
    flatten_complete_bundle,
    restore_complete_bundle,
    runtime_reuse_manifest,
)
from train_v1 import move_targets, sha256, write_json_create


def gradient_report(model: torch.nn.Module, prefixes) -> dict:
    values = []
    finite = True
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes) or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all())
        values.append(float(grad.abs().sum().item()))
    return {"parameters_with_grad": len(values), "finite": finite, "absolute_sum": sum(values), "nonzero": sum(values) > 0.0}


def max_output_difference(left: dict, right: dict) -> dict:
    result = {"segmentation": float((left["segmentation"] - right["segmentation"]).abs().max().item())}
    result["detection_count_equal"] = len(left["detections"]) == len(right["detections"])
    maximum = 0.0
    shapes_equal = True
    for left_item, right_item in zip(left["detections"], right["detections"]):
        if set(left_item) != set(right_item):
            shapes_equal = False
            continue
        for key in left_item:
            if left_item[key].shape != right_item[key].shape:
                shapes_equal = False
                continue
            if left_item[key].numel():
                maximum = max(maximum, float((left_item[key].float() - right_item[key].float()).abs().max().item()))
    result["detection_shapes_equal"] = shapes_equal
    result["detection_max_abs"] = maximum
    result["tolerance"] = 1e-6
    result["pass"] = shapes_equal and bool(result["detection_count_equal"]) and maximum <= 1e-6 and result["segmentation"] <= 1e-6
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = args.dataset_source.resolve(strict=True)
    dataset_link = experiment_dir / "dataset"
    if not dataset_link.exists():
        dataset_link.symlink_to(dataset_source, target_is_directory=True)
    elif dataset_link.resolve() != dataset_source:
        raise SystemExit(f"existing dataset link targets {dataset_link.resolve()}, expected {dataset_source}")
    registered_config = experiment_dir / "registered_config.json"
    shutil.copy2(args.config, registered_config)
    if (experiment_dir / "preflight.json").exists():
        raise SystemExit("refusing to overwrite completed preflight")
    start = time.monotonic()

    rows = read_manifest(dataset_source / "manifest.csv")
    splits = {name: [row for row in rows if row.get("split") == name] for name in ("train", "val", "test")}
    split_counts = {name: len(values) for name, values in splits.items()}
    test_path_refs = [row["sample_id"] for row in rows if "_test_" in " ".join(str(value) for value in row.values())]
    dataset_check = {
        "counts": split_counts,
        "expected": {"train": 6600, "val": 3588, "test": 0},
        "test_path_references": len(test_path_refs),
        "pass": split_counts == {"train": 6600, "val": 3588, "test": 0} and not test_path_refs,
    }
    if not dataset_check["pass"]:
        raise SystemExit(f"dataset check failed: {dataset_check}")

    actual_weight_sha = sha256(args.weights)
    weight_check = {
        "source": config["weights"]["url"],
        "path": str(args.weights.resolve()),
        "expected_sha256": config["weights"]["sha256"],
        "actual_sha256": actual_weight_sha,
        "enum": config["weights"]["enum"],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "license": "BSD-3-Clause",
        "license_file": str((Path(__file__).resolve().parent / "licenses" / "torchvision-BSD-3-Clause.txt")),
        "pass": actual_weight_sha == config["weights"]["sha256"],
    }
    if not weight_check["pass"]:
        raise SystemExit(f"weight SHA mismatch: {weight_check}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda")
    set_reproducible_seeds(int(config["training_seed"]))
    object_rows = load_object_boxes(dataset_source / "object_boxes.csv")
    width, height = map(int, config["input_size"])
    dataset = RouteBFasterRCNNDataset(
        dataset_source, splits["train"], object_rows, (width, height), training=True,
        flip_probability=float(config["flip_probability"]),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate)
    rgb, radar, targets, metadata = next(iter(loader))
    alignment = {
        "rgb_channels": [int(value.shape[0]) for value in rgb],
        "radar_channels": [int(value.shape[0]) for value in radar],
        "frame_ids_equal": [value["frame_id"] == value["radar_frame_id"] for value in metadata],
        "timestamp_abs_error_s": [abs(value["timestamp"] - value["radar_timestamp"]) for value in metadata],
    }
    alignment["pass"] = alignment["rgb_channels"] == [3, 3] and alignment["radar_channels"] == [4, 4] and all(alignment["frame_ids_equal"]) and max(alignment["timestamp_abs_error_s"]) <= 1e-6
    if not alignment["pass"]:
        raise SystemExit(f"modality alignment failed: {alignment}")

    model = build_model(pretrained=True, input_size=(width, height)).to(device)
    model.train()
    freeze_batch_norm(model.detector)
    rgb_device = [value.to(device) for value in rgb]
    radar_device = [value.to(device) for value in radar]
    targets_device = move_targets(targets, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(rgb_device, radar_device, targets_device)
        loss = sum(output["losses"].values())
    loss.backward()
    forward_backward = {
        "loss": float(loss.detach().item()),
        "amp_dtype": "bfloat16",
        "losses": {key: float(value.detach().item()) for key, value in output["losses"].items()},
        "finite_loss": bool(torch.isfinite(loss)),
        "gradients": {
            "rgb_detector": gradient_report(model, ("detector.rpn.", "detector.roi_heads.box_head.", "detector.roi_heads.box_predictor.")),
            "radar_encoder": gradient_report(model, ("radar_encoder.",)),
            "roi_localization": gradient_report(model, ("radar_roi_embed.", "roi_localization_head.")),
            "segmentation_decoder": gradient_report(model, ("segmentation_decoder.",)),
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }
    forward_backward["pass"] = forward_backward["finite_loss"] and all(
        value["finite"] and value["nonzero"] for value in forward_backward["gradients"].values()
    )
    if not forward_backward["pass"]:
        raise SystemExit(f"forward/backward failed: {forward_backward}")

    model.eval()
    one_rgb, one_radar = rgb_device[:1], radar_device[:1]
    with torch.inference_mode():
        monolithic = model(one_rgb, one_radar)
        bundle = model.encode_front(one_rgb, one_radar)
        split = model.decode_tail(bundle, bundle["original_image_sizes"])
    parity = max_output_difference(monolithic, split)
    manifest = boundary_manifest(bundle)
    flattened = flatten_complete_bundle(bundle)
    restored = restore_complete_bundle(
        flattened,
        {
            "image_batch_shape": bundle["image_batch_shape"],
            "image_sizes": bundle["image_sizes"],
            "original_image_sizes": bundle["original_image_sizes"],
        },
    )
    reuse_check = {
        "manifest": runtime_reuse_manifest(),
        "flattened_names": list(flattened.keys()),
        "complete_level_count": len(flattened),
        "restore_shapes_equal": all(
            restored[group][level].shape == tensor.shape
            for group in ("rgb_fpn", "radar_fpn")
            for level, tensor in bundle[group].items()
        ),
    }
    reuse_check["pass"] = reuse_check["complete_level_count"] == 10 and reuse_check["restore_shapes_equal"]
    tail_signature = str(inspect.signature(model.decode_tail))
    split_check = {
        "parity": parity,
        "boundary": manifest,
        "decode_tail_signature": tail_signature,
        "tail_raw_argument_absent": "rgb" not in tail_signature and "radar" not in tail_signature,
    }
    split_check["pass"] = parity["pass"] and split_check["tail_raw_argument_absent"] and not manifest["raw_rgb_present"] and not manifest["raw_radar_present"] and reuse_check["pass"]
    if not split_check["pass"]:
        raise SystemExit(f"split check failed: {split_check}")

    report = {
        "status": "PASS",
        "runtime_seconds": time.monotonic() - start,
        "python": sys.version,
        "device": torch.cuda.get_device_name(device),
        "dataset": dataset_check,
        "weights": weight_check,
        "alignment": alignment,
        "forward_backward_amp": forward_backward,
        "split_inference": split_check,
        "runtime_reuse": reuse_check,
        "coco_mapping": model.coco_mapping,
    }
    write_json_create(experiment_dir / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
