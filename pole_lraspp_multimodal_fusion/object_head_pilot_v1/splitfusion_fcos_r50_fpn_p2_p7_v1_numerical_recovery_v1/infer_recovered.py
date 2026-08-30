from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .audit import audit_tree, require_finite_audit
from .base_runtime import load_base
from .contracts import (atomic_json, atomic_text, canonical_hash, current_commit, load_json, package_hashes,
                        require_qualified, resolve_repo_path, sha256, verify_original_provenance)
from .recovery_model import build_recovery_model
from .state_guard import model_hash


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualified recovered-checkpoint validation inference")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--qualification-dir", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int, choices=(16, 22, 26))
    parser.add_argument("--execute-validation-inference", required=True, choices=("TRAINING_COMPLETE_AND_AUTHORIZED",))
    args = parser.parse_args()
    qualified, _qualification = require_qualified(args.qualification_dir, args.authorization)
    verify_original_provenance(checkpoint_metadata=False)
    experiment = args.experiment.resolve(strict=True)
    if not (experiment / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("validation prediction access forbidden before recovered epoch26 completes")
    provenance = load_json(experiment / "RECOVERY_PROVENANCE.json")
    if provenance.get("source_commit") != current_commit() or provenance.get("source_files_sha256") != canonical_hash(package_hashes()):
        raise RuntimeError("recovered experiment source provenance drift")
    base = load_base(); immutable = load_json(Path(__file__).with_name("recovery_config.json"))
    original_experiment = resolve_repo_path(immutable["original"]["experiment"])
    original_config = load_json(resolve_repo_path(immutable["original"]["config"]))
    priors = load_json(original_experiment / "TRAIN_ONLY_PRIORS.json")
    checkpoint_path = experiment / f"checkpoints/epoch_{args.epoch:03d}.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1" or int(checkpoint["epoch"]) != args.epoch:
        raise RuntimeError("recovered checkpoint schema/epoch drift")
    recovery = checkpoint["recovery"]
    if (recovery.get("source_commit") != current_commit()
            or recovery.get("source_files_sha256") != canonical_hash(package_hashes())
            or recovery.get("qualified_config_sha256") != canonical_hash(qualified)
            or recovery.get("ceilings") != qualified["ceilings"]
            or recovery.get("original_checkpoint_sha256") != qualified["original_checkpoint_sha256"]
            or float(recovery.get("selected_tau")) != float(qualified["selected_tau"])):
        raise RuntimeError("recovered checkpoint qualification binding drift")
    if not torch.cuda.is_available():
        raise RuntimeError("registered inference requires CUDA")
    device = torch.device("cuda:0"); model, _ = build_recovery_model(priors, float(qualified["selected_tau"]), device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    dataset_root = (base.common.ROOT / original_config["dataset_root"]).resolve(strict=True)
    dataset = base.data.InferenceDataset(dataset_root, "val")
    output = experiment / f"predictions/epoch_{args.epoch:03d}"
    output.mkdir(parents=True, exist_ok=False); segmentation_dir = output / "segmentation"; segmentation_dir.mkdir()
    detections_path = output / "detections.csv"; segmentation_manifest = output / "segmentation_manifest.csv"
    rows, count = [], 0; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base.infer.FIELDS); writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                fused, row, calibration = dataset[index]
                fused = fused.unsqueeze(0).to(device, non_blocking=True)
                require_finite_audit(audit_tree(fused, "input"))
                calibration_gpu = {name: value.to(device) for name, value in calibration.items()}
                outputs = model(fused, dense=False); require_finite_audit(audit_tree(outputs, "outputs"))
                detections = model.postprocess(outputs, [calibration_gpu])[0]
                require_finite_audit(audit_tree(detections, "decoded_and_postprocessed"))
                for prediction_index in range(len(detections["scores"])):
                    writer.writerow(base.infer.record(detections, row, prediction_index))
                count += len(detections["scores"])
                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                labels = F.interpolate(outputs["semantic_logits"].float(), size=source_hw,
                                       mode="bilinear", align_corners=False).argmax(1)[0]
                labels_np = labels.cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"; path = output / relative
                if not cv2.imwrite(str(path), labels_np):
                    raise RuntimeError(f"failed segmentation write {path}")
                rows.append({"sample_id": row["sample_id"], "prediction_path": str(relative),
                             "width": labels_np.shape[1], "height": labels_np.shape[0], "sha256": sha256(path)})
    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader(); writer.writerows(rows)
    manifest = {"schema": "splitfusion_fcos_recovered_inference_manifest_v1", "epoch": args.epoch,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash, "checkpoint_model_sha256": model_hash(model),
        "validation_frames": len(dataset), "inference_pass_count": 1, "score_floor": 0.02,
        "derived_threshold": 0.20, "topk_per_level": 1000, "nms_iou": 0.60, "detections_per_image": 100,
        "world_nms": False, "candidate_identity": ["image", "level", "flattened_point", "internal_class"],
        "native_object_grid": [108, 192], "fpn_levels": list(base.model.LEVELS),
        "detection_predictions": count, "detections_sha256": sha256(detections_path),
        "segmentation_manifest_sha256": sha256(segmentation_manifest),
        "prediction_set_sha256": hashlib.sha256((sha256(detections_path) + sha256(segmentation_manifest)).encode()).hexdigest(),
        "depth_paths_opened": 0, "depth_labels_used": False, "semantic_gt_paths_opened": 0,
        "raw_transport_bytes": 22020096, "transport_dtype": "float32", "transport_codec": "identity",
        "wall_seconds": time.monotonic() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "external_detection_fields": list(base.infer.FIELDS),
        "tensor_finiteness_audited_every_frame": True, "zero_detection_frames_supported": True}
    atomic_json(output / "inference_manifest.json", manifest)
    atomic_text(output / "INFERENCE_COMPLETE", "ONE_SCORE_FLOOR_0_02_PASS_COMPLETE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
