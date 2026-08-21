#!/usr/bin/env python3
"""Regenerate retained prediction lists offline on the frozen val or test split.

This module does not instantiate a CARLA client, open sockets, invoke OAI, or
train. It writes only under its own experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
AB = HERE.parents[3]
DATASET = AB / "fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
for package_path in (AB / "pole_lraspp_multimodal_fusion", AB, AB / "rl_agent/feature_ae"):
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    CLASS_NAMES,
    load_config,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor, load_mask  # noqa: E402
from pole_lraspp_multimodal_fusion.model import (  # noqa: E402
    OBJECT_HEAD_CHANNELS,
    build_multitask_fusion_lraspp,
)
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    decode_objects,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)
from pole_lraspp_multimodal_fusion.split_runtime import (  # noqa: E402
    MultimodalLRASPPSplitModel,
    deserialize_backbone_features,
    serialize_backbone_features,
)


CONFIG_PATH = AB / "pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
CHECKPOINTS = {
    "noae": AB / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt",
    "ae32": AB / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
    "ae64": AB / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
    "ae128": AB / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
}
EVAL = {
    "score_threshold": 0.20,
    "image_nms_radius_px": 2,
    "topk": 120,
    "max_distance_m": 40.0,
    "min_gt_area_px": 12.0,
}
ENTROPY_CODER = "zstd"
ZSTD_LEVEL = 3
SAMPLE_RE = re.compile(r"_(\d{6})_frame(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_freeze() -> dict[str, Any]:
    freeze = json.loads((HERE / "PREINFERENCE_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise AssertionError("Pre-inference freeze is absent or invalid")
    for record in freeze["frozen_files"]:
        path = HERE / record["path"]
        if sha256(path) != record["sha256"]:
            raise AssertionError(f"Frozen file changed after preregistration: {path}")
    return freeze


def build_model(ckpt_path: Path, device: torch.device, config: dict[str, Any]):
    train_cfg, fusion_cfg = config["training"], config.get("fusion", {})
    object_cfg = config.get("object_heads", {})
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt = checkpoint if isinstance(checkpoint, dict) else {}
    object_class_names = tuple(ckpt.get("object_class_names") or object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
    input_size = tuple(int(value) for value in (ckpt.get("input_size") or train_cfg.get("input_size", [512, 288])))
    model = build_multitask_fusion_lraspp(
        num_classes=int(train_cfg.get("num_classes", 3)),
        radar_channels=int(ckpt.get("radar_channels") or fusion_cfg.get("radar_channels", 4)),
        pretrained=False,
        object_channels=int(ckpt.get("object_channels") or object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(ckpt.get("fuse_low_into_object_head")) or bool(object_cfg.get("fuse_low_feature", False)),
        head_arch=str(ckpt.get("object_head_arch") or object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(ckpt.get("object_use_coordconv")) or bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(ckpt.get("object_head_depth") or object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(ckpt.get("object_use_groundplane_prior")) or bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(ckpt.get("object_groundplane_params") or object_cfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    ae_bottleneck = int((ckpt.get("trial") or {}).get("ae_bottleneck", 0))
    if ae_bottleneck > 0:
        from ae_model import build_ae
        high_channels = int(model.classifier.cbr[0].in_channels)
        model.feature_ae = build_ae(
            str((ckpt.get("trial") or {}).get("ae_arch", "v2")),
            high_channels,
            ae_bottleneck,
        ).to(device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()
    predict_bbox2d = bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False))
    return model, input_size, object_class_names, ae_bottleneck, predict_bbox2d


def roi_gate(features: OrderedDict, keep_order: dict[tuple[int, int], torch.Tensor], fraction: float) -> OrderedDict:
    if fraction <= 0.0:
        return features
    gated = OrderedDict()
    for name, feature in features.items():
        flat = feature.flatten(start_dim=2).clone()
        count = int(round(float(fraction) * flat.shape[-1]))
        if count > 0:
            flat[:, :, keep_order[feature.shape[-2:]][:count]] = 0
        gated[name] = flat.reshape_as(feature)
    return gated


def payload_size(serialized: dict[str, Any]) -> int:
    import zstandard as zstd
    blobs: list[bytes] = []
    for entry in serialized.values():
        values = entry.values() if isinstance(entry, dict) else [entry]
        blobs.extend(bytes(value) for value in values if isinstance(value, (bytes, bytearray)))
    return len(zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(b"".join(blobs)))


def json_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        return float("nan")
    return number


def compact_prediction(prediction: dict[str, Any], source_order: int) -> dict[str, Any]:
    return {
        "source_order": source_order,
        "class_name": str(prediction.get("class_name", "")),
        "score": json_float(prediction["score"]),
        "world_x": json_float(prediction["world_x"]),
        "world_y": json_float(prediction["world_y"]),
        "center_x_px": json_float(prediction.get("center_x_px", float("nan"))),
        "center_y_px": json_float(prediction.get("center_y_px", float("nan"))),
    }


def compact_gt(gt: dict[str, Any], source_order: int) -> dict[str, Any]:
    return {
        "source_order": source_order,
        "class_name": str(gt.get("class_name", "")),
        "world_x": json_float(gt["world_x"]),
        "world_y": json_float(gt["world_y"]),
        "bbox_area_px": json_float(gt.get("area", float("nan"))),
    }


def parse_sample(sample_id: str) -> tuple[int, int]:
    match = SAMPLE_RE.search(sample_id)
    if not match:
        raise ValueError(f"Unexpected sample identifier: {sample_id}")
    return int(match.group(1)), int(match.group(2))


def load_rows(split: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    all_rows = read_manifest(DATASET / "manifest.csv")
    maxima: dict[str, int] = defaultdict(int)
    for row in all_rows:
        index, _ = parse_sample(str(row["sample_id"]))
        maxima[str(row["experiment_id"])] = max(maxima[str(row["experiment_id"])], index)
    rows = [row for row in all_rows if row.get("split") == split]
    expected_ids = (HERE / "split_identifiers" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    actual_ids = [str(row["sample_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise AssertionError(f"{split} manifest identifiers changed after freeze")
    return rows, dict(maxima)


def trajectory_group(row: dict[str, str], maxima: dict[str, int]) -> str:
    experiment = str(row["experiment_id"])
    index, _ = parse_sample(str(row["sample_id"]))
    loop = min(7, int(math.floor(index * 8 / max(1, maxima[experiment] + 1))))
    return f"{experiment}::loop{loop}"


def profile_output_path(split: str, profile_id: str) -> Path:
    return HERE / "raw_predictions" / split / f"{profile_id}.jsonl.gz"


def write_profile(
    *,
    split_name: str,
    profile: dict[str, Any],
    rows: list[dict[str, str]],
    maxima: dict[str, int],
    boxes: dict[str, list[dict[str, str]]],
    model: torch.nn.Module,
    split_model: MultimodalLRASPPSplitModel,
    input_size: tuple[int, int],
    object_class_names: tuple[str, ...],
    ae_bottleneck: int,
    predict_bbox2d: bool,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    target = profile_output_path(split_name, str(profile["profile_id"]))
    if target.exists():
        return {"profile_id": profile["profile_id"], "status": "existing_complete", "path": str(target.relative_to(HERE)), "sha256": sha256(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    import carla_split_inference_udp_data_collect as transport_module

    transport = transport_module.TransportConfig(
        quantization_mode=str(profile["quantization"]),
        entropy_coder_name=ENTROPY_CODER,
        zstd_level=ZSTD_LEVEL,
        roi_objectness_threshold=0.0,
        bypass_rcnn_transform=False,
    )
    codecs: dict[str, Any] = {}
    out_hw = (int(input_size[1]), int(input_size[0]))
    num_seg_classes = int(config["training"].get("num_classes", 3))
    n_heat = int(getattr(model, "heatmap_channels", len(object_class_names)))
    ae = getattr(model, "feature_ae", None)
    started = time.time()
    latency_values: list[float] = []
    with gzip.open(partial, "wt", encoding="utf-8", newline="") as handle, torch.inference_mode():
        for row_index, row in enumerate(rows):
            fused, output_hw, original_size = load_fused_tensor(row, DATASET, input_size, device)
            matrix = parse_matrix(row.get("camera_matrix_json", ""))
            if matrix is None:
                raise AssertionError(f"Missing camera matrix: {row['sample_id']}")
            gt_objects = valid_localization_objects(
                boxes.get(row["sample_id"], []),
                image_width=int(original_size[0]),
                image_height=int(original_size[1]),
                min_area_px=EVAL["min_gt_area_px"],
                object_class_names=object_class_names,
                max_distance_m=EVAL["max_distance_m"],
            )
            features = split_model.encode(fused)
            object_maps = split_model.decode_object_maps(features, out_hw)
            objectness = torch.sigmoid(object_maps[:, :n_heat]).amax(dim=1, keepdim=True)
            keep_order: dict[tuple[int, int], torch.Tensor] = {}
            for feature in features.values():
                shape = feature.shape[-2:]
                if shape not in keep_order:
                    pooled = F.adaptive_max_pool2d(objectness, shape).reshape(-1).float()
                    keep_order[shape] = pooled.argsort()
            gated = roi_gate(features, keep_order, float(profile["roi_drop_fraction"]))
            if ae is not None:
                gated = OrderedDict((name, ae.encode(value) if name == "high" else value) for name, value in gated.items())
            serialized, _ = serialize_backbone_features(gated, transport, codecs)
            nbytes = payload_size(serialized)
            reconstructed = deserialize_backbone_features(serialized, device=device, transport=transport, feature_codecs=codecs)
            if ae is not None:
                reconstructed = OrderedDict((name, ae.decode(value) if name == "high" else value) for name, value in reconstructed.items())

            torch.cuda.synchronize(device)
            decoder_start = time.perf_counter_ns()
            outputs = split_model.decode_outputs(reconstructed, out_hw)
            predictions = decode_objects(
                outputs["object"],
                camera_matrix=matrix,
                topk=EVAL["topk"],
                score_threshold=EVAL["score_threshold"],
                nms_radius_px=EVAL["image_nms_radius_px"],
                object_class_names=object_class_names,
                predict_bbox2d=predict_bbox2d,
            )
            camera_center = np.asarray(matrix)[:3, 3]
            predictions = [
                prediction for prediction in predictions
                if math.hypot(float(prediction["world_x"]) - camera_center[0], float(prediction["world_y"]) - camera_center[1])
                <= EVAL["max_distance_m"]
            ]
            torch.cuda.synchronize(device)
            decoder_latency_ms = (time.perf_counter_ns() - decoder_start) / 1e6
            latency_values.append(decoder_latency_ms)

            segmentation = F.interpolate(outputs["out"], size=output_hw, mode="bilinear", align_corners=False)
            segmentation_prediction = segmentation.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
            segmentation_gt = load_mask(DATASET / row["mask_path"])
            confusion = np.zeros((num_seg_classes, num_seg_classes), dtype=np.int64)
            update_confusion(confusion, segmentation_prediction, segmentation_gt, num_seg_classes)
            collection_index, frame_from_id = parse_sample(str(row["sample_id"]))
            record = {
                "split": split_name,
                "profile_id": profile["profile_id"],
                "family": profile["family"],
                "quantization": profile["quantization"],
                "roi_drop_fraction": profile["roi_drop_fraction"],
                "role": profile["role"],
                "normal_gate": profile["normal_gate"],
                "sample_id": row["sample_id"],
                "frame_id": row.get("frame_id", ""),
                "frame_from_sample_id": frame_from_id,
                "collection_index": collection_index,
                "experiment_id": row.get("experiment_id", ""),
                "scenario_group": trajectory_group(row, maxima),
                "payload_bytes": nbytes,
                "decoder_latency_ms": decoder_latency_ms,
                "segmentation_confusion": confusion.reshape(-1).tolist(),
                "predictions": [compact_prediction(prediction, index) for index, prediction in enumerate(predictions)],
                "gt": [compact_gt(gt, index) for index, gt in enumerate(gt_objects)],
            }
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n")
            if (row_index + 1) % 100 == 0:
                elapsed = time.time() - started
                print(
                    f"[{split_name}/{profile['profile_id']}] {row_index + 1}/{len(rows)} "
                    f"elapsed={elapsed / 60:.1f}m eta={elapsed / (row_index + 1) * (len(rows) - row_index - 1) / 60:.1f}m",
                    flush=True,
                )
    partial.replace(target)
    values = np.asarray(latency_values[25:], dtype=float)
    return {
        "profile_id": profile["profile_id"],
        "status": "generated",
        "path": str(target.relative_to(HERE)),
        "sha256": sha256(target),
        "size_bytes": target.stat().st_size,
        "frames": len(rows),
        "checkpoint": str(CHECKPOINTS[str(profile["family"])].relative_to(AB)),
        "checkpoint_sha256": sha256(CHECKPOINTS[str(profile["family"])]),
        "ae_bottleneck": ae_bottleneck,
        "decoder_latency_warmup_excluded": min(25, len(latency_values)),
        "decoder_latency_p50_ms": float(np.quantile(values, 0.50)),
        "decoder_latency_p95_ms": float(np.quantile(values, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("val", "test"))
    args = parser.parse_args()
    freeze = verify_freeze()
    resolved = json.loads((HERE / "resolved_config.json").read_text(encoding="utf-8"))
    if args.split == "test":
        selection_path = HERE / "frozen_selection.json"
        if not selection_path.is_file():
            raise RuntimeError("Test inference is forbidden before validation selection is frozen")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selected_candidate") not in ("world_suppression_1m", "world_suppression_2m"):
            raise RuntimeError("No eligible conservative setting is frozen; test inference is forbidden")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the frozen latency/evaluation contract; CPU fallback is forbidden")
    device = torch.device("cuda")
    marker = HERE / f"{args.split.upper()}_INFERENCE_STARTED.json"
    if not marker.exists():
        atomic_json(marker, {
            "split": args.split,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_validation_identifier_hash": next(
                item["sha256"] for item in freeze["frozen_files"] if item["path"] == "split_identifiers/val.txt"
            ),
            "frozen_test_identifier_hash": next(
                item["sha256"] for item in freeze["frozen_files"] if item["path"] == "split_identifiers/test.txt"
            ),
            "resume_policy": "completed profile files are never overwritten; absent profiles may resume after interruption",
        })

    rows, maxima = load_rows(args.split)
    boxes = load_object_boxes(DATASET / "object_boxes.csv")
    config = load_config(str(CONFIG_PATH))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in resolved["profiles"]:
        by_family[str(profile["family"])].append(profile)
    profile_results: list[dict[str, Any]] = []
    started = time.time()
    for family, profiles in by_family.items():
        pending = [profile for profile in profiles if not profile_output_path(args.split, str(profile["profile_id"])).exists()]
        if not pending:
            profile_results.extend({
                "profile_id": profile["profile_id"],
                "status": "existing_complete",
                "path": str(profile_output_path(args.split, str(profile["profile_id"])).relative_to(HERE)),
                "sha256": sha256(profile_output_path(args.split, str(profile["profile_id"]))),
            } for profile in profiles)
            continue
        model, input_size, classes, ae_bottleneck, predict_bbox2d = build_model(CHECKPOINTS[family], device, config)
        split_model = MultimodalLRASPPSplitModel(model, device, input_size=input_size)
        for profile in profiles:
            profile_results.append(write_profile(
                split_name=args.split,
                profile=profile,
                rows=rows,
                maxima=maxima,
                boxes=boxes,
                model=model,
                split_model=split_model,
                input_size=input_size,
                object_class_names=classes,
                ae_bottleneck=ae_bottleneck,
                predict_bbox2d=predict_bbox2d,
                config=config,
                device=device,
            ))
        del model, split_model
        torch.cuda.empty_cache()

    complete = {
        "split": args.split,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_version": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
        "profile_results": profile_results,
    }
    atomic_json(HERE / f"{args.split.upper()}_INFERENCE_COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
