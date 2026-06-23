from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from .common import (
    CLASS_NAMES,
    class_iou_from_confusion,
    load_config,
    read_manifest,
    save_json,
    setup_logger,
    update_confusion,
    utc_iso,
)
from .model import OBJECT_HEAD_CHANNELS, build_lraspp, build_multitask_fusion_lraspp
from .object_targets import (
    OBJECT_CLASS_NAMES,
    decode_objects,
    greedy_match_predictions,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)


def find_best_checkpoint(exp_dir: Path) -> Path:
    summaries = sorted((exp_dir / "checkpoints").glob("*/trial_summary.json"))
    best_path: Optional[Path] = None
    best_score = -math.inf
    for summary_path in summaries:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        score = float(payload.get("best_selection_score", payload.get("best_miou", float("nan"))))
        checkpoint = Path(str(payload.get("best_checkpoint", ""))).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (exp_dir / checkpoint).resolve()
        if checkpoint.exists() and score > best_score:
            best_score = score
            best_path = checkpoint
    if best_path is None:
        raise FileNotFoundError(f"No usable best checkpoint found under {exp_dir / 'checkpoints'}")
    return best_path


def _rgb_normalized_tensor(image: Image.Image, input_size: Tuple[int, int]) -> torch.Tensor:
    resized = image.resize(input_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def load_fused_tensor(row: Dict[str, str], dataset_dir: Path, input_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    image = Image.open(dataset_dir / row["rgb_path"]).convert("RGB")
    original_width, original_height = image.size
    image_tensor = _rgb_normalized_tensor(image, input_size)
    radar_path = dataset_dir / row["radar_tensor_path"]
    payload = np.load(radar_path)
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            radar = payload["radar"].astype(np.float32)
        else:
            radar = np.asarray(payload, dtype=np.float32)
    finally:
        if hasattr(payload, "close"):
            payload.close()
    if radar.shape[2] != input_size[0] or radar.shape[1] != input_size[1]:
        channels = []
        for idx, channel in enumerate(radar):
            interpolation = cv2.INTER_NEAREST if idx == 0 else cv2.INTER_LINEAR
            channels.append(cv2.resize(channel, input_size, interpolation=interpolation))
        radar = np.stack(channels, axis=0).astype(np.float32)
    radar_tensor = torch.from_numpy(np.ascontiguousarray(radar))
    fused = torch.cat([image_tensor, radar_tensor], dim=0).unsqueeze(0).to(device)
    return fused, (original_height, original_width), (original_width, original_height)


def load_rgb_tensor(row: Dict[str, str], dataset_dir: Path, input_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(dataset_dir / row["rgb_path"]).convert("RGB")
    width, height = image.size
    return _rgb_normalized_tensor(image, input_size).unsqueeze(0).to(device), (height, width)


def load_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, class_names: Sequence[str], output_dir: Path, name: str) -> None:
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{name} normalized confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_dir / f"{name}_confusion_matrix")


def yaw_error_deg(pred: Dict[str, float], gt: Dict[str, float]) -> float:
    pred_angle = math.atan2(float(pred["yaw_sin"]), float(pred["yaw_cos"]))
    gt_angle = math.atan2(float(gt["yaw_sin"]), float(gt["yaw_cos"]))
    diff = math.atan2(math.sin(pred_angle - gt_angle), math.cos(pred_angle - gt_angle))
    return abs(math.degrees(diff))


def maybe_classical_radar_diagnostic(row: Dict[str, str], boxes: Sequence[Dict[str, str]], dataset_dir: Path) -> List[float]:
    radar_path = dataset_dir / row.get("radar_points_path", "")
    if not radar_path.exists():
        return []
    errors: List[float] = []
    with np.load(radar_path) as radar_points:
        world_xyz = radar_points["world_xyz"]
        u = radar_points["u"]
        v = radar_points["v"]
        valid = radar_points["valid_projection"].astype(bool)
    for box in boxes:
        if box.get("object_world_x", "") == "" or box.get("object_world_y", "") == "" or world_xyz.size == 0:
            continue
        x0 = float(box.get("gt_bbox_x", 0.0) or 0.0)
        y0 = float(box.get("gt_bbox_y", 0.0) or 0.0)
        x1 = x0 + float(box.get("gt_bbox_w", 0.0) or 0.0)
        y1 = y0 + float(box.get("gt_bbox_h", 0.0) or 0.0)
        inside = valid & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        if not np.any(inside):
            continue
        pred_xy = np.median(world_xyz[inside, :2], axis=0)
        errors.append(float(np.linalg.norm(pred_xy - np.array([float(box["object_world_x"]), float(box["object_world_y"])], dtype=np.float32))))
    return errors


def resolve_device(args: argparse.Namespace) -> torch.device:
    requested = str(getattr(args, "device", "auto")).lower()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        if getattr(args, "require_cuda", False) and not cuda_available:
            raise RuntimeError("CUDA was required for evaluation, but torch.cuda.is_available() is false.")
        return torch.device("cuda" if cuda_available else "cpu")
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("Evaluation was requested on CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        if getattr(args, "require_cuda", False):
            raise RuntimeError("--require-cuda cannot be used with --device cpu.")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    dataset_dir = exp_dir / "dataset"
    log = setup_logger(exp_dir / "supervisor.log")
    device = resolve_device(args)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log(f"Evaluating on device={device} ({device_name})")
    rows = [row for row in read_manifest(dataset_dir / "manifest.csv") if row.get("split") == args.split]
    if args.sample_id_contains:
        rows = [row for row in rows if args.sample_id_contains in row.get("sample_id", "")]
    if not rows:
        raise RuntimeError(f"No rows found for split={args.split}.")
    train_cfg = config["training"]
    fusion_cfg = config.get("fusion", {})
    object_cfg = config.get("object_heads", {})
    eval_cfg = dict(config.get("evaluation", {}))
    if args.object_score_threshold is not None:
        eval_cfg["object_score_threshold"] = float(args.object_score_threshold)
    if args.object_nms_radius_px is not None:
        eval_cfg["object_nms_radius_px"] = int(args.object_nms_radius_px)
    if args.topk_objects is not None:
        eval_cfg["topk_objects"] = int(args.topk_objects)
    if args.match_distance_m is not None:
        eval_cfg["match_distance_m"] = float(args.match_distance_m)
    num_classes = int(train_cfg.get("num_classes", 3))
    class_names = list(CLASS_NAMES[:num_classes])
    input_size = tuple(int(v) for v in args.input_size) if args.input_size else tuple(int(v) for v in train_cfg.get("input_size", [512, 288]))
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    object_class_names = tuple(
        (checkpoint.get("object_class_names") if isinstance(checkpoint, dict) else None)
        or object_cfg.get("object_classes", OBJECT_CLASS_NAMES)
    )
    if isinstance(checkpoint, dict) and "input_size" in checkpoint:
        input_size = tuple(int(v) for v in checkpoint["input_size"])
    radar_channels = int((checkpoint.get("radar_channels") if isinstance(checkpoint, dict) else None) or fusion_cfg.get("radar_channels", 4))
    object_channels = int((checkpoint.get("object_channels") if isinstance(checkpoint, dict) else None) or object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS))
    fuse_low_into_object_head = bool(
        checkpoint.get("fuse_low_into_object_head") if isinstance(checkpoint, dict) else None
    ) or bool(object_cfg.get("fuse_low_feature", False))
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        object_channels=object_channels,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=fuse_low_into_object_head,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()

    baseline_model = None
    baseline_input_size = input_size
    baseline_checkpoint_path = str(train_cfg.get("baseline_rgb_checkpoint", ""))
    if baseline_checkpoint_path:
        baseline_model = build_lraspp(num_classes, pretrained=False).to(device)
        baseline_checkpoint = torch.load(Path(baseline_checkpoint_path).expanduser(), map_location=device)
        if isinstance(baseline_checkpoint, dict) and "input_size" in baseline_checkpoint:
            baseline_input_size = tuple(int(v) for v in baseline_checkpoint["input_size"])
        baseline_state = baseline_checkpoint["model"] if isinstance(baseline_checkpoint, dict) and "model" in baseline_checkpoint else baseline_checkpoint
        baseline_model.load_state_dict(baseline_state)
        baseline_model.eval()

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    baseline_confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")
    object_metric_rows: List[Dict[str, object]] = []
    loc_errors: List[float] = []
    loc_sq_errors: List[float] = []
    dim_abs_errors: List[float] = []
    yaw_errors: List[float] = []
    parked_correct = 0
    parked_total = 0
    tp = fp = fn = 0
    per_class_stats: Dict[str, Dict[str, object]] = {
        str(name): {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []}
        for name in object_class_names
    }
    classical_errors: List[float] = []
    with torch.inference_mode():
        for row in rows:
            fused_tensor, output_hw, original_size = load_fused_tensor(row, dataset_dir, input_size, device)
            outputs = model(fused_tensor)
            logits = F.interpolate(outputs["out"], size=output_hw, mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
            gt = load_mask(dataset_dir / row["mask_path"])
            update_confusion(confusion, pred, gt, num_classes)
            if baseline_model is not None:
                rgb_tensor, baseline_hw = load_rgb_tensor(row, dataset_dir, baseline_input_size, device)
                baseline_logits = baseline_model(rgb_tensor)["out"]
                baseline_logits = F.interpolate(baseline_logits, size=baseline_hw, mode="bilinear", align_corners=False)
                baseline_pred = baseline_logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
                update_confusion(baseline_confusion, baseline_pred, gt, num_classes)

            matrix = parse_matrix(row.get("camera_matrix_json", ""))
            gt_objects = valid_localization_objects(
                object_boxes.get(row["sample_id"], []),
                image_width=int(original_size[0]),
                image_height=int(original_size[1]),
                min_area_px=float(object_cfg.get("min_gt_area_px", 24.0)),
                object_class_names=object_class_names,
            )
            predictions: List[Dict[str, float]] = []
            if matrix is not None:
                predictions = decode_objects(
                    outputs["object"],
                    camera_matrix=matrix,
                    topk=int(eval_cfg.get("topk_objects", 40)),
                    score_threshold=float(eval_cfg.get("object_score_threshold", 0.25)),
                    nms_radius_px=int(eval_cfg.get("object_nms_radius_px", 5)),
                    object_class_names=object_class_names,
                )
            matches = greedy_match_predictions(
                predictions,
                gt_objects,
                max_distance_m=float(eval_cfg.get("match_distance_m", 5.0)),
                class_aware=True,
            )
            tp += len(matches)
            fp += max(0, len(predictions) - len(matches))
            fn += max(0, len(gt_objects) - len(matches))
            matched_pred = {pred_idx for pred_idx, _, _ in matches}
            matched_gt = {gt_idx for _, gt_idx, _ in matches}
            for pred_idx, gt_idx, dist in matches:
                pred_obj = predictions[pred_idx]
                gt_obj = gt_objects[gt_idx]
                gt_class = str(gt_obj.get("class_name", "object"))
                per_class_stats.setdefault(gt_class, {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []})
                per_class_stats[gt_class]["tp"] = int(per_class_stats[gt_class]["tp"]) + 1
                per_class_stats[gt_class]["loc_errors"].append(float(dist))
                loc_errors.append(float(dist))
                loc_sq_errors.append(float(dist * dist))
                dim_err = float(np.mean(np.abs(np.array([pred_obj["size_x"], pred_obj["size_y"], pred_obj["size_z"]]) - np.array([gt_obj["size_x"], gt_obj["size_y"], gt_obj["size_z"]]))))
                dim_abs_errors.append(dim_err)
                yaw_err = yaw_error_deg(pred_obj, gt_obj)
                yaw_errors.append(yaw_err)
                pred_parked = float(pred_obj["parked_score"]) >= 0.5
                gt_parked = float(gt_obj["parked"]) >= 0.5
                parked_correct += int(pred_parked == gt_parked)
                parked_total += 1
                object_metric_rows.append(
                    {
                        "split": args.split,
                        "sample_id": row["sample_id"],
                        "frame_id": row.get("frame_id", ""),
                        "traffic_light_id": row.get("traffic_light_id", ""),
                        "match_status": "tp",
                        "class_name": gt_class,
                        "pred_class_name": pred_obj.get("class_name", ""),
                        "gt_class_name": gt_class,
                        "score": pred_obj["score"],
                        "global_xy_error_m": dist,
                        "dimension_mae_m": dim_err,
                        "yaw_error_deg": yaw_err,
                        "parked_correct": int(pred_parked == gt_parked),
                        "pred_world_x": pred_obj["world_x"],
                        "pred_world_y": pred_obj["world_y"],
                        "pred_size_x": pred_obj["size_x"],
                        "pred_size_y": pred_obj["size_y"],
                        "pred_size_z": pred_obj["size_z"],
                        "gt_world_x": gt_obj["world_x"],
                        "gt_world_y": gt_obj["world_y"],
                        "gt_size_x": gt_obj["size_x"],
                        "gt_size_y": gt_obj["size_y"],
                        "gt_size_z": gt_obj["size_z"],
                    }
                )
            for pred_idx, pred_obj in enumerate(predictions):
                if pred_idx not in matched_pred:
                    pred_class = str(pred_obj.get("class_name", "object"))
                    per_class_stats.setdefault(pred_class, {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []})
                    per_class_stats[pred_class]["fp"] = int(per_class_stats[pred_class]["fp"]) + 1
                    object_metric_rows.append(
                        {
                            "split": args.split,
                            "sample_id": row["sample_id"],
                            "match_status": "fp",
                            "class_name": pred_class,
                            "pred_class_name": pred_class,
                            "score": pred_obj["score"],
                        }
                    )
            for gt_idx, gt_obj in enumerate(gt_objects):
                if gt_idx not in matched_gt:
                    gt_class = str(gt_obj.get("class_name", "object"))
                    per_class_stats.setdefault(gt_class, {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []})
                    per_class_stats[gt_class]["fn"] = int(per_class_stats[gt_class]["fn"]) + 1
                    object_metric_rows.append(
                        {
                            "split": args.split,
                            "sample_id": row["sample_id"],
                            "match_status": "fn",
                            "class_name": gt_class,
                            "gt_class_name": gt_class,
                            "gt_world_x": gt_obj["world_x"],
                            "gt_world_y": gt_obj["world_y"],
                        }
                    )
            if bool(eval_cfg.get("classical_radar_diagnostic", False)):
                boxes = [box for box in object_boxes.get(row["sample_id"], []) if box.get("label") == "vehicle"]
                classical_errors.extend(maybe_classical_radar_diagnostic(row, boxes, dataset_dir))

    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-9, precision + recall))
    metrics: Dict[str, object] = {
        "split": args.split,
        "sample_id_contains": args.sample_id_contains or "",
        "checkpoint": str(checkpoint_path),
        "samples": len(rows),
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "generated_at": utc_iso(),
        "device": str(device),
        "device_name": device_name,
        "learned_object_tp": tp,
        "learned_object_fp": fp,
        "learned_object_fn": fn,
        "learned_object_precision": precision,
        "learned_object_recall": recall,
        "learned_object_f1": f1,
        "learned_object_class_names": list(object_class_names),
        "learned_global_xy_mae_m": float(np.mean(loc_errors)) if loc_errors else float("nan"),
        "learned_global_xy_rmse_m": float(math.sqrt(np.mean(loc_sq_errors))) if loc_sq_errors else float("nan"),
        "learned_dimension_mae_m": float(np.mean(dim_abs_errors)) if dim_abs_errors else float("nan"),
        "learned_yaw_mae_deg": float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        "learned_parked_accuracy": float(parked_correct / max(1, parked_total)) if parked_total else float("nan"),
        "learned_localization_method": "neural_object_head_direct_regression",
    }
    for class_name, stats in per_class_stats.items():
        class_tp = int(stats.get("tp", 0))
        class_fp = int(stats.get("fp", 0))
        class_fn = int(stats.get("fn", 0))
        class_precision = float(class_tp / max(1, class_tp + class_fp))
        class_recall = float(class_tp / max(1, class_tp + class_fn))
        class_f1 = float(2.0 * class_precision * class_recall / max(1e-9, class_precision + class_recall))
        class_errors = [float(v) for v in stats.get("loc_errors", [])]
        metrics[f"learned_{class_name}_object_tp"] = class_tp
        metrics[f"learned_{class_name}_object_fp"] = class_fp
        metrics[f"learned_{class_name}_object_fn"] = class_fn
        metrics[f"learned_{class_name}_object_precision"] = class_precision
        metrics[f"learned_{class_name}_object_recall"] = class_recall
        metrics[f"learned_{class_name}_object_f1"] = class_f1
        metrics[f"learned_{class_name}_global_xy_mae_m"] = (
            float(np.mean(class_errors)) if class_errors else float("nan")
        )
    for idx, name in enumerate(class_names):
        metrics[f"{name}_iou"] = float(ious[idx])
    if baseline_model is not None:
        b_miou, b_ious, b_pixel_acc = class_iou_from_confusion(baseline_confusion)
        metrics.update(
            {
                "baseline_rgb_checkpoint": baseline_checkpoint_path,
                "baseline_rgb_miou": b_miou,
                "baseline_rgb_pixel_accuracy": b_pixel_acc,
                "fusion_miou_delta_vs_rgb": float(miou - b_miou),
            }
        )
        for idx, name in enumerate(class_names):
            metrics[f"baseline_rgb_{name}_iou"] = float(b_ious[idx])
    if classical_errors:
        metrics["classical_radar_localization_diagnostic_mae_m"] = float(np.mean(classical_errors))
        metrics["classical_radar_localization_diagnostic_count"] = len(classical_errors)

    metrics_dir = exp_dir / "metrics"
    figures_dir = exp_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / f"{args.split}_fusion_evaluation_metrics.json", metrics)
    plot_confusion_matrix(confusion, class_names, figures_dir, f"{args.split}_fusion")
    if baseline_model is not None:
        plot_confusion_matrix(baseline_confusion, class_names, figures_dir, f"{args.split}_rgb_baseline")
    object_csv = metrics_dir / f"{args.split}_learned_object_metrics.csv"
    if object_metric_rows:
        with object_csv.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = sorted({key for row in object_metric_rows for key in row.keys()})
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(object_metric_rows)
    log(
        f"Evaluation split={args.split} miou={miou:.4f} vehicle_iou={metrics.get('vehicle_iou', float('nan')):.4f} "
        f"learned_xy_mae={metrics['learned_global_xy_mae_m']}; metrics={metrics_dir}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--input-size", nargs=2, type=int, default=None)
    parser.add_argument("--object-score-threshold", type=float, default=None)
    parser.add_argument("--object-nms-radius-px", type=int, default=None)
    parser.add_argument("--topk-objects", type=int, default=None)
    parser.add_argument("--match-distance-m", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--sample-id-contains", default="")
    args = parser.parse_args()
    raise SystemExit(evaluate_checkpoint(args))


if __name__ == "__main__":
    main()
