from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
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
from .train_lraspp import build_lraspp


SAMPLE_METRIC_FIELDS = (
    "split",
    "sample_id",
    "frame_id",
    "traffic_light_id",
    "view_id",
    "class_name",
    "gt_mask_area_px",
    "pred_mask_area_px",
    "mask_area_error_px",
    "relative_mask_area_error",
)

OBJECT_METRIC_FIELDS = (
    "split",
    "sample_id",
    "frame_id",
    "traffic_light_id",
    "view_id",
    "label",
    "row_type",
    "component_index",
    "gt_actor_id",
    "gt_source",
    "gt_actor_type_id",
    "bbox_iou",
    "pred_mask_area_px",
    "pred_bbox_x",
    "pred_bbox_y",
    "pred_bbox_w",
    "pred_bbox_h",
    "pred_bbox_area_px",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_w",
    "gt_bbox_h",
    "gt_bbox_area_px",
    "pred_bbox_to_gt_area_ratio",
)


def load_image_tensor(image_path: Path, input_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized = image.resize(input_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device), (height, width)


def load_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)


def predict_mask(
    model: torch.nn.Module,
    image_path: Path,
    input_size: Tuple[int, int],
    output_hw: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    tensor, _ = load_image_tensor(image_path, input_size, device)
    with torch.inference_mode():
        logits = model(tensor)["out"]
        logits = F.interpolate(logits, size=output_hw, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 0.0 else 0.0


def connected_components(mask: np.ndarray, min_area: int) -> List[Dict[str, float]]:
    binary = mask.astype(np.uint8)
    if not binary.any():
        return []
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components: List[Dict[str, float]] = []
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        w = int(stats[component_index, cv2.CC_STAT_WIDTH])
        h = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "component_index": component_index,
                "mask_area": float(area),
                "bbox": (float(x), float(y), float(w), float(h)),
                "bbox_area": float(w * h),
            }
        )
    components.sort(key=lambda item: float(item["mask_area"]), reverse=True)
    return components


def load_object_boxes(path: Path) -> Dict[str, List[Dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped


def match_components_to_boxes(
    components: Sequence[Dict[str, float]],
    boxes: Sequence[Dict[str, str]],
    min_iou: float,
) -> Dict[int, Tuple[int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for component_index, component in enumerate(components):
        comp_bbox = component["bbox"]
        for box_index, box in enumerate(boxes):
            gt_bbox = (
                float(box.get("gt_bbox_x", 0.0) or 0.0),
                float(box.get("gt_bbox_y", 0.0) or 0.0),
                float(box.get("gt_bbox_w", 0.0) or 0.0),
                float(box.get("gt_bbox_h", 0.0) or 0.0),
            )
            iou = bbox_iou(comp_bbox, gt_bbox)
            if iou >= float(min_iou):
                candidates.append((iou, component_index, box_index))
    candidates.sort(reverse=True)
    matches: Dict[int, Tuple[int, float]] = {}
    used_components = set()
    used_boxes = set()
    for iou, component_index, box_index in candidates:
        if component_index in used_components or box_index in used_boxes:
            continue
        used_components.add(component_index)
        used_boxes.add(box_index)
        matches[component_index] = (box_index, float(iou))
    return matches


def find_best_checkpoint(exp_dir: Path) -> Path:
    summaries = sorted((exp_dir / "checkpoints").glob("*/trial_summary.json"))
    best_path: Optional[Path] = None
    best_miou = -math.inf
    for summary_path in summaries:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        miou = float(payload.get("best_miou", float("nan")))
        checkpoint = Path(str(payload.get("best_checkpoint", ""))).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (exp_dir / checkpoint).resolve()
        if checkpoint.exists() and miou > best_miou:
            best_miou = miou
            best_path = checkpoint
    if best_path is None:
        raise FileNotFoundError(f"No usable best checkpoint found under {exp_dir / 'checkpoints'}")
    return best_path


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, class_names: Sequence[str], output_dir: Path, split: str) -> None:
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{split} normalized confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", color="black")
    save_figure(fig, output_dir / f"{split}_confusion_matrix")


def plot_class_iou(metrics: Dict[str, float], class_names: Sequence[str], output_dir: Path, split: str) -> None:
    values = [metrics.get(f"{name}_iou", float("nan")) for name in class_names]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(class_names, values, color=["#6b7280", "#2563eb", "#f97316"][: len(class_names)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title(f"{split} class IoU")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, output_dir / f"{split}_class_iou")


def plot_temporal_variation(object_metrics_path: Path, output_dir: Path, split: str) -> Optional[Path]:
    if not object_metrics_path.exists():
        return None
    df = pd.read_csv(object_metrics_path)
    matched = df[df["row_type"] == "matched"].copy()
    if matched.empty:
        return None
    for column in ("frame_id", "pred_mask_area_px", "gt_bbox_area_px", "bbox_iou", "pred_bbox_to_gt_area_ratio"):
        matched[column] = pd.to_numeric(matched[column], errors="coerce")
    matched["object_key"] = (
        matched["gt_actor_id"].astype(str)
        + " | "
        + matched["label"].astype(str)
        + " | "
        + matched["gt_source"].astype(str)
    )
    counts = matched.groupby("object_key").size().sort_values(ascending=False)
    top_keys = counts.head(8).index.tolist()
    top = matched[matched["object_key"].isin(top_keys)].copy()

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for key in top_keys:
        subset = top[top["object_key"] == key].sort_values("frame_id")
        label = key.split(" | ")[0]
        ax.plot(subset["frame_id"], subset["pred_mask_area_px"], linewidth=1.5, label=f"pred {label}")
        ax.plot(subset["frame_id"], subset["gt_bbox_area_px"], linewidth=1.1, linestyle="--", label=f"GT {label}")
    ax.set_title(f"{split} predicted segmentation area vs CARLA projected GT")
    ax.set_xlabel("CARLA frame")
    ax.set_ylabel("Area (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, output_dir / f"{split}_overlay_area_vs_gt_over_time")

    summary = (
        matched.groupby(["gt_actor_id", "gt_source", "gt_actor_type_id", "label"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            pred_mask_area_mean_px=("pred_mask_area_px", "mean"),
            pred_mask_area_std_px=("pred_mask_area_px", "std"),
            gt_bbox_area_mean_px=("gt_bbox_area_px", "mean"),
            gt_bbox_area_std_px=("gt_bbox_area_px", "std"),
            mean_bbox_iou=("bbox_iou", "mean"),
            mean_pred_bbox_to_gt_area_ratio=("pred_bbox_to_gt_area_ratio", "mean"),
        )
        .reset_index()
    )
    summary["pred_mask_area_cv"] = summary["pred_mask_area_std_px"] / summary["pred_mask_area_mean_px"].replace(0, np.nan)
    summary["gt_bbox_area_cv"] = summary["gt_bbox_area_std_px"] / summary["gt_bbox_area_mean_px"].replace(0, np.nan)
    summary = summary.sort_values(["samples", "mean_bbox_iou"], ascending=[False, False])
    summary_path = output_dir / f"{split}_object_temporal_variation_summary.csv"
    summary.to_csv(summary_path, index=False)

    top_summary = summary.head(10)
    x = np.arange(len(top_summary))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - 0.18, top_summary["pred_mask_area_cv"].fillna(0.0), width=0.36, label="pred mask CV")
    ax.bar(x + 0.18, top_summary["gt_bbox_area_cv"].fillna(0.0), width=0.36, label="GT box CV")
    ax.set_xticks(x, top_summary["gt_actor_id"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title(f"{split} object area variation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / f"{split}_object_area_cv")
    return summary_path


def evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    dataset_dir = exp_dir / "dataset"
    log = setup_logger(exp_dir / "supervisor.log")
    split = str(args.split)
    rows = [row for row in read_manifest(dataset_dir / "manifest.csv") if row.get("split") == split]
    if not rows:
        raise RuntimeError(f"No rows found for split={split} in {dataset_dir / 'manifest.csv'}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else find_best_checkpoint(exp_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_cfg = config["training"]
    num_classes = int(train_cfg.get("num_classes", 3))
    input_width, input_height = [int(v) for v in checkpoint.get("input_size", checkpoint.get("trial", {}).get("input_size", train_cfg.get("input_size", [512, 288])))]
    class_names = list(CLASS_NAMES[:num_classes])
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(args.cpu) else "cpu")
    model = build_lraspp(num_classes, bool(train_cfg.get("pretrained", True))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log(f"Evaluating {checkpoint_path} on {split} split with {len(rows)} rows on {device}.")

    object_boxes = load_object_boxes(dataset_dir / "object_boxes.csv")
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    sample_metrics: List[Dict] = []
    object_metrics: List[Dict] = []
    min_component_area = int(args.min_component_area)
    min_iou = float(args.gt_match_iou)

    for row in rows:
        gt = load_mask(dataset_dir / row["mask_path"])
        pred = predict_mask(model, dataset_dir / row["rgb_path"], (input_width, input_height), gt.shape, device)
        update_confusion(confusion, pred, gt, num_classes)
        sample_id = str(row["sample_id"])
        frame_id = int(row.get("frame_id", 0) or 0)
        for class_id, class_name in enumerate(class_names):
            gt_area = int(np.sum(gt == class_id))
            pred_area = int(np.sum(pred == class_id))
            error = abs(pred_area - gt_area)
            sample_metrics.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "frame_id": frame_id,
                    "traffic_light_id": row.get("traffic_light_id", ""),
                    "view_id": row.get("view_id", ""),
                    "class_name": class_name,
                    "gt_mask_area_px": gt_area,
                    "pred_mask_area_px": pred_area,
                    "mask_area_error_px": error,
                    "relative_mask_area_error": float(error / gt_area) if gt_area > 0 else float("nan"),
                }
            )

        for class_id, label in ((1, "vehicle"), (2, "person")):
            components = connected_components(pred == class_id, min_component_area)
            boxes = [box for box in object_boxes.get(sample_id, []) if box.get("label") == label]
            matches = match_components_to_boxes(components, boxes, min_iou)
            matched_box_indices = {box_index for box_index, _ in matches.values()}
            for comp_index, component in enumerate(components):
                pred_bbox = component["bbox"]
                metric = {
                    "split": split,
                    "sample_id": sample_id,
                    "frame_id": frame_id,
                    "traffic_light_id": row.get("traffic_light_id", ""),
                    "view_id": row.get("view_id", ""),
                    "label": label,
                    "row_type": "matched" if comp_index in matches else "unmatched_prediction",
                    "component_index": int(component["component_index"]),
                    "pred_mask_area_px": component["mask_area"],
                    "pred_bbox_x": pred_bbox[0],
                    "pred_bbox_y": pred_bbox[1],
                    "pred_bbox_w": pred_bbox[2],
                    "pred_bbox_h": pred_bbox[3],
                    "pred_bbox_area_px": component["bbox_area"],
                }
                if comp_index in matches:
                    box_index, iou = matches[comp_index]
                    box = boxes[box_index]
                    gt_area = float(box.get("gt_bbox_area_px", 0.0) or 0.0)
                    metric.update(
                        {
                            "gt_actor_id": box.get("gt_actor_id", ""),
                            "gt_source": box.get("gt_source", ""),
                            "gt_actor_type_id": box.get("gt_actor_type_id", ""),
                            "bbox_iou": iou,
                            "gt_bbox_x": box.get("gt_bbox_x", ""),
                            "gt_bbox_y": box.get("gt_bbox_y", ""),
                            "gt_bbox_w": box.get("gt_bbox_w", ""),
                            "gt_bbox_h": box.get("gt_bbox_h", ""),
                            "gt_bbox_area_px": gt_area,
                            "pred_bbox_to_gt_area_ratio": float(component["bbox_area"] / gt_area) if gt_area > 0 else float("nan"),
                        }
                    )
                object_metrics.append(metric)
            for box_index, box in enumerate(boxes):
                if box_index in matched_box_indices:
                    continue
                object_metrics.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "frame_id": frame_id,
                        "traffic_light_id": row.get("traffic_light_id", ""),
                        "view_id": row.get("view_id", ""),
                        "label": label,
                        "row_type": "unmatched_gt",
                        "component_index": -1,
                        "gt_actor_id": box.get("gt_actor_id", ""),
                        "gt_source": box.get("gt_source", ""),
                        "gt_actor_type_id": box.get("gt_actor_type_id", ""),
                        "gt_bbox_x": box.get("gt_bbox_x", ""),
                        "gt_bbox_y": box.get("gt_bbox_y", ""),
                        "gt_bbox_w": box.get("gt_bbox_w", ""),
                        "gt_bbox_h": box.get("gt_bbox_h", ""),
                        "gt_bbox_area_px": box.get("gt_bbox_area_px", ""),
                    }
                )

    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    metrics = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "updated_at": utc_iso(),
        "samples": len(rows),
        "miou": miou,
        "pixel_accuracy": pixel_acc,
    }
    for idx, name in enumerate(class_names):
        metrics[f"{name}_iou"] = float(ious[idx])
    vehicle_rows = [r for r in sample_metrics if r["class_name"] == "vehicle" and not math.isnan(float(r["relative_mask_area_error"]))]
    person_rows = [r for r in sample_metrics if r["class_name"] == "person" and not math.isnan(float(r["relative_mask_area_error"]))]
    metrics["vehicle_mean_relative_mask_area_error"] = float(np.mean([r["relative_mask_area_error"] for r in vehicle_rows])) if vehicle_rows else float("nan")
    metrics["person_mean_relative_mask_area_error"] = float(np.mean([r["relative_mask_area_error"] for r in person_rows])) if person_rows else float("nan")

    metrics_dir = exp_dir / "metrics"
    figures_dir = exp_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sample_metrics_path = metrics_dir / f"{split}_sample_metrics.csv"
    object_metrics_path = metrics_dir / f"{split}_object_metrics.csv"
    with sample_metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_METRIC_FIELDS)
        writer.writeheader()
        for metric in sample_metrics:
            writer.writerow({field: metric.get(field, "") for field in SAMPLE_METRIC_FIELDS})
    with object_metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBJECT_METRIC_FIELDS)
        writer.writeheader()
        for metric in object_metrics:
            writer.writerow({field: metric.get(field, "") for field in OBJECT_METRIC_FIELDS})

    save_json(metrics_dir / f"{split}_evaluation_metrics.json", metrics)
    plot_confusion_matrix(confusion, class_names, figures_dir, split)
    plot_class_iou(metrics, class_names, figures_dir, split)
    variation_summary = plot_temporal_variation(object_metrics_path, figures_dir, split)
    if variation_summary is not None:
        metrics["temporal_variation_summary"] = str(variation_summary)
        save_json(metrics_dir / f"{split}_evaluation_metrics.json", metrics)
    log(f"{split} mIoU={miou:.4f} vehicle_iou={metrics.get('vehicle_iou', float('nan')):.4f} person_iou={metrics.get('person_iou', float('nan')):.4f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--gt-match-iou", type=float, default=0.05)
    parser.add_argument("--min-component-area", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    raise SystemExit(evaluate(args))


if __name__ == "__main__":
    main()
