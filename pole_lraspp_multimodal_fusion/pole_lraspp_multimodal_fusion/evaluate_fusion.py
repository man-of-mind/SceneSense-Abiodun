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
    if int(getattr(args, "limit_rows", 0) or 0) > 0:
        rows = rows[: int(args.limit_rows)]  # smoke-test / quick subset
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
    ckpt = checkpoint if isinstance(checkpoint, dict) else {}
    object_head_arch = str(ckpt.get("object_head_arch") or object_cfg.get("head_arch", "shared"))
    object_use_coordconv = bool(ckpt.get("object_use_coordconv")) or bool(object_cfg.get("use_coordconv", False))
    object_head_depth = int(ckpt.get("object_head_depth") or object_cfg.get("head_depth", 2))
    object_use_groundplane = bool(ckpt.get("object_use_groundplane_prior")) or bool(object_cfg.get("use_groundplane_prior", False))
    object_predict_bbox2d = bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False))
    object_groundplane_params = dict(ckpt.get("object_groundplane_params") or object_cfg.get("groundplane_params", {}) or {})
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        object_channels=object_channels,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=fuse_low_into_object_head,
        head_arch=object_head_arch,
        use_coordconv=object_use_coordconv,
        head_depth=object_head_depth,
        predict_bbox2d=object_predict_bbox2d,
        use_groundplane_prior=object_use_groundplane,
        groundplane_params=object_groundplane_params,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()

    # Optional split-inference codec round-trip (accuracy-vs-compression). Only the intermediate features
    # are quantized between encode and decode; ALL matching/IoU/recall/loc code below is reused unchanged,
    # so numbers are directly comparable to the uncompressed baseline (quantization_mode="").
    split_codec = None
    if getattr(args, "quantization_mode", ""):
        import carla_split_inference_udp_data_collect as od_collect
        from .split_runtime import (MultimodalLRASPPSplitModel, serialize_backbone_features,
                                    deserialize_backbone_features)
        split_codec = {
            "split": MultimodalLRASPPSplitModel(model, device, input_size=input_size),
            "transport": od_collect.TransportConfig(
                quantization_mode=str(args.quantization_mode),
                entropy_coder_name=str(getattr(args, "entropy_coder", "zlib")),
                zstd_level=int(getattr(args, "zstd_level", 3)),
                roi_objectness_threshold=0.0, bypass_rcnn_transform=False),
            "codecs": {},
            "serialize": serialize_backbone_features,
            "deserialize": deserialize_backbone_features,
            "out_hw": (int(input_size[1]), int(input_size[0])),  # decode at input resolution to match model()
            "roi_threshold": float(getattr(args, "roi_threshold", 0.0) or 0.0),
            "n_heat": int(getattr(model, "heatmap_channels", len(object_class_names))),
        }

    # Optional feature-AE: compress the 'high' feature to the bottleneck (payload accounted on the
    # bottleneck), composing AFTER ROI drop and BEFORE the codec quantization.
    ae = None
    if getattr(args, "ae_checkpoint", ""):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl_agent" / "feature_ae"))
        from ae_model import build_ae
        ae_ckpt = torch.load(Path(args.ae_checkpoint).expanduser(), map_location=device)
        ae = build_ae(ae_ckpt.get("arch", "v1"), int(ae_ckpt["in_channels"]), int(ae_ckpt["bottleneck"])).to(device)
        ae.load_state_dict(ae_ckpt["ae_state"])
        ae.eval()
        args.ae_bottleneck = int(ae_ckpt["bottleneck"])

    def _roi_gate(feats):
        """Front-side ROI drop: zero the lowest-objectness fraction q of backbone-feature cells by RANK
        (drop the k=round(q*N) lowest-objectness cells). MUST match the training-time drop
        (model._objectness_drop): a quantile VALUE threshold ties on the focal-biased objectness floor
        (~sigmoid(-4.6)) and keeps everything, so it must be rank-based. Max-pool keeps a cell if any
        high-objectness pixel falls in it, so real detections survive."""
        q = split_codec["roi_threshold"]
        if q <= 0:
            return feats
        obj_maps = split_codec["split"].decode_object_maps(feats, split_codec["out_hw"])
        objness = torch.sigmoid(obj_maps[:, :split_codec["n_heat"]]).amax(dim=1, keepdim=True)  # (1,1,H,W)
        from collections import OrderedDict as _OD
        gated = _OD()
        for name, feat in feats.items():
            pooled = F.adaptive_max_pool2d(objness, feat.shape[-2:]).reshape(-1).float()
            n = pooled.numel()
            k = int(round(float(q) * n))
            keep = torch.ones_like(pooled)
            if k > 0:
                keep[pooled.argsort()[:k]] = 0.0  # drop the k lowest-objectness cells by rank
            gated[name] = feat * keep.reshape(1, 1, feat.shape[-2], feat.shape[-1]).to(feat.dtype)
        return gated

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
    payload_bytes: List[float] = []  # offline payload size = serialized codec bytes per frame (no CARLA)
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
            if split_codec is not None:
                feats = split_codec["split"].encode(fused_tensor)
                feats = _roi_gate(feats)                                   # ROI drop on the 960-ch 'high'
                if ae is not None:                                         # then AE-encode 'high' -> bottleneck
                    feats = type(feats)((k, (ae.encode(v) if k == "high" else v)) for k, v in feats.items())
                serialized, _ = split_codec["serialize"](feats, split_codec["transport"], split_codec["codecs"])
                # On-wire payload = entropy-coded size of the quantized buffers. Entropy coding is what
                # turns ROI-drop's zeros into fewer bytes, so we must measure POST-entropy (the raw
                # quantized size is invariant to zeroing). Captures all axes: quant (buffer size),
                # entropy coder, ROI (zero-compressibility), AE (bottleneck channel count).
                _blob = b"".join(
                    _b for _entry in serialized.values()
                    for _b in (_entry.values() if isinstance(_entry, dict) else [_entry])
                    if isinstance(_b, (bytes, bytearray)))
                _ec = str(getattr(args, "entropy_coder", "zlib") or "zlib")
                if _ec == "zlib":
                    import zlib as _zlib
                    _nb = len(_zlib.compress(_blob, 6))
                elif _ec == "zstd":
                    import zstandard as _zstd
                    _nb = len(_zstd.ZstdCompressor(level=int(getattr(args, "zstd_level", 3))).compress(_blob))
                else:
                    _nb = len(_blob)
                payload_bytes.append(float(_nb))
                feats_rt = split_codec["deserialize"](serialized, device=device,
                                                      transport=split_codec["transport"],
                                                      feature_codecs=split_codec["codecs"])
                if ae is not None:                                         # AE-decode bottleneck -> 960-ch 'high'
                    feats_rt = type(feats_rt)((k, (ae.decode(v) if k == "high" else v)) for k, v in feats_rt.items())
                outputs = split_codec["split"].decode_outputs(feats_rt, split_codec["out_hw"])
            else:
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
                max_distance_m=args.max_gt_distance_m,
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
                    predict_bbox2d=object_predict_bbox2d,
                )
                # Operating-range gate on predictions: drop detections beyond range so
                # far-field false positives are not scored (matches the GT gate above).
                if args.max_gt_distance_m is not None and matrix is not None:
                    cam_c = np.asarray(matrix)[:3, 3]
                    predictions = [p for p in predictions
                                   if math.hypot(float(p["world_x"]) - cam_c[0],
                                                 float(p["world_y"]) - cam_c[1]) <= float(args.max_gt_distance_m)]
                # Radar-gated decoding: drop detections with no radar occupancy nearby (channel 0
                # of the radar tensor = fused[3]). Heatmap false positives sit where no radar
                # returns exist; true near objects do. Optionally class-restricted.
                if int(args.radar_gate_px) > 0:
                    occ = fused_tensor[0, 3].detach().cpu().numpy()  # radar occupancy @ input res
                    oh, ow = occ.shape
                    gate_cls = {c.strip() for c in args.radar_gate_classes.split(",") if c.strip()}
                    w = int(args.radar_gate_px)
                    kept = []
                    for p in predictions:
                        if gate_cls and p.get("class_name") not in gate_cls:
                            kept.append(p); continue
                        cx, cy = int(round(p["center_x_px"])), int(round(p["center_y_px"]))
                        y0, y1 = max(0, cy - w), min(oh, cy + w + 1)
                        x0, x1 = max(0, cx - w), min(ow, cx + w + 1)
                        if y0 < y1 and x0 < x1 and float(occ[y0:y1, x0:x1].max()) > 0.0:
                            kept.append(p)
                    predictions = kept
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
                        **(
                            {
                                # pred box in input-image px; gt box in original-image px.
                                # Stored with both frame sizes so 2D-IoU can be computed offline.
                                "pred_bbox_x0": pred_obj.get("bbox_x0", float("nan")),
                                "pred_bbox_y0": pred_obj.get("bbox_y0", float("nan")),
                                "pred_bbox_x1": pred_obj.get("bbox_x1", float("nan")),
                                "pred_bbox_y1": pred_obj.get("bbox_y1", float("nan")),
                                "input_w": int(input_size[0]),
                                "input_h": int(input_size[1]),
                                "gt_center_x": gt_obj.get("center_x", float("nan")),
                                "gt_center_y": gt_obj.get("center_y", float("nan")),
                                "gt_bbox_w": gt_obj.get("bbox_w", float("nan")),
                                "gt_bbox_h": gt_obj.get("bbox_h", float("nan")),
                                "orig_w": int(original_size[0]),
                                "orig_h": int(original_size[1]),
                            }
                            if object_predict_bbox2d
                            else {}
                        ),
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
                            "pred_world_x": pred_obj["world_x"],
                            "pred_world_y": pred_obj["world_y"],
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
        "quantization_mode": str(getattr(args, "quantization_mode", "") or "uncompressed"),
        "entropy_coder": str(getattr(args, "entropy_coder", "") if getattr(args, "quantization_mode", "") else ""),
        "roi_threshold": float(getattr(args, "roi_threshold", 0.0) or 0.0),
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
        # --- split-inference knob settings + offline payload (bytes), for the knob matrix ---
        "quantization_mode": str(getattr(args, "quantization_mode", "") or ""),
        "entropy_coder": str(getattr(args, "entropy_coder", "") or ""),
        "roi_threshold": float(getattr(args, "roi_threshold", 0.0) or 0.0),
        "ae_bottleneck": int(getattr(args, "ae_bottleneck", 0) or 0),
        "payload_bytes_mean": float(np.mean(payload_bytes)) if payload_bytes else float("nan"),
        "payload_bytes_p95": float(np.percentile(payload_bytes, 95)) if payload_bytes else float("nan"),
        "payload_frames": int(len(payload_bytes)),
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
    parser.add_argument("--max-gt-distance-m", type=float, default=None,
                        help="Operating-range gate: ignore GT and predictions beyond this range (m).")
    parser.add_argument("--radar-gate-px", type=int, default=0,
                        help="If >0, drop decoded detections with no radar occupancy within this "
                             "window (px) of the center -> kills heatmap false positives.")
    parser.add_argument("--radar-gate-classes", type=str, default="",
                        help="Comma-separated classes the radar gate applies to (empty = all). "
                             "e.g. 'vehicle' to spare pedestrians (weak radar return).")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--sample-id-contains", default="")
    parser.add_argument("--quantization-mode", default="",
                        help="If set, route inference through the split-inference codec round-trip at this "
                             "quantization (per_tensor_uint8 / per_channel_uint8 / per_channel_uint4) to "
                             "measure accuracy-vs-compression. Empty = uncompressed baseline.")
    parser.add_argument("--entropy-coder", default="zlib", choices=("zlib", "zstd", "none"))
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--roi-threshold", type=float, default=0.0,
                        help="ROI drop FRACTION in [0,1) (with --quantization-mode set): drop the lowest-"
                             "objectness fraction of backbone-feature cells before the codec (quantile-based, "
                             "importance drop). e.g. 0.3 = drop bottom 30%%. Needs a quant mode active (use "
                             "per_channel_uint8 to isolate the ROI effect).")
    parser.add_argument("--ae-checkpoint", default="",
                        help="Feature-AE checkpoint (ae_bN.pt). If set, the 'high' feature is AE-encoded to the "
                             "bottleneck before the codec (payload = quantized bottleneck bytes) and AE-decoded "
                             "after, matching deployment. Composes after ROI drop.")
    parser.add_argument("--ae-bottleneck", type=int, default=0, help="recorded in metrics; auto-set from checkpoint")
    parser.add_argument("--limit-rows", type=int, default=0, help="evaluate only the first N rows (smoke/subset)")
    args = parser.parse_args()
    raise SystemExit(evaluate_checkpoint(args))


if __name__ == "__main__":
    main()
