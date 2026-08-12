#!/usr/bin/env python3
"""Replay retained pedestrian inputs and score the exact center/origin metric."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from data_collection import carla_fusion_policy_corpus_collector as policy
from data_collection import run_policy_corpus as corpus_runner


base = policy.base
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "ae_integrated_20260710"
    / "noae_baseline"
    / "checkpoints"
    / "mprime_joint_noae"
    / "best.pt"
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def target_radar_hit_count(
    gt: Mapping[str, object], radar: Mapping[str, np.ndarray]
) -> int:
    """Count current-measurement radar returns inside the projected target box."""

    display_w, display_h = (int(value) for value in radar["display_size"])
    model_w, model_h = (int(value) for value in radar["model_size"])
    x0 = float(gt["bbox_x1"]) * model_w / display_w
    y0 = float(gt["bbox_y1"]) * model_h / display_h
    x1 = float(gt["bbox_x2"]) * model_w / display_w
    y1 = float(gt["bbox_y2"]) * model_h / display_h
    u = np.asarray(radar["points_u"], dtype=np.float32)
    v = np.asarray(radar["points_v"], dtype=np.float32)
    valid = np.asarray(radar["points_valid_projection"], dtype=bool)
    inside = valid & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    return int(inside.sum())


def _single_csv(run_dir: Path, suffix: str) -> Path:
    return corpus_runner._single_csv(run_dir, suffix)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _load_model(checkpoint: Path, device: torch.device):
    model_args = argparse.Namespace(
        fusion_checkpoint=str(checkpoint),
        fusion_experiment_dir="",
        model_input_width=0,
        model_input_height=0,
        num_classes=3,
        object_hidden_channels=128,
        roi_threshold=0.0,
        ae_checkpoint="",
    )
    return base.load_fusion_model(model_args, device)


def _decode(
    logits: torch.Tensor | np.ndarray,
    *,
    camera_matrix: np.ndarray,
    score_threshold: float,
    nms_radius_px: int,
    topk: int,
    class_names: Sequence[str],
    predict_bbox2d: bool,
) -> List[Dict[str, float]]:
    tensor = logits if isinstance(logits, torch.Tensor) else torch.from_numpy(logits)
    return policy._DECODE_OBJECTS(
        tensor,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        topk=int(topk),
        score_threshold=float(score_threshold),
        nms_radius_px=int(nms_radius_px),
        object_class_names=list(class_names),
        predict_bbox2d=bool(predict_bbox2d),
    )


def _target_match(
    predictions: Iterable[Mapping[str, object]],
    gt: Mapping[str, object],
    gate_m: float,
) -> Tuple[bool, Optional[float], Optional[float]]:
    target = np.asarray([float(gt["origin_x"]), float(gt["origin_y"])])
    candidates: List[Tuple[float, float]] = []
    for prediction in predictions:
        if str(prediction.get("class_name", "")).strip().lower() not in {
            "person",
            "pedestrian",
            "walker",
        }:
            continue
        point = np.asarray(
            [float(prediction["world_x"]), float(prediction["world_y"])]
        )
        distance = float(np.linalg.norm(point - target))
        candidates.append((distance, float(prediction["score"])))
    if not candidates:
        return False, None, None
    distance, score = min(candidates, key=lambda item: item[0])
    return bool(distance <= float(gate_m)), score, distance


def _prepare_input(
    rgb_path: Path,
    radar_tensor: np.ndarray,
    model_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    frame_bgr = base.cv2.imread(str(rgb_path), base.cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"unable to read retained RGB {rgb_path}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return base.prepare_fusion_input(
        frame_bgr=frame_bgr,
        radar_tensor_chw=np.asarray(radar_tensor, dtype=np.float32),
        model_size=model_size,
        device=device,
        rgb_mean=mean,
        rgb_std=std,
    )


def _method_summary(rows: pd.DataFrame, method: str) -> Dict[str, object]:
    subset = rows[rows["method"] == method]
    total = int(len(subset))
    matched = int(subset["matched"].astype(bool).sum())
    low, high = wilson_interval(matched, total)
    score = pd.to_numeric(subset["nearest_person_score"], errors="coerce").dropna()
    error = pd.to_numeric(subset["nearest_person_error_m"], errors="coerce").dropna()
    return {
        "eligible_rows": total,
        "matched_rows": matched,
        "recall": float(matched / total) if total else None,
        "recall_pct": float(100.0 * matched / total) if total else None,
        "wilson_95_pct": (
            [float(100.0 * low), float(100.0 * high)] if total else None
        ),
        "nearest_person_score_median": float(score.median()) if len(score) else None,
        "nearest_person_score_p10": float(score.quantile(0.10)) if len(score) else None,
        "nearest_person_error_median_m": float(error.median()) if len(error) else None,
    }


def replay(
    run_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    *,
    role_prefix: str,
    headline_range_m: float,
    score_threshold: float,
    association_gate_m: float,
    nms_radius_px: int,
    topk: int,
    device_name: str,
    training_reference_recall: float,
    recovery_tolerance_pp: float,
) -> Dict[str, object]:
    retained_root = run_dir / "retained_inputs"
    index = pd.read_csv(retained_root / "frames.csv")
    complete = index[
        index["rgb_path"].notna()
        & index["radar_path"].notna()
        & index["logits_path"].notna()
    ].copy()
    gt = pd.read_csv(_single_csv(run_dir, "_object_ground_truth.csv"))
    role = gt.get("role_name", pd.Series("", index=gt.index)).astype(str)
    eligible = gt[
        (gt["class_name"].astype(str).str.lower() == "pedestrian")
        & role.str.startswith(role_prefix)
        & _truthy(gt["in_camera_frustum"])
        & (pd.to_numeric(gt["distance_m"], errors="coerce") <= headline_range_m)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no retained close/in-frustum controlled pedestrian rows")
    duplicate_frames = eligible["frame_id"].duplicated(keep=False)
    if duplicate_frames.any():
        raise RuntimeError("controlled target role prefix resolved multiple actors in one frame")
    eligible = eligible.merge(complete, on="frame_id", how="inner", validate="one_to_one")
    if eligible.empty:
        raise RuntimeError("no complete retained inputs overlap controlled pedestrian GT")

    requested_device = device_name.lower()
    if requested_device == "auto":
        front_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        front_device = torch.device(requested_device)
    if front_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requested but torch.cuda.is_available() is false")
    back_device = torch.device("cpu")
    front_model, model_size = _load_model(checkpoint, front_device)
    back_model, back_size = _load_model(checkpoint, back_device)
    if tuple(model_size) != tuple(back_size):
        raise RuntimeError("front/back replay model input sizes differ")
    class_names = list(getattr(front_model, "object_class_names", ["vehicle", "person"]))
    predict_bbox2d = bool(getattr(front_model, "object_predict_bbox2d", False))
    transport = base.od_collect.TransportConfig(
        quantization_mode="per_channel_uint8",
        entropy_coder_name="zlib",
        zstd_level=3,
        roi_objectness_threshold=0.0,
        bypass_rcnn_transform=False,
    )
    front_codecs: Dict[str, object] = {}
    back_codecs: Dict[str, object] = {}
    entropy_coder = transport.make_entropy_coder()
    details: List[Dict[str, object]] = []
    replay_logits_dir = output_dir / "replay_logits"
    replay_logits_dir.mkdir(parents=True, exist_ok=False)

    for row in eligible.sort_values("frame_id").to_dict("records"):
        frame_id = int(row["frame_id"])
        with np.load(retained_root / str(row["radar_path"])) as payload:
            radar = {name: np.asarray(payload[name]) for name in payload.files}
        with np.load(retained_root / str(row["logits_path"])) as payload:
            live_logits = np.asarray(payload["object_logits"], dtype=np.float32)
        camera_matrix = np.asarray(radar["camera_matrix"], dtype=np.float64)
        fused = _prepare_input(
            retained_root / str(row["rgb_path"]),
            radar["radar"],
            tuple(int(value) for value in radar["model_size"]),
            front_device,
        )
        with torch.inference_mode():
            monolithic_outputs = front_model.model(fused)
            monolithic_logits = monolithic_outputs["object"]
            features = front_model.encode(fused)
            features = base._front_compress(
                front_model,
                features,
                (int(model_size[1]), int(model_size[0])),
            )
        serialized, _raw_bytes, _raw_levels, _compressed_levels = (
            base.od_collect.serialize_feature_maps(
                features,
                front_codecs,
                quantization_mode=transport.quantization_mode,
                per_level_compress_probe=False,
                entropy_coder=entropy_coder,
            )
        )
        replay_features = base.od_collect.deserialize_feature_maps(
            serialized,
            back_device,
            batch_size=1,
            feature_codecs=back_codecs,
            quantization_mode=transport.quantization_mode,
        )
        replay_features = base._back_decompress(back_model, replay_features)
        with torch.inference_mode():
            replay_logits = back_model.decode_outputs(
                replay_features,
                output_size=(int(model_size[1]), int(model_size[0])),
            )["object"]
        output_path = replay_logits_dir / f"frame_{frame_id:08d}.npz"
        with output_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                monolithic_object_logits=monolithic_logits.detach()
                .to("cpu", dtype=torch.float32)
                .numpy(),
                split_replay_object_logits=replay_logits.detach()
                .to("cpu", dtype=torch.float32)
                .numpy(),
            )

        methods = {
            "retained_live_logits": live_logits,
            "identical_input_split_replay": replay_logits,
            "identical_input_monolithic": monolithic_logits,
        }
        for method, logits in methods.items():
            predictions = _decode(
                logits,
                camera_matrix=camera_matrix,
                score_threshold=score_threshold,
                nms_radius_px=nms_radius_px,
                topk=topk,
                class_names=class_names,
                predict_bbox2d=predict_bbox2d,
            )
            matched, nearest_score, nearest_error = _target_match(
                predictions, row, association_gate_m
            )
            details.append(
                {
                    "frame_id": frame_id,
                    "carla_timestamp": float(row["carla_timestamp_x"]),
                    "actor_id": int(row["actor_id"]),
                    "method": method,
                    "matched": int(matched),
                    "nearest_person_score": nearest_score,
                    "nearest_person_error_m": nearest_error,
                    "target_distance_m": float(row["distance_m"]),
                    "target_radar_hit_count": target_radar_hit_count(row, radar),
                    "raw_radar_points": int(len(radar["points_u"])),
                    "radar_occupancy_nonzero_px": int(
                        np.count_nonzero(np.asarray(radar["radar"])[0])
                    ),
                    "replay_logits_path": str(output_path.relative_to(output_dir)),
                }
            )

    detail_frame = pd.DataFrame(details)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_frame.to_csv(output_dir / "per_frame_replay.csv", index=False)
    methods = (
        "retained_live_logits",
        "identical_input_split_replay",
        "identical_input_monolithic",
    )
    summaries = {method: _method_summary(detail_frame, method) for method in methods}
    split_recall = float(summaries["identical_input_split_replay"]["recall"] or 0.0)
    live_recall = float(summaries["retained_live_logits"]["recall"] or 0.0)
    monolithic_recall = float(summaries["identical_input_monolithic"]["recall"] or 0.0)
    recovery_floor = float(training_reference_recall - recovery_tolerance_pp / 100.0)
    replay_agrees_live = abs(split_recall - live_recall) <= 0.02
    if replay_agrees_live and split_recall >= recovery_floor:
        verdict = "B_CONFIRMED_SENSOR_CONTRACT_MISMATCH"
    elif monolithic_recall >= recovery_floor and split_recall < recovery_floor:
        verdict = "B_CONFIRMED_LIVE_SPLIT_PIPELINE_MISMATCH"
    else:
        verdict = "C_ESCALATE_GENUINE_MODEL_WEAKNESS"
    radar_rows = detail_frame[detail_frame["method"] == "retained_live_logits"]
    result = {
        "schema": "pedestrian_on_contract_replay.v1",
        "verdict": verdict,
        "global_halt_remains": True,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "score_threshold": float(score_threshold),
        "association": "class-aware actor-origin XY center distance",
        "association_gate_m": float(association_gate_m),
        "headline_range_m": float(headline_range_m),
        "nms_radius_px": int(nms_radius_px),
        "topk": int(topk),
        "training_reference_recall": float(training_reference_recall),
        "recovery_tolerance_pp": float(recovery_tolerance_pp),
        "recovery_floor": recovery_floor,
        "split_replay_agrees_with_retained_live_within_2pp": replay_agrees_live,
        "methods": summaries,
        "target_radar_hit_count": {
            "min": int(radar_rows["target_radar_hit_count"].min()),
            "median": float(radar_rows["target_radar_hit_count"].median()),
            "max": int(radar_rows["target_radar_hit_count"].max()),
            "frames_with_hit_pct": float(
                100.0 * (radar_rows["target_radar_hit_count"] > 0).mean()
            ),
        },
        "raw_radar_points_per_frame": {
            "median": float(radar_rows["raw_radar_points"].median()),
            "p10": float(radar_rows["raw_radar_points"].quantile(0.10)),
            "p90": float(radar_rows["raw_radar_points"].quantile(0.90)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--role-prefix", default="pedestrian_blocker_v4")
    parser.add_argument("--headline-range-m", type=float, default=25.0)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--association-gate-m", type=float, default=5.0)
    parser.add_argument("--nms-radius-px", type=int, default=2)
    parser.add_argument("--topk", type=int, default=120)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--training-reference-recall", type=float, default=0.855)
    parser.add_argument("--recovery-tolerance-pp", type=float, default=10.0)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "on_contract_replay"
    )
    result = replay(
        run_dir,
        output_dir,
        args.checkpoint.expanduser().resolve(),
        role_prefix=args.role_prefix,
        headline_range_m=args.headline_range_m,
        score_threshold=args.score_threshold,
        association_gate_m=args.association_gate_m,
        nms_radius_px=args.nms_radius_px,
        topk=args.topk,
        device_name=args.device,
        training_reference_recall=args.training_reference_recall,
        recovery_tolerance_pp=args.recovery_tolerance_pp,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
