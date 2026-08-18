#!/usr/bin/env python3
"""Analyze matched medium/crowded Low-vs-Epic production-collector trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scenesense_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data_collection.analyze_phase2_renderer_quality_gate import (
    _one_to_one_counts,
    canonical_prediction_class,
)
from data_collection.run_phase2_renderer_dense_confirmation import (
    CONFIG_BY_QUALITY,
    load_stage_config,
)


CLASS_NAMES = ("pedestrian", "vehicle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _single_csv(run_dir: Path, suffix: str) -> Path:
    matches = sorted((run_dir / "streams").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"expected one *{suffix} under {run_dir}, found {len(matches)}")
    return matches[0]


def _load_batch(root: Path, quality: str) -> tuple[dict, dict]:
    root = Path(root).resolve()
    if not (root / "COMPLETED.json").is_file() or (root / "FAILED.json").exists():
        raise ValueError(f"{quality} dense confirmation is not complete: {root}")
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "collection_complete_pending_verification":
        raise ValueError(f"{quality} batch manifest is incomplete")
    config_path = root / "resolved_collection_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    renderer = config.get("renderer_quality", {})
    if renderer.get("declared_quality_level") != quality or renderer.get(
        "required_server_launch_flag"
    ) != f"-quality-level={quality}":
        raise ValueError(f"{quality} resolved renderer declaration is invalid")
    if len(manifest.get("runs", [])) != 2:
        raise ValueError("each quality must contain exactly medium and crowded")
    complete = {"complete", "complete_with_teardown_warning"}
    if any(item.get("status") not in complete for item in manifest["runs"]):
        raise ValueError(f"{quality} contains an incomplete run")
    return manifest, config


def _load_tables(root: Path, quality: str) -> dict[str, pd.DataFrame]:
    manifest, config = _load_batch(root, quality)
    metrics_frames = []
    truth_frames = []
    prediction_frames = []
    for record in manifest["runs"]:
        episode_id = str(record["episode_id"])
        family = str(record["scenario_family"])
        run_dir = root / "runs" / episode_id
        metrics = pd.read_csv(_single_csv(run_dir, "_metrics.csv"))
        metrics = metrics.sort_values("frame_id").reset_index(drop=True)
        metrics["frame_ordinal"] = np.arange(len(metrics), dtype=int)
        ordinals = metrics[["frame_id", "frame_ordinal"]]
        truth = pd.read_csv(_single_csv(run_dir, "_object_ground_truth.csv"))
        predictions = pd.read_csv(_single_csv(run_dir, "_object_predictions.csv"))
        truth = truth.merge(ordinals, on="frame_id", how="inner", validate="many_to_one")
        predictions = predictions.merge(
            ordinals, on="frame_id", how="inner", validate="many_to_one"
        )
        if not predictions.empty:
            predictions["class_name"] = predictions["class_name"].map(
                canonical_prediction_class
            )
        for frame in (metrics, truth, predictions):
            frame["quality"] = quality
            frame["episode_id"] = episode_id
            frame["scenario_family"] = family
        metrics_frames.append(metrics)
        truth_frames.append(truth)
        prediction_frames.append(predictions)
    return {
        "metrics": pd.concat(metrics_frames, ignore_index=True),
        "truth": pd.concat(truth_frames, ignore_index=True),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "config": config,
    }


def detection_curve(tables: Mapping[str, pd.DataFrame], evaluation: Mapping[str, object]) -> pd.DataFrame:
    thresholds = [float(value) for value in evaluation["postdecoder_score_thresholds"]]
    ranges = {
        str(key): float(value)
        for key, value in evaluation["near_range_m_by_class"].items()
    }
    gate_m = float(evaluation["localization_match_gate_m"])
    rows = []
    for (quality, episode_id, family), metric_group in tables["metrics"].groupby(
        ["quality", "episode_id", "scenario_family"], sort=True
    ):
        truth_group = tables["truth"]
        pred_group = tables["predictions"]
        for field, value in (
            ("quality", quality),
            ("episode_id", episode_id),
            ("scenario_family", family),
        ):
            truth_group = truth_group[truth_group[field] == value]
            pred_group = pred_group[pred_group[field] == value]
        ordinals = sorted(int(value) for value in metric_group["frame_ordinal"].unique())
        for class_name in CLASS_NAMES:
            near_truth = truth_group[
                (truth_group["class_name"] == class_name)
                & (pd.to_numeric(truth_group["in_camera_frustum"], errors="coerce") == 1)
                & (pd.to_numeric(truth_group["distance_m"], errors="coerce") <= ranges[class_name])
            ]
            near_predictions = pred_group[
                (pred_group["class_name"] == class_name)
                & (pd.to_numeric(pred_group["distance_m"], errors="coerce") <= ranges[class_name])
            ]
            for threshold in thresholds:
                thresholded = near_predictions[
                    pd.to_numeric(near_predictions["score"], errors="coerce") >= threshold
                ]
                tp = fp = fn = 0
                for ordinal in ordinals:
                    frame_tp, frame_fp, frame_fn = _one_to_one_counts(
                        near_truth[near_truth["frame_ordinal"] == ordinal],
                        thresholded[thresholded["frame_ordinal"] == ordinal],
                        gate_m,
                    )
                    tp += frame_tp
                    fp += frame_fp
                    fn += frame_fn
                precision = float(tp / (tp + fp)) if tp + fp else float("nan")
                recall = float(tp / (tp + fn)) if tp + fn else float("nan")
                rows.append(
                    {
                        "quality": quality,
                        "episode_id": episode_id,
                        "scenario_family": family,
                        "class_name": class_name,
                        "near_range_m": ranges[class_name],
                        "score_threshold": threshold,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "precision": precision,
                        "recall": recall,
                        "metric_scope": "actor_origin_one_to_one_postdecoder_floor_0.05",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    grouped = curve.groupby(
        ["quality", "class_name", "near_range_m", "score_threshold"], as_index=False
    )[["true_positives", "false_positives", "false_negatives"]].sum()
    grouped["precision"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_positives"]
    ).replace(0, np.nan)
    grouped["recall"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_negatives"]
    ).replace(0, np.nan)
    grouped["metric_scope"] = "actor_origin_one_to_one_postdecoder_floor_0.05"
    return grouped.sort_values(["quality", "class_name", "score_threshold"])


def capture_diagnostics(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for keys, frame in tables["metrics"].groupby(
        ["quality", "episode_id", "scenario_family"], sort=True
    ):
        quality, episode_id, family = keys
        received = frame["result_received"].astype(str).str.lower().isin(("true", "1"))
        rows.append(
            {
                "quality": quality,
                "episode_id": episode_id,
                "scenario_family": family,
                "frames": int(len(frame)),
                "result_received_frames": int(received.sum()),
                "radar_projected_points_median": float(
                    pd.to_numeric(frame["radar_projected_points"], errors="coerce").median()
                ),
                "semantic_gt_frames": int(
                    pd.to_numeric(frame["gt_camera_available"], errors="coerce").fillna(0).sum()
                ),
                "segmentation_miou": float(
                    pd.to_numeric(frame["miou_3class_macro"], errors="coerce").mean()
                ),
                "vehicle_iou": float(
                    pd.to_numeric(frame["miou_vehicle_iou"], errors="coerce").mean()
                ),
                "person_iou": float(
                    pd.to_numeric(frame["miou_person_iou"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def population_diagnostics(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    requested = {"medium": {"vehicle": 20, "pedestrian": 25}, "crowded": {"vehicle": 28, "pedestrian": 35}}
    rows = []
    for keys, frame in tables["truth"].groupby(
        ["quality", "episode_id", "scenario_family"], sort=True
    ):
        quality, episode_id, family = keys
        for class_name in CLASS_NAMES:
            class_frame = frame[frame["class_name"] == class_name]
            per_frame = class_frame.groupby("frame_ordinal")["actor_id"].nunique()
            realized = int(per_frame.max()) if len(per_frame) else 0
            target = int(requested[family][class_name])
            rows.append(
                {
                    "quality": quality,
                    "episode_id": episode_id,
                    "scenario_family": family,
                    "class_name": class_name,
                    "requested_count": target,
                    "maximum_realized_count": realized,
                    "realized_fraction": float(realized / target),
                }
            )
    return pd.DataFrame(rows)


def _greedy_position_matches(left: pd.DataFrame, right: pd.DataFrame, tolerance_m: float) -> int:
    if left.empty or right.empty:
        return 0
    left_xyz = left[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=float)
    right_xyz = right[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(left_xyz[:, None, :] - right_xyz[None, :, :], axis=2)
    candidates = sorted(
        (float(distances[i, j]), i, j)
        for i in range(len(left))
        for j in range(len(right))
        if float(distances[i, j]) <= tolerance_m
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _distance, i, j in candidates:
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
    return len(used_left)


def matched_world_diagnostics(
    low: Mapping[str, pd.DataFrame], epic: Mapping[str, pd.DataFrame], tolerance_m: float
) -> pd.DataFrame:
    rows = []
    for family in ("medium", "crowded"):
        for class_name in CLASS_NAMES:
            low_group = low["truth"][(low["truth"]["scenario_family"] == family) & (low["truth"]["class_name"] == class_name)]
            epic_group = epic["truth"][(epic["truth"]["scenario_family"] == family) & (epic["truth"]["class_name"] == class_name)]
            low_first = low_group[low_group["frame_ordinal"] == low_group["frame_ordinal"].min()]
            epic_first = epic_group[epic_group["frame_ordinal"] == epic_group["frame_ordinal"].min()]
            matched = _greedy_position_matches(low_first, epic_first, tolerance_m)
            denominator = max(len(low_first), len(epic_first), 1)
            rows.append(
                {
                    "scenario_family": family,
                    "class_name": class_name,
                    "low_initial_count": int(len(low_first)),
                    "epic_initial_count": int(len(epic_first)),
                    "position_matches": int(matched),
                    "position_tolerance_m": tolerance_m,
                    "position_match_fraction": float(matched / denominator),
                    "count_relative_delta": float(abs(len(low_first) - len(epic_first)) / denominator),
                }
            )
    return pd.DataFrame(rows)


def _plot_pr(curve: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, class_name in zip(axes, CLASS_NAMES):
        subset = curve[curve["class_name"] == class_name]
        for quality, frame in subset.groupby("quality"):
            axis.plot(frame["recall"], frame["precision"], marker="o", label=quality)
        axis.set(title=class_name.capitalize(), xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Training-density renderer confirmation (postdecoder floor 0.05)")
    return figure


def analyze(low_root: Path, epic_root: Path, output_dir: Path) -> dict:
    _path, reference = load_stage_config("Low")
    evaluation = reference["evaluation"]
    low = _load_tables(Path(low_root).resolve(), "Low")
    epic = _load_tables(Path(epic_root).resolve(), "Epic")
    combined = {
        key: pd.concat([low[key], epic[key]], ignore_index=True)
        for key in ("metrics", "truth", "predictions")
    }
    curve_by_run = detection_curve(combined, evaluation)
    curve = aggregate_curve(curve_by_run)
    capture = capture_diagnostics(combined)
    population = population_diagnostics(combined)
    world_match = matched_world_diagnostics(
        low, epic, float(evaluation["initial_position_tolerance_m"])
    )

    primary = float(evaluation["primary_score_threshold"])
    primary_rows = curve[np.isclose(curve["score_threshold"], primary)].copy()
    recall = {
        (str(row.quality), str(row.class_name)): float(row.recall)
        for row in primary_rows.itertuples()
    }
    segmentation = capture.groupby("quality")["segmentation_miou"].mean().to_dict()
    weights = evaluation["reward_v5_task_weights"]
    utility = {
        quality: float(
            float(weights["segmentation"]) * float(segmentation[quality])
            + float(weights["pedestrian_recall"]) * recall[(quality, "pedestrian")]
            + float(weights["vehicle_recall"]) * recall[(quality, "vehicle")]
        )
        for quality in ("Low", "Epic")
    }
    components = {
        quality: {
            "segmentation": float(segmentation[quality]),
            "pedestrian_recall": recall[(quality, "pedestrian")],
            "vehicle_recall": recall[(quality, "vehicle")],
        }
        for quality in ("Low", "Epic")
    }

    expected_frames = 120
    structural_pass = bool(
        (capture["frames"] == expected_frames).all()
        and (capture["result_received_frames"] == expected_frames).all()
        and (capture["semantic_gt_frames"] == expected_frames).all()
    )
    radar = capture.groupby("quality")["radar_projected_points_median"].median()
    radar_delta = abs(float(radar["Epic"]) - float(radar["Low"])) / max(float(radar["Low"]), 1.0)
    radar_pass = radar_delta <= float(evaluation["radar_median_relative_tolerance"])
    population_pass = bool(
        (population["realized_fraction"] >= float(evaluation["minimum_realized_spawn_fraction"])).all()
    )
    matched_world_pass = bool(
        (world_match["count_relative_delta"] <= float(evaluation["matched_population_count_relative_tolerance"])).all()
        and (world_match["position_match_fraction"] >= float(evaluation["minimum_initial_position_match_fraction"])).all()
    )
    finite_components = all(math.isfinite(value) for values in components.values() for value in values.values())
    validity_pass = structural_pass and radar_pass and population_pass and matched_world_pass and finite_components

    if not validity_pass:
        verdict = "HOLD_INVALID_MATCHED_CAPTURE"
        interpretation = "The Low/Epic trials are not a valid matched renderer comparison; do not choose a corpus renderer from them."
    else:
        winner = max(utility, key=utility.get)
        loser = "Low" if winner == "Epic" else "Epic"
        utility_gap = float(utility[winner] - utility[loser])
        worst_regression = max(
            components[loser][name] - components[winner][name]
            for name in components[winner]
        )
        dominates = utility_gap >= float(evaluation["dominance_utility_margin"]) and worst_regression <= float(evaluation["maximum_component_regression"])
        if dominates:
            verdict = f"PIN_{winner.upper()}_PRIMARY"
            interpretation = f"Pin {winner} as the primary renderer; retain {loser} only as an explicit stress stratum."
        else:
            verdict = str(evaluation["inconclusive_fallback"])
            interpretation = "Neither renderer dominates the reward-v5 task components; renderer quality remains an explicit corpus stratum."

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    curve_by_run.to_csv(output_dir / "pr_curve_by_run.csv", index=False)
    curve.to_csv(output_dir / "pr_curve.csv", index=False)
    primary_rows.to_csv(output_dir / "primary_threshold_metrics.csv", index=False)
    capture.to_csv(output_dir / "capture_and_segmentation_metrics.csv", index=False)
    population.to_csv(output_dir / "population_realization.csv", index=False)
    world_match.to_csv(output_dir / "matched_world_diagnostics.csv", index=False)
    figure = _plot_pr(curve)
    figure.savefig(output_dir / "renderer_pr_curve.png", dpi=300)
    figure.savefig(output_dir / "renderer_pr_curve.pdf")
    plt.close(figure)

    summary = _finite_or_none(
        {
            "schema": "scenesense.phase2_renderer_dense_confirmation_decision.v1",
            "verdict": verdict,
            "interpretation": interpretation,
            "comparison_id": reference["renderer_quality"]["comparison_id"],
            "reward_v5_task_utility": utility,
            "reward_v5_task_components": components,
            "checks": {
                "structural_capture": {"pass": structural_pass},
                "radar_invariance": {
                    "pass": radar_pass,
                    "low_median": float(radar["Low"]),
                    "epic_median": float(radar["Epic"]),
                    "relative_delta": radar_delta,
                },
                "population_realization": {"pass": population_pass},
                "matched_world": {"pass": matched_world_pass},
                "finite_task_components": {"pass": finite_components},
            },
            "decision_rule": {
                "dominance_utility_margin": float(evaluation["dominance_utility_margin"]),
                "maximum_component_regression": float(evaluation["maximum_component_regression"]),
                "fallback": str(evaluation["inconclusive_fallback"]),
            },
            "limitations": [
                "Two fixed-seed density trials per renderer are a contract decision, not a generalization study.",
                "Detector PR is conditional on postdecoder candidates emitted at score >= 0.05.",
                "CARLA has no renderer-quality RPC; quality is operator-declared from the server launch flag.",
                "Traffic sanity here covers deterministic population realization, matched initial geometry, and actor cleanup; the generic collector has no collision sensor.",
                "Shared-GPU inference timing is non-citable and is excluded from the renderer decision.",
            ],
            "inputs": {"low_root": str(Path(low_root).resolve()), "epic_root": str(Path(epic_root).resolve())},
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    decision_path = output_dir / "RENDERER_DENSE_CONFIRMATION_DECISION.json"
    with decision_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    with (output_dir / "artifact_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--epic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.low_root, args.epic_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
