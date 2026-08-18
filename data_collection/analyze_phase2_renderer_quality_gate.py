#!/usr/bin/env python3
"""Compare the paired Low/Epic Phase-2 renderer-quality gate captures.

Detection metrics use actor-origin, one-to-one, 5 m center matching and only
postdecoder candidates emitted at score >= 0.05.  Consequently the PR sweep is
explicitly conditional on that decoder floor.  Segmentation has no semantic-GT
camera in the causal pilot; it is reported as paired prediction stability, not
as accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scenesense_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data_collection.run_phase2_renderer_quality_gate import (
    DEFAULT_CONFIG,
    QUALITY_LEVELS,
    load_gate_config,
)


CLASS_NAMES = ("pedestrian", "vehicle")
SEGMENTATION_CLASS_NAMES = {0: "background", 1: "vehicle", 2: "pedestrian"}
PREDICTION_CLASS_ALIASES = {"person": "pedestrian", "pedestrian": "pedestrian", "vehicle": "vehicle"}


def canonical_prediction_class(value: object) -> str:
    label = str(value).strip().lower()
    if label not in PREDICTION_CLASS_ALIASES:
        raise ValueError(f"unexpected detector class label: {value!r}")
    return PREDICTION_CLASS_ALIASES[label]


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


def _read_single_csv(directory: Path, pattern: str) -> pd.DataFrame:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {directory}, found {len(matches)}")
    return pd.read_csv(matches[0])


def _launch_manifest_for(batch_root: Path) -> dict:
    candidates = []
    for path in batch_root.parent.glob("*.launch.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if Path(str(payload.get("batch_root", ""))).resolve() == batch_root.resolve():
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one detached launch manifest for {batch_root}, found {len(candidates)}"
        )
    path, payload = candidates[0]
    payload["_path"] = str(path)
    return payload


def _load_batch(batch_root: Path, expected_quality: str) -> dict:
    batch_root = Path(batch_root).resolve()
    if not (batch_root / "COMPLETED.json").is_file():
        raise ValueError(f"renderer stage is incomplete: {batch_root}")
    if (batch_root / "FAILED.json").exists():
        raise ValueError(f"renderer stage has a failure sentinel: {batch_root}")
    config_path = batch_root / "resolved_integration_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream)
    renderer = resolved.get("renderer_quality")
    if not isinstance(renderer, Mapping) or renderer.get(
        "declared_quality_level"
    ) != expected_quality:
        raise ValueError(f"resolved renderer declaration is not {expected_quality}")
    if renderer.get("required_server_launch_flag") != f"-quality-level={expected_quality}":
        raise ValueError("resolved renderer launch flag differs from declaration")
    launch = _launch_manifest_for(batch_root)
    if launch.get("declared_renderer_quality") != expected_quality:
        raise ValueError("detached launch and resolved renderer declarations differ")
    manifest = json.loads((batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or len(manifest.get("trajectories", [])) != 2:
        raise ValueError("renderer stage batch manifest is not a complete matched pair")
    if any(item.get("status") != "complete" for item in manifest["trajectories"]):
        raise ValueError("renderer stage contains an incomplete trajectory")
    return {
        "root": batch_root,
        "quality": expected_quality,
        "resolved_config": resolved,
        "resolved_config_path": config_path,
        "launch": launch,
        "manifest": manifest,
    }


def _load_role_tables(batch: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    detections = []
    truth = []
    metrics = []
    quality = str(batch["quality"])
    root = Path(batch["root"])
    scenario_by_trajectory = {
        str(item["trajectory_id"]): str(item["scenario_role"])
        for item in batch["manifest"]["trajectories"]
    }
    for trajectory_id, scenario_role in scenario_by_trajectory.items():
        for source_role in ("helper", "recipient"):
            role_root = root / trajectory_id / source_role
            metric = _read_single_csv(role_root / "streams", "*_metrics.csv")
            metric = metric.sort_values("frame_id").reset_index(drop=True)
            metric["frame_ordinal"] = np.arange(len(metric), dtype=int)
            metric["quality"] = quality
            metric["trajectory_id"] = trajectory_id
            metric["scenario_role"] = scenario_role
            metric["source_role"] = source_role
            metrics.append(metric)
            frame_ordinals = metric[["frame_id", "frame_ordinal"]].drop_duplicates()

            detected = pd.read_csv(role_root / "runtime" / "final_detections.csv")
            detected["class_name"] = detected["class_name"].map(
                canonical_prediction_class
            )
            detected = detected.merge(frame_ordinals, on="frame_id", how="left", validate="many_to_one")
            ego = pd.read_csv(role_root / "runtime" / "ego_states.csv")
            ego = ego[["frame_id", "world_x", "world_y", "world_z"]].rename(
                columns={
                    "world_x": "ego_x",
                    "world_y": "ego_y",
                    "world_z": "ego_z",
                }
            )
            detected = detected.merge(ego, on="frame_id", how="left", validate="many_to_one")
            detected["distance_m"] = np.sqrt(
                (detected["world_x"] - detected["ego_x"]) ** 2
                + (detected["world_y"] - detected["ego_y"]) ** 2
                + (detected["world_z"] - detected["ego_z"]) ** 2
            )
            detected["quality"] = quality
            detected["scenario_role"] = scenario_role
            detections.append(detected)

            actual = _read_single_csv(role_root / "evaluation_truth", "*_object_ground_truth.csv")
            actual = actual.merge(frame_ordinals, on="frame_id", how="left", validate="many_to_one")
            actual["quality"] = quality
            actual["trajectory_id"] = trajectory_id
            actual["scenario_role"] = scenario_role
            actual["source_role"] = source_role
            truth.append(actual)
    return {
        "detections": pd.concat(detections, ignore_index=True),
        "truth": pd.concat(truth, ignore_index=True),
        "metrics": pd.concat(metrics, ignore_index=True),
    }


def _one_to_one_counts(
    truth: pd.DataFrame, predictions: pd.DataFrame, gate_m: float
) -> tuple[int, int, int]:
    if truth.empty:
        return 0, int(len(predictions)), 0
    if predictions.empty:
        return 0, 0, int(len(truth))
    truth_xyz = truth[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=float)
    pred_xyz = predictions[["world_x", "world_y", "world_z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(truth_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
    candidates = [
        (float(distances[i, j]), i, j)
        for i in range(distances.shape[0])
        for j in range(distances.shape[1])
        if float(distances[i, j]) <= float(gate_m)
    ]
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for _distance, truth_index, prediction_index in sorted(candidates):
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
    true_positives = len(matched_truth)
    return (
        true_positives,
        int(len(predictions) - true_positives),
        int(len(truth) - true_positives),
    )


def detection_curve(
    tables: Mapping[str, pd.DataFrame], gate_config: Mapping[str, object]
) -> pd.DataFrame:
    metrics = gate_config["metrics"]
    thresholds = [float(value) for value in metrics["postdecoder_score_thresholds"]]
    match_gate = float(metrics["localization_match_gate_m"])
    ranges = {
        str(key): float(value)
        for key, value in metrics["near_range_m_by_class"].items()
    }
    truth = tables["truth"]
    predictions = tables["detections"]
    rows = []
    group_fields = ["quality", "trajectory_id", "scenario_role", "source_role"]
    groups = tables["metrics"][group_fields].drop_duplicates().to_dict("records")
    for group in groups:
        truth_group = truth
        pred_group = predictions
        for field, value in group.items():
            truth_group = truth_group[truth_group[field] == value]
            pred_group = pred_group[pred_group[field] == value]
        frame_ordinals = sorted(
            int(value)
            for value in tables["metrics"].loc[
                np.logical_and.reduce(
                    [tables["metrics"][field] == value for field, value in group.items()]
                ),
                "frame_ordinal",
            ].unique()
        )
        for class_name in CLASS_NAMES:
            range_m = ranges[class_name]
            class_truth = truth_group[
                (truth_group["class_name"] == class_name)
                & (truth_group["in_camera_frustum"].astype(int) == 1)
                & (truth_group["distance_m"].astype(float) <= range_m)
            ]
            class_predictions = pred_group[
                (pred_group["class_name"] == class_name)
                & (pred_group["distance_m"].astype(float) <= range_m)
            ]
            for threshold in thresholds:
                tp = fp = fn = 0
                thresholded = class_predictions[
                    class_predictions["score"].astype(float) >= threshold
                ]
                for ordinal in frame_ordinals:
                    observed = class_truth[class_truth["frame_ordinal"] == ordinal]
                    emitted = thresholded[thresholded["frame_ordinal"] == ordinal]
                    frame_tp, frame_fp, frame_fn = _one_to_one_counts(
                        observed, emitted, match_gate
                    )
                    tp += frame_tp
                    fp += frame_fp
                    fn += frame_fn
                precision = float(tp / (tp + fp)) if tp + fp else float("nan")
                recall = float(tp / (tp + fn)) if tp + fn else float("nan")
                f1 = (
                    float(2.0 * precision * recall / (precision + recall))
                    if math.isfinite(precision)
                    and math.isfinite(recall)
                    and precision + recall > 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        **group,
                        "class_name": class_name,
                        "near_range_m": range_m,
                        "score_threshold": threshold,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "metric_scope": "postdecoder_candidates_score_gte_0.05",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        curve.groupby(["quality", "class_name", "near_range_m", "score_threshold"], as_index=False)[
            ["true_positives", "false_positives", "false_negatives"]
        ]
        .sum()
        .sort_values(["quality", "class_name", "score_threshold"])
    )
    grouped["precision"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_positives"]
    ).replace(0, np.nan)
    grouped["recall"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_negatives"]
    ).replace(0, np.nan)
    grouped["f1"] = 2.0 * grouped["precision"] * grouped["recall"] / (
        grouped["precision"] + grouped["recall"]
    ).replace(0, np.nan)
    grouped["metric_scope"] = "postdecoder_candidates_score_gte_0.05"
    return grouped


def capture_diagnostics(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metrics = tables["metrics"]
    for keys, frame in metrics.groupby(
        ["quality", "trajectory_id", "scenario_role", "source_role"], sort=True
    ):
        quality, trajectory_id, scenario_role, source_role = keys
        received = frame["result_received"].astype(str).str.lower().isin(("true", "1"))
        rows.append(
            {
                "quality": quality,
                "trajectory_id": trajectory_id,
                "scenario_role": scenario_role,
                "source_role": source_role,
                "metric_frames": int(len(frame)),
                "result_received_frames": int(received.sum()),
                "dropped_required_frames": int((~received).sum()),
                "radar_projected_points_median": float(
                    pd.to_numeric(frame["radar_projected_points"], errors="coerce").median()
                ),
                "radar_projected_points_p05": float(
                    pd.to_numeric(frame["radar_projected_points"], errors="coerce").quantile(0.05)
                ),
                "gt_camera_available_frames": int(
                    pd.to_numeric(frame["gt_camera_available"], errors="coerce").fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _logit_files(batch_root: Path) -> dict[tuple[str, str], list[Path]]:
    result = {}
    manifest = json.loads((batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["trajectories"]:
        trajectory_id = str(item["trajectory_id"])
        scenario_role = str(item["scenario_role"])
        for source_role in ("helper", "recipient"):
            result[(scenario_role, source_role)] = sorted(
                (batch_root / trajectory_id / source_role / "retained_inputs").glob(
                    "frame_*_logits.npz"
                )
            )
    return result


def segmentation_stability(
    low_root: Path, epic_root: Path, stride: int
) -> pd.DataFrame:
    low = _logit_files(low_root)
    epic = _logit_files(epic_root)
    if set(low) != set(epic):
        raise ValueError("Low/Epic retained-logit role sets differ")
    rows = []
    for key in sorted(low):
        low_files = low[key]
        epic_files = epic[key]
        if len(low_files) != len(epic_files):
            raise ValueError(f"Low/Epic retained-logit counts differ for {key}")
        for ordinal in range(0, len(low_files), int(stride)):
            with np.load(low_files[ordinal]) as low_npz, np.load(epic_files[ordinal]) as epic_npz:
                low_mask = np.argmax(low_npz["out"], axis=1)[0]
                epic_mask = np.argmax(epic_npz["out"], axis=1)[0]
            if low_mask.shape != epic_mask.shape:
                raise ValueError("Low/Epic segmentation output shapes differ")
            agreement = float(np.mean(low_mask == epic_mask))
            for class_id, class_name in SEGMENTATION_CLASS_NAMES.items():
                low_class = low_mask == class_id
                epic_class = epic_mask == class_id
                union = int(np.logical_or(low_class, epic_class).sum())
                intersection = int(np.logical_and(low_class, epic_class).sum())
                rows.append(
                    {
                        "scenario_role": key[0],
                        "source_role": key[1],
                        "retained_frame_ordinal": ordinal,
                        "class_id": class_id,
                        "class_name": class_name,
                        "argmax_pixel_agreement": agreement,
                        "paired_class_iou": (
                            float(intersection / union) if union else float("nan")
                        ),
                        "low_class_fraction": float(np.mean(low_class)),
                        "epic_class_fraction": float(np.mean(epic_class)),
                        "metric_scope": "paired_prediction_stability_not_accuracy",
                    }
                )
    if not rows:
        raise ValueError("no retained segmentation logits were available")
    return pd.DataFrame(rows)


def plot_pr_curve(aggregate: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, class_name in zip(axes, CLASS_NAMES):
        subset = aggregate[aggregate["class_name"] == class_name]
        for quality, frame in subset.groupby("quality"):
            axis.plot(
                frame["recall"],
                frame["precision"],
                marker="o",
                label=str(quality),
            )
        axis.set_title(class_name.capitalize())
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0.0, 1.02)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Renderer gate: postdecoder PR (candidate floor = 0.05)")
    return figure


def analyze(
    low_root: Path, epic_root: Path, output_dir: Path, config_path: Path
) -> dict:
    gate = load_gate_config(config_path)
    batches = {
        "Low": _load_batch(low_root, "Low"),
        "Epic": _load_batch(epic_root, "Epic"),
    }
    tables = {quality: _load_role_tables(batch) for quality, batch in batches.items()}
    combined = {
        key: pd.concat([tables[quality][key] for quality in QUALITY_LEVELS], ignore_index=True)
        for key in ("detections", "truth", "metrics")
    }
    curve_by_role = detection_curve(combined, gate)
    curve = aggregate_curve(curve_by_role)
    diagnostics = capture_diagnostics(combined)
    segmentation = segmentation_stability(
        Path(low_root).resolve(),
        Path(epic_root).resolve(),
        int(gate["metrics"]["segmentation_comparison_stride_frames"]),
    )

    primary = float(gate["metrics"]["primary_score_threshold"])
    primary_rows = curve[np.isclose(curve["score_threshold"], primary)].copy()
    recall_by_quality = {
        (str(row.quality), str(row.class_name)): float(row.recall)
        for row in primary_rows.itertuples()
    }
    recall_shifts = {
        class_name: 100.0
        * (recall_by_quality[("Epic", class_name)] - recall_by_quality[("Low", class_name)])
        for class_name in CLASS_NAMES
    }
    max_shift = float(gate["metrics"]["maximum_absolute_recall_shift_pp"])
    recall_pass = all(abs(value) <= max_shift for value in recall_shifts.values())

    radar_by_quality = diagnostics.groupby("quality")[
        "radar_projected_points_median"
    ].median()
    radar_relative_delta = abs(float(radar_by_quality["Epic"]) - float(radar_by_quality["Low"])) / max(
        float(radar_by_quality["Low"]), 1.0
    )
    radar_pass = radar_relative_delta <= float(
        gate["metrics"]["radar_median_relative_tolerance"]
    )
    dropped_total = int(diagnostics["dropped_required_frames"].sum())
    structural_pass = dropped_total <= int(
        gate["metrics"]["maximum_dropped_required_frames"]
    ) and bool((diagnostics["metric_frames"] == 120).all())

    agreement = float(segmentation["argmax_pixel_agreement"].median())
    class_iou = (
        segmentation.groupby("class_name")["paired_class_iou"].median().dropna().to_dict()
    )
    segmentation_pass = agreement >= float(
        gate["metrics"]["minimum_segmentation_argmax_agreement"]
    ) and all(
        float(value) >= float(gate["metrics"]["minimum_segmentation_class_iou"])
        for value in class_iou.values()
    )
    passed = structural_pass and radar_pass and recall_pass and segmentation_pass
    verdict = "PASS_FAIL_FAST" if passed else "HOLD_RENDERER_CONTRACT_REVIEW"

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    curve_by_role.to_csv(output_dir / "pr_curve_by_role.csv", index=False)
    curve.to_csv(output_dir / "pr_curve.csv", index=False)
    primary_rows.to_csv(output_dir / "primary_threshold_metrics.csv", index=False)
    diagnostics.to_csv(output_dir / "capture_diagnostics.csv", index=False)
    segmentation.to_csv(output_dir / "segmentation_prediction_stability.csv", index=False)
    figure = plot_pr_curve(curve)
    figure.savefig(output_dir / "renderer_pr_curve.png", dpi=300)
    figure.savefig(output_dir / "renderer_pr_curve.pdf")
    plt.close(figure)

    summary = {
        "schema": "scenesense.phase2_renderer_quality_decision.v1",
        "verdict": verdict,
        "interpretation": (
            "Epic is pinned as the primary corpus renderer; Low remains a stress stratum."
            if passed
            else "Corpus collection remains on hold pending renderer/training-contract review."
        ),
        "confirmatory_claim_allowed": False,
        "comparison_id": gate["comparison"]["comparison_id"],
        "quality_verification": gate["comparison"]["quality_verification"],
        "quality_empirically_introspected": False,
        "limitations": [
            "One matched positive/benign pair is a fail-fast gate, not confirmatory evidence.",
            "PR is conditional on postdecoder candidates emitted at score >= 0.05.",
            "Segmentation is paired prediction stability because semantic GT is disabled in the causal pilot; it is not accuracy.",
            "Shared-GPU wall time and inference latency are diagnostic, not citable performance measurements.",
        ],
        "checks": {
            "structural_capture": {
                "pass": structural_pass,
                "dropped_required_frames": dropped_total,
            },
            "radar_invariance": {
                "pass": radar_pass,
                "low_projected_median": float(radar_by_quality["Low"]),
                "epic_projected_median": float(radar_by_quality["Epic"]),
                "relative_delta": radar_relative_delta,
            },
            "near_range_recall_stability": {
                "pass": recall_pass,
                "epic_minus_low_pp": recall_shifts,
                "maximum_absolute_shift_pp": max_shift,
            },
            "segmentation_prediction_stability": {
                "pass": segmentation_pass,
                "median_argmax_pixel_agreement": agreement,
                "median_paired_class_iou": class_iou,
            },
        },
        "inputs": {
            quality.lower(): {
                "batch_root": str(batch["root"]),
                "resolved_config": str(batch["resolved_config_path"]),
                "resolved_config_sha256": _sha256(batch["resolved_config_path"]),
                "launch_manifest": batch["launch"]["_path"],
            }
            for quality, batch in batches.items()
        },
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary = _finite_or_none(summary)
    with (output_dir / "RENDERER_QUALITY_DECISION.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    artifact_manifest = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    with (output_dir / "artifact_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(artifact_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--epic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = analyze(args.low_root, args.epic_root, args.output_dir, args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
