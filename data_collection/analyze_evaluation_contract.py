#!/usr/bin/env python3
"""Audit object-detection acceptance on an immutable policy corpus.

The analysis deliberately operates on saved CSV artifacts only.  It selects a
per-class score threshold on whole validation trajectories, evaluates that
threshold on held-out test trajectories, reports recall by ground-truth range,
and compares the corpus radar density with a retained-input reference run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = (
    REPO_ROOT
    / "data_collection/experiments/policy_corpus_advisor_rich_v4"
    / "20260813_014501_full"
)
DEFAULT_REFERENCE_RUN = (
    REPO_ROOT
    / "data_collection/experiments/pedestrian_on_contract_diagnostic_v1"
    / "20260812_213148_smoke/runs/pedestrian_on_contract_smoke_v1"
)
CLASSES = ("pedestrian", "vehicle")
MATCH_GATE_M = 5.0
MAX_RANGE_M = 25.0
NEAR_RANGE_M = 12.0
RANGE_EDGES_M = (0.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0)
THRESHOLDS = tuple(np.round(np.arange(0.05, 1.0001, 0.005), 3))


@dataclass(frozen=True)
class RunData:
    episode_id: str
    scenario_family: str
    split: str
    run_dir: Path
    gt: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class FrameMatchData:
    gt_xy: np.ndarray
    prediction_xy: np.ndarray
    prediction_scores: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single(run_dir: Path, pattern: str) -> Path:
    matches = sorted((run_dir / "streams").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {run_dir}, found {len(matches)}")
    return matches[0]


def _normalise_class(value: object) -> str:
    value = str(value).strip().lower()
    if value in {"person", "pedestrian", "walker"}:
        return "pedestrian"
    return "vehicle" if "vehicle" in value else value


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _load_run(item: Mapping[str, object]) -> RunData:
    run_dir = Path(str(item["run_dir"])).resolve()
    gt = pd.read_csv(_single(run_dir, "*_object_ground_truth.csv"))
    predictions = pd.read_csv(_single(run_dir, "*_object_predictions.csv"))
    gt["class_name"] = gt["class_name"].map(_normalise_class)
    predictions["class_name"] = predictions["class_name"].map(_normalise_class)
    for frame in (gt, predictions):
        for column in ("frame_id", "world_x", "world_y", "distance_m"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if {"origin_x", "origin_y"}.issubset(gt.columns):
        gt["world_x"] = pd.to_numeric(gt["origin_x"], errors="coerce")
        gt["world_y"] = pd.to_numeric(gt["origin_y"], errors="coerce")
    predictions["score"] = pd.to_numeric(predictions["score"], errors="coerce")
    gt = gt[
        gt["class_name"].isin(CLASSES)
        & _truthy(gt["in_camera_frustum"])
        & gt["distance_m"].between(0.0, MAX_RANGE_M, inclusive="both")
        & gt[["world_x", "world_y"]].notna().all(axis=1)
    ].copy()
    predictions = predictions[
        predictions["class_name"].isin(CLASSES)
        & predictions["distance_m"].between(0.0, MAX_RANGE_M, inclusive="both")
        & predictions[["world_x", "world_y", "score"]].notna().all(axis=1)
    ].copy()
    return RunData(
        episode_id=str(item["episode_id"]),
        scenario_family=str(item["scenario_family"]),
        split=str(item["split"]),
        run_dir=run_dir,
        gt=gt,
        predictions=predictions,
    )


def load_runs(batch_dir: Path, excluded: Iterable[str]) -> Tuple[List[RunData], list]:
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded_ids = set(excluded)
    items = list(manifest["runs"])
    runs = [_load_run(item) for item in items if str(item["episode_id"]) not in excluded_ids]
    found_ids = {str(item["episode_id"]) for item in items}
    missing = excluded_ids - found_ids
    if missing:
        raise ValueError(f"excluded episodes not found in batch: {sorted(missing)}")
    return runs, items


def _match_frame(
    gt_frame: pd.DataFrame, prediction_frame: pd.DataFrame
) -> Tuple[List[int], List[int]]:
    if gt_frame.empty or prediction_frame.empty:
        return [], []
    gt_xy = gt_frame[["world_x", "world_y"]].to_numpy(dtype=float)
    pred_xy = prediction_frame[["world_x", "world_y"]].to_numpy(dtype=float)
    distances = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
    pairs = np.argwhere(distances <= MATCH_GATE_M)
    ordered = sorted(
        ((float(distances[g, p]), int(g), int(p)) for g, p in pairs),
        key=lambda row: row[0],
    )
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    for _distance, gt_index, prediction_index in ordered:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
    gt_labels = list(gt_frame.index)
    prediction_labels = list(prediction_frame.index)
    return (
        [int(gt_labels[index]) for index in sorted(used_gt)],
        [int(prediction_labels[index]) for index in sorted(used_predictions)],
    )


def match_run(
    run: RunData, class_name: str, threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    gt = run.gt[run.gt["class_name"] == class_name]
    predictions = run.predictions[
        (run.predictions["class_name"] == class_name)
        & (run.predictions["score"] >= threshold - 1e-12)
    ]
    matched_gt: List[int] = []
    matched_predictions: List[int] = []
    frame_ids = sorted(set(gt["frame_id"].astype(int)) | set(predictions["frame_id"].astype(int)))
    for frame_id in frame_ids:
        frame_gt = gt[gt["frame_id"] == frame_id]
        frame_predictions = predictions[predictions["frame_id"] == frame_id]
        gt_indices, prediction_indices = _match_frame(frame_gt, frame_predictions)
        matched_gt.extend(gt_indices)
        matched_predictions.extend(prediction_indices)
    return gt, predictions, matched_gt, matched_predictions


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def prepare_match_cache(
    runs: Sequence[RunData],
) -> Dict[Tuple[str, str], Tuple[FrameMatchData, ...]]:
    cache: Dict[Tuple[str, str], Tuple[FrameMatchData, ...]] = {}
    for run in runs:
        for class_name in CLASSES:
            gt = run.gt[run.gt["class_name"] == class_name]
            predictions = run.predictions[run.predictions["class_name"] == class_name]
            frames: List[FrameMatchData] = []
            frame_ids = sorted(
                set(gt["frame_id"].astype(int)) | set(predictions["frame_id"].astype(int))
            )
            for frame_id in frame_ids:
                gt_frame = gt[gt["frame_id"] == frame_id]
                prediction_frame = predictions[predictions["frame_id"] == frame_id]
                frames.append(
                    FrameMatchData(
                        gt_xy=gt_frame[["world_x", "world_y"]].to_numpy(dtype=float),
                        prediction_xy=prediction_frame[["world_x", "world_y"]].to_numpy(
                            dtype=float
                        ),
                        prediction_scores=prediction_frame["score"].to_numpy(dtype=float),
                    )
                )
            cache[(run.episode_id, class_name)] = tuple(frames)
    return cache


def _match_array_count(gt_xy: np.ndarray, prediction_xy: np.ndarray) -> int:
    if not len(gt_xy) or not len(prediction_xy):
        return 0
    distances = np.linalg.norm(gt_xy[:, None, :] - prediction_xy[None, :, :], axis=2)
    ordered = sorted(
        (
            (float(distances[gt_index, prediction_index]), int(gt_index), int(prediction_index))
            for gt_index, prediction_index in np.argwhere(distances <= MATCH_GATE_M)
        ),
        key=lambda row: row[0],
    )
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    for _distance, gt_index, prediction_index in ordered:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
    return len(used_gt)


def score_threshold(
    runs: Sequence[RunData],
    cache: Mapping[Tuple[str, str], Tuple[FrameMatchData, ...]],
    class_name: str,
    threshold: float,
    split: str,
) -> Dict[str, object]:
    selected = runs if split == "all" else [run for run in runs if run.split == split]
    gt_count = prediction_count = true_positives = 0
    for run in selected:
        for frame in cache[(run.episode_id, class_name)]:
            selected_predictions = frame.prediction_scores >= threshold - 1e-12
            gt_count += len(frame.gt_xy)
            prediction_count += int(np.count_nonzero(selected_predictions))
            true_positives += _match_array_count(
                frame.gt_xy, frame.prediction_xy[selected_predictions]
            )
    precision = _safe_ratio(true_positives, prediction_count)
    recall = _safe_ratio(true_positives, gt_count)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0.0
        else 0.0
    )
    return {
        "split": split,
        "class_name": class_name,
        "score_threshold": threshold,
        "eligible_gt_rows": gt_count,
        "prediction_rows": prediction_count,
        "true_positives": true_positives,
        "false_positives": prediction_count - true_positives,
        "false_negatives": gt_count - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_pr_curves(
    runs: Sequence[RunData],
    cache: Mapping[Tuple[str, str], Tuple[FrameMatchData, ...]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in ("validation", "test"):
        for class_name in CLASSES:
            for threshold in THRESHOLDS:
                rows.append(
                    score_threshold(runs, cache, class_name, threshold, split)
                )
    return pd.DataFrame(rows)


def choose_validation_thresholds(pr_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_name in CLASSES:
        candidates = pr_curves[
            (pr_curves["split"] == "validation")
            & (pr_curves["class_name"] == class_name)
        ].copy()
        best_f1 = float(candidates["f1"].max())
        # Prefer the stricter threshold when a rounded grid produces an exact tie.
        chosen = candidates[np.isclose(candidates["f1"], best_f1, atol=1e-12)].sort_values(
            "score_threshold", ascending=False
        ).iloc[0]
        rows.append(
            {
                **chosen.to_dict(),
                "selection_rule": "maximum validation F1; tie -> higher threshold",
                "at_decoder_floor": bool(
                    math.isclose(float(chosen["score_threshold"]), min(THRESHOLDS))
                ),
            }
        )
    return pd.DataFrame(rows)


def _range_label(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:g}"


def build_range_coverage(
    runs: Sequence[RunData], chosen: Mapping[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_rows: List[pd.DataFrame] = []
    for run in runs:
        for class_name in CLASSES:
            for contract, threshold in (
                ("decoder_floor_0.05", 0.05),
                ("inherited_0.20", 0.20),
                ("validation_f1", float(chosen[class_name])),
            ):
                gt, _predictions, matched_gt, _matched_predictions = match_run(
                    run, class_name, threshold
                )
                marked = gt.copy()
                marked["matched"] = marked.index.isin(matched_gt)
                marked["episode_id"] = run.episode_id
                marked["scenario_family"] = run.scenario_family
                marked["split"] = run.split
                marked["contract"] = contract
                marked["score_threshold"] = threshold
                gt_rows.append(marked)
    marked = pd.concat(gt_rows, ignore_index=True)
    labels = [
        _range_label(RANGE_EDGES_M[index], RANGE_EDGES_M[index + 1])
        for index in range(len(RANGE_EDGES_M) - 1)
    ]
    marked["range_bin_m"] = pd.cut(
        marked["distance_m"],
        bins=RANGE_EDGES_M,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    by_range_rows: List[Dict[str, object]] = []
    cumulative_rows: List[Dict[str, object]] = []
    per_run_rows: List[Dict[str, object]] = []
    for split in ("validation", "test", "all"):
        split_frame = marked if split == "all" else marked[marked["split"] == split]
        for (contract, class_name), class_frame in split_frame.groupby(
            ["contract", "class_name"], observed=True
        ):
            threshold = float(class_frame["score_threshold"].iloc[0])
            for label in labels:
                group = class_frame[class_frame["range_bin_m"] == label]
                matched_count = int(group["matched"].sum())
                by_range_rows.append(
                    {
                        "split": split,
                        "contract": contract,
                        "class_name": class_name,
                        "score_threshold": threshold,
                        "range_bin_m": label,
                        "eligible_gt_rows": len(group),
                        "matched_gt_rows": matched_count,
                        "recall": _safe_ratio(matched_count, len(group)),
                    }
                )
            for upper in RANGE_EDGES_M[1:]:
                group = class_frame[class_frame["distance_m"] <= upper + 1e-12]
                matched_count = int(group["matched"].sum())
                cumulative_rows.append(
                    {
                        "split": split,
                        "contract": contract,
                        "class_name": class_name,
                        "score_threshold": threshold,
                        "range_upper_m": upper,
                        "eligible_gt_rows": len(group),
                        "matched_gt_rows": matched_count,
                        "recall": _safe_ratio(matched_count, len(group)),
                    }
                )
    for (episode_id, contract, class_name), group in marked.groupby(
        ["episode_id", "contract", "class_name"], observed=True
    ):
        near = group[group["distance_m"] <= NEAR_RANGE_M]
        per_run_rows.append(
            {
                "episode_id": episode_id,
                "scenario_family": str(group["scenario_family"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "contract": contract,
                "class_name": class_name,
                "score_threshold": float(group["score_threshold"].iloc[0]),
                "eligible_gt_rows_le12m": len(near),
                "matched_gt_rows_le12m": int(near["matched"].sum()),
                "recall_le12m": _safe_ratio(int(near["matched"].sum()), len(near)),
                "eligible_gt_rows_le25m": len(group),
                "matched_gt_rows_le25m": int(group["matched"].sum()),
                "recall_le25m": _safe_ratio(int(group["matched"].sum()), len(group)),
            }
        )
    return pd.DataFrame(by_range_rows), pd.DataFrame(cumulative_rows), pd.DataFrame(per_run_rows)


def _numeric_summary(values: pd.Series) -> Dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "frames": int(len(values)),
        "mean": float(values.mean()),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build_radar_audit(
    runs: Sequence[RunData], reference_run: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows: List[Dict[str, object]] = []
    corpus_values: List[pd.Series] = []
    for run in runs:
        metrics_path = _single(run.run_dir, "*_metrics.csv")
        config_path = next((run.run_dir / "manifests").glob("*_resolved_config.json"))
        metrics = pd.read_csv(metrics_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        values = pd.to_numeric(metrics["radar_projected_points"], errors="coerce").dropna()
        corpus_values.append(values)
        per_run_rows.append(
            {
                "episode_id": run.episode_id,
                "scenario_family": run.scenario_family,
                "split": run.split,
                **_numeric_summary(values),
                "sensor_detection_hz": config.get("fps"),
                "world_control_hz": config.get("world_tick_hz"),
                "sensor_every_tick": config.get("sensor_every_tick"),
                "radar_points_per_second": config.get("radar_points_per_second"),
                "radar_sensor_tick_s": 1.0 / float(config["fps"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256(metrics_path),
                "resolved_config_sha256": _sha256(config_path),
            }
        )
    reference_metrics_path = _single(reference_run, "*_metrics.csv")
    reference_config_path = next((reference_run / "manifests").glob("*_resolved_config.json"))
    reference_metrics = pd.read_csv(reference_metrics_path)
    reference_config = json.loads(reference_config_path.read_text(encoding="utf-8"))
    corpus = pd.concat(corpus_values, ignore_index=True)
    reference = pd.to_numeric(
        reference_metrics["radar_projected_points"], errors="coerce"
    ).dropna()
    summary_rows = [
        {
            "source": (
                f"{runs[0].run_dir.parents[2].name}_included_runs"
                if runs else "policy_corpus_included_runs"
            ),
            **_numeric_summary(corpus),
            "sensor_detection_hz": float(per_run_rows[0]["sensor_detection_hz"]),
            "world_control_hz": float(per_run_rows[0]["world_control_hz"]),
            "sensor_every_tick": bool(per_run_rows[0]["sensor_every_tick"]),
            "radar_points_per_second": int(per_run_rows[0]["radar_points_per_second"]),
        },
        {
            "source": "retained_on_contract_reference",
            **_numeric_summary(reference),
            "sensor_detection_hz": float(reference_config["fps"]),
            "world_control_hz": float(
                reference_config.get("world_tick_hz") or reference_config["fps"]
            ),
            "sensor_every_tick": bool(reference_config["sensor_every_tick"]),
            "radar_points_per_second": int(reference_config["radar_points_per_second"]),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    reference_median = float(summary.loc[summary["source"] == "retained_on_contract_reference", "median"].iloc[0])
    summary["median_fraction_of_reference"] = summary["median"] / reference_median
    return pd.DataFrame(per_run_rows), summary


def build_input_inventory(runs: Sequence[RunData]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        for artifact_type, pattern in (
            ("ground_truth", "*_object_ground_truth.csv"),
            ("predictions", "*_object_predictions.csv"),
            ("metrics", "*_metrics.csv"),
        ):
            path = _single(run.run_dir, pattern)
            rows.append(
                {
                    "episode_id": run.episode_id,
                    "scenario_family": run.scenario_family,
                    "split": run.split,
                    "artifact_type": artifact_type,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return pd.DataFrame(rows)


def plot_pr_curves(
    curves: pd.DataFrame, chosen: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    colours = {"validation": "#0072B2", "test": "#D55E00"}
    for axis, class_name in zip(axes, CLASSES):
        for split in ("validation", "test"):
            rows = curves[
                (curves["class_name"] == class_name) & (curves["split"] == split)
            ].sort_values("recall")
            axis.plot(
                rows["recall"], rows["precision"], label=split, color=colours[split]
            )
        selected_threshold = float(
            chosen.loc[chosen["class_name"] == class_name, "score_threshold"].iloc[0]
        )
        point = curves[
            (curves["class_name"] == class_name)
            & (curves["split"] == "validation")
            & np.isclose(curves["score_threshold"], selected_threshold)
        ].iloc[0]
        axis.scatter(
            [point["recall"]], [point["precision"]], marker="*", s=120, color="#009E73",
            zorder=3, label=f"chosen t={selected_threshold:.3f}"
        )
        axis.set_title(class_name.capitalize())
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0.0, 1.02)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("Trajectory-held-out precision-recall curves (5 m center match, GT/pred <=25 m)")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"precision_recall_by_class.{suffix}", dpi=300)
    plt.close(fig)


def plot_range_coverage(by_range: pd.DataFrame, output_dir: Path) -> None:
    selected = by_range[
        (by_range["contract"] == "validation_f1")
        & (by_range["split"].isin(["validation", "test"]))
    ].copy()
    labels = list(dict.fromkeys(selected["range_bin_m"].astype(str)))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, class_name in zip(axes, CLASSES):
        for offset, (split, colour) in enumerate(
            (("validation", "#0072B2"), ("test", "#D55E00"))
        ):
            rows = selected[
                (selected["class_name"] == class_name) & (selected["split"] == split)
            ].set_index("range_bin_m").reindex(labels)
            axis.bar(
                x + (offset - 0.5) * 0.38,
                100.0 * rows["recall"].to_numpy(dtype=float),
                width=0.38,
                label=split,
                color=colour,
            )
        axis.axvline(2.5, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(class_name.capitalize())
        axis.set_xlabel("Ground-truth range (m)")
        axis.set_ylabel("Recall (%)")
        axis.set_xticks(x, labels, rotation=25)
        axis.set_ylim(0.0, 105.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("Coverage versus range at validation-selected per-class thresholds")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"coverage_vs_range_by_class.{suffix}", dpi=300)
    plt.close(fig)


def write_manifest(
    output_dir: Path,
    batch_dir: Path,
    reference_run: Path,
    excluded: Sequence[str],
    chosen: pd.DataFrame,
    radar_summary: pd.DataFrame,
) -> None:
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    thresholds = {
        str(row.class_name): float(row.score_threshold)
        for row in chosen.itertuples(index=False)
    }
    corpus_ratio = float(
        radar_summary.loc[
            radar_summary["source"].str.startswith("policy_corpus"),
            "median_fraction_of_reference",
        ].iloc[0]
    )
    manifest = {
        "schema": "evaluation_contract_desk_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RECOLLECT_REQUIRED_SENSOR_CONTRACT_DRIFT",
        "analysis_only": True,
        "carla_launched": False,
        "batch_dir": str(batch_dir),
        "batch_manifest_sha256": _sha256(batch_dir / "batch_manifest.json"),
        "reference_run": str(reference_run),
        "excluded_episode_ids": list(excluded),
        "matching_contract": {
            "representation": "actor-origin XY center",
            "class_aware": True,
            "association_gate_m": MATCH_GATE_M,
            "gt_eligibility": "in_camera_frustum and 0 <= distance_m <= 25",
            "prediction_scope": "0 <= decoded distance_m <= 25",
        },
        "threshold_selection": {
            "split": "validation whole trajectories only",
            "rule": "maximum F1 on 0.005 score grid; tie -> higher threshold",
            "selected": thresholds,
        },
        "radar_density_median_fraction_of_reference": corpus_ratio,
        "artifacts": artifacts,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def run(
    batch_dir: Path,
    reference_run: Path,
    output_dir: Path,
    excluded: Sequence[str],
) -> Path:
    batch_dir = batch_dir.resolve()
    reference_run = reference_run.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {output_dir}")
    output_dir.mkdir(parents=True)
    runs, _items = load_runs(batch_dir, excluded)
    match_cache = prepare_match_cache(runs)
    curves = build_pr_curves(runs, match_cache)
    chosen = choose_validation_thresholds(curves)
    chosen_map = {
        str(row.class_name): float(row.score_threshold)
        for row in chosen.itertuples(index=False)
    }
    by_range, cumulative, per_run_coverage = build_range_coverage(runs, chosen_map)
    radar_per_run, radar_summary = build_radar_audit(runs, reference_run)
    input_inventory = build_input_inventory(runs)
    selected_metrics = pd.DataFrame(
        [
            score_threshold(runs, match_cache, class_name, threshold, split)
            for class_name, threshold in chosen_map.items()
            for split in ("validation", "test", "all")
        ]
    )

    curves.to_csv(output_dir / "precision_recall_curve.csv", index=False)
    chosen.to_csv(output_dir / "validation_selected_thresholds.csv", index=False)
    selected_metrics.to_csv(output_dir / "selected_threshold_metrics.csv", index=False)
    by_range.to_csv(output_dir / "coverage_by_range.csv", index=False)
    cumulative.to_csv(output_dir / "coverage_cumulative_range.csv", index=False)
    per_run_coverage.to_csv(output_dir / "coverage_per_run.csv", index=False)
    radar_per_run.to_csv(output_dir / "radar_density_per_run.csv", index=False)
    radar_summary.to_csv(output_dir / "radar_density_summary.csv", index=False)
    input_inventory.to_csv(output_dir / "input_inventory.csv", index=False)
    plot_pr_curves(curves, chosen, output_dir)
    plot_range_coverage(by_range, output_dir)
    write_manifest(output_dir, batch_dir, reference_run, excluded, chosen, radar_summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-episode", action="append", default=[]
    )
    args = parser.parse_args()
    print(run(args.batch_dir, args.reference_run, args.output_dir, args.exclude_episode))


if __name__ == "__main__":
    main()
