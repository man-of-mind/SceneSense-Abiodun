#!/usr/bin/env python3
"""Analyze frozen offline lists, select on validation, and finalize one-shot test evidence."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
AB = HERE.parents[3]
import sys
if str(AB / "pole_lraspp_multimodal_fusion") not in sys.path:
    sys.path.insert(0, str(AB / "pole_lraspp_multimodal_fusion"))
from pole_lraspp_multimodal_fusion.object_targets import greedy_match_predictions  # noqa: E402


CLASSES = ("vehicle", "person")
CANDIDATE_RADIUS = {"baseline": 0.0, "world_suppression_1m": 1.0, "world_suppression_2m": 2.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_freeze() -> None:
    frozen = json.loads((HERE / "PREINFERENCE_FREEZE.json").read_text(encoding="utf-8"))
    for record in frozen["frozen_files"]:
        if sha256(HERE / record["path"]) != record["sha256"]:
            raise AssertionError(f"Frozen preregistration changed: {record['path']}")


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_candidate(predictions: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    radius = CANDIDATE_RADIUS[candidate]
    if radius <= 0.0:
        return list(predictions)
    kept: list[dict[str, Any]] = []
    ordered = sorted(predictions, key=lambda item: (-float(item["score"]), int(item["source_order"])))
    for prediction in ordered:
        suppress = False
        for accepted in kept:
            if str(accepted["class_name"]) != str(prediction["class_name"]):
                continue
            distance = math.hypot(
                float(accepted["world_x"]) - float(prediction["world_x"]),
                float(accepted["world_y"]) - float(prediction["world_y"]),
            )
            if distance <= radius:
                suppress = True
                break
        if not suppress:
            kept.append(prediction)
    return kept


def frame_metrics(record: dict[str, Any], candidate: str) -> dict[str, Any]:
    predictions = apply_candidate(record["predictions"], candidate)
    ground_truth = record["gt"]
    matches = greedy_match_predictions(predictions, ground_truth, max_distance_m=5.0, class_aware=True)
    matched_predictions = {int(prediction_index) for prediction_index, _, _ in matches}
    matched_gt = {int(gt_index) for _, gt_index, _ in matches}
    output: dict[str, Any] = {
        "split": record["split"],
        "profile_id": record["profile_id"],
        "family": record["family"],
        "role": record["role"],
        "normal_gate": bool(record["normal_gate"]),
        "candidate": candidate,
        "sample_id": record["sample_id"],
        "scenario_group": record["scenario_group"],
        "payload_bytes": int(record["payload_bytes"]),
        "decoder_latency_ms": float(record["decoder_latency_ms"]),
        "prediction_count": len(predictions),
        "gt_count": len(ground_truth),
    }
    for class_name in CLASSES:
        prefix = "veh" if class_name == "vehicle" else "ped"
        output[f"{prefix}_tp"] = 0
        output[f"{prefix}_fp"] = 0
        output[f"{prefix}_fn"] = 0
        output[f"{prefix}_error_sum"] = 0.0
        output[f"{prefix}_error_sq_sum"] = 0.0
        output[f"{prefix}_prediction_count"] = sum(str(item["class_name"]) == class_name for item in predictions)
        output[f"{prefix}_gt_count"] = sum(str(item["class_name"]) == class_name for item in ground_truth)
    for prediction_index, gt_index, distance in matches:
        class_name = str(ground_truth[int(gt_index)]["class_name"])
        prefix = "veh" if class_name == "vehicle" else "ped"
        output[f"{prefix}_tp"] += 1
        output[f"{prefix}_error_sum"] += float(distance)
        output[f"{prefix}_error_sq_sum"] += float(distance) ** 2
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index not in matched_predictions:
            prefix = "veh" if str(prediction["class_name"]) == "vehicle" else "ped"
            output[f"{prefix}_fp"] += 1
    for gt_index, gt in enumerate(ground_truth):
        if gt_index not in matched_gt:
            prefix = "veh" if str(gt["class_name"]) == "vehicle" else "ped"
            output[f"{prefix}_fn"] += 1
    output["all_fp"] = output["veh_fp"] + output["ped_fp"]
    return output


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def aggregate_metrics(frame_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["split", "profile_id", "family", "role", "normal_gate", "candidate"]
    for keys, group in frame_table.groupby(grouping, sort=True, dropna=False):
        base = dict(zip(grouping, keys))
        all_fp = float(group["all_fp"].sum())
        for class_name, prefix in (("vehicle", "veh"), ("person", "ped")):
            tp = float(group[f"{prefix}_tp"].sum())
            fp = float(group[f"{prefix}_fp"].sum())
            fn = float(group[f"{prefix}_fn"].sum())
            precision = ratio(tp, tp + fp)
            recall = ratio(tp, tp + fn)
            rows.append({
                **base,
                "class_name": class_name,
                "frames": int(len(group)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "prediction_count": int(group[f"{prefix}_prediction_count"].sum()),
                "gt_count": int(group[f"{prefix}_gt_count"].sum()),
                "precision": precision,
                "recall": recall,
                "f1": ratio(2.0 * precision * recall, precision + recall),
                "xy_mae_m": ratio(float(group[f"{prefix}_error_sum"].sum()), tp),
                "xy_rmse_m": math.sqrt(ratio(float(group[f"{prefix}_error_sq_sum"].sum()), tp)),
                "fp_per_frame_class": ratio(fp, len(group)),
                "fp_per_frame_all": ratio(all_fp, len(group)),
            })
    return pd.DataFrame(rows)


def secondary_metrics(raw_by_profile: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_id, records in sorted(raw_by_profile.items()):
        confusion = np.asarray([record["segmentation_confusion"] for record in records], dtype=np.int64).sum(axis=0).reshape(3, 3)
        ious: list[float] = []
        for class_index in range(3):
            tp = int(confusion[class_index, class_index])
            fp = int(confusion[:, class_index].sum()) - tp
            fn = int(confusion[class_index, :].sum()) - tp
            ious.append(ratio(tp, tp + fp + fn))
        payloads = np.asarray([record["payload_bytes"] for record in records], dtype=float)
        rows.append({
            "split": records[0]["split"],
            "profile_id": profile_id,
            "family": records[0]["family"],
            "role": records[0]["role"],
            "normal_gate": bool(records[0]["normal_gate"]),
            "frames": len(records),
            "payload_bytes_mean": float(payloads.mean()),
            "payload_bytes_p50": float(np.quantile(payloads, 0.50)),
            "payload_bytes_p95": float(np.quantile(payloads, 0.95)),
            "miou": float(np.nanmean(ious)),
            "iou_background": ious[0],
            "iou_vehicle": ious[1],
            "iou_person": ious[2],
            "decoder_invariance": "mathematically unchanged across retained-list candidates",
        })
    return pd.DataFrame(rows)


def metric_from_totals(totals: np.ndarray, metric: str) -> float:
    # Columns: veh tp/fp/fn/err, ped tp/fp/fn/err, all fp, frames.
    values = totals.sum(axis=0)
    if metric == "vehicle_precision":
        return ratio(values[0], values[0] + values[1])
    if metric == "vehicle_recall":
        return ratio(values[0], values[0] + values[2])
    if metric == "vehicle_xy_mae_m":
        return ratio(values[3], values[0])
    if metric == "person_precision":
        return ratio(values[4], values[4] + values[5])
    if metric == "person_recall":
        return ratio(values[4], values[4] + values[6])
    if metric == "person_xy_mae_m":
        return ratio(values[7], values[4])
    if metric == "fp_per_frame_all":
        return ratio(values[8], values[9])
    raise KeyError(metric)


def numeric_table(group: pd.DataFrame, unit_column: str) -> tuple[np.ndarray, list[str]]:
    columns = [
        "veh_tp", "veh_fp", "veh_fn", "veh_error_sum",
        "ped_tp", "ped_fp", "ped_fn", "ped_error_sum", "all_fp",
    ]
    working = group[[unit_column, *columns]].copy()
    working["frames"] = 1
    aggregated = working.groupby(unit_column, sort=True)[[*columns, "frames"]].sum()
    return aggregated.to_numpy(dtype=float), list(aggregated.index.astype(str))


def paired_bootstrap(frame_table: pd.DataFrame, candidate: str, config: dict[str, Any]) -> pd.DataFrame:
    repetitions = int(config["uncertainty"]["bootstrap_replicates"])
    seed = int(config["uncertainty"]["seed"])
    metrics = (
        "vehicle_precision", "vehicle_recall", "vehicle_xy_mae_m",
        "person_precision", "person_recall", "person_xy_mae_m", "fp_per_frame_all",
    )
    scopes: list[tuple[str, pd.DataFrame]] = []
    for profile_id, group in frame_table.groupby("profile_id", sort=True):
        scopes.append((str(profile_id), group))
    scopes.append(("pooled_normal", frame_table.loc[frame_table.normal_gate.astype(bool)].copy()))
    output: list[dict[str, Any]] = []
    for scope_index, (scope, scope_table) in enumerate(scopes):
        baseline = scope_table.loc[scope_table.candidate == "baseline"]
        selected = scope_table.loc[scope_table.candidate == candidate]
        if baseline.empty or selected.empty:
            raise AssertionError(f"Missing paired rows for {scope}/{candidate}")
        for unit_index, (unit_name, unit_column) in enumerate((("frame", "sample_id"), ("scenario", "scenario_group"))):
            baseline_array, baseline_units = numeric_table(baseline, unit_column)
            selected_array, selected_units = numeric_table(selected, unit_column)
            if baseline_units != selected_units:
                raise AssertionError(f"Paired {unit_name} units differ for {scope}/{candidate}")
            rng = np.random.default_rng(seed + scope_index * 1009 + unit_index * 9173 + int(CANDIDATE_RADIUS[candidate] * 100))
            indices = rng.integers(0, len(baseline_units), size=(repetitions, len(baseline_units)))
            for metric in metrics:
                baseline_point = metric_from_totals(baseline_array, metric)
                candidate_point = metric_from_totals(selected_array, metric)
                deltas = np.empty(repetitions, dtype=float)
                for repetition in range(repetitions):
                    sample = indices[repetition]
                    deltas[repetition] = metric_from_totals(selected_array[sample], metric) - metric_from_totals(baseline_array[sample], metric)
                output.append({
                    "split": str(scope_table["split"].iloc[0]),
                    "scope": scope,
                    "candidate": candidate,
                    "resampling_unit": unit_name,
                    "metric": metric,
                    "baseline": baseline_point,
                    "candidate_value": candidate_point,
                    "observed_delta": candidate_point - baseline_point,
                    "delta_ci95_low": float(np.nanquantile(deltas, 0.025)),
                    "delta_ci95_high": float(np.nanquantile(deltas, 0.975)),
                    "bootstrap_replicates": repetitions,
                    "independent_units": len(baseline_units),
                })
    return pd.DataFrame(output)


def latency_evidence(
    split_name: str,
    raw_by_profile: dict[str, list[dict[str, Any]]],
    candidates: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gpu_rows: list[dict[str, Any]] = []
    list_rows: list[dict[str, Any]] = []
    repeats = int(config["latency"]["list_repeats_per_frame_candidate"])
    for profile_id, records in sorted(raw_by_profile.items()):
        for frame_index, record in enumerate(records):
            gpu_rows.append({
                "split": split_name,
                "profile_id": profile_id,
                "sample_id": record["sample_id"],
                "frame_index": frame_index,
                "warmup_excluded": frame_index < int(config["latency"]["warmup_frames_excluded_per_profile"]),
                "gpu_decoder_end_to_end_ms": float(record["decoder_latency_ms"]),
            })
            for candidate in candidates:
                predictions = record["predictions"]
                apply_candidate(predictions, candidate)
                for repeat in range(repeats):
                    started = time.perf_counter_ns()
                    apply_candidate(predictions, candidate)
                    elapsed = time.perf_counter_ns() - started
                    list_rows.append({
                        "split": split_name,
                        "profile_id": profile_id,
                        "sample_id": record["sample_id"],
                        "candidate": candidate,
                        "repeat": repeat,
                        "incremental_list_latency_ms": elapsed / 1e6,
                    })
    gpu = pd.DataFrame(gpu_rows)
    list_detail = pd.DataFrame(list_rows)
    summary: list[dict[str, Any]] = []
    for profile_id, group in gpu.loc[~gpu.warmup_excluded].groupby("profile_id", sort=True):
        values = group.gpu_decoder_end_to_end_ms.to_numpy(dtype=float)
        summary.append({
            "split": split_name,
            "profile_id": profile_id,
            "candidate": "decoder_envelope_shared",
            "latency_scope": "gpu_decoder_end_to_end",
            "samples": len(values),
            "p50_ms": float(np.quantile(values, 0.50)),
            "p90_ms": float(np.quantile(values, 0.90)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "max_ms": float(values.max()),
        })
    for (profile_id, candidate), group in list_detail.groupby(["profile_id", "candidate"], sort=True):
        values = group.incremental_list_latency_ms.to_numpy(dtype=float)
        summary.append({
            "split": split_name,
            "profile_id": profile_id,
            "candidate": candidate,
            "latency_scope": "incremental_retained_list",
            "samples": len(values),
            "p50_ms": float(np.quantile(values, 0.50)),
            "p90_ms": float(np.quantile(values, 0.90)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "max_ms": float(values.max()),
        })
    return gpu, list_detail, pd.DataFrame(summary)


def floor_checks(metrics: pd.DataFrame, candidate: str, config: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    floors = config["normal_service_floors"]
    output: list[dict[str, Any]] = []
    normal = metrics.loc[(metrics.candidate == candidate) & metrics.normal_gate.astype(bool)]
    expected_profiles = {str(item["profile_id"]) for item in config["profiles"] if item["normal_gate"]}
    if set(normal.profile_id.astype(str)) != expected_profiles:
        raise AssertionError(f"Normal profile coverage mismatch for {candidate}")
    for profile_id, group in normal.groupby("profile_id", sort=True):
        vehicle = group.loc[group.class_name == "vehicle"].iloc[0]
        person = group.loc[group.class_name == "person"].iloc[0]
        checks = {
            "vehicle_recall": (float(vehicle.recall), ">=", float(floors["vehicle_recall_min"])),
            "person_recall": (float(person.recall), ">=", float(floors["person_recall_min"])),
            "vehicle_precision": (float(vehicle.precision), ">=", float(floors["vehicle_precision_min"])),
            "person_precision": (float(person.precision), ">=", float(floors["person_precision_min"])),
            "vehicle_xy_mae_m": (float(vehicle.xy_mae_m), "<=", float(floors["vehicle_xy_mae_m_max"])),
            "person_xy_mae_m": (float(person.xy_mae_m), "<=", float(floors["person_xy_mae_m_max"])),
            "fp_per_frame_all": (float(vehicle.fp_per_frame_all), "<=", float(floors["fp_per_frame_max"])),
        }
        for metric, (value, operator, floor) in checks.items():
            passed = value >= floor if operator == ">=" else value <= floor
            output.append({
                "profile_id": profile_id,
                "candidate": candidate,
                "metric": metric,
                "value": value,
                "operator": operator,
                "floor": floor,
                "pass": bool(passed),
            })
    return all(record["pass"] for record in output), output


def material_checks(uncertainty: pd.DataFrame, candidate: str, config: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rules = config["material_improvement"]
    pooled = uncertainty.loc[(uncertainty.scope == "pooled_normal") & (uncertainty.candidate == candidate)]
    output: list[dict[str, Any]] = []
    point_requirements = {
        "vehicle_precision": (">=", float(rules["vehicle_precision_point_delta_min"])),
        "person_precision": (">=", float(rules["person_precision_point_delta_min"])),
        "fp_per_frame_all": ("<=", float(rules["fp_per_frame_point_delta_max"])),
    }
    for metric, (operator, threshold) in point_requirements.items():
        point_values = pooled.loc[pooled.metric == metric, "observed_delta"].unique()
        if len(point_values) != 1:
            raise AssertionError(f"Missing pooled point estimate: {candidate}/{metric}")
        value = float(point_values[0])
        passed = value >= threshold if operator == ">=" else value <= threshold
        output.append({"candidate": candidate, "check": f"point_{metric}", "value": value, "operator": operator, "threshold": threshold, "pass": bool(passed)})
    for unit in ("frame", "scenario"):
        for metric in ("vehicle_precision", "person_precision"):
            row = pooled.loc[(pooled.resampling_unit == unit) & (pooled.metric == metric)].iloc[0]
            value = float(row.delta_ci95_low)
            output.append({"candidate": candidate, "check": f"{unit}_{metric}_ci_low", "value": value, "operator": ">", "threshold": 0.0, "pass": bool(value > 0.0)})
        row = pooled.loc[(pooled.resampling_unit == unit) & (pooled.metric == "fp_per_frame_all")].iloc[0]
        value = float(row.delta_ci95_high)
        output.append({"candidate": candidate, "check": f"{unit}_fp_per_frame_ci_high", "value": value, "operator": "<", "threshold": 0.0, "pass": bool(value < 0.0)})
    return all(record["pass"] for record in output), output


def load_raw(split_name: str, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    expected_ids = (HERE / "split_identifiers" / f"{split_name}.txt").read_text(encoding="utf-8").splitlines()
    expected_set = set(expected_ids)
    result: dict[str, list[dict[str, Any]]] = {}
    for profile in config["profiles"]:
        profile_id = str(profile["profile_id"])
        path = HERE / "raw_predictions" / split_name / f"{profile_id}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw predictions: {path}")
        records = load_json_lines(path)
        ids = [str(record["sample_id"]) for record in records]
        if ids != expected_ids or set(ids) != expected_set:
            raise AssertionError(f"Raw {split_name}/{profile_id} identifiers differ from frozen order/set")
        if any(str(record["profile_id"]) != profile_id for record in records):
            raise AssertionError(f"Profile identity contamination in {path}")
        result[profile_id] = records
    return result


def analyze_split(split_name: str) -> dict[str, Any]:
    verify_freeze()
    config = json.loads((HERE / "resolved_config.json").read_text(encoding="utf-8"))
    if not (HERE / f"{split_name.upper()}_INFERENCE_COMPLETE.json").is_file():
        raise RuntimeError(f"{split_name} inference has not completed")
    raw = load_raw(split_name, config)
    if split_name == "val":
        candidates = ["baseline", "world_suppression_1m", "world_suppression_2m"]
    else:
        selection = json.loads((HERE / "frozen_selection.json").read_text(encoding="utf-8"))
        selected = str(selection.get("selected_candidate"))
        if selected not in ("world_suppression_1m", "world_suppression_2m"):
            raise RuntimeError("Test analysis requires an eligible frozen setting")
        candidates = ["baseline", selected]

    frame_rows: list[dict[str, Any]] = []
    for profile_id, records in raw.items():
        for record in records:
            for candidate in candidates:
                frame_rows.append(frame_metrics(record, candidate))
    frames = pd.DataFrame(frame_rows)
    metrics = aggregate_metrics(frames)
    secondary = secondary_metrics(raw)
    uncertainty_parts = [paired_bootstrap(frames, candidate, config) for candidate in candidates if candidate != "baseline"]
    uncertainty = pd.concat(uncertainty_parts, ignore_index=True)
    gpu_latency, list_latency, latency_summary = latency_evidence(split_name, raw, candidates, config)

    frames.to_csv(HERE / f"paired_per_frame_{split_name}.csv.gz", index=False, compression="gzip")
    metrics.to_csv(HERE / f"per_profile_class_metrics_{split_name}.csv", index=False)
    secondary.to_csv(HERE / f"secondary_payload_segmentation_{split_name}.csv", index=False)
    uncertainty.to_csv(HERE / f"paired_bootstrap_{split_name}.csv", index=False)
    gpu_latency.to_csv(HERE / f"latency_gpu_samples_{split_name}.csv.gz", index=False, compression="gzip")
    list_latency.to_csv(HERE / f"latency_list_samples_{split_name}.csv.gz", index=False, compression="gzip")
    latency_summary.to_csv(HERE / f"latency_summary_{split_name}.csv", index=False)

    decision: dict[str, Any] = {
        "split": split_name,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": {},
    }
    checks_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate == "baseline":
            continue
        floor_pass, floor_records = floor_checks(metrics, candidate, config)
        material_pass, material_records = material_checks(uncertainty, candidate, config)
        checks_rows.extend({**record, "check_type": "absolute_floor"} for record in floor_records)
        checks_rows.extend({**record, "check_type": "material_paired"} for record in material_records)
        decision["candidates"][candidate] = {
            "all_normal_profile_floors_pass": floor_pass,
            "material_paired_improvement_pass": material_pass,
            "eligible": floor_pass and material_pass,
        }
    pd.DataFrame(checks_rows).to_csv(HERE / f"decision_checks_{split_name}.csv", index=False)

    if split_name == "val":
        selected_candidate: str | None = None
        for candidate in ("world_suppression_1m", "world_suppression_2m"):
            if decision["candidates"][candidate]["eligible"]:
                selected_candidate = candidate
                break
        decision["selected_candidate"] = selected_candidate
        decision["selection_rule_result"] = (
            "least_radius_eligible_candidate" if selected_candidate else "no_conservative_candidate_eligible"
        )
        decision["validation_evidence_hashes"] = {
            path.name: sha256(path)
            for path in (
                HERE / "paired_per_frame_val.csv.gz",
                HERE / "per_profile_class_metrics_val.csv",
                HERE / "paired_bootstrap_val.csv",
                HERE / "decision_checks_val.csv",
                HERE / "latency_summary_val.csv",
            )
        }
        atomic_json(HERE / "frozen_selection.json", decision)
    else:
        selected_candidate = candidates[1]
        decision["selected_candidate"] = selected_candidate
        decision["confirmatory_pass"] = bool(decision["candidates"][selected_candidate]["eligible"])
        atomic_json(HERE / "test_confirmation.json", decision)
    return decision


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def pilot_proposal() -> str:
    return """# Bounded AE64 retraining pilot proposal (not started)

This proposal is review-only. It creates no checkpoint and authorizes no training.

## Bounds

- Central family: AE64 only; initialize from the frozen v1 AE64 checkpoint.
- One preregistered training configuration, at most three fixed seeds, the existing
  train/validation split, and one final untouched-test evaluation after seed/config
  selection on validation.
- Candidate outputs must use a new versioned directory and filename; `best.pt` and
  every v1 checkpoint remain read-only.
- Freeze checkpoint, config, training/evaluator/decoder source, dependency, split,
  and dataset-manifest hashes for each candidate.

## Service-aware selection and promotion rule

First apply a feasibility filter. Vehicle and person precision must be superior
to AE64-v1 with paired 95% lower bounds above zero. Vehicle/person recall, each
class's world-XY MAE, secondary segmentation, payload, and compute must be
non-inferior within preregistered margins; no requirement says every scalar must
strictly improve. Suggested validation margins are recall delta >= -0.01,
XY-MAE delta <= +0.05 m, mIoU delta >= -0.01, payload P95 <= +2%, GPU decoder
P95 <= +5%, and total inference P95 <= +5%.

Among feasible candidates only, rank by the frozen service score:
`0.35*vehicle_precision_gain + 0.20*person_precision_gain +
0.15*minimum_recall_margin + 0.15*minimum_XY_margin +
0.05*segmentation_margin + 0.05*payload_margin + 0.05*compute_margin`,
with each term normalized by its preregistered margin. Use seed-stability and
lower compute as tie-breaks. A later promotion still requires human review and
regeneration of affected detection/localization catalog rows.
"""


def finalize() -> dict[str, Any]:
    verify_freeze()
    config = json.loads((HERE / "resolved_config.json").read_text(encoding="utf-8"))
    selection_path = HERE / "frozen_selection.json"
    if not selection_path.is_file():
        conclusion = "INSUFFICIENT_EVIDENCE"
        selected = None
        reason = "Validation analysis/selection is missing."
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection.get("selected_candidate")
        if selected is None:
            conclusion = "RETRAINING_PILOT_JUSTIFIED"
            reason = "Neither 1 m nor 2 m met every frozen validation floor and paired material-improvement requirement."
        elif not (HERE / "test_confirmation.json").is_file():
            conclusion = "INSUFFICIENT_EVIDENCE"
            reason = "A setting was frozen on validation, but one-shot untouched-test confirmation is missing."
        else:
            confirmation = json.loads((HERE / "test_confirmation.json").read_text(encoding="utf-8"))
            if confirmation.get("selected_candidate") != selected:
                conclusion = "INSUFFICIENT_EVIDENCE"
                reason = "Test candidate differs from the validation-frozen selection."
            elif confirmation.get("confirmatory_pass"):
                conclusion = "CONSERVATIVE_POSTPROCESSING_READY_FOR_PROMOTION_REVIEW"
                reason = "The least-radius eligible validation setting passed the one-shot test floors and paired material-improvement rule."
            else:
                conclusion = "RETRAINING_PILOT_JUSTIFIED"
                reason = "The validation-frozen conservative setting failed the one-shot confirmatory test rule."

    split_integrity = json.loads((HERE / "split_integrity.json").read_text(encoding="utf-8"))
    validation_metrics = pd.read_csv(HERE / "per_profile_class_metrics_val.csv") if (HERE / "per_profile_class_metrics_val.csv").is_file() else pd.DataFrame()
    test_metrics = pd.read_csv(HERE / "per_profile_class_metrics_test.csv") if (HERE / "per_profile_class_metrics_test.csv").is_file() else pd.DataFrame()
    validation_uncertainty = pd.read_csv(HERE / "paired_bootstrap_val.csv") if (HERE / "paired_bootstrap_val.csv").is_file() else pd.DataFrame()
    test_uncertainty = pd.read_csv(HERE / "paired_bootstrap_test.csv") if (HERE / "paired_bootstrap_test.csv").is_file() else pd.DataFrame()
    latency_val = pd.read_csv(HERE / "latency_summary_val.csv") if (HERE / "latency_summary_val.csv").is_file() else pd.DataFrame()
    latency_test = pd.read_csv(HERE / "latency_summary_test.csv") if (HERE / "latency_summary_test.csv").is_file() else pd.DataFrame()

    report_sections = [
        "# Conservative decoder validation report",
        "",
        f"Audit: `{config['audit_id']}`  ",
        f"Conclusion: **`{conclusion}`**  ",
        f"Frozen global setting: **`{selected or 'none'}`**",
        "",
        "## Outcome",
        "",
        reason,
        "This is an offline promotion-review result, not deployment approval. No production file, checkpoint, CARLA/OAI path, registry, runtime, launcher, controller, or map server was changed.",
        "",
        "## Provenance and split integrity",
        "",
        f"The original manifest contains {split_integrity['split_counts']['train']:,} train, {split_integrity['split_counts']['val']:,} validation, and {split_integrity['split_counts']['test']:,} test identifiers. Pairwise overlap is zero, and all four checkpoint families' saved split files exactly match the manifest in identity and order. The frozen validation identifier SHA-256 is `{split_integrity['frozen_identifier_files']['val']['sha256']}`.",
        "",
        "All source/checkpoint/data hashes are in `input_hash_manifest.json`; preregistration completion precedes both inference markers.",
        "",
        "## Validation per-profile/class metrics",
        "",
    ]
    metric_columns = ["profile_id", "candidate", "class_name", "precision", "recall", "xy_mae_m", "fp_per_frame_all"]
    if not validation_metrics.empty:
        report_sections.append(markdown_table(validation_metrics[metric_columns], metric_columns))
    else:
        report_sections.append("Validation metrics unavailable.")
    report_sections.extend(["", "## Validation pooled-normal paired uncertainty", ""])
    uncertainty_columns = ["candidate", "resampling_unit", "metric", "observed_delta", "delta_ci95_low", "delta_ci95_high"]
    if not validation_uncertainty.empty:
        focus = validation_uncertainty.loc[(validation_uncertainty.scope == "pooled_normal") & validation_uncertainty.metric.isin(["vehicle_precision", "person_precision", "vehicle_recall", "person_recall", "vehicle_xy_mae_m", "person_xy_mae_m", "fp_per_frame_all"])]
        report_sections.append(markdown_table(focus[uncertainty_columns], uncertainty_columns))
    else:
        report_sections.append("Validation uncertainty unavailable.")
    report_sections.extend(["", "## One-shot untouched-test evidence", ""])
    if not test_metrics.empty:
        report_sections.append(markdown_table(test_metrics[metric_columns], metric_columns))
    else:
        report_sections.append("No test metrics were produced because no candidate was eligible or required evidence was unavailable.")
    if not test_uncertainty.empty:
        report_sections.extend(["", "Pooled-normal paired test deltas:", ""])
        focus = test_uncertainty.loc[(test_uncertainty.scope == "pooled_normal") & test_uncertainty.metric.isin(["vehicle_precision", "person_precision", "vehicle_recall", "person_recall", "vehicle_xy_mae_m", "person_xy_mae_m", "fp_per_frame_all"])]
        report_sections.append(markdown_table(focus[uncertainty_columns], uncertainty_columns))
    report_sections.extend([
        "",
        "## Latency",
        "",
        "GPU end-to-end decoder latency is CUDA-synchronized feature-to-retained-list time. Incremental list latency measures only the predicted-only suppression on an already-retained list. These scopes are intentionally not combined.",
        "",
    ])
    latency_columns = ["split", "profile_id", "candidate", "latency_scope", "samples", "p50_ms", "p95_ms", "max_ms"]
    latency = pd.concat([latency_val, latency_test], ignore_index=True) if not latency_val.empty or not latency_test.empty else pd.DataFrame()
    if not latency.empty:
        report_sections.append(markdown_table(latency[latency_columns], latency_columns, digits=6))
    else:
        report_sections.append("Latency evidence unavailable.")
    report_sections.extend([
        "",
        "## Secondary evidence and catalog implication",
        "",
        "Retained-list suppression does not alter feature serialization, payload, or segmentation logits. Their per-profile measurements are preserved in `secondary_payload_segmentation_{val,test}.csv` and versioned separately. If a later review promotes this decoder, all affected detection/localization catalog rows must be regenerated globally; payload/segmentation rows may remain only with this invariance link.",
        "",
        "## Decision boundary",
        "",
        f"**`{conclusion}`**. {reason} Human review remains mandatory; deployment is neither performed nor approved here.",
        "",
    ])
    (HERE / "REPORT.md").write_text("\n".join(report_sections), encoding="utf-8")
    if conclusion == "RETRAINING_PILOT_JUSTIFIED":
        (HERE / "RETRAINING_PILOT_PROPOSAL.md").write_text(pilot_proposal(), encoding="utf-8")

    review = {
        "status": "REVIEW_REQUIRED",
        "conclusion": conclusion,
        "selected_global_decoder_setting": selected,
        "deployment_approved": False,
        "production_edits_made": False,
        "training_started": False,
        "required_review": [
            "Review frozen split/provenance hashes and paired evidence.",
            "Review all normal profiles, not only pooled values.",
            "If promotion is authorized later, regenerate every affected detection/localization catalog row.",
        ],
    }
    atomic_json(HERE / "REVIEW_REQUIRED", review)
    atomic_json(HERE / "RESULTS_SUMMARY.json", {
        "audit_id": config["audit_id"],
        "conclusion": conclusion,
        "reason": reason,
        "selected_candidate": selected,
        "deployment_approved": False,
        "review_required": True,
    })

    excluded = {"manifest.json", "IMMUTABLE.sha256"}
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in HERE.rglob("*") if item.is_file() and item.name not in excluded and not item.name.endswith(".partial")):
        files.append({
            "path": str(path.relative_to(HERE)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "audit_id": config["audit_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "selected_candidate": selected,
        "file_count_excluding_manifest_and_immutable_seal": len(files),
        "files": files,
    }
    atomic_json(HERE / "manifest.json", manifest)
    seal_records = files + [{"path": "manifest.json", "size_bytes": (HERE / "manifest.json").stat().st_size, "sha256": sha256(HERE / "manifest.json")}]
    (HERE / "IMMUTABLE.sha256").write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in sorted(seal_records, key=lambda item: item["path"])),
        encoding="utf-8",
    )
    return {"conclusion": conclusion, "selected_candidate": selected, "reason": reason, "files": len(files) + 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validation", "test", "finalize"))
    arguments = parser.parse_args()
    if arguments.command == "validation":
        result = analyze_split("val")
    elif arguments.command == "test":
        result = analyze_split("test")
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
