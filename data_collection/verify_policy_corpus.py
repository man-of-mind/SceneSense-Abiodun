#!/usr/bin/env python3
"""Apply the locked §5 gates to a completed policy-corpus batch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_agent.policy.catalog import flatten_actions, load_profile_catalog
from rl_agent.policy.channel import ChannelProcess, ChannelSurface
from rl_agent.policy.config import load_config
from rl_agent.policy.env import SurrogateEnv
from rl_agent.policy.replay import (
    _greedy_prediction_matches,
    _normalize_class,
    _prediction_score_mask,
    discover_trace_registry,
    load_trace_episode,
)
from data_collection import analyze_evaluation_contract as evaluation_contract

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _single_csv(run_dir: Path, suffix: str) -> Path:
    matches = sorted((run_dir / "streams").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} under {run_dir}, found {len(matches)}")
    return matches[0]


def _actor_speeds(frame: pd.DataFrame) -> pd.Series:
    rows: List[pd.Series] = []
    for _actor_id, group in frame.groupby("actor_id"):
        group = group.sort_values("carla_timestamp")
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        speed = np.hypot(dx, dy) / dt
        speed[(dt <= 0.0) | ~np.isfinite(speed)] = np.nan
        rows.append(pd.Series(speed.to_numpy(), index=group.index))
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows).sort_index()


def _safe_pct(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _quantile(values: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else None


def _longest_true_dwell(mask: Sequence[bool], timestamps: Sequence[float]) -> float:
    values = np.asarray(mask, dtype=bool)
    times = np.asarray(timestamps, dtype=float)
    if not len(values):
        return 0.0
    finite_deltas = np.diff(times)
    finite_deltas = finite_deltas[np.isfinite(finite_deltas) & (finite_deltas > 0.0)]
    nominal_dt = float(np.median(finite_deltas)) if len(finite_deltas) else 0.0
    longest = 0.0
    start: int | None = None
    previous: int | None = None
    for index, enabled in enumerate(values):
        contiguous = (
            previous is None
            or nominal_dt <= 0.0
            or times[index] - times[previous] <= 1.5 * nominal_dt
        )
        if enabled:
            if start is None or not contiguous:
                if start is not None and previous is not None:
                    longest = max(longest, times[previous] - times[start] + nominal_dt)
                start = index
        elif start is not None:
            assert previous is not None
            longest = max(longest, times[previous] - times[start] + nominal_dt)
            start = None
        previous = index
    if start is not None and previous is not None:
        longest = max(longest, times[previous] - times[start] + nominal_dt)
    return float(longest)


def _inspect_run(
    run_spec: Mapping[str, object],
    run_dir: Path,
    collection_config: Mapping[str, object],
    prediction_score_min_by_class: Mapping[str, float] | None = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    verify = collection_config["verification"]
    timing = collection_config["timing_gate"]
    range_m = float(verify["headline_range_m"])
    gt_path = _single_csv(run_dir, "_object_ground_truth.csv")
    prediction_path = _single_csv(run_dir, "_object_predictions.csv")
    metrics_path = _single_csv(run_dir, "_metrics.csv")
    gt = pd.read_csv(gt_path)
    predictions = pd.read_csv(prediction_path)
    metrics = pd.read_csv(metrics_path)
    failures: List[str] = []

    required_gt = {
        "frame_id", "carla_timestamp", "actor_id", "class_name", "origin_x", "origin_y",
        "distance_m", "in_camera_frustum", "height_m", "width_m",
    }
    missing = sorted(required_gt - set(gt.columns))
    if missing:
        raise ValueError(f"missing GT fields in {gt_path}: {missing}")
    numeric_columns = ["origin_x", "origin_y", "distance_m"]
    numeric = gt[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        failures.append("nonfinite_gt_position_or_distance")
    if (numeric["distance_m"] < 0.0).any():
        failures.append("negative_distance")
    coordinate_max = float(verify["town_coordinate_abs_max_m"])
    if (numeric[["origin_x", "origin_y"]].abs() > coordinate_max).any().any():
        failures.append("position_outside_declared_town_bounds")
    frustum_values = set(gt["in_camera_frustum"].astype(str).str.lower())
    if not frustum_values.issubset({"0", "1", "false", "true"}):
        failures.append("invalid_frustum_flag")

    gt_match = gt[_truthy(gt["in_camera_frustum"])].copy()
    gt_match = gt_match[pd.to_numeric(gt_match["distance_m"], errors="coerce") <= range_m].copy()
    gt_match["class_name"] = gt_match["class_name"].map(_normalize_class)
    gt_match["world_x"] = pd.to_numeric(gt_match["origin_x"], errors="coerce")
    gt_match["world_y"] = pd.to_numeric(gt_match["origin_y"], errors="coerce")
    predictions = predictions.copy()
    predictions["class_name"] = predictions["class_name"].map(_normalize_class)
    prediction_score_min_by_class = dict(prediction_score_min_by_class or {})
    predictions = predictions[
        _prediction_score_mask(
            predictions,
            {
                "prediction_score_min": float(verify["prediction_score_min"]),
                "prediction_score_min_by_class": prediction_score_min_by_class,
            },
        )
    ].copy()
    matches = _greedy_prediction_matches(
        gt_match, predictions, float(verify["association_gate_m"])
    )
    coverage_rows: List[Dict[str, object]] = []
    for class_name in sorted(set(gt_match["class_name"]) | {"vehicle", "pedestrian"}):
        class_gt = gt_match[gt_match["class_name"] == class_name]
        class_matches = (
            matches[matches["class_name"] == class_name]
            if not matches.empty
            else pd.DataFrame()
        )
        gt_frames = int(class_gt["frame_id"].nunique())
        matched_frames = 0
        if len(class_matches):
            matched_keys = set(
                zip(class_matches["actor_id"].astype(int), class_matches["timestamp"].astype(float))
            )
            matched_frames = int(
                class_gt[
                    class_gt.apply(
                        lambda row: (int(row["actor_id"]), float(row["carla_timestamp"]))
                        in matched_keys,
                        axis=1,
                    )
                ]["frame_id"].nunique()
            )
        coverage_rows.append(
            {
                "episode_id": str(run_spec["episode_id"]),
                "scenario_family": str(run_spec["scenario_family"]),
                "split": str(run_spec["split"]),
                "class_name": class_name,
                "eligible_gt_rows": int(len(class_gt)),
                "matched_rows": int(len(class_matches)),
                "object_row_coverage_pct": _safe_pct(len(class_matches), len(class_gt)),
                "eligible_frames": gt_frames,
                "matched_frames": matched_frames,
                "frame_coverage_pct": _safe_pct(matched_frames, gt_frames),
            }
        )

    gt["class_name"] = gt["class_name"].map(_normalize_class)
    speeds = _actor_speeds(gt)
    gt["derived_speed_mps"] = speeds
    pedestrian = gt[gt["class_name"] == "pedestrian"]
    pedestrian_eligible = gt_match[gt_match["class_name"] == "pedestrian"]
    pedestrian_speeds = pd.to_numeric(
        pedestrian["derived_speed_mps"], errors="coerce"
    ).dropna()
    active_pedestrian_speeds = pedestrian_speeds[pedestrian_speeds > 0.2]
    height_low, height_high = map(float, verify["pedestrian_height_range_m"])
    width_low, width_high = map(float, verify["pedestrian_width_range_m"])
    if len(pedestrian_speeds) and (pedestrian_speeds > float(verify["pedestrian_speed_max_mps"])).any():
        failures.append("implausible_pedestrian_speed")
    pedestrian_speed_rows_above_max = int(
        (pedestrian_speeds > float(verify["pedestrian_speed_max_mps"])).sum()
    )
    if len(pedestrian):
        height = pd.to_numeric(pedestrian["height_m"], errors="coerce")
        width = pd.to_numeric(pedestrian["width_m"], errors="coerce")
        if (~height.between(height_low, height_high)).any():
            failures.append("implausible_pedestrian_height")
        if (~width.between(width_low, width_high)).any():
            failures.append("implausible_pedestrian_width")
    requires_pedestrian = bool(
        verify.get(
            "require_pedestrian_gt_per_run",
            str(run_spec["scenario_family"]) in {"ped_crossing", "mixed_urban"},
        )
    )
    pedestrian_matches = 0 if matches.empty else int((matches["class_name"] == "pedestrian").sum())
    if requires_pedestrian and pedestrian_eligible.empty:
        failures.append("no_in_range_in_frustum_pedestrian_gt")
    if (
        requires_pedestrian
        and pedestrian_matches == 0
        and not prediction_score_min_by_class
    ):
        failures.append("no_pedestrian_prediction_match")

    wait = pd.to_numeric(metrics["camera_frame_wait_ms"], errors="coerce").dropna()
    result_received_pct = 100.0 * float(_truthy(metrics["result_received"]).mean())
    requested = int(run_spec.get("requested_frames", collection_config["requested_frames"]))
    if len(metrics) < int(math.ceil(float(timing["minimum_processed_fraction"]) * requested)):
        failures.append("too_few_processed_frames")
    if predictions.empty:
        failures.append("empty_predictions")
    if result_received_pct < float(timing["minimum_result_received_pct"]):
        failures.append("result_receive_collapse")
    if not len(wait):
        failures.append("missing_camera_frame_wait")
    else:
        if float(wait.median()) > float(timing["median_max_ms"]):
            failures.append("camera_wait_median")
        if float(wait.quantile(0.95)) > float(timing["p95_max_ms"]):
            failures.append("camera_wait_p95")

    in_scope = gt_match.groupby("frame_id")["actor_id"].nunique()
    in_scope_vehicle_frames = int(
        gt_match.loc[gt_match["class_name"] == "vehicle", "frame_id"].nunique()
    )
    vehicle_speed = pd.to_numeric(
        gt.loc[gt["class_name"] == "vehicle", "derived_speed_mps"], errors="coerce"
    ).dropna()
    diagnostics_present = pd.to_numeric(
        metrics.get("decode_diagnostics_present", pd.Series(0, index=metrics.index)),
        errors="coerce",
    ).fillna(0).astype(bool)
    diagnostics_fraction = float(diagnostics_present.mean()) if len(metrics) else 0.0
    minimum_diagnostics = float(verify.get("minimum_decoder_diagnostics_fraction", 0.0))
    if diagnostics_fraction < minimum_diagnostics:
        failures.append("decoder_diagnostics_below_minimum")
    pre_topk = pd.to_numeric(
        metrics.get(
            "decode_pre_topk_above_threshold_count", pd.Series(dtype=float)
        ),
        errors="coerce",
    ).dropna()
    expected_topk = collection_config.get("collection_contract", {}).get(
        "required_effective_args", {}
    ).get("--topk-objects")
    if expected_topk is not None:
        runtime_topk = pd.to_numeric(
            metrics.get("decode_topk_limit", pd.Series(dtype=float)), errors="coerce"
        ).where(diagnostics_present).dropna()
        observed_limits = set(runtime_topk.astype(int))
        if diagnostics_present.any() and observed_limits != {int(expected_topk)}:
            failures.append("decoder_topk_runtime_mismatch")

    exact_target_rows = pd.DataFrame()
    exact_target_in_scope_fast = pd.Series(dtype=bool)
    exact_target_speed = pd.Series(dtype=float)
    exact_fast_dwell_s: float | None = None
    exact_role_name = str(verify.get("exact_fast_role_name", ""))
    is_exact_fast = (
        bool(exact_role_name)
        and str(run_spec.get("scenario_family")) == "exact_fast_convoy"
    )
    if is_exact_fast:
        if "role_name" not in gt.columns:
            failures.append("missing_role_name_for_exact_fast_gate")
            exact_target_rows = gt.iloc[0:0].copy()
        else:
            exact_target_rows = gt[gt["role_name"].astype(str) == exact_role_name].copy()
        exact_target_rows = exact_target_rows.sort_values("carla_timestamp")
        exact_target_speed = pd.to_numeric(
            exact_target_rows["derived_speed_mps"], errors="coerce"
        )
        exact_target_in_scope_fast = (
            _truthy(exact_target_rows["in_camera_frustum"])
            & (
                pd.to_numeric(exact_target_rows["distance_m"], errors="coerce")
                <= float(verify["exact_fast_range_max_m"])
            )
            & (exact_target_speed >= float(verify["exact_fast_speed_min_mps"]))
        )
        exact_fast_dwell_s = _longest_true_dwell(
            exact_target_in_scope_fast.tolist(),
            pd.to_numeric(exact_target_rows["carla_timestamp"], errors="coerce").tolist(),
        )
        if exact_target_rows.empty:
            failures.append("missing_exact_fast_target_gt")
        elif exact_fast_dwell_s < float(verify["exact_fast_dwell_min_s"]):
            failures.append("exact_fast_target_dwell_below_minimum")

    summary: Dict[str, object] = {
        "episode_id": str(run_spec["episode_id"]),
        "run_group": str(run_spec["run_group"]),
        "scenario_family": str(run_spec["scenario_family"]),
        "scenario_variant": str(run_spec.get("scenario_variant", run_spec["scenario_family"])),
        "split": str(run_spec["split"]),
        "seed": int(run_spec["seed"]),
        "processed_frames": int(len(metrics)),
        "gt_rows": int(len(gt)),
        "prediction_rows_at_operating_threshold": int(len(predictions)),
        "in_scope_gt_rows": int(len(gt_match)),
        "mean_in_scope_actors_per_frame": float(in_scope.mean()) if len(in_scope) else 0.0,
        "max_in_scope_actors_per_frame": int(in_scope.max()) if len(in_scope) else 0,
        "pedestrian_gt_rows": int(len(pedestrian)),
        "in_scope_pedestrian_gt_rows": int(len(pedestrian_eligible)),
        "pedestrian_matches": pedestrian_matches,
        "pedestrian_speed_median_mps": _quantile(pedestrian_speeds, 0.50),
        "pedestrian_speed_max_mps": _quantile(pedestrian_speeds, 1.0),
        "pedestrian_speed_rows_above_max": pedestrian_speed_rows_above_max,
        "active_pedestrian_speed_p50_mps": _quantile(active_pedestrian_speeds, 0.50),
        "active_pedestrian_speed_p95_mps": _quantile(active_pedestrian_speeds, 0.95),
        "in_scope_vehicle_frames": in_scope_vehicle_frames,
        "in_scope_vehicle_frame_pct": _safe_pct(in_scope_vehicle_frames, requested),
        "vehicle_speed_p50_mps": _quantile(vehicle_speed, 0.50),
        "vehicle_speed_p95_mps": _quantile(vehicle_speed, 0.95),
        "decoder_diagnostics_fraction": diagnostics_fraction,
        "decode_pre_topk_max": int(pre_topk.max()) if len(pre_topk) else None,
        "decode_topk_saturated_frames": int(
            pd.to_numeric(
                metrics.get("decode_topk_saturated", pd.Series(0, index=metrics.index)),
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        ),
        "exact_fast_target_gt_rows": int(len(exact_target_rows)),
        "exact_fast_target_speed_p50_mps": _quantile(exact_target_speed, 0.50),
        "exact_fast_target_in_scope_fast_rows": int(exact_target_in_scope_fast.sum()),
        "exact_fast_target_dwell_s": exact_fast_dwell_s,
        "result_received_pct": result_received_pct,
        "camera_frame_wait_median_ms": _quantile(wait, 0.50),
        "camera_frame_wait_p95_ms": _quantile(wait, 0.95),
        "max_abs_origin_coordinate_m": float(
            numeric[["origin_x", "origin_y"]].abs().max().max()
        ) if len(numeric) else None,
        "pass": not failures,
        "failures": "|".join(sorted(set(failures))),
        "ground_truth_sha256": _sha256(gt_path),
        "prediction_sha256": _sha256(prediction_path),
        "metrics_sha256": _sha256(metrics_path),
    }
    return summary, coverage_rows


def _run_surrogate_gate(
    batch_dir: Path,
    split_manifest_path: Path,
    prediction_score_min_by_class: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    config = copy.deepcopy(load_config())
    config["replay"]["roots"] = [str(batch_dir / "runs")]
    config["replay"]["split_manifest_csv"] = str(split_manifest_path)
    # The verification split is the accepted corpus inventory and may omit a
    # predeclared invalid trajectory that remains immutable on disk.
    config["replay"]["allow_unlisted_episodes"] = True
    config["replay"]["max_episode_steps"] = 1200
    config["replay"]["prediction_score_min_by_class"] = dict(
        prediction_score_min_by_class or {}
    )
    config["safety"]["epsilon_m"] = 2.0
    config["safety"]["range_m"] = 25.0
    config["actions"]["preferred_core_kib"] = 90
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(profiles, config["actions"]["fps"], 90)
    surface = ChannelSurface(config)
    records = discover_trace_registry(config)
    split_frame = pd.read_csv(split_manifest_path)
    variant_by_episode = {
        str(row.episode_id): str(row.scenario_variant)
        for row in split_frame.itertuples(index=False)
    }
    rows: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        frames = load_trace_episode(record, config, range_m=25.0, max_steps=1200)
        seed = 810_000 + index
        channel = ChannelProcess(config, surface, seed)
        env = SurrogateEnv(
            config,
            frames,
            actions,
            channel,
            surface,
            seed + 10_000,
            latency_mode="sample",
            latency_crn_by_tick=True,
        )
        frame_count = send_needed = selected_split = selected_skip = capture_attempt = 0
        split_raw_safe = skip_raw_safe = infeasible = shield_ood = 0
        truth_objects = observed_objects = 0
        truth_by_class = {"vehicle": 0, "pedestrian": 0}
        observed_by_class = {"vehicle": 0, "pedestrian": 0}
        while not env.done:
            truth_objects += len(env.frame.truth_objects)
            observed_objects += len(env.frame.observed_objects)
            for obj in env.frame.truth_objects:
                class_name = _normalize_class(obj.class_name)
                if class_name in truth_by_class:
                    truth_by_class[class_name] += 1
            for obj in env.frame.observed_objects:
                class_name = _normalize_class(obj.class_name)
                if class_name in observed_by_class:
                    observed_by_class[class_name] += 1
            decision = env.shielded_decision()
            raw = decision.raw_safe_action_ids
            split_is_raw_safe = any(item.startswith("SPLIT::") for item in raw)
            split_raw_safe += int(split_is_raw_safe)
            skip_raw_safe += int("SKIP" in raw)
            send_needed += int(
                "SKIP" not in raw and split_is_raw_safe
            )
            selected_split += int(decision.selected.action.mode == "SPLIT")
            selected_skip += int(decision.selected.action.mode == "SKIP")
            infeasible += int(not decision.feasible)
            shield_ood += int(decision.shield_ood)
            step = env.step(decision.selected.action)
            capture_attempt += int(pd.notna(step.get("actual_delivery")))
            frame_count += 1
        rows.append(
            {
                "episode_id": record.episode_id,
                "scenario_family": record.scenario_family,
                "scenario_variant": variant_by_episode[record.episode_id],
                "split": record.split,
                "frames": frame_count,
                "send_needed_frames": send_needed,
                "send_needed_pct": _safe_pct(send_needed, frame_count),
                "split_raw_safe_frames": split_raw_safe,
                "split_raw_safe_pct": _safe_pct(split_raw_safe, frame_count),
                "skip_raw_safe_frames": skip_raw_safe,
                "skip_raw_safe_pct": _safe_pct(skip_raw_safe, frame_count),
                "selected_split_frames": selected_split,
                "selected_split_pct": _safe_pct(selected_split, frame_count),
                "selected_skip_frames": selected_skip,
                "selected_skip_pct": _safe_pct(selected_skip, frame_count),
                "infeasible_frames": infeasible,
                "infeasible_pct": _safe_pct(infeasible, frame_count),
                "shield_ood_frames": shield_ood,
                "shield_ood_pct": _safe_pct(shield_ood, frame_count),
                "capture_attempt_frames": capture_attempt,
                "capture_attempt_pct": _safe_pct(capture_attempt, frame_count),
                "truth_objects": truth_objects,
                "observed_objects": observed_objects,
                "observation_coverage_pct": _safe_pct(observed_objects, truth_objects),
                "truth_vehicle_objects": truth_by_class["vehicle"],
                "observed_vehicle_objects": observed_by_class["vehicle"],
                "vehicle_observation_coverage_pct": _safe_pct(
                    observed_by_class["vehicle"], truth_by_class["vehicle"]
                ),
                "truth_pedestrian_objects": truth_by_class["pedestrian"],
                "observed_pedestrian_objects": observed_by_class["pedestrian"],
                "pedestrian_observation_coverage_pct": _safe_pct(
                    observed_by_class["pedestrian"], truth_by_class["pedestrian"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def _trajectory_bootstrap_recall(
    per_run: pd.DataFrame,
    *,
    range_by_class: Mapping[str, float],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for class_name in ("pedestrian", "vehicle"):
        range_m = float(range_by_class[class_name])
        selected = per_run[
            (per_run["split"] == "test")
            & (per_run["contract"] == "validation_f1")
            & (per_run["class_name"] == class_name)
        ].copy()
        eligible_column = f"eligible_gt_rows_le{range_m:g}m"
        matched_column = f"matched_gt_rows_le{range_m:g}m"
        selected = selected[pd.to_numeric(selected[eligible_column], errors="coerce") > 0]
        recalls: List[float] = []
        if len(selected):
            eligible = selected[eligible_column].to_numpy(dtype=float)
            matched = selected[matched_column].to_numpy(dtype=float)
            for _unused in range(samples):
                indices = rng.integers(0, len(selected), size=len(selected))
                denominator = float(eligible[indices].sum())
                if denominator:
                    recalls.append(float(matched[indices].sum()) / denominator)
        point_denominator = int(selected[eligible_column].sum()) if len(selected) else 0
        point_numerator = int(selected[matched_column].sum()) if len(selected) else 0
        rows.append(
            {
                "split": "test",
                "class_name": class_name,
                "range_upper_m": range_m,
                "trajectory_count": int(len(selected)),
                "eligible_gt_rows": point_denominator,
                "matched_gt_rows": point_numerator,
                "recall": (
                    float(point_numerator) / point_denominator
                    if point_denominator else float("nan")
                ),
                "bootstrap_samples": samples,
                "ci95_lower": float(np.quantile(recalls, 0.025)) if recalls else None,
                "ci95_upper": float(np.quantile(recalls, 0.975)) if recalls else None,
            }
        )
    return pd.DataFrame(rows)


def _localization_error_summary(
    runs: Sequence[evaluation_contract.RunData],
    thresholds: Mapping[str, float],
    range_by_class: Mapping[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for class_name in ("pedestrian", "vehicle"):
        threshold = float(thresholds[class_name])
        range_m = float(range_by_class[class_name])
        errors: List[float] = []
        for run in runs:
            if run.split != "test":
                continue
            gt = run.gt[
                (run.gt["class_name"] == class_name)
                & (run.gt["distance_m"] <= range_m)
            ].copy()
            predictions = run.predictions[
                (run.predictions["class_name"] == class_name)
                & (run.predictions["score"] >= threshold - 1e-12)
            ].copy()
            matches = _greedy_prediction_matches(
                gt, predictions, evaluation_contract.MATCH_GATE_M
            )
            if len(matches):
                errors.extend(
                    pd.to_numeric(matches["match_error_m"], errors="coerce")
                    .dropna()
                    .tolist()
                )
        series = pd.Series(errors, dtype=float)
        rows.append(
            {
                "split": "test",
                "class_name": class_name,
                "range_upper_m": range_m,
                "score_threshold": threshold,
                "matched_rows": int(len(series)),
                "localization_error_median_m": (
                    float(series.median()) if len(series) else None
                ),
                "localization_error_p95_m": (
                    float(series.quantile(0.95)) if len(series) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def verify(
    batch_dir: Path,
    skip_surrogate: bool = False,
    evaluation_contract_path: Path | None = None,
) -> Path:
    batch_dir = batch_dir.resolve()
    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if str(batch_manifest.get("mode")) != "full":
        raise ValueError("corpus verification requires a full collection batch")
    if str(batch_manifest.get("status")) != "collection_complete_pending_verification":
        raise ValueError(
            "corpus batch is not complete and pending verification: "
            f"{batch_manifest.get('status')}"
        )
    config_path = batch_dir / "resolved_collection_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        collection_config = yaml.safe_load(stream)
    evaluation_spec: Mapping[str, object] | None = None
    if evaluation_contract_path is not None:
        evaluation_contract_path = evaluation_contract_path.resolve()
        with evaluation_contract_path.open("r", encoding="utf-8") as stream:
            loaded_evaluation_spec = yaml.safe_load(stream)
        if int(loaded_evaluation_spec.get("schema_version", 0)) != 1:
            raise ValueError("evaluation contract schema_version must be 1")
        evaluation_spec = loaded_evaluation_spec
    all_run_specs: Sequence[Mapping[str, object]] = batch_manifest["runs"]
    excluded_episode_ids = (
        [str(value) for value in evaluation_spec.get("excluded_episode_ids", [])]
        if evaluation_spec is not None else []
    )
    all_episode_ids = {str(item["episode_id"]) for item in all_run_specs}
    missing_exclusions = set(excluded_episode_ids) - all_episode_ids
    if missing_exclusions:
        raise ValueError(
            "excluded episodes are absent from the batch: "
            + ", ".join(sorted(missing_exclusions))
        )
    run_specs = [
        item for item in all_run_specs
        if str(item["episode_id"]) not in set(excluded_episode_ids)
    ]
    if not run_specs:
        raise ValueError("evaluation contract excludes every corpus episode")
    verification_dir = batch_dir / "verification" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    verification_dir.mkdir(parents=True, exist_ok=False)

    selected_thresholds: Dict[str, float] = {}
    selected_metrics = pd.DataFrame()
    cumulative_coverage = pd.DataFrame()
    diagnostic_recall_ci = pd.DataFrame()
    localization_error = pd.DataFrame()
    radar_summary = pd.DataFrame()
    if evaluation_spec is not None:
        evaluation_runs, _items = evaluation_contract.load_runs(
            batch_dir, excluded_episode_ids
        )
        match_cache = evaluation_contract.prepare_match_cache(evaluation_runs)
        pr_curves = evaluation_contract.build_pr_curves(evaluation_runs, match_cache)
        selected = evaluation_contract.choose_validation_thresholds(pr_curves)
        selected_thresholds = {
            str(row.class_name): float(row.score_threshold)
            for row in selected.itertuples(index=False)
        }
        by_range, cumulative_coverage, per_run_coverage = (
            evaluation_contract.build_range_coverage(
                evaluation_runs, selected_thresholds
            )
        )
        selected_metrics = pd.DataFrame(
            [
                evaluation_contract.score_threshold(
                    evaluation_runs,
                    match_cache,
                    class_name,
                    threshold,
                    split,
                )
                for class_name, threshold in selected_thresholds.items()
                for split in ("validation", "test", "all")
            ]
        )
        diagnostic_ranges = {
            str(class_name): float(range_m)
            for class_name, range_m in evaluation_spec[
                "diagnostic_recall_range_m_by_class"
            ].items()
        }
        if set(diagnostic_ranges) != {"pedestrian", "vehicle"}:
            raise ValueError(
                "diagnostic_recall_range_m_by_class must define pedestrian and vehicle"
            )
        diagnostic_recall_ci = _trajectory_bootstrap_recall(
            per_run_coverage,
            range_by_class=diagnostic_ranges,
            samples=int(evaluation_spec.get("trajectory_bootstrap_samples", 10000)),
            seed=int(evaluation_spec.get("trajectory_bootstrap_seed", 20260813)),
        )
        localization_error = _localization_error_summary(
            evaluation_runs, selected_thresholds, diagnostic_ranges
        )
        reference_run = REPO_ROOT / str(evaluation_spec["radar_reference_run"])
        radar_per_run, radar_summary = evaluation_contract.build_radar_audit(
            evaluation_runs, reference_run
        )
        pr_curves.to_csv(verification_dir / "precision_recall_curve.csv", index=False)
        selected.to_csv(
            verification_dir / "validation_selected_thresholds.csv", index=False
        )
        selected_metrics.to_csv(
            verification_dir / "selected_threshold_metrics.csv", index=False
        )
        by_range.to_csv(verification_dir / "coverage_by_range.csv", index=False)
        cumulative_coverage.to_csv(
            verification_dir / "coverage_cumulative_range.csv", index=False
        )
        per_run_coverage.to_csv(
            verification_dir / "coverage_per_run.csv", index=False
        )
        diagnostic_recall_ci.to_csv(
            verification_dir / "diagnostic_recall_trajectory_bootstrap_ci.csv",
            index=False,
        )
        localization_error.to_csv(
            verification_dir / "diagnostic_localization_error.csv", index=False
        )
        radar_per_run.to_csv(
            verification_dir / "radar_density_per_run.csv", index=False
        )
        radar_summary.to_csv(
            verification_dir / "radar_density_summary.csv", index=False
        )
        evaluation_contract.plot_pr_curves(pr_curves, selected, verification_dir)
        evaluation_contract.plot_range_coverage(by_range, verification_dir)

    run_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    for run_spec in run_specs:
        run_dir = Path(str(run_spec["run_dir"]))
        summary, coverage = _inspect_run(
            run_spec,
            run_dir,
            collection_config,
            selected_thresholds,
        )
        run_rows.append(summary)
        coverage_rows.extend(coverage)
        split_rows.append(
            {
                "episode_id": str(run_spec["episode_id"]),
                "run_group": str(run_spec["run_group"]),
                "scenario_family": str(run_spec["scenario_family"]),
                "scenario_variant": str(run_spec.get("scenario_variant", run_spec["scenario_family"])),
                "split": str(run_spec["split"]),
            }
        )
    runs = pd.DataFrame(run_rows)
    coverage = pd.DataFrame(coverage_rows)
    split_manifest = pd.DataFrame(split_rows)
    runs.to_csv(verification_dir / "run_verification.csv", index=False)
    coverage.to_csv(verification_dir / "class_coverage.csv", index=False)
    split_manifest_path = verification_dir / "replay_split_manifest.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    gate_failures: List[str] = []
    structural_failures: List[str] = []
    for run_spec in run_specs:
        episode_id = str(run_spec["episode_id"])
        if str(run_spec.get("status")) not in {
            "complete", "complete_with_teardown_warning"
        }:
            structural_failures.append(f"{episode_id}:collection_status")
        for name in (
            "basic_gate", "radar_density_gate", "traffic_sanity",
            "exact_fast_scenario_gate",
        ):
            value = run_spec.get(name, {})
            if not isinstance(value, Mapping) or not bool(value.get("pass", False)):
                structural_failures.append(f"{episode_id}:{name}")
        if int(run_spec.get("traffic_sanity", {}).get("collision_events", 0)) != 0:
            structural_failures.append(f"{episode_id}:traffic_collision")
        postflight = run_spec.get("postflight_dynamic_actor_counts", {})
        if not isinstance(postflight, Mapping) or any(
            int(value) != 0 for value in postflight.values()
        ):
            structural_failures.append(f"{episode_id}:postflight_actor_leak")
    if structural_failures:
        gate_failures.append("one_or_more_structural_collection_gates_failed")
    if not bool(runs["pass"].all()):
        gate_failures.append("one_or_more_run_level_gates_failed")
    aggregate_coverage = (
        coverage.groupby("class_name", as_index=False)[
            ["eligible_gt_rows", "matched_rows", "eligible_frames", "matched_frames"]
        ].sum()
    )
    aggregate_coverage["object_row_coverage_pct"] = 100.0 * aggregate_coverage[
        "matched_rows"
    ] / aggregate_coverage["eligible_gt_rows"].replace(0, np.nan)
    aggregate_coverage["frame_coverage_pct"] = 100.0 * aggregate_coverage[
        "matched_frames"
    ] / aggregate_coverage["eligible_frames"].replace(0, np.nan)
    direct_match_coverage = _safe_pct(
        int(coverage["matched_rows"].sum()), int(coverage["eligible_gt_rows"].sum())
    )
    verify_config = collection_config["verification"]
    ped_coverage = aggregate_coverage[aggregate_coverage["class_name"] == "pedestrian"]
    expects_pedestrian = bool(verify_config.get("require_pedestrian_replay", any(
        str(run_spec["scenario_family"]) in {"ped_crossing", "mixed_urban"}
        for run_spec in run_specs
    )))
    if expects_pedestrian and (
        ped_coverage.empty or int(ped_coverage.iloc[0]["matched_rows"]) == 0
    ):
        gate_failures.append("no_pedestrian_match_denominator")
    if evaluation_spec is not None:
        recall_gate_mode = str(evaluation_spec.get("recall_gate_mode", "hard"))
        if recall_gate_mode not in {"hard", "report_only"}:
            raise ValueError("recall_gate_mode must be hard or report_only")
        if recall_gate_mode == "hard":
            near_minimums = {
                str(class_name): float(value)
                for class_name, value in evaluation_spec[
                    "minimum_test_near_recall"
                ].items()
            }
            for class_name, minimum in near_minimums.items():
                row = diagnostic_recall_ci[
                    diagnostic_recall_ci["class_name"] == class_name
                ]
                if row.empty or not math.isfinite(float(row.iloc[0]["recall"])):
                    gate_failures.append(f"{class_name}_test_recall_missing")
                elif float(row.iloc[0]["recall"]) < minimum:
                    gate_failures.append(f"{class_name}_test_recall_below_minimum")
        corpus_radar = radar_summary[
            radar_summary["source"].str.startswith("policy_corpus")
        ]
        density_ratio = (
            float(corpus_radar.iloc[0]["median_fraction_of_reference"])
            if len(corpus_radar) else float("nan")
        )
        density_tolerance = float(evaluation_spec["radar_relative_tolerance"])
        if (
            not math.isfinite(density_ratio)
            or abs(density_ratio - 1.0) > density_tolerance
        ):
            gate_failures.append("corpus_radar_density_outside_contract")
    for class_name in ("pedestrian", "vehicle"):
        row = aggregate_coverage[aggregate_coverage["class_name"] == class_name]
        if row.empty or int(row.iloc[0]["eligible_gt_rows"]) == 0:
            gate_failures.append(f"missing_{class_name}_ground_truth_population")

    surrogate = pd.DataFrame()
    surrogate_summary = pd.DataFrame()
    if not skip_surrogate:
        surrogate = _run_surrogate_gate(
            batch_dir, split_manifest_path, selected_thresholds
        )
        surrogate.to_csv(verification_dir / "surrogate_send_path.csv", index=False)
        surrogate_summary = surrogate.groupby(
            ["scenario_family", "scenario_variant"], as_index=False
        )[
            [
                "frames", "send_needed_frames", "split_raw_safe_frames", "skip_raw_safe_frames",
                "selected_split_frames", "selected_skip_frames", "infeasible_frames",
                "shield_ood_frames", "capture_attempt_frames",
                "truth_objects", "observed_objects", "truth_vehicle_objects",
                "observed_vehicle_objects", "truth_pedestrian_objects",
                "observed_pedestrian_objects",
            ]
        ].sum()
        for numerator in (
            "send_needed", "split_raw_safe", "skip_raw_safe", "selected_split",
            "selected_skip", "infeasible", "shield_ood", "capture_attempt",
        ):
            surrogate_summary[f"{numerator}_pct"] = 100.0 * surrogate_summary[
                f"{numerator}_frames"
            ] / surrogate_summary["frames"]
        surrogate_summary["observation_coverage_pct"] = 100.0 * surrogate_summary[
            "observed_objects"
        ] / surrogate_summary["truth_objects"].replace(0, np.nan)
        for class_name in ("vehicle", "pedestrian"):
            surrogate_summary[f"{class_name}_observation_coverage_pct"] = 100.0 * (
                surrogate_summary[f"observed_{class_name}_objects"]
                / surrogate_summary[f"truth_{class_name}_objects"].replace(0, np.nan)
            )
        overall_frames = int(surrogate["frames"].sum())
        overall_send = _safe_pct(int(surrogate["send_needed_frames"].sum()), overall_frames)
        overall_split = _safe_pct(int(surrogate["selected_split_frames"].sum()), overall_frames)
        vehicle_replay_coverage = _safe_pct(
            int(surrogate["observed_vehicle_objects"].sum()),
            int(surrogate["truth_vehicle_objects"].sum()),
        )
        legacy_vehicle_coverage = float(
            verify_config.get(
                "legacy_vehicle_observation_coverage_pct",
                verify_config.get("legacy_observation_coverage_pct", 45.18),
            )
        )
        if (
            evaluation_spec is None
            and bool(verify_config.get("require_vehicle_replay_above_legacy", True))
            and (
            vehicle_replay_coverage <= legacy_vehicle_coverage
            )
        ):
            gate_failures.append("vehicle_replay_observation_coverage_not_above_legacy")
        if expects_pedestrian and int(surrogate["truth_pedestrian_objects"].sum()) == 0:
            gate_failures.append("no_pedestrian_truth_in_replay")
        if expects_pedestrian and int(surrogate["observed_pedestrian_objects"].sum()) == 0:
            gate_failures.append("no_observed_pedestrian_in_replay")
        pedestrian_replay_coverage = _safe_pct(
            int(surrogate["observed_pedestrian_objects"].sum()),
            int(surrogate["truth_pedestrian_objects"].sum()),
        )
        minimum_pedestrian_coverage = verify_config.get(
            "minimum_pedestrian_replay_observation_coverage_pct"
        )
        if (
            evaluation_spec is None
            and
            expects_pedestrian
            and minimum_pedestrian_coverage is not None
            and pedestrian_replay_coverage < float(minimum_pedestrian_coverage)
        ):
            gate_failures.append("pedestrian_replay_observation_coverage_below_minimum")
        if bool(verify_config.get("require_send_needed_above_legacy", True)) and (
            overall_send <= float(verify_config["legacy_send_needed_pct"])
        ):
            gate_failures.append("send_needed_not_above_legacy")
        if bool(verify_config.get("require_selected_split_above_legacy", True)) and (
            overall_split <= float(verify_config["legacy_selected_split_pct"])
        ):
            gate_failures.append("selected_split_not_above_legacy")
        fast_convoy = surrogate_summary[
            (surrogate_summary["scenario_family"] == "dense_fast")
            & (surrogate_summary["scenario_variant"] == "fast_convoy")
        ]
        if bool(verify_config.get("require_fast_convoy_send_needed_above_legacy", True)) and (
            fast_convoy.empty
            or float(fast_convoy.iloc[0]["send_needed_pct"])
            <= float(verify_config["legacy_send_needed_pct"])
        ):
            gate_failures.append("fast_convoy_send_needed_not_above_legacy")

    family_summary = runs.groupby(
        ["scenario_family", "scenario_variant"], as_index=False
    ).agg(
        runs=("episode_id", "count"),
        mean_in_scope_actors=("mean_in_scope_actors_per_frame", "mean"),
        vehicle_speed_p50_mps=("vehicle_speed_p50_mps", "median"),
        vehicle_speed_p95_mps=("vehicle_speed_p95_mps", "median"),
        in_scope_vehicle_frame_pct=("in_scope_vehicle_frame_pct", "median"),
        active_pedestrian_speed_p50_mps=("active_pedestrian_speed_p50_mps", "median"),
        active_pedestrian_speed_p95_mps=("active_pedestrian_speed_p95_mps", "median"),
        pedestrian_speed_max_mps=("pedestrian_speed_max_mps", "max"),
        pedestrian_speed_rows_above_max=("pedestrian_speed_rows_above_max", "sum"),
        decoder_diagnostics_fraction=("decoder_diagnostics_fraction", "mean"),
        decode_pre_topk_max=("decode_pre_topk_max", "max"),
        decode_topk_saturated_frames=("decode_topk_saturated_frames", "sum"),
        exact_fast_target_speed_p50_mps=("exact_fast_target_speed_p50_mps", "median"),
        exact_fast_target_dwell_s=("exact_fast_target_dwell_s", "min"),
        camera_wait_median_ms=("camera_frame_wait_median_ms", "median"),
        camera_wait_p95_ms=("camera_frame_wait_p95_ms", "median"),
    )
    if skip_surrogate and not gate_failures:
        status = "PARTIAL_PASS_SMOKE_ONLY"
    else:
        status = "PASS" if not gate_failures else "FAIL_QUARANTINED"
    report = [
        "# Policy corpus verification",
        "",
        f"**Status: {status}**",
        "",
        f"Batch: `{batch_dir}`",
        "",
        "## Gate summary",
        "",
        f"- Corpus scope: `{verify_config.get('corpus_scope', 'multiclass')}`.",
        f"- Accepted trajectories: {len(run_specs)}/{len(all_run_specs)}; excluded before threshold selection and replay: {excluded_episode_ids or 'none'}.",
        f"- Online structural gates (sensor contract, traffic, completion, cleanup): {'PASS' if not structural_failures else 'FAIL'}.",
        f"- Run-level schema, position, realized-regime, decoder-telemetry, and timing gates: {'PASS' if bool(runs['pass'].all()) else 'FAIL'}",
        f"- Direct same-frame object-row match coverage: {direct_match_coverage:.2f}% (reported diagnostically; not compared to held-track replay coverage)",
    ]
    if evaluation_spec is not None:
        report.extend(
            [
                f"- Frozen per-class thresholds selected by validation F1: {selected_thresholds}.",
                f"- Held-out actor-origin recall is `{evaluation_spec.get('recall_gate_mode', 'hard')}` and reported with trajectory-bootstrap 95% CIs:",
                _markdown_table(diagnostic_recall_ci),
                "- Localization error among matched held-out objects:",
                _markdown_table(localization_error),
                f"- Radar density relative to retained on-contract reference:",
                _markdown_table(radar_summary),
            ]
        )
    if expects_pedestrian:
        report.append(
            f"- Pedestrian matched rows: {int(ped_coverage.iloc[0]['matched_rows']) if not ped_coverage.empty else 0}"
        )
    if not skip_surrogate:
        overall_frames = int(surrogate["frames"].sum())
        surrogate_lines = [
                f"- Send-needed: {_safe_pct(int(surrogate['send_needed_frames'].sum()), overall_frames):.2f}% (legacy 14.66%)",
                f"- Selected SPLIT: {_safe_pct(int(surrogate['selected_split_frames'].sum()), overall_frames):.2f}% (legacy 5.83%)",
                f"- Infeasible/over-budget frames: {_safe_pct(int(surrogate['infeasible_frames'].sum()), overall_frames):.2f}%",
                f"- Shield OOD frames: {_safe_pct(int(surrogate['shield_ood_frames'].sum()), overall_frames):.2f}%",
                f"- Capture attempts: {_safe_pct(int(surrogate['capture_attempt_frames'].sum()), overall_frames):.2f}%",
                f"- Vehicle replay observation coverage: {_safe_pct(int(surrogate['observed_vehicle_objects'].sum()), int(surrogate['truth_vehicle_objects'].sum())):.2f}% (vehicle-only legacy 45.18%)",
        ]
        if expects_pedestrian:
            surrogate_lines.append(
                f"- Pedestrian replay observation coverage: {_safe_pct(int(surrogate['observed_pedestrian_objects'].sum()), int(surrogate['truth_pedestrian_objects'].sum())):.2f}% ({int(surrogate['observed_pedestrian_objects'].sum())} observed object-frames; configured minimum {verify_config.get('minimum_pedestrian_replay_observation_coverage_pct', 'nonzero only')}%)"
            )
        if not bool(verify_config.get("require_send_needed_above_legacy", True)):
            surrogate_lines.append("- Send-needed and selected-action rates are diagnostics, not collection gates for this scope.")
        report.extend(surrogate_lines)
    if gate_failures:
        report.extend(["", "Failures: " + ", ".join(gate_failures)])
    failed_runs = runs.loc[
        ~runs["pass"],
        [
            "episode_id", "failures", "pedestrian_speed_rows_above_max",
            "pedestrian_speed_max_mps",
        ],
    ]
    if not failed_runs.empty:
        report.extend(
            [
                "",
                "## Failed run details",
                "",
                _markdown_table(failed_runs),
            ]
        )
    report.extend(
        [
            "",
            "## Per-class coverage",
            "",
            _markdown_table(aggregate_coverage),
            "",
            "## Realized scenario conditions",
            "",
            _markdown_table(family_summary),
        ]
    )
    if not skip_surrogate:
        report.extend(
            [
                "",
                "## Surrogate send-path by family",
                "",
                _markdown_table(surrogate_summary),
            ]
        )
    report.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This is corpus and surrogate verification, not a live safety guarantee. Actor-origin fields are the matching/replay coordinates; bbox-center world fields remain diagnostics. A failed batch is quarantined and must not be used for policy training or headline evaluation.",
            "",
        ]
    )
    report_path = verification_dir / "CORPUS_VERIFICATION.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    artifacts = {}
    for path in sorted(verification_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    verification_manifest: MutableMapping[str, object] = {
        "schema": "policy_corpus_verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate_failures": gate_failures,
        "batch_manifest_sha256": _sha256(batch_manifest_path),
        "collection_config_sha256": _sha256(config_path),
        "evaluation_contract_path": (
            str(evaluation_contract_path) if evaluation_contract_path else None
        ),
        "evaluation_contract_sha256": (
            _sha256(evaluation_contract_path) if evaluation_contract_path else None
        ),
        "acceptance_basis": "structural_collection_gates",
        "accepted_run_count": int(len(run_specs)),
        "excluded_episode_ids": excluded_episode_ids,
        "recall_gate_mode": (
            str(evaluation_spec.get("recall_gate_mode", "hard"))
            if evaluation_spec is not None else None
        ),
        "structural_failures": structural_failures,
        "prediction_score_min_by_class": selected_thresholds,
        "artifacts": artifacts,
    }
    manifest_path = verification_dir / "verification_manifest.json"
    manifest_path.write_text(
        json.dumps(verification_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return verification_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--skip-surrogate", action="store_true")
    parser.add_argument("--evaluation-contract", type=Path)
    args = parser.parse_args()
    print(
        verify(
            args.batch_dir,
            args.skip_surrogate,
            args.evaluation_contract,
        )
    )


if __name__ == "__main__":
    main()
