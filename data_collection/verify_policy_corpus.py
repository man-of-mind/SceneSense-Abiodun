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
    discover_trace_registry,
    load_trace_episode,
)

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


def _inspect_run(
    run_spec: Mapping[str, object],
    run_dir: Path,
    collection_config: Mapping[str, object],
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
    scores = pd.to_numeric(
        predictions.get("score", pd.Series(1.0, index=predictions.index)), errors="coerce"
    )
    predictions = predictions[scores >= float(verify["prediction_score_min"])].copy()
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
    requires_pedestrian = str(run_spec["scenario_family"]) in {"ped_crossing", "mixed_urban"}
    pedestrian_matches = 0 if matches.empty else int((matches["class_name"] == "pedestrian").sum())
    if requires_pedestrian and pedestrian_eligible.empty:
        failures.append("no_in_range_in_frustum_pedestrian_gt")
    if requires_pedestrian and pedestrian_matches == 0:
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
    summary: Dict[str, object] = {
        "episode_id": str(run_spec["episode_id"]),
        "run_group": str(run_spec["run_group"]),
        "scenario_family": str(run_spec["scenario_family"]),
        "scenario_variant": str(run_spec.get("scenario_variant", run_spec["scenario_family"])),
        "split": str(run_spec["split"]),
        "seed": int(run_spec["seed"]),
        "processed_frames": int(len(metrics)),
        "gt_rows": int(len(gt)),
        "prediction_rows_score_ge_0_20": int(len(predictions)),
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
) -> pd.DataFrame:
    config = copy.deepcopy(load_config())
    config["replay"]["roots"] = [str(batch_dir / "runs")]
    config["replay"]["split_manifest_csv"] = str(split_manifest_path)
    config["replay"]["max_episode_steps"] = 1200
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


def verify(batch_dir: Path, skip_surrogate: bool = False) -> Path:
    batch_dir = batch_dir.resolve()
    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    config_path = batch_dir / "resolved_collection_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        collection_config = yaml.safe_load(stream)
    run_specs: Sequence[Mapping[str, object]] = batch_manifest["runs"]
    verification_dir = batch_dir / "verification" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    verification_dir.mkdir(parents=True, exist_ok=False)

    run_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    for run_spec in run_specs:
        run_dir = Path(str(run_spec["run_dir"]))
        summary, coverage = _inspect_run(run_spec, run_dir, collection_config)
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
    expects_pedestrian = any(
        str(run_spec["scenario_family"]) in {"ped_crossing", "mixed_urban"}
        for run_spec in run_specs
    )
    if expects_pedestrian and (
        ped_coverage.empty or int(ped_coverage.iloc[0]["matched_rows"]) == 0
    ):
        gate_failures.append("no_pedestrian_match_denominator")

    surrogate = pd.DataFrame()
    surrogate_summary = pd.DataFrame()
    if not skip_surrogate:
        surrogate = _run_surrogate_gate(batch_dir, split_manifest_path)
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
        if vehicle_replay_coverage <= legacy_vehicle_coverage:
            gate_failures.append("vehicle_replay_observation_coverage_not_above_legacy")
        if int(surrogate["truth_pedestrian_objects"].sum()) == 0:
            gate_failures.append("no_pedestrian_truth_in_replay")
        if int(surrogate["observed_pedestrian_objects"].sum()) == 0:
            gate_failures.append("no_observed_pedestrian_in_replay")
        if overall_send <= float(verify_config["legacy_send_needed_pct"]):
            gate_failures.append("send_needed_not_above_legacy")
        if overall_split <= float(verify_config["legacy_selected_split_pct"]):
            gate_failures.append("selected_split_not_above_legacy")
        fast_convoy = surrogate_summary[
            (surrogate_summary["scenario_family"] == "dense_fast")
            & (surrogate_summary["scenario_variant"] == "fast_convoy")
        ]
        if fast_convoy.empty or float(fast_convoy.iloc[0]["send_needed_pct"]) <= float(
            verify_config["legacy_send_needed_pct"]
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
        f"- Run-level schema, pedestrian, position, speed, and timing gates: {'PASS' if bool(runs['pass'].all()) else 'FAIL'}",
        f"- Direct same-frame object-row match coverage: {direct_match_coverage:.2f}% (reported diagnostically; not compared to held-track replay coverage)",
        f"- Pedestrian matched rows: {int(ped_coverage.iloc[0]['matched_rows']) if not ped_coverage.empty else 0}",
    ]
    if not skip_surrogate:
        overall_frames = int(surrogate["frames"].sum())
        report.extend(
            [
                f"- Send-needed: {_safe_pct(int(surrogate['send_needed_frames'].sum()), overall_frames):.2f}% (legacy 14.66%)",
                f"- Selected SPLIT: {_safe_pct(int(surrogate['selected_split_frames'].sum()), overall_frames):.2f}% (legacy 5.83%)",
                f"- Infeasible/over-budget frames: {_safe_pct(int(surrogate['infeasible_frames'].sum()), overall_frames):.2f}%",
                f"- Shield OOD frames: {_safe_pct(int(surrogate['shield_ood_frames'].sum()), overall_frames):.2f}%",
                f"- Capture attempts: {_safe_pct(int(surrogate['capture_attempt_frames'].sum()), overall_frames):.2f}%",
                f"- Vehicle replay observation coverage: {_safe_pct(int(surrogate['observed_vehicle_objects'].sum()), int(surrogate['truth_vehicle_objects'].sum())):.2f}% (vehicle-only legacy 45.18%)",
                f"- Pedestrian replay observation coverage: {_safe_pct(int(surrogate['observed_pedestrian_objects'].sum()), int(surrogate['truth_pedestrian_objects'].sum())):.2f}% ({int(surrogate['observed_pedestrian_objects'].sum())} observed object-frames)",
            ]
        )
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
            "This is corpus and surrogate verification, not a live safety guarantee. A failed batch is quarantined and must not be used for policy training or headline evaluation.",
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
    args = parser.parse_args()
    print(verify(args.batch_dir, args.skip_surrogate))


if __name__ == "__main__":
    main()
