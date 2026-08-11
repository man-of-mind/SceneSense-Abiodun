#!/usr/bin/env python3
"""Controller-independent freshness rescore for an immutable policy corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_agent.policy.config import load_config
from rl_agent.policy.replay import (
    _greedy_prediction_matches,
    _normalize_class,
    discover_trace_registry,
    load_trace_episode,
)
from rl_agent.policy.types import SceneFrame

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "freshness_rescore_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_pct(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _quantile(values: Iterable[float], q: float) -> float | None:
    series = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna()
    if not len(series):
        return None
    ordered = np.sort(series.to_numpy(dtype=float))
    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    if lower_index == upper_index or lower == upper:
        return lower
    if math.isinf(lower) or math.isinf(upper):
        return upper if position > lower_index else lower
    fraction = position - lower_index
    return lower + fraction * (upper - lower)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _single_csv(run_dir: Path, suffix: str) -> Path:
    matches = sorted((run_dir / "streams").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} under {run_dir}, found {len(matches)}")
    return matches[0]


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("freshness rescore config schema_version must be 1")
    return value


def _validate_reference(config: Mapping[str, object]) -> None:
    spec = config["freshness"]
    catalog_path = REPO_ROOT / str(config["provenance"]["action_catalog_csv"])
    catalog = pd.read_csv(catalog_path)
    row = catalog[catalog["profile_id"] == str(spec["reference_profile_id"])]
    if len(row) != 1:
        raise ValueError("reference profile is not unique in the action catalog")
    measured = float(row.iloc[0]["base_loc_calibrated_m"])
    declared = float(spec["reference_base_loc_m"])
    if not math.isclose(measured, declared, abs_tol=1e-9):
        raise ValueError(f"declared base_loc {declared} differs from catalog {measured}")
    if declared >= float(spec["epsilon_m"]):
        raise ValueError("reference base localization must be below epsilon")


def _split_manifest(batch_manifest: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for item in batch_manifest["runs"]:
        if str(item["status"]) not in {"complete", "complete_with_teardown_warning"}:
            raise ValueError(f"incomplete input run: {item['episode_id']} ({item['status']})")
        rows.append(
            {
                "episode_id": str(item["episode_id"]),
                "run_group": str(item["run_group"]),
                "scenario_family": str(item["scenario_family"]),
                "scenario_variant": str(item.get("scenario_variant", item["scenario_family"])),
                "split": str(item["split"]),
                "run_dir": str(item["run_dir"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["episode_id"].duplicated().any():
        raise ValueError("batch manifest contains duplicate episode IDs")
    return frame


def _policy_records(batch_dir: Path, split_path: Path, range_m: float) -> Dict[str, Tuple[object, List[SceneFrame]]]:
    policy_config = copy.deepcopy(load_config())
    policy_config["replay"]["roots"] = [str(batch_dir / "runs")]
    policy_config["replay"]["split_manifest_csv"] = str(split_path)
    policy_config["replay"]["max_episode_steps"] = 2000
    records = discover_trace_registry(policy_config)
    return {
        record.episode_id: (
            record,
            load_trace_episode(record, policy_config, range_m=range_m, max_steps=2000),
        )
        for record in records
    }


def frames_to_object_table(
    frames: Sequence[SceneFrame],
    metadata: Mapping[str, object],
    pedestrian_speed_max_mps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    exclusions: List[Dict[str, object]] = []
    for frame in frames:
        observed_keys = {obj.track_key for obj in frame.observed_objects}
        for obj in frame.truth_objects:
            row = {
                "episode_id": str(metadata["episode_id"]),
                "scenario_family": str(metadata["scenario_family"]),
                "scenario_variant": str(metadata["scenario_variant"]),
                "split": str(metadata["split"]),
                "step_index": int(frame.step_index),
                "timestamp_s": float(frame.timestamp_s),
                "actor_id": int(obj.track_key[1]),
                "class_name": _normalize_class(obj.class_name),
                "speed_mps": float(obj.speed_mps),
                "observed": obj.track_key in observed_keys,
            }
            if row["class_name"] == "pedestrian" and row["speed_mps"] > pedestrian_speed_max_mps:
                exclusions.append({**row, "reason": "resampled_pedestrian_speed_above_qc_max"})
                continue
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def score_seeded_view(
    objects: pd.DataFrame,
    view: str,
    epsilon_m: float,
    base_loc_m: float,
    hz: float,
    liveness_ticks: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if view not in {"gt_seeded_motion_only", "detection_seeded_deployable"}:
        raise ValueError(f"unknown freshness view: {view}")
    keys = ["episode_id", "actor_id"]
    if view == "gt_seeded_motion_only":
        seeds = objects.groupby(keys)["timestamp_s"].min().rename("seed_timestamp_s")
    else:
        seeds = (
            objects.loc[objects["observed"]]
            .groupby(keys)["timestamp_s"]
            .min()
            .rename("seed_timestamp_s")
        )
    scored = objects.merge(seeds.reset_index(), on=keys, how="left")
    scored["view"] = view
    scored["mapped"] = scored["seed_timestamp_s"].notna() & (
        scored["timestamp_s"] >= scored["seed_timestamp_s"] - 1e-12
    )
    scored["aoi_s"] = np.where(
        scored["mapped"], scored["timestamp_s"] - scored["seed_timestamp_s"], np.nan
    )
    displacement_budget = math.sqrt(max(0.0, epsilon_m**2 - base_loc_m**2))
    speed = scored["speed_mps"].to_numpy(dtype=float)
    freshness_budget = np.full(len(scored), np.inf, dtype=float)
    np.divide(
        displacement_budget,
        speed,
        out=freshness_budget,
        where=speed > 1e-12,
    )
    scored["freshness_budget_s"] = freshness_budget
    scored["remaining_to_breach_s"] = np.where(
        scored["mapped"], scored["freshness_budget_s"] - scored["aoi_s"], np.nan
    )
    scored["localization_error_m"] = np.where(
        scored["mapped"], np.hypot(base_loc_m, speed * scored["aoi_s"]), np.nan
    )
    scored["mapped_over_epsilon"] = scored["mapped"] & (
        scored["localization_error_m"] > epsilon_m + 1e-12
    )
    scored["strict_gt_unsafe"] = ~scored["mapped"] | scored["mapped_over_epsilon"]
    scored["mapped_fresh"] = scored["mapped"] & ~scored["mapped_over_epsilon"]
    for ticks in liveness_ticks:
        horizon = float(ticks) / hz
        scored[f"near_breach_{ticks}tick"] = scored["mapped_fresh"] & (
            scored["remaining_to_breach_s"] <= horizon + 1e-12
        )

    track_rows: List[Dict[str, object]] = []
    for (_episode_id, _actor_id), group in scored.groupby(keys, sort=False):
        group = group.sort_values("timestamp_s")
        seeded = bool(group["mapped"].any())
        mapped = group[group["mapped"]]
        first = group.iloc[0]
        row: Dict[str, object] = {
            "view": view,
            "episode_id": str(first["episode_id"]),
            "scenario_family": str(first["scenario_family"]),
            "scenario_variant": str(first["scenario_variant"]),
            "split": str(first["split"]),
            "actor_id": int(first["actor_id"]),
            "class_name": str(first["class_name"]),
            "seeded": seeded,
            "truth_object_frames": int(len(group)),
            "pre_seed_or_never_mapped_frames": int((~group["mapped"]).sum()),
        }
        if not seeded:
            row.update(
                {
                    "seed_timestamp_s": None,
                    "initial_speed_mps": None,
                    "initial_freshness_budget_s": None,
                    "breached": False,
                    "right_censored": False,
                    "time_to_first_sampled_breach_s": None,
                    "censor_time_s": None,
                    "event_or_censor_time_s": None,
                }
            )
        else:
            seed_time = float(mapped.iloc[0]["seed_timestamp_s"])
            breaches = mapped[mapped["mapped_over_epsilon"]]
            breached = not breaches.empty
            last_time = float(mapped["timestamp_s"].max())
            breach_time = (
                float(breaches.iloc[0]["timestamp_s"]) - seed_time if breached else None
            )
            censor_time = max(0.0, last_time - seed_time) if not breached else None
            row.update(
                {
                    "seed_timestamp_s": seed_time,
                    "initial_speed_mps": float(mapped.iloc[0]["speed_mps"]),
                    "initial_freshness_budget_s": float(mapped.iloc[0]["freshness_budget_s"]),
                    "breached": breached,
                    "right_censored": not breached,
                    "time_to_first_sampled_breach_s": breach_time,
                    "censor_time_s": censor_time,
                    "event_or_censor_time_s": breach_time if breached else censor_time,
                }
            )
        track_rows.append(row)
    return scored, pd.DataFrame(track_rows)


def dwell_segments(
    objects: pd.DataFrame,
    hz: float,
    slow_max_mps: float,
    fast_thresholds: Sequence[float],
) -> pd.DataFrame:
    regimes = [(f"slow_le_{slow_max_mps:g}", "le", slow_max_mps)] + [
        (f"fast_ge_{value:g}", "ge", value) for value in fast_thresholds
    ]
    rows: List[Dict[str, object]] = []
    max_gap = 1.5 / hz
    for (_episode_id, _actor_id), group in objects.groupby(["episode_id", "actor_id"], sort=False):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        first = group.iloc[0]
        for regime, comparator, threshold in regimes:
            flag = (
                group["speed_mps"].to_numpy(dtype=float) <= threshold + 1e-12
                if comparator == "le"
                else group["speed_mps"].to_numpy(dtype=float) >= threshold - 1e-12
            )
            start: int | None = None
            for index in range(len(group) + 1):
                active = index < len(group) and bool(flag[index])
                contiguous = (
                    index == 0
                    or float(group.iloc[index]["timestamp_s"] - group.iloc[index - 1]["timestamp_s"])
                    <= max_gap + 1e-12
                ) if index < len(group) else False
                if active and start is None:
                    start = index
                elif start is not None and (not active or not contiguous):
                    end = index - 1
                    count = end - start + 1
                    rows.append(
                        {
                            "episode_id": str(first["episode_id"]),
                            "scenario_family": str(first["scenario_family"]),
                            "scenario_variant": str(first["scenario_variant"]),
                            "split": str(first["split"]),
                            "actor_id": int(first["actor_id"]),
                            "class_name": str(first["class_name"]),
                            "regime": regime,
                            "threshold_mps": float(threshold),
                            "start_timestamp_s": float(group.iloc[start]["timestamp_s"]),
                            "end_timestamp_s": float(group.iloc[end]["timestamp_s"]),
                            "object_frames": int(count),
                            "dwell_s": float(count / hz),
                        }
                    )
                    start = index if active else None
    return pd.DataFrame(rows)


def _km_quantile(durations: Sequence[float], events: Sequence[bool], q: float) -> float | None:
    if not durations:
        return None
    frame = pd.DataFrame({"duration": durations, "event": events}).sort_values("duration")
    survival = 1.0
    for duration, group in frame.groupby("duration", sort=True):
        at_risk = int((frame["duration"] >= duration - 1e-12).sum())
        event_count = int(group["event"].sum())
        if at_risk:
            survival *= 1.0 - event_count / at_risk
        if 1.0 - survival >= q - 1e-12:
            return float(duration)
    return None


def _group_specs() -> Sequence[Tuple[str, Sequence[str]]]:
    return (
        ("run", ["episode_id", "scenario_family", "scenario_variant", "split"]),
        ("family_variant", ["scenario_family", "scenario_variant"]),
        ("split", ["split"]),
        ("corpus", []),
    )


def _iter_groups(frame: pd.DataFrame, columns: Sequence[str]):
    if columns:
        grouper = columns[0] if len(columns) == 1 else list(columns)
        for key, group in frame.groupby(grouper, dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            yield dict(zip(columns, values)), group
    else:
        yield {}, frame


def summarize_speed(
    objects: pd.DataFrame,
    dwell: pd.DataFrame,
    slow_max: float,
    fast_thresholds: Sequence[float],
    regime_min_dwell_s: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scope, columns in _group_specs():
        for group_values, outer in _iter_groups(objects, columns):
            for class_name, group in outer.groupby("class_name", sort=True):
                values = group["speed_mps"]
                row: Dict[str, object] = {
                    "scope": scope,
                    **group_values,
                    "class_name": class_name,
                    "tracks": int(group[["episode_id", "actor_id"]].drop_duplicates().shape[0]),
                    "object_frames": int(len(group)),
                    "speed_p10_mps": _quantile(values, 0.10),
                    "speed_p50_mps": _quantile(values, 0.50),
                    "speed_p90_mps": _quantile(values, 0.90),
                    "speed_max_mps": _quantile(values, 1.0),
                    "slow_object_frame_pct": _safe_pct((values <= slow_max).sum(), len(values)),
                }
                for threshold in fast_thresholds:
                    row[f"fast_ge_{threshold:g}_object_frame_pct"] = _safe_pct(
                        (values >= threshold).sum(), len(values)
                    )
                segment_subset = dwell[dwell["class_name"] == class_name]
                for name, value in group_values.items():
                    if name in segment_subset.columns:
                        segment_subset = segment_subset[segment_subset[name] == value]
                for regime in [f"slow_le_{slow_max:g}"] + [
                    f"fast_ge_{value:g}" for value in fast_thresholds
                ]:
                    selected = segment_subset[segment_subset["regime"] == regime]
                    row[f"{regime}_max_dwell_s"] = _quantile(selected["dwell_s"], 1.0)
                    row[f"{regime}_runs_ge_min_dwell"] = int(
                        selected.loc[selected["dwell_s"] >= regime_min_dwell_s, "episode_id"].nunique()
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_freshness(scored: pd.DataFrame, liveness_ticks: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scope, columns in _group_specs():
        for group_values, outer in _iter_groups(scored, columns):
            for view, view_group in outer.groupby("view", sort=True):
                class_groups = [("all", view_group)]
                class_groups.extend(view_group.groupby("class_name", sort=True))
                for class_name, group in class_groups:
                    mapped = group[group["mapped"]]
                    frame_keys = ["episode_id", "step_index"]
                    all_frames = group[frame_keys].drop_duplicates()
                    mapped_frames = mapped[frame_keys].drop_duplicates()
                    row: Dict[str, object] = {
                        "scope": scope,
                        **group_values,
                        "view": view,
                        "class_name": class_name,
                        "truth_object_frames": int(len(group)),
                        "mapped_object_frames": int(len(mapped)),
                        "unmapped_object_frames": int((~group["mapped"]).sum()),
                        "mapped_over_epsilon_object_frames": int(group["mapped_over_epsilon"].sum()),
                        "strict_gt_unsafe_object_frames": int(group["strict_gt_unsafe"].sum()),
                        "mapped_pressure_pct": _safe_pct(
                            group["mapped_over_epsilon"].sum(), len(mapped)
                        ),
                        "strict_gt_unsafe_pct": _safe_pct(
                            group["strict_gt_unsafe"].sum(), len(group)
                        ),
                        "all_in_scope_frames": int(len(all_frames)),
                        "frames_with_mapped_object": int(len(mapped_frames)),
                    }
                    for ticks in liveness_ticks:
                        near_frames = group.loc[
                            group[f"near_breach_{ticks}tick"], frame_keys
                        ].drop_duplicates()
                        row[f"near_breach_{ticks}tick_frames"] = int(len(near_frames))
                        row[f"near_breach_{ticks}tick_pct_in_scope_frames"] = _safe_pct(
                            len(near_frames), len(all_frames)
                        )
                        row[f"near_breach_{ticks}tick_pct_mapped_frames"] = _safe_pct(
                            len(near_frames), len(mapped_frames)
                        )
                    breached_frames = group.loc[
                        group["mapped_over_epsilon"], frame_keys
                    ].drop_duplicates()
                    row["already_breached_frames"] = int(len(breached_frames))
                    row["already_breached_pct_in_scope_frames"] = _safe_pct(
                        len(breached_frames), len(all_frames)
                    )
                    row["already_breached_pct_mapped_frames"] = _safe_pct(
                        len(breached_frames), len(mapped_frames)
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_breach_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seeded = tracks[tracks["seeded"]].copy()
    for scope, columns in _group_specs():
        for group_values, outer in _iter_groups(seeded, columns):
            for view, view_group in outer.groupby("view", sort=True):
                class_groups = [("all", view_group)]
                class_groups.extend(view_group.groupby("class_name", sort=True))
                for class_name, group in class_groups:
                    duration = group["event_or_censor_time_s"].astype(float).tolist()
                    event = group["breached"].astype(bool).tolist()
                    breached_times = group.loc[
                        group["breached"], "time_to_first_sampled_breach_s"
                    ]
                    rows.append(
                        {
                            "scope": scope,
                            **group_values,
                            "view": view,
                            "class_name": class_name,
                            "seeded_tracks": int(len(group)),
                            "breached_tracks": int(group["breached"].sum()),
                            "right_censored_tracks": int(group["right_censored"].sum()),
                            "breached_track_pct": _safe_pct(
                                group["breached"].sum(), len(group)
                            ),
                            "initial_budget_p10_s": _quantile(
                                group["initial_freshness_budget_s"], 0.10
                            ),
                            "initial_budget_p50_s": _quantile(
                                group["initial_freshness_budget_s"], 0.50
                            ),
                            "initial_budget_p90_s": _quantile(
                                group["initial_freshness_budget_s"], 0.90
                            ),
                            "observed_breach_p10_s": _quantile(breached_times, 0.10),
                            "observed_breach_p50_s": _quantile(breached_times, 0.50),
                            "observed_breach_p90_s": _quantile(breached_times, 0.90),
                            "km_breach_p10_s": _km_quantile(duration, event, 0.10),
                            "km_breach_p50_s": _km_quantile(duration, event, 0.50),
                            "km_breach_p90_s": _km_quantile(duration, event, 0.90),
                        }
                    )
    return pd.DataFrame(rows)


def _direct_coverage(
    run_dir: Path,
    metadata: Mapping[str, object],
    range_m: float,
    score_min: float,
    association_gate_m: float,
    pedestrian_speed_max_mps: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    gt_path = _single_csv(run_dir, "_object_ground_truth.csv")
    prediction_path = _single_csv(run_dir, "_object_predictions.csv")
    gt = pd.read_csv(gt_path)
    predictions = pd.read_csv(prediction_path)
    gt["class_name"] = gt["class_name"].map(_normalize_class)
    gt["world_x"] = pd.to_numeric(gt["origin_x"], errors="coerce")
    gt["world_y"] = pd.to_numeric(gt["origin_y"], errors="coerce")
    eligible = gt[_truthy(gt["in_camera_frustum"])].copy()
    eligible = eligible[pd.to_numeric(eligible["distance_m"], errors="coerce") <= range_m].copy()
    predictions["class_name"] = predictions["class_name"].map(_normalize_class)
    scores = pd.to_numeric(
        predictions.get("score", pd.Series(1.0, index=predictions.index)), errors="coerce"
    )
    predictions = predictions[scores >= score_min].copy()
    matches = _greedy_prediction_matches(eligible, predictions, association_gate_m)
    coverage: List[Dict[str, object]] = []
    for class_name in sorted(set(eligible["class_name"]) | {"vehicle", "pedestrian"}):
        class_gt = eligible[eligible["class_name"] == class_name]
        class_matches = matches[matches["class_name"] == class_name] if not matches.empty else matches
        matched_timestamps = set(class_matches["timestamp"].astype(float)) if len(class_matches) else set()
        matched_frames = int(
            class_gt.loc[class_gt["carla_timestamp"].astype(float).isin(matched_timestamps), "frame_id"].nunique()
        )
        coverage.append(
            {
                **{name: metadata[name] for name in (
                    "episode_id", "scenario_family", "scenario_variant", "split"
                )},
                "class_name": class_name,
                "eligible_gt_rows": int(len(class_gt)),
                "matched_rows": int(len(class_matches)),
                "eligible_frames": int(class_gt["frame_id"].nunique()),
                "matched_frames": matched_frames,
                "ground_truth_sha256": _sha256(gt_path),
                "prediction_sha256": _sha256(prediction_path),
            }
        )
    raw_qc: List[Dict[str, object]] = []
    pedestrian = gt[gt["class_name"] == "pedestrian"].copy()
    for _actor_id, group in pedestrian.groupby("actor_id"):
        group = group.sort_values("carla_timestamp")
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        step = np.hypot(
            pd.to_numeric(group["origin_x"], errors="coerce").diff(),
            pd.to_numeric(group["origin_y"], errors="coerce").diff(),
        )
        speed = step / dt
        for index in group.index[(speed > pedestrian_speed_max_mps) & np.isfinite(speed)]:
            item = group.loc[index]
            raw_qc.append(
                {
                    **{name: metadata[name] for name in (
                        "episode_id", "scenario_family", "scenario_variant", "split"
                    )},
                    "frame_id": int(item["frame_id"]),
                    "carla_timestamp": float(item["carla_timestamp"]),
                    "actor_id": int(item["actor_id"]),
                    "class_name": "pedestrian",
                    "forward_displacement_speed_mps": float(speed.loc[index]),
                    "origin_x": float(item["origin_x"]),
                    "origin_y": float(item["origin_y"]),
                    "origin_z": float(item.get("origin_z", np.nan)),
                    "distance_m": float(item["distance_m"]),
                    "in_camera_frustum": bool(_truthy(pd.Series([item["in_camera_frustum"]])).iloc[0]),
                    "inside_headline_scope": bool(
                        _truthy(pd.Series([item["in_camera_frustum"]])).iloc[0]
                        and float(item["distance_m"]) <= range_m
                    ),
                    "disposition": "flag_raw_only_preserve_source",
                }
            )
    return coverage, raw_qc


def summarize_coverage(run_coverage: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    replay_rows = []
    for keys, group in objects.groupby(
        ["episode_id", "scenario_family", "scenario_variant", "split", "class_name"],
        sort=True,
    ):
        truth_tracks = group[["episode_id", "actor_id"]].drop_duplicates()
        detected_tracks = group.loc[group["observed"], ["episode_id", "actor_id"]].drop_duplicates()
        replay_rows.append(
            {
                "episode_id": keys[0], "scenario_family": keys[1],
                "scenario_variant": keys[2], "split": keys[3], "class_name": keys[4],
                "truth_object_frames": int(len(group)),
                "observed_object_frames": int(group["observed"].sum()),
                "truth_tracks": int(len(truth_tracks)),
                "ever_detected_tracks": int(len(detected_tracks)),
            }
        )
    replay = pd.DataFrame(replay_rows)
    run = run_coverage.merge(
        replay,
        on=["episode_id", "scenario_family", "scenario_variant", "split", "class_name"],
        how="left",
    ).fillna(
        {"truth_object_frames": 0, "observed_object_frames": 0, "truth_tracks": 0, "ever_detected_tracks": 0}
    )
    rows = []
    metric_columns = [
        "eligible_gt_rows", "matched_rows", "eligible_frames", "matched_frames",
        "truth_object_frames", "observed_object_frames", "truth_tracks", "ever_detected_tracks",
    ]
    for scope, columns in _group_specs():
        for values, group in _iter_groups(run, columns):
            for class_name, class_group in group.groupby("class_name", sort=True):
                sums = class_group[metric_columns].sum()
                rows.append(
                    {
                        "scope": scope,
                        **values,
                        "class_name": class_name,
                        **{name: int(sums[name]) for name in metric_columns},
                        "direct_object_row_coverage_pct": _safe_pct(sums["matched_rows"], sums["eligible_gt_rows"]),
                        "direct_frame_coverage_pct": _safe_pct(sums["matched_frames"], sums["eligible_frames"]),
                        "replay_observation_coverage_pct": _safe_pct(
                            sums["observed_object_frames"], sums["truth_object_frames"]
                        ),
                        "ever_detected_track_pct": _safe_pct(sums["ever_detected_tracks"], sums["truth_tracks"]),
                        "never_detected_tracks": int(sums["truth_tracks"] - sums["ever_detected_tracks"]),
                    }
                )
    return pd.DataFrame(rows)


def regime_split_coverage(
    dwell: pd.DataFrame,
    split_manifest: pd.DataFrame,
    min_dwell_s: float,
    minimum_runs: int,
) -> pd.DataFrame:
    rows = []
    classes = sorted(dwell["class_name"].unique())
    regimes = sorted(dwell["regime"].unique())
    for split in ("train", "validation", "test"):
        total_runs = int((split_manifest["split"] == split).sum())
        for class_name in classes:
            for regime in regimes:
                selected = dwell[
                    (dwell["split"] == split)
                    & (dwell["class_name"] == class_name)
                    & (dwell["regime"] == regime)
                    & (dwell["dwell_s"] >= min_dwell_s)
                ]
                count = int(selected["episode_id"].nunique())
                rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "regime": regime,
                        "descriptive_min_dwell_s": min_dwell_s,
                        "batch_runs_in_split": total_runs,
                        "runs_with_regime": count,
                        "minimum_runs_heuristic": minimum_runs,
                        "meets_two_run_heuristic": count >= minimum_runs,
                    }
                )
    return pd.DataFrame(rows)


def regime_concentration(objects: pd.DataFrame, slow_max: float, fast_threshold: float) -> pd.DataFrame:
    rows = []
    for class_name, group in objects.groupby("class_name", sort=True):
        for regime, mask in (
            (f"slow_le_{slow_max:g}", group["speed_mps"] <= slow_max),
            (f"fast_ge_{fast_threshold:g}", group["speed_mps"] >= fast_threshold),
        ):
            selected = group[mask]
            counts = selected.groupby("episode_id").size().sort_values(ascending=False)
            total = int(counts.sum())
            rows.append(
                {
                    "class_name": class_name,
                    "regime": regime,
                    "object_frames": total,
                    "contributing_runs": int(len(counts)),
                    "top_1_run_fraction_pct": _safe_pct(counts.iloc[:1].sum(), total),
                    "top_2_runs_fraction_pct": _safe_pct(counts.iloc[:2].sum(), total),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.dropna(axis=1, how="all")
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def _latest_prior_verification(batch_dir: Path, expected_status: str) -> Path:
    candidates = []
    for path in (batch_dir / "verification").glob("*/verification_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(manifest.get("status")) == expected_status:
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(f"no prior {expected_status} verification under {batch_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def rescore(batch_dir: Path, config_path: Path, output_dir: Path | None = None) -> Path:
    batch_dir = batch_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    _validate_reference(config)
    freshness = config["freshness"]
    speed_config = config["speed"]
    matching = config["matching"]
    heuristic = config["human_heuristic"]
    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    split_frame = _split_manifest(batch_manifest)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = batch_dir / "freshness_rescore" / timestamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    split_path = output_dir / "replay_split_manifest.csv"
    split_frame.drop(columns=["run_dir"]).to_csv(split_path, index=False)
    prior_dir = _latest_prior_verification(
        batch_dir, str(config["provenance"]["prior_verification_status"])
    )

    records = _policy_records(batch_dir, split_path, float(freshness["headline_range_m"]))
    object_frames: List[pd.DataFrame] = []
    resampled_exclusions: List[pd.DataFrame] = []
    direct_rows: List[Dict[str, object]] = []
    raw_qc_rows: List[Dict[str, object]] = []
    for item in split_frame.to_dict(orient="records"):
        episode_id = str(item["episode_id"])
        if episode_id not in records:
            raise ValueError(f"replay loader did not discover {episode_id}")
        _record, frames = records[episode_id]
        frame, excluded = frames_to_object_table(
            frames, item, float(speed_config["pedestrian_qc_max_mps"])
        )
        object_frames.append(frame)
        if not excluded.empty:
            resampled_exclusions.append(excluded)
        coverage, qc = _direct_coverage(
            Path(str(item["run_dir"])), item,
            float(freshness["headline_range_m"]),
            float(matching["prediction_score_min"]),
            float(matching["association_gate_m"]),
            float(speed_config["pedestrian_qc_max_mps"]),
        )
        direct_rows.extend(coverage)
        raw_qc_rows.extend(qc)
    objects = pd.concat(object_frames, ignore_index=True)
    if objects.empty:
        raise ValueError("no in-scope object frames were produced")
    raw_qc = pd.DataFrame(raw_qc_rows)
    raw_qc.to_csv(output_dir / "raw_qc_flags.csv", index=False)
    resampled_qc = (
        pd.concat(resampled_exclusions, ignore_index=True)
        if resampled_exclusions else pd.DataFrame(columns=list(objects.columns) + ["reason"])
    )
    resampled_qc.to_csv(output_dir / "scored_qc_exclusions.csv", index=False)

    views = []
    track_views = []
    for view in ("gt_seeded_motion_only", "detection_seeded_deployable"):
        scored, tracks = score_seeded_view(
            objects, view,
            float(freshness["epsilon_m"]),
            float(freshness["reference_base_loc_m"]),
            float(freshness["clock_hz"]),
            [int(value) for value in freshness["liveness_ticks"]],
        )
        views.append(scored)
        track_views.append(tracks)
    scored = pd.concat(views, ignore_index=True)
    tracks = pd.concat(track_views, ignore_index=True)
    scored.to_csv(output_dir / "freshness_object_frames.csv", index=False)
    tracks.to_csv(output_dir / "freshness_tracks.csv", index=False)

    fast_thresholds = [float(value) for value in speed_config["fast_thresholds_mps"]]
    slow_max = float(speed_config["slow_max_mps"])
    min_dwell = float(speed_config["descriptive_regime_min_dwell_s"])
    dwell = dwell_segments(objects, float(freshness["clock_hz"]), slow_max, fast_thresholds)
    dwell.to_csv(output_dir / "speed_dwell_segments.csv", index=False)
    speed_summary = summarize_speed(objects, dwell, slow_max, fast_thresholds, min_dwell)
    speed_summary.to_csv(output_dir / "speed_summary.csv", index=False)
    freshness_summary = summarize_freshness(
        scored, [int(value) for value in freshness["liveness_ticks"]]
    )
    freshness_summary.to_csv(output_dir / "freshness_summary.csv", index=False)
    breach_summary = summarize_breach_tracks(tracks)
    breach_summary.to_csv(output_dir / "breach_time_summary.csv", index=False)
    coverage = summarize_coverage(pd.DataFrame(direct_rows), objects)
    coverage.to_csv(output_dir / "detection_coverage.csv", index=False)
    split_regimes = regime_split_coverage(
        dwell, split_frame, min_dwell, int(heuristic["minimum_runs_per_split"])
    )
    split_regimes.to_csv(output_dir / "regime_split_coverage.csv", index=False)
    concentration = regime_concentration(
        objects, slow_max, float(heuristic["sustained_fast_threshold_mps"])
    )
    concentration.to_csv(output_dir / "regime_concentration.csv", index=False)

    corpus_speed = speed_summary[speed_summary["scope"] == "corpus"]
    corpus_freshness = freshness_summary[freshness_summary["scope"] == "corpus"]
    corpus_breach = breach_summary[breach_summary["scope"] == "corpus"]
    family_speed = speed_summary[speed_summary["scope"] == "family_variant"]
    family_freshness = freshness_summary[freshness_summary["scope"] == "family_variant"]
    family_breach = breach_summary[breach_summary["scope"] == "family_variant"]
    corpus_coverage = coverage[coverage["scope"] == "corpus"]
    per_run_speed = speed_summary[speed_summary["scope"] == "run"]
    vehicle_run = per_run_speed[per_run_speed["class_name"] == "vehicle"].copy()
    run_pressure = freshness_summary[
        (freshness_summary["scope"] == "run")
        & (freshness_summary["view"] == "gt_seeded_motion_only")
        & (freshness_summary["class_name"] == "all")
    ].copy()
    run_pressure = run_pressure.rename(
        columns={
            "truth_object_frames": "gt_object_frames",
            "mapped_over_epsilon_object_frames": "gt_over_epsilon_object_frames",
            "mapped_pressure_pct": "gt_pressure_pct",
        }
    )
    per_run_report = vehicle_run.merge(
        run_pressure,
        on=["episode_id", "scenario_family", "scenario_variant", "split"],
        how="outer",
        suffixes=("_vehicle", ""),
    )
    per_run_columns = [
        "episode_id", "scenario_family", "scenario_variant", "split",
        "speed_p10_mps", "speed_p50_mps", "speed_p90_mps", "speed_max_mps",
        "slow_object_frame_pct", "fast_ge_10_object_frame_pct", "fast_ge_10_max_dwell_s",
        "gt_pressure_pct", "near_breach_3tick_frames", "near_breach_3tick_pct_in_scope_frames",
        "near_breach_5tick_frames", "near_breach_5tick_pct_in_scope_frames",
        "near_breach_10tick_frames", "near_breach_10tick_pct_in_scope_frames",
    ]
    per_run_report = per_run_report[[column for column in per_run_columns if column in per_run_report]]
    per_run_report.to_csv(output_dir / "per_run_human_review.csv", index=False)

    prior_report = prior_dir / "CORPUS_VERIFICATION.md"
    prior_manifest = prior_dir / "verification_manifest.json"
    fast_regime = f"fast_ge_{float(heuristic['sustained_fast_threshold_mps']):g}"
    split_fast = split_regimes[
        (split_regimes["class_name"] == "vehicle") & (split_regimes["regime"] == fast_regime)
    ]
    split_slow = split_regimes[split_regimes["regime"] == f"slow_le_{slow_max:g}"]
    report = [
        "# Policy corpus freshness rescore",
        "",
        "**Status: HUMAN_REVIEW_REQUIRED**",
        "",
        f"Batch: `{batch_dir}`",
        f"Prior immutable verification: `{prior_report}` (`{config['provenance']['prior_verification_status']}`)",
        "",
        "This report supersedes the old send-needed metric only for the corpus salvage/top-up disposition. It does not rewrite the prior report or its pre-registered result.",
        "",
        "## Locked counterfactual",
        "",
        f"- epsilon={float(freshness['epsilon_m']):g} m; range<={float(freshness['headline_range_m']):g} m; {float(freshness['clock_hz']):g} Hz.",
        f"- Fixed `{freshness['reference_profile_id']}` base localization={float(freshness['reference_base_loc_m']):.2f} m.",
        "- Seed once instantaneously, then never resend; no channel process is used.",
        "- A channel-forced SKIP is not labelled safe/free here; it remains a distinct flagged over-budget outcome.",
        "- GT-seeded motion-only and detection-seeded deployable/strict-GT views are reported separately.",
        "- Near-breach excludes already-breached frames; time-to-breach summaries retain right-censored tracks.",
        "",
        "## Corpus speed tails by class",
        "",
        _markdown_table(corpus_speed),
        "",
        "## Counterfactual pressure and liveness by scoring view and class",
        "",
        _markdown_table(corpus_freshness),
        "",
        "## Per-object time to epsilon breach (right-censor aware)",
        "",
        _markdown_table(corpus_breach),
        "",
        "## Detection coverage by class",
        "",
        _markdown_table(corpus_coverage),
        "",
        "Pedestrian coverage is an independent gating concern for any pedestrian-freshness claim; no post-hoc automatic threshold is applied here.",
        "",
        "## Human salvage/top-up heuristic (not an automatic verdict)",
        "",
        "1. Inspect whether speeds span abundant slow object-frames through a non-trivial sustained vehicle >=10 m/s tail.",
        "2. Inspect whether GT-seeded pressure is materially above zero.",
        "3. Require each needed regime in at least two runs in train, validation, and test; top up only a missing regime.",
        "",
        "### Sustained-fast split distribution",
        "",
        _markdown_table(split_fast),
        "",
        "### Slow-regime split distribution",
        "",
        _markdown_table(split_slow),
        "",
        "## Regime concentration",
        "",
        _markdown_table(concentration),
        "",
        "## Per-family speed",
        "",
        _markdown_table(family_speed),
        "",
        "## Per-family pressure and liveness by scoring view",
        "",
        _markdown_table(family_freshness),
        "",
        "## Per-family time to epsilon breach",
        "",
        _markdown_table(family_breach),
        "",
        "## Per-run pressure and vehicle fast tail",
        "",
        _markdown_table(per_run_report),
        "",
        "## Raw pedestrian QC flags",
        "",
        f"Raw flags: {len(raw_qc)}; inside the <=25 m headline scope: {int(raw_qc['inside_headline_scope'].sum()) if len(raw_qc) else 0}. Raw source rows were not edited.",
        "",
        _markdown_table(raw_qc),
        "",
        "## Interpretation guardrail",
        "",
        "The skip-only reference is an idealized controller-independent motion diagnostic, not an executable policy and not a safety guarantee. Abiodun and local Claude make the salvage-versus-targeted-top-up decision from these distributions.",
        "",
    ]
    report_path = output_dir / "FRESHNESS_RESCORE.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    resolved_config = output_dir / "resolved_freshness_rescore_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest: MutableMapping[str, object] = {
        "schema": "policy_corpus_freshness_rescore.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_REVIEW_REQUIRED",
        "batch_dir": str(batch_dir),
        "batch_manifest_sha256": _sha256(batch_manifest_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "prior_verification_dir": str(prior_dir),
        "prior_verification_report_sha256": _sha256(prior_report),
        "prior_verification_manifest_sha256": _sha256(prior_manifest),
        "input_runs": int(len(split_frame)),
        "scored_object_frames": int(len(objects)),
        "raw_qc_flags": int(len(raw_qc)),
        "scored_qc_exclusions": int(len(resampled_qc)),
        "artifacts": artifacts,
    }
    (output_dir / "freshness_rescore_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(rescore(args.batch_dir, args.config, args.output_dir))


if __name__ == "__main__":
    main()
