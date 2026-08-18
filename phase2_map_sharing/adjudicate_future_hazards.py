"""Evaluation-only future-trajectory hazard and stopping-outcome adjudication.

This module consumes a completed paired replay plus synchronized CARLA truth.
It never writes to runtime directories and never returns truth labels to a
controller.  Matching is class-constrained, one-to-one, and performed afresh
from warning-track coordinates rather than trusting the replay's diagnostic
truth join.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment


SCHEMA = "scenesense.phase2_future_hazard_adjudication.v2"
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ROLE_NAMES = ("helper", "recipient")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_output_name(value: str) -> str:
    name = str(value).strip()
    if not OUTPUT_NAME_PATTERN.fullmatch(name) or not name.startswith(
        "hazard_adjudication"
    ):
        raise ValueError(
            "output name must be one safe basename beginning with "
            f"'hazard_adjudication': {value!r}"
        )
    return name


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _normalize_class(value: object) -> str:
    name = str(value).strip().lower()
    if name in {"person", "walker"}:
        return "pedestrian"
    if name in {"bike", "bicycle"}:
        return "cyclist"
    return name


def _git_value(repository_root: Path, *arguments: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def match_warnings_one_to_one(
    warnings: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    gate_m: float,
) -> dict[int, dict]:
    """Maximum-cardinality, minimum-distance matching within each class."""

    if gate_m <= 0.0 or not math.isfinite(gate_m):
        raise ValueError("gate_m must be finite and positive")
    required_warning = {"class_name", "track_world_x", "track_world_y"}
    required_truth = {"class_name", "actor_id", "origin_x", "origin_y", "role_name"}
    if missing := required_warning - set(warnings.columns):
        raise ValueError(f"warning rows are missing fields: {sorted(missing)}")
    if missing := required_truth - set(truth.columns):
        raise ValueError(f"truth rows are missing fields: {sorted(missing)}")

    results = {
        int(index): {
            "current_truth_matched": 0,
            "current_truth_actor_id": None,
            "current_truth_role_name": None,
            "current_truth_distance_m": None,
        }
        for index in warnings.index
    }
    warning_classes = warnings["class_name"].map(_normalize_class)
    truth_classes = truth["class_name"].map(_normalize_class)
    for class_name in sorted(set(warning_classes)):
        warning_group = warnings[warning_classes == class_name].sort_index()
        truth_group = truth[truth_classes == class_name].copy()
        truth_group["_actor_sort"] = truth_group["actor_id"].astype(str)
        truth_group = truth_group.sort_values("_actor_sort")
        if warning_group.empty or truth_group.empty:
            continue

        warning_xy = warning_group[["track_world_x", "track_world_y"]].to_numpy(
            dtype=float
        )
        truth_xy = truth_group[["origin_x", "origin_y"]].to_numpy(dtype=float)
        distances = np.linalg.norm(
            warning_xy[:, np.newaxis, :] - truth_xy[np.newaxis, :, :], axis=2
        )
        row_count, truth_count = distances.shape
        unmatched_cost = (row_count + truth_count + 1) * float(gate_m)
        invalid_cost = unmatched_cost * 3.0
        cost = np.full(
            (row_count, truth_count + row_count), unmatched_cost, dtype=float
        )
        valid = distances <= float(gate_m) + 1e-12
        cost[:, :truth_count] = np.where(valid, distances, invalid_cost)
        # Stable tie-breaks are far below the measurement precision.
        cost[:, :truth_count] += (
            np.arange(truth_count, dtype=float)[np.newaxis, :] * 1e-12
        )
        assigned_rows, assigned_columns = linear_sum_assignment(cost)
        for row_position, column_position in zip(assigned_rows, assigned_columns):
            if column_position >= truth_count or not valid[row_position, column_position]:
                continue
            warning_index = int(warning_group.index[row_position])
            truth_row = truth_group.iloc[column_position]
            results[warning_index] = {
                "current_truth_matched": 1,
                "current_truth_actor_id": str(truth_row["actor_id"]),
                "current_truth_role_name": str(truth_row["role_name"]),
                "current_truth_distance_m": float(
                    distances[row_position, column_position]
                ),
            }
    return results


def _rectangle_vertices(
    x_m: float,
    y_m: float,
    yaw_deg: float,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    half_length, half_width = float(length_m) / 2.0, float(width_m) / 2.0
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    angle = math.radians(float(yaw_deg))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    return local @ rotation.T + np.asarray([float(x_m), float(y_m)])


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first, second = q - p, r - p
        return float(first[0] * second[1] - first[1] * second[0])

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    if (ab_c > 1e-12 and ab_d < -1e-12 or ab_c < -1e-12 and ab_d > 1e-12) and (
        cd_a > 1e-12 and cd_b < -1e-12 or cd_a < -1e-12 and cd_b > 1e-12
    ):
        return True
    return bool(
        abs(ab_c) <= 1e-12 and on_segment(a, c, b)
        or abs(ab_d) <= 1e-12 and on_segment(a, d, b)
        or abs(cd_a) <= 1e-12 and on_segment(c, a, d)
        or abs(cd_b) <= 1e-12 and on_segment(c, b, d)
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-15:
        return float(np.linalg.norm(point - start))
    fraction = max(
        0.0, min(1.0, float(np.dot(point - start, segment)) / denominator)
    )
    return float(np.linalg.norm(point - (start + fraction * segment)))


def oriented_box_clearance_m(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Exact non-negative 2-D clearance between two oriented rectangles."""

    first_vertices = _rectangle_vertices(*[float(item) for item in first])
    second_vertices = _rectangle_vertices(*[float(item) for item in second])
    first_edges = [
        (first_vertices[index], first_vertices[(index + 1) % 4])
        for index in range(4)
    ]
    second_edges = [
        (second_vertices[index], second_vertices[(index + 1) % 4])
        for index in range(4)
    ]
    if any(
        _segments_intersect(a, b, c, d)
        for a, b in first_edges
        for c, d in second_edges
    ):
        return 0.0
    distances = []
    for vertex in first_vertices:
        distances.extend(
            _point_segment_distance(vertex, start, end)
            for start, end in second_edges
        )
    for vertex in second_vertices:
        distances.extend(
            _point_segment_distance(vertex, start, end)
            for start, end in first_edges
        )
    return float(min(distances))


def _first_sustained_stop(
    ego_trace: pd.DataFrame,
    *,
    not_before_frame: int,
    speed_threshold_mps: float,
    dwell_s: float,
    cadence_s: float,
) -> Optional[pd.Series]:
    required = max(1, int(math.ceil(float(dwell_s) / float(cadence_s) - 1e-12)))
    eligible = ego_trace[ego_trace["frame_id"].astype(int) >= int(not_before_frame)].copy()
    eligible = eligible.sort_values("frame_id")
    run: list[pd.Series] = []
    previous_frame: Optional[int] = None
    for _, row in eligible.iterrows():
        frame_id = int(row["frame_id"])
        if previous_frame is not None and frame_id != previous_frame + 1:
            run = []
        previous_frame = frame_id
        if float(row["recipient_speed_mps"]) <= float(speed_threshold_mps) + 1e-12:
            run.append(row)
            if len(run) >= required:
                return run[0]
        else:
            run = []
    return None


def _resolve_ego_dimensions(
    role_dir: Path, truth: pd.DataFrame
) -> tuple[Optional[tuple[float, float]], str]:
    manifest = _load_json(_single(role_dir / "manifests", "*_manifest.json"))
    type_id = str(manifest.get("anchor", {}).get("type_id", ""))
    candidates = truth[truth["type_id"].astype(str) == type_id]
    if candidates.empty:
        return None, "unavailable_ego_bbox_not_logged"
    lengths = pd.to_numeric(candidates["length_m"]).dropna()
    widths = pd.to_numeric(candidates["width_m"]).dropna()
    if lengths.empty or widths.empty:
        return None, "unavailable_ego_bbox_not_logged"
    return (
        (float(lengths.median()), float(widths.median())),
        "same_blueprint_truth_proxy_requires_direct_ego_bbox_in_full_corpus",
    )


def _future_label(
    event: Mapping[str, object],
    actor_truth: pd.DataFrame,
    ego_trace: pd.DataFrame,
    *,
    horizon_s: float,
    safety_radius_m: float,
    cadence_s: float,
    ego_dimensions: Optional[tuple[float, float]],
) -> dict:
    warning_at_s = float(event["warning_at_s"])
    required_end_s = warning_at_s + float(horizon_s)
    future = actor_truth[
        (pd.to_numeric(actor_truth["carla_timestamp"]) >= warning_at_s - 1e-9)
        & (pd.to_numeric(actor_truth["carla_timestamp"]) <= required_end_s + 1e-9)
    ].copy()
    merged = future.merge(ego_trace, on="frame_id", how="inner")
    if merged.empty:
        return {
            "future_label": "censored_no_aligned_future_truth",
            "future_truth_censored": 1,
            "future_horizon_observed_s": 0.0,
            "truth_hazard_positive": None,
            "false_warning": None,
            "minimum_future_center_distance_m": None,
            "minimum_future_surface_clearance_m": None,
            "minimum_distance_frame_id": None,
            "minimum_distance_after_warning_s": None,
        }
    merged["center_distance_m"] = np.hypot(
        pd.to_numeric(merged["recipient_x"]) - pd.to_numeric(merged["origin_x"]),
        pd.to_numeric(merged["recipient_y"]) - pd.to_numeric(merged["origin_y"]),
    )
    if ego_dimensions is not None:
        ego_length, ego_width = ego_dimensions
        merged["surface_clearance_m"] = [
            oriented_box_clearance_m(
                (
                    row.recipient_x,
                    row.recipient_y,
                    row.recipient_yaw_deg,
                    ego_length,
                    ego_width,
                ),
                (
                    row.origin_x,
                    row.origin_y,
                    row.yaw_deg,
                    row.length_m,
                    row.width_m,
                ),
            )
            for row in merged.itertuples(index=False)
        ]
    else:
        merged["surface_clearance_m"] = np.nan
    minimum_index = merged["center_distance_m"].idxmin()
    minimum_row = merged.loc[minimum_index]
    observed_end_s = float(pd.to_numeric(merged["carla_timestamp"]).max())
    observed_horizon_s = max(0.0, observed_end_s - warning_at_s)
    hazard_positive = float(minimum_row["center_distance_m"]) <= float(
        safety_radius_m
    ) + 1e-12
    censored = observed_end_s < required_end_s - float(cadence_s) * 0.5
    if hazard_positive:
        label, false_warning = "truth_hazard_positive", 0
    elif censored:
        label, false_warning = "censored_before_full_horizon", None
    else:
        label, false_warning = "truth_hazard_negative", 1
    surface = pd.to_numeric(merged["surface_clearance_m"]).dropna()
    return {
        "future_label": label,
        "future_truth_censored": int(censored and not hazard_positive),
        "future_horizon_observed_s": observed_horizon_s,
        "truth_hazard_positive": int(hazard_positive),
        "false_warning": false_warning,
        "minimum_future_center_distance_m": float(minimum_row["center_distance_m"]),
        "minimum_future_surface_clearance_m": (
            float(surface.min()) if not surface.empty else None
        ),
        "minimum_distance_frame_id": int(minimum_row["frame_id"]),
        "minimum_distance_after_warning_s": float(
            minimum_row["carla_timestamp"] - warning_at_s
        ),
    }


def _aligned_counterfactual_ego(
    reference_ego: pd.DataFrame,
    donor_ego: pd.DataFrame,
    *,
    cadence_s: float,
) -> pd.DataFrame:
    """Put a matched no-intervention ego trace on the reference frame IDs.

    Paired trajectories share their initial state, seed, route, and 10 Hz
    timebase. Their CARLA frame IDs differ because the world is reloaded, so
    alignment is by elapsed time. This fails closed rather than interpolating
    a mismatched pair.
    """

    required = {
        "frame_id",
        "elapsed_s",
        "recipient_x",
        "recipient_y",
        "recipient_yaw_deg",
        "recipient_speed_mps",
    }
    for label, trace in (("reference", reference_ego), ("donor", donor_ego)):
        if missing := required - set(trace.columns):
            raise ValueError(
                f"{label} ego trace lacks counterfactual fields: {sorted(missing)}"
            )
    if cadence_s <= 0.0 or not math.isfinite(cadence_s):
        raise ValueError("cadence_s must be finite and positive")

    reference = reference_ego.sort_values("elapsed_s").reset_index(drop=True)
    donor = donor_ego.sort_values("elapsed_s").reset_index(drop=True)
    if len(reference) != len(donor):
        raise ValueError(
            "matched counterfactual ego traces have different row counts: "
            f"reference={len(reference)}, donor={len(donor)}"
        )
    time_error = np.abs(
        pd.to_numeric(reference["elapsed_s"]).to_numpy(dtype=float)
        - pd.to_numeric(donor["elapsed_s"]).to_numpy(dtype=float)
    )
    if len(time_error) and float(time_error.max()) > float(cadence_s) * 0.5 + 1e-9:
        raise ValueError(
            "matched counterfactual ego traces exceed half-cadence alignment: "
            f"max_error_s={float(time_error.max()):.9f}"
        )

    aligned = donor.copy()
    aligned["counterfactual_source_frame_id"] = donor["frame_id"].astype(int)
    aligned["frame_id"] = reference["frame_id"].astype(int)
    aligned["elapsed_s"] = pd.to_numeric(reference["elapsed_s"]).to_numpy(
        dtype=float
    )
    return aligned


def _runtime_hashes(batch_root: Path, integration_config: Mapping[str, object]) -> dict[str, str]:
    hashes = {}
    for trajectory in integration_config["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in ROLE_NAMES:
            runtime_dir = batch_root / trajectory_id / role / "runtime"
            for path in sorted(runtime_dir.glob("*")):
                if path.is_file():
                    hashes[str(path.relative_to(batch_root))] = _sha256(path)
    return hashes


def _episode_count(timestamps: Sequence[float], gap_s: float) -> int:
    ordered = sorted(set(float(value) for value in timestamps))
    if not ordered:
        return 0
    return 1 + sum(
        current - previous > float(gap_s) + 1e-12
        for previous, current in zip(ordered, ordered[1:])
    )


def _trajectory_safety_outcome(
    batch_root: Path,
    trajectory: Mapping[str, object],
    batch_entry: Mapping[str, object],
    config: Mapping[str, object],
) -> dict:
    trajectory_id = str(trajectory["trajectory_id"])
    recipient_dir = batch_root / trajectory_id / "recipient"
    truth = pd.read_csv(_single(recipient_dir / "evaluation_truth", "*_ground_truth.csv"))
    ego = pd.read_csv(batch_root / trajectory_id / "ego_motion_trace.csv")
    ego_dimensions, ego_dimension_source = _resolve_ego_dimensions(recipient_dir, truth)
    target_prefix = str(trajectory.get("target_truth_role_prefix", ""))
    target = truth[
        truth["role_name"].astype(str).str.startswith(target_prefix)
    ] if target_prefix else truth.iloc[0:0]
    base = {
        "trajectory_id": trajectory_id,
        "scenario_role": trajectory["scenario_role"],
        "target_present": int(not target.empty),
        "target_truth_role_prefix": target_prefix,
        "collision_count": int(
            batch_entry.get("integrity", {}).get("unintended_collision_count", 0)
        ),
        "warnings_actuated": int(
            bool(config["attribution"]["warnings_actuated"])
        ),
        "outcome_attribution": str(config["attribution"]["current_pilot"]),
        "ego_dimension_source": ego_dimension_source,
        "ego_length_m": ego_dimensions[0] if ego_dimensions else None,
        "ego_width_m": ego_dimensions[1] if ego_dimensions else None,
        "clearance_band_status": "continuous_report_only_thresholds_not_frozen",
    }
    if target.empty:
        return {
            **base,
            "target_actor_id": None,
            "target_motion_start_frame": None,
            "minimum_center_distance_m": None,
            "minimum_surface_clearance_m": None,
            "minimum_clearance_frame_id": None,
            "sustained_stop_detected": 0,
            "stop_frame_id": None,
            "stop_center_distance_m": None,
            "stop_surface_clearance_m": None,
        }

    actor_counts = target.groupby("actor_id").size().sort_values(ascending=False)
    actor_id = str(actor_counts.index[0])
    target = target[target["actor_id"].astype(str) == actor_id].sort_values("frame_id")
    first = target.iloc[0]
    moved = np.hypot(
        pd.to_numeric(target["origin_x"]) - float(first["origin_x"]),
        pd.to_numeric(target["origin_y"]) - float(first["origin_y"]),
    )
    moving_rows = target[moved >= float(config["stopping"]["target_motion_threshold_m"])]
    motion_start_frame = int(
        moving_rows.iloc[0]["frame_id"] if not moving_rows.empty else first["frame_id"]
    )
    merged = target.merge(ego, on="frame_id", how="inner")
    merged["center_distance_m"] = np.hypot(
        pd.to_numeric(merged["recipient_x"]) - pd.to_numeric(merged["origin_x"]),
        pd.to_numeric(merged["recipient_y"]) - pd.to_numeric(merged["origin_y"]),
    )
    if ego_dimensions:
        ego_length, ego_width = ego_dimensions
        merged["surface_clearance_m"] = [
            oriented_box_clearance_m(
                (
                    row.recipient_x,
                    row.recipient_y,
                    row.recipient_yaw_deg,
                    ego_length,
                    ego_width,
                ),
                (
                    row.origin_x,
                    row.origin_y,
                    row.yaw_deg,
                    row.length_m,
                    row.width_m,
                ),
            )
            for row in merged.itertuples(index=False)
        ]
    else:
        merged["surface_clearance_m"] = np.nan
    minimum_row = merged.loc[merged["center_distance_m"].idxmin()]
    stop = _first_sustained_stop(
        ego,
        not_before_frame=motion_start_frame,
        speed_threshold_mps=float(config["stopping"]["speed_threshold_mps"]),
        dwell_s=float(config["stopping"]["minimum_dwell_s"]),
        cadence_s=float(config["timebase"]["cadence_s"]),
    )
    stop_merged = None
    if stop is not None:
        candidates = merged[merged["frame_id"].astype(int) == int(stop["frame_id"])]
        if not candidates.empty:
            stop_merged = candidates.iloc[0]
    surface = pd.to_numeric(merged["surface_clearance_m"]).dropna()
    return {
        **base,
        "target_actor_id": actor_id,
        "target_motion_start_frame": motion_start_frame,
        "minimum_center_distance_m": float(minimum_row["center_distance_m"]),
        "minimum_surface_clearance_m": (
            float(surface.min()) if not surface.empty else None
        ),
        "minimum_clearance_frame_id": int(minimum_row["frame_id"]),
        "sustained_stop_detected": int(stop is not None),
        "stop_frame_id": int(stop["frame_id"]) if stop is not None else None,
        "stop_center_distance_m": (
            float(stop_merged["center_distance_m"])
            if stop_merged is not None
            else None
        ),
        "stop_surface_clearance_m": (
            float(stop_merged["surface_clearance_m"])
            if stop_merged is not None
            and pd.notna(stop_merged["surface_clearance_m"])
            else None
        ),
    }


def adjudicate(
    batch_root: Path,
    integration_config: Mapping[str, object],
    adjudication_config: Mapping[str, object],
    *,
    evaluation_name: str,
    output_name: str,
    integration_config_path: Path,
    adjudication_config_path: Path,
) -> dict:
    output_name = _safe_output_name(output_name)
    output_dir = batch_root / output_name
    if output_dir.exists():
        raise FileExistsError(f"adjudication output already exists: {output_dir}")
    if adjudication_config.get("authorization") != "evaluation_only_no_runtime_feedback":
        raise ValueError("adjudication config must be evaluation-only")
    evaluation_dir = batch_root / str(evaluation_name)
    replay_summary = _load_json(evaluation_dir / "replay_summary.json")
    if replay_summary.get("status") != "complete":
        raise ValueError("input replay is not complete")
    replay_provenance = _load_json(evaluation_dir / "analysis_provenance.json")
    if replay_provenance.get("analysis_config", {}).get("semantic_sha256") != _semantic_sha256(
        integration_config
    ):
        raise ValueError("integration config does not match input replay provenance")

    runtime_before = _runtime_hashes(batch_root, integration_config)
    warning_events = pd.read_csv(evaluation_dir / "warning_events.csv")
    required_warning_columns = {
        "trajectory_id",
        "scenario_role",
        "arm_id",
        "frame_id",
        "warning_at_s",
        "canonical_track_id",
        "class_name",
        "track_world_x",
        "track_world_y",
    }
    if missing := required_warning_columns - set(warning_events.columns):
        raise ValueError(f"input replay lacks adjudication fields: {sorted(missing)}")

    trajectory_context: dict[str, dict] = {}
    safety_radii = {
        _normalize_class(key): float(value)
        for key, value in adjudication_config["future_hazard"][
            "safety_radius_m_by_class"
        ].items()
    }
    trajectories = list(integration_config["trajectories"])
    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        recipient_dir = batch_root / trajectory_id / "recipient"
        truth = pd.read_csv(
            _single(recipient_dir / "evaluation_truth", "*_ground_truth.csv")
        )
        ego = pd.read_csv(batch_root / trajectory_id / "ego_motion_trace.csv")
        ego_dimensions, ego_dimension_source = _resolve_ego_dimensions(
            recipient_dir, truth
        )
        frame_times = truth[["frame_id", "carla_timestamp"]].drop_duplicates(
            "frame_id"
        )
        trajectory_context[trajectory_id] = {
            "trajectory": trajectory,
            "truth": truth,
            "ego": ego,
            "ego_dimensions": ego_dimensions,
            "ego_dimension_source": ego_dimension_source,
            "last_truth_s": float(pd.to_numeric(frame_times["carla_timestamp"]).max()),
            "frame_times": frame_times,
            "target_prefix": str(trajectory.get("target_truth_role_prefix", "")),
        }

    pair_members: dict[str, list[Mapping[str, object]]] = {}
    for trajectory in trajectories:
        pair_id = str(trajectory.get("matched_pair_id", "")).strip()
        if not pair_id:
            raise ValueError(
                f"trajectory {trajectory['trajectory_id']} lacks matched_pair_id"
            )
        pair_members.setdefault(pair_id, []).append(trajectory)

    cadence_s = float(adjudication_config["timebase"]["cadence_s"])
    positive_basis = str(
        adjudication_config["future_hazard"]["positive_ego_trajectory_basis"]
    )
    benign_basis = str(
        adjudication_config["future_hazard"]["benign_ego_trajectory_basis"]
    )
    if positive_basis != "matched_benign_no_target_recipient_counterfactual":
        raise ValueError(f"unsupported positive trajectory basis: {positive_basis}")
    if benign_basis != "realized_nonactuated_recipient_trajectory":
        raise ValueError(f"unsupported benign trajectory basis: {benign_basis}")

    for pair_id, members in pair_members.items():
        benign = [
            item
            for item in members
            if str(item["scenario_role"]) == "matched_benign_negative"
        ]
        positive = [
            item
            for item in members
            if str(item["scenario_role"]) == "controlled_positive_occlusion"
        ]
        if len(benign) != 1 or len(positive) != 1:
            raise ValueError(
                f"matched pair {pair_id!r} must contain exactly one positive and "
                f"one benign trajectory, found positive={len(positive)}, "
                f"benign={len(benign)}"
            )
        benign_id = str(benign[0]["trajectory_id"])
        positive_id = str(positive[0]["trajectory_id"])
        trajectory_context[positive_id]["hazard_ego"] = _aligned_counterfactual_ego(
            trajectory_context[positive_id]["ego"],
            trajectory_context[benign_id]["ego"],
            cadence_s=cadence_s,
        )
        trajectory_context[positive_id]["hazard_ego_basis"] = positive_basis
        trajectory_context[positive_id]["counterfactual_source_trajectory_id"] = (
            benign_id
        )
        trajectory_context[benign_id]["hazard_ego"] = trajectory_context[benign_id][
            "ego"
        ]
        trajectory_context[benign_id]["hazard_ego_basis"] = benign_basis
        trajectory_context[benign_id]["counterfactual_source_trajectory_id"] = None

    rows: list[dict] = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        context = trajectory_context[trajectory_id]
        truth = context["truth"]
        ego_dimensions = context["ego_dimensions"]
        trajectory_warnings = warning_events[
            warning_events["trajectory_id"].astype(str) == trajectory_id
        ]
        for (_, _), frame_warnings in trajectory_warnings.groupby(
            ["arm_id", "frame_id"], sort=True
        ):
            frame_id = int(frame_warnings.iloc[0]["frame_id"])
            truth_frame = truth[truth["frame_id"].astype(int) == frame_id]
            matches = match_warnings_one_to_one(
                frame_warnings,
                truth_frame,
                gate_m=float(adjudication_config["matching"]["center_gate_m"]),
            )
            for warning_index, event in frame_warnings.iterrows():
                base = dict(event)
                match = matches[int(warning_index)]
                base.update(match)
                base["hazard_ego_trajectory_basis"] = context[
                    "hazard_ego_basis"
                ]
                base["counterfactual_source_trajectory_id"] = context[
                    "counterfactual_source_trajectory_id"
                ]
                base["target_hazard_match_adjudicated"] = int(
                    bool(match["current_truth_matched"])
                    and bool(context["target_prefix"])
                    and str(match["current_truth_role_name"]).startswith(
                        context["target_prefix"]
                    )
                )
                if not match["current_truth_matched"]:
                    base.update(
                        {
                            "future_label": "unmatched_false_warning",
                            "future_truth_censored": 0,
                            "future_horizon_observed_s": 0.0,
                            "truth_hazard_positive": 0,
                            "false_warning_adjudicated": 1,
                            "minimum_future_center_distance_m": None,
                            "minimum_future_surface_clearance_m": None,
                            "minimum_distance_frame_id": None,
                            "minimum_distance_after_warning_s": None,
                        }
                    )
                else:
                    actor_truth = truth[
                        truth["actor_id"].astype(str)
                        == str(match["current_truth_actor_id"])
                    ]
                    result = _future_label(
                        base,
                        actor_truth,
                        context["hazard_ego"],
                        horizon_s=float(
                            adjudication_config["future_hazard"]["horizon_s"]
                        ),
                        safety_radius_m=float(
                            safety_radii.get(_normalize_class(base["class_name"]), 3.0)
                        ),
                        cadence_s=cadence_s,
                        ego_dimensions=ego_dimensions,
                    )
                    result["false_warning_adjudicated"] = result.pop(
                        "false_warning"
                    )
                    base.update(result)
                rows.append(base)

    adjudicated = pd.DataFrame(rows)
    arm_rows: list[dict] = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        context = trajectory_context[trajectory_id]
        eligible_end = context["last_truth_s"] - float(
            adjudication_config["future_hazard"]["horizon_s"]
        )
        eligible_frames = context["frame_times"][
            pd.to_numeric(context["frame_times"]["carla_timestamp"])
            <= eligible_end + 1e-9
        ]
        eligible_frame_ids = set(eligible_frames["frame_id"].astype(int))
        duration_minutes = (
            len(eligible_frame_ids)
            * float(adjudication_config["timebase"]["cadence_s"])
            / 60.0
        )
        for arm_id in ("ego_only", "send_everything", "hazard_only"):
            group = adjudicated[
                (adjudicated["trajectory_id"].astype(str) == trajectory_id)
                & (adjudicated["arm_id"].astype(str) == arm_id)
            ]
            eligible_group = group[group["frame_id"].astype(int).isin(eligible_frame_ids)]
            false_group = eligible_group[
                pd.to_numeric(
                    eligible_group["false_warning_adjudicated"], errors="coerce"
                )
                == 1
            ]
            false_frames = int(false_group["frame_id"].nunique())
            target_group = group[
                pd.to_numeric(group["target_hazard_match_adjudicated"]).astype(int)
                == 1
            ]
            first_target = (
                float(pd.to_numeric(target_group["warning_at_s"]).min())
                if not target_group.empty
                else None
            )
            arm_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "scenario_role": trajectory["scenario_role"],
                    "arm_id": arm_id,
                    "hazard_ego_trajectory_basis": context["hazard_ego_basis"],
                    "counterfactual_source_trajectory_id": context[
                        "counterfactual_source_trajectory_id"
                    ],
                    "warning_event_count": len(group),
                    "one_to_one_matched_event_count": int(
                        pd.to_numeric(group["current_truth_matched"]).sum()
                    ),
                    "unmatched_false_warning_event_count": int(
                        (group["future_label"] == "unmatched_false_warning").sum()
                    ),
                    "truth_hazard_positive_event_count": int(
                        (pd.to_numeric(group["truth_hazard_positive"], errors="coerce") == 1).sum()
                    ),
                    "adjudicated_false_warning_event_count": int(
                        (pd.to_numeric(group["false_warning_adjudicated"], errors="coerce") == 1).sum()
                    ),
                    "censored_warning_event_count": int(
                        (pd.to_numeric(group["future_truth_censored"], errors="coerce") == 1).sum()
                    ),
                    "eligible_full_horizon_frame_count": len(eligible_frame_ids),
                    "false_warning_active_frame_count": false_frames,
                    "false_warning_active_frame_rate": (
                        false_frames / len(eligible_frame_ids)
                        if eligible_frame_ids
                        else None
                    ),
                    "false_warning_episode_count": _episode_count(
                        pd.to_numeric(false_group["warning_at_s"]).tolist(),
                        float(adjudication_config["episodes"]["quiet_gap_s"]),
                    ),
                    "false_warning_episodes_per_minute": (
                        _episode_count(
                            pd.to_numeric(false_group["warning_at_s"]).tolist(),
                            float(adjudication_config["episodes"]["quiet_gap_s"]),
                        )
                        / duration_minutes
                        if duration_minutes > 0.0
                        else None
                    ),
                    "first_registered_target_warning_s": first_target,
                    "missed_registered_target": int(
                        bool(context["target_prefix"]) and first_target is None
                    ),
                    "confirmatory_performance_evidence": False,
                }
            )
    arm_summary = pd.DataFrame(arm_rows)
    for trajectory_id, group in arm_summary.groupby("trajectory_id"):
        ego_values = group[group["arm_id"] == "ego_only"]
        ego_first = (
            float(ego_values.iloc[0]["first_registered_target_warning_s"])
            if not ego_values.empty
            and pd.notna(ego_values.iloc[0]["first_registered_target_warning_s"])
            else None
        )
        for index in group.index:
            first = arm_summary.at[index, "first_registered_target_warning_s"]
            arm_summary.at[index, "warning_lead_vs_ego_s"] = (
                ego_first - float(first)
                if ego_first is not None and pd.notna(first)
                else None
            )

    batch_manifest = _load_json(batch_root / "batch_manifest.json")
    batch_entries = {
        str(item["trajectory_id"]): item
        for item in batch_manifest.get("trajectories", [])
    }
    safety_outcomes = [
        _trajectory_safety_outcome(
            batch_root,
            trajectory,
            batch_entries[str(trajectory["trajectory_id"])],
            adjudication_config,
        )
        for trajectory in integration_config["trajectories"]
    ]
    runtime_after = _runtime_hashes(batch_root, integration_config)
    runtime_unchanged = runtime_before == runtime_after

    duplicate_matches = (
        adjudicated[pd.to_numeric(adjudicated["current_truth_matched"]).astype(int) == 1]
        .groupby(["trajectory_id", "arm_id", "frame_id", "class_name", "current_truth_actor_id"])
        .size()
    )
    verification_failures = []
    if len(adjudicated) != len(warning_events):
        verification_failures.append("adjudicated row count differs from replay warnings")
    if not duplicate_matches.empty and int(duplicate_matches.max()) > 1:
        verification_failures.append("one-to-one truth identity reused within an arm/frame/class")
    matched_distances = pd.to_numeric(
        adjudicated.loc[
            pd.to_numeric(adjudicated["current_truth_matched"]).astype(int) == 1,
            "current_truth_distance_m",
        ]
    )
    if (
        not matched_distances.empty
        and matched_distances.max()
        > float(adjudication_config["matching"]["center_gate_m"]) + 1e-9
    ):
        verification_failures.append("one-to-one match exceeds center gate")
    if not runtime_unchanged:
        verification_failures.append("runtime artifacts changed during evaluation")
    if any(int(row["warnings_actuated"]) != 0 for row in safety_outcomes):
        verification_failures.append("pilot unexpectedly claims actuated warning outcomes")
    positive_basis_ok = True
    positive_target_hazard_ok = True
    for trajectory in trajectories:
        if str(trajectory["scenario_role"]) != "controlled_positive_occlusion":
            continue
        trajectory_id = str(trajectory["trajectory_id"])
        context = trajectory_context[trajectory_id]
        positive_basis_ok &= (
            context["hazard_ego_basis"]
            == "matched_benign_no_target_recipient_counterfactual"
            and bool(context["counterfactual_source_trajectory_id"])
        )
        target_rows = adjudicated[
            (adjudicated["trajectory_id"].astype(str) == trajectory_id)
            & (pd.to_numeric(adjudicated["target_hazard_match_adjudicated"]) == 1)
        ]
        if target_rows.empty or not (
            pd.to_numeric(target_rows["truth_hazard_positive"], errors="coerce") == 1
        ).any():
            positive_target_hazard_ok = False
    if not positive_basis_ok:
        verification_failures.append(
            "positive trajectory did not use the declared matched benign counterfactual"
        )
    if not positive_target_hazard_ok:
        verification_failures.append(
            "registered positive target never becomes future-hazard-positive under "
            "the matched no-yield counterfactual"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    adjudicated.to_csv(output_dir / "adjudicated_warning_events.csv", index=False)
    arm_summary.to_csv(output_dir / "arm_adjudication_summary.csv", index=False)
    pd.DataFrame(safety_outcomes).to_csv(
        output_dir / "trajectory_safety_outcomes.csv", index=False
    )
    repository_root = Path(__file__).resolve().parents[1]
    source_files = {
        "warning_events.csv": _sha256(evaluation_dir / "warning_events.csv"),
        "replay_summary.json": _sha256(evaluation_dir / "replay_summary.json"),
        "replay_provenance.json": _sha256(
            evaluation_dir / "analysis_provenance.json"
        ),
        "batch_manifest.json": _sha256(batch_root / "batch_manifest.json"),
        "integration_config": _sha256(integration_config_path),
        "adjudication_config": _sha256(adjudication_config_path),
    }
    provenance = {
        "schema": "scenesense.phase2_future_hazard_provenance.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_root": str(batch_root),
        "input_evaluation_namespace": str(evaluation_name),
        "output_namespace": output_name,
        "source_files": source_files,
        "runtime_artifact_count": len(runtime_before),
        "runtime_hashes_unchanged": runtime_unchanged,
        "analysis_code": {
            "repository_commit": _git_value(repository_root, "rev-parse", "HEAD"),
            "module": str(Path(__file__).resolve().relative_to(repository_root)),
            "module_sha256": _sha256(Path(__file__).resolve()),
        },
        "integration_config_semantic_sha256": _semantic_sha256(integration_config),
        "adjudication_config_semantic_sha256": _semantic_sha256(adjudication_config),
        "truth_usage": "evaluation_only_no_runtime_feedback",
        "positive_hazard_ego_trajectory_basis": positive_basis,
        "benign_hazard_ego_trajectory_basis": benign_basis,
    }
    (output_dir / "adjudication_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verification = {
        "schema": "scenesense.phase2_future_hazard_verification.v2",
        "verdict": "PASS" if not verification_failures else "FAIL_HOLD",
        "failures": verification_failures,
        "gates": {
            "input_warning_rows_preserved": len(adjudicated) == len(warning_events),
            "one_to_one_matching": duplicate_matches.empty
            or int(duplicate_matches.max()) <= 1,
            "center_gate_respected": matched_distances.empty
            or float(matched_distances.max())
            <= float(adjudication_config["matching"]["center_gate_m"]) + 1e-9,
            "runtime_artifacts_unchanged": runtime_unchanged,
            "current_pilot_outcomes_non_actuated": all(
                int(row["warnings_actuated"]) == 0 for row in safety_outcomes
            ),
            "positive_counterfactual_basis_valid": bool(positive_basis_ok),
            "registered_positive_target_future_hazard_observed": bool(
                positive_target_hazard_ok
            ),
        },
    }
    (output_dir / "adjudication_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete" if not verification_failures else "failed_hold",
        "verdict": verification["verdict"],
        "confirmatory_performance_evidence": False,
        "warning_event_rows": len(adjudicated),
        "arm_summary_rows": len(arm_summary),
        "trajectory_safety_rows": len(safety_outcomes),
        "runtime_truth_leakage_detected": not runtime_unchanged,
        "stopping_reward_status": (
            "continuous_outcome_computable_but_not_policy_attributable_until_warning_actuation"
        ),
        "positive_hazard_ego_trajectory_basis": positive_basis,
        "supersedes": "hazard_adjudication_v1_intervention_contaminated",
    }
    (output_dir / "adjudication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "schema": "scenesense.phase2_future_hazard_artifact_manifest.v2",
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path != manifest_path
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if verification_failures:
        raise RuntimeError(f"future-hazard verification failed: {verification_failures}")
    return summary


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument(
        "--integration-config",
        type=Path,
        default=repository_root
        / "data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml",
    )
    parser.add_argument(
        "--adjudication-config",
        type=Path,
        default=repository_root
        / "phase2_map_sharing/configs/future_hazard_adjudication_v2.yaml",
    )
    parser.add_argument("--evaluation-name", default="evaluation_v4")
    parser.add_argument("--output-name", default="hazard_adjudication_v2")
    args = parser.parse_args()
    integration_config_path = args.integration_config.resolve()
    adjudication_config_path = args.adjudication_config.resolve()
    integration_config = yaml.safe_load(
        integration_config_path.read_text(encoding="utf-8")
    )
    adjudication_config = yaml.safe_load(
        adjudication_config_path.read_text(encoding="utf-8")
    )
    if not isinstance(integration_config, Mapping) or not isinstance(
        adjudication_config, Mapping
    ):
        raise ValueError("config roots must be mappings")
    result = adjudicate(
        args.batch_root.resolve(),
        integration_config,
        adjudication_config,
        evaluation_name=str(args.evaluation_name),
        output_name=str(args.output_name),
        integration_config_path=integration_config_path,
        adjudication_config_path=adjudication_config_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
