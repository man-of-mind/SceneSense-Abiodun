"""Offline three-arm replay for a completed paired-causal pilot.

The replay consumes only source-local runtime tracks and recipient ego state.
CARLA identity is opened later, in the evaluation-only truth join.  Pilot
parameters are explicitly provisional: this command establishes C2
computability and must not be presented as confirmatory performance evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd
import yaml

from phase2_map_sharing.engine_v2 import RecipientMapEngineV2
from phase2_map_sharing.schemas_v2 import (
    MapContributionV2,
    MapObjectObservationV2,
    RecipientStateV2,
    with_exact_payload_bytes_v2,
)


ARMS = ("ego_only", "send_everything", "hazard_only")
CLOCK_ID = "carla_simulation_elapsed_seconds"
ZERO_SHA256 = "0" * 64
STATE_COVARIANCE = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 4.0, 0.0,
    0.0, 0.0, 0.0, 4.0,
)
PROCESS_NOISE = (
    0.25, 0.0, 0.0, 0.0,
    0.0, 0.25, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_output_name(value: str, prefix: str) -> str:
    name = str(value).strip()
    if not OUTPUT_NAME_PATTERN.fullmatch(name) or not name.startswith(prefix):
        raise ValueError(
            f"output name must be one safe basename beginning with {prefix!r}: {value!r}"
        )
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


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _class_name(value: object) -> str:
    name = str(value).strip().lower()
    if name in {"person", "walker"}:
        return "pedestrian"
    return name


def _latest_ego_state(ego: pd.DataFrame, timestamp_s: float) -> pd.Series:
    eligible = ego[pd.to_numeric(ego["carla_timestamp"]) <= float(timestamp_s) + 1e-9]
    if eligible.empty:
        raise ValueError(f"no causal recipient state exists at {timestamp_s:.6f}s")
    return eligible.sort_values("carla_timestamp").iloc[-1]


def _recipient_state(row: pd.Series, available_at_s: float) -> RecipientStateV2:
    state = RecipientStateV2(
        recipient_ue_id="recipient",
        observed_at_s=float(row["carla_timestamp"]),
        available_at_s=float(available_at_s),
        clock_id=CLOCK_ID,
        x_m=float(row["world_x"]),
        y_m=float(row["world_y"]),
        vx_mps=float(row["velocity_x"]),
        vy_mps=float(row["velocity_y"]),
        state_covariance=STATE_COVARIANCE,
        motion_model_id="CV",
        process_noise_model_id="provisional_diagonal_v1",
        process_noise_covariance_per_s=PROCESS_NOISE,
    )
    state.validate()
    return state


def _hazard_score(track: pd.Series, recipient: RecipientStateV2, horizon_s: float = 5.0) -> float:
    observed_at_s = float(track["last_observed_timestamp_s"])
    dt = max(0.0, recipient.available_at_s - observed_at_s)
    object_x = float(track["world_x"]) + float(track["velocity_x"]) * dt
    object_y = float(track["world_y"]) + float(track["velocity_y"]) * dt
    recipient_dt = max(0.0, recipient.available_at_s - recipient.observed_at_s)
    recipient_x = recipient.x_m + recipient.vx_mps * recipient_dt
    recipient_y = recipient.y_m + recipient.vy_mps * recipient_dt
    rx = object_x - recipient_x
    ry = object_y - recipient_y
    rvx = float(track["velocity_x"]) - recipient.vx_mps
    rvy = float(track["velocity_y"]) - recipient.vy_mps
    speed_sq = rvx * rvx + rvy * rvy
    tca = 0.0 if speed_sq <= 1e-12 else max(
        0.0, min(float(horizon_s), -(rx * rvx + ry * rvy) / speed_sq)
    )
    closest = math.hypot(rx + rvx * tca, ry + rvy * tca)
    radius = 3.0 if _class_name(track["class_name"]) != "pedestrian" else 2.5
    return max(0.0, min(1.0, 1.0 - closest / radius)) if closest <= radius else 0.0


def _objects(
    tracks: pd.DataFrame,
    *,
    captured_at_s: float,
    publication_action: str,
    recipient: RecipientStateV2,
) -> tuple[MapObjectObservationV2, ...]:
    objects = []
    for _, row in tracks.iterrows():
        hazard_score = _hazard_score(row, recipient)
        if publication_action == "PUBLISH_HAZARD_SUBSET" and hazard_score <= 0.0:
            continue
        obj = MapObjectObservationV2(
            source_track_id=str(row["source_track_id"]),
            tracker_id=str(row["source_role"]),
            tracker_version=str(row["tracker_version"]),
            class_name=_class_name(row["class_name"]),
            x_m=float(row["world_x"]),
            y_m=float(row["world_y"]),
            vx_mps=float(row["velocity_x"]),
            vy_mps=float(row["velocity_y"]),
            confidence=float(row["score"]),
            measured_at_s=float(row["last_observed_timestamp_s"]),
            state_covariance=STATE_COVARIANCE,
            motion_model_id="CV",
            process_noise_model_id="provisional_diagonal_v1",
            process_noise_covariance_per_s=PROCESS_NOISE,
            validity_horizon_s=1.0,
            hazard_score=hazard_score,
            hazard_source=(
                "recipient_relative_cv_provisional_v1"
                if publication_action == "PUBLISH_HAZARD_SUBSET"
                else "none"
            ),
            recipient_state_observed_at_s=(
                recipient.observed_at_s
                if publication_action == "PUBLISH_HAZARD_SUBSET"
                else None
            ),
            recipient_state_available_at_s=(
                recipient.available_at_s
                if publication_action == "PUBLISH_HAZARD_SUBSET"
                else None
            ),
        )
        obj.validate()
        objects.append(obj)
    return tuple(objects)


def _contribution(
    *,
    trajectory_id: str,
    source_role: str,
    sequence: int,
    captured_at_s: float,
    tracks: pd.DataFrame,
    publication_action: str,
    recipient: RecipientStateV2,
    model_sha256: str,
    config_sha256: str,
) -> MapContributionV2:
    contribution = MapContributionV2(
        contribution_id=f"{trajectory_id}:{source_role}:{sequence}",
        source_ue_id=source_role,
        recipient_ue_id="recipient",
        sequence_number=int(sequence),
        captured_at_s=float(captured_at_s),
        placement_decision_id=f"{trajectory_id}:{source_role}:placement:{sequence}",
        placement_decision_at_s=float(captured_at_s),
        inference_completed_at_s=float(captured_at_s),
        publication_decision_id=f"{trajectory_id}:{source_role}:publication:{sequence}",
        publication_decision_at_s=float(captured_at_s),
        published_at_s=float(captured_at_s),
        clock_id=CLOCK_ID,
        publication_decision_locus=source_role,
        inference_placement="SPLIT_FEATURE",
        publication_action=publication_action,
        profile_id="mprime_200k_fast_nms2_top120",
        target_fps=10.0,
        model_id="M_prime",
        model_sha256=model_sha256,
        config_sha256=config_sha256,
        code_revision="workspace_pilot_replay_v1",
        source_sensor_ids=(f"{source_role}_rgb", f"{source_role}_radar"),
        calibration_ids=(f"{source_role}_camera_intrinsics", f"{source_role}_radar_extrinsics"),
        transport_chunk_bytes=60000,
        chunk_count=1,
        application_payload_bytes=0,
        objects=_objects(
            tracks,
            captured_at_s=captured_at_s,
            publication_action=publication_action,
            recipient=recipient,
        ),
    )
    return with_exact_payload_bytes_v2(contribution)


def _truth_match(
    warning_track: Mapping[str, object],
    truth: pd.DataFrame,
    target_role_prefix: str,
    gate_m: float = 5.0,
) -> tuple[bool, bool, Optional[str], Optional[float]]:
    candidates = []
    for _, row in truth.iterrows():
        if _class_name(row["class_name"]) != _class_name(warning_track["class_name"]):
            continue
        distance = math.hypot(
            float(row["origin_x"]) - float(warning_track["x_m"]),
            float(row["origin_y"]) - float(warning_track["y_m"]),
        )
        if distance <= gate_m:
            candidates.append((distance, str(row["role_name"]), str(row["actor_id"])))
    if not candidates:
        return False, False, None, None
    distance, role_name, actor_id = min(candidates)
    target = bool(target_role_prefix and role_name.startswith(target_role_prefix))
    return True, target, actor_id, float(distance)


def _select_target_chain_warning(
    warning_rows: Sequence[Mapping[str, object]],
    positive_trajectory_id: str,
) -> Optional[dict]:
    """Choose a deterministic registered-target exemplar, never an arbitrary warning."""

    arm_priority = {"hazard_only": 0, "send_everything": 1, "ego_only": 2}
    candidates = [
        dict(row)
        for row in warning_rows
        if str(row.get("trajectory_id")) == str(positive_trajectory_id)
        and int(row.get("target_hazard_match", 0)) == 1
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row["warning_at_s"]),
            arm_priority.get(str(row.get("arm_id")), 99),
            int(row["frame_id"]),
            str(row.get("canonical_track_id", "")),
        ),
    )


def _warning_diagnostics(
    metric_rows: Sequence[Mapping[str, object]],
    warning_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict], list[dict]]:
    """Return exposure-based warning burden and track-fragmentation diagnostics.

    `non_target` is deliberately named as a proxy.  A warning about a different
    truth object is not scientifically a false alarm until scenario truth says
    that object is non-hazardous.
    """

    warnings = pd.DataFrame(warning_rows)
    diagnostics: list[dict] = []
    for metric in metric_rows:
        trajectory_id = str(metric["trajectory_id"])
        arm_id = str(metric["arm_id"])
        frame_count = int(metric["frame_count"])
        if warnings.empty:
            group = warnings
        else:
            group = warnings[
                (warnings["trajectory_id"].astype(str) == trajectory_id)
                & (warnings["arm_id"].astype(str) == arm_id)
            ]

        def _subset(column: str, value: int) -> pd.DataFrame:
            if group.empty:
                return group
            return group[pd.to_numeric(group[column]).astype(int) == int(value)]

        target = _subset("target_hazard_match", 1)
        non_target = _subset("target_hazard_match", 0)
        unmatched = _subset("truth_matched", 0)
        matched_non_target = non_target[
            pd.to_numeric(non_target.get("truth_matched", pd.Series(dtype=int))).astype(int)
            == 1
        ] if not non_target.empty else non_target

        def _frame_count(rows: pd.DataFrame) -> int:
            return 0 if rows.empty else int(rows["frame_id"].nunique())

        def _track_count(rows: pd.DataFrame) -> int:
            return 0 if rows.empty else int(rows["canonical_track_id"].nunique())

        warning_frames = _frame_count(group)
        diagnostics.append(
            {
                "trajectory_id": trajectory_id,
                "scenario_role": metric["scenario_role"],
                "arm_id": arm_id,
                "frame_count": frame_count,
                "warning_event_count": len(group),
                "warning_frame_count": warning_frames,
                "warning_frame_rate": warning_frames / frame_count if frame_count else 0.0,
                "warning_events_per_active_frame": (
                    len(group) / warning_frames if warning_frames else 0.0
                ),
                "unique_warning_track_count": _track_count(group),
                "target_warning_event_count": len(target),
                "target_warning_frame_count": _frame_count(target),
                "target_warning_track_count": _track_count(target),
                "non_target_warning_event_count": len(non_target),
                "non_target_warning_frame_count": _frame_count(non_target),
                "non_target_warning_frame_rate": (
                    _frame_count(non_target) / frame_count if frame_count else 0.0
                ),
                "non_target_warning_track_count": _track_count(non_target),
                "matched_non_target_warning_event_count": len(matched_non_target),
                "unmatched_warning_event_count": len(unmatched),
                "unmatched_warning_frame_count": _frame_count(unmatched),
                "unmatched_warning_frame_rate": (
                    _frame_count(unmatched) / frame_count if frame_count else 0.0
                ),
                "unmatched_warning_track_count": _track_count(unmatched),
                "false_warning_adjudication_status": (
                    "provisional_non_target_proxy_not_hazard_adjudicated"
                ),
            }
        )

    fragmentation: list[dict] = []
    if not warnings.empty:
        matched = warnings[pd.to_numeric(warnings["truth_matched"]).astype(int) == 1]
        for keys, group in matched.groupby(
            ["trajectory_id", "scenario_role", "arm_id", "class_name", "evaluation_truth_id"],
            dropna=False,
        ):
            trajectory_id, scenario_role, arm_id, class_name, truth_id = keys
            fragmentation.append(
                {
                    "trajectory_id": trajectory_id,
                    "scenario_role": scenario_role,
                    "arm_id": arm_id,
                    "class_name": class_name,
                    "truth_scope": "matched_truth_object",
                    "evaluation_truth_id": truth_id,
                    "target_hazard_match": int(
                        pd.to_numeric(group["target_hazard_match"]).astype(int).max()
                    ),
                    "warning_event_count": len(group),
                    "warning_frame_count": int(group["frame_id"].nunique()),
                    "canonical_warning_track_count": int(
                        group["canonical_track_id"].nunique()
                    ),
                    "canonical_warning_track_ids": json.dumps(
                        sorted(group["canonical_track_id"].astype(str).unique())
                    ),
                }
            )
        unmatched = warnings[pd.to_numeric(warnings["truth_matched"]).astype(int) == 0]
        for keys, group in unmatched.groupby(
            ["trajectory_id", "scenario_role", "arm_id", "class_name"],
            dropna=False,
        ):
            trajectory_id, scenario_role, arm_id, class_name = keys
            fragmentation.append(
                {
                    "trajectory_id": trajectory_id,
                    "scenario_role": scenario_role,
                    "arm_id": arm_id,
                    "class_name": class_name,
                    "truth_scope": "unmatched",
                    "evaluation_truth_id": None,
                    "target_hazard_match": 0,
                    "warning_event_count": len(group),
                    "warning_frame_count": int(group["frame_id"].nunique()),
                    "canonical_warning_track_count": int(
                        group["canonical_track_id"].nunique()
                    ),
                    "canonical_warning_track_ids": json.dumps(
                        sorted(group["canonical_track_id"].astype(str).unique())
                    ),
                }
            )
    return diagnostics, fragmentation


def replay_trajectory(
    batch_root: Path,
    trajectory: Mapping[str, object],
) -> tuple[list[dict], list[dict], dict]:
    trajectory_id = str(trajectory["trajectory_id"])
    helper_dir = batch_root / trajectory_id / "helper"
    recipient_dir = batch_root / trajectory_id / "recipient"
    helper_tracks = pd.read_csv(helper_dir / "runtime/causal_tracks.csv")
    recipient_tracks = pd.read_csv(recipient_dir / "runtime/causal_tracks.csv")
    recipient_ego = pd.read_csv(recipient_dir / "runtime/ego_states.csv")
    truth = pd.read_csv(_single(recipient_dir / "evaluation_truth", "*_ground_truth.csv"))
    helper_metrics = pd.read_csv(_single(helper_dir / "streams", "*_metrics.csv"))
    recipient_metrics = pd.read_csv(_single(recipient_dir / "streams", "*_metrics.csv"))
    recipient_manifest_path = _single(recipient_dir / "manifests", "*_manifest.json")
    recipient_manifest = json.loads(recipient_manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(recipient_manifest["checkpoint_path"]))
    model_sha = _sha256(checkpoint) if checkpoint.is_file() else ZERO_SHA256
    config_path = _single(recipient_dir / "manifests", "*_resolved_config.json")
    config_sha = _sha256(config_path)

    common_frames = sorted(
        set(helper_metrics["frame_id"].astype(int))
        & set(recipient_metrics["frame_id"].astype(int))
    )
    engines = {
        arm: RecipientMapEngineV2(
            "recipient",
            warning_emission_confidence_floor=0.05,
            warning_sigma_multiplier=0.0,
        )
        for arm in ARMS
    }
    state_ids = {arm: id(engine.tracks) for arm, engine in engines.items()}
    if len(set(state_ids.values())) != len(ARMS):
        raise RuntimeError("counterfactual arms unexpectedly share map state")
    helper_metric_by_frame = helper_metrics.set_index("frame_id")
    capture_timestamp_by_frame = (
        recipient_metrics.set_index("frame_id")["carla_timestamp"].astype(float).to_dict()
    )
    warning_rows: list[dict] = []
    arm_accumulator = {
        arm: {
            "application_bytes": 0,
            "on_wire_bytes": 0,
            "capture_to_install_ms": [],
            "map_aoi_s": [],
            "target_warning_times": [],
            "false_warnings": 0,
            "warning_count": 0,
            "map_track_counts": [],
        }
        for arm in ARMS
    }

    for sequence, frame_id in enumerate(common_frames):
        frame_helper = helper_tracks[helper_tracks["frame_id"].astype(int) == frame_id]
        frame_recipient = recipient_tracks[
            recipient_tracks["frame_id"].astype(int) == frame_id
        ]
        timestamps = [
            *pd.to_numeric(frame_helper.get("carla_timestamp", pd.Series(dtype=float))).tolist(),
            *pd.to_numeric(frame_recipient.get("carla_timestamp", pd.Series(dtype=float))).tolist(),
        ]
        captured_at_s = float(
            min(timestamps) if timestamps else capture_timestamp_by_frame[int(frame_id)]
        )
        ego_row = _latest_ego_state(recipient_ego, captured_at_s)
        recipient_state = _recipient_state(ego_row, captured_at_s)
        truth_frame = truth[truth["frame_id"].astype(int) == int(frame_id)]
        recipient_contribution = _contribution(
            trajectory_id=trajectory_id,
            source_role="recipient",
            sequence=sequence,
            captured_at_s=captured_at_s,
            tracks=frame_recipient,
            publication_action="PUBLISH_ALL",
            recipient=recipient_state,
            model_sha256=model_sha,
            config_sha256=config_sha,
        )
        for arm, engine in engines.items():
            result = engine.install(
                recipient_contribution, captured_at_s, CLOCK_ID
            )
            if result != "accepted":
                raise RuntimeError(f"{arm} rejected recipient contribution: {result}")
            if arm != "ego_only":
                action = (
                    "PUBLISH_ALL" if arm == "send_everything" else "PUBLISH_HAZARD_SUBSET"
                )
                helper_contribution = _contribution(
                    trajectory_id=trajectory_id,
                    source_role="helper",
                    sequence=sequence,
                    captured_at_s=captured_at_s,
                    tracks=frame_helper,
                    publication_action=action,
                    recipient=recipient_state,
                    model_sha256=model_sha,
                    config_sha256=config_sha,
                )
                result = engine.install(helper_contribution, captured_at_s, CLOCK_ID)
                if result != "accepted":
                    raise RuntimeError(f"{arm} rejected helper contribution: {result}")
                accumulator = arm_accumulator[arm]
                accumulator["application_bytes"] += helper_contribution.application_payload_bytes
                accumulator["on_wire_bytes"] += (
                    helper_contribution.application_payload_bytes
                    + helper_contribution.chunk_count * 36
                )
                if frame_id in helper_metric_by_frame.index:
                    latency = helper_metric_by_frame.loc[frame_id, "total_pipeline_ms_estimate"]
                    if isinstance(latency, pd.Series):
                        latency = latency.iloc[0]
                    if pd.notna(latency):
                        accumulator["capture_to_install_ms"].append(float(latency))

            warnings = engine.warnings(recipient_state)
            snapshot = engine.snapshot(captured_at_s, CLOCK_ID)
            snapshot_by_id = {
                str(track["canonical_track_id"]): track for track in snapshot["tracks"]
            }
            arm_accumulator[arm]["map_track_counts"].append(len(snapshot["tracks"]))
            for warning in warnings:
                arm_accumulator[arm]["warning_count"] += 1
                arm_accumulator[arm]["map_aoi_s"].append(float(warning.map_aoi_s))
                track = snapshot_by_id[warning.canonical_track_id]
                matched, target, truth_id, distance = _truth_match(
                    track,
                    truth_frame,
                    str(trajectory.get("target_truth_role_prefix", "")),
                )
                if target:
                    arm_accumulator[arm]["target_warning_times"].append(captured_at_s)
                if not target:
                    arm_accumulator[arm]["false_warnings"] += 1
                warning_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "scenario_role": trajectory["scenario_role"],
                        "arm_id": arm,
                        "frame_id": int(frame_id),
                        "warning_at_s": captured_at_s,
                        "canonical_track_id": warning.canonical_track_id,
                        "class_name": warning.class_name,
                        "track_world_x": track["x_m"],
                        "track_world_y": track["y_m"],
                        "track_velocity_x": track["vx_mps"],
                        "track_velocity_y": track["vy_mps"],
                        "track_position_sigma_m": track["position_sigma_m"],
                        "time_to_closest_approach_s": warning.time_to_closest_approach_s,
                        "closest_approach_m": warning.closest_approach_m,
                        "uncertainty_expanded_closest_approach_m": (
                            warning.uncertainty_expanded_closest_approach_m
                        ),
                        "position_sigma_at_closest_approach_m": (
                            warning.position_sigma_at_closest_approach_m
                        ),
                        "map_aoi_s": warning.map_aoi_s,
                        "evidence_sources": json.dumps(list(warning.evidence_sources)),
                        "evidence_track_ids": json.dumps(list(warning.evidence_track_ids)),
                        "evidence_scope": warning.evidence_scope,
                        "truth_matched": int(matched),
                        "target_hazard_match": int(target),
                        "evaluation_truth_id": truth_id,
                        "truth_distance_m": distance,
                    }
                )

    target_exists = bool(
        str(trajectory.get("target_truth_role_prefix", ""))
        and truth["role_name"].astype(str).str.startswith(
            str(trajectory["target_truth_role_prefix"])
        ).any()
    )
    first_by_arm = {
        arm: (
            min(values["target_warning_times"])
            if values["target_warning_times"]
            else None
        )
        for arm, values in arm_accumulator.items()
    }
    ego_first = first_by_arm["ego_only"]
    metric_rows = []
    for arm, values in arm_accumulator.items():
        first = first_by_arm[arm]
        lead = None if first is None or ego_first is None else ego_first - first
        metric_rows.append(
            {
                "trajectory_id": trajectory_id,
                "scenario_role": trajectory["scenario_role"],
                "arm_id": arm,
                "first_warning_s": first,
                "warning_lead_s": lead,
                "false_warning": int(values["false_warnings"] > 0),
                "false_warning_definition": (
                    "legacy_any_non_target_warning_proxy_not_hazard_adjudicated"
                ),
                "missed_hazard": int(target_exists and first is None),
                "frame_count": len(common_frames),
                "warning_count": values["warning_count"],
                "application_bytes": values["application_bytes"],
                "on_wire_bytes": values["on_wire_bytes"],
                "capture_to_install_ms": (
                    sum(values["capture_to_install_ms"]) / len(values["capture_to_install_ms"])
                    if values["capture_to_install_ms"]
                    else 0.0
                ),
                "capture_to_install_timing_status": (
                    "non_citable_shared_gpu_correctness_pilot"
                ),
                "map_aoi_s": (
                    sum(values["map_aoi_s"]) / len(values["map_aoi_s"])
                    if values["map_aoi_s"]
                    else 0.0
                ),
                "mean_map_track_count": (
                    sum(values["map_track_counts"]) / len(values["map_track_counts"])
                    if values["map_track_counts"]
                    else 0.0
                ),
                "evidence_provenance": (
                    "recipient_local_tracks"
                    if arm == "ego_only"
                    else "recipient_plus_helper_all"
                    if arm == "send_everything"
                    else "recipient_plus_helper_causal_hazard_subset"
                ),
            }
        )
    isolation = {
        "trajectory_id": trajectory_id,
        "state_object_ids": state_ids,
        "independent": len(set(state_ids.values())) == len(ARMS),
        "map_engine_diagnostics": {
            arm: {
                "created_canonical_tracks": engine.next_track_number - 1,
                "final_active_tracks": len(engine.tracks),
                "counters": dict(engine.counters),
            }
            for arm, engine in engines.items()
        },
    }
    return metric_rows, warning_rows, isolation


def _analysis_provenance(
    batch_root: Path,
    config: Mapping[str, object],
    config_path: Optional[Path],
    evaluation_name: str,
) -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    resolved_config_path = batch_root / "resolved_integration_config.yaml"
    launch_manifest_path = batch_root.parent / f"{batch_root.name}.launch.json"
    resolved_config = (
        yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        if resolved_config_path.is_file()
        else None
    )
    launch_manifest = (
        json.loads(launch_manifest_path.read_text(encoding="utf-8"))
        if launch_manifest_path.is_file()
        else None
    )
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("verify_paired_pilot.py"),
        Path(__file__).resolve().with_name("engine_v2.py"),
        Path(__file__).resolve().with_name("schemas_v2.py"),
    ]
    source_manifests = {}
    for trajectory in config["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in ("helper", "recipient"):
            path = batch_root / trajectory_id / role / "artifact_manifest.json"
            source_manifests[f"{trajectory_id}:{role}"] = {
                "path": str(path.relative_to(batch_root)),
                "sha256": _sha256(path),
            }
    config_file_sha = _sha256(config_path) if config_path and config_path.is_file() else None
    launched_config_sha = launch_manifest.get("config_sha256") if launch_manifest else None
    status = _git_value(repository_root, "status", "--porcelain")
    return {
        "schema": "scenesense.phase2_analysis_provenance.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_root": str(batch_root),
        "evaluation_namespace": evaluation_name,
        "analysis_config": {
            "path": str(config_path) if config_path else None,
            "file_sha256": config_file_sha,
            "semantic_sha256": _semantic_sha256(config),
        },
        "capture_resolved_config": {
            "path": str(resolved_config_path.relative_to(batch_root)),
            "file_sha256": _sha256(resolved_config_path)
            if resolved_config_path.is_file()
            else None,
            "semantic_sha256": _semantic_sha256(resolved_config)
            if resolved_config is not None
            else None,
            "semantic_match_to_analysis_config": resolved_config == config,
        },
        "detached_launch": {
            "path": str(launch_manifest_path),
            "file_sha256": _sha256(launch_manifest_path)
            if launch_manifest_path.is_file()
            else None,
            "recorded_config_sha256": launched_config_sha,
            "analysis_config_file_matches_launch": (
                config_file_sha == launched_config_sha
                if config_file_sha is not None and launched_config_sha is not None
                else None
            ),
            "inference_timing_citable": launch_manifest.get("inference_timing_citable")
            if launch_manifest
            else None,
        },
        "capture_artifacts": {
            "batch_manifest_sha256": _sha256(batch_root / "batch_manifest.json"),
            "completion_sentinel_sha256": _sha256(batch_root / "COMPLETED.json"),
            "role_artifact_manifests": source_manifests,
        },
        "analysis_code": {
            "repository_commit": _git_value(repository_root, "rev-parse", "HEAD"),
            "repository_dirty": bool(status),
            "repository_status_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None
            ),
            "files": {
                str(path.relative_to(repository_root)): _sha256(path) for path in code_paths
            },
        },
    }


def _write_evaluation_manifest(evaluation_dir: Path) -> None:
    manifest_path = evaluation_dir / "evaluation_artifact_manifest.json"
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(evaluation_dir.iterdir())
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "scenesense.phase2_evaluation_artifact_manifest.v1",
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def replay(
    batch_root: Path,
    config: Mapping[str, object],
    *,
    evaluation_name: str = "evaluation",
    config_path: Optional[Path] = None,
) -> dict:
    evaluation_name = _validated_output_name(evaluation_name, "evaluation")
    evaluation_dir = batch_root / evaluation_name
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    all_metrics: list[dict] = []
    all_warnings: list[dict] = []
    isolation_rows: list[dict] = []
    for trajectory in config["trajectories"]:
        metrics, warnings, isolation = replay_trajectory(batch_root, trajectory)
        all_metrics.extend(metrics)
        all_warnings.extend(warnings)
        isolation_rows.append(isolation)
    warning_diagnostics, fragmentation_diagnostics = _warning_diagnostics(
        all_metrics, all_warnings
    )
    diagnostics_by_key = {
        (str(row["trajectory_id"]), str(row["arm_id"])): row
        for row in warning_diagnostics
    }
    diagnostic_columns = {
        "warning_event_count",
        "warning_frame_count",
        "warning_frame_rate",
        "warning_events_per_active_frame",
        "unique_warning_track_count",
        "target_warning_event_count",
        "target_warning_frame_count",
        "target_warning_track_count",
        "non_target_warning_event_count",
        "non_target_warning_frame_count",
        "non_target_warning_frame_rate",
        "non_target_warning_track_count",
        "matched_non_target_warning_event_count",
        "unmatched_warning_event_count",
        "unmatched_warning_frame_count",
        "unmatched_warning_frame_rate",
        "unmatched_warning_track_count",
        "false_warning_adjudication_status",
    }
    for metric in all_metrics:
        diagnostic = diagnostics_by_key[(str(metric["trajectory_id"]), str(metric["arm_id"]))]
        metric.update({key: diagnostic[key] for key in diagnostic_columns})
    pd.DataFrame(all_metrics).to_csv(evaluation_dir / "arm_metrics.csv", index=False)
    pd.DataFrame(all_warnings).to_csv(evaluation_dir / "warning_events.csv", index=False)
    pd.DataFrame(warning_diagnostics).to_csv(
        evaluation_dir / "warning_diagnostics.csv", index=False
    )
    pd.DataFrame(fragmentation_diagnostics).to_csv(
        evaluation_dir / "warning_fragmentation.csv", index=False
    )
    (evaluation_dir / "arm_state_manifest.json").write_text(
        json.dumps(
            {
                "schema": "scenesense.phase2_arm_state_isolation.v1",
                "independent_state_per_arm": all(item["independent"] for item in isolation_rows),
                "shared_mutable_state_detected": not all(
                    item["independent"] for item in isolation_rows
                ),
                "trajectories": isolation_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (evaluation_dir / "paired_semantics.json").write_text(
        json.dumps(
            {
                "schema": "scenesense.phase2_paired_semantics.v1",
                "declared_arm_differences": ["publication_selection"],
                "hidden_world_state_divergence": False,
                "world_capture_reused_across_arms": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    positive = next(
        item for item in config["trajectories"] if item["scenario_role"] == "controlled_positive_occlusion"
    )
    positive_id = str(positive["trajectory_id"])
    role_dir = batch_root / positive_id / "helper"
    warning = _select_target_chain_warning(all_warnings, positive_id)
    if warning:
        frame_id = int(warning["frame_id"])
        evidence_sources = json.loads(str(warning["evidence_sources"]))
        source_roles = sorted(set(str(item) for item in evidence_sources))
        warning_status = "observed_registered_target"
    else:
        source_roles = ["helper", "recipient"]
        role_dir = batch_root / positive_id / "helper"
        raw_candidate = sorted((role_dir / "retained_inputs").glob("frame_*_inputs.npz"))[0]
        frame_id = int(raw_candidate.stem.split("_")[1])
        warning_status = "missed_but_computable"
    source_artifacts = {}
    for source_role in source_roles:
        role_dir = batch_root / positive_id / source_role
        paths = {
            "capture": role_dir / f"retained_inputs/frame_{frame_id:08d}_inputs.npz",
            "inference": role_dir / f"retained_inputs/frame_{frame_id:08d}_logits.npz",
            "tracking": role_dir / "runtime/causal_tracks.csv",
            "action": role_dir / "runtime/causal_decisions.jsonl",
        }
        for stage, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"target-chain {source_role} {stage} artifact is missing: {path}"
                )
        source_artifacts[source_role] = {
            stage: str(path.relative_to(batch_root)) for stage, path in paths.items()
        }
    chain = {
        "schema": "scenesense.phase2_capture_warning_truth_chain.v3",
        "target_contract": {
            "trajectory_id": positive_id,
            "scenario_role": positive["scenario_role"],
            "target_truth_role_prefix": positive["target_truth_role_prefix"],
            "required_target_hazard_match": 1,
            "selection": "earliest_registered_target_warning_hazard_only_tie_preferred",
        },
        "capture": {
            "frame_id": frame_id,
            "source_roles": source_roles,
            "artifacts_by_source": {
                role: artifacts["capture"] for role, artifacts in source_artifacts.items()
            },
        },
        "inference": {
            "artifacts_by_source": {
                role: artifacts["inference"] for role, artifacts in source_artifacts.items()
            }
        },
        "tracking": {
            "artifacts_by_source": {
                role: artifacts["tracking"] for role, artifacts in source_artifacts.items()
            }
        },
        "action": {
            "artifacts_by_source": {
                role: artifacts["action"] for role, artifacts in source_artifacts.items()
            }
        },
        "transport": {
            "mode": "local_exact_v2_bytes",
            "artifact": f"{evaluation_name}/arm_metrics.csv",
        },
        "map_install": {"engine": "RecipientMapEngineV2", "arm_state_isolated": True},
        "warning": {"status": warning_status, "event": warning},
        "truth_score": {
            "namespace": "evaluation_truth",
            "artifact": str(
                _single(
                    batch_root / positive_id / "recipient" / "evaluation_truth",
                    "*_ground_truth.csv",
                ).relative_to(batch_root)
            ),
        },
    }
    (evaluation_dir / "capture_warning_truth_chain.json").write_text(
        json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = _analysis_provenance(
        batch_root,
        config,
        config_path.resolve() if config_path else None,
        evaluation_name,
    )
    (evaluation_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "scenesense.phase2_pilot_replay.v3",
        "status": "complete",
        "confirmatory_performance_evidence": False,
        "parameter_status": "provisional_for_computability_only",
        "warning_calibration_status": "not_frozen_from_two_trajectory_pilot",
        "false_warning_status": "non_target_proxy_only_not_hazard_adjudicated",
        "trajectory_count": len(config["trajectories"]),
        "arm_metric_rows": len(all_metrics),
        "warning_event_rows": len(all_warnings),
        "warning_diagnostic_rows": len(warning_diagnostics),
        "fragmentation_diagnostic_rows": len(fragmentation_diagnostics),
        "target_chain_status": warning_status,
        "evaluation_namespace": evaluation_name,
    }
    (evaluation_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_evaluation_manifest(evaluation_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data_collection/configs/phase2_paired_causal_pilot_integration_v1.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="create-only output directory basename (for example evaluation_v2)",
    )
    args = parser.parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("integration config root must be a mapping")
    result = replay(
        args.batch_root.resolve(),
        config,
        evaluation_name=args.evaluation_name,
        config_path=args.config.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
