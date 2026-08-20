"""Post-capture recipient-availability replay for the exact-16 factor smoke.

The adapter consumes immutable paired role artifacts.  It replays the frozen
v3 source tracker, serializes v2 map contributions, installs them in a v3
recipient map through an explicitly local-loopback transport, and records the
install-to-consumer boundary.  CARLA truth is joined only after that causal
replay to identify the registered target for endpoint evaluation.

This module has no CARLA/OAI launcher and cannot admit a batch by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from phase2_map_sharing.engine_v3 import RecipientMapEngineV3
from phase2_map_sharing.factor_smoke_runtime_contract import (
    CausalPolicyRuntimeAuditor,
    FeatureComponent,
    FeatureSample,
    LOCAL_LOOPBACK,
    RecipientAvailabilityRecorder,
    aggregate_guardrail_reports,
    analyze_installed_track_guardrails,
    build_recipient_map_target_match,
    build_recipient_available_endpoint,
    canonical_sha256,
    summarize_policy_audits,
)
from phase2_map_sharing.replay_paired_pilot import (
    CLOCK_ID,
    _contribution,
    _latest_ego_state,
    _recipient_state,
)
from phase2_map_sharing.replay_warning_repair_screen import (
    _replay_source_tracker_v3,
)
from phase2_map_sharing.source_tracker_v3 import TRACKER_V3_VERSION
from phase2_map_sharing.static_truth_adjudication_v1 import (
    load_verified_static_catalog_v1,
    normalize_static_semantic_class_v1,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POSTFLIGHT_SCHEMA = "scenesense.phase2_factor_smoke_postflight.v1"
BATCH_VALIDATION_FILENAME = "factor_smoke_validation.json"
BATCH_RESULT_FILENAME = "factor_smoke_results.json"
DEPENDENCY_PATHS = {
    "factor_smoke_postflight": Path(__file__).resolve(),
    "factor_smoke_runtime_contract": REPO_ROOT
    / "phase2_map_sharing/factor_smoke_runtime_contract.py",
    "causal_contract": REPO_ROOT / "phase2_map_sharing/causal_contract.py",
    "schemas_v2": REPO_ROOT / "phase2_map_sharing/schemas_v2.py",
    "source_tracker_v3": REPO_ROOT / "phase2_map_sharing/source_tracker_v3.py",
    "engine_v2": REPO_ROOT / "phase2_map_sharing/engine_v2.py",
    "engine_v3": REPO_ROOT / "phase2_map_sharing/engine_v3.py",
    "replay_warning_repair_screen": REPO_ROOT
    / "phase2_map_sharing/replay_warning_repair_screen.py",
    "replay_paired_pilot": REPO_ROOT / "phase2_map_sharing/replay_paired_pilot.py",
    "replay_calibration_grid": REPO_ROOT
    / "phase2_map_sharing/replay_calibration_grid.py",
    "static_truth_adjudication_v1": REPO_ROOT
    / "phase2_map_sharing/static_truth_adjudication_v1.py",
    "factor_smoke_validator": REPO_ROOT
    / "data_collection/validate_phase2_factor_realization_smoke.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} under {directory}, got {len(matches)}")
    return matches[0]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected YAML mapping: {path}")
    return dict(value)


def _verified_dependency_fingerprints(
    smoke_config: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    pinned = smoke_config["recipient_endpoint_runtime"]["dependency_sha256"]
    if not isinstance(pinned, Mapping) or set(pinned) != set(DEPENDENCY_PATHS):
        raise ValueError("recipient endpoint dependency fingerprint keys drifted")
    result: dict[str, dict[str, str]] = {}
    for name, expected_path in DEPENDENCY_PATHS.items():
        contract = pinned[name]
        if not isinstance(contract, Mapping) or set(contract) != {"path", "sha256"}:
            raise ValueError(f"dependency {name} contract must contain path and sha256")
        configured_path = _resolve(contract["path"])
        if configured_path != expected_path.resolve():
            raise ValueError(f"dependency {name} path drifted")
        observed = _sha256_file(configured_path)
        if observed != str(contract["sha256"]):
            raise ValueError(f"dependency {name} hash drifted")
        result[name] = {"path": str(configured_path), "sha256": observed}
    return result


def _captured_role_provenance(
    role_dirs: Mapping[str, Path],
) -> dict[str, dict[str, str]]:
    """Require a capture-time checkpoint digest, then recheck current bytes."""

    result: dict[str, dict[str, str]] = {}
    for role, role_dir in role_dirs.items():
        manifest_path = _single(role_dir / "manifests", "*_manifest.json")
        resolved_path = _single(role_dir / "manifests", "*_resolved_config.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        nested = manifest.get("phase2_paired_causal")
        if not isinstance(nested, Mapping):
            raise ValueError(f"{role} manifest lacks Phase-2 checkpoint identity")
        identity_fields = (
            "checkpoint_path_at_capture",
            "checkpoint_sha256",
            "checkpoint_identity_basis",
        )
        if any(manifest.get(field) != nested.get(field) for field in identity_fields):
            raise ValueError(f"{role} root/nested checkpoint identity differs")
        if manifest.get("checkpoint_identity_basis") != "capture_time_file_bytes":
            raise ValueError(f"{role} checkpoint identity basis drifted")
        captured_sha = str(manifest.get("checkpoint_sha256", "")).lower()
        if len(captured_sha) != 64 or any(
            character not in "0123456789abcdef" for character in captured_sha
        ):
            raise ValueError(f"{role} manifest lacks capture-time checkpoint_sha256")
        checkpoint = Path(str(manifest.get("checkpoint_path_at_capture", ""))).resolve()
        legacy_path = Path(str(manifest.get("checkpoint_path", ""))).resolve()
        if checkpoint != legacy_path:
            raise ValueError(f"{role} checkpoint capture path differs from model path")
        if not checkpoint.is_file():
            raise ValueError(f"{role} captured checkpoint is unavailable: {checkpoint}")
        current_sha = _sha256_file(checkpoint)
        if current_sha != captured_sha:
            raise ValueError(f"{role} checkpoint bytes differ from capture-time SHA-256")
        result[role] = {
            "model_sha256": captured_sha,
            "config_sha256": _sha256_file(resolved_path),
            "checkpoint_hash_status": "capture_time_sha256_recomputed_equal",
            "checkpoint_path_at_capture": str(checkpoint),
            "checkpoint_sha256_at_capture": captured_sha,
            "checkpoint_sha256_recomputed": current_sha,
            "checkpoint_sha256_equal": True,
            "checkpoint_identity_basis": "capture_time_file_bytes",
            "manifest_sha256": _sha256_file(manifest_path),
        }
    if len({record["model_sha256"] for record in result.values()}) != 1:
        raise ValueError("helper and recipient used different checkpoint bytes")
    return result


def _verify_role_artifact_manifest(role_dir: Path) -> dict[str, Any]:
    """Rehash every collector artifact listed in its sealed role manifest."""

    manifest_path = role_dir / "artifact_manifest.json"
    payload = _load_json_mapping(manifest_path, "collector artifact manifest")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"collector artifact manifest is empty: {manifest_path}")
    verified: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("collector artifact-manifest entry is invalid")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("collector artifact-manifest path escapes role directory")
        path = (role_dir / relative).resolve()
        try:
            path.relative_to(role_dir.resolve())
        except ValueError as exc:
            raise ValueError("collector artifact path escapes role directory") from exc
        if not path.is_file():
            raise ValueError(f"collector artifact is missing: {path}")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != int(entry.get("bytes", -1)) or digest != entry.get("sha256"):
            raise ValueError(f"collector artifact bytes/hash drifted: {path}")
        verified.append(
            {"path": str(relative), "bytes": size, "sha256": digest}
        )
    if payload.get("trajectory_id") != role_dir.parent.name:
        raise ValueError("collector artifact manifest trajectory ID drifted")
    if payload.get("source_role") != role_dir.name:
        raise ValueError("collector artifact manifest source role drifted")
    return {
        "source_role": role_dir.name,
        "artifact_manifest_path": str(manifest_path.resolve()),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "listed_file_count": len(verified),
        "listed_total_bytes": sum(item["bytes"] for item in verified),
        "listed_entries_sha256": canonical_sha256(verified),
        "all_listed_files_rehashed_equal": True,
    }


def _tracks_frame(
    cache: Mapping[int, Sequence[Mapping[str, object]]], frame_id: int
) -> pd.DataFrame:
    rows = list(cache.get(int(frame_id), ()))
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=(
            "source_track_id",
            "source_role",
            "tracker_version",
            "class_name",
            "world_x",
            "world_y",
            "world_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "score",
            "last_observed_timestamp_s",
        )
    )


def _track_observation_payload(role: str, frame_id: int, row: Mapping[str, object]) -> dict:
    return {
        "source_role": str(role),
        "source_track_id": str(row["source_track_id"]),
        "source_sample_id": f"{role}:{int(frame_id)}",
        "class_name": str(row["class_name"]),
        "world_x": float(row["world_x"]),
        "world_y": float(row["world_y"]),
        "world_z": float(row.get("world_z", 0.0)),
        "velocity_x": float(row["velocity_x"]),
        "velocity_y": float(row["velocity_y"]),
        "velocity_z": float(row.get("velocity_z", 0.0)),
        "score": float(row["score"]),
        "measured_at_s": float(row["last_observed_timestamp_s"]),
        "tracker_version": str(row["tracker_version"]),
    }


def _map_id_for_source(
    engine: RecipientMapEngineV3,
    role: str,
    track_id: str,
    installed_at_s: float,
) -> str:
    candidates = [
        str(canonical_track_id)
        for canonical_track_id, track in engine.tracks.items()
        if str(track.source_track_ids.get(role, "")) == str(track_id)
        and abs(float(track.source_capture_at_s.get(role, float("-inf"))) - installed_at_s)
        <= 1e-9
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"source-to-recipient association is not one-to-one: {role}:{track_id} -> {candidates}"
        )
    return candidates[0]


def _target_source_track(
    role: str,
    role_dir: Path,
    tracks_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    hazard_class: str,
    target_role_prefix: str,
    center_gate_m: float,
) -> str | None:
    truth = pd.read_csv(_single(role_dir / "evaluation_truth", "*_ground_truth.csv"))
    target = truth[truth["role_name"].astype(str).str.startswith(target_role_prefix)].copy()
    if target.empty:
        return None
    if target["actor_id"].astype(str).nunique() != 1:
        raise ValueError(f"registered target is not unique: {role_dir}")
    target_by_frame = target.set_index(target["frame_id"].astype(int), drop=False)
    candidates: list[tuple[float, float, str]] = []
    for frame_id, tracks in tracks_by_frame.items():
        if frame_id not in target_by_frame.index:
            continue
        truth_row = target_by_frame.loc[frame_id]
        if isinstance(truth_row, pd.DataFrame):
            raise ValueError("registered target truth duplicates a frame")
        for track in tracks:
            if normalize_static_semantic_class_v1(track["class_name"]) != hazard_class:
                continue
            distance = math.hypot(
                float(track["world_x"]) - float(truth_row["origin_x"]),
                float(track["world_y"]) - float(truth_row["origin_y"]),
            )
            if distance <= center_gate_m + 1e-12:
                candidates.append(
                    (
                        float(track["last_observed_timestamp_s"]),
                        distance,
                        str(track["source_track_id"]),
                    )
                )
    return None if not candidates else min(candidates)[2]


def _truth_match_attempts(
    trajectory_dir: Path,
    role_dirs: Mapping[str, Path],
    attempts: Sequence[Mapping[str, Any]],
    observation_payload_by_sha: Mapping[str, Mapping[str, Any]],
    *,
    center_gate_m: float,
) -> tuple[dict[str, bool | None], dict[str, Any]]:
    truth_by_role_frame: dict[tuple[str, int], pd.DataFrame] = {}
    for role, role_dir in role_dirs.items():
        truth = pd.read_csv(_single(role_dir / "evaluation_truth", "*_ground_truth.csv"))
        for frame_id, frame in truth.groupby(truth["frame_id"].astype(int)):
            truth_by_role_frame[(role, int(frame_id))] = frame
    static_path = trajectory_dir / "static_environment_truth/static_environment_objects.csv"
    static_truth: pd.DataFrame | None = None
    if static_path.is_file():
        static_truth = load_verified_static_catalog_v1(static_path.parent)
    result: dict[str, bool | None] = {}
    dynamic_matches = 0
    static_matches = 0
    for attempt in attempts:
        observation = observation_payload_by_sha.get(
            str(attempt["source_observation_sha256"])
        )
        if observation is None:
            result[str(attempt["attempt_id"])] = None
            continue
        role = str(observation["source_role"])
        frame_id = int(str(observation["source_sample_id"]).rsplit(":", 1)[1])
        truth = truth_by_role_frame.get((role, frame_id))
        if truth is None:
            result[str(attempt["attempt_id"])] = None
            continue
        normalized_class = normalize_static_semantic_class_v1(observation["class_name"])
        matches = []
        for row in truth.to_dict("records"):
            if normalize_static_semantic_class_v1(row["class_name"]) != normalized_class:
                continue
            matches.append(
                math.hypot(
                    float(observation["world_x"]) - float(row["origin_x"]),
                    float(observation["world_y"]) - float(row["origin_y"]),
                )
            )
        dynamic_match = bool(matches and min(matches) <= center_gate_m + 1e-12)
        static_match = False
        if not dynamic_match and static_truth is not None:
            static_rows = static_truth[
                static_truth["semantic_class"].map(normalize_static_semantic_class_v1)
                == normalized_class
            ]
            if not static_rows.empty:
                distances = [
                    math.hypot(
                        float(observation["world_x"])
                        - float(row["bbox_center_x_m"]),
                        float(observation["world_y"])
                        - float(row["bbox_center_y_m"]),
                    )
                    for row in static_rows.to_dict("records")
                ]
                static_match = bool(
                    distances and min(distances) <= center_gate_m + 1e-12
                )
        dynamic_matches += int(dynamic_match)
        static_matches += int(static_match)
        result[str(attempt["attempt_id"])] = dynamic_match or static_match
    coverage = {
        "basis": "dynamic_actor_truth_plus_verified_static_vehicle_catalog",
        "dynamic_truth_available": True,
        "static_truth_available": static_truth is not None,
        "static_truth_path": str(static_path) if static_path.is_file() else None,
        "attempt_count": len(attempts),
        "dynamic_match_count": dynamic_matches,
        "static_only_match_count": static_matches,
        "coverage_complete_for_dynamic_actors_and_static_car_truck_bus": (
            static_truth is not None
        ),
        "report_only": True,
    }
    coverage["coverage_sha256"] = canonical_sha256(coverage)
    return result, coverage


def _relative_diagnostics(
    objects: Sequence[Mapping[str, float]],
    *,
    recipient_x: float,
    recipient_y: float,
    recipient_vx: float,
    recipient_vy: float,
) -> tuple[float, float]:
    if not objects:
        raise ValueError("relative diagnostics require at least one causal object")
    ttc_values = []
    clearance_values = []
    for item in objects:
        rx = float(item["x_m"]) - recipient_x
        ry = float(item["y_m"]) - recipient_y
        rvx = float(item["vx_mps"]) - recipient_vx
        rvy = float(item["vy_mps"]) - recipient_vy
        speed_sq = rvx * rvx + rvy * rvy
        tca = 0.0 if speed_sq <= 1e-12 else max(0.0, -(rx * rvx + ry * rvy) / speed_sq)
        ttc_values.append(tca)
        clearance_values.append(math.hypot(rx + rvx * tca, ry + rvy * tca))
    return min(ttc_values), min(clearance_values)


def _snapshot_track(snapshot: Mapping[str, Any], map_track_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in snapshot.get("tracks", [])
        if str(item.get("canonical_track_id")) == str(map_track_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"recipient map snapshot lacks unique canonical track {map_track_id}"
        )
    row = matches[0]
    return {
        "class_name": str(row["class_name"]),
        "x_m": float(row["x_m"]),
        "y_m": float(row["y_m"]),
        "snapshot_at_s": float(snapshot["timestamp_s"]),
    }


def _registered_target_truth_state(
    target_truth: pd.DataFrame,
    *,
    frame_id: int,
    available_at_s: float,
    target_role_prefix: str,
) -> dict[str, Any]:
    rows = target_truth[
        (target_truth["frame_id"].astype(int) == int(frame_id))
        & target_truth["role_name"].astype(str).str.startswith(target_role_prefix)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"registered target truth is not unique at frame {frame_id}"
        )
    row = rows.iloc[0]
    truth_at_s = float(row["carla_timestamp"])
    if abs(truth_at_s - float(available_at_s)) > 1e-9:
        raise ValueError("target truth and recipient availability are not time-aligned")
    return {
        "class_name": str(row["class_name"]),
        "x_m": float(row["origin_x"]),
        "y_m": float(row["origin_y"]),
        "observed_at_s": truth_at_s,
    }


def _policy_samples(
    *,
    stage: str,
    smoke_config: Mapping[str, Any],
    decision_at_s: float,
    prior_available_at_s: float,
    helper_state: Mapping[str, Any],
    recipient_state_row: Mapping[str, Any],
    recipient_acceleration_mps2: float,
    prior_helper_track_count: int,
    installed_snapshot: Mapping[str, Any],
    helper_contribution: Any,
) -> dict[str, FeatureSample]:
    feature_names = smoke_config["policy_feature_contract"][f"{stage}_features"]
    fixture = smoke_config["policy_projection_exercise"]["fixture_backed_fields"]
    samples: dict[str, FeatureSample] = {}
    for name in feature_names:
        if name in fixture:
            samples[name] = FeatureSample(
                value=fixture[name]["value"],
                source_stage=fixture[name]["runtime_source_stage"],
                observed_at_s=prior_available_at_s,
                available_at_s=prior_available_at_s,
                evidence_kind="preregistered_fixture",
            )

    recipient_x = float(recipient_state_row["world_x"])
    recipient_y = float(recipient_state_row["world_y"])
    recipient_vx = float(recipient_state_row["velocity_x"])
    recipient_vy = float(recipient_state_row["velocity_y"])
    helper_x = float(helper_state["world_x"])
    helper_y = float(helper_state["world_y"])
    helper_vx = float(helper_state["velocity_x"])
    helper_vy = float(helper_state["velocity_y"])
    components = (
        FeatureComponent("helper_localization", decision_at_s, decision_at_s),
        FeatureComponent("recipient_state_transport", decision_at_s, decision_at_s),
    )

    installed_tracks = list(installed_snapshot["tracks"])
    installed_ttc, _ = _relative_diagnostics(
        installed_tracks,
        recipient_x=recipient_x,
        recipient_y=recipient_y,
        recipient_vx=recipient_vx,
        recipient_vy=recipient_vy,
    )
    installed_values = {
        "installed_map_object_count": len(installed_tracks),
        "installed_map_max_aoi_s": max(float(item["map_aoi_s"]) for item in installed_tracks),
        "installed_map_max_position_sigma_m": max(
            float(item["position_sigma_m"]) for item in installed_tracks
        ),
        "installed_map_min_estimated_ttc_s": installed_ttc,
    }
    for name, value in installed_values.items():
        if name in feature_names:
            samples[name] = FeatureSample(
                value=value,
                source_stage="recipient_map_feedback_transport",
                observed_at_s=decision_at_s,
                available_at_s=decision_at_s,
                evidence_kind="local_loopback_transport_abstraction",
            )
    placement_values = {
        "prior_source_track_count": (prior_helper_track_count, "causal_tracker"),
        "recipient_speed_mps": (math.hypot(recipient_vx, recipient_vy), "recipient_state_transport"),
        "recipient_acceleration_mps2": (
            recipient_acceleration_mps2,
            "recipient_state_transport",
        ),
    }
    for name, (value, source) in placement_values.items():
        if name in feature_names:
            sample_at_s = (
                prior_available_at_s
                if name == "prior_source_track_count"
                else decision_at_s
            )
            samples[name] = FeatureSample(
                value=value,
                source_stage=source,
                observed_at_s=sample_at_s,
                available_at_s=sample_at_s,
                evidence_kind=(
                    "local_loopback_transport_abstraction"
                    if source == "recipient_state_transport"
                    else "observed"
                ),
            )
    relative_values = {
        "helper_recipient_relative_x_m": helper_x - recipient_x,
        "helper_recipient_relative_y_m": helper_y - recipient_y,
        "helper_recipient_relative_vx_mps": helper_vx - recipient_vx,
        "helper_recipient_relative_vy_mps": helper_vy - recipient_vy,
    }
    for name, value in relative_values.items():
        if name in feature_names:
            samples[name] = FeatureSample(
                value=value,
                source_stage="derived_relative_kinematics",
                observed_at_s=decision_at_s,
                available_at_s=decision_at_s,
                component_provenance=components,
                evidence_kind="local_loopback_transport_abstraction",
            )

    if stage == "publication":
        objects = [
            {
                "x_m": obj.x_m,
                "y_m": obj.y_m,
                "vx_mps": obj.vx_mps,
                "vy_mps": obj.vy_mps,
            }
            for obj in helper_contribution.objects
        ]
        current_ttc, current_clearance = _relative_diagnostics(
            objects,
            recipient_x=recipient_x,
            recipient_y=recipient_y,
            recipient_vx=recipient_vx,
            recipient_vy=recipient_vy,
        )
        current_values = {
            "current_causal_track_count": len(helper_contribution.objects),
            "current_min_track_confidence": min(
                float(obj.confidence) for obj in helper_contribution.objects
            ),
            "current_min_estimated_ttc_s": current_ttc,
            "current_min_estimated_clearance_m": current_clearance,
            "current_max_position_sigma_m": max(
                math.sqrt(max(float(obj.state_covariance[0]), float(obj.state_covariance[5])))
                for obj in helper_contribution.objects
            ),
            "recipient_state_age_s": 0.0,
        }
        for name, value in current_values.items():
            samples[name] = FeatureSample(
                value=value,
                source_stage=(
                    "recipient_state_transport"
                    if name == "recipient_state_age_s"
                    else "causal_tracker"
                ),
                observed_at_s=decision_at_s,
                available_at_s=decision_at_s,
                evidence_kind=(
                    "local_loopback_transport_abstraction"
                    if name == "recipient_state_age_s"
                    else "observed"
                ),
            )
    missing = sorted(set(feature_names) - set(samples))
    if missing:
        raise ValueError(f"postflight cannot truthfully populate policy features: {missing}")
    return samples


def analyze_trajectory_artifacts(
    *,
    trajectory_dir: Path,
    trajectory_row: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
    target_role_prefix: str = "phase2_registered_target_",
    evaluation_center_gate_m: float = 5.0,
) -> dict[str, Any]:
    """Replay one completed real trajectory and return bundle-ready evidence."""

    trajectory_dir = Path(trajectory_dir).resolve()
    trajectory_id = str(trajectory_row["trajectory_id"])
    if trajectory_dir.name != trajectory_id:
        raise ValueError("trajectory directory does not match trajectory row")
    role_dirs = {role: trajectory_dir / role for role in ("helper", "recipient")}
    runtime = smoke_config["recipient_endpoint_runtime"]
    warning_path = _resolve(runtime["warning_repair_config"])
    replay_path = _resolve(runtime["replay_config"])
    if _sha256_file(warning_path) != runtime["warning_repair_config_sha256"]:
        raise ValueError("warning-repair dependency hash drifted")
    if _sha256_file(replay_path) != runtime["replay_config_sha256"]:
        raise ValueError("replay dependency hash drifted")
    if runtime["tracker_version"] != TRACKER_V3_VERSION:
        raise ValueError("source tracker version drifted")
    if runtime["map_engine"] != "RecipientMapEngineV3":
        raise ValueError("recipient map engine drifted")
    if runtime["transport_mode"] != LOCAL_LOOPBACK:
        raise ValueError("factor smoke endpoint must remain local-loopback")
    handoff_delay_s = float(runtime["consumer_handoff_delay_s"])
    if handoff_delay_s != 0.0 or not math.isfinite(handoff_delay_s):
        raise ValueError("exact factor smoke requires its pinned zero-delay loopback handoff")

    dependency_fingerprints = _verified_dependency_fingerprints(smoke_config)

    warning_config = _load_yaml(warning_path)
    # This dependency carries the prior replay/truth contract.  It is pinned
    # and fingerprinted even though warning generation is intentionally absent.
    _load_yaml(replay_path)
    factor_path = trajectory_dir / "scenario/factor_realization.json"
    realized_trace_path = trajectory_dir / "scenario/realized_trace.csv"
    factor_artifact = _load_json_mapping(factor_path, "factor realization")
    if factor_artifact.get("trajectory_id") != trajectory_id:
        raise ValueError("factor realization trajectory ID drifted")
    controlled_positive = bool(trajectory_row.get("controlled_hazard_present"))
    if controlled_positive:
        realized_factors = factor_artifact.get("realized_factors")
        if not isinstance(realized_factors, Mapping):
            raise ValueError("positive factor realization lacks realized factors")
        realized_onset_s: float | None = float(
            realized_factors["realized_hazard_onset_s"]
        )
    else:
        if factor_artifact.get("registered_target_absent") is not True:
            raise ValueError("benign factor realization lacks target-absence evidence")
        realized_onset_s = None

    input_paths = [
        warning_path,
        replay_path,
        factor_path,
        realized_trace_path,
        *DEPENDENCY_PATHS.values(),
    ]
    static_truth_dir = trajectory_dir / "static_environment_truth"
    input_paths.extend(
        [
            static_truth_dir / "artifact_manifest.json",
            static_truth_dir / "static_environment_objects.csv",
            static_truth_dir / "static_environment_snapshot.json",
        ]
    )
    for role_dir in role_dirs.values():
        manifest_path = _single(role_dir / "manifests", "*_manifest.json")
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(
            str(manifest_payload.get("checkpoint_path_at_capture", ""))
        ).resolve()
        input_paths.extend(
            [
                role_dir / "runtime/final_detections.csv",
                role_dir / "runtime/ego_states.csv",
                _single(role_dir / "streams", "*_metrics.csv"),
                _single(role_dir / "evaluation_truth", "*_ground_truth.csv"),
                manifest_path,
                role_dir / "artifact_manifest.json",
                _single(role_dir / "manifests", "*_resolved_config.json"),
                checkpoint_path,
            ]
        )
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    before = {str(path): _sha256_file(path) for path in input_paths}

    tracks_by_role: dict[str, dict[int, list[dict]]] = {}
    tracker_diagnostics = {}
    for role, role_dir in role_dirs.items():
        tracks, diagnostics = _replay_source_tracker_v3(
            role_dir, role, warning_config
        )
        tracks_by_role[role] = tracks
        tracker_diagnostics[role] = diagnostics

    helper_metrics = pd.read_csv(_single(role_dirs["helper"] / "streams", "*_metrics.csv"))
    recipient_metrics = pd.read_csv(
        _single(role_dirs["recipient"] / "streams", "*_metrics.csv")
    )
    if helper_metrics["frame_id"].duplicated().any() or recipient_metrics[
        "frame_id"
    ].duplicated().any():
        raise ValueError("paired role metrics contain duplicate frame IDs")
    helper_times = helper_metrics.set_index("frame_id")["carla_timestamp"].astype(float)
    recipient_times = recipient_metrics.set_index("frame_id")["carla_timestamp"].astype(float)
    common_frames = sorted(set(helper_times.index.astype(int)) & set(recipient_times.index.astype(int)))
    if len(common_frames) != len(helper_times) or len(common_frames) != len(recipient_times):
        raise ValueError("paired role frame coverage differs")
    if any(
        abs(float(helper_times.loc[frame]) - float(recipient_times.loc[frame])) > 1e-9
        for frame in common_frames
    ):
        raise ValueError("paired role timestamps differ")

    map_config = warning_config["recipient_map"]
    engine = RecipientMapEngineV3(
        "recipient",
        association_gate_m=float(map_config["association_gate_m"]),
        association_sigma_multiplier=float(map_config["association_sigma_multiplier"]),
        warning_sigma_multiplier=float(map_config["warning_uncertainty_multiplier"]),
        track_ttl_s=float(map_config["track_ttl_s"]),
        max_transport_age_s=float(map_config["max_transport_age_s"]),
        warning_horizon_s=float(map_config["warning_horizon_s"]),
        warning_emission_confidence_floor=float(
            map_config["warning_emission_confidence_floor"]
        ),
        safety_radius_m_by_class=map_config["safety_radius_m_by_class"],
    )
    recorder = RecipientAvailabilityRecorder(
        trajectory_id=trajectory_id,
        clock_id=CLOCK_ID,
        transport_mode=LOCAL_LOOPBACK,
    )
    role_provenance = _captured_role_provenance(role_dirs)
    recipient_ego = pd.read_csv(role_dirs["recipient"] / "runtime/ego_states.csv")
    helper_ego = pd.read_csv(role_dirs["helper"] / "runtime/ego_states.csv")
    policy_auditor = CausalPolicyRuntimeAuditor.from_config(
        smoke_config,
        trajectory_id=trajectory_id,
        arm_id="fixed_local_loopback_projection_audit",
        clock_id=CLOCK_ID,
        decision_locus="helper",
    )
    policy_audit_exercised = False
    registered_observations: set[tuple[str, str, str]] = set()
    confirmed_tracks: set[tuple[str, str]] = set()
    observation_payload_by_sha: dict[str, dict[str, Any]] = {}
    previous_timestamp_s: float | None = None
    previous_helper_track_count = 0
    previous_recipient_speed_mps: float | None = None
    map_state_by_install_ref: dict[tuple[str, str], dict[str, Any]] = {}
    frame_by_install_ref: dict[tuple[str, str], int] = {}

    for sequence, frame_id in enumerate(common_frames):
        captured_at_s = float(recipient_times.loc[frame_id])
        available_at_s = captured_at_s + handoff_delay_s
        recipient_state = _recipient_state(
            _latest_ego_state(recipient_ego, captured_at_s), captured_at_s
        )
        recipient_ego_row = _latest_ego_state(recipient_ego, captured_at_s)
        helper_ego_row = _latest_ego_state(helper_ego, captured_at_s)
        frames = {
            role: _tracks_frame(tracks_by_role[role], frame_id)
            for role in ("recipient", "helper")
        }
        observation_sha_by_role_track: dict[tuple[str, str], str] = {}
        for role, frame in frames.items():
            for row in frame.to_dict("records"):
                payload = _track_observation_payload(role, frame_id, row)
                observation_sha = canonical_sha256(payload)
                observation_payload_by_sha[observation_sha] = payload
                key = (role, str(row["source_track_id"]), observation_sha)
                if key not in registered_observations:
                    recorder.register_source_observation(
                        source_role=role,
                        source_track_id=str(row["source_track_id"]),
                        observation_sha256=observation_sha,
                        observed_at_s=float(row["last_observed_timestamp_s"]),
                    )
                    registered_observations.add(key)
                observation_sha_by_role_track[(role, str(row["source_track_id"]))] = observation_sha
                confirmation_key = (role, str(row["source_track_id"]))
                if confirmation_key not in confirmed_tracks:
                    recorder.record_source_confirmation(
                        source_role=role,
                        source_track_id=str(row["source_track_id"]),
                        confirmed_at_s=captured_at_s,
                    )
                    confirmed_tracks.add(confirmation_key)

        helper_contribution = _contribution(
            trajectory_id=trajectory_id,
            source_role="helper",
            sequence=sequence,
            captured_at_s=captured_at_s,
            tracks=frames["helper"],
            publication_action="PUBLISH_ALL",
            recipient=recipient_state,
            model_sha256=role_provenance["helper"]["model_sha256"],
            config_sha256=role_provenance["helper"]["config_sha256"],
        )
        snapshot_before_current_actions = engine.snapshot(captured_at_s, CLOCK_ID)
        policy_auditor.record_policy_state_exposure(
            sample_at_s=captured_at_s,
            source_track_count=len(helper_contribution.objects),
            installed_map_track_count=len(snapshot_before_current_actions["tracks"]),
        )
        recipient_speed_mps = math.hypot(
            float(recipient_ego_row["velocity_x"]),
            float(recipient_ego_row["velocity_y"]),
        )
        recipient_acceleration_mps2 = 0.0
        if previous_timestamp_s is not None and previous_recipient_speed_mps is not None:
            dt = captured_at_s - previous_timestamp_s
            if dt > 1e-12:
                recipient_acceleration_mps2 = (
                    recipient_speed_mps - previous_recipient_speed_mps
                ) / dt
        if (
            not policy_audit_exercised
            and previous_timestamp_s is not None
            and snapshot_before_current_actions["tracks"]
            and helper_contribution.objects
        ):
            placement_samples = _policy_samples(
                stage="placement",
                smoke_config=smoke_config,
                decision_at_s=captured_at_s,
                prior_available_at_s=previous_timestamp_s,
                helper_state=helper_ego_row,
                recipient_state_row=recipient_ego_row,
                recipient_acceleration_mps2=recipient_acceleration_mps2,
                prior_helper_track_count=previous_helper_track_count,
                installed_snapshot=snapshot_before_current_actions,
                helper_contribution=helper_contribution,
            )
            publication_samples = _policy_samples(
                stage="publication",
                smoke_config=smoke_config,
                decision_at_s=captured_at_s,
                prior_available_at_s=previous_timestamp_s,
                helper_state=helper_ego_row,
                recipient_state_row=recipient_ego_row,
                recipient_acceleration_mps2=recipient_acceleration_mps2,
                prior_helper_track_count=previous_helper_track_count,
                installed_snapshot=snapshot_before_current_actions,
                helper_contribution=helper_contribution,
            )
            policy_auditor.consume(
                stage="placement",
                decision_id=f"{trajectory_id}:placement_projection:{sequence}",
                decision_at_s=captured_at_s,
                action=smoke_config["policy_projection_exercise"]["fixed_actions"]["placement"],
                samples=placement_samples,
            )
            policy_auditor.consume(
                stage="publication",
                decision_id=f"{trajectory_id}:publication_projection:{sequence}",
                decision_at_s=captured_at_s,
                action=smoke_config["policy_projection_exercise"]["fixed_actions"]["publication"],
                samples=publication_samples,
            )
            policy_auditor.exercise_forbidden_canary(
                stage="placement",
                decision_id=f"{trajectory_id}:forbidden_canary:{sequence}",
                decision_at_s=captured_at_s,
                action=smoke_config["policy_projection_exercise"]["fixed_actions"]["placement"],
                valid_samples=placement_samples,
            )
            policy_audit_exercised = True

        recipient_contribution = _contribution(
            trajectory_id=trajectory_id,
            source_role="recipient",
            sequence=sequence,
            captured_at_s=captured_at_s,
            tracks=frames["recipient"],
            publication_action="PUBLISH_ALL",
            recipient=recipient_state,
            model_sha256=role_provenance["recipient"]["model_sha256"],
            config_sha256=role_provenance["recipient"]["config_sha256"],
        )
        result = engine.install(recipient_contribution, captured_at_s, CLOCK_ID)
        if result != "accepted":
            raise RuntimeError(f"recipient contribution rejected: {result}")
        recipient_snapshot = engine.snapshot(available_at_s, CLOCK_ID)
        for obj in recipient_contribution.objects:
            map_id = _map_id_for_source(
                engine, "recipient", obj.source_track_id, captured_at_s
            )
            local_install_id = (
                f"{recipient_contribution.contribution_id}:{obj.source_track_id}"
            )
            recorder.record_recipient_local_install(
                local_install_id=local_install_id,
                source_track_id=obj.source_track_id,
                source_observation_sha256=observation_sha_by_role_track[
                    ("recipient", obj.source_track_id)
                ],
                recipient_map_track_id=map_id,
                confirmed_at_s=captured_at_s,
                installed_at_s=captured_at_s,
                available_at_s=available_at_s,
            )
            ref = ("recipient_local_install", local_install_id)
            map_state_by_install_ref[ref] = _snapshot_track(recipient_snapshot, map_id)
            frame_by_install_ref[ref] = int(frame_id)

        result = engine.install(helper_contribution, captured_at_s, CLOCK_ID)
        if result != "accepted":
            raise RuntimeError(f"helper contribution rejected: {result}")
        helper_snapshot = engine.snapshot(available_at_s, CLOCK_ID)
        for obj in helper_contribution.objects:
            map_id = _map_id_for_source(
                engine, "helper", obj.source_track_id, captured_at_s
            )
            attempt_id = f"{helper_contribution.contribution_id}:{obj.source_track_id}"
            recorder.record_install_attempt(
                attempt_id=attempt_id,
                contribution_id=helper_contribution.contribution_id,
                source_role="helper",
                source_track_id=obj.source_track_id,
                source_observation_sha256=observation_sha_by_role_track[
                    ("helper", obj.source_track_id)
                ],
                published_at_s=helper_contribution.published_at_s,
                attempted_at_s=captured_at_s,
                install_status="accepted",
                recipient_map_track_id=map_id,
                installed_at_s=captured_at_s,
                available_at_s=available_at_s,
            )
            ref = ("helper_install_attempt", attempt_id)
            map_state_by_install_ref[ref] = _snapshot_track(helper_snapshot, map_id)
            frame_by_install_ref[ref] = int(frame_id)
        previous_timestamp_s = captured_at_s
        previous_helper_track_count = len(helper_contribution.objects)
        previous_recipient_speed_mps = recipient_speed_mps

    availability = recorder.to_record()
    causal_policy_audit = policy_auditor.to_record()
    truth_match, truth_coverage = _truth_match_attempts(
        trajectory_dir,
        role_dirs,
        availability["install_attempts"],
        observation_payload_by_sha,
        center_gate_m=evaluation_center_gate_m,
    )
    guardrails = analyze_installed_track_guardrails(
        availability,
        evaluation_truth_match_by_attempt_id=truth_match,
    )
    retention = _retention_window_evidence(
        trajectory_dir,
        realized_onset_s=realized_onset_s,
    )
    result: dict[str, Any] = {
        "schema": POSTFLIGHT_SCHEMA,
        "trajectory_id": trajectory_id,
        "transport_scope": "local_loopback_only_no_oai_claim",
        "warnings_generated": False,
        "recipient_availability_provenance": availability,
        "installed_track_guardrails": guardrails,
        "causal_policy_audit": causal_policy_audit,
        "evaluation_truth_match_by_attempt_id": truth_match,
        "evaluation_truth_coverage": truth_coverage,
        "tracker_diagnostics": tracker_diagnostics,
        "role_model_provenance": role_provenance,
        "retention_window_evidence": retention,
        "dependency_fingerprints": dependency_fingerprints,
        "input_fingerprints": before,
    }
    if controlled_positive:
        hazard_class = str(trajectory_row["hazard_class"])
        helper_target = _target_source_track(
            "helper",
            role_dirs["helper"],
            tracks_by_role["helper"],
            hazard_class=hazard_class,
            target_role_prefix=target_role_prefix,
            center_gate_m=evaluation_center_gate_m,
        )
        recipient_target = _target_source_track(
            "recipient",
            role_dirs["recipient"],
            tracks_by_role["recipient"],
            hazard_class=hazard_class,
            target_role_prefix=target_role_prefix,
            center_gate_m=evaluation_center_gate_m,
        )
        recipient_target_truth = pd.read_csv(
            _single(role_dirs["recipient"] / "evaluation_truth", "*_ground_truth.csv")
        )
        map_target_matches: list[dict[str, Any]] = []
        for item in availability["install_attempts"]:
            if (
                item["install_status"] != "accepted"
                or item["source_role"] != "helper"
                or item["source_track_id"] != helper_target
            ):
                continue
            ref = ("helper_install_attempt", str(item["attempt_id"]))
            target_truth = _registered_target_truth_state(
                recipient_target_truth,
                frame_id=frame_by_install_ref[ref],
                available_at_s=float(item["available_at_s"]),
                target_role_prefix=target_role_prefix,
            )
            map_target_matches.append(
                build_recipient_map_target_match(
                    trajectory_id=trajectory_id,
                    install_kind=ref[0],
                    install_ref_id=ref[1],
                    source_role="helper",
                    source_track_id=str(item["source_track_id"]),
                    recipient_map_track_id=str(item["recipient_map_track_id"]),
                    available_at_s=float(item["available_at_s"]),
                    canonical_map_state=map_state_by_install_ref[ref],
                    target_truth_state=target_truth,
                    center_gate_m=evaluation_center_gate_m,
                )
            )
        for item in availability["recipient_local_installs"]:
            if item["source_track_id"] != recipient_target:
                continue
            ref = ("recipient_local_install", str(item["local_install_id"]))
            target_truth = _registered_target_truth_state(
                recipient_target_truth,
                frame_id=frame_by_install_ref[ref],
                available_at_s=float(item["available_at_s"]),
                target_role_prefix=target_role_prefix,
            )
            map_target_matches.append(
                build_recipient_map_target_match(
                    trajectory_id=trajectory_id,
                    install_kind=ref[0],
                    install_ref_id=ref[1],
                    source_role="recipient",
                    source_track_id=str(item["source_track_id"]),
                    recipient_map_track_id=str(item["recipient_map_track_id"]),
                    available_at_s=float(item["available_at_s"]),
                    canonical_map_state=map_state_by_install_ref[ref],
                    target_truth_state=target_truth,
                    center_gate_m=evaluation_center_gate_m,
                )
            )
        helper_map = next(
            (
                item["recipient_map_track_id"]
                for item in sorted(
                    map_target_matches, key=lambda value: value["available_at_s"]
                )
                if item["install_kind"] == "helper_install_attempt"
                and item["usable_target_match"]
            ),
            None,
        )
        horizon = max(float(recipient_times.loc[frame]) for frame in common_frames)
        result["evaluation_recipient_map_target_matches"] = map_target_matches
        result["installed_track_endpoint"] = build_recipient_available_endpoint(
            availability,
            helper_source_track_id=helper_target,
            recipient_source_track_id=recipient_target,
            recipient_map_track_id=helper_map,
            evaluation_horizon_s=horizon,
            evaluation_recipient_map_target_matches=map_target_matches,
        )

    after = {str(path): _sha256_file(path) for path in input_paths}
    if before != after:
        raise RuntimeError("postflight inputs changed during analysis")
    result["collector_artifact_manifests"] = {
        role: _verify_role_artifact_manifest(role_dir)
        for role, role_dir in role_dirs.items()
    }
    result["postflight_sha256"] = canonical_sha256(result)
    return result


def _write_json_create_only_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish JSON without ever replacing an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON mapping: {path}")
    return dict(value)


def _structural_capture_evidence(
    trajectory_dir: Path,
    batch_record: Mapping[str, Any],
    collision_relevance_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the capture gates and their traffic-sanity source artifacts."""

    trajectory_id = trajectory_dir.name
    traffic = batch_record.get("traffic_sanity")
    verification = batch_record.get("trajectory_verification")
    if not isinstance(traffic, Mapping) or traffic.get("pass") is not True:
        raise ValueError(f"traffic sanity did not pass: {trajectory_id}")
    if int(traffic.get("collision_events", -1)) != 0:
        raise ValueError(f"traffic collision gate did not pass: {trajectory_id}")
    if not isinstance(verification, Mapping) or verification.get("pass") is not True:
        raise ValueError(f"trajectory verification did not pass: {trajectory_id}")
    gate_names = (
        "matched_pair_initial_realization_gate",
        "matched_pair_owned_nontreatment_gate",
        "matched_pair_static_environment_gate",
        "matched_pair_full_trajectory_gate",
    )
    gates: dict[str, Any] = {}
    for name in gate_names:
        gate = batch_record.get(name)
        if not isinstance(gate, Mapping) or gate.get("pass") is not True:
            raise ValueError(f"matched-pair structural gate did not pass: {name}: {trajectory_id}")
        gates[name] = dict(gate)

    traffic_dir = trajectory_dir / "traffic_sanity"
    files: dict[str, dict[str, Any]] = {}
    for name in (
        "traffic_sanity_summary.json",
        "npc_collision_events.csv",
        "npc_trajectories.csv",
        "ambient_actor_trajectories.csv",
    ):
        path = traffic_dir / name
        if not path.is_file():
            raise ValueError(f"traffic-sanity artifact is missing: {path}")
        files[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    summary = _load_json_mapping(
        traffic_dir / "traffic_sanity_summary.json", "traffic sanity summary"
    )
    if summary != dict(traffic):
        raise ValueError(f"traffic sanity summary differs from batch record: {trajectory_id}")
    result = {
        "schema": "scenesense.phase2_factor_structural_capture_evidence.v1",
        "trajectory_id": trajectory_id,
        "traffic_sanity": dict(traffic),
        "trajectory_verification": dict(verification),
        "matched_pair_gates": gates,
        "collision_relevance_contract": dict(collision_relevance_contract),
        "traffic_sanity_artifacts": files,
        "structural_capture_pass": True,
    }
    result["structural_capture_sha256"] = canonical_sha256(result)
    return result


def _retention_window_evidence(
    trajectory_dir: Path,
    *,
    realized_onset_s: float | None,
) -> dict[str, Any]:
    trace_path = trajectory_dir / "scenario/realized_trace.csv"
    trace = pd.read_csv(trace_path)
    required_trace_columns = {"frame_id", "elapsed_s"}
    if not required_trace_columns.issubset(trace.columns):
        raise ValueError("realized trace lacks frame_id/elapsed_s")
    trace_frame_ids = trace["frame_id"].astype(int)
    if trace_frame_ids.duplicated().any():
        raise ValueError("realized trace contains duplicate frame IDs")
    trace_by_frame = trace.set_index(trace_frame_ids, drop=False)
    roles: dict[str, Any] = {}
    frame_sets: dict[str, list[int]] = {}
    relative_times: dict[str, list[float]] = {}
    for role in ("helper", "recipient"):
        role_dir = trajectory_dir / role
        path_by_frame = {
            int(path.name.split("_")[1]): path
            for path in (role_dir / "retained_inputs").glob("frame_*_inputs.npz")
        }
        frames = sorted(path_by_frame)
        if len(frames) != 40 or len(set(frames)) != 40:
            raise ValueError(f"{role} must retain exactly 40 unique input frames")
        metrics_path = _single(role_dir / "streams", "*_metrics.csv")
        metrics = pd.read_csv(metrics_path)
        by_frame = metrics.set_index(metrics["frame_id"].astype(int), drop=False)
        if any(frame not in by_frame.index for frame in frames):
            raise ValueError(f"{role} retained frame lacks a metrics timestamp")
        timestamps = [float(by_frame.loc[frame]["carla_timestamp"]) for frame in frames]
        if any(
            abs((right - left) - 0.1) > 1e-6
            for left, right in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(f"{role} retained inputs are not an exact 10 Hz window")
        if any(frame not in trace_by_frame.index for frame in frames):
            raise ValueError(f"{role} retained frame lacks realized-trace elapsed time")
        rel = [float(trace_by_frame.loc[frame]["elapsed_s"]) for frame in frames]
        if any(
            abs((right - left) - 0.1) > 1e-6
            for left, right in zip(rel, rel[1:])
        ):
            raise ValueError(f"{role} realized-trace retention cadence is not 10 Hz")
        frame_sets[role] = frames
        relative_times[role] = rel
        roles[role] = {
            "retained_input_frame_count": len(frames),
            "retained_input_frame_ids": frames,
            "first_carla_timestamp_s": timestamps[0],
            "last_carla_timestamp_s": timestamps[-1],
            "first_episode_relative_s": rel[0],
            "last_episode_relative_s": rel[-1],
            "measured_window_span_s": timestamps[-1] - timestamps[0],
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": _sha256_file(metrics_path),
        }
    if frame_sets["helper"] != frame_sets["recipient"]:
        raise ValueError("helper/recipient retained input frame IDs differ")
    maximum_pair_error = max(
        abs(left - right)
        for left, right in zip(relative_times["helper"], relative_times["recipient"])
    )
    if maximum_pair_error > 1e-9:
        raise ValueError("helper/recipient retained input timestamps differ")
    result: dict[str, Any] = {
        "schema": "scenesense.phase2_factor_retention_window_evidence.v1",
        "trajectory_id": trajectory_dir.name,
        "roles": roles,
        "exact_aligned_40_input_frames_at_10_hz": True,
        "maximum_pair_timestamp_error_s": maximum_pair_error,
        "elapsed_time_basis": "scenario_realized_trace_frame_id_join",
        "realized_trace_path": str(trace_path.resolve()),
        "realized_trace_sha256": _sha256_file(trace_path),
        "realized_onset_status": (
            "not_applicable_matched_benign"
            if realized_onset_s is None
            else "measured_positive"
        ),
        "guarantee_semantics": (
            "nominal_3s_after_authored_command_and_at_least_2p8s_after_realized_onset"
        ),
    }
    if realized_onset_s is not None:
        onset = float(realized_onset_s)
        if not math.isfinite(onset):
            raise ValueError("realized onset must be finite")
        first = relative_times["recipient"][0]
        last = relative_times["recipient"][-1]
        pre = onset - first
        post = last - onset
        if pre < -1e-9 or post < 2.8 - 1e-9:
            raise ValueError(
                "retained inputs do not cover realized onset through at least 2.8 s"
            )
        result.update(
            {
                "realized_hazard_onset_s": onset,
                "measured_pre_realized_onset_span_s": pre,
                "measured_post_realized_onset_span_s": post,
                "minimum_required_post_realized_onset_span_s": 2.8,
            }
        )
    result["retention_evidence_sha256"] = canonical_sha256(result)
    return result


def analyze_and_persist_trajectory_artifacts(
    *,
    trajectory_dir: Path,
    trajectory_row: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail early on one real row and publish its immutable postflight."""

    result = analyze_trajectory_artifacts(
        trajectory_dir=trajectory_dir,
        trajectory_row=trajectory_row,
        smoke_config=smoke_config,
    )
    path = Path(trajectory_dir) / "scenario/factor_smoke_postflight.json"
    _write_json_create_only_atomic(path, result)
    return result


def analyze_batch_artifacts(
    *,
    batch_root: Path,
    smoke_config: Mapping[str, Any],
    factor_plan: Mapping[str, Any],
    write_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and atomically validate the exact-16 bundle from real artifacts.

    This function cannot produce a PASS from capture-only sentinels: it
    requires all sixteen completed batch records, every factor-realization
    artifact, actual v3 tracker/map replay, exact-loader audit, and both
    recipient-consumer endpoint chains.  No partial subset is admitted.
    """

    from data_collection.validate_phase2_factor_realization_smoke import (
        RESULT_SCHEMA,
        validate_results,
    )

    root = Path(batch_root).resolve()
    if not root.is_dir():
        raise ValueError(f"factor batch root is missing: {root}")
    rows = factor_plan.get("rows")
    if (
        not isinstance(rows, list)
        or int(factor_plan.get("trajectory_count", -1)) != 16
        or len(rows) != 16
        or int(factor_plan.get("group_count", -1)) != 8
    ):
        raise ValueError("factor postflight requires the exact 16-row/8-group plan")
    if factor_plan.get("stage_id") != smoke_config.get("stage_id"):
        raise ValueError("factor plan and smoke stage IDs differ")
    batch = _load_json_mapping(root / "batch_manifest.json", "raw batch manifest")
    resolved_audit = _load_yaml(root / "resolved_config.yaml")
    collision_relevance_contract = {
        "minimum_static_collision_horizontal_impulse": float(
            resolved_audit["ambient_traffic"]["traffic_sanity_gate"]
            ["minimum_static_collision_horizontal_impulse"]
        ),
        "same_pair_incident_separation_frames_strictly_greater_than": 10,
        "actor_to_actor_contact_rule": "other_actor_id_gt_zero",
        "static_contact_rule": "static_type_and_minimum_horizontal_impulse",
    }
    batch_rows = batch.get("trajectories")
    if not isinstance(batch_rows, list) or len(batch_rows) != 16:
        raise ValueError("raw batch manifest does not contain exactly 16 trajectories")
    batch_by_id = {str(item.get("trajectory_id")): item for item in batch_rows}
    if len(batch_by_id) != 16 or any(
        item.get("status") != "complete" for item in batch_rows
    ):
        raise ValueError("all exact-16 raw trajectory records must be complete")
    plan_by_id = {str(item["trajectory_id"]): item for item in rows}
    if set(batch_by_id) != set(plan_by_id):
        raise ValueError("raw batch trajectory IDs differ from the exact factor plan")

    result_rows: list[dict[str, Any]] = []
    audits: list[Mapping[str, Any]] = []
    guardrails: list[Mapping[str, Any]] = []
    dependency_sets: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        trajectory_dir = root / trajectory_id
        factor_path = trajectory_dir / "scenario/factor_realization.json"
        factor = _load_json_mapping(factor_path, "factor realization")
        for field, expected in (
            ("trajectory_id", trajectory_id),
            ("trajectory_row_sha256", row["trajectory_row_sha256"]),
            ("scenario_role", row["scenario_role"]),
        ):
            if factor.get(field) != expected:
                raise ValueError(f"factor artifact {field} drifted: {trajectory_id}")
        if factor.get("requested_factors") != row["requested_factor_contract"]:
            raise ValueError(f"factor requested controls drifted: {trajectory_id}")
        context_sha = str(factor.get("nontreatment_plan_sha256", ""))
        if len(context_sha) != 64:
            raise ValueError(f"factor non-treatment plan hash is invalid: {trajectory_id}")
        recomputed_postflight = analyze_trajectory_artifacts(
            trajectory_dir=trajectory_dir,
            trajectory_row=row,
            smoke_config=smoke_config,
        )
        postflight_path = trajectory_dir / "scenario/factor_smoke_postflight.json"
        if postflight_path.is_file():
            postflight = _load_json_mapping(
                postflight_path, "per-trajectory factor postflight"
            )
            if postflight != recomputed_postflight:
                raise ValueError(
                    f"per-trajectory postflight is not byte-content reproducible: {trajectory_id}"
                )
        else:
            postflight = recomputed_postflight
            if write_outputs:
                _write_json_create_only_atomic(postflight_path, postflight)
        retention = postflight["retention_window_evidence"]
        structural_capture = _structural_capture_evidence(
            trajectory_dir,
            batch_by_id[trajectory_id],
            collision_relevance_contract,
        )
        evidence_manifest = {
            "factor_realization_path": str(factor_path),
            "factor_realization_sha256": _sha256_file(factor_path),
            "postflight_sha256": postflight["postflight_sha256"],
            "postflight_artifact_path": str(postflight_path.resolve()),
            "postflight_artifact_sha256": (
                _sha256_file(postflight_path) if postflight_path.is_file() else None
            ),
            "input_fingerprints": postflight["input_fingerprints"],
            "dependency_fingerprints": postflight["dependency_fingerprints"],
            "collector_artifact_manifests": postflight[
                "collector_artifact_manifests"
            ],
            "retention_evidence_sha256": retention[
                "retention_evidence_sha256"
            ],
            "structural_capture_sha256": structural_capture[
                "structural_capture_sha256"
            ],
            "batch_trajectory_record_sha256": canonical_sha256(
                batch_by_id[trajectory_id]
            ),
        }
        result_row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "trajectory_row_sha256": row["trajectory_row_sha256"],
            "artifact_manifest_sha256": canonical_sha256(evidence_manifest),
            "artifact_evidence": evidence_manifest,
            "group_id": row["group_id"],
            "scenario_role": row["scenario_role"],
            "nontreatment_plan_sha256": context_sha,
            "requested_factors": factor["requested_factors"],
            "causal_policy_audit": postflight["causal_policy_audit"],
            "recipient_availability_provenance": postflight[
                "recipient_availability_provenance"
            ],
            "installed_track_guardrails": postflight[
                "installed_track_guardrails"
            ],
            "evaluation_truth_match_by_attempt_id": postflight[
                "evaluation_truth_match_by_attempt_id"
            ],
            "evaluation_truth_coverage": postflight["evaluation_truth_coverage"],
            "capture_model_identity": postflight["role_model_provenance"],
            "retention_window_evidence": retention,
            "structural_capture_gates": structural_capture,
            "transport_scope": postflight["transport_scope"],
            "warnings_generated": postflight["warnings_generated"],
        }
        if bool(row["controlled_hazard_present"]):
            gate = factor.get("factor_realization_gate")
            if not isinstance(gate, Mapping) or gate.get("pass") is not True:
                raise ValueError(f"positive factor gate did not pass: {trajectory_id}")
            if not isinstance(factor.get("realized_factors"), Mapping):
                raise ValueError(f"positive realized factors are missing: {trajectory_id}")
            result_row["realized_factors"] = factor["realized_factors"]
            result_row["evaluation_recipient_map_target_matches"] = postflight[
                "evaluation_recipient_map_target_matches"
            ]
            result_row["installed_track_endpoint"] = postflight[
                "installed_track_endpoint"
            ]
        else:
            if factor.get("registered_target_absent") is not True:
                raise ValueError(f"benign target absence is not explicit: {trajectory_id}")
            result_row.update(
                {
                    "registered_target_absent": True,
                    "realized_factors_status": factor["realized_factors_status"],
                    "factor_reference_trajectory_id": factor[
                        "factor_reference_trajectory_id"
                    ],
                }
            )
        result_rows.append(result_row)
        audits.append(postflight["causal_policy_audit"])
        guardrails.append(postflight["installed_track_guardrails"])
        dependency_sets[canonical_sha256(postflight["dependency_fingerprints"])] = (
            postflight["dependency_fingerprints"]
        )
    if len(dependency_sets) != 1:
        raise ValueError("postflight dependencies changed within the exact tranche")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "stage_id": smoke_config["stage_id"],
        "source_manifest_sha256": factor_plan["source_manifest_sha256"],
        "plan_sha256": factor_plan["plan_sha256"],
        "policy_feature_contract_sha256": factor_plan[
            "policy_feature_contract_sha256"
        ],
        "atomic_exact_trajectory_count": 16,
        "partial_admission": False,
        "warnings_actuated": False,
        "oai_executed": False,
        "downstream_stage_chained": False,
        "policy_action_selected_from_features": False,
        "policy_performance_evaluated": False,
        "observed_policy_state_complete": False,
        "policy_feature_projection": summarize_policy_audits(audits),
        "installed_track_quality_guardrails": aggregate_guardrail_reports(
            guardrails
        ),
        "dependency_fingerprints": dict(next(iter(dependency_sets.values()))),
        "batch_input_evidence": {
            "batch_manifest_prepostflight": batch,
            "batch_manifest_prepostflight_sha256": canonical_sha256(batch),
            "raw_plan_path": str((root / "plan.json").resolve()),
            "raw_plan_sha256": _sha256_file(root / "plan.json"),
            "resolved_config_path": str((root / "resolved_config.yaml").resolve()),
            "resolved_config_sha256": _sha256_file(root / "resolved_config.yaml"),
            "factor_plan_sha256": canonical_sha256(factor_plan),
            "collision_relevance_contract": collision_relevance_contract,
        },
        "trajectories": result_rows,
    }
    result["result_bundle_sha256"] = canonical_sha256(result)
    validation = validate_results(result, smoke_config, factor_plan)
    if validation.get("verdict") != "PASS_ATOMIC_EXACT_16_ADMITTED":
        raise RuntimeError("factor postflight validator did not return atomic PASS")
    if write_outputs:
        _write_json_create_only_atomic(root / BATCH_RESULT_FILENAME, result)
        _write_json_create_only_atomic(root / BATCH_VALIDATION_FILENAME, validation)
    return result, validation
