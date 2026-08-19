"""Apply the preregistered decision-opportunity gates to one immutable pilot.

This is an offline, pilot-only decision stage.  It replays the frozen v3 source
tracker and recipient map from captured lightweight detections; it does not
decode retained logits, launch CARLA/OAI, run a grid, or chain another stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from phase2_map_sharing.engine_v3 import RecipientMapEngineV3
from phase2_map_sharing.replay_calibration_grid import (
    _adjudicate_warnings,
    _candidate_diagnostics,
    _enrich_arm_metrics,
    _replay_trajectory_setting,
    _single,
    _truth_contexts,
    _verify_capture_complete,
    load_replay_config,
)
from phase2_map_sharing.replay_warning_repair_screen import (
    _replay_source_tracker_v3,
    _screen_setting,
    load_config as load_warning_config,
)
from phase2_map_sharing.source_tracker_v3 import TRACKER_V3_VERSION
from phase2_map_sharing.static_truth_adjudication_v1 import (
    normalize_static_semantic_class_v1,
)


SCHEMA = "scenesense.phase2_decision_opportunity_pilot_analysis.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "phase2_map_sharing/configs/decision_opportunity_pilot_analysis_v1.yaml"
)
OUTPUT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*_decision$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _safe_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} keys drifted")
    return value


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path) -> tuple[dict, dict, dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _exact_keys(
        payload,
        {
            "schema_version", "authorization", "claim_scope", "pilot_contract",
            "frozen_dependencies", "target_detection_gate", "decision_gates",
            "reporting",
        },
        "analysis config",
    )
    if root["schema_version"] != SCHEMA:
        raise ValueError("analysis schema drifted")
    if root["authorization"] != "offline_immutable_pilot_decision_only_no_downstream":
        raise ValueError("analysis authorization drifted")
    if root["claim_scope"] != "three_trajectory_local_carla_pilot_stop_go_not_c2_or_oai_evidence":
        raise ValueError("analysis claim scope drifted")

    contract = _exact_keys(
        root["pilot_contract"],
        {
            "stage_id", "pilot_schema", "completion_status", "batch_status",
            "trajectory_ids", "scenario_roles", "target_role_prefix", "source_roles",
            "cooperative_arms", "all_arms", "world_hz", "frame_count_per_trajectory",
        },
        "pilot_contract",
    )
    expected_ids = [
        "sa_curbside_bus_occluded_pedestrian_low_short_r00_pos",
        "sa_curbside_bus_occluded_pedestrian_low_short_r00_ben",
        "sb_town10hd_opt_signalized_demo_region_r00_natural",
    ]
    expected_roles = [
        "controlled_positive_occlusion",
        "matched_benign_negative",
        "naturalistic_operation",
    ]
    expected_contract = {
        "stage_id": "phase2_decision_opportunity_pilot_v1",
        "pilot_schema": "scenesense.phase2_decision_opportunity_pilot.v1",
        "completion_status": "audit_complete_stop_for_human_gate",
        "batch_status": "audit_capture_and_per_trajectory_verification_complete",
        "trajectory_ids": expected_ids,
        "scenario_roles": expected_roles,
        "target_role_prefix": "phase2_registered_target_",
        "source_roles": ["helper", "recipient"],
        "cooperative_arms": ["send_everything", "hazard_only"],
        "all_arms": ["ego_only", "send_everything", "hazard_only"],
        "world_hz": 10.0,
        "frame_count_per_trajectory": 120,
    }
    if dict(contract) != expected_contract:
        raise ValueError("pilot contract drifted")

    dependencies = _exact_keys(
        root["frozen_dependencies"],
        {
            "decision_contract", "decision_contract_sha256", "warning_repair_config",
            "warning_repair_config_sha256", "replay_truth_config",
            "replay_truth_config_sha256", "setting_id", "tracker_version",
        },
        "frozen_dependencies",
    )
    expected_hashes = {
        "decision_contract_sha256": "1b5f04e1b9e680f6be65c1c789d893f0942bca795a39dee458aa3124063cb234",
        "warning_repair_config_sha256": "b08c7837aab8002477a5f2c47d500a6470391827a15c4dc2b72d2acd37719dd8",
        "replay_truth_config_sha256": "a9307c81885f0840e926aeb0b3d4f98810057400238e7d5b251d925e9d325806",
    }
    for field, expected in expected_hashes.items():
        if dependencies[field] != expected:
            raise ValueError(f"{field} drifted")
    if dependencies["setting_id"] != "c20_a30_t05_u00":
        raise ValueError("setting_id drifted")
    if dependencies["tracker_version"] != TRACKER_V3_VERSION:
        raise ValueError("tracker version drifted")
    for path_field, hash_field in (
        ("decision_contract", "decision_contract_sha256"),
        ("warning_repair_config", "warning_repair_config_sha256"),
        ("replay_truth_config", "replay_truth_config_sha256"),
    ):
        dependency_path = _repo_path(dependencies[path_field])
        if _sha256(dependency_path) != dependencies[hash_field]:
            raise ValueError(f"frozen dependency hash drifted: {path_field}")

    detection = _exact_keys(
        root["target_detection_gate"],
        {"normalized_class", "score_floor", "actor_origin_center_gate_m", "minimum_consecutive_helper_frames"},
        "target_detection_gate",
    )
    if dict(detection) != {
        "normalized_class": "pedestrian", "score_floor": 0.05,
        "actor_origin_center_gate_m": 5.0, "minimum_consecutive_helper_frames": 5,
    }:
        raise ValueError("target detection gate drifted")
    gates = _exact_keys(
        root["decision_gates"],
        {
            "minimum_recipient_confirmation_delay_s", "minimum_helper_warning_lead_vs_ego_s",
            "minimum_recipient_speed_at_helper_warning_mps",
            "require_helper_warning_before_first_hidden_actor_yield",
            "require_zero_registered_target_misses_all_arms",
            "maximum_benign_false_warning_active_frame_rate",
            "maximum_cooperative_excess_false_warning_active_frame_rate",
        },
        "decision_gates",
    )
    if dict(gates) != {
        "minimum_recipient_confirmation_delay_s": 1.0,
        "minimum_helper_warning_lead_vs_ego_s": 0.5,
        "minimum_recipient_speed_at_helper_warning_mps": 2.0,
        "require_helper_warning_before_first_hidden_actor_yield": True,
        "require_zero_registered_target_misses_all_arms": True,
        "maximum_benign_false_warning_active_frame_rate": 0.10,
        "maximum_cooperative_excess_false_warning_active_frame_rate": 0.02,
    }:
        raise ValueError("decision gates drifted")
    reporting = _exact_keys(
        root["reporting"],
        {"naturalistic_status", "false_warning_episode_rate_status", "oai_status", "downstream_status"},
        "reporting",
    )
    if dict(reporting) != {
        "naturalistic_status": "report_only",
        "false_warning_episode_rate_status": "report_only_insufficient_short_exposure",
        "oai_status": "not_run_not_authorized",
        "downstream_status": "not_authorized",
    }:
        raise ValueError("reporting contract drifted")

    warning_config = load_warning_config(_repo_path(dependencies["warning_repair_config"]))
    replay_config = load_replay_config(_repo_path(dependencies["replay_truth_config"]))
    setting = _screen_setting(warning_config, replay_config)
    if setting["setting_id"] != dependencies["setting_id"]:
        raise ValueError("frozen c20/a30/t05/u00 setting drifted")
    if warning_config["source_tracker"]["algorithm"] != TRACKER_V3_VERSION:
        raise ValueError("frozen v3 tracker drifted")
    return dict(root), warning_config, replay_config


def _validate_output(batch_root: Path, output_dir: Path) -> None:
    source = batch_root.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == source or output.is_relative_to(source):
        raise ValueError("analysis output must be a create-only capture sibling")
    if not OUTPUT_PATTERN.fullmatch(output.name):
        raise ValueError("analysis output basename must be safe and end in _decision")


def _validate_batch(batch_root: Path, config: Mapping[str, object]) -> tuple[dict, dict, dict, dict]:
    completed, manifest, plan = _verify_capture_complete(batch_root)
    contract = config["pilot_contract"]
    resolved = yaml.safe_load((batch_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(resolved, Mapping):
        raise ValueError("capture resolved config is invalid")
    if completed.get("status") != contract["completion_status"]:
        raise ValueError("pilot completion status drifted")
    if manifest.get("status") != contract["batch_status"]:
        raise ValueError("pilot batch status drifted")
    if manifest.get("stage_id") != contract["stage_id"] or resolved.get("stage_id") != contract["stage_id"]:
        raise ValueError("pilot stage_id drifted")
    ids = [str(row["trajectory_id"]) for row in plan["trajectories"]]
    roles = [str(row["scenario_role"]) for row in plan["trajectories"]]
    manifest_ids = [str(row["trajectory_id"]) for row in manifest["trajectories"]]
    if ids != contract["trajectory_ids"] or manifest_ids != ids:
        raise ValueError("pilot trajectory IDs or order drifted")
    if roles != contract["scenario_roles"]:
        raise ValueError("pilot scenario roles or order drifted")
    for row in manifest["trajectories"]:
        if row.get("status") != "complete" or row.get("trajectory_verification", {}).get("pass") is not True:
            raise ValueError(f"trajectory is not capture-complete: {row['trajectory_id']}")
        if int(row.get("captured_frame_count", -1)) != int(contract["frame_count_per_trajectory"]):
            raise ValueError("trajectory frame count drifted")
        static = row.get("static_environment_truth")
        if not isinstance(static, Mapping) or static.get("status") != "complete":
            raise ValueError("static environment truth is not complete")
    if float(resolved.get("clock", {}).get("world_hz", -1)) != float(contract["world_hz"]):
        raise ValueError("pilot world cadence drifted")
    if int(resolved.get("clock", {}).get("frames_per_trajectory", -1)) != int(contract["frame_count_per_trajectory"]):
        raise ValueError("pilot resolved frame count drifted")
    if resolved.get("capture", {}).get("warnings_actuated") is not False:
        raise ValueError("pilot warnings unexpectedly actuated")
    if resolved.get("static_environment_truth", {}).get("enabled") is not True:
        raise ValueError("pilot static truth declaration drifted")
    provenance = resolved.get("pilot_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema") != contract["pilot_schema"]:
        raise ValueError("pilot provenance schema drifted")
    if provenance.get("trajectory_ids") != contract["trajectory_ids"]:
        raise ValueError("pilot provenance trajectory IDs drifted")
    if provenance.get("decision_contract_sha256") != config["frozen_dependencies"]["decision_contract_sha256"]:
        raise ValueError("captured decision contract hash drifted")
    if provenance.get("visual_acceptance", {}).get("status") != "accepted":
        raise ValueError("pilot visual acceptance is absent")
    if provenance.get("no_oai_or_downstream_chaining") is not True:
        raise ValueError("pilot downstream authorization drifted")
    return completed, manifest, plan, dict(resolved)


def _verify_role_inputs(batch_root: Path, plan: Mapping[str, object]) -> list[Path]:
    paths: list[Path] = []
    for trajectory in plan["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in ("helper", "recipient"):
            role_dir = batch_root / trajectory_id / role
            manifest_path = role_dir / "artifact_manifest.json"
            artifact_manifest = _load_json(manifest_path)
            entries = {str(row["path"]): row for row in artifact_manifest.get("files", [])}
            required = [
                role_dir / "runtime/final_detections.csv",
                role_dir / "runtime/ego_states.csv",
                _single(role_dir / "streams", "*_metrics.csv"),
                _single(role_dir / "evaluation_truth", "*_ground_truth.csv"),
                _single(role_dir / "manifests", "*_manifest.json"),
                _single(role_dir / "manifests", "*_resolved_config.json"),
            ]
            for candidate in required:
                relative = str(candidate.relative_to(role_dir))
                entry = entries.get(relative)
                if entry is None:
                    raise ValueError(f"consumed role artifact is unmanifested: {candidate}")
                if int(entry["bytes"]) != candidate.stat().st_size or str(entry["sha256"]) != _sha256(candidate):
                    raise ValueError(f"consumed role artifact hash drifted: {candidate}")
            paths.extend([manifest_path, *required])
        paths.extend(
            [
                batch_root / trajectory_id / "scenario/realized_trace.csv",
                batch_root / trajectory_id / "scenario/realization_summary.json",
                batch_root / trajectory_id / "static_environment_truth/artifact_manifest.json",
                batch_root / trajectory_id / "static_environment_truth/static_environment_objects.csv",
                batch_root / trajectory_id / "static_environment_truth/static_environment_snapshot.json",
            ]
        )
    return paths


def _fingerprints(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"consumed input must be a regular file: {path}")
        result[str(path)] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _scenario_trace(trajectory_dir: Path, frame_count: int) -> pd.DataFrame:
    trace = pd.read_csv(trajectory_dir / "scenario/realized_trace.csv").sort_values("frame_id")
    required = {
        "frame_id", "elapsed_s", "recipient_speed_mps", "recipient_direct_route_yield_active",
        "recipient_direct_route_yield_actor_id", "recipient_direct_route_yield_actor_type",
    }
    if missing := required - set(trace.columns):
        raise ValueError(f"scenario trace lacks pilot gate fields: {sorted(missing)}")
    if len(trace) != frame_count or trace["frame_id"].duplicated().any():
        raise ValueError("scenario trace frame coverage drifted")
    return trace


def _target_truth(role_dir: Path, prefix: str, expected_frames: Sequence[int]) -> pd.DataFrame:
    truth = pd.read_csv(_single(role_dir / "evaluation_truth", "*_ground_truth.csv"))
    target = truth[truth["role_name"].astype(str).str.startswith(prefix)].copy()
    if target.empty or target["actor_id"].astype(str).nunique() != 1:
        raise ValueError(f"positive role lacks exactly one registered target: {role_dir}")
    if target["frame_id"].duplicated().any() or target["frame_id"].astype(int).tolist() != list(expected_frames):
        raise ValueError("registered target truth does not cover every frame exactly once")
    target["normalized_class"] = target["class_name"].map(normalize_static_semantic_class_v1)
    if not target["normalized_class"].eq("pedestrian").all():
        raise ValueError("registered target is not a pedestrian")
    return target


def _consecutive_runs(evidence: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    start = None
    previous = None
    for position, row in enumerate(evidence.itertuples(index=False)):
        if int(row.qualified) == 1:
            if start is None:
                start = position
            previous = position
        elif start is not None:
            section = evidence.iloc[start : previous + 1]
            rows.append(_run_row(section))
            start = previous = None
    if start is not None:
        rows.append(_run_row(evidence.iloc[start : previous + 1]))
    return rows


def _run_row(section: pd.DataFrame) -> dict:
    return {
        "source_role": str(section.iloc[0]["source_role"]),
        "start_frame_id": int(section.iloc[0]["frame_id"]),
        "end_frame_id": int(section.iloc[-1]["frame_id"]),
        "start_carla_timestamp_s": float(section.iloc[0]["carla_timestamp"]),
        "end_carla_timestamp_s": float(section.iloc[-1]["carla_timestamp"]),
        "frame_count": len(section),
        "minimum_score": float(pd.to_numeric(section["score"]).min()),
        "maximum_actor_origin_error_m": float(pd.to_numeric(section["actor_origin_error_m"]).max()),
    }


def _raw_target_evidence(
    role_dir: Path,
    role: str,
    target: pd.DataFrame,
    detection_gate: Mapping[str, object],
) -> tuple[pd.DataFrame, list[dict]]:
    metrics = pd.read_csv(_single(role_dir / "streams", "*_metrics.csv")).sort_values("frame_id")
    if metrics["frame_id"].duplicated().any():
        raise ValueError("source metrics contain duplicate frame IDs")
    timestamps = pd.to_numeric(metrics["carla_timestamp"]).to_numpy(dtype=float)
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 0.1, rtol=0.0, atol=1e-6):
        raise ValueError("source stream is not a consecutive 10 Hz clock")
    detections = pd.read_csv(role_dir / "runtime/final_detections.csv")
    target_by_frame = target.set_index("frame_id")
    rows: list[dict] = []
    for state in metrics.itertuples(index=False):
        frame_id = int(state.frame_id)
        truth = target_by_frame.loc[frame_id]
        candidates = detections[detections["frame_id"].astype(int) == frame_id].copy()
        candidates["normalized_class"] = candidates["class_name"].map(normalize_static_semantic_class_v1)
        candidates = candidates[
            candidates["normalized_class"].eq(detection_gate["normalized_class"])
            & (pd.to_numeric(candidates["score"]) >= float(detection_gate["score_floor"]) - 1e-12)
        ].copy()
        candidates["actor_origin_error_m"] = np.hypot(
            pd.to_numeric(candidates["world_x"]) - float(truth["origin_x"]),
            pd.to_numeric(candidates["world_y"]) - float(truth["origin_y"]),
        )
        candidates = candidates[
            candidates["actor_origin_error_m"] <= float(detection_gate["actor_origin_center_gate_m"]) + 1e-12
        ].sort_values(["actor_origin_error_m", "score", "detection_index"], ascending=[True, False, True])
        selected = candidates.iloc[0] if not candidates.empty else None
        rows.append(
            {
                "source_role": role,
                "frame_id": frame_id,
                "carla_timestamp": float(state.carla_timestamp),
                "target_actor_id": str(truth["actor_id"]),
                "qualified": int(selected is not None),
                "detection_index": None if selected is None else int(selected["detection_index"]),
                "class_name": None if selected is None else str(selected["class_name"]),
                "score": None if selected is None else float(selected["score"]),
                "actor_origin_error_m": None if selected is None else float(selected["actor_origin_error_m"]),
            }
        )
    evidence = pd.DataFrame(rows)
    return evidence, _consecutive_runs(evidence)


def _target_track_evidence(
    role: str,
    tracks_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    target: pd.DataFrame,
    trace: pd.DataFrame,
    gate_m: float,
) -> pd.DataFrame:
    target_by_frame = target.set_index("frame_id")
    elapsed_by_frame = trace.set_index("frame_id")["elapsed_s"].astype(float).to_dict()
    rows: list[dict] = []
    for frame_id in target["frame_id"].astype(int):
        truth = target_by_frame.loc[frame_id]
        candidates = []
        for track in tracks_by_frame.get(frame_id, ()):
            if normalize_static_semantic_class_v1(track["class_name"]) != "pedestrian":
                continue
            distance = math.hypot(
                float(track["world_x"]) - float(truth["origin_x"]),
                float(track["world_y"]) - float(truth["origin_y"]),
            )
            if distance <= gate_m + 1e-12:
                candidates.append((distance, str(track["source_track_id"]), track))
        if not candidates:
            continue
        distance, _, track = min(candidates, key=lambda item: (item[0], item[1]))
        rows.append(
            {
                "source_role": role,
                "frame_id": frame_id,
                "carla_timestamp": float(truth["carla_timestamp"]),
                "elapsed_s": float(elapsed_by_frame[frame_id]),
                "target_actor_id": str(truth["actor_id"]),
                "source_track_id": str(track["source_track_id"]),
                "confirmed_at_frame_id": int(track["confirmed_at_frame_id"]),
                "confirmation_hits": int(track["confirmation_hits"]),
                "consecutive_hits": int(track["consecutive_hits"]),
                "score": float(track["score"]),
                "actor_origin_error_m": float(distance),
            }
        )
    return pd.DataFrame(rows)


def _active_evidence_pairs(row: Mapping[str, object]) -> list[tuple[str, str]]:
    sources = json.loads(str(row["evidence_sources"]))
    track_ids = json.loads(str(row["evidence_track_ids"]))
    if not isinstance(sources, list) or not isinstance(track_ids, list) or len(sources) != len(track_ids):
        raise ValueError("warning active evidence arrays are malformed")
    return [(str(source), str(track_id)) for source, track_id in zip(sources, track_ids)]


def _augment_warning_evidence(
    adjudicated: Sequence[Mapping[str, object]], target_tracks: pd.DataFrame
) -> list[dict]:
    # A recipient-map warning can causally retain a source estimate for the
    # frozen 0.5 s TTL even when that source has no new detection on the exact
    # warning frame.  Attribute the active evidence ID to the registered target
    # using only actor-origin matches observed at or before the warning.  Do not
    # require an exact-frame re-observation, and do not use a future match to
    # label an earlier warning.
    target_frames_by_track: dict[tuple[str, str], list[int]] = {}
    for row in target_tracks.itertuples(index=False):
        target_frames_by_track.setdefault(
            (str(row.source_role), str(row.source_track_id)), []
        ).append(int(row.frame_id))
    for frames in target_frames_by_track.values():
        frames.sort()
    rows: list[dict] = []
    for item in adjudicated:
        row = dict(item)
        pairs = _active_evidence_pairs(row)
        helper_active = any(source == "helper" for source, _ in pairs)
        warning_frame = int(row["frame_id"])
        helper_target_frames = [
            matched_frame
            for source, track_id in pairs
            if source == "helper"
            for matched_frame in target_frames_by_track.get((source, track_id), ())
            if matched_frame <= warning_frame
        ]
        helper_target = bool(helper_target_frames)
        row["active_evidence_pairs"] = json.dumps(pairs)
        row["helper_active_evidence"] = int(helper_active)
        row["helper_target_track_active_evidence"] = int(helper_target)
        row["helper_target_last_matched_frame_id"] = (
            max(helper_target_frames) if helper_target_frames else None
        )
        row["helper_target_attribution_basis"] = (
            "same_confirmed_source_track_actor_origin_matched_at_or_before_warning"
            if helper_target
            else "no_prior_or_current_target_match_for_active_helper_track"
        )
        row["helper_evidence_temporal_scope"] = (
            "active_under_map_ttl_not_historical" if helper_active else "no_active_helper"
        )
        rows.append(row)
    return rows


def _first_row(frame: pd.DataFrame, sort: Sequence[str]) -> dict | None:
    if frame.empty:
        return None
    return frame.sort_values(list(sort)).iloc[0].to_dict()


def _evaluate_gates(
    *,
    config: Mapping[str, object],
    helper_runs: Sequence[Mapping[str, object]],
    confirmations: Mapping[str, Mapping[str, object] | None],
    warning_evidence: Mapping[str, Mapping[str, object]],
    target_misses: Mapping[str, int],
    benign_rates: Mapping[str, float],
) -> dict[str, object]:
    detection_gate = config["target_detection_gate"]
    gates = config["decision_gates"]
    cooperative = config["pilot_contract"]["cooperative_arms"]
    helper = confirmations.get("helper")
    recipient = confirmations.get("recipient")
    confirmation_margin = (
        None
        if helper is None or recipient is None
        else float(recipient["carla_timestamp"]) - float(helper["carla_timestamp"])
    )
    raw_gate = max((int(row["frame_count"]) for row in helper_runs), default=0) >= int(
        detection_gate["minimum_consecutive_helper_frames"]
    )
    helper_warning_present = all(warning_evidence[arm]["helper_warning"] is not None for arm in cooperative)
    lead_gate = helper_warning_present and all(
        warning_evidence[arm]["lead_vs_ego_s"] is not None
        and float(warning_evidence[arm]["lead_vs_ego_s"])
        >= float(gates["minimum_helper_warning_lead_vs_ego_s"]) - 1e-12
        for arm in cooperative
    )
    pre_yield = helper_warning_present and all(
        bool(warning_evidence[arm]["before_first_hidden_actor_yield"])
        for arm in cooperative
    )
    speed_gate = helper_warning_present and all(
        warning_evidence[arm]["recipient_speed_mps"] is not None
        and float(warning_evidence[arm]["recipient_speed_mps"])
        >= float(gates["minimum_recipient_speed_at_helper_warning_mps"]) - 1e-12
        for arm in cooperative
    )
    absolute = all(
        float(benign_rates[arm]) <= float(gates["maximum_benign_false_warning_active_frame_rate"]) + 1e-12
        for arm in config["pilot_contract"]["all_arms"]
    )
    excess = all(
        float(benign_rates[arm])
        <= float(benign_rates["ego_only"])
        + float(gates["maximum_cooperative_excess_false_warning_active_frame_rate"])
        + 1e-12
        for arm in cooperative
    )
    results = {
        "five_consecutive_helper_target_detections": bool(raw_gate),
        "confirmed_v3_helper_target_track": helper is not None,
        "recipient_v3_confirmation_at_least_1s_later": (
            confirmation_margin is not None
            and confirmation_margin >= float(gates["minimum_recipient_confirmation_delay_s"]) - 1e-12
        ),
        "helper_derived_truth_positive_warning_present_both_cooperative_arms": bool(helper_warning_present),
        "helper_warning_lead_at_least_0p5s_both_cooperative_arms": bool(lead_gate),
        "helper_warning_before_first_hidden_actor_yield_both_cooperative_arms": bool(pre_yield),
        "recipient_speed_at_least_2mps_at_helper_warning_both_cooperative_arms": bool(speed_gate),
        "zero_registered_target_misses_all_arms": all(int(value) == 0 for value in target_misses.values()),
        "benign_false_warning_active_rate_at_most_10pct_all_arms": bool(absolute),
        "cooperative_benign_excess_at_most_2pp": bool(excess),
        "naturalistic_operation": "REPORT_ONLY",
        "false_warning_episode_rate": "REPORT_ONLY_SHORT_EXPOSURE",
    }
    return results


def _write_manifest(output_dir: Path) -> None:
    path = output_dir / "artifact_manifest.json"
    files = [
        {"path": item.name, "bytes": item.stat().st_size, "sha256": _sha256(item)}
        for item in sorted(output_dir.iterdir())
        if item.is_file() and item != path
    ]
    path.write_text(
        json.dumps({"schema": f"{SCHEMA}.artifacts", "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(batch_root: Path, config_path: Path, output_dir: Path) -> dict:
    batch_root = batch_root.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    _validate_output(batch_root, output_dir)
    config, warning_config, replay_config = load_config(config_path)
    _completed, manifest, plan, resolved = _validate_batch(batch_root, config)
    consumed = _verify_role_inputs(batch_root, plan)
    consumed.extend(
        [
            batch_root / "COMPLETED.json", batch_root / "batch_manifest.json",
            batch_root / "plan.json", batch_root / "resolved_config.yaml", config_path,
            _repo_path(config["frozen_dependencies"]["decision_contract"]),
            _repo_path(config["frozen_dependencies"]["warning_repair_config"]),
            _repo_path(config["frozen_dependencies"]["replay_truth_config"]),
            Path(__file__), REPO_ROOT / "phase2_map_sharing/source_tracker_v3.py",
            REPO_ROOT / "phase2_map_sharing/engine_v3.py",
            REPO_ROOT / "phase2_map_sharing/replay_warning_repair_screen.py",
            REPO_ROOT / "phase2_map_sharing/replay_calibration_grid.py",
            REPO_ROOT / "phase2_map_sharing/static_truth_adjudication_v1.py",
        ]
    )
    before = _fingerprints(consumed)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    contexts = _truth_contexts(batch_root, plan, replay_config)
    if not all("static_catalog" in context for context in contexts.values()):
        raise ValueError("verified static truth is incomplete for the pilot")
    setting = _screen_setting(warning_config, replay_config)
    positive = plan["trajectories"][0]
    positive_id = str(positive["trajectory_id"])
    trace = _scenario_trace(
        batch_root / positive_id,
        int(config["pilot_contract"]["frame_count_per_trajectory"]),
    )
    frame_ids = trace["frame_id"].astype(int).tolist()
    target_by_role: dict[str, pd.DataFrame] = {}
    raw_frames: list[pd.DataFrame] = []
    raw_runs: list[dict] = []
    source_tracks: dict[tuple[str, str], dict[int, list[dict]]] = {}
    tracker_diagnostics: list[dict] = []
    target_track_frames: list[pd.DataFrame] = []
    for trajectory in plan["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in config["pilot_contract"]["source_roles"]:
            role_dir = batch_root / trajectory_id / str(role)
            tracks, diagnostic = _replay_source_tracker_v3(role_dir, str(role), warning_config)
            source_tracks[(trajectory_id, str(role))] = tracks
            tracker_diagnostics.append({"trajectory_id": trajectory_id, **diagnostic})
            if trajectory_id == positive_id:
                target = _target_truth(
                    role_dir,
                    str(config["pilot_contract"]["target_role_prefix"]),
                    frame_ids,
                )
                target_by_role[str(role)] = target
                evidence, runs = _raw_target_evidence(
                    role_dir, str(role), target, config["target_detection_gate"]
                )
                raw_frames.append(evidence)
                raw_runs.extend(runs)
                target_track_frames.append(
                    _target_track_evidence(
                        str(role), tracks, target, trace,
                        float(config["target_detection_gate"]["actor_origin_center_gate_m"]),
                    )
                )

    metrics: list[dict] = []
    warnings: list[dict] = []
    isolation: list[dict] = []
    for trajectory in plan["trajectories"]:
        rows, events, state = _replay_trajectory_setting(
            batch_root, trajectory, setting, source_tracks, replay_config,
            engine_class=RecipientMapEngineV3,
        )
        metrics.extend(rows)
        warnings.extend(events)
        isolation.append(state)
    adjudicated = _adjudicate_warnings(warnings, contexts, replay_config)
    target_tracks = pd.concat(target_track_frames, ignore_index=True)
    adjudicated = _augment_warning_evidence(adjudicated, target_tracks)
    enriched = _enrich_arm_metrics(metrics, adjudicated, contexts, replay_config)
    diagnostics = _candidate_diagnostics(
        enriched, cadence_s=float(replay_config["truth_evaluation"]["cadence_s"])
    )
    metrics_frame = pd.DataFrame(enriched)
    adjudicated_frame = pd.DataFrame(adjudicated)
    diagnostics_frame = pd.DataFrame(diagnostics)

    confirmations: dict[str, dict | None] = {}
    for role in config["pilot_contract"]["source_roles"]:
        role_rows = target_tracks[target_tracks["source_role"].astype(str) == str(role)]
        confirmations[str(role)] = _first_row(role_rows, ("carla_timestamp", "source_track_id"))
    confirmation_margin = (
        None
        if any(confirmations[role] is None for role in ("helper", "recipient"))
        else float(confirmations["recipient"]["carla_timestamp"])
        - float(confirmations["helper"]["carla_timestamp"])
    )

    positive_warnings = adjudicated_frame[
        adjudicated_frame["trajectory_id"].astype(str).eq(positive_id)
        & pd.to_numeric(adjudicated_frame["target_hazard_match_adjudicated"]).eq(1)
        & pd.to_numeric(adjudicated_frame["truth_hazard_positive"]).eq(1)
    ].copy()
    ego_first = _first_row(
        positive_warnings[positive_warnings["arm_id"].astype(str).eq("ego_only")],
        ("warning_at_s", "frame_id", "canonical_track_id"),
    )
    target_actor_id = str(target_by_role["recipient"]["actor_id"].astype(str).iloc[0])
    yields = trace[pd.to_numeric(trace["recipient_direct_route_yield_active"]).fillna(0).astype(int).eq(1)]
    first_yield = _first_row(yields, ("elapsed_s", "frame_id"))
    if first_yield is not None:
        yield_actor = str(int(float(first_yield["recipient_direct_route_yield_actor_id"])))
        if yield_actor != target_actor_id:
            raise ValueError("first hidden-actor yield is not attributed to the registered target")
        realization = _load_json(batch_root / positive_id / "scenario/realization_summary.json")
        summary_yield = realization.get("first_direct_route_yield_by_role", {}).get("recipient")
        if not isinstance(summary_yield, Mapping) or int(summary_yield["frame_id"]) != int(first_yield["frame_id"]):
            raise ValueError("scenario trace and realization summary disagree on first yield")

    trace_by_frame = trace.set_index("frame_id")
    warning_evidence: dict[str, dict] = {}
    for arm in config["pilot_contract"]["cooperative_arms"]:
        arm_rows = positive_warnings[
            positive_warnings["arm_id"].astype(str).eq(str(arm))
            & pd.to_numeric(positive_warnings["helper_target_track_active_evidence"]).eq(1)
        ]
        helper_warning = _first_row(arm_rows, ("warning_at_s", "frame_id", "canonical_track_id"))
        if helper_warning is None:
            warning_evidence[str(arm)] = {
                "helper_warning": None, "lead_vs_ego_s": None,
                "recipient_speed_mps": None, "before_first_hidden_actor_yield": False,
            }
            continue
        scenario_row = trace_by_frame.loc[int(helper_warning["frame_id"])]
        warning_elapsed = float(scenario_row["elapsed_s"])
        helper_warning = {
            **helper_warning,
            "elapsed_s": warning_elapsed,
            "recipient_speed_mps": float(scenario_row["recipient_speed_mps"]),
            "active_vs_historical": "active_under_map_ttl_not_historical",
        }
        warning_evidence[str(arm)] = {
            "helper_warning": helper_warning,
            "lead_vs_ego_s": (
                None if ego_first is None else float(ego_first["warning_at_s"]) - float(helper_warning["warning_at_s"])
            ),
            "recipient_speed_mps": float(scenario_row["recipient_speed_mps"]),
            "before_first_hidden_actor_yield": (
                first_yield is not None and warning_elapsed < float(first_yield["elapsed_s"]) - 1e-12
            ),
        }

    positive_metrics = metrics_frame[metrics_frame["scenario_role"].astype(str).eq("controlled_positive_occlusion")]
    target_misses = {
        str(row.arm_id): int(row.missed_registered_target)
        for row in positive_metrics.itertuples(index=False)
    }
    benign_rates = {
        str(row.arm_id): float(row.suite_a_benign_false_warning_active_frame_rate)
        for row in diagnostics_frame.itertuples(index=False)
    }
    helper_runs = [row for row in raw_runs if row["source_role"] == "helper"]
    gate_results = _evaluate_gates(
        config=config,
        helper_runs=helper_runs,
        confirmations=confirmations,
        warning_evidence=warning_evidence,
        target_misses=target_misses,
        benign_rates=benign_rates,
    )
    scientific_bools = [value for value in gate_results.values() if isinstance(value, bool)]
    scientific_verdict = (
        "PASS_PILOT_GATES_STOP_FOR_HUMAN_DECISION"
        if all(scientific_bools)
        else "FAIL_HOLD_STOP_NO_DOWNSTREAM"
    )

    per_arm_rows = []
    for row in metrics_frame.to_dict("records"):
        group = adjudicated_frame[
            adjudicated_frame["trajectory_id"].astype(str).eq(str(row["trajectory_id"]))
            & adjudicated_frame["arm_id"].astype(str).eq(str(row["arm_id"]))
        ]
        target_group = group[
            pd.to_numeric(group["target_hazard_match_adjudicated"]).eq(1)
            & pd.to_numeric(group["truth_hazard_positive"]).eq(1)
        ]
        helper_group = target_group[pd.to_numeric(target_group["helper_target_track_active_evidence"]).eq(1)]
        per_arm_rows.append(
            {
                **row,
                "target_truth_positive_warning_event_count": len(target_group),
                "target_truth_positive_warning_active_frame_count": int(target_group["frame_id"].nunique()),
                "helper_derived_target_warning_event_count": len(helper_group),
                "helper_derived_target_warning_active_frame_count": int(helper_group["frame_id"].nunique()),
            }
        )
    per_arm_frame = pd.DataFrame(per_arm_rows)
    naturalistic = per_arm_frame[
        per_arm_frame["scenario_role"].astype(str).eq("naturalistic_operation")
    ].copy()
    naturalistic["reporting_status"] = "REPORT_ONLY"

    after = _fingerprints(consumed)
    if before != after:
        raise RuntimeError("consumed source inputs changed during analysis")
    input_hashes_sha256 = hashlib.sha256(
        json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    first_ego_evidence = None
    if ego_first is not None:
        ego_trace = trace_by_frame.loc[int(ego_first["frame_id"])]
        first_ego_evidence = {**ego_first, "elapsed_s": float(ego_trace["elapsed_s"])}

    evidence = {
        "raw_helper_detection_runs": helper_runs,
        "first_target_v3_confirmation_by_role": confirmations,
        "recipient_minus_helper_confirmation_s": confirmation_margin,
        "first_ego_only_truth_positive_warning": first_ego_evidence,
        "cooperative_helper_derived_warning_by_arm": warning_evidence,
        "first_hidden_actor_yield": (
            None
            if first_yield is None
            else {
                "frame_id": int(first_yield["frame_id"]),
                "elapsed_s": float(first_yield["elapsed_s"]),
                "actor_id": target_actor_id,
                "actor_type": str(first_yield["recipient_direct_route_yield_actor_type"]),
                "recipient_speed_mps": float(first_yield["recipient_speed_mps"]),
            }
        ),
        "target_missed_by_arm": target_misses,
        "benign_false_warning_active_frame_rate_by_arm": benign_rates,
        "benign_cooperative_excess_vs_ego_by_arm": {
            arm: float(benign_rates[arm]) - float(benign_rates["ego_only"])
            for arm in config["pilot_contract"]["cooperative_arms"]
        },
    }

    pd.concat(raw_frames, ignore_index=True).to_csv(output_dir / "target_detection_evidence.csv", index=False)
    pd.DataFrame(raw_runs).to_csv(output_dir / "target_detection_runs.csv", index=False)
    target_tracks.to_csv(output_dir / "target_v3_track_evidence.csv", index=False)
    pd.DataFrame(tracker_diagnostics).to_csv(output_dir / "source_tracker_diagnostics.csv", index=False)
    per_arm_frame.to_csv(output_dir / "arm_trajectory_metrics.csv", index=False)
    pd.DataFrame(warnings).to_csv(output_dir / "warning_events.csv", index=False)
    adjudicated_frame.to_csv(output_dir / "adjudicated_warning_events.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    naturalistic.to_csv(output_dir / "naturalistic_report.csv", index=False)
    (output_dir / "arm_state_isolation.json").write_text(
        json.dumps(_safe_json(isolation), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_gate_evidence.json").write_text(
        json.dumps(_safe_json(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema": f"{SCHEMA}.provenance",
        "source_batch": str(batch_root),
        "source_batch_stage_id": manifest["stage_id"],
        "analysis_config": str(config_path),
        "setting_id": setting["setting_id"],
        "source_tracker_version": TRACKER_V3_VERSION,
        "map_engine": "RecipientMapEngineV3",
        "warning_evidence_semantics": "evidence_sources_and_ids_are_active_under_map_ttl_not_historical",
        "truth_usage": "evaluation_only_after_causal_warning_generation",
        "static_truth_verified_all_trajectories": True,
        "retained_logit_decode_run": False,
        "baseline_replay_run": False,
        "grid_run": False,
        "carla_run": False,
        "oai_run": False,
        "downstream_authorized": False,
        "consumed_input_fingerprint_sha256": input_hashes_sha256,
        "consumed_input_fingerprints": before,
        "capture_resolved_config_stage_id": resolved["stage_id"],
    }
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(_safe_json(provenance), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete_stop_no_downstream",
        "technical_verdict": "PASS",
        "scientific_verdict": scientific_verdict,
        "claim_scope": config["claim_scope"],
        "source_batch": str(batch_root),
        "output_dir": str(output_dir),
        "setting_id": setting["setting_id"],
        "gate_results": gate_results,
        "gate_evidence": evidence,
        "per_arm_metrics": per_arm_rows,
        "naturalistic_reporting_status": "REPORT_ONLY",
        "retained_logit_decode_run": False,
        "baseline_replay_run": False,
        "grid_run": False,
        "oai_status": "not_run_not_authorized",
        "downstream_status": "not_authorized",
        "next_action": (
            "human_decision_only_no_automatic_downstream"
            if scientific_verdict.startswith("PASS")
            else "stop_no_collection_oai_controller_or_rl"
        ),
        "consumed_input_fingerprint_sha256": input_hashes_sha256,
    }
    (output_dir / "RESULTS_SUMMARY.json").write_text(
        json.dumps(_safe_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete_stop_no_downstream",
                "technical_verdict": "PASS",
                "scientific_verdict": scientific_verdict,
                "downstream_authorized": False,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "data_collection/experiments/phase2_decision_opportunity_analysis_v1" / f"{timestamp}_decision"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    try:
        result = run(args.batch_root, args.config, output_dir)
    except Exception as exc:
        if output_dir.is_dir() and not (output_dir / "COMPLETED.json").exists():
            (output_dir / "FAILED.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "failed_stop_no_downstream",
                        "technical_verdict": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                        "downstream_authorized": False,
                        "written_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_manifest(output_dir)
        raise
    print(json.dumps(_safe_json(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
