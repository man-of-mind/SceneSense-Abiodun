"""Run one bounded v3 warning repair screen on the immutable audit batch.

This runner deliberately evaluates only the already least-nuisance v2 setting.
It is a structural smoke for a versioned causal tracker and recipient fusion
repair, not a calibration search, a parameter-selection run, or C2 evidence.
It never launches CARLA or OAI and never writes inside the source capture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

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
    grid_settings,
    load_replay_config,
)
from phase2_map_sharing.source_tracker_v3 import (
    TRACKER_V3_VERSION,
    SourceLocalCausalTrackerV3,
)


SCHEMA = "scenesense.phase2_warning_repair_screen.v3"
EXPECTED_SETTING_ID = "c20_a30_t05_u00"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "phase2_map_sharing/configs/warning_repair_screen_v3.yaml"
DEFAULT_REPLAY_CONFIG = (
    REPO_ROOT / "phase2_map_sharing/configs/calibration_replay_sufficiency_v1.yaml"
)
FROZEN_TRACKER = {
    "algorithm": TRACKER_V3_VERSION,
    "association_gate_m": 5.0,
    "maximum_missed_frames": 3,
    "minimum_confirmation_hits": 2,
    "duplicate_suppression_radius_m": 0.75,
    "velocity_smoothing_alpha": 0.5,
    "speed_plausibility_slack_m": 0.75,
    "maximum_vertical_speed_mps": 8.0,
    "maximum_speed_mps_by_class": {
        "person": 12.0,
        "pedestrian": 12.0,
        "cyclist": 25.0,
        "vehicle": 60.0,
        "object": 40.0,
    },
    "publish_missed_tracks": False,
}
FROZEN_RECIPIENT_MAP = {
    "algorithm": "quality_weighted_moment_equal_time_v1",
    "warning_emission_confidence_floor": 0.20,
    "association_gate_m": 3.0,
    "association_sigma_multiplier": 2.0,
    "warning_uncertainty_multiplier": 0.0,
    "track_ttl_s": 0.5,
    "max_transport_age_s": 1.0,
    "warning_horizon_s": 5.0,
    "safety_radius_m_by_class": {
        "pedestrian": 2.5,
        "cyclist": 3.0,
        "vehicle": 3.0,
    },
}
FROZEN_SCREEN_GATES = {
    "maximum_suite_a_benign_false_warning_active_frame_rate": 0.10,
    "maximum_cooperative_excess_false_warning_active_frame_rate": 0.02,
    "minimum_cooperative_target_lead_s": 0.50,
    "require_zero_registered_target_misses": True,
    "episode_rate_status": "report_only_insufficient_seven_second_exposure",
    "static_truth_status": "incomplete_actor_only_truth_manual_rgb_audit_required",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_exact(observed: object, expected: object, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{label} keys drifted")
        for key, value in expected.items():
            _require_exact(observed[key], value, f"{label}.{key}")
        return
    if isinstance(expected, bool):
        if observed is not expected:
            raise ValueError(f"{label} drifted")
        return
    if isinstance(expected, float):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) != expected
        ):
            raise ValueError(f"{label} drifted")
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f"{label} drifted")


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("warning repair config root must be a mapping")
    config = dict(payload)
    expected_root = {
        "schema_version",
        "authorization",
        "claim_scope",
        "source_detection_floor",
        "source_tracker",
        "recipient_map",
        "screen_gates",
    }
    if set(config) != expected_root:
        raise ValueError("warning repair config root keys drifted")
    if config["schema_version"] != SCHEMA:
        raise ValueError("unexpected warning repair schema")
    if config["authorization"] != "offline_immutable_batch_structural_screen_only":
        raise ValueError("warning repair screen is not authorized beyond offline analysis")
    if config["claim_scope"] != (
        "tracker_and_fusion_repair_diagnostic_not_parameter_selection_or_c2"
    ):
        raise ValueError("warning repair claim scope drifted")
    if _finite("source_detection_floor", config["source_detection_floor"]) != 0.05:
        raise ValueError("source detection floor must remain 0.05")

    tracker = config["source_tracker"]
    required_tracker = {
        "algorithm",
        "association_gate_m",
        "maximum_missed_frames",
        "minimum_confirmation_hits",
        "duplicate_suppression_radius_m",
        "velocity_smoothing_alpha",
        "speed_plausibility_slack_m",
        "maximum_vertical_speed_mps",
        "maximum_speed_mps_by_class",
        "publish_missed_tracks",
    }
    if not isinstance(tracker, Mapping) or set(tracker) != required_tracker:
        raise ValueError("source tracker contract keys drifted")
    _require_exact(tracker, FROZEN_TRACKER, "source_tracker")

    recipient = config["recipient_map"]
    required_recipient = {
        "algorithm",
        "warning_emission_confidence_floor",
        "association_gate_m",
        "association_sigma_multiplier",
        "warning_uncertainty_multiplier",
        "track_ttl_s",
        "max_transport_age_s",
        "warning_horizon_s",
        "safety_radius_m_by_class",
    }
    if not isinstance(recipient, Mapping) or set(recipient) != required_recipient:
        raise ValueError("recipient map contract keys drifted")
    _require_exact(recipient, FROZEN_RECIPIENT_MAP, "recipient_map")
    setting = (
        f"c{round(100 * _finite('confidence', recipient['warning_emission_confidence_floor'])):02d}_"
        f"a{round(10 * _finite('association', recipient['association_gate_m'])):02d}_"
        f"t{round(10 * _finite('ttl', recipient['track_ttl_s'])):02d}_"
        f"u{round(10 * _finite('uncertainty', recipient['warning_uncertainty_multiplier'])):02d}"
    )
    if setting != EXPECTED_SETTING_ID:
        raise ValueError("screen must use the preregistered least-nuisance v2 setting")

    gates = config["screen_gates"]
    required_gates = {
        "maximum_suite_a_benign_false_warning_active_frame_rate",
        "maximum_cooperative_excess_false_warning_active_frame_rate",
        "minimum_cooperative_target_lead_s",
        "require_zero_registered_target_misses",
        "episode_rate_status",
        "static_truth_status",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise ValueError("screen gate keys drifted")
    _require_exact(gates, FROZEN_SCREEN_GATES, "screen_gates")
    return config


def _tracker_kwargs(config: Mapping[str, object]) -> dict:
    tracker = config["source_tracker"]
    return {
        "association_gate_m": float(tracker["association_gate_m"]),
        "maximum_missed_frames": int(tracker["maximum_missed_frames"]),
        "minimum_confirmation_hits": int(tracker["minimum_confirmation_hits"]),
        "duplicate_suppression_radius_m": float(
            tracker["duplicate_suppression_radius_m"]
        ),
        "velocity_smoothing_alpha": float(tracker["velocity_smoothing_alpha"]),
        "speed_plausibility_slack_m": float(tracker["speed_plausibility_slack_m"]),
        "maximum_speed_mps_by_class": dict(
            tracker["maximum_speed_mps_by_class"]
        ),
        "maximum_vertical_speed_mps": float(tracker["maximum_vertical_speed_mps"]),
    }


def _replay_source_tracker_v3(
    role_dir: Path,
    role: str,
    config: Mapping[str, object],
) -> tuple[dict[int, list[dict]], dict]:
    tracker = SourceLocalCausalTrackerV3(role, **_tracker_kwargs(config))
    detections = pd.read_csv(role_dir / "runtime/final_detections.csv")
    processed = pd.read_csv(_single(role_dir / "streams", "*_metrics.csv")).sort_values(
        "frame_id"
    )
    floor = float(config["source_detection_floor"])
    by_frame: dict[int, list[dict]] = {}
    associations: list[dict] = []
    raw_detection_rows = 0
    published_rows = 0
    published_track_ids: set[str] = set()
    for state in processed.itertuples(index=False):
        frame_id = int(state.frame_id)
        timestamp = float(state.carla_timestamp)
        frame = detections[
            (detections["frame_id"].astype(int) == frame_id)
            & (pd.to_numeric(detections["score"]) >= floor - 1e-12)
        ].sort_values("detection_index")
        raw_detection_rows += len(frame)
        tracks, frame_associations = tracker.update(
            frame_id=frame_id,
            timestamp_s=timestamp,
            detections=frame.to_dict("records"),
        )
        # A confirmed track remains internal across the configured miss grace,
        # but only a current measurement is published.  Recipient TTL handles
        # causal short gaps without manufacturing a fresh source observation.
        visible = [row for row in tracks if int(row["missed_frames"]) == 0]
        by_frame[frame_id] = visible
        published_rows += len(visible)
        published_track_ids.update(str(row["source_track_id"]) for row in visible)
        associations.extend(frame_associations)
    association_frame = pd.DataFrame(associations)
    counts = (
        association_frame["association"].value_counts().sort_index().to_dict()
        if not association_frame.empty
        else {}
    )
    return by_frame, {
        "source_role": role,
        "frame_count": len(processed),
        "raw_detection_rows": raw_detection_rows,
        "published_observed_track_rows": published_rows,
        "published_unique_track_ids": len(published_track_ids),
        "association_counts": counts,
        "velocity_limited_association_count": int(
            pd.to_numeric(
                association_frame.get("velocity_limited", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .astype(bool)
            .sum()
        ),
    }


def _screen_setting(config: Mapping[str, object], replay_config: Mapping[str, object]) -> dict:
    expected = next(
        row for row in grid_settings(replay_config) if row["setting_id"] == EXPECTED_SETTING_ID
    )
    recipient = config["recipient_map"]
    observed = {
        "warning_emission_confidence_floor": float(
            recipient["warning_emission_confidence_floor"]
        ),
        "map_association_gate_m": float(recipient["association_gate_m"]),
        "map_track_ttl_s": float(recipient["track_ttl_s"]),
        "warning_uncertainty_multiplier": float(
            recipient["warning_uncertainty_multiplier"]
        ),
    }
    if any(observed[key] != float(expected[key]) for key in observed):
        raise ValueError("v3 screen setting differs from registered v2 comparison point")
    return expected


def _safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_artifact_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.json"
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "scenesense.phase2_warning_repair_screen_artifacts.v3",
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run(
    batch_root: Path,
    baseline_replay: Path,
    config_path: Path,
    output_dir: Path,
) -> dict:
    batch_root = batch_root.resolve()
    baseline_replay = baseline_replay.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == batch_root or output_dir.is_relative_to(batch_root):
        raise ValueError("screen output must be a create-only capture sibling")
    config = load_config(config_path)
    replay_config = load_replay_config(DEFAULT_REPLAY_CONFIG)
    setting = _screen_setting(config, replay_config)
    _, _, plan = _verify_capture_complete(batch_root)
    # Fail before creating the output or replaying tracks if this capture
    # declared static truth but any trajectory catalog is missing or corrupt.
    contexts = _truth_contexts(batch_root, plan, replay_config)
    if not (baseline_replay / "COMPLETED.json").is_file():
        raise ValueError("baseline replay is not complete")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    input_hashes = {
        "capture_completed_sha256": _sha256(batch_root / "COMPLETED.json"),
        "capture_batch_manifest_sha256": _sha256(batch_root / "batch_manifest.json"),
        "capture_plan_sha256": _sha256(batch_root / "plan.json"),
        "baseline_completed_sha256": _sha256(baseline_replay / "COMPLETED.json"),
        "baseline_manifest_sha256": _sha256(baseline_replay / "artifact_manifest.json"),
        "screen_config_sha256": _sha256(config_path),
        "screen_runner_code_sha256": _sha256(Path(__file__)),
        "replay_grid_code_sha256": _sha256(
            REPO_ROOT / "phase2_map_sharing/replay_calibration_grid.py"
        ),
        "tracker_code_sha256": _sha256(
            REPO_ROOT / "phase2_map_sharing/source_tracker_v3.py"
        ),
        "engine_code_sha256": _sha256(REPO_ROOT / "phase2_map_sharing/engine_v3.py"),
    }

    source_tracks: dict[tuple[str, str], dict[int, list[dict]]] = {}
    tracker_diagnostics: list[dict] = []
    for trajectory in plan["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in ("helper", "recipient"):
            role_dir = batch_root / trajectory_id / role
            tracks, diagnostic = _replay_source_tracker_v3(role_dir, role, config)
            source_tracks[(trajectory_id, role)] = tracks
            tracker_diagnostics.append(
                {"trajectory_id": trajectory_id, **diagnostic}
            )

    metric_rows: list[dict] = []
    warning_rows: list[dict] = []
    isolation_rows: list[dict] = []
    for trajectory in plan["trajectories"]:
        rows, warnings, isolation = _replay_trajectory_setting(
            batch_root,
            trajectory,
            setting,
            source_tracks,
            replay_config,
            engine_class=RecipientMapEngineV3,
        )
        metric_rows.extend(rows)
        warning_rows.extend(warnings)
        isolation_rows.append(isolation)

    adjudicated = _adjudicate_warnings(warning_rows, contexts, replay_config)
    enriched = _enrich_arm_metrics(metric_rows, adjudicated, contexts, replay_config)
    diagnostics = _candidate_diagnostics(
        enriched,
        cadence_s=float(replay_config["truth_evaluation"]["cadence_s"]),
    )

    metrics_frame = pd.DataFrame(enriched)
    diagnostics_frame = pd.DataFrame(diagnostics)
    baseline = pd.read_csv(baseline_replay / "candidate_diagnostics.csv")
    baseline = baseline[baseline["setting_id"].astype(str) == EXPECTED_SETTING_ID]
    comparison = baseline.merge(
        diagnostics_frame,
        on=["setting_id", "arm_id"],
        suffixes=("_v2", "_v3"),
        validate="one_to_one",
    )
    comparison["benign_false_active_rate_change_v3_minus_v2"] = (
        comparison["suite_a_benign_false_warning_active_frame_rate_v3"]
        - comparison["suite_a_benign_false_warning_active_frame_rate_v2"]
    )
    comparison["naturalistic_false_active_rate_change_v3_minus_v2"] = (
        comparison["naturalistic_false_warning_active_frame_rate_v3"]
        - comparison["naturalistic_false_warning_active_frame_rate_v2"]
    )

    benign_by_arm = diagnostics_frame.set_index("arm_id")[
        "suite_a_benign_false_warning_active_frame_rate"
    ].astype(float)
    positive = metrics_frame[
        metrics_frame["scenario_role"].astype(str) == "controlled_positive_occlusion"
    ].set_index("arm_id")
    gates = config["screen_gates"]
    maximum_false = float(
        gates["maximum_suite_a_benign_false_warning_active_frame_rate"]
    )
    maximum_excess = float(
        gates["maximum_cooperative_excess_false_warning_active_frame_rate"]
    )
    minimum_lead = float(gates["minimum_cooperative_target_lead_s"])
    absolute_false_gate = bool((benign_by_arm <= maximum_false + 1e-12).all())
    cooperative_excess_gate = all(
        float(benign_by_arm[arm])
        <= float(benign_by_arm["ego_only"]) + maximum_excess + 1e-12
        for arm in ("send_everything", "hazard_only")
    )
    zero_miss_gate = bool(
        (pd.to_numeric(positive["missed_registered_target"]) == 0).all()
    )
    lead_gate = all(
        pd.notna(positive.loc[arm, "warning_lead_vs_ego_s"])
        and float(positive.loc[arm, "warning_lead_vs_ego_s"]) >= minimum_lead - 1e-12
        for arm in ("send_everything", "hazard_only")
    )
    gate_results = {
        "absolute_false_warning_active_frame_rate": absolute_false_gate,
        "cooperative_false_warning_noninferiority": cooperative_excess_gate,
        "zero_registered_target_misses": zero_miss_gate,
        "minimum_cooperative_target_lead": lead_gate,
        "episode_rate": "REPORT_ONLY_SHORT_EXPOSURE",
        "static_truth_completeness": "INCOMPLETE_ACTOR_ONLY",
    }
    scientific_verdict = (
        "PROMISING_REPAIR_READY_FOR_HUMAN_REVIEW"
        if all((absolute_false_gate, cooperative_excess_gate, zero_miss_gate, lead_gate))
        else "FAIL_HOLD_STOP_NO_COLLECTION"
    )

    pd.DataFrame(tracker_diagnostics).to_csv(
        output_dir / "source_tracker_diagnostics.csv", index=False
    )
    metrics_frame.to_csv(output_dir / "arm_trajectory_metrics.csv", index=False)
    pd.DataFrame(warning_rows).to_csv(output_dir / "warning_events.csv", index=False)
    pd.DataFrame(adjudicated).to_csv(
        output_dir / "adjudicated_warning_events.csv", index=False
    )
    diagnostics_frame.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    comparison.to_csv(output_dir / "v2_v3_comparison.csv", index=False)
    (output_dir / "arm_state_isolation.json").write_text(
        json.dumps(_safe_json(isolation_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "technical_verdict": "PASS",
        "scientific_verdict": scientific_verdict,
        "claim_scope": config["claim_scope"],
        "source_batch": str(batch_root),
        "baseline_replay": str(baseline_replay),
        "setting_id": EXPECTED_SETTING_ID,
        "trajectory_count": len(plan["trajectories"]),
        "arm_metric_rows": len(metrics_frame),
        "warning_rows": len(warning_rows),
        "gate_results": gate_results,
        "benign_false_warning_active_frame_rate_by_arm": {
            arm: float(value) for arm, value in benign_by_arm.items()
        },
        "target_missed_by_arm": {
            arm: int(positive.loc[arm, "missed_registered_target"])
            for arm in positive.index
        },
        "target_lead_vs_ego_s_by_arm": {
            arm: (
                None
                if pd.isna(positive.loc[arm, "warning_lead_vs_ego_s"])
                else float(positive.loc[arm, "warning_lead_vs_ego_s"])
            )
            for arm in positive.index
        },
        "static_truth_note": (
            "Retained RGB proves actor-only truth omits real parked Town10 vehicles. "
            "This screen compares nuisance consistently but cannot complete static IDs "
            "or authorize parameter selection."
        ),
        "next_action": (
            "human_review_only_no_collection_or_oai"
            if scientific_verdict.startswith("PROMISING")
            else "stop_and_keep_batch_as_excluded_development_fixture"
        ),
        "input_hashes": input_hashes,
    }
    (output_dir / "RESULTS_SUMMARY.json").write_text(
        json.dumps(_safe_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete_stop_for_human_gate",
                "technical_verdict": "PASS",
                "scientific_verdict": scientific_verdict,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(output_dir)
    return summary


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "data_collection/experiments/phase2_warning_repair_screen_v3"
        / f"{timestamp}_screen"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--baseline-replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir()
    try:
        summary = run(
            args.batch_root,
            args.baseline_replay,
            args.config,
            output_dir,
        )
    except Exception as exc:
        if output_dir.is_dir() and not (output_dir / "COMPLETED.json").exists():
            (output_dir / "FAILED.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "failed_stop",
                        "error": f"{type(exc).__name__}: {exc}",
                        "written_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(_safe_json(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
