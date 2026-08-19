"""Replay the frozen Phase-2 warning grid from an immutable audit capture.

This is an evaluation-only sufficiency audit.  It never feeds CARLA truth back
into source tracking, publication, map association, or warning generation, and
it cannot select an operating point or establish C2 from the small audit batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import re
import sys
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FUSION_PACKAGE_ROOT = _REPOSITORY_ROOT / "pole_lraspp_multimodal_fusion"
if str(_FUSION_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FUSION_PACKAGE_ROOT))

from data_collection.phase2_causal_runtime import (
    SourceLocalCausalTracker,
    TRACKER_VERSION,
)
from phase2_map_sharing.adjudicate_future_hazards import (
    _aligned_counterfactual_ego,
    _episode_count,
    _future_label,
    match_warnings_one_to_one,
)
from phase2_map_sharing.static_truth_adjudication_v1 import (
    TRUTH_SOURCE_STATIC,
    constant_static_future_truth_v1,
    load_trajectory_static_catalogs_v1,
    match_unmatched_warnings_to_static_v1,
)
from phase2_map_sharing.engine_v2 import RecipientMapEngineV2
from phase2_map_sharing.replay_paired_pilot import (
    ARMS,
    CLOCK_ID,
    _contribution,
    _latest_ego_state,
    _recipient_state,
)
from pole_lraspp_multimodal_fusion.object_targets import decode_objects


SCHEMA = "scenesense.phase2_calibration_replay_sufficiency.v1"
OUTPUT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ROLE_NAMES = ("helper", "recipient")
FROZEN_GRID = {
    "warning_emission_confidence_floors": (0.05, 0.10, 0.15, 0.20),
    "map_association_gates_m": (2.0, 3.0, 4.0),
    "map_track_ttls_s": (0.5, 1.0),
    "warning_uncertainty_multipliers": (0.0, 1.0, 2.0),
}
FROZEN_SOURCE_DECODE = {
    "captured_floor": 0.05,
    "topk": 120,
    "nms_radius_px": 2,
    "maximum_objects": 30,
    "object_class_names": ["vehicle", "person"],
    "predict_bbox2d": True,
    "parity_score_tolerance": 1.0e-6,
    "parity_position_tolerance_m": 1.0e-5,
}
FROZEN_SOURCE_TRACKER = {
    "algorithm": "source_local_nearest_cv.v1",
    "association_gate_m": 5.0,
    "maximum_missed_frames": 3,
    "tuned": False,
}
FROZEN_MAP_ENGINE_FIXED = {
    "association_sigma_multiplier": 2.0,
    "warning_horizon_s": 5.0,
    "max_transport_age_s": 1.0,
    "multi_source_install_order": ["recipient", "helper"],
    "equal_measurement_time_tie_semantics": (
        "latest_installed_observation_state_and_confidence_wins"
    ),
    "safety_radius_m_by_class": {
        "pedestrian": 2.5,
        "cyclist": 3.0,
        "vehicle": 3.0,
    },
}
FROZEN_TRUTH_EVALUATION = {
    "center_match_gate_m": 5.0,
    "future_horizon_s": 5.0,
    "cadence_s": 0.1,
    "false_warning_quiet_gap_s": 1.0,
    "registered_target_role_prefix": "phase2_registered_target_",
    "positive_ego_basis": "matched_benign_no_target_recipient_counterfactual",
    "benign_and_naturalistic_ego_basis": (
        "realized_nonactuated_recipient_trajectory"
    ),
}
FROZEN_INTEGRITY_CONTRACT = {
    "require_capture_completed": True,
    "require_all_role_artifact_hashes": True,
    "require_decode_parity_on_every_retained_logit": True,
    "require_baseline_tracker_parity": True,
    "require_runtime_hashes_unchanged": True,
    "output_is_create_only_sibling": True,
}
ROOT_CONFIG_KEYS = {
    "schema_version",
    "authorization",
    "claim_scope",
    "execution",
    "source_decode",
    "source_tracker",
    "map_engine_fixed",
    "replay_grid",
    "truth_evaluation",
    "integrity",
}
REQUIRED_RUNTIME_TRACK_COLUMNS = (
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
    "last_observed_frame_id",
    "missed_frames",
)
_WORKER_BATCH_ROOT: Optional[Path] = None
_WORKER_TRAJECTORIES: Optional[Sequence[Mapping[str, object]]] = None
_WORKER_SOURCE_TRACKS: Optional[
    Mapping[tuple[str, str], Mapping[int, Sequence[Mapping[str, object]]]]
] = None
_WORKER_CONFIG: Optional[Mapping[str, object]] = None


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


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _normalize_class(value: object) -> str:
    name = str(value).strip().lower()
    if name in {"person", "walker"}:
        return "pedestrian"
    if name in {"bike", "bicycle"}:
        return "cyclist"
    return name


def _require_exact_contract(
    observed: object,
    expected: object,
    label: str,
) -> None:
    """Compare a frozen scientific contract without bool/numeric coercion."""

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise ValueError(f"{label} must be a mapping")
        observed_keys = {str(key) for key in observed}
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            raise ValueError(
                f"{label} keys drifted: missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
        for key, expected_value in expected.items():
            _require_exact_contract(observed[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{label} drifted from the frozen ordered list")
        for index, expected_value in enumerate(expected):
            _require_exact_contract(observed[index], expected_value, f"{label}[{index}]")
        return
    if isinstance(expected, bool):
        if not isinstance(observed, bool) or observed is not expected:
            raise ValueError(f"{label} drifted: observed={observed!r}, frozen={expected!r}")
        return
    if isinstance(expected, int):
        if type(observed) is not int or observed != expected:
            raise ValueError(f"{label} drifted: observed={observed!r}, frozen={expected!r}")
        return
    if isinstance(expected, float):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(f"{label} must be the frozen numeric value {expected!r}")
        if not math.isfinite(float(observed)) or float(observed) != float(expected):
            raise ValueError(f"{label} drifted: observed={observed!r}, frozen={expected!r}")
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f"{label} drifted: observed={observed!r}, frozen={expected!r}")


def load_replay_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("replay config root must be a mapping")
    config = dict(payload)
    if set(config) != ROOT_CONFIG_KEYS:
        raise ValueError(
            "replay config root keys drifted: "
            f"missing={sorted(ROOT_CONFIG_KEYS - set(config))}, "
            f"extra={sorted(set(config) - ROOT_CONFIG_KEYS)}"
        )
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"unexpected replay schema: {config.get('schema_version')!r}")
    if config.get("authorization") != "evaluation_only_no_runtime_feedback":
        raise ValueError("replay must be evaluation-only with no runtime feedback")
    if config.get("claim_scope") != (
        "retained_artifact_replay_sufficiency_not_parameter_selection_or_c2_evidence"
    ):
        raise ValueError("replay claim scope drifted")
    _require_exact_contract(
        config.get("source_decode"), FROZEN_SOURCE_DECODE, "source_decode"
    )
    _require_exact_contract(
        config.get("source_tracker"), FROZEN_SOURCE_TRACKER, "source_tracker"
    )
    if str(config["source_tracker"]["algorithm"]) != TRACKER_VERSION:
        raise ValueError("source-local tracker version does not match runtime code")
    _require_exact_contract(
        config.get("map_engine_fixed"),
        FROZEN_MAP_ENGINE_FIXED,
        "map_engine_fixed",
    )
    _require_exact_contract(
        config.get("truth_evaluation"),
        FROZEN_TRUTH_EVALUATION,
        "truth_evaluation",
    )
    _require_exact_contract(
        config.get("integrity"), FROZEN_INTEGRITY_CONTRACT, "integrity"
    )
    grid = config.get("replay_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("replay_grid mapping is required")
    expected_grid_keys = {*FROZEN_GRID, "expected_combinations"}
    if set(grid) != expected_grid_keys:
        raise ValueError(
            "replay_grid keys drifted: "
            f"missing={sorted(expected_grid_keys - set(grid))}, "
            f"extra={sorted(set(grid) - expected_grid_keys)}"
        )
    for key, frozen in FROZEN_GRID.items():
        _require_exact_contract(
            grid.get(key), list(frozen), f"replay_grid.{key}"
        )
    product = math.prod(len(values) for values in FROZEN_GRID.values())
    _require_exact_contract(
        grid.get("expected_combinations"), product, "replay_grid.expected_combinations"
    )
    if product != 72:
        raise ValueError("frozen replay grid must contain exactly 72 settings")
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {"parallel_workers"}:
        raise ValueError("execution must contain only parallel_workers")
    workers = execution.get("parallel_workers")
    if type(workers) is not int:
        raise ValueError("execution.parallel_workers must be an integer")
    if not 1 <= workers <= 24:
        raise ValueError("execution.parallel_workers must be within [1, 24]")
    return config


def grid_settings(config: Mapping[str, object]) -> list[dict]:
    grid = config["replay_grid"]
    rows = []
    for confidence, association, ttl, uncertainty in itertools.product(
        grid["warning_emission_confidence_floors"],
        grid["map_association_gates_m"],
        grid["map_track_ttls_s"],
        grid["warning_uncertainty_multipliers"],
    ):
        parameters = {
            "warning_emission_confidence_floor": float(confidence),
            "map_association_gate_m": float(association),
            "map_track_ttl_s": float(ttl),
            "warning_uncertainty_multiplier": float(uncertainty),
        }
        rows.append(
            {
                "setting_id": (
                    f"c{int(round(100 * confidence)):02d}_"
                    f"a{int(round(10 * association)):02d}_"
                    f"t{int(round(10 * ttl)):02d}_"
                    f"u{int(round(10 * uncertainty)):02d}"
                ),
                **parameters,
                "setting_sha256": _semantic_sha256(parameters),
            }
        )
    if len(rows) != 72 or len({row["setting_id"] for row in rows}) != 72:
        raise AssertionError("frozen grid did not generate 72 unique settings")
    return rows


def default_output_dir(batch_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        batch_root.parent.parent
        / "phase2_calibration_replay_v1"
        / f"{timestamp}_replay"
    )


def _validate_output_dir(batch_root: Path, output_dir: Path) -> None:
    batch = batch_root.resolve()
    output = output_dir.resolve()
    if output == batch or output.is_relative_to(batch):
        raise ValueError("reanalysis output must be a sibling, never inside the capture")
    if not OUTPUT_PATTERN.fullmatch(output.name):
        raise ValueError(f"unsafe output directory basename: {output.name!r}")
    if output.exists():
        raise FileExistsError(f"create-only replay output already exists: {output}")


def _verify_capture_complete(batch_root: Path) -> tuple[dict, dict, dict]:
    completed = _load_json(batch_root / "COMPLETED.json")
    manifest = _load_json(batch_root / "batch_manifest.json")
    plan = _load_json(batch_root / "plan.json")
    if completed.get("status") != "audit_complete_stop_for_human_gate":
        raise ValueError("capture completion sentinel is not accepted")
    if manifest.get("status") != "audit_capture_and_per_trajectory_verification_complete":
        raise ValueError("batch manifest is not capture-complete")
    planned = [str(item["trajectory_id"]) for item in plan.get("trajectories", [])]
    completed_ids = [str(item["trajectory_id"]) for item in manifest.get("trajectories", [])]
    if not planned or planned != completed_ids:
        raise ValueError("plan and completed trajectory order differ")
    return completed, manifest, plan


def _role_directories(batch_root: Path, plan: Mapping[str, object]):
    for trajectory in plan["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        for role in ROLE_NAMES:
            role_dir = batch_root / trajectory_id / role
            if not role_dir.is_dir():
                raise FileNotFoundError(role_dir)
            yield trajectory, role, role_dir


def _capture_artifact_fingerprints(batch_root: Path) -> dict[str, dict[str, object]]:
    """Hash the complete immutable source tree, including unmanifested truth traces."""

    if not batch_root.is_dir():
        raise FileNotFoundError(batch_root)
    fingerprints: dict[str, dict[str, object]] = {}
    for path in sorted(batch_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"capture source must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(batch_root))
        fingerprints[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if not fingerprints:
        raise ValueError(f"capture source tree is empty: {batch_root}")
    return fingerprints


def _named_path_fingerprints(
    paths: Mapping[str, Path],
    *,
    require_files: bool,
) -> dict[str, dict[str, object]]:
    fingerprints: dict[str, dict[str, object]] = {}
    for label, raw_path in sorted(paths.items()):
        if raw_path.is_symlink():
            raise ValueError(f"result-defining path must not be a symlink: {raw_path}")
        path = raw_path.resolve()
        if not path.is_file():
            if require_files:
                raise FileNotFoundError(path)
            fingerprints[label] = {"path": str(path), "exists": False}
            continue
        fingerprints[label] = {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return fingerprints


def _result_defining_dependency_paths(config_path: Path) -> dict[str, Path]:
    return {
        "analysis_config": config_path,
        "replay_calibration_grid.py": Path(__file__),
        "data_collection/phase2_causal_runtime.py": (
            _REPOSITORY_ROOT / "data_collection/phase2_causal_runtime.py"
        ),
        "phase2_map_sharing/causal_contract.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/causal_contract.py"
        ),
        "phase2_map_sharing/retention.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/retention.py"
        ),
        "phase2_map_sharing/replay_paired_pilot.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/replay_paired_pilot.py"
        ),
        "phase2_map_sharing/adjudicate_future_hazards.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/adjudicate_future_hazards.py"
        ),
        "phase2_map_sharing/engine_v2.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/engine_v2.py"
        ),
        "phase2_map_sharing/schemas_v2.py": (
            _REPOSITORY_ROOT / "phase2_map_sharing/schemas_v2.py"
        ),
        "pole_lraspp_multimodal_fusion/object_targets.py": (
            _FUSION_PACKAGE_ROOT
            / "pole_lraspp_multimodal_fusion/object_targets.py"
        ),
        "phase2_map_sharing/WARNING_EVALUATION_DESIGN_FREEZE.md": (
            _REPOSITORY_ROOT
            / "phase2_map_sharing/WARNING_EVALUATION_DESIGN_FREEZE.md"
        ),
    }


def _checkpoint_paths(
    batch_root: Path, plan: Mapping[str, object]
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for trajectory, role, role_dir in _role_directories(batch_root, plan):
        manifest = _load_json(_single(role_dir / "manifests", "*_manifest.json"))
        raw_path = str(manifest.get("checkpoint_path", "")).strip()
        if raw_path:
            paths[f"{trajectory['trajectory_id']}:{role}:checkpoint"] = Path(raw_path)
    return paths


def _write_fingerprint_snapshot(
    path: Path,
    *,
    schema: str,
    fingerprints: Mapping[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {"schema": schema, "fingerprints": fingerprints},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fingerprint_drift(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(
            key for key in before_keys & after_keys if before[key] != after[key]
        ),
    }


def _verify_role_artifact_manifests(
    batch_root: Path,
    plan: Mapping[str, object],
    source_fingerprints: Mapping[str, Mapping[str, object]],
) -> dict:
    streams = {}
    for trajectory, role, role_dir in _role_directories(batch_root, plan):
        label = f"{trajectory['trajectory_id']}:{role}"
        manifest = _load_json(role_dir / "artifact_manifest.json")
        files = manifest.get("files", [])
        if not files:
            raise ValueError(f"empty role artifact manifest: {label}")
        total_bytes = 0
        for item in files:
            path = role_dir / str(item["path"])
            resolved = path.resolve()
            if not resolved.is_relative_to(role_dir.resolve()):
                raise ValueError(f"manifested artifact escapes role directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"manifested artifact missing: {path}")
            relative = str(path.relative_to(batch_root))
            observed = source_fingerprints.get(relative)
            if observed is None:
                raise ValueError(f"manifested artifact was not pre-hashed: {path}")
            if int(observed["bytes"]) != int(item["bytes"]):
                raise ValueError(f"manifested artifact size drift: {path}")
            if str(observed["sha256"]) != str(item["sha256"]):
                raise ValueError(f"manifested artifact hash drift: {path}")
            total_bytes += int(item["bytes"])
        streams[label] = {"file_count": len(files), "bytes": total_bytes}
    return streams


def _runtime_hashes(batch_root: Path, plan: Mapping[str, object]) -> dict[str, str]:
    hashes = {}
    for _, _, role_dir in _role_directories(batch_root, plan):
        for path in sorted((role_dir / "runtime").glob("*")):
            if path.is_file():
                hashes[str(path.relative_to(batch_root))] = _sha256(path)
    return hashes


def _decode_retained_parity(
    batch_root: Path,
    plan: Mapping[str, object],
    config: Mapping[str, object],
) -> list[dict]:
    source = config["source_decode"]
    rows = []
    for trajectory, role, role_dir in _role_directories(batch_root, plan):
        trajectory_id = str(trajectory["trajectory_id"])
        inputs = {
            int(path.stem.split("_")[1]): path
            for path in sorted((role_dir / "retained_inputs").glob("frame_*_inputs.npz"))
        }
        logits = {
            int(path.stem.split("_")[1]): path
            for path in sorted((role_dir / "retained_inputs").glob("frame_*_logits.npz"))
        }
        if not inputs or inputs.keys() != logits.keys():
            raise ValueError(
                f"retained input/logit frames are missing or misaligned: {trajectory_id}:{role}"
            )
        live = pd.read_csv(role_dir / "runtime/final_detections.csv")
        for frame_id in sorted(inputs):
            with np.load(inputs[frame_id], allow_pickle=False) as input_npz:
                camera_matrix = np.asarray(input_npz["camera_matrix"], dtype=np.float64)
                stored_frame = int(np.asarray(input_npz["frame_id"]).reshape(-1)[0])
            with np.load(logits[frame_id], allow_pickle=False) as logit_npz:
                object_logits = np.asarray(logit_npz["object"], dtype=np.float32)
            if stored_frame != frame_id:
                raise ValueError(f"retained input frame ID mismatch: {inputs[frame_id]}")
            decoded = decode_objects(
                torch.from_numpy(object_logits),
                camera_matrix=camera_matrix,
                topk=int(source["topk"]),
                score_threshold=float(source["captured_floor"]),
                nms_radius_px=int(source["nms_radius_px"]),
                object_class_names=tuple(source["object_class_names"]),
                predict_bbox2d=bool(source["predict_bbox2d"]),
            )[: int(source["maximum_objects"])]
            expected = (
                live[live["frame_id"].astype(int) == frame_id]
                .sort_values("detection_index")
                .reset_index(drop=True)
            )
            if len(decoded) != len(expected):
                raise ValueError(
                    f"decode count parity failed for {trajectory_id}:{role}:{frame_id}: "
                    f"decoded={len(decoded)} runtime={len(expected)}"
                )
            score_error = 0.0
            position_error = 0.0
            for index, prediction in enumerate(decoded):
                expected_row = expected.iloc[index]
                if int(expected_row["detection_index"]) != index:
                    raise ValueError("runtime detection indices are not contiguous decode order")
                if str(prediction["class_name"]) != str(expected_row["class_name"]):
                    raise ValueError(
                        f"decode class parity failed for {trajectory_id}:{role}:{frame_id}:{index}"
                    )
                score_error = max(
                    score_error,
                    abs(float(prediction["score"]) - float(expected_row["score"])),
                )
                position_error = max(
                    position_error,
                    *(abs(float(prediction[key]) - float(expected_row[key]))
                      for key in ("world_x", "world_y", "world_z")),
                )
            if score_error > float(source["parity_score_tolerance"]):
                raise ValueError("retained decode score parity exceeded tolerance")
            if position_error > float(source["parity_position_tolerance_m"]):
                raise ValueError("retained decode position parity exceeded tolerance")
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_role": role,
                    "frame_id": frame_id,
                    "decoded_count": len(decoded),
                    "runtime_count": len(expected),
                    "maximum_score_absolute_error": score_error,
                    "maximum_world_coordinate_absolute_error_m": position_error,
                    "pass": 1,
                }
            )
    return rows


def _replay_source_tracker(
    role_dir: Path,
    role: str,
    config: Mapping[str, object],
) -> tuple[dict[int, list[dict]], list[dict]]:
    tracker_config = config["source_tracker"]
    source_detection_floor = float(config["source_decode"]["captured_floor"])
    tracker = SourceLocalCausalTracker(
        role,
        association_gate_m=float(tracker_config["association_gate_m"]),
        maximum_missed_frames=int(tracker_config["maximum_missed_frames"]),
    )
    detections = pd.read_csv(role_dir / "runtime/final_detections.csv")
    processed = pd.read_csv(_single(role_dir / "streams", "*_metrics.csv")).sort_values(
        "frame_id"
    )
    by_frame: dict[int, list[dict]] = {}
    all_associations: list[dict] = []
    for state in processed.itertuples(index=False):
        frame_id = int(state.frame_id)
        timestamp = float(state.carla_timestamp)
        frame = detections[
            (detections["frame_id"].astype(int) == frame_id)
            & (
                pd.to_numeric(detections["score"])
                >= float(source_detection_floor) - 1e-12
            )
        ].sort_values("detection_index")
        tracks, associations = tracker.update(
            frame_id=frame_id,
            timestamp_s=timestamp,
            detections=frame.to_dict("records"),
        )
        augmented = []
        for track in tracks:
            augmented.append(
                {
                    **track,
                    "source_role": role,
                    "tracker_version": TRACKER_VERSION,
                    "frame_id": frame_id,
                    "carla_timestamp": timestamp,
                }
            )
        by_frame[frame_id] = augmented
        all_associations.extend(
            {
                **association,
                "source_role": role,
                "carla_timestamp": timestamp,
            }
            for association in associations
        )
    return by_frame, all_associations


def _max_numeric_error(left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]) -> float:
    maximum = 0.0
    for column in columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        mismatched_nan = np.isnan(a) ^ np.isnan(b)
        if mismatched_nan.any():
            return float("inf")
        if (~both_nan).any():
            maximum = max(maximum, float(np.max(np.abs(a[~both_nan] - b[~both_nan]))))
    return maximum


def _tracker_parity(
    role_dir: Path,
    replayed: Mapping[int, Sequence[Mapping[str, object]]],
    associations: Sequence[Mapping[str, object]],
) -> dict:
    generated_tracks = pd.DataFrame(
        [row for frame_rows in replayed.values() for row in frame_rows]
    )
    captured_tracks = pd.read_csv(role_dir / "runtime/causal_tracks.csv")
    track_sort = ["frame_id", "source_track_id"]
    generated_tracks = generated_tracks.sort_values(track_sort).reset_index(drop=True)
    captured_tracks = captured_tracks.sort_values(track_sort).reset_index(drop=True)
    if len(generated_tracks) != len(captured_tracks):
        raise ValueError("baseline source tracker row count differs from capture")
    text_columns = ["source_track_id", "source_role", "tracker_version", "class_name"]
    for column in text_columns:
        if generated_tracks[column].astype(str).tolist() != captured_tracks[column].astype(str).tolist():
            raise ValueError(f"baseline source tracker text parity failed: {column}")
    numeric_columns = [
        "frame_id", "carla_timestamp", "score", "world_x", "world_y", "world_z",
        "velocity_x", "velocity_y", "velocity_z", "last_observed_timestamp_s",
        "last_observed_frame_id", "missed_frames",
    ]
    track_error = _max_numeric_error(generated_tracks, captured_tracks, numeric_columns)
    if track_error > 1e-8:
        raise ValueError(f"baseline source tracker numeric parity failed: {track_error}")

    generated_associations = pd.DataFrame(associations)
    captured_associations = pd.read_csv(role_dir / "runtime/tracker_associations.csv")
    association_sort = ["frame_id", "source_track_id", "association"]
    generated_associations = generated_associations.sort_values(association_sort).reset_index(drop=True)
    captured_associations = captured_associations.sort_values(association_sort).reset_index(drop=True)
    if len(generated_associations) != len(captured_associations):
        raise ValueError("baseline source association row count differs from capture")
    for column in ("source_track_id", "source_role", "association", "class_name"):
        if generated_associations[column].astype(str).tolist() != captured_associations[column].astype(str).tolist():
            raise ValueError(f"baseline source association text parity failed: {column}")
    association_error = _max_numeric_error(
        generated_associations,
        captured_associations,
        ("frame_id", "carla_timestamp", "detection_index", "association_distance_m"),
    )
    if association_error > 1e-8:
        raise ValueError(f"baseline source association numeric parity failed: {association_error}")
    return {
        "track_rows": len(generated_tracks),
        "association_rows": len(generated_associations),
        "maximum_track_numeric_error": track_error,
        "maximum_association_numeric_error": association_error,
        "pass": 1,
    }


def _checkpoint_and_config_hash(role_dir: Path) -> tuple[str, str, str]:
    manifest_path = _single(role_dir / "manifests", "*_manifest.json")
    resolved_path = _single(role_dir / "manifests", "*_resolved_config.json")
    manifest = _load_json(manifest_path)
    checkpoint = Path(str(manifest.get("checkpoint_path", "")))
    checkpoint_sha = _sha256(checkpoint) if checkpoint.is_file() else "0" * 64
    checkpoint_status = (
        "current_file_hash_not_capture_time_authenticated"
        if checkpoint.is_file()
        else "checkpoint_file_unavailable"
    )
    return checkpoint_sha, _sha256(resolved_path), checkpoint_status


def _paired_role_provenance(
    helper_dir: Path, recipient_dir: Path
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role, role_dir in (("helper", helper_dir), ("recipient", recipient_dir)):
        model_sha, config_sha, checkpoint_status = _checkpoint_and_config_hash(
            role_dir
        )
        result[role] = {
            "model_sha256": model_sha,
            "config_sha256": config_sha,
            "checkpoint_hash_status": checkpoint_status,
        }
    return result


def _empty_tracks() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_RUNTIME_TRACK_COLUMNS)


def _tracks_frame(
    cache: Mapping[int, Sequence[Mapping[str, object]]], frame_id: int
) -> pd.DataFrame:
    rows = list(cache.get(int(frame_id), ()))
    return pd.DataFrame(rows) if rows else _empty_tracks()


def _replay_trajectory_setting(
    batch_root: Path,
    trajectory: Mapping[str, object],
    setting: Mapping[str, object],
    source_tracks: Mapping[tuple[str, str], Mapping[int, Sequence[Mapping[str, object]]]],
    config: Mapping[str, object],
    *,
    engine_class: type[RecipientMapEngineV2] = RecipientMapEngineV2,
) -> tuple[list[dict], list[dict], dict]:
    trajectory_id = str(trajectory["trajectory_id"])
    helper_dir = batch_root / trajectory_id / "helper"
    recipient_dir = batch_root / trajectory_id / "recipient"
    recipient_ego = pd.read_csv(recipient_dir / "runtime/ego_states.csv")
    helper_metrics = pd.read_csv(_single(helper_dir / "streams", "*_metrics.csv"))
    recipient_metrics = pd.read_csv(_single(recipient_dir / "streams", "*_metrics.csv"))
    if helper_metrics["frame_id"].duplicated().any() or recipient_metrics[
        "frame_id"
    ].duplicated().any():
        raise ValueError(f"paired role metrics contain duplicate frame IDs: {trajectory_id}")
    common_frames = sorted(
        set(recipient_metrics["frame_id"].astype(int))
        & set(helper_metrics["frame_id"].astype(int))
    )
    if len(common_frames) != len(recipient_metrics) or len(common_frames) != len(helper_metrics):
        raise ValueError(f"paired role frame coverage differs: {trajectory_id}")
    helper_timestamp_by_frame = (
        helper_metrics.set_index("frame_id")["carla_timestamp"].astype(float).to_dict()
    )
    recipient_timestamp_by_frame = (
        recipient_metrics.set_index("frame_id")["carla_timestamp"]
        .astype(float)
        .to_dict()
    )
    maximum_pair_timestamp_error_s = max(
        (
            abs(
                float(helper_timestamp_by_frame[frame_id])
                - float(recipient_timestamp_by_frame[frame_id])
            )
            for frame_id in common_frames
        ),
        default=0.0,
    )
    if maximum_pair_timestamp_error_s > 1e-9:
        raise ValueError(
            f"paired helper/recipient timestamps differ for {trajectory_id}: "
            f"maximum_error_s={maximum_pair_timestamp_error_s}"
        )
    confidence = float(setting["warning_emission_confidence_floor"])
    # The source detector and tracker are frozen at their captured 0.05/5 m/
    # three-miss contract.  Only the recipient warning-emission confidence
    # floor varies, after installation and association have already occurred.
    helper_tracks = source_tracks[(trajectory_id, "helper")]
    recipient_tracks = source_tracks[(trajectory_id, "recipient")]
    fixed = config["map_engine_fixed"]
    engines = {
        arm: engine_class(
            "recipient",
            association_gate_m=float(setting["map_association_gate_m"]),
            association_sigma_multiplier=float(fixed["association_sigma_multiplier"]),
            warning_sigma_multiplier=float(setting["warning_uncertainty_multiplier"]),
            track_ttl_s=float(setting["map_track_ttl_s"]),
            max_transport_age_s=float(fixed["max_transport_age_s"]),
            warning_horizon_s=float(fixed["warning_horizon_s"]),
            warning_emission_confidence_floor=confidence,
            safety_radius_m_by_class=fixed["safety_radius_m_by_class"],
        )
        for arm in ARMS
    }
    state_ids = {arm: id(engine.tracks) for arm, engine in engines.items()}
    if len(set(state_ids.values())) != len(ARMS):
        raise RuntimeError("counterfactual arms share mutable map state")
    role_provenance = _paired_role_provenance(helper_dir, recipient_dir)
    recipient_provenance = role_provenance["recipient"]
    helper_provenance = role_provenance["helper"]
    metrics = {
        arm: {
            "application_bytes": 0,
            "on_wire_bytes": 0,
            "warning_count": 0,
            "warning_frames": set(),
            "map_aoi_s": [],
            "map_track_counts": [],
        }
        for arm in ARMS
    }
    warning_rows: list[dict] = []
    timestamp_by_frame = recipient_timestamp_by_frame
    for sequence, frame_id in enumerate(common_frames):
        captured_at_s = float(timestamp_by_frame[frame_id])
        ego_row = _latest_ego_state(recipient_ego, captured_at_s)
        recipient_state = _recipient_state(ego_row, captured_at_s)
        recipient_contribution = _contribution(
            trajectory_id=trajectory_id,
            source_role="recipient",
            sequence=sequence,
            captured_at_s=captured_at_s,
            tracks=_tracks_frame(recipient_tracks, frame_id),
            publication_action="PUBLISH_ALL",
            recipient=recipient_state,
            model_sha256=recipient_provenance["model_sha256"],
            config_sha256=recipient_provenance["config_sha256"],
        )
        for arm, engine in engines.items():
            result = engine.install(recipient_contribution, captured_at_s, CLOCK_ID)
            if result != "accepted":
                raise RuntimeError(
                    f"{setting['setting_id']}:{trajectory_id}:{arm} rejected recipient: {result}"
                )
            if arm != "ego_only":
                action = (
                    "PUBLISH_ALL"
                    if arm == "send_everything"
                    else "PUBLISH_HAZARD_SUBSET"
                )
                helper_contribution = _contribution(
                    trajectory_id=trajectory_id,
                    source_role="helper",
                    sequence=sequence,
                    captured_at_s=captured_at_s,
                    tracks=_tracks_frame(helper_tracks, frame_id),
                    publication_action=action,
                    recipient=recipient_state,
                    model_sha256=helper_provenance["model_sha256"],
                    config_sha256=helper_provenance["config_sha256"],
                )
                result = engine.install(helper_contribution, captured_at_s, CLOCK_ID)
                if result != "accepted":
                    raise RuntimeError(
                        f"{setting['setting_id']}:{trajectory_id}:{arm} rejected helper: {result}"
                    )
                metrics[arm]["application_bytes"] += int(
                    helper_contribution.application_payload_bytes
                )
                metrics[arm]["on_wire_bytes"] += int(
                    helper_contribution.application_payload_bytes
                    + helper_contribution.chunk_count * 36
                )
            warnings = engine.warnings(recipient_state)
            snapshot = engine.snapshot(captured_at_s, CLOCK_ID)
            snapshot_by_id = {
                str(track["canonical_track_id"]): track for track in snapshot["tracks"]
            }
            metrics[arm]["map_track_counts"].append(len(snapshot["tracks"]))
            if warnings:
                metrics[arm]["warning_frames"].add(frame_id)
            for warning in warnings:
                track = snapshot_by_id[str(warning.canonical_track_id)]
                metrics[arm]["warning_count"] += 1
                metrics[arm]["map_aoi_s"].append(float(warning.map_aoi_s))
                warning_rows.append(
                    {
                        **setting,
                        "trajectory_id": trajectory_id,
                        "group_id": trajectory.get("group_id"),
                        "suite_id": trajectory.get("suite_id"),
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
                    }
                )
    metric_rows = []
    for arm, values in metrics.items():
        metric_rows.append(
            {
                **setting,
                "trajectory_id": trajectory_id,
                "group_id": trajectory.get("group_id"),
                "suite_id": trajectory.get("suite_id"),
                "scenario_role": trajectory["scenario_role"],
                "arm_id": arm,
                "frame_count": len(common_frames),
                "warning_count": int(values["warning_count"]),
                "warning_active_frame_count": len(values["warning_frames"]),
                "warning_active_frame_rate": (
                    len(values["warning_frames"]) / len(common_frames)
                    if common_frames
                    else None
                ),
                "application_bytes": int(values["application_bytes"]),
                "on_wire_bytes_loopback_framing_proxy": int(values["on_wire_bytes"]),
                "oai_on_wire_bytes_status": "unmeasured_remains_blocking",
                "mean_warning_map_aoi_s": (
                    float(np.mean(values["map_aoi_s"])) if values["map_aoi_s"] else 0.0
                ),
                "mean_map_track_count": (
                    float(np.mean(values["map_track_counts"]))
                    if values["map_track_counts"]
                    else 0.0
                ),
                "recipient_checkpoint_hash_status": recipient_provenance[
                    "checkpoint_hash_status"
                ],
                "helper_checkpoint_hash_status": helper_provenance[
                    "checkpoint_hash_status"
                ],
                "role_checkpoint_sha256_equal": (
                    recipient_provenance["model_sha256"]
                    == helper_provenance["model_sha256"]
                ),
                "role_resolved_config_sha256_equal": (
                    recipient_provenance["config_sha256"]
                    == helper_provenance["config_sha256"]
                ),
                "confirmatory_performance_evidence": 0,
            }
        )
    isolation = {
        "setting_id": setting["setting_id"],
        "trajectory_id": trajectory_id,
        "independent": len(set(state_ids.values())) == len(ARMS),
        "maximum_helper_recipient_timestamp_error_s": (
            maximum_pair_timestamp_error_s
        ),
        "engine_counters": {arm: dict(engine.counters) for arm, engine in engines.items()},
    }
    return metric_rows, warning_rows, isolation


def _initialize_replay_worker(
    batch_root: str,
    trajectories: Sequence[Mapping[str, object]],
    source_tracks: Mapping[
        tuple[str, str], Mapping[int, Sequence[Mapping[str, object]]]
    ],
    config: Mapping[str, object],
) -> None:
    global _WORKER_BATCH_ROOT
    global _WORKER_TRAJECTORIES
    global _WORKER_SOURCE_TRACKS
    global _WORKER_CONFIG
    _WORKER_BATCH_ROOT = Path(batch_root)
    _WORKER_TRAJECTORIES = trajectories
    _WORKER_SOURCE_TRACKS = source_tracks
    _WORKER_CONFIG = config


def _replay_setting_worker(
    setting: Mapping[str, object],
) -> tuple[list[dict], list[dict], list[dict]]:
    if any(
        value is None
        for value in (
            _WORKER_BATCH_ROOT,
            _WORKER_TRAJECTORIES,
            _WORKER_SOURCE_TRACKS,
            _WORKER_CONFIG,
        )
    ):
        raise RuntimeError("replay worker was not initialized")
    metrics: list[dict] = []
    warnings: list[dict] = []
    isolation: list[dict] = []
    assert _WORKER_BATCH_ROOT is not None
    assert _WORKER_TRAJECTORIES is not None
    assert _WORKER_SOURCE_TRACKS is not None
    assert _WORKER_CONFIG is not None
    for trajectory in _WORKER_TRAJECTORIES:
        rows, events, state = _replay_trajectory_setting(
            _WORKER_BATCH_ROOT,
            trajectory,
            setting,
            _WORKER_SOURCE_TRACKS,
            _WORKER_CONFIG,
        )
        metrics.extend(rows)
        warnings.extend(events)
        isolation.append(state)
    return metrics, warnings, isolation


def _ego_trace(trajectory_dir: Path) -> pd.DataFrame:
    # This trace is evaluation-only physical truth.  It is opened only after
    # warnings are generated and is never passed to either causal controller.
    trace = pd.read_csv(trajectory_dir / "scenario/realized_trace.csv").sort_values(
        "frame_id"
    )
    required = {
        "frame_id",
        "elapsed_s",
        "recipient_x",
        "recipient_y",
        "recipient_yaw_deg",
        "recipient_speed_mps",
    }
    if missing := required - set(trace.columns):
        raise ValueError(f"realized scenario trace lacks ego fields: {sorted(missing)}")
    return trace[list(sorted(required))].copy()


def _truth_contexts(
    batch_root: Path,
    plan: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, dict]:
    truth_config = config["truth_evaluation"]
    contexts: dict[str, dict] = {}
    static_catalogs, static_truth_requirement = (
        load_trajectory_static_catalogs_v1(
            batch_root,
            [trajectory["trajectory_id"] for trajectory in plan["trajectories"]],
            declared_sources=(("replay_config", config),),
        )
    )
    for trajectory in plan["trajectories"]:
        trajectory_id = str(trajectory["trajectory_id"])
        role_dir = batch_root / trajectory_id / "recipient"
        truth = pd.read_csv(_single(role_dir / "evaluation_truth", "*_ground_truth.csv"))
        frame_times = truth[["frame_id", "carla_timestamp"]].drop_duplicates("frame_id")
        target_prefix = (
            str(truth_config["registered_target_role_prefix"])
            if str(trajectory["scenario_role"]) == "controlled_positive_occlusion"
            else ""
        )
        if target_prefix:
            target = truth[truth["role_name"].astype(str).str.startswith(target_prefix)]
            if target.empty or target["actor_id"].astype(str).nunique() != 1:
                raise ValueError(
                    f"positive trajectory lacks exactly one registered target: {trajectory_id}"
                )
        contexts[trajectory_id] = {
            "trajectory": trajectory,
            "truth": truth,
            "frame_times": frame_times,
            "last_truth_s": float(pd.to_numeric(frame_times["carla_timestamp"]).max()),
            "ego": _ego_trace(batch_root / trajectory_id),
            "target_prefix": target_prefix,
            "static_truth_requirement": static_truth_requirement,
        }
        # Historical audit batches without a declaration keep their actor-only
        # semantics.  Declared future-pilot catalogs were required and verified
        # as a complete set before any per-trajectory truth was opened.
        static_catalog = static_catalogs.get(trajectory_id)
        if static_catalog is not None:
            contexts[trajectory_id]["static_catalog"] = static_catalog
    pairs: dict[str, list[Mapping[str, object]]] = {}
    for trajectory in plan["trajectories"]:
        pair_id = str(trajectory.get("matched_pair_id") or "").strip()
        if pair_id:
            pairs.setdefault(pair_id, []).append(trajectory)
    cadence = float(truth_config["cadence_s"])
    for pair_id, members in pairs.items():
        positive = [m for m in members if m["scenario_role"] == "controlled_positive_occlusion"]
        benign = [m for m in members if m["scenario_role"] == "matched_benign_negative"]
        if len(positive) != 1 or len(benign) != 1:
            raise ValueError(
                f"matched pair {pair_id} must contain one positive and one benign"
            )
        positive_id = str(positive[0]["trajectory_id"])
        benign_id = str(benign[0]["trajectory_id"])
        contexts[positive_id]["hazard_ego"] = _aligned_counterfactual_ego(
            contexts[positive_id]["ego"],
            contexts[benign_id]["ego"],
            cadence_s=cadence,
        )
        contexts[positive_id]["hazard_ego_basis"] = str(
            truth_config["positive_ego_basis"]
        )
        contexts[positive_id]["counterfactual_source_trajectory_id"] = benign_id
        contexts[benign_id]["hazard_ego"] = contexts[benign_id]["ego"]
        contexts[benign_id]["hazard_ego_basis"] = str(
            truth_config["benign_and_naturalistic_ego_basis"]
        )
        contexts[benign_id]["counterfactual_source_trajectory_id"] = None
    for trajectory_id, context in contexts.items():
        if "hazard_ego" not in context:
            context["hazard_ego"] = context["ego"]
            context["hazard_ego_basis"] = str(
                truth_config["benign_and_naturalistic_ego_basis"]
            )
            context["counterfactual_source_trajectory_id"] = None
    return contexts


def _adjudicate_warnings(
    warning_rows: Sequence[Mapping[str, object]],
    contexts: Mapping[str, Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict]:
    if not warning_rows:
        return []
    truth_config = config["truth_evaluation"]
    warnings = pd.DataFrame(warning_rows)
    safety_radii = {
        _normalize_class(key): float(value)
        for key, value in config["map_engine_fixed"]["safety_radius_m_by_class"].items()
    }
    rows: list[dict] = []
    group_columns = ["setting_id", "trajectory_id", "arm_id", "frame_id"]
    for (_, trajectory_id, _, frame_id), frame_warnings in warnings.groupby(
        group_columns, sort=True
    ):
        context = contexts[str(trajectory_id)]
        truth = context["truth"]
        truth_frame = truth[truth["frame_id"].astype(int) == int(frame_id)]
        dynamic_matches = match_warnings_one_to_one(
            frame_warnings,
            truth_frame,
            gate_m=float(truth_config["center_match_gate_m"]),
        )
        static_catalog = context.get("static_catalog")
        matches = (
            match_unmatched_warnings_to_static_v1(
                frame_warnings,
                dynamic_matches,
                static_catalog,
                gate_m=float(truth_config["center_match_gate_m"]),
            )
            if isinstance(static_catalog, pd.DataFrame)
            else dynamic_matches
        )
        for warning_index, event in frame_warnings.iterrows():
            base = dict(event)
            match = matches[int(warning_index)]
            base.update(match)
            base["hazard_ego_trajectory_basis"] = context["hazard_ego_basis"]
            base["counterfactual_source_trajectory_id"] = context[
                "counterfactual_source_trajectory_id"
            ]
            target_match = bool(
                match["current_truth_matched"]
                and context["target_prefix"]
                and str(match["current_truth_role_name"]).startswith(
                    str(context["target_prefix"])
                )
            )
            base["target_hazard_match_adjudicated"] = int(target_match)
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
                if match.get("truth_source") == TRUTH_SOURCE_STATIC:
                    if not isinstance(static_catalog, pd.DataFrame):
                        raise RuntimeError(
                            "static warning match lacks its verified catalog"
                        )
                    actor_truth = constant_static_future_truth_v1(
                        static_catalog,
                        actor_id=str(match["current_truth_actor_id"]),
                        frame_times=context["frame_times"],
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
                    horizon_s=float(truth_config["future_horizon_s"]),
                    safety_radius_m=float(
                        safety_radii.get(_normalize_class(base["class_name"]), 3.0)
                    ),
                    cadence_s=float(truth_config["cadence_s"]),
                    ego_dimensions=None,
                )
                result["false_warning_adjudicated"] = result.pop("false_warning")
                base.update(result)
            rows.append(base)
    return rows


def _enrich_arm_metrics(
    metric_rows: Sequence[Mapping[str, object]],
    adjudicated_rows: Sequence[Mapping[str, object]],
    contexts: Mapping[str, Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict]:
    metrics = pd.DataFrame(metric_rows)
    adjudicated = pd.DataFrame(adjudicated_rows)
    truth_config = config["truth_evaluation"]
    enriched: list[dict] = []
    for metric in metrics.to_dict("records"):
        trajectory_id = str(metric["trajectory_id"])
        setting_id = str(metric["setting_id"])
        arm_id = str(metric["arm_id"])
        context = contexts[trajectory_id]
        eligible_end = float(context["last_truth_s"]) - float(
            truth_config["future_horizon_s"]
        )
        eligible_frames = context["frame_times"][
            pd.to_numeric(context["frame_times"]["carla_timestamp"])
            <= eligible_end + 1e-9
        ]
        eligible_frame_ids = set(eligible_frames["frame_id"].astype(int))
        if adjudicated.empty:
            group = adjudicated
        else:
            group = adjudicated[
                (adjudicated["trajectory_id"].astype(str) == trajectory_id)
                & (adjudicated["setting_id"].astype(str) == setting_id)
                & (adjudicated["arm_id"].astype(str) == arm_id)
            ]
        eligible = (
            group[group["frame_id"].astype(int).isin(eligible_frame_ids)]
            if not group.empty
            else group
        )
        false_rows = (
            eligible[
                pd.to_numeric(
                    eligible["false_warning_adjudicated"], errors="coerce"
                )
                == 1
            ]
            if not eligible.empty
            else eligible
        )
        unmatched_rows = (
            eligible[eligible["future_label"] == "unmatched_false_warning"]
            if not eligible.empty
            else eligible
        )
        target_rows = (
            group[
                (
                    pd.to_numeric(
                        group["target_hazard_match_adjudicated"], errors="coerce"
                    )
                    == 1
                )
                & (
                    pd.to_numeric(
                        group["truth_hazard_positive"], errors="coerce"
                    )
                    == 1
                )
            ]
            if not group.empty
            else group
        )
        false_frames = int(false_rows["frame_id"].nunique()) if not false_rows.empty else 0
        unmatched_frames = (
            int(unmatched_rows["frame_id"].nunique()) if not unmatched_rows.empty else 0
        )
        duration_minutes = (
            len(eligible_frame_ids) * float(truth_config["cadence_s"]) / 60.0
        )
        episodes = _episode_count(
            pd.to_numeric(false_rows["warning_at_s"]).tolist()
            if not false_rows.empty
            else [],
            float(truth_config["false_warning_quiet_gap_s"]),
        )
        first_target = (
            float(pd.to_numeric(target_rows["warning_at_s"]).min())
            if not target_rows.empty
            else None
        )
        enriched.append(
            {
                **metric,
                "eligible_full_horizon_frame_count": len(eligible_frame_ids),
                "one_to_one_matched_warning_count": (
                    int(pd.to_numeric(group["current_truth_matched"]).sum())
                    if not group.empty
                    else 0
                ),
                "adjudicated_false_warning_count": len(false_rows),
                "false_warning_active_frame_count": false_frames,
                "false_warning_active_frame_rate": (
                    false_frames / len(eligible_frame_ids)
                    if eligible_frame_ids
                    else None
                ),
                "false_warning_episode_count": episodes,
                "false_warning_episodes_per_minute": (
                    episodes / duration_minutes if duration_minutes > 0.0 else None
                ),
                "unmatched_warning_active_frame_count": unmatched_frames,
                "unmatched_warning_active_frame_rate": (
                    unmatched_frames / len(eligible_frame_ids)
                    if eligible_frame_ids
                    else None
                ),
                "censored_warning_count": (
                    int(
                        (
                            pd.to_numeric(
                                group["future_truth_censored"], errors="coerce"
                            )
                            == 1
                        ).sum()
                    )
                    if not group.empty
                    else 0
                ),
                "truth_hazard_positive_warning_count": (
                    int(
                        (
                            pd.to_numeric(
                                group["truth_hazard_positive"], errors="coerce"
                            )
                            == 1
                        ).sum()
                    )
                    if not group.empty
                    else 0
                ),
                "first_registered_target_warning_s": first_target,
                "missed_registered_target": int(
                    bool(context["target_prefix"]) and first_target is None
                ),
                "hazard_ego_trajectory_basis": context["hazard_ego_basis"],
                "counterfactual_source_trajectory_id": context[
                    "counterfactual_source_trajectory_id"
                ],
            }
        )
    frame = pd.DataFrame(enriched)
    frame["warning_lead_vs_ego_s"] = np.nan
    for (_, trajectory_id), group in frame.groupby(["setting_id", "trajectory_id"]):
        ego = group[group["arm_id"] == "ego_only"]
        ego_first = (
            float(ego.iloc[0]["first_registered_target_warning_s"])
            if not ego.empty
            and pd.notna(ego.iloc[0]["first_registered_target_warning_s"])
            else None
        )
        for index in group.index:
            first = frame.at[index, "first_registered_target_warning_s"]
            if ego_first is not None and pd.notna(first):
                frame.at[index, "warning_lead_vs_ego_s"] = ego_first - float(first)
    return frame.to_dict("records")


def _candidate_diagnostics(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    cadence_s: float,
) -> list[dict]:
    if not math.isfinite(float(cadence_s)) or float(cadence_s) <= 0.0:
        raise ValueError("candidate diagnostic cadence_s must be finite and positive")
    metrics = pd.DataFrame(metric_rows)
    rows = []
    for (setting_id, arm_id), group in metrics.groupby(["setting_id", "arm_id"], sort=True):
        benign = group[
            group["scenario_role"].astype(str) == "matched_benign_negative"
        ]
        natural = group[group["scenario_role"].astype(str) == "naturalistic_operation"]
        eligible = int(benign["eligible_full_horizon_frame_count"].sum())
        false_frames = int(benign["false_warning_active_frame_count"].sum())
        exposure_minutes = eligible * float(cadence_s) / 60.0
        false_episodes = int(benign["false_warning_episode_count"].sum())
        base = group.iloc[0]
        rows.append(
            {
                "setting_id": setting_id,
                "arm_id": arm_id,
                "warning_emission_confidence_floor": base[
                    "warning_emission_confidence_floor"
                ],
                "map_association_gate_m": base["map_association_gate_m"],
                "map_track_ttl_s": base["map_track_ttl_s"],
                "warning_uncertainty_multiplier": base[
                    "warning_uncertainty_multiplier"
                ],
                "suite_a_benign_trajectory_count": len(benign),
                "suite_a_benign_eligible_frame_count": eligible,
                "suite_a_benign_false_warning_active_frame_rate": (
                    false_frames / eligible if eligible else None
                ),
                "suite_a_benign_false_warning_episodes_per_minute": (
                    false_episodes / exposure_minutes if exposure_minutes > 0.0 else None
                ),
                "naturalistic_trajectory_count": len(natural),
                "naturalistic_false_warning_active_frame_rate": (
                    int(natural["false_warning_active_frame_count"].sum())
                    / int(natural["eligible_full_horizon_frame_count"].sum())
                    if int(natural["eligible_full_horizon_frame_count"].sum())
                    else None
                ),
                "application_bytes_total": int(group["application_bytes"].sum()),
                "audit_gate_interpretation": (
                    "diagnostic_only_too_few_trajectory_clusters_for_parameter_selection"
                ),
            }
        )
    return rows


def _append_progress(output_dir: Path, event: str, **fields: object) -> None:
    row = {
        "schema": "scenesense.phase2_calibration_replay_progress.v1",
        "event": event,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


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
                "schema": "scenesense.phase2_calibration_replay_artifacts.v1",
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _decision_markdown(summary: Mapping[str, object], diagnostics: pd.DataFrame) -> str:
    benign = diagnostics[
        diagnostics["arm_id"].astype(str).isin(("ego_only", "send_everything", "hazard_only"))
    ]
    min_false = (
        float(pd.to_numeric(benign["suite_a_benign_false_warning_active_frame_rate"]).min())
        if not benign.empty
        else float("nan")
    )
    max_false = (
        float(pd.to_numeric(benign["suite_a_benign_false_warning_active_frame_rate"]).max())
        if not benign.empty
        else float("nan")
    )
    return f"""# Phase-2 calibration replay-sufficiency decision

Verdict: **{summary['verdict']}**

This create-only offline analysis executed all **{summary['setting_count']}**
settings from the binding warning-evaluation freeze across
**{summary['trajectory_count']} trajectories** and three isolated publication
arms. It is a replay-sufficiency audit only: it is not a powered calibration,
does not select an operating point, and is not C2 evidence.

## Integrity result

- Retained-logit canonical decode parity: **{summary['decode_parity_frames']} / {summary['decode_parity_frames']} frames**.
- Baseline source-tracker parity: **{summary['tracker_parity_streams']} / {summary['tracker_parity_streams']} streams**.
- Candidate/trajectory/arm rows: **{summary['arm_metric_rows']}**.
- All capture source artifacts unchanged: **{str(summary['all_capture_source_artifacts_unchanged']).lower()}**.
- Result-defining code and config unchanged: **{str(summary['result_defining_dependencies_unchanged']).lower()}**.
- OAI enqueue/on-wire/reassembly/install fields: **still blocking and unmeasured**.

## Diagnostic boundary

On this deliberately tiny subset, the Suite-A benign false-warning active-frame
rates span approximately **{min_false:.3f} to {max_false:.3f}** across settings
and arms. These values are useful for detecting an obviously broken warning
surface, but one positive/benign pair is not a trajectory-cluster calibration
set. The frozen 10% and 1/min gates must be pooled over the complete calibration
split; they are not applied independently to this 12-second trajectory.

The accepted capture remains immutable. All derived files are in this sibling
analysis directory and truth was joined only after causal replay.
"""


def replay(
    batch_root: Path,
    config: Mapping[str, object],
    output_dir: Path,
    *,
    config_path: Optional[Path] = None,
) -> dict:
    batch_root = batch_root.resolve()
    output_dir = output_dir.resolve()
    if config_path is None:
        raise ValueError("config_path is required for an auditable replay")
    config_path = config_path.resolve()
    file_config = load_replay_config(config_path)
    if _semantic_sha256(file_config) != _semantic_sha256(config):
        raise ValueError("in-memory replay config differs from the hashed config file")
    _validate_output_dir(batch_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _append_progress(output_dir, "replay_started", batch_root=str(batch_root))
        dependency_paths = _result_defining_dependency_paths(config_path)
        dependency_before = _named_path_fingerprints(
            dependency_paths, require_files=True
        )
        _write_fingerprint_snapshot(
            output_dir / "result_defining_dependencies_before.json",
            schema="scenesense.phase2_result_defining_dependencies.v1",
            fingerprints=dependency_before,
        )
        _append_progress(output_dir, "source_artifact_hashing_started")
        source_before = _capture_artifact_fingerprints(batch_root)
        _write_fingerprint_snapshot(
            output_dir / "source_artifact_fingerprints_before.json",
            schema="scenesense.phase2_source_artifact_fingerprints.v1",
            fingerprints=source_before,
        )
        _, batch_manifest, plan = _verify_capture_complete(batch_root)
        # Resolve truth completeness before decoding logits or starting the
        # replay worker pool.  A pilot that declared static truth cannot spend a
        # long offline run only to discover a missing/tampered catalog later.
        contexts = _truth_contexts(batch_root, plan, config)
        checkpoint_paths = _checkpoint_paths(batch_root, plan)
        checkpoint_before = _named_path_fingerprints(
            checkpoint_paths, require_files=False
        )
        _write_fingerprint_snapshot(
            output_dir / "external_checkpoint_fingerprints_before.json",
            schema="scenesense.phase2_external_checkpoint_fingerprints.v1",
            fingerprints=checkpoint_before,
        )
        settings = grid_settings(config)
        runtime_before = _runtime_hashes(batch_root, plan)
        _append_progress(output_dir, "capture_contract_validated")

        artifact_streams = _verify_role_artifact_manifests(
            batch_root, plan, source_before
        )
        _append_progress(
            output_dir,
            "source_artifact_integrity_passed",
            stream_count=len(artifact_streams),
        )
        decode_rows = _decode_retained_parity(batch_root, plan, config)
        decode_frame_expected = sum(
            len(list((role_dir / "retained_inputs").glob("frame_*_logits.npz")))
            for _, _, role_dir in _role_directories(batch_root, plan)
        )
        if len(decode_rows) != decode_frame_expected or not all(
            int(row["pass"]) == 1 for row in decode_rows
        ):
            raise RuntimeError("retained decoder parity is incomplete")
        pd.DataFrame(decode_rows).to_csv(output_dir / "decode_parity.csv", index=False)
        _append_progress(
            output_dir, "decode_parity_passed", frame_count=len(decode_rows)
        )

        captured_floor = float(config["source_decode"]["captured_floor"])
        source_tracks: dict[
            tuple[str, str], Mapping[int, Sequence[Mapping[str, object]]]
        ] = {}
        tracker_rows = []
        for trajectory, role, role_dir in _role_directories(batch_root, plan):
            trajectory_id = str(trajectory["trajectory_id"])
            replayed, associations = _replay_source_tracker(
                role_dir, role, config
            )
            parity = _tracker_parity(role_dir, replayed, associations)
            source_tracks[(trajectory_id, role)] = replayed
            tracker_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "source_role": role,
                    "source_detector_floor": captured_floor,
                    "source_tracker_association_gate_m": float(
                        config["source_tracker"]["association_gate_m"]
                    ),
                    "source_tracker_maximum_missed_frames": int(
                        config["source_tracker"]["maximum_missed_frames"]
                    ),
                    **parity,
                }
            )
        pd.DataFrame(tracker_rows).to_csv(output_dir / "tracker_parity.csv", index=False)
        _append_progress(
            output_dir, "source_tracker_parity_passed", stream_count=len(tracker_rows)
        )

        all_metrics: list[dict] = []
        all_warnings: list[dict] = []
        isolation_rows: list[dict] = []
        worker_count = min(int(config["execution"]["parallel_workers"]), len(settings))
        worker_trajectories = [
            {
                key: trajectory.get(key)
                for key in (
                    "trajectory_id",
                    "group_id",
                    "suite_id",
                    "scenario_role",
                )
            }
            for trajectory in plan["trajectories"]
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_replay_worker,
            initargs=(str(batch_root), worker_trajectories, source_tracks, config),
        ) as executor:
            results = executor.map(_replay_setting_worker, settings, chunksize=1)
            for ordinal, (setting, result) in enumerate(
                zip(settings, results), start=1
            ):
                metrics, warnings, isolation = result
                all_metrics.extend(metrics)
                all_warnings.extend(warnings)
                isolation_rows.extend(isolation)
                _append_progress(
                    output_dir,
                    "setting_complete",
                    ordinal=ordinal,
                    total=len(settings),
                    setting_id=setting["setting_id"],
                )

        expected_metric_rows = len(settings) * len(plan["trajectories"]) * len(ARMS)
        if len(all_metrics) != expected_metric_rows:
            raise RuntimeError(
                f"arm metric row count mismatch: {len(all_metrics)} != {expected_metric_rows}"
            )
        if not all(bool(item["independent"]) for item in isolation_rows):
            raise RuntimeError("one or more replay arms shared mutable map state")
        adjudicated = _adjudicate_warnings(all_warnings, contexts, config)
        if len(adjudicated) != len(all_warnings):
            raise RuntimeError("truth adjudication did not preserve every warning row")
        enriched_metrics = _enrich_arm_metrics(
            all_metrics, adjudicated, contexts, config
        )
        diagnostics = _candidate_diagnostics(
            enriched_metrics,
            cadence_s=float(config["truth_evaluation"]["cadence_s"]),
        )

        settings_frame = pd.DataFrame(settings)
        metrics_frame = pd.DataFrame(enriched_metrics)
        warnings_frame = pd.DataFrame(all_warnings)
        adjudicated_frame = pd.DataFrame(adjudicated)
        diagnostics_frame = pd.DataFrame(diagnostics)
        settings_frame.to_csv(output_dir / "grid_settings.csv", index=False)
        metrics_frame.to_csv(output_dir / "arm_trajectory_metrics.csv", index=False)
        warnings_frame.to_csv(output_dir / "warning_events.csv", index=False)
        adjudicated_frame.to_csv(
            output_dir / "adjudicated_warning_events.csv", index=False
        )
        diagnostics_frame.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
        (output_dir / "arm_state_isolation.json").write_text(
            json.dumps(
                {
                    "schema": "scenesense.phase2_grid_arm_isolation.v1",
                    "independent_state_per_arm": True,
                    "rows": isolation_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        runtime_after = _runtime_hashes(batch_root, plan)
        runtime_unchanged = runtime_before == runtime_after
        if not runtime_unchanged:
            raise RuntimeError("capture runtime artifacts changed during replay")
        source_after = _capture_artifact_fingerprints(batch_root)
        dependency_after = _named_path_fingerprints(
            dependency_paths, require_files=True
        )
        checkpoint_after = _named_path_fingerprints(
            checkpoint_paths, require_files=False
        )
        _write_fingerprint_snapshot(
            output_dir / "source_artifact_fingerprints_after.json",
            schema="scenesense.phase2_source_artifact_fingerprints.v1",
            fingerprints=source_after,
        )
        _write_fingerprint_snapshot(
            output_dir / "result_defining_dependencies_after.json",
            schema="scenesense.phase2_result_defining_dependencies.v1",
            fingerprints=dependency_after,
        )
        _write_fingerprint_snapshot(
            output_dir / "external_checkpoint_fingerprints_after.json",
            schema="scenesense.phase2_external_checkpoint_fingerprints.v1",
            fingerprints=checkpoint_after,
        )
        drift = {
            "source_artifacts": _fingerprint_drift(source_before, source_after),
            "result_defining_dependencies": _fingerprint_drift(
                dependency_before, dependency_after
            ),
            "external_checkpoints": _fingerprint_drift(
                checkpoint_before, checkpoint_after
            ),
        }
        (output_dir / "immutability_drift_report.json").write_text(
            json.dumps(
                {
                    "schema": "scenesense.phase2_replay_immutability_drift.v1",
                    "drift": drift,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if source_before != source_after:
            raise RuntimeError("one or more capture source artifacts changed during replay")
        if dependency_before != dependency_after:
            raise RuntimeError("result-defining code or config changed during replay")
        if checkpoint_before != checkpoint_after:
            raise RuntimeError("an externally referenced checkpoint changed during replay")
        setting_counts = metrics_frame.groupby("setting_id").size()
        every_setting_complete = bool(
            len(setting_counts) == 72
            and (setting_counts == len(plan["trajectories"]) * len(ARMS)).all()
        )
        if not every_setting_complete:
            raise RuntimeError("one or more frozen settings lacks complete arm/trajectory rows")
        positive_metrics = metrics_frame[
            metrics_frame["scenario_role"].astype(str)
            == "controlled_positive_occlusion"
        ]
        target_endpoint_computable = bool(
            pd.to_numeric(
                positive_metrics["first_registered_target_warning_s"], errors="coerce"
            ).notna().any()
        )
        if not target_endpoint_computable:
            raise RuntimeError("registered-target warning endpoint is not computable in any setting")

        integrity = {
            "schema": "scenesense.phase2_calibration_replay_integrity.v1",
            "verdict": "PASS",
            "gates": {
                "capture_complete": True,
                "role_artifact_hashes_verified": True,
                "decode_parity_all_retained_frames": True,
                "baseline_source_tracker_parity": True,
                "frozen_grid_exactly_72": len(settings) == 72,
                "every_setting_complete": every_setting_complete,
                "counterfactual_arm_state_isolated": True,
                "truth_join_row_preservation": len(adjudicated) == len(all_warnings),
                "registered_target_endpoint_computable": target_endpoint_computable,
                "runtime_artifacts_unchanged": runtime_unchanged,
                "all_capture_source_artifacts_unchanged": source_before == source_after,
                "result_defining_dependencies_unchanged": (
                    dependency_before == dependency_after
                ),
                "external_checkpoints_unchanged": (
                    checkpoint_before == checkpoint_after
                ),
            },
            "capture_artifact_streams": artifact_streams,
            "capture_source_artifact_count": len(source_before),
            "accepted_capture_resolved_config_sha256": source_before[
                "resolved_config.yaml"
            ]["sha256"],
            "accepted_capture_grid_declaration_status": (
                "superseded_96_point_provenance_not_executed"
            ),
        }
        (output_dir / "integrity_report.json").write_text(
            json.dumps(integrity, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema": "scenesense.phase2_calibration_replay_provenance.v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_batch": str(batch_root),
            "source_batch_manifest_sha256": source_before["batch_manifest.json"][
                "sha256"
            ],
            "source_completion_sha256": source_before["COMPLETED.json"]["sha256"],
            "source_artifact_fingerprint_file": (
                "source_artifact_fingerprints_before.json"
            ),
            "source_artifact_count": len(source_before),
            "analysis_config": str(config_path),
            "analysis_config_sha256": dependency_before["analysis_config"]["sha256"],
            "analysis_config_semantic_sha256": _semantic_sha256(config),
            "result_defining_dependencies": dependency_before,
            "external_checkpoint_fingerprints": checkpoint_before,
            "truth_usage": "evaluation_only_after_causal_warning_generation",
            "source_perception_surface": (
                "fixed_detector_floor_0.05_and_fixed_source_tracker_not_tuned"
            ),
            "checkpoint_hash_status": (
                "current_file_hash_recorded_in_contributions_not_capture_time_authenticated"
            ),
            "multi_source_install_order": config["map_engine_fixed"][
                "multi_source_install_order"
            ],
            "equal_measurement_time_tie_semantics": config["map_engine_fixed"][
                "equal_measurement_time_tie_semantics"
            ],
            "tie_semantics_scope": (
                "deterministic_replay_sufficiency_baseline_not_c2_evidence"
            ),
            "output_is_create_only_sibling": True,
        }
        (output_dir / "analysis_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema": SCHEMA,
            "status": "complete_stop_for_human_gate",
            "verdict": "PASS",
            "claim_scope": config["claim_scope"],
            "source_batch": str(batch_root),
            "output_dir": str(output_dir),
            "trajectory_count": len(plan["trajectories"]),
            "setting_count": len(settings),
            "arm_metric_rows": len(enriched_metrics),
            "warning_event_rows": len(all_warnings),
            "adjudicated_warning_rows": len(adjudicated),
            "decode_parity_frames": len(decode_rows),
            "decode_parity_detections": int(
                pd.DataFrame(decode_rows)["decoded_count"].sum()
            ),
            "tracker_parity_streams": len(tracker_rows),
            "runtime_artifacts_unchanged": runtime_unchanged,
            "all_capture_source_artifacts_unchanged": source_before == source_after,
            "result_defining_dependencies_unchanged": (
                dependency_before == dependency_after
            ),
            "parameter_selection_authorized": False,
            "c2_claim_authorized": False,
            "oai_field_gate": batch_manifest.get("oai_field_gate"),
            "next_action": "human_review_replay_sufficiency_and_warning_surface",
        }
        (output_dir / "replay_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "REPLAY_SUFFICIENCY_DECISION.md").write_text(
            _decision_markdown(summary, diagnostics_frame), encoding="utf-8"
        )
        _append_progress(
            output_dir,
            "stage_complete",
            verdict="PASS",
            next_action=summary["next_action"],
        )
        (output_dir / "COMPLETED.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "RESULTS_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_artifact_manifest(output_dir)
        return summary
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "failed_hold",
            "verdict": "FAIL_HOLD",
            "error": f"{type(exc).__name__}: {exc}",
            "source_batch": str(batch_root),
            "output_dir": str(output_dir),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _append_progress(output_dir, "stage_failed", error=failure["error"])
        _write_artifact_manifest(output_dir)
        raise


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            repository_root
            / "phase2_map_sharing/configs/calibration_replay_sufficiency_v1.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_replay_config(config_path)
    batch_root = args.batch_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(batch_root).resolve()
    )
    summary = replay(
        batch_root,
        config,
        output_dir,
        config_path=config_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
