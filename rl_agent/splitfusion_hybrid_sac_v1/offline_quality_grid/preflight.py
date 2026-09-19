"""CPU/metadata-only Phase-A1a input verification.

This module intentionally has no torch import.  It opens the two allowlisted
episode manifests, the frozen model-validation manifest, catalog/source text,
and checkpoint bytes for SHA-256 only.  It never loads a checkpoint or reads a
sensor payload.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .contract import (
    ACTION_CATALOG_Q_E4,
    ACTION_CATALOG_RELPATH,
    ACTION_CATALOG_SHA256,
    ALLOWED_EPISODE_IDS,
    BEHAVIORAL_SOURCE_FILE_ROLES,
    BEHAVIORAL_SOURCE_PACKAGE_ROLES,
    CHECKPOINTS,
    CONTRACT_SHA256,
    EPISODES,
    EXPECTED_GRID_ROWS,
    FAMILIES,
    MODEL_DATASET_MANIFEST_RELPATH,
    MODEL_DATASET_MANIFEST_SHA256,
    MODEL_EVALUATION_ROOT_RELPATH,
    MODEL_VALIDATION_FRAMES,
    Q_E4_GRID,
    QUANTIZERS,
    REQUIRED_BEHAVIORAL_SOURCE_ROLES,
    RUNTIME_BEHAVIORAL_ARTIFACTS,
    VALIDATION_AVO_TABLE_RELPATH,
    VALIDATION_AVO_TABLE_SHA256,
    OfflineGridContractError,
    canonical_sha256,
    mode_inventory,
    repository_root,
    sha256_file,
)

REQUIRED_MANIFEST_FIELDS = (
    "experiment_id",
    "sample_id",
    "split",
    "rgb_path",
    "mask_path",
    "instance_raw_path",
    "radar_tensor_path",
    "radar_points_path",
    "frame_id",
    "timestamp",
    "camera_width",
    "camera_height",
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "anchor_pitch",
    "anchor_yaw",
    "anchor_roll",
    "vehicle_pixels",
    "person_pixels",
)
SOURCE_PAYLOAD_FIELDS = (
    "rgb_path",
    "mask_path",
    "instance_raw_path",
    "radar_tensor_path",
    "radar_points_path",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise OfflineGridContractError(f"CSV has no header: {path}")
        missing = sorted(set(REQUIRED_MANIFEST_FIELDS).difference(reader.fieldnames))
        if missing:
            raise OfflineGridContractError(f"manifest fields missing at {path}: {missing}")
        return list(reader)


def _safe_relative(raw: str, field_name: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise OfflineGridContractError(
            f"{field_name} must be a non-traversing relative path, got {raw!r}"
        )
    return path


def _validate_episode_rows(binding: Any, rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != binding.expected_rows:
        raise OfflineGridContractError(
            f"{binding.episode_id}: {len(rows)} rows != {binding.expected_rows}"
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise OfflineGridContractError(f"{binding.episode_id}: duplicate sample_id")
    frame_ids: list[int] = []
    for row in rows:
        if row["experiment_id"] != binding.episode_id:
            raise OfflineGridContractError(
                f"foreign episode in {binding.episode_id} manifest: {row['experiment_id']!r}"
            )
        if row["split"] != "val":
            raise OfflineGridContractError(
                f"{binding.episode_id}: non-validation row {row['sample_id']}"
            )
        if not row["sample_id"].startswith(binding.episode_id + "_"):
            raise OfflineGridContractError(
                f"sample identity is not episode-bound: {row['sample_id']!r}"
            )
        try:
            frame_id = int(row["frame_id"])
            width, height = int(row["camera_width"]), int(row["camera_height"])
        except ValueError as exc:
            raise OfflineGridContractError(
                f"non-integer manifest identity at {row['sample_id']}"
            ) from exc
        if frame_id < 0 or (width, height) != (1280, 720):
            raise OfflineGridContractError(
                f"manifest geometry/identity drift at {row['sample_id']}"
            )
        frame_ids.append(frame_id)
        for field_name in SOURCE_PAYLOAD_FIELDS:
            _safe_relative(row[field_name], field_name)
    if len(frame_ids) != len(set(frame_ids)):
        raise OfflineGridContractError(f"{binding.episode_id}: duplicate frame_id")
    return {
        "rows": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_frame_ids": len(set(frame_ids)),
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
    }


def _load_model_validation_ids(
    root: Path,
) -> tuple[set[str], Counter[str], int, dict[str, dict[str, str]]]:
    path = root / MODEL_DATASET_MANIFEST_RELPATH
    if sha256_file(path) != MODEL_DATASET_MANIFEST_SHA256:
        raise OfflineGridContractError("frozen model dataset manifest SHA-256 drift")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    rows_by_id: dict[str, dict[str, str]] = {}
    total_val = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("split") != "val":
                continue
            total_val += 1
            episode = str(row.get("experiment_id", ""))
            if episode not in ALLOWED_EPISODE_IDS:
                raise OfflineGridContractError(
                    f"frozen validation manifest contains a foreign episode {episode!r}"
                )
            sample_id = str(row.get("sample_id", ""))
            if sample_id in ids:
                raise OfflineGridContractError(
                    f"frozen model manifest duplicate sample_id {sample_id!r}"
                )
            ids.add(sample_id)
            rows_by_id[sample_id] = dict(row)
            counts[episode] += 1
    if total_val != MODEL_VALIDATION_FRAMES or len(ids) != MODEL_VALIDATION_FRAMES:
        raise OfflineGridContractError(
            f"model validation population drift: {total_val}/{len(ids)}"
        )
    return ids, counts, total_val, rows_by_id


def _validate_catalog(root: Path) -> dict[str, Any]:
    path = root / ACTION_CATALOG_RELPATH
    digest = sha256_file(path)
    if digest != ACTION_CATALOG_SHA256:
        raise OfflineGridContractError("action catalog SHA-256 drift")
    document = json.loads(path.read_text(encoding="utf-8"))
    order = document.get("action_order", {})
    if tuple(order.get("family", ())) != FAMILIES:
        raise OfflineGridContractError("action catalog family order drift")
    if tuple(order.get("quantizer", ())) != QUANTIZERS:
        raise OfflineGridContractError("action catalog quantizer order drift")
    if tuple(order.get("q_e4", ())) != ACTION_CATALOG_Q_E4:
        raise OfflineGridContractError("action catalog q_e4 order drift")
    modes = mode_inventory()
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(modes) * len(
        ACTION_CATALOG_Q_E4
    ):
        raise OfflineGridContractError("action catalog profile cardinality drift")
    by_action_id: dict[int, Mapping[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise OfflineGridContractError("action catalog profile is not an object")
        action_id = profile.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise OfflineGridContractError("action catalog action_id is not an exact int")
        if action_id in by_action_id:
            raise OfflineGridContractError("action catalog has duplicate action_id")
        by_action_id[action_id] = profile
    for mode_id, family, quantizer in modes:
        for anchor_index, q_e4 in enumerate(ACTION_CATALOG_Q_E4):
            action_id = mode_id * len(ACTION_CATALOG_Q_E4) + anchor_index
            profile = by_action_id.get(action_id)
            if profile is None or (
                profile.get("family"), profile.get("quantizer"), profile.get("q_e4")
            ) != (family, quantizer, q_e4):
                raise OfflineGridContractError(
                    f"action catalog identity drift at action {action_id}"
                )
    return {
        "path": ACTION_CATALOG_RELPATH,
        "sha256": digest,
        "catalog_anchor_q_e4": list(order.get("q_e4", ())),
        "offline_exact_q_e4": list(Q_E4_GRID),
        "modes": [
            {"mode_id": mode_id, "family": family, "quantizer": quantizer}
            for mode_id, family, quantizer in modes
        ],
    }


def behavioral_source_bindings(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Hash the explicit repository-wide behavioral source closure."""

    roles: dict[str, str] = {}

    def register(relative: str, role: str) -> None:
        previous = roles.get(relative)
        if previous is not None and previous != role:
            raise OfflineGridContractError(
                f"behavioral source has conflicting roles: {relative}"
            )
        roles[relative] = role

    for relative, role in BEHAVIORAL_SOURCE_PACKAGE_ROLES.items():
        package = root / relative
        if not package.is_dir():
            raise OfflineGridContractError(f"behavioral source package is missing: {relative}")
        files = [
            path for path in sorted(package.rglob("*.py"))
            if "__pycache__" not in path.parts
        ]
        if not files:
            raise OfflineGridContractError(f"behavioral source package is empty: {relative}")
        for path in files:
            register(str(path.relative_to(root)), role)
    for relative, role in BEHAVIORAL_SOURCE_FILE_ROLES.items():
        if not (root / relative).is_file():
            raise OfflineGridContractError(f"behavioral source file is missing: {relative}")
        register(relative, role)
    for relative, role in REQUIRED_BEHAVIORAL_SOURCE_ROLES.items():
        if roles.get(relative) != role:
            raise OfflineGridContractError(
                f"required behavioral source/role is absent: {relative}"
            )
    hashes = {relative: sha256_file(root / relative) for relative in sorted(roles)}
    for relative, (expected, _role) in RUNTIME_BEHAVIORAL_ARTIFACTS.items():
        if hashes.get(relative) != expected:
            raise OfflineGridContractError(
                f"runtime behavioral artifact SHA-256 drift: {relative}"
            )
    return hashes, dict(sorted(roles.items()))


def run_metadata_preflight(root: Path | None = None) -> dict[str, Any]:
    """Verify all fixed identities without importing/loading any model."""

    root = repository_root() if root is None else Path(root).resolve(strict=True)
    catalog = _validate_catalog(root)
    model_ids, model_counts, total_val, model_rows = _load_model_validation_ids(root)

    episode_reports: list[dict[str, Any]] = []
    executable_total = 0
    for binding in EPISODES:
        manifest_path = root / binding.manifest_relpath
        observed = sha256_file(manifest_path)
        if observed != binding.manifest_sha256:
            raise OfflineGridContractError(
                f"{binding.episode_id} manifest SHA-256 drift: {observed}"
            )
        rows = _read_csv_rows(manifest_path)
        report = _validate_episode_rows(binding, rows)
        source_ids = {row["sample_id"] for row in rows}
        executable = source_ids.intersection(model_ids)
        excluded = sorted(source_ids.difference(model_ids))
        if len(executable) < binding.selected_rows:
            raise OfflineGridContractError(
                f"{binding.episode_id} has only {len(executable)} executable rows"
            )
        if len(executable) != model_counts[binding.episode_id]:
            raise OfflineGridContractError(
                f"model/source intersection drift for {binding.episode_id}"
            )
        source_by_id = {row["sample_id"]: row for row in rows}
        identity_fields = (
            "frame_id", "timestamp", "camera_width", "camera_height", "camera_fx",
            "camera_fy", "camera_cx", "camera_cy", "camera_matrix_json",
        )
        for sample_id in executable:
            source_row = source_by_id[sample_id]
            model_row = model_rows[sample_id]
            for field_name in identity_fields:
                if model_row.get(field_name) != source_row.get(field_name):
                    raise OfflineGridContractError(
                        f"model/source {field_name} drift at {sample_id}"
                    )
            for field_name in ("rgb_path", "radar_tensor_path"):
                expected = f"{binding.episode_id}/{source_row[field_name]}"
                if model_row.get(field_name) != expected:
                    raise OfflineGridContractError(
                        f"model/source {field_name} binding drift at {sample_id}"
                    )
        executable_total += len(executable)
        episode_reports.append(
            {
                "episode_id": binding.episode_id,
                "grid_split": binding.grid_split,
                "manifest_path": binding.manifest_relpath,
                "manifest_sha256": observed,
                "selected_rows": binding.selected_rows,
                "executable_intersection_rows": len(executable),
                "excluded_before_frozen_model_dataset_count": len(excluded),
                "excluded_before_frozen_model_dataset_sample_ids": excluded,
                "model_source_identity_rows_verified": len(executable),
                **report,
            }
        )
    if executable_total != MODEL_VALIDATION_FRAMES:
        raise OfflineGridContractError("episode executable intersections do not reconcile")

    checkpoint_reports: dict[str, Any] = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        path = root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise OfflineGridContractError(f"{name} checkpoint SHA-256 drift")
        checkpoint_reports[name] = {
            "path": relative,
            "sha256": observed,
            "bytes": path.stat().st_size,
            "loaded": False,
        }

    source_bindings, source_roles = behavioral_source_bindings(root)

    evaluation_source_bindings = {
        f"{MODEL_EVALUATION_ROOT_RELPATH}/object_boxes.csv": sha256_file(
            root / MODEL_EVALUATION_ROOT_RELPATH / "object_boxes.csv"
        ),
        f"{MODEL_EVALUATION_ROOT_RELPATH}/target_manifest.csv": sha256_file(
            root / MODEL_EVALUATION_ROOT_RELPATH / "target_manifest.csv"
        ),
        VALIDATION_AVO_TABLE_RELPATH: sha256_file(root / VALIDATION_AVO_TABLE_RELPATH),
    }
    if evaluation_source_bindings[VALIDATION_AVO_TABLE_RELPATH] != VALIDATION_AVO_TABLE_SHA256:
        raise OfflineGridContractError("validation AVO table SHA-256 drift")

    report: dict[str, Any] = {
        "schema": "splitfusion_exact_offline_quality_grid_preflight_v1",
        "status": "PASS_METADATA_ONLY_NO_MODEL_OR_CUDA_INFERENCE",
        "contract_sha256": CONTRACT_SHA256,
        "repository_root": str(root),
        "episodes": episode_reports,
        "frozen_model_validation_manifest": {
            "path": MODEL_DATASET_MANIFEST_RELPATH,
            "sha256": MODEL_DATASET_MANIFEST_SHA256,
            "validation_rows": total_val,
            "episode_counts": dict(sorted(model_counts.items())),
        },
        "catalog": catalog,
        "checkpoints": checkpoint_reports,
        "source_bindings": source_bindings,
        "source_roles": source_roles,
        "behavioral_source_closure": {
            "policy": (
                "package-wide transitive Python binding plus known-SHA runtime "
                "configs, locks, priors, checkpoint-selection decisions and actual checkpoints"
            ),
            "files": len(source_bindings),
            "required_paths": len(REQUIRED_BEHAVIORAL_SOURCE_ROLES),
            "runtime_behavioral_artifacts": len(RUNTIME_BEHAVIORAL_ARTIFACTS),
            "complete_for_repository_behavioral_inputs_used_by_executor": True,
            "third_party_dependencies": (
                "outside repository-source closure; frozen Phase-11 preflight and "
                "runtime numerical guards additionally apply at execution"
            ),
        },
        "evaluation_source_bindings": evaluation_source_bindings,
        "expected_grid_rows": EXPECTED_GRID_ROWS,
        "reward_spec": {
            "status": "REQUIRED_FOR_EXECUTION_NOT_REGISTERED_IN_REPOSITORY",
            "policy": "execution requires canonical RewardSpecV1 JSON and caller-pinned SHA-256",
            "quality_role": "PROVISIONAL_RECOMPUTABLE_QUALITY_CALIBRATION",
            "raw_sufficient_statistics_are_primary": True,
            "scalar_reward_weights_used_by_extraction": False,
        },
        "test_episode_access": "NONE_ALLOWLIST_ONLY_05_06",
        "cuda_queried": False,
        "checkpoint_objects_loaded": False,
        "preflight_binding_sha256": "",
    }
    report["preflight_binding_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "preflight_binding_sha256"}
    )
    return report
