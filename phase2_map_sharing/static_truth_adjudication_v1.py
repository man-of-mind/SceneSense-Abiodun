"""Evaluation-only adapter for sealed static CARLA environment truth.

Dynamic actor-origin truth remains the primary truth source.  This module only
offers still-unmatched warnings to a verified, enabled static-object catalog,
using class-constrained one-to-one center matching.  A static match is an
identity association, not a hazard label; callers must run the matched object
through the same future-trajectory hazard calculation used for dynamic actors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from data_collection.phase2_static_environment_truth_v1 import (
    OBJECTS_CSV_NAME,
    verify_static_environment_truth_v1,
)


TRUTH_SOURCE_DYNAMIC = "dynamic_actor_origin"
TRUTH_SOURCE_STATIC = "static_environment_catalog_v1"
TRUTH_SOURCE_UNMATCHED = "unmatched"
STATIC_TRUTH_REQUIREMENT_SCHEMA = (
    "scenesense.phase2_static_truth_adjudication_requirement.v1"
)
DECISION_OPPORTUNITY_PILOT_STAGE_ID = "phase2_decision_opportunity_pilot_v1"
DECISION_OPPORTUNITY_PILOT_SCHEMA = (
    "scenesense.phase2_decision_opportunity_pilot.v1"
)


def normalize_static_semantic_class_v1(value: object) -> str:
    """Map CARLA catalog labels onto the policy/replay class vocabulary."""

    name = str(value).strip().lower()
    if name in {"car", "truck", "bus", "vehicle"}:
        return "vehicle"
    if name in {"person", "walker", "pedestrian"}:
        return "pedestrian"
    if name in {"bike", "bicycle", "cyclist"}:
        return "cyclist"
    return name


def load_verified_static_catalog_v1(static_truth_dir: Path) -> pd.DataFrame:
    """Verify the sealed artifact before returning enabled canonical objects."""

    root = Path(static_truth_dir)
    verification = verify_static_environment_truth_v1(root)
    catalog = pd.read_csv(root / OBJECTS_CSV_NAME, dtype={"enabled": str})
    enabled = catalog["enabled"].astype(str).str.strip().str.lower()
    if not enabled.isin({"true", "false"}).all():
        # The artifact verifier should already catch this.  Keep the adapter
        # independently fail-closed if its parser contract ever drifts.
        raise ValueError("static truth enabled state is invalid")
    catalog = catalog[enabled == "true"].copy()
    catalog["class_name"] = catalog["semantic_class"].map(
        normalize_static_semantic_class_v1
    )
    catalog["actor_id"] = catalog["environment_object_id"].astype(str)
    catalog["role_name"] = "static_environment_object"
    catalog["origin_x"] = pd.to_numeric(catalog["bbox_center_x_m"])
    catalog["origin_y"] = pd.to_numeric(catalog["bbox_center_y_m"])
    catalog["origin_z"] = pd.to_numeric(catalog["bbox_center_z_m"])
    catalog["yaw_deg"] = pd.to_numeric(catalog["bbox_rotation_yaw_deg"])
    catalog["length_m"] = 2.0 * pd.to_numeric(catalog["bbox_extent_x_m"])
    catalog["width_m"] = 2.0 * pd.to_numeric(catalog["bbox_extent_y_m"])
    catalog["height_m"] = 2.0 * pd.to_numeric(catalog["bbox_extent_z_m"])
    catalog["truth_source"] = TRUTH_SOURCE_STATIC
    catalog.attrs["verification"] = dict(verification)
    catalog.attrs["static_truth_dir"] = str(root.resolve())
    return catalog.reset_index(drop=True)


def maybe_load_verified_static_catalog_v1(
    trajectory_dir: Path,
    *,
    required: bool = False,
) -> Optional[pd.DataFrame]:
    """Return ``None`` only when a historical trajectory has no catalog.

    An existing but incomplete/tampered catalog always raises.  This distinction
    preserves the historical actor-only path without silently accepting corrupt
    future-pilot truth.
    """

    static_truth_dir = Path(trajectory_dir) / "static_environment_truth"
    if not static_truth_dir.exists():
        if bool(required):
            raise FileNotFoundError(
                "declared-required static environment truth is missing: "
                f"{static_truth_dir}"
            )
        return None
    return load_verified_static_catalog_v1(static_truth_dir)


def _load_optional_mapping(path: Path, label: str) -> Optional[Mapping[str, object]]:
    """Read an optional provenance object while rejecting malformed evidence."""

    candidate = Path(path)
    if not candidate.exists():
        return None
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} must be a regular file: {candidate}")
    try:
        if candidate.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        else:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {candidate}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping: {candidate}")
    return payload


def _static_truth_requirement_reasons(
    value: Mapping[str, object], source: str
) -> list[str]:
    """Return explicit reasons one provenance object requires static truth."""

    reasons: list[str] = []
    static_contract = value.get("static_environment_truth")
    if static_contract is not None:
        if not isinstance(static_contract, Mapping):
            raise ValueError(
                f"{source}.static_environment_truth must be a mapping"
            )
        if "enabled" in static_contract:
            enabled = static_contract["enabled"]
            if type(enabled) is not bool:
                raise ValueError(
                    f"{source}.static_environment_truth.enabled must be boolean"
                )
            if enabled:
                reasons.append(f"{source}:static_environment_truth.enabled=true")

    stage_id = str(value.get("stage_id", "")).strip()
    if stage_id == DECISION_OPPORTUNITY_PILOT_STAGE_ID:
        reasons.append(f"{source}:stage_id={stage_id}")
    for field in ("schema", "schema_version"):
        if str(value.get(field, "")).strip() == DECISION_OPPORTUNITY_PILOT_SCHEMA:
            reasons.append(f"{source}:{field}={DECISION_OPPORTUNITY_PILOT_SCHEMA}")

    if "pilot_provenance" in value:
        pilot = value["pilot_provenance"]
        if not isinstance(pilot, Mapping):
            raise ValueError(f"{source}.pilot_provenance must be a mapping")
        reasons.append(f"{source}:pilot_provenance_present")
    return reasons


def resolve_static_truth_requirement_v1(
    batch_root: Path,
    *,
    declared_sources: Sequence[tuple[str, Mapping[str, object]]] = (),
) -> dict[str, object]:
    """Resolve whether a capture/replay contract makes static truth mandatory.

    Historical batches that contain none of the declarations remain compatible
    with their actor-only evaluation.  A declaration is never inferred from the
    mere presence of a catalog: the requirement comes from capture provenance,
    while a present catalog is always integrity-verified by its loader.
    """

    root = Path(batch_root)
    sources: list[tuple[str, Mapping[str, object]]] = []
    for name, path in (
        ("capture_resolved_config", root / "resolved_config.yaml"),
        ("capture_batch_manifest", root / "batch_manifest.json"),
    ):
        payload = _load_optional_mapping(path, name)
        if payload is not None:
            sources.append((name, payload))
    for name, value in declared_sources:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        sources.append((str(name), value))

    reasons: list[str] = []
    for source, value in sources:
        reasons.extend(_static_truth_requirement_reasons(value, source))
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema": STATIC_TRUTH_REQUIREMENT_SCHEMA,
        "required": bool(unique_reasons),
        "reasons": unique_reasons,
        "inspected_sources": [name for name, _value in sources],
        "historical_actor_only_compatibility": not bool(unique_reasons),
    }


def load_trajectory_static_catalogs_v1(
    batch_root: Path,
    trajectory_ids: Sequence[object],
    *,
    declared_sources: Sequence[tuple[str, Mapping[str, object]]] = (),
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Load every present catalog and enforce any provenance declaration."""

    root = Path(batch_root)
    identifiers = [str(value).strip() for value in trajectory_ids]
    if any(not value for value in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("trajectory IDs must be non-empty and unique")
    requirement = resolve_static_truth_requirement_v1(
        root, declared_sources=declared_sources
    )
    catalogs: dict[str, pd.DataFrame] = {}
    for trajectory_id in identifiers:
        catalog = maybe_load_verified_static_catalog_v1(
            root / trajectory_id,
            required=bool(requirement["required"]),
        )
        if catalog is not None:
            catalogs[trajectory_id] = catalog
    return catalogs, requirement


def _unmatched_record() -> dict[str, object]:
    return {
        "current_truth_matched": 0,
        "current_truth_actor_id": None,
        "current_truth_role_name": None,
        "current_truth_distance_m": None,
        "truth_source": TRUTH_SOURCE_UNMATCHED,
        "current_truth_static_environment_object_id": None,
    }


def match_unmatched_warnings_to_static_v1(
    warnings: pd.DataFrame,
    dynamic_matches: Mapping[int, Mapping[str, object]],
    static_catalog: pd.DataFrame,
    *,
    gate_m: float,
) -> dict[int, dict[str, object]]:
    """Preserve dynamic matches, then match remaining warnings to static truth."""

    if not math.isfinite(float(gate_m)) or float(gate_m) <= 0.0:
        raise ValueError("gate_m must be finite and positive")
    required_warning = {"class_name", "track_world_x", "track_world_y"}
    required_static = {
        "class_name",
        "actor_id",
        "role_name",
        "origin_x",
        "origin_y",
        "environment_object_id",
    }
    if missing := required_warning - set(warnings.columns):
        raise ValueError(f"warning rows are missing fields: {sorted(missing)}")
    if missing := required_static - set(static_catalog.columns):
        raise ValueError(f"static truth rows are missing fields: {sorted(missing)}")
    warning_indices = {int(index) for index in warnings.index}
    if set(int(index) for index in dynamic_matches) != warning_indices:
        raise ValueError("dynamic match keys must cover every warning exactly")

    results: dict[int, dict[str, object]] = {}
    unmatched_indices: list[int] = []
    for index in warnings.index:
        warning_index = int(index)
        dynamic = dict(dynamic_matches[warning_index])
        if int(dynamic.get("current_truth_matched", 0)) == 1:
            dynamic["truth_source"] = TRUTH_SOURCE_DYNAMIC
            dynamic["current_truth_static_environment_object_id"] = None
            results[warning_index] = dynamic
        else:
            results[warning_index] = _unmatched_record()
            unmatched_indices.append(warning_index)

    if not unmatched_indices or static_catalog.empty:
        return results

    warning_classes = warnings["class_name"].map(
        normalize_static_semantic_class_v1
    )
    static_classes = static_catalog["class_name"].map(
        normalize_static_semantic_class_v1
    )
    for class_name in sorted(set(warning_classes.loc[unmatched_indices])):
        warning_group = warnings.loc[
            [
                index
                for index in unmatched_indices
                if warning_classes.loc[index] == class_name
            ]
        ].sort_index()
        static_group = static_catalog[static_classes == class_name].copy()
        static_group["_actor_sort"] = static_group["actor_id"].astype(str)
        static_group = static_group.sort_values("_actor_sort")
        if warning_group.empty or static_group.empty:
            continue

        warning_xy = warning_group[["track_world_x", "track_world_y"]].to_numpy(
            dtype=float
        )
        static_xy = static_group[["origin_x", "origin_y"]].to_numpy(dtype=float)
        if not np.isfinite(warning_xy).all() or not np.isfinite(static_xy).all():
            raise ValueError("warning/static match coordinates must be finite")
        distances = np.linalg.norm(
            warning_xy[:, np.newaxis, :] - static_xy[np.newaxis, :, :], axis=2
        )
        row_count, static_count = distances.shape
        unmatched_cost = (row_count + static_count + 1) * float(gate_m)
        invalid_cost = unmatched_cost * 3.0
        cost = np.full(
            (row_count, static_count + row_count), unmatched_cost, dtype=float
        )
        valid = distances <= float(gate_m) + 1e-12
        cost[:, :static_count] = np.where(valid, distances, invalid_cost)
        cost[:, :static_count] += (
            np.arange(static_count, dtype=float)[np.newaxis, :] * 1e-12
        )
        assigned_rows, assigned_columns = linear_sum_assignment(cost)
        for row_position, column_position in zip(assigned_rows, assigned_columns):
            if column_position >= static_count or not valid[
                row_position, column_position
            ]:
                continue
            warning_index = int(warning_group.index[row_position])
            static_row = static_group.iloc[column_position]
            results[warning_index] = {
                "current_truth_matched": 1,
                "current_truth_actor_id": str(static_row["actor_id"]),
                "current_truth_role_name": str(static_row["role_name"]),
                "current_truth_distance_m": float(
                    distances[row_position, column_position]
                ),
                "truth_source": TRUTH_SOURCE_STATIC,
                "current_truth_static_environment_object_id": str(
                    static_row["environment_object_id"]
                ),
            }
    return results


def constant_static_future_truth_v1(
    static_catalog: pd.DataFrame,
    *,
    actor_id: str,
    frame_times: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat one immutable OBB on the causal/counterfactual frame timebase."""

    required_times = {"frame_id", "carla_timestamp"}
    if missing := required_times - set(frame_times.columns):
        raise ValueError(f"frame times are missing fields: {sorted(missing)}")
    selected = static_catalog[
        static_catalog["actor_id"].astype(str) == str(actor_id)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"static actor ID must resolve exactly once: {actor_id!r}, rows={len(selected)}"
        )
    obj = selected.iloc[0]
    times = frame_times[["frame_id", "carla_timestamp"]].copy()
    if times["frame_id"].duplicated().any():
        raise ValueError("static future frame IDs must be unique")
    if not np.isfinite(
        times[["frame_id", "carla_timestamp"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("static future frame times must be finite")

    # Keep the native OBB fields as well as the derived full dimensions.  This
    # makes later surface-clearance calculations reproducible without guessing
    # whether the CARLA values were half-extents or full dimensions.
    copied_fields = (
        "bbox_center_x_m",
        "bbox_center_y_m",
        "bbox_center_z_m",
        "bbox_extent_x_m",
        "bbox_extent_y_m",
        "bbox_extent_z_m",
        "bbox_rotation_roll_deg",
        "bbox_rotation_pitch_deg",
        "bbox_rotation_yaw_deg",
        "bbox_coordinate_frame",
        "environment_object_id",
        "semantic_class",
        "map_name",
        "map_sha256",
    )
    constants = {
        "actor_id": str(obj["actor_id"]),
        "role_name": str(obj["role_name"]),
        "class_name": str(obj["class_name"]),
        "origin_x": float(obj["origin_x"]),
        "origin_y": float(obj["origin_y"]),
        "origin_z": float(obj["origin_z"]),
        "yaw_deg": float(obj["yaw_deg"]),
        "length_m": float(obj["length_m"]),
        "width_m": float(obj["width_m"]),
        "height_m": float(obj["height_m"]),
        "truth_source": TRUTH_SOURCE_STATIC,
        **{field: obj[field] for field in copied_fields},
    }
    for field, value in constants.items():
        times[field] = value
    return times
