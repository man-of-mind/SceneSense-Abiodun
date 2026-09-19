"""Deterministic SI/P40 and object-density stratified frame selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..scene_descriptors import (
    InvalidRadarRangesError,
    RadarUnavailableError,
    SCHEMA_SHA256 as SCENE_DESCRIPTOR_SHA256,
    camera_spatial_information,
    radar_proximity_p40,
)
from .contract import (
    CONTRACT_SHA256,
    EPISODES,
    MODEL_EVALUATION_ROOT_RELPATH,
    SAMPLING_SEED,
    SELECTION_SCHEMA_ID,
    STRATIFICATION,
    TOTAL_SELECTED_FRAMES,
    OfflineGridContractError,
    canonical_json_bytes,
    canonical_sha256,
    repository_root,
    sha256_file,
)
from .preflight import (
    SOURCE_PAYLOAD_FIELDS,
    _load_model_validation_ids,
    _read_csv_rows,
    _safe_relative,
    run_metadata_preflight,
)

SELECTION_SOURCE_GLOBAL_FILES = (
    "manifest.csv",
    "metadata.json",
    "resolved_config.json",
    "object_boxes.csv",
    "object_visibility.csv",
    "depth_frames.csv",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    episode_id: str
    grid_split: str
    source_position: int
    sample_id: str
    frame_id: int
    timestamp: float
    source_paths: Mapping[str, str]
    camera_si: float | None
    camera_si_valid: bool
    camera_si_status: str
    radar_p40: float | None
    radar_p40_valid: bool
    radar_p40_status: str
    current_sweep_raw_returns: int
    current_sweep_valid_returns: int
    current_sweep_rejected_returns: int
    vehicle_pixels: int
    person_pixels: int
    vehicle_present: bool
    person_present: bool
    ego_world_pose: tuple[float, float, float, float, float, float]
    si_bin: int = -1
    p40_bin: int = -1
    vehicle_density_bin: int = -1
    person_density_bin: int = -1

    @property
    def stratum(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.camera_si_valid),
            self.si_bin,
            int(self.radar_p40_valid),
            self.p40_bin,
            self.vehicle_density_bin,
            self.person_density_bin,
        )


def _sha_rank(*parts: Any) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quantile_edges(values: Sequence[float], quantiles: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        raise OfflineGridContractError("non-finite stratification value")
    return tuple(float(value) for value in np.quantile(array, quantiles, method="linear"))


def _bin(value: float | None, valid: bool, edges: Sequence[float]) -> int:
    if not valid or value is None:
        return -1
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))


def _density_bin(pixels: int, positive_edges: Sequence[float]) -> int:
    if pixels == 0:
        return 0
    return 1 + int(
        np.searchsorted(np.asarray(positive_edges, dtype=np.float64), pixels, side="right")
    )


def _load_camera_si(path: Path) -> float:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (720, 1280, 3) or bgr.dtype != np.uint8:
        raise OfflineGridContractError(f"invalid source RGB for SI: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
    luma = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return camera_spatial_information(luma)


def _load_current_sweep_p40(path: Path) -> tuple[float | None, bool, str, int, int, int]:
    raw_count = valid_count = rejected = 0
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "sweep_offset" not in payload.files or "original_range_m" not in payload.files:
                raise OfflineGridContractError(
                    f"radar provenance lacks sweep_offset/original_range_m: {path}"
                )
            offset = np.asarray(payload["sweep_offset"])
            ranges = np.asarray(payload["original_range_m"])
        if offset.ndim != 1 or ranges.ndim != 1 or offset.shape != ranges.shape:
            raise OfflineGridContractError(f"malformed radar provenance arrays: {path}")
        current = np.ascontiguousarray(ranges[offset == 0])
        valid_mask = np.isfinite(current) & (current > 0.0) & (current <= 120.0)
        raw_count = int(current.size)
        valid_count = int(np.count_nonzero(valid_mask))
        rejected = raw_count - valid_count
        # The frozen descriptor owns validation.  In particular, one malformed
        # element invalidates the whole current sweep; silently filtering it
        # would change both P40 and the deterministic selection stratum.
        value = radar_proximity_p40(current)
        return value, True, "VALID", raw_count, valid_count, rejected
    except RadarUnavailableError as exc:
        return None, False, type(exc).__name__, raw_count, valid_count, rejected
    except InvalidRadarRangesError as exc:
        return None, False, type(exc).__name__, raw_count, valid_count, rejected


def _source_hashes(
    episode_root: Path, row: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for field_name in SOURCE_PAYLOAD_FIELDS:
        relative = row[field_name]
        path = episode_root / relative
        if not path.is_file():
            raise OfflineGridContractError(
                f"selected source payload is missing: {row['sample_id']}:{relative}"
            )
        paths[field_name] = relative
        hashes[field_name] = sha256_file(path)
    return paths, hashes


def _evaluation_source_hashes(root: Path, sample_id: str) -> tuple[dict[str, str], dict[str, str]]:
    base = Path(MODEL_EVALUATION_ROOT_RELPATH)
    paths = {
        "segmentation_gt_path": str(base / "segmentation_masks" / f"{sample_id}.png"),
        "object_ignore_mask_path": str(base / "object_ignore_masks" / f"{sample_id}.png"),
    }
    hashes: dict[str, str] = {}
    for name, relative in paths.items():
        path = root / relative
        if not path.is_file():
            raise OfflineGridContractError(
                f"selected evaluation payload is missing: {sample_id}:{relative}"
            )
        hashes[name] = sha256_file(path)
    return paths, hashes


def describe_candidates(
    *,
    root: Path,
    episode_id: str,
    grid_split: str,
    rows: Sequence[Mapping[str, str]],
    executable_ids: set[str],
    progress_every: int = 100,
) -> list[Candidate]:
    episode_root = root / "data_collection/experiments/route_b_perception_v3" / episode_id
    candidates: list[Candidate] = []
    for position, row in enumerate(rows):
        sample_id = row["sample_id"]
        if sample_id not in executable_ids:
            continue
        paths = {
            field_name: str(_safe_relative(row[field_name], field_name))
            for field_name in SOURCE_PAYLOAD_FIELDS
        }
        camera_si = _load_camera_si(episode_root / paths["rgb_path"])
        p40, p40_valid, p40_status, raw, valid, rejected = _load_current_sweep_p40(
            episode_root / paths["radar_points_path"]
        )
        vehicle_pixels, person_pixels = int(row["vehicle_pixels"]), int(row["person_pixels"])
        if min(vehicle_pixels, person_pixels) < 0:
            raise OfflineGridContractError(f"negative density at {sample_id}")
        candidates.append(
            Candidate(
                episode_id=episode_id,
                grid_split=grid_split,
                source_position=position,
                sample_id=sample_id,
                frame_id=int(row["frame_id"]),
                timestamp=float(row["timestamp"]),
                source_paths=paths,
                camera_si=camera_si,
                camera_si_valid=True,
                camera_si_status="VALID",
                radar_p40=p40,
                radar_p40_valid=p40_valid,
                radar_p40_status=p40_status,
                current_sweep_raw_returns=raw,
                current_sweep_valid_returns=valid,
                current_sweep_rejected_returns=rejected,
                vehicle_pixels=vehicle_pixels,
                person_pixels=person_pixels,
                vehicle_present=vehicle_pixels > 0,
                person_present=person_pixels > 0,
                ego_world_pose=tuple(
                    float(row[f"anchor_{name}"])
                    for name in ("x", "y", "z", "pitch", "yaw", "roll")
                ),
            )
        )
        if progress_every and len(candidates) % progress_every == 0:
            print(
                f"[selection] {episode_id}: described {len(candidates)}/{len(executable_ids)}",
                flush=True,
            )
    if len(candidates) != len(executable_ids):
        raise OfflineGridContractError(
            f"{episode_id}: descriptor coverage {len(candidates)} != {len(executable_ids)}"
        )
    return candidates


def assign_strata(candidates: Sequence[Candidate]) -> tuple[list[Candidate], dict[str, Any]]:
    si_edges = _quantile_edges(
        [float(item.camera_si) for item in candidates if item.camera_si_valid],
        (0.25, 0.50, 0.75),
    )
    p40_edges = _quantile_edges(
        [float(item.radar_p40) for item in candidates if item.radar_p40_valid],
        (0.25, 0.50, 0.75),
    )
    vehicle_edges = _quantile_edges(
        [float(item.vehicle_pixels) for item in candidates if item.vehicle_pixels > 0],
        (1.0 / 3.0, 2.0 / 3.0),
    )
    person_edges = _quantile_edges(
        [float(item.person_pixels) for item in candidates if item.person_pixels > 0],
        (1.0 / 3.0, 2.0 / 3.0),
    )
    assigned = [
        replace(
            item,
            si_bin=_bin(item.camera_si, item.camera_si_valid, si_edges),
            p40_bin=_bin(item.radar_p40, item.radar_p40_valid, p40_edges),
            vehicle_density_bin=_density_bin(item.vehicle_pixels, vehicle_edges),
            person_density_bin=_density_bin(item.person_pixels, person_edges),
        )
        for item in candidates
    ]
    return assigned, {
        "camera_si_quartile_edges": list(si_edges),
        "radar_p40_quartile_edges": list(p40_edges),
        "vehicle_positive_pixel_tertile_edges": list(vehicle_edges),
        "person_positive_pixel_tertile_edges": list(person_edges),
    }


def proportional_stratified_select(
    candidates: Sequence[Candidate], target: int, *, seed: str = SAMPLING_SEED
) -> tuple[list[tuple[Candidate, float, float]], dict[str, Any]]:
    """Hamilton-allocate a without-replacement sample and explicit weights."""

    if target <= 0 or target > len(candidates):
        raise OfflineGridContractError(
            f"selection target {target} is incompatible with {len(candidates)} candidates"
        )
    strata: dict[tuple[int, ...], list[Candidate]] = defaultdict(list)
    for item in candidates:
        strata[item.stratum].append(item)
    population = len(candidates)
    allocation: dict[tuple[int, ...], int] = {}
    remainders: list[tuple[float, str, tuple[int, ...]]] = []
    for key, values in strata.items():
        exact = target * len(values) / population
        base = min(len(values), int(math.floor(exact)))
        allocation[key] = base
        remainders.append((exact - base, _sha_rank(seed, "allocation", key), key))
    remaining = target - sum(allocation.values())
    for _remainder, _tie, key in sorted(remainders, key=lambda row: (-row[0], row[1])):
        if remaining == 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise OfflineGridContractError("stratified allocation did not reach its target")

    selected: list[tuple[Candidate, float, float]] = []
    stratum_report: list[dict[str, Any]] = []
    for key in sorted(strata):
        values = sorted(
            strata[key],
            key=lambda item: (_sha_rank(seed, item.episode_id, key, item.sample_id), item.sample_id),
        )
        count = allocation[key]
        probability = count / len(values) if count else 0.0
        weight = len(values) / count if count else 0.0
        selected.extend((item, probability, weight) for item in values[:count])
        stratum_report.append(
            {
                "stratum": list(key),
                "population": len(values),
                "selected": count,
                "inclusion_probability": probability,
                "sampling_weight": weight,
            }
        )
    selected.sort(
        key=lambda row: (
            row[0].stratum,
            _sha_rank(seed, row[0].episode_id, row[0].stratum, row[0].sample_id),
            row[0].sample_id,
        )
    )
    if len(selected) != target or len({item.sample_id for item, _, _ in selected}) != target:
        raise OfflineGridContractError("selection cardinality/uniqueness failure")
    return selected, {
        "population": population,
        "selected": target,
        "strata": stratum_report,
        "oversampled_with_replacement": False,
        "sampling_weights_stored": True,
    }


def _candidate_record(
    root: Path, item: Candidate, rank: int, probability: float, weight: float
) -> dict[str, Any]:
    episode_root = root / "data_collection/experiments/route_b_perception_v3" / item.episode_id
    paths, hashes = _source_hashes(
        episode_root,
        {
            "experiment_id": item.episode_id,
            "sample_id": item.sample_id,
            "frame_id": str(item.frame_id),
            **dict(item.source_paths),
        },
    )
    evaluation_paths, evaluation_hashes = _evaluation_source_hashes(root, item.sample_id)
    source_binding = canonical_sha256(
        {
            "episode_id": item.episode_id,
            "sample_id": item.sample_id,
            "frame_id": item.frame_id,
            "paths": paths,
            "sha256": hashes,
            "evaluation_paths": evaluation_paths,
            "evaluation_sha256": evaluation_hashes,
        }
    )
    return {
        "selection_rank_within_split": rank,
        "episode_id": item.episode_id,
        "sample_id": item.sample_id,
        "frame_id": item.frame_id,
        "timestamp": item.timestamp,
        "grid_split": item.grid_split,
        "source_position": item.source_position,
        "source_paths": paths,
        "source_sha256": hashes,
        "evaluation_source_paths": evaluation_paths,
        "evaluation_source_sha256": evaluation_hashes,
        "source_binding_sha256": source_binding,
        "camera_si": item.camera_si,
        "camera_si_valid": item.camera_si_valid,
        "camera_si_status": item.camera_si_status,
        "radar_p40": item.radar_p40,
        "radar_p40_valid": item.radar_p40_valid,
        "radar_p40_status": item.radar_p40_status,
        "current_sweep_raw_returns": item.current_sweep_raw_returns,
        "current_sweep_valid_returns": item.current_sweep_valid_returns,
        "current_sweep_rejected_returns": item.current_sweep_rejected_returns,
        "vehicle_pixels": item.vehicle_pixels,
        "person_pixels": item.person_pixels,
        "vehicle_present": item.vehicle_present,
        "person_present": item.person_present,
        "ego_world_pose": {
            name: value
            for name, value in zip(
                ("x", "y", "z", "pitch", "yaw", "roll"), item.ego_world_pose
            )
        },
        "stratum": list(item.stratum),
        "inclusion_probability": probability,
        "sampling_weight": weight,
    }


def build_selection_manifest(
    root: Path | None = None, *, progress_every: int = 100
) -> dict[str, Any]:
    root = repository_root() if root is None else Path(root).resolve(strict=True)
    preflight = run_metadata_preflight(root)
    executable_ids, _counts, _total, _model_rows = _load_model_validation_ids(root)
    all_selected: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    global_bindings: dict[str, dict[str, str]] = {}

    for binding in EPISODES:
        episode_root = root / binding.root_relpath
        rows = _read_csv_rows(episode_root / "manifest.csv")
        episode_ids = {row["sample_id"] for row in rows}
        executable = episode_ids.intersection(executable_ids)
        described = describe_candidates(
            root=root,
            episode_id=binding.episode_id,
            grid_split=binding.grid_split,
            rows=rows,
            executable_ids=executable,
            progress_every=progress_every,
        )
        assigned, edges = assign_strata(described)
        selected, sampling = proportional_stratified_select(
            assigned, binding.selected_rows
        )
        records = [
            _candidate_record(root, item, rank, probability, weight)
            for rank, (item, probability, weight) in enumerate(selected)
        ]
        all_selected.extend(records)
        split_reports.append(
            {
                "episode_id": binding.episode_id,
                "grid_split": binding.grid_split,
                "edges": edges,
                "sampling": sampling,
                "selected_sample_id_sha256": hashlib.sha256(
                    "".join(record["sample_id"] + "\n" for record in records).encode("utf-8")
                ).hexdigest(),
            }
        )
        global_bindings[binding.episode_id] = {
            name: sha256_file(episode_root / name)
            for name in SELECTION_SOURCE_GLOBAL_FILES
        }

    if len(all_selected) != TOTAL_SELECTED_FRAMES:
        raise OfflineGridContractError("combined selection count drift")
    if len({record["sample_id"] for record in all_selected}) != TOTAL_SELECTED_FRAMES:
        raise OfflineGridContractError("combined selection has duplicate sample IDs")
    document: dict[str, Any] = {
        "schema": SELECTION_SCHEMA_ID,
        "contract_sha256": CONTRACT_SHA256,
        "preflight_binding_sha256": preflight["preflight_binding_sha256"],
        "scene_descriptor_schema_sha256": SCENE_DESCRIPTOR_SHA256,
        "descriptor_runtime_versions": {
            "numpy": str(np.__version__),
            "opencv": str(cv2.__version__),
        },
        "sampling_seed": SAMPLING_SEED,
        "stratification": dict(STRATIFICATION),
        "source_global_sha256": global_bindings,
        "split_reports": split_reports,
        "selected_frame_count": len(all_selected),
        "selected_frames": all_selected,
        "test_episode_access": "NONE_ALLOWLIST_ONLY_05_06",
        "selection_manifest_sha256": "",
    }
    document["selection_manifest_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "selection_manifest_sha256"}
    )
    return document


def validate_selection_manifest(document: Mapping[str, Any]) -> str:
    if document.get("schema") != SELECTION_SCHEMA_ID:
        raise OfflineGridContractError("wrong selection manifest schema")
    if document.get("contract_sha256") != CONTRACT_SHA256:
        raise OfflineGridContractError("selection manifest contract binding drift")
    frames = document.get("selected_frames")
    if not isinstance(frames, list) or len(frames) != TOTAL_SELECTED_FRAMES:
        raise OfflineGridContractError("selection manifest frame count drift")
    ids = [str(row.get("sample_id", "")) for row in frames]
    if len(set(ids)) != len(ids):
        raise OfflineGridContractError("selection manifest duplicate sample ID")
    by_episode = Counter(str(row.get("episode_id", "")) for row in frames)
    expected_by_episode = {item.episode_id: item.selected_rows for item in EPISODES}
    if dict(by_episode) != expected_by_episode:
        raise OfflineGridContractError("selection manifest episode/count drift")
    expected_splits = {item.episode_id: item.grid_split for item in EPISODES}
    for row in frames:
        episode_id = str(row.get("episode_id", ""))
        if row.get("grid_split") != expected_splits.get(episode_id):
            raise OfflineGridContractError("selection manifest split/episode mismatch")
        if not str(row.get("sample_id", "")).startswith(episode_id + "_"):
            raise OfflineGridContractError("selection sample identity is not episode-bound")
        if set(row.get("source_paths", {})) != set(SOURCE_PAYLOAD_FIELDS):
            raise OfflineGridContractError("selection source path inventory drift")
        if set(row.get("source_sha256", {})) != set(SOURCE_PAYLOAD_FIELDS):
            raise OfflineGridContractError("selection source hash inventory drift")
        if set(row.get("evaluation_source_paths", {})) != {
            "segmentation_gt_path", "object_ignore_mask_path"
        }:
            raise OfflineGridContractError("selection evaluation path inventory drift")
        if set(row.get("evaluation_source_sha256", {})) != {
            "segmentation_gt_path", "object_ignore_mask_path"
        }:
            raise OfflineGridContractError("selection evaluation hash inventory drift")
        source_binding = canonical_sha256(
            {
                "episode_id": episode_id,
                "sample_id": row["sample_id"],
                "frame_id": row["frame_id"],
                "paths": row["source_paths"],
                "sha256": row["source_sha256"],
                "evaluation_paths": row["evaluation_source_paths"],
                "evaluation_sha256": row["evaluation_source_sha256"],
            }
        )
        if row.get("source_binding_sha256") != source_binding:
            raise OfflineGridContractError("selection per-frame source binding drift")
    expected = canonical_sha256(
        {key: value for key, value in document.items() if key != "selection_manifest_sha256"}
    )
    if document.get("selection_manifest_sha256") != expected:
        raise OfflineGridContractError("selection manifest self-digest drift")
    return expected


def verify_selection_sources(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and exact-compare selection before any model/CUDA action.

    A self-digest proves only internal consistency.  This verification reruns
    SI/P40, episode-local bins, Hamilton allocation, SHA ranks, inclusion
    probabilities and weights from the two allowlisted raw episodes.  Thus a
    scientifically altered manifest cannot authorize itself by recomputing its
    own digest.
    """

    digest = validate_selection_manifest(document)
    root = Path(root).resolve(strict=True)
    rebuilt = build_selection_manifest(root, progress_every=0)
    if canonical_json_bytes(dict(document)) != canonical_json_bytes(rebuilt):
        raise OfflineGridContractError(
            "selection manifest differs from deterministic source rebuild"
        )
    expected_globals = document.get("source_global_sha256")
    if not isinstance(expected_globals, Mapping):
        raise OfflineGridContractError("selection global source bindings are missing")
    checked = 0
    for binding in EPISODES:
        observed = {
            name: sha256_file(root / binding.root_relpath / name)
            for name in SELECTION_SOURCE_GLOBAL_FILES
        }
        if observed != expected_globals.get(binding.episode_id):
            raise OfflineGridContractError(
                f"selection global source drift: {binding.episode_id}"
            )
    for row in document["selected_frames"]:
        episode_root = (
            root
            / "data_collection/experiments/route_b_perception_v3"
            / str(row["episode_id"])
        )
        for field_name in SOURCE_PAYLOAD_FIELDS:
            relative = _safe_relative(str(row["source_paths"][field_name]), field_name)
            observed = sha256_file(episode_root / relative)
            if observed != row["source_sha256"][field_name]:
                raise OfflineGridContractError(
                    f"selected source drift: {row['sample_id']}:{field_name}"
                )
            checked += 1
        for field_name in ("segmentation_gt_path", "object_ignore_mask_path"):
            relative = _safe_relative(
                str(row["evaluation_source_paths"][field_name]), field_name
            )
            observed = sha256_file(root / relative)
            if observed != row["evaluation_source_sha256"][field_name]:
                raise OfflineGridContractError(
                    f"selected evaluation source drift: {row['sample_id']}:{field_name}"
                )
            checked += 1
    return {
        "selection_manifest_sha256": digest,
        "selected_frames": len(document["selected_frames"]),
        "selected_payload_files_hashed": checked,
        "test_episode_access": "NONE_ALLOWLIST_ONLY_05_06",
        "deterministic_source_rebuild_exact": True,
    }


def write_json_create_only(path: Path, document: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise OfflineGridContractError(f"create-only output exists: {path}")
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)
