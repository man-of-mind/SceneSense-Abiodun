"""Read-only corrected P40 sidecar for the completed exact quality grid.

The completed grid is immutable.  Its selector counted range-valid current-
sweep returns, but accidentally passed the *unfiltered* vector to the strict
``radar_proximity_p40`` descriptor.  Route-B collection explicitly declares
that downstream reducers retain only finite returns satisfying
``0 < original_range_m <= configured_range_m``.  This module applies that
already-declared source rule and emits a separately bound, auditable sidecar;
it never rewrites the selection manifest, SQLite grid, or any source NPZ.

Import is inert.  The exact audit reads files only when
``load_exact_corrected_p40_sidecar`` is called.  The CLI writes only when an
explicit ``--output`` path is supplied, and that write is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .offline_quality_grid.contract import (
    FIT_SELECTION_COUNT,
    HELD_SCENE_SELECTION_COUNT,
    TOTAL_SELECTED_FRAMES,
    canonical_json_bytes,
    canonical_sha256,
    repository_root,
)
from .offline_quality_grid.selection import validate_selection_manifest
from .scene_descriptors import RADAR_SENSOR_RANGE_M, radar_proximity_p40

__all__ = [
    "CorrectedP40Error",
    "BundleBindingError",
    "SourceRadarError",
    "DuplicateSampleIdError",
    "UnknownSampleIdError",
    "ExactGridBundleBinding",
    "EXACT_COMPLETED_GRID_BINDING",
    "CorrectedP40Record",
    "CorrectedP40Sidecar",
    "load_exact_corrected_p40_sidecar",
    "main",
]


SCHEMA_ID = "splitfusion_corrected_p40_sidecar_v1"
EVIDENCE_CLASS = "DERIVED_FROM_HASH_VERIFIED_IMMUTABLE_ROUTE_B_SOURCE"
DEFAULT_BUNDLE_RELPATH = (
    "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
    "20260918_exact_continuous_q_grid_a1b_full"
)
ROUTE_B_ROOT_RELPATH = "data_collection/experiments/route_b_perception_v3"
IMPLEMENTATION_RELPATH = (
    "rl_agent/splitfusion_hybrid_sac_v1/corrected_p40_sidecar.py"
)
EXPECTED_SELECTION_INTERNAL_SHA256 = (
    "9a03f415edf6f5482115bbec7644c9e6174d13dad9bd1505576e3e6ef93df905"
)
REPAIR_CLASS_FILTERED = "CORRECTED_BY_DECLARED_ROUTE_B_RANGE_FILTER"
REPAIR_CLASS_AUDIT = "ORIGINAL_VALID_VALUE_EXACTLY_REDERIVED"
REPAIR_AUTHORITY = (
    "Route-B collection metadata declares that a downstream reducer retains "
    "only finite returns with 0 < original_range_m <= configured range; the "
    "offline selector computed that mask and its counts but omitted applying "
    "the mask before the strict P40 call"
)
EXPECTED_RAW_RANGE_NOTE = (
    "CARLA occasionally reports original_range_m beyond the configured range "
    "(measured 0.043% of returns, up to 172.7 m at a 120 m setting). Raw "
    "provenance is saved unmodified and unclamped; a downstream reducer should "
    "treat only finite returns with 0 < original_range_m <= range_m as range-valid."
)
EXPECTED_EPISODE_METADATA_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "canonical_v3_05_val_30_30_s601_tm1601": (
            "ba22facb550378dbd6cee8367d428922ed97a539354b0845c97598e06a7e5928"
        ),
        "canonical_v3_06_val_50_50_s602_tm1602": (
            "0426b276b1c8dc1b542fc2209ea81273328c88c872bf3825443f53b882045569"
        ),
    }
)


class CorrectedP40Error(RuntimeError):
    """Base class for fail-closed sidecar errors."""


class BundleBindingError(CorrectedP40Error):
    """The immutable completed-grid bundle does not match its exact binding."""


class SourceRadarError(CorrectedP40Error):
    """A selected source NPZ cannot support the declared correction."""


class DuplicateSampleIdError(CorrectedP40Error):
    """The selection or sidecar contains a duplicate sample identity."""


class UnknownSampleIdError(KeyError, CorrectedP40Error):
    """A sample is outside the exact sidecar inventory."""


@dataclass(frozen=True, slots=True)
class ExactGridBundleBinding:
    """External byte bindings for one immutable completed quality-grid bundle."""

    run_manifest_file_sha256: str
    selection_manifest_file_sha256: str
    completion_file_sha256: str
    quality_database_file_sha256: str
    selection_manifest_internal_sha256: str
    expected_rows: int


EXACT_COMPLETED_GRID_BINDING = ExactGridBundleBinding(
    run_manifest_file_sha256=(
        "8869d085e585bd4cb4d8231abdfea4b358095578e85a3cbfb811bf073bfc9b36"
    ),
    selection_manifest_file_sha256=(
        "a25b93d35a5bb70e062d275daf8767ce73a5ab7ba34079ab6a468bda5ffe4c4d"
    ),
    completion_file_sha256=(
        "de73546968bd6d632a4b48120c4980936111896206adeeafddd05efe79324c44"
    ),
    quality_database_file_sha256=(
        "8b2c0754f135873cc339dc62cc7e0e09e02e3a85426d0027c2cd38b6a2a68c88"
    ),
    selection_manifest_internal_sha256=EXPECTED_SELECTION_INTERNAL_SHA256,
    expected_rows=101_376,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise BundleBindingError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BundleBindingError(f"cannot parse strict {label}: {path}") from exc
    if type(value) is not dict:
        raise BundleBindingError(f"{label} must be a JSON object")
    return value


def _validate_radar_metadata_document(
    document: Mapping[str, Any], episode_id: str
) -> None:
    if document.get("experiment_id") != episode_id:
        raise BundleBindingError(f"Route-B metadata episode identity drift: {episode_id}")
    radar = document.get("radar")
    if not isinstance(radar, Mapping):
        raise BundleBindingError(f"Route-B radar metadata is missing: {episode_id}")
    configured = radar.get("configured_attributes")
    if not isinstance(configured, Mapping):
        raise BundleBindingError(
            f"Route-B configured radar attributes are missing: {episode_id}"
        )
    range_m = radar.get("range_m")
    configured_range = configured.get("range")
    if isinstance(range_m, bool) or isinstance(configured_range, bool):
        raise BundleBindingError(f"Route-B radar range has boolean type: {episode_id}")
    try:
        numeric_range = float(range_m)
        numeric_configured_range = float(configured_range)
    except (TypeError, ValueError) as exc:
        raise BundleBindingError(f"Route-B radar range is not numeric: {episode_id}") from exc
    if (
        not math.isfinite(numeric_range)
        or not math.isfinite(numeric_configured_range)
        or numeric_range != RADAR_SENSOR_RANGE_M
        or numeric_configured_range != RADAR_SENSOR_RANGE_M
    ):
        raise BundleBindingError(
            f"Route-B configured radar range is not {RADAR_SENSOR_RANGE_M:g} m: "
            f"{episode_id}"
        )
    if radar.get("raw_range_note") != EXPECTED_RAW_RANGE_NOTE:
        raise BundleBindingError(
            f"Route-B downstream range-filter authority drift: {episode_id}"
        )


def _verify_repair_authority_metadata(
    root: Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    globals_by_episode = selection.get("source_global_sha256")
    if not isinstance(globals_by_episode, Mapping):
        raise BundleBindingError("selection global source binding is missing")
    episodes: dict[str, Any] = {}
    for episode_id, expected_sha256 in EXPECTED_EPISODE_METADATA_SHA256.items():
        selected_globals = globals_by_episode.get(episode_id)
        if not isinstance(selected_globals, Mapping):
            raise BundleBindingError(
                f"selection lacks global binding for episode: {episode_id}"
            )
        if selected_globals.get("metadata.json") != expected_sha256:
            raise BundleBindingError(
                f"selection Route-B metadata binding drift: {episode_id}"
            )
        relative = f"{ROUTE_B_ROOT_RELPATH}/{episode_id}/metadata.json"
        path = root / relative
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise BundleBindingError(
                f"physical Route-B metadata hash drift: {episode_id}"
            )
        document = _strict_json_object(path, f"Route-B metadata for {episode_id}")
        _validate_radar_metadata_document(document, episode_id)
        episodes[episode_id] = {
            "metadata_relative_path": relative,
            "metadata_sha256": observed_sha256,
            "configured_range_m": RADAR_SENSOR_RANGE_M,
            "raw_range_note": EXPECTED_RAW_RANGE_NOTE,
        }
    return {
        "rule": "finite AND 0 < original_range_m <= configured range_m",
        "episodes": episodes,
    }


def _safe_source_path(
    root: Path, episode_id: str, relative_text: str
) -> Path:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SourceRadarError(f"unsafe radar source path: {relative_text!r}")
    try:
        episode_root = (
            root / ROUTE_B_ROOT_RELPATH / episode_id
        ).resolve(strict=True)
        candidate = (episode_root / Path(*relative.parts)).resolve(strict=True)
    except OSError as exc:
        raise SourceRadarError(
            f"selected radar source is missing/unresolvable: {episode_id}:{relative_text}"
        ) from exc
    try:
        candidate.relative_to(episode_root)
    except ValueError as exc:
        raise SourceRadarError(
            f"radar source escapes its episode root: {relative_text!r}"
        ) from exc
    if not candidate.is_file():
        raise SourceRadarError(f"radar source is not a regular file: {candidate}")
    return candidate


@dataclass(frozen=True, slots=True)
class CorrectedP40Record:
    """One selected frame's original status and corrected descriptor."""

    sample_id: str
    episode_id: str
    grid_split: str
    frame_id: int
    radar_source_path: str
    radar_source_sha256: str
    original_p40: Optional[float]
    original_p40_valid: bool
    original_p40_status: str
    original_raw_returns: int
    original_valid_returns: int
    original_rejected_returns: int
    current_sweep_raw_returns: int
    current_sweep_valid_returns: int
    current_sweep_rejected_returns: int
    current_sweep_rejected_fraction: float
    corrected_p40: float
    correction_classification: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "grid_split": self.grid_split,
            "frame_id": self.frame_id,
            "radar_source_path": self.radar_source_path,
            "radar_source_sha256": self.radar_source_sha256,
            "original_p40": self.original_p40,
            "original_p40_valid": self.original_p40_valid,
            "original_p40_status": self.original_p40_status,
            "original_raw_returns": self.original_raw_returns,
            "original_valid_returns": self.original_valid_returns,
            "original_rejected_returns": self.original_rejected_returns,
            "current_sweep_raw_returns": self.current_sweep_raw_returns,
            "current_sweep_valid_returns": self.current_sweep_valid_returns,
            "current_sweep_rejected_returns": self.current_sweep_rejected_returns,
            "current_sweep_rejected_fraction": self.current_sweep_rejected_fraction,
            "corrected_p40": self.corrected_p40,
            "correction_classification": self.correction_classification,
        }


def _derive_record(root: Path, row: Mapping[str, Any]) -> CorrectedP40Record:
    try:
        sample_id = str(row["sample_id"])
        episode_id = str(row["episode_id"])
        grid_split = str(row["grid_split"])
        frame_id = int(row["frame_id"])
        source_paths = row["source_paths"]
        source_hashes = row["source_sha256"]
        relative = str(source_paths["radar_points_path"])
        expected_source_sha256 = str(source_hashes["radar_points_path"])
        original_valid = row["radar_p40_valid"]
        original_status = str(row["radar_p40_status"])
        original_p40_raw = row["radar_p40"]
        old_raw = int(row["current_sweep_raw_returns"])
        old_valid = int(row["current_sweep_valid_returns"])
        old_rejected = int(row["current_sweep_rejected_returns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceRadarError("malformed selected-frame P40/source record") from exc
    if not sample_id or not episode_id or grid_split not in {"fit", "held_scene"}:
        raise SourceRadarError(f"invalid selected-frame identity: {sample_id!r}")
    if type(original_valid) is not bool:
        raise SourceRadarError(f"non-boolean original P40 validity: {sample_id}")
    if min(old_raw, old_valid, old_rejected) < 0 or old_valid + old_rejected != old_raw:
        raise SourceRadarError(f"incoherent original P40 counts: {sample_id}")

    path = _safe_source_path(root, episode_id, relative)
    observed_source_sha256 = _sha256_file(path)
    if observed_source_sha256 != expected_source_sha256:
        raise SourceRadarError(
            f"selected radar source hash drift: {sample_id}: "
            f"expected {expected_source_sha256}, observed {observed_source_sha256}"
        )

    try:
        with np.load(path, allow_pickle=False) as payload:
            if "sweep_offset" not in payload.files or "original_range_m" not in payload.files:
                raise SourceRadarError(
                    f"radar source lacks sweep_offset/original_range_m: {sample_id}"
                )
            offset = np.asarray(payload["sweep_offset"])
            ranges = np.asarray(payload["original_range_m"])
    except SourceRadarError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SourceRadarError(f"cannot read selected radar source: {sample_id}") from exc
    if offset.ndim != 1 or ranges.ndim != 1 or offset.shape != ranges.shape:
        raise SourceRadarError(f"malformed radar provenance arrays: {sample_id}")
    if not np.issubdtype(offset.dtype, np.integer):
        raise SourceRadarError(f"sweep_offset is not integer typed: {sample_id}")
    if ranges.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise SourceRadarError(f"original_range_m is not float32/float64: {sample_id}")

    current = np.ascontiguousarray(ranges[offset == 0])
    raw_count = int(current.size)
    if raw_count == 0:
        raise SourceRadarError(
            f"current 100-ms radar sweep is missing/empty; no zero imputation: {sample_id}"
        )
    valid_mask = (
        np.isfinite(current)
        & (current > 0.0)
        & (current <= RADAR_SENSOR_RANGE_M)
    )
    filtered = np.ascontiguousarray(current[valid_mask])
    valid_count = int(filtered.size)
    rejected_count = raw_count - valid_count
    if valid_count == 0:
        raise SourceRadarError(
            f"current radar sweep has no range-valid returns; no zero imputation: {sample_id}"
        )
    if (raw_count, valid_count, rejected_count) != (old_raw, old_valid, old_rejected):
        raise SourceRadarError(
            f"selection P40 count drift for {sample_id}: "
            f"manifest={(old_raw, old_valid, old_rejected)}, "
            f"source={(raw_count, valid_count, rejected_count)}"
        )

    corrected = radar_proximity_p40(filtered)
    original_p40: Optional[float]
    if original_valid:
        if original_status != "VALID" or isinstance(original_p40_raw, bool):
            raise SourceRadarError(f"incoherent valid original P40: {sample_id}")
        try:
            original_p40 = float(original_p40_raw)
        except (TypeError, ValueError) as exc:
            raise SourceRadarError(f"invalid original P40 value: {sample_id}") from exc
        if not math.isfinite(original_p40) or not 0.0 <= original_p40 <= 1.0:
            raise SourceRadarError(f"out-of-range original P40 value: {sample_id}")
        if rejected_count != 0 or corrected != original_p40:
            raise SourceRadarError(
                f"original-valid P40 does not exactly rederive: {sample_id}"
            )
        classification = REPAIR_CLASS_AUDIT
    else:
        if original_p40_raw is not None:
            raise SourceRadarError(f"invalid original P40 must be null: {sample_id}")
        if original_status != "InvalidRadarRangesError" or rejected_count == 0:
            raise SourceRadarError(
                f"original invalidity is not explained by filtered ranges: {sample_id}"
            )
        original_p40 = None
        classification = REPAIR_CLASS_FILTERED

    return CorrectedP40Record(
        sample_id=sample_id,
        episode_id=episode_id,
        grid_split=grid_split,
        frame_id=frame_id,
        radar_source_path=relative,
        radar_source_sha256=observed_source_sha256,
        original_p40=original_p40,
        original_p40_valid=original_valid,
        original_p40_status=original_status,
        original_raw_returns=old_raw,
        original_valid_returns=old_valid,
        original_rejected_returns=old_rejected,
        current_sweep_raw_returns=raw_count,
        current_sweep_valid_returns=valid_count,
        current_sweep_rejected_returns=rejected_count,
        current_sweep_rejected_fraction=rejected_count / raw_count,
        corrected_p40=corrected,
        correction_classification=classification,
    )


def _derive_records(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_total: int,
    expected_split_counts: Mapping[str, int],
) -> tuple[CorrectedP40Record, ...]:
    if len(rows) != expected_total:
        raise SourceRadarError(
            f"selected-frame count drift: {len(rows)} != {expected_total}"
        )
    records: list[CorrectedP40Record] = []
    seen: set[str] = set()
    split_counts = {name: 0 for name in expected_split_counts}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in seen:
            raise DuplicateSampleIdError(f"duplicate selected sample ID: {sample_id}")
        seen.add(sample_id)
        record = _derive_record(root, row)
        if record.grid_split not in split_counts:
            raise SourceRadarError(f"foreign grid split: {record.grid_split}")
        split_counts[record.grid_split] += 1
        records.append(record)
    if split_counts != dict(expected_split_counts):
        raise SourceRadarError(
            f"grid-split count drift: {split_counts} != {dict(expected_split_counts)}"
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class CorrectedP40Sidecar:
    """Immutable provider and audit report for all corrected selected frames.

    ``lookup(sample_id)`` is the narrow response-surface provider protocol and
    returns the corrected float.  ``record(sample_id)`` retains the richer
    audit evidence.
    """

    records: tuple[CorrectedP40Record, ...]
    source_binding: Mapping[str, Any]
    binding_sha256: str
    _by_sample_id: Mapping[str, CorrectedP40Record] = field(
        repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        records: Sequence[CorrectedP40Record],
        source_binding: Mapping[str, Any],
    ) -> "CorrectedP40Sidecar":
        frozen_records = tuple(records)
        by_id: dict[str, CorrectedP40Record] = {}
        for record in frozen_records:
            if record.sample_id in by_id:
                raise DuplicateSampleIdError(
                    f"duplicate corrected sample ID: {record.sample_id}"
                )
            by_id[record.sample_id] = record
        frozen_binding = _deep_freeze(dict(source_binding))
        binding_document = {
            "schema": SCHEMA_ID,
            "evidence_class": EVIDENCE_CLASS,
            "repair_authority": REPAIR_AUTHORITY,
            "source_binding": _thaw(frozen_binding),
            "records": [record.to_canonical_dict() for record in frozen_records],
        }
        digest = canonical_sha256(binding_document)
        return cls(
            records=frozen_records,
            source_binding=frozen_binding,
            binding_sha256=digest,
            _by_sample_id=MappingProxyType(by_id),
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    def lookup(self, sample_id: str) -> float:
        return self.record(sample_id).corrected_p40

    def record(self, sample_id: str) -> CorrectedP40Record:
        try:
            return self._by_sample_id[str(sample_id)]
        except KeyError as exc:
            raise UnknownSampleIdError(str(sample_id)) from exc

    def canonical_report(self) -> dict[str, Any]:
        total_raw = sum(row.current_sweep_raw_returns for row in self.records)
        total_valid = sum(row.current_sweep_valid_returns for row in self.records)
        total_rejected = sum(row.current_sweep_rejected_returns for row in self.records)
        split_counts = {
            split: sum(row.grid_split == split for row in self.records)
            for split in ("fit", "held_scene")
        }
        classifications = {
            name: sum(row.correction_classification == name for row in self.records)
            for name in (REPAIR_CLASS_FILTERED, REPAIR_CLASS_AUDIT)
        }
        document = {
            "schema": SCHEMA_ID,
            "evidence_class": EVIDENCE_CLASS,
            "status": "COMPLETE",
            "repair_scope": (
                "READ_ONLY_DERIVED_SIDECAR; ORIGINAL_GRID_SELECTION_AND_SOURCES_UNCHANGED"
            ),
            "repair_authority": REPAIR_AUTHORITY,
            "no_zero_imputation": True,
            "source_binding": _thaw(self.source_binding),
            "inventory": {
                "records": len(self.records),
                "valid_corrected_p40": len(self.records),
                "split_counts": split_counts,
                "classification_counts": classifications,
                "current_sweep_raw_returns": total_raw,
                "current_sweep_valid_returns": total_valid,
                "current_sweep_rejected_returns": total_rejected,
                "current_sweep_rejected_fraction": total_rejected / total_raw,
            },
            "records": [record.to_canonical_dict() for record in self.records],
            "binding_sha256": self.binding_sha256,
        }
        document["canonical_report_sha256"] = canonical_sha256(document)
        return document

    def canonical_report_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_report()) + b"\n"


def _verify_exact_bundle(
    bundle: Path,
    binding: ExactGridBundleBinding,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = Path(bundle).resolve(strict=True)
    paths = {
        "run_manifest": bundle / "run_manifest.json",
        "selection_manifest": bundle / "selection_manifest.json",
        "completion": bundle / "COMPLETE.json",
        "quality_database": bundle / "quality_rows.sqlite3",
    }
    expected = {
        "run_manifest": binding.run_manifest_file_sha256,
        "selection_manifest": binding.selection_manifest_file_sha256,
        "completion": binding.completion_file_sha256,
        "quality_database": binding.quality_database_file_sha256,
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        differences = {
            name: {"expected": expected[name], "observed": observed[name]}
            for name in expected
            if expected[name] != observed[name]
        }
        raise BundleBindingError(f"completed-grid byte binding drift: {differences}")

    run_manifest = _strict_json_object(paths["run_manifest"], "run manifest")
    selection = _strict_json_object(paths["selection_manifest"], "selection manifest")
    completion = _strict_json_object(paths["completion"], "completion record")
    try:
        internal_selection_sha = validate_selection_manifest(selection)
    except Exception as exc:
        raise BundleBindingError("selection manifest semantic validation failed") from exc
    if internal_selection_sha != binding.selection_manifest_internal_sha256:
        raise BundleBindingError("selection manifest internal binding drift")
    if run_manifest.get("execution_status") != "COMPLETE":
        raise BundleBindingError("run manifest is not COMPLETE")
    if int(run_manifest.get("expected_rows", -1)) != binding.expected_rows:
        raise BundleBindingError("run manifest expected-row count drift")
    artifacts = run_manifest.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise BundleBindingError("run manifest artifact binding is missing")
    if artifacts.get("selection_manifest.json") != expected["selection_manifest"]:
        raise BundleBindingError("run manifest selection binding drift")
    if artifacts.get("quality_rows.sqlite3") != expected["quality_database"]:
        raise BundleBindingError("run manifest database binding drift")
    if completion.get("status") != "COMPLETE":
        raise BundleBindingError("completion status is not COMPLETE")
    if int(completion.get("rows", -1)) != binding.expected_rows:
        raise BundleBindingError("completion row count drift")
    if completion.get("run_manifest_file_sha256") != expected["run_manifest"]:
        raise BundleBindingError("completion/run-manifest cross-reference drift")
    if int(selection.get("selected_frame_count", -1)) != TOTAL_SELECTED_FRAMES:
        raise BundleBindingError("selection selected-frame count drift")
    return run_manifest, selection, completion


def load_exact_corrected_p40_sidecar(
    *,
    root: Optional[Path] = None,
    bundle: Optional[Path] = None,
) -> CorrectedP40Sidecar:
    """Audit all 768 selected NPZs and build the exact immutable sidecar."""

    resolved_root = repository_root() if root is None else Path(root).resolve(strict=True)
    resolved_bundle = (
        resolved_root / DEFAULT_BUNDLE_RELPATH
        if bundle is None
        else Path(bundle)
    ).resolve(strict=True)
    expected_bundle = (resolved_root / DEFAULT_BUNDLE_RELPATH).resolve(strict=True)
    if resolved_bundle != expected_bundle:
        raise BundleBindingError(
            "exact P40 repair accepts only the registered completed bundle location: "
            f"{expected_bundle}"
        )
    _run, selection, _complete = _verify_exact_bundle(
        resolved_bundle, EXACT_COMPLETED_GRID_BINDING
    )
    metadata_binding = _verify_repair_authority_metadata(resolved_root, selection)
    rows = selection.get("selected_frames")
    if not isinstance(rows, list):
        raise BundleBindingError("selection selected_frames is not a list")
    records = _derive_records(
        resolved_root,
        rows,
        expected_total=TOTAL_SELECTED_FRAMES,
        expected_split_counts={
            "fit": FIT_SELECTION_COUNT,
            "held_scene": HELD_SCENE_SELECTION_COUNT,
        },
    )
    implementation_path = Path(__file__).resolve(strict=True)
    try:
        observed_implementation_relpath = str(
            implementation_path.relative_to(resolved_root)
        )
    except ValueError as exc:
        raise BundleBindingError(
            "corrected-P40 implementation is outside the repository root"
        ) from exc
    if observed_implementation_relpath != IMPLEMENTATION_RELPATH:
        raise BundleBindingError(
            "corrected-P40 implementation path drift: "
            f"{observed_implementation_relpath!r}"
        )
    source_binding = {
        "bundle_relative_path": DEFAULT_BUNDLE_RELPATH,
        "run_manifest_file_sha256": EXACT_COMPLETED_GRID_BINDING.run_manifest_file_sha256,
        "selection_manifest_file_sha256": (
            EXACT_COMPLETED_GRID_BINDING.selection_manifest_file_sha256
        ),
        "selection_manifest_internal_sha256": (
            EXACT_COMPLETED_GRID_BINDING.selection_manifest_internal_sha256
        ),
        "completion_file_sha256": EXACT_COMPLETED_GRID_BINDING.completion_file_sha256,
        "quality_database_file_sha256": (
            EXACT_COMPLETED_GRID_BINDING.quality_database_file_sha256
        ),
        "radar_sensor_range_m": RADAR_SENSOR_RANGE_M,
        "repair_implementation_relative_path": observed_implementation_relpath,
        "repair_implementation_sha256": _sha256_file(implementation_path),
        "repair_authority_metadata": metadata_binding,
        "selected_npz_files_hash_verified": len(records),
    }
    return CorrectedP40Sidecar.create(records, source_binding)


def _protected_output_roots(root: Path) -> tuple[Path, Path]:
    resolved_root = Path(root).resolve(strict=True)
    try:
        bundle = (resolved_root / DEFAULT_BUNDLE_RELPATH).resolve(strict=True)
        route_b = (resolved_root / ROUTE_B_ROOT_RELPATH).resolve(strict=True)
    except OSError as exc:
        raise CorrectedP40Error(
            "cannot resolve immutable evidence roots before output validation"
        ) from exc
    return bundle, route_b


def _validate_output_path(path: Path, root: Path) -> Path:
    """Resolve symlinks and reject any write into immutable evidence trees.

    This runs before creating a parent directory.  ``strict=False`` resolves
    every existing symlink component while allowing a new final path.  The
    caller repeats it after parent creation to narrow the remaining race.
    """

    candidate = Path(path).expanduser().resolve(strict=False)
    for protected in _protected_output_roots(root):
        if candidate == protected or candidate.is_relative_to(protected):
            raise CorrectedP40Error(
                f"output is inside immutable evidence tree {protected}: {candidate}"
            )
    return candidate


def _write_create_only(path: Path, payload: bytes, *, root: Path) -> str:
    path = _validate_output_path(path, root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve again now that the parent exists, catching a pre-existing
    # symlinked parent before opening either final or partial output.
    path = _validate_output_path(path, root)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise CorrectedP40Error(f"create-only output already exists: {path}")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # ``os.replace`` would overwrite a file created between the existence
        # check and rename.  A same-directory hard link is atomic and refuses
        # an existing destination, preserving the create-only contract even
        # under a concurrent writer.
        os.link(partial, path)
        partial.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return _sha256_file(path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional create-only canonical JSON output; omitted means stdout only",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    resolved_root = (
        repository_root()
        if args.root is None
        else Path(args.root).resolve(strict=True)
    )
    sidecar = load_exact_corrected_p40_sidecar(
        root=resolved_root, bundle=args.bundle
    )
    payload = sidecar.canonical_report_bytes()
    if args.output is None:
        print(payload.decode("utf-8"), end="")
    else:
        digest = _write_create_only(args.output, payload, root=resolved_root)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "output": str(args.output),
                    "output_sha256": digest,
                    "binding_sha256": sidecar.binding_sha256,
                    "records": len(sidecar.records),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
