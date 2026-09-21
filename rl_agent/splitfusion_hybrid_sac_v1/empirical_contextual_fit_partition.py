"""Isolated, reward-blind fit partition for the empirical contextual pilot.

This contract partitions only the 476 D1-eligible *fit* scenes and the 399
joint-valid radio-calibration rows.  It never queries or represents a quality
held-scene database row and exposes no held-scene API.  Membership depends only on
pre-action temporal block identities; no quality, payload, latency, delivery,
or reward scalar is inspected.

The resulting fit-validation split is reward-held, but it is deliberately not
covariate-normalization-held: D1 preregistered its unsupervised SI/SNR scaling
from all 512 fit covariates and all 399 valid calibration rows before this
partition was defined.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .action_contract import CATALOG_SHA256
from .empirical_contextual_contract import (
    MODELED_SMOKE_SUPPORT_SHA256,
    PILOT_UTILITY_SPEC_SHA256,
    PROFILE_ORDER_RNG_CONTRACT_SHA256,
    SURFACE_QUALIFICATION_REPORT_SHA256,
    EmpiricalPilotBindingV1,
)
from .empirical_contextual_environment import (
    _implementation_bundle_sha256,
    build_registered_freshness_policy,
    build_registered_normalization_spec,
)
from .payload_network_surrogate import build_payload_network_surrogate
from .empirical_quality_surface import (
    BUNDLE_RELATIVE_PATH,
    DATABASE_FILE_SHA256,
    REWARD_SPEC_FILE_SHA256,
    SELECTION_FILE_SHA256,
)
from .empirical_radio_context import (
    CALIBRATION_SHA256,
    OaiRadioCalibrationStoreV1,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_ORDER,
    assert_policy_features_exclude_forbidden_fields,
)
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "FIT_PARTITION_SCHEMA",
    "REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256",
    "FitPartitionError",
    "EmpiricalFitPartitionV1",
    "RadioBlockInventoryV1",
    "RadioFitAssignmentV1",
    "SceneBlockInventoryV1",
    "SceneFitAssignmentV1",
    "load_registered_empirical_fit_partition",
]


FIT_PARTITION_SCHEMA = "splitfusion.empirical_fit_partition.v1"
TRAIN_SPLIT = "train"
FIT_VALIDATION_SPLIT = "fit_validation"
SCENE_HASH_DOMAIN = "splitfusion.empirical_fit_partition.v1"
SCENE_VALIDATION_MODULUS = 5
SCENE_VALIDATION_RESIDUE = 0
RADIO_BLOCK_STEPS = 10
RADIO_VALIDATION_MODULUS = 5
RADIO_VALIDATION_RESIDUE = 0

# Exact D1 identities this isolated contract consumes but does not rederive.
D1_CORRECTED_P40_BINDING_SHA256 = (
    "7f52e0a315c7a1276b179b2e17e7149d6162d25cfcfcc584925965455f4ce7fa"
)
D1_QUALITY_SURFACE_BINDING_SHA256 = (
    "e3741b29ef657a0dc0d6616816dbf3b746c60f7a87e4897d8a3bb4e9706766a7"
)
D1_ELIGIBLE_FIT_CONTEXT_INDEX_SHA256 = (
    "b93fa95e1e742a2618854f5bed394354b1af0179513b8d17322bc46c5789a812"
)


class FitPartitionError(ValueError):
    """Partition evidence, inventory, or registered identity drifted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FitPartitionError(f"{name} must be a lowercase SHA-256")
    return value


def _scene_split(temporal_block: int) -> str:
    domain = f"{SCENE_HASH_DOMAIN}:{temporal_block}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(domain).digest(), "big")
    return (
        FIT_VALIDATION_SPLIT
        if value % SCENE_VALIDATION_MODULUS == SCENE_VALIDATION_RESIDUE
        else TRAIN_SPLIT
    )


def _radio_split(trace_step_block: int) -> str:
    return (
        FIT_VALIDATION_SPLIT
        if trace_step_block % RADIO_VALIDATION_MODULUS
        == RADIO_VALIDATION_RESIDUE
        else TRAIN_SPLIT
    )


@dataclass(frozen=True, slots=True)
class SceneFitAssignmentV1:
    sample_id: str
    episode_id: str
    frame_id: int
    temporal_block: int
    split: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.episode_id:
            raise FitPartitionError("scene assignment identities must be non-empty")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise FitPartitionError("scene frame_id must be non-negative")
        if self.temporal_block != self.frame_id // 100:
            raise FitPartitionError("scene temporal block is not frame_id // 100")
        if self.split != _scene_split(self.temporal_block):
            raise FitPartitionError("scene assignment contradicts hash rule")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "sample_id": self.sample_id,
            "split": self.split,
            "temporal_block": self.temporal_block,
        }


@dataclass(frozen=True, slots=True)
class RadioFitAssignmentV1:
    csv_row_number: int
    network_profile: str
    trace_id: str
    trace_step_index: int
    trace_step_block: int
    row_sha256: str
    split: str

    def __post_init__(self) -> None:
        if type(self.csv_row_number) is not int or self.csv_row_number < 2:
            raise FitPartitionError("radio CSV row number must include header offset")
        if not self.network_profile or not self.trace_id:
            raise FitPartitionError("radio assignment identities must be non-empty")
        if type(self.trace_step_index) is not int or self.trace_step_index < 0:
            raise FitPartitionError("radio trace step must be non-negative")
        if self.trace_step_block != self.trace_step_index // RADIO_BLOCK_STEPS:
            raise FitPartitionError("radio block is not trace_step_index // 10")
        if self.split != _radio_split(self.trace_step_block):
            raise FitPartitionError("radio assignment contradicts block rule")
        _require_sha256(self.row_sha256, "radio row_sha256")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "csv_row_number": self.csv_row_number,
            "network_profile": self.network_profile,
            "row_sha256": self.row_sha256,
            "split": self.split,
            "trace_id": self.trace_id,
            "trace_step_block": self.trace_step_block,
            "trace_step_index": self.trace_step_index,
        }


@dataclass(frozen=True, slots=True)
class SceneBlockInventoryV1:
    temporal_block: int
    split: str
    sample_ids: Tuple[str, ...]

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.sample_ids),
            "sample_ids": list(self.sample_ids),
            "split": self.split,
            "temporal_block": self.temporal_block,
        }


@dataclass(frozen=True, slots=True)
class RadioBlockInventoryV1:
    network_profile: str
    trace_id: str
    trace_step_block: int
    split: str
    csv_row_numbers: Tuple[int, ...]

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.csv_row_numbers),
            "csv_row_numbers": list(self.csv_row_numbers),
            "network_profile": self.network_profile,
            "split": self.split,
            "trace_id": self.trace_id,
            "trace_step_block": self.trace_step_block,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalFitPartitionV1:
    scene_assignments: Tuple[SceneFitAssignmentV1, ...]
    radio_assignments: Tuple[RadioFitAssignmentV1, ...]
    scene_blocks: Tuple[SceneBlockInventoryV1, ...]
    radio_blocks: Tuple[RadioBlockInventoryV1, ...]
    source_bindings: Mapping[str, str]
    policy_feature_order: Tuple[str, ...]
    disclosures: Tuple[str, ...]
    schema: str = FIT_PARTITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FIT_PARTITION_SCHEMA:
            raise FitPartitionError("fit partition schema drift")
        object.__setattr__(
            self,
            "source_bindings",
            MappingProxyType(dict(self.source_bindings)),
        )
        for name, digest in self.source_bindings.items():
            _require_sha256(digest, f"source_bindings[{name!r}]")
        if self.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise FitPartitionError("D1 policy feature order drift")
        assert_policy_features_exclude_forbidden_fields()
        self._validate_inventory()

    def _validate_inventory(self) -> None:
        if len(self.scene_assignments) != 476:
            raise FitPartitionError("scene assignment count must be 476")
        if len(self.radio_assignments) != 399:
            raise FitPartitionError("radio assignment count must be 399")
        sample_ids = tuple(row.sample_id for row in self.scene_assignments)
        if len(set(sample_ids)) != len(sample_ids):
            raise FitPartitionError("scene assignment duplicated a sample")
        radio_rows = tuple(row.csv_row_number for row in self.radio_assignments)
        if len(set(radio_rows)) != len(radio_rows):
            raise FitPartitionError("radio assignment duplicated a CSV row")
        scene_counts = {
            split: sum(row.split == split for row in self.scene_assignments)
            for split in (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
        }
        radio_counts = {
            split: sum(row.split == split for row in self.radio_assignments)
            for split in (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
        }
        if scene_counts != {TRAIN_SPLIT: 391, FIT_VALIDATION_SPLIT: 85}:
            raise FitPartitionError(f"scene split count drift: {scene_counts}")
        if radio_counts != {TRAIN_SPLIT: 319, FIT_VALIDATION_SPLIT: 80}:
            raise FitPartitionError(f"radio split count drift: {radio_counts}")

        scene_block_split: Dict[int, str] = {}
        for row in self.scene_assignments:
            prior = scene_block_split.setdefault(row.temporal_block, row.split)
            if prior != row.split:
                raise FitPartitionError("scene temporal block crosses splits")
        radio_block_split: Dict[Tuple[str, str, int], str] = {}
        for row in self.radio_assignments:
            key = (row.network_profile, row.trace_id, row.trace_step_block)
            prior = radio_block_split.setdefault(key, row.split)
            if prior != row.split:
                raise FitPartitionError("radio trace-local block crosses splits")

        expected_scene_blocks = {
            (row.temporal_block, row.split): tuple(
                sorted(
                    item.sample_id
                    for item in self.scene_assignments
                    if item.temporal_block == row.temporal_block
                )
            )
            for row in self.scene_assignments
        }
        observed_scene_blocks = {
            (row.temporal_block, row.split): row.sample_ids
            for row in self.scene_blocks
        }
        if observed_scene_blocks != expected_scene_blocks:
            raise FitPartitionError("scene block inventory does not close")

        expected_radio_blocks = {
            (row.network_profile, row.trace_id, row.trace_step_block, row.split): tuple(
                sorted(
                    item.csv_row_number
                    for item in self.radio_assignments
                    if (
                        item.network_profile,
                        item.trace_id,
                        item.trace_step_block,
                    )
                    == (row.network_profile, row.trace_id, row.trace_step_block)
                )
            )
            for row in self.radio_assignments
        }
        observed_radio_blocks = {
            (row.network_profile, row.trace_id, row.trace_step_block, row.split): (
                row.csv_row_numbers
            )
            for row in self.radio_blocks
        }
        if observed_radio_blocks != expected_radio_blocks:
            raise FitPartitionError("radio block inventory does not close")

    @property
    def scene_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: sum(row.split == split for row in self.scene_assignments)
                for split in (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
            }
        )

    @property
    def radio_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: sum(row.split == split for row in self.radio_assignments)
                for split in (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
            }
        )

    @property
    def radio_profile_counts(self) -> Mapping[str, Mapping[str, int]]:
        profiles = sorted({row.network_profile for row in self.radio_assignments})
        return MappingProxyType(
            {
                profile: MappingProxyType(
                    {
                        split: sum(
                            row.network_profile == profile and row.split == split
                            for row in self.radio_assignments
                        )
                        for split in (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
                    }
                )
                for profile in profiles
            }
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "disclosures": list(self.disclosures),
            "policy_feature_order": list(self.policy_feature_order),
            "radio_assignments": [
                row.to_canonical_dict() for row in self.radio_assignments
            ],
            "radio_blocks": [row.to_canonical_dict() for row in self.radio_blocks],
            "radio_counts": dict(self.radio_counts),
            "radio_profile_counts": {
                profile: dict(counts)
                for profile, counts in self.radio_profile_counts.items()
            },
            "radio_rule": {
                "block": "trace_step_index // 10 within (profile, trace_id)",
                "fit_validation": "trace_step_block mod 5 == 0",
                "rationale": (
                    "deterministic periodic one-in-five temporal-block class; "
                    "chosen from row counts/identities before outcome inspection"
                ),
            },
            "scene_assignments": [
                row.to_canonical_dict() for row in self.scene_assignments
            ],
            "scene_blocks": [row.to_canonical_dict() for row in self.scene_blocks],
            "scene_counts": dict(self.scene_counts),
            "scene_rule": {
                "block": "frame_id // 100",
                "domain": "splitfusion.empirical_fit_partition.v1:{block}",
                "fit_validation": "sha256(domain) interpreted as integer mod 5 == 0",
                "rationale": (
                    "approximate 80/20 temporal-block partition selected from "
                    "covariates/counts only before reward inspection"
                ),
            },
            "schema": self.schema,
            "source_bindings": dict(self.source_bindings),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def _fit_scene_assignments(project_root: Path) -> Tuple[SceneFitAssignmentV1, ...]:
    bundle = project_root / BUNDLE_RELATIVE_PATH
    selection_path = bundle / "selection_manifest.json"
    database_path = bundle / "quality_rows.sqlite3"
    if _sha256_file(selection_path) != SELECTION_FILE_SHA256:
        raise FitPartitionError("selection manifest SHA-256 drift")
    if _sha256_file(database_path) != DATABASE_FILE_SHA256:
        raise FitPartitionError("quality database SHA-256 drift")

    connection = sqlite3.connect(
        f"file:{database_path.resolve(strict=True)}?mode=ro&immutable=1",
        uri=True,
        timeout=30.0,
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            "SELECT sample_id, MIN(CAST(json_extract(row_json, '$.quality_valid') "
            "AS INTEGER)), MAX(CAST(json_extract(row_json, '$.quality_valid') "
            "AS INTEGER)), COUNT(*) FROM quality_rows WHERE grid_split='fit' "
            "GROUP BY sample_id ORDER BY sample_id"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 512 or any(count != 132 for _sid, _lo, _hi, count in rows):
        raise FitPartitionError("D1 fit validity inventory drift")
    if any(lo != hi for _sid, lo, hi, _count in rows):
        raise FitPartitionError("D1 invalidity is no longer action-independent")
    eligible = {sid for sid, lo, _hi, _count in rows if lo == 1}
    if len(eligible) != 476:
        raise FitPartitionError("D1 eligible scene count drift")

    document = json.loads(selection_path.read_text(encoding="utf-8"))
    assignments = []
    for row in document["selected_frames"]:
        if row.get("grid_split") != "fit" or row.get("sample_id") not in eligible:
            continue
        # Validate the only covariate consulted by D1 state construction, but
        # do not include it in the split rule or serialized partition.
        camera_si = float(row["camera_si"])
        sampling_weight = float(row["sampling_weight"])
        if not math.isfinite(camera_si) or not math.isfinite(sampling_weight):
            raise FitPartitionError("nonfinite fit covariate/design weight")
        frame_id = int(row["frame_id"])
        block = frame_id // 100
        assignments.append(
            SceneFitAssignmentV1(
                sample_id=str(row["sample_id"]),
                episode_id=str(row["episode_id"]),
                frame_id=frame_id,
                temporal_block=block,
                split=_scene_split(block),
            )
        )
    assignments.sort(key=lambda row: row.sample_id)
    if {row.sample_id for row in assignments} != eligible:
        raise FitPartitionError("eligible fit samples do not close exactly")
    return tuple(assignments)


def _radio_assignments(
    store: OaiRadioCalibrationStoreV1,
) -> Tuple[RadioFitAssignmentV1, ...]:
    assignments = tuple(
        RadioFitAssignmentV1(
            csv_row_number=row.csv_row_number,
            network_profile=row.network_profile,
            trace_id=row.trace_id,
            trace_step_index=row.trace_step_index,
            trace_step_block=row.trace_step_index // RADIO_BLOCK_STEPS,
            row_sha256=row.row_sha256,
            split=_radio_split(row.trace_step_index // RADIO_BLOCK_STEPS),
        )
        for row in sorted(store.rows, key=lambda item: item.csv_row_number)
    )
    if {row.csv_row_number for row in assignments} != {
        row.csv_row_number for row in store.rows
    }:
        raise FitPartitionError("valid radio rows do not close exactly")
    return assignments


def _scene_blocks(
    assignments: Tuple[SceneFitAssignmentV1, ...],
) -> Tuple[SceneBlockInventoryV1, ...]:
    keys = sorted({(row.temporal_block, row.split) for row in assignments})
    return tuple(
        SceneBlockInventoryV1(
            temporal_block=block,
            split=split,
            sample_ids=tuple(
                sorted(row.sample_id for row in assignments if row.temporal_block == block)
            ),
        )
        for block, split in keys
    )


def _radio_blocks(
    assignments: Tuple[RadioFitAssignmentV1, ...],
) -> Tuple[RadioBlockInventoryV1, ...]:
    keys = sorted(
        {
            (row.network_profile, row.trace_id, row.trace_step_block, row.split)
            for row in assignments
        }
    )
    return tuple(
        RadioBlockInventoryV1(
            network_profile=profile,
            trace_id=trace_id,
            trace_step_block=block,
            split=split,
            csv_row_numbers=tuple(
                sorted(
                    row.csv_row_number
                    for row in assignments
                    if (row.network_profile, row.trace_id, row.trace_step_block)
                    == (profile, trace_id, block)
                )
            ),
        )
        for profile, trace_id, block, split in keys
    )


def load_registered_empirical_fit_partition(
    *, project_root: Optional[Path] = None
) -> EmpiricalFitPartitionV1:
    """Load only fit identities plus the registered radio calibration rows."""
    root = (
        Path(__file__).resolve().parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    scenes = _fit_scene_assignments(root)
    radio_store = OaiRadioCalibrationStoreV1.load_registered(project_root=root)
    radios = _radio_assignments(radio_store)
    normalization = build_registered_normalization_spec()
    freshness = build_registered_freshness_policy()
    network_surrogate_sha256 = build_payload_network_surrogate().canonical_sha256()
    d1_binding = EmpiricalPilotBindingV1(
        corrected_p40_binding_sha256=D1_CORRECTED_P40_BINDING_SHA256,
        quality_surface_binding_sha256=D1_QUALITY_SURFACE_BINDING_SHA256,
        network_surrogate_sha256=network_surrogate_sha256,
        radio_calibration_sha256=CALIBRATION_SHA256,
        modeled_smoke_support_sha256=MODELED_SMOKE_SUPPORT_SHA256,
        utility_spec_sha256=PILOT_UTILITY_SPEC_SHA256,
        normalization_spec_sha256=normalization.canonical_sha256(),
        freshness_policy_sha256=freshness.canonical_sha256(),
        eligible_fit_context_index_sha256=(
            D1_ELIGIBLE_FIT_CONTEXT_INDEX_SHA256
        ),
        action_catalog_sha256=CATALOG_SHA256,
        surface_qualification_report_sha256=(
            SURFACE_QUALIFICATION_REPORT_SHA256
        ),
        profile_order_rng_contract_sha256=PROFILE_ORDER_RNG_CONTRACT_SHA256,
        implementation_bundle_sha256=_implementation_bundle_sha256(),
    )
    source_bindings = {
        "action_catalog_sha256": CATALOG_SHA256,
        "corrected_p40_binding_sha256": D1_CORRECTED_P40_BINDING_SHA256,
        "d1_eligible_fit_context_index_sha256": (
            D1_ELIGIBLE_FIT_CONTEXT_INDEX_SHA256
        ),
        "d1_pilot_binding_sha256": d1_binding.canonical_sha256(),
        "d1_freshness_policy_sha256": freshness.canonical_sha256(),
        "d1_normalization_spec_sha256": normalization.canonical_sha256(),
        "d1_profile_order_rng_contract_sha256": (
            PROFILE_ORDER_RNG_CONTRACT_SHA256
        ),
        "d1_quality_surface_binding_sha256": (
            D1_QUALITY_SURFACE_BINDING_SHA256
        ),
        "d1_surface_qualification_report_sha256": (
            SURFACE_QUALIFICATION_REPORT_SHA256
        ),
        "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "network_surrogate_sha256": network_surrogate_sha256,
        "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "quality_database_sha256": DATABASE_FILE_SHA256,
        "quality_reward_definition_sha256": REWARD_SPEC_FILE_SHA256,
        "quality_selection_sha256": SELECTION_FILE_SHA256,
        "radio_calibration_sha256": CALIBRATION_SHA256,
    }
    partition = EmpiricalFitPartitionV1(
        scene_assignments=scenes,
        radio_assignments=radios,
        scene_blocks=_scene_blocks(scenes),
        radio_blocks=_radio_blocks(radios),
        source_bindings=source_bindings,
        policy_feature_order=tuple(POLICY_FEATURE_ORDER),
        disclosures=(
            "PARTITION_RULES_USE_TEMPORAL_IDENTITIES_AND_COUNTS_ONLY; NO_REWARD_"
            "QUALITY_PAYLOAD_LATENCY_OR_DELIVERY_VALUE_SELECTED_MEMBERSHIP",
            "QUALITY_HELD_SCENE_ROWS_ARE_NOT_QUERIED_REPRESENTED_OR_EXPOSED",
            "FIT_VALIDATION_IS_REWARD_HELD_BUT_NOT_COVARIATE_NORMALIZATION_HELD;_"
            "D1_UNSUPERVISED_NORMALIZATION_USED_ALL_512_FIT_COVARIATES_AND_ALL_"
            "399_JOINT_VALID_RADIO_ROWS_BEFORE_PARTITIONING",
            "NETWORK_PROFILE_TRACE_AND_ROW_IDENTITIES_ARE_ENVIRONMENT_HIDDEN_AND_"
            "ABSENT_FROM_THE_EXACT_D1_POLICY_FEATURE_ORDER",
            "SCENE_TEMPORAL_BLOCKS_AND_PROFILE_TRACE_LOCAL_RADIO_BLOCKS_NEVER_"
            "CROSS_TRAIN_AND_FIT_VALIDATION",
        ),
    )
    if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
        raise FitPartitionError("registered empirical fit-partition hash drift")
    return partition


REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256 = (
    "44dad342d09cc39cab37e069627b8b9019e8cb9221fff25e1a8a8f240e8e1061"
)
