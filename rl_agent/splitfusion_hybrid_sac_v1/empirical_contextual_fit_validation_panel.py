"""Frozen deterministic fit-validation panel for empirical Hybrid-SAC.

The panel crosses every registered fit-validation scene with every network
profile.  Within each profile, its 20 validation radio rows are assigned to
the canonically sorted 85 scenes by deterministic round robin.  Assignment
uses identities and split labels only; reward, quality, payload, latency,
delivery, action, and outcome values are neither accepted nor represented.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_fit_partition import (
    FIT_PARTITION_SCHEMA,
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    EmpiricalFitPartitionV1,
    load_registered_empirical_fit_partition,
)
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "FIT_VALIDATION_PANEL_SCHEMA",
    "PANEL_ASSIGNMENT_INPUT_FIELDS",
    "REGISTERED_FIT_VALIDATION_PANEL_SHA256",
    "FitValidationPanelError",
    "FitValidationPanelEntryV1",
    "FitValidationPanelManifestV1",
    "load_registered_fit_validation_panel",
]


FIT_VALIDATION_PANEL_SCHEMA = "splitfusion.empirical_fit_validation_panel.v1"
PANEL_ASSIGNMENT_INPUT_FIELDS: Tuple[str, ...] = (
    "scene.sample_id",
    "scene.episode_id",
    "scene.frame_id",
    "scene.split",
    "radio.csv_row_number",
    "radio.network_profile",
    "radio.trace_id",
    "radio.trace_step_index",
    "radio.row_sha256",
    "radio.split",
)


class FitValidationPanelError(ValueError):
    """The frozen validation-panel identity or allocation drifted."""


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FitValidationPanelError(f"{name} must be a lowercase SHA-256")
    return value


def _scene_key(row: object) -> Tuple[str, str, int]:
    return (row.sample_id, row.episode_id, row.frame_id)


def _radio_key(row: object) -> Tuple[str, int, int, str]:
    return (
        row.trace_id,
        row.trace_step_index,
        row.csv_row_number,
        row.row_sha256,
    )


@dataclass(frozen=True, slots=True)
class FitValidationPanelEntryV1:
    panel_index: int
    scene_rank: int
    scene_sample_id: str
    scene_episode_id: str
    scene_frame_id: int
    scene_split: str
    profile_rank: int
    network_profile: str
    radio_rank: int
    radio_csv_row_number: int
    radio_trace_id: str
    radio_trace_step_index: int
    radio_row_sha256: str
    radio_split: str
    fit_partition_sha256: str
    d1_pilot_binding_sha256: str
    source_bindings_sha256: str

    def __post_init__(self) -> None:
        for name in ("panel_index", "scene_rank", "scene_frame_id", "profile_rank",
                     "radio_rank", "radio_csv_row_number", "radio_trace_step_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise FitValidationPanelError(f"{name} must be a non-negative integer")
        if not self.scene_sample_id or not self.scene_episode_id or not self.radio_trace_id:
            raise FitValidationPanelError("panel identities must be non-empty")
        if self.scene_split != FIT_VALIDATION_SPLIT or self.radio_split != FIT_VALIDATION_SPLIT:
            raise FitValidationPanelError("panel entries must be fit-validation only")
        if self.profile_rank >= len(NETWORK_PROFILE_ORDER):
            raise FitValidationPanelError("profile rank is outside the registered order")
        if self.network_profile != NETWORK_PROFILE_ORDER[self.profile_rank]:
            raise FitValidationPanelError("network profile contradicts profile rank")
        if self.radio_rank >= 20:
            raise FitValidationPanelError("radio rank must be in [0, 19]")
        _require_sha256(self.radio_row_sha256, "radio_row_sha256")
        _require_sha256(self.fit_partition_sha256, "fit_partition_sha256")
        _require_sha256(self.d1_pilot_binding_sha256, "d1_pilot_binding_sha256")
        _require_sha256(self.source_bindings_sha256, "source_bindings_sha256")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FitValidationPanelManifestV1:
    entries: Tuple[FitValidationPanelEntryV1, ...]
    profile_order: Tuple[str, ...]
    assignment_input_fields: Tuple[str, ...]
    fit_partition_schema: str
    fit_partition_sha256: str
    d1_pilot_binding_sha256: str
    source_bindings: Mapping[str, str]
    source_bindings_sha256: str
    schema: str = FIT_VALIDATION_PANEL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_bindings", MappingProxyType(dict(self.source_bindings)))
        if self.schema != FIT_VALIDATION_PANEL_SCHEMA:
            raise FitValidationPanelError("validation-panel schema drift")
        if self.fit_partition_schema != FIT_PARTITION_SCHEMA:
            raise FitValidationPanelError("fit-partition schema drift")
        if self.profile_order != tuple(NETWORK_PROFILE_ORDER):
            raise FitValidationPanelError("registered network-profile order drift")
        if self.assignment_input_fields != PANEL_ASSIGNMENT_INPUT_FIELDS:
            raise FitValidationPanelError("assignment input-field inventory drift")
        _require_sha256(self.fit_partition_sha256, "fit_partition_sha256")
        _require_sha256(self.d1_pilot_binding_sha256, "d1_pilot_binding_sha256")
        _require_sha256(self.source_bindings_sha256, "source_bindings_sha256")
        if canonical_sha256(dict(self.source_bindings)) != self.source_bindings_sha256:
            raise FitValidationPanelError("source-binding aggregate hash drift")
        if self.source_bindings.get("d1_pilot_binding_sha256") != self.d1_pilot_binding_sha256:
            raise FitValidationPanelError("D1 pilot binding does not close")
        self._validate_panel()

    def _validate_panel(self) -> None:
        if len(self.entries) != 340:
            raise FitValidationPanelError("validation panel must contain 340 entries")
        if tuple(row.panel_index for row in self.entries) != tuple(range(340)):
            raise FitValidationPanelError("panel indexes must be contiguous and ordered")
        for row in self.entries:
            if (
                row.fit_partition_sha256 != self.fit_partition_sha256
                or row.d1_pilot_binding_sha256 != self.d1_pilot_binding_sha256
                or row.source_bindings_sha256 != self.source_bindings_sha256
            ):
                raise FitValidationPanelError("entry provenance binding drift")

        scenes = sorted(
            {
                (row.scene_sample_id, row.scene_episode_id, row.scene_frame_id)
                for row in self.entries
            }
        )
        if len(scenes) != 85:
            raise FitValidationPanelError("validation panel must contain 85 scenes")
        expected_order = []
        for scene_rank, scene in enumerate(scenes):
            rows_for_scene = [
                row
                for row in self.entries
                if (row.scene_sample_id, row.scene_episode_id, row.scene_frame_id) == scene
            ]
            if len(rows_for_scene) != 4 or {row.network_profile for row in rows_for_scene} != set(self.profile_order):
                raise FitValidationPanelError("each scene must cross all four profiles exactly once")
            for profile_rank, profile in enumerate(self.profile_order):
                row = next(item for item in rows_for_scene if item.network_profile == profile)
                if row.scene_rank != scene_rank or row.profile_rank != profile_rank:
                    raise FitValidationPanelError("scene/profile ranks drifted")
                expected_order.append(row)
        if tuple(expected_order) != self.entries:
            raise FitValidationPanelError("entries are not in canonical scene-major order")

        for profile in self.profile_order:
            profile_entries = [row for row in self.entries if row.network_profile == profile]
            radio_identities = sorted(
                {
                    (row.radio_trace_id, row.radio_trace_step_index,
                     row.radio_csv_row_number, row.radio_row_sha256)
                    for row in profile_entries
                }
            )
            if len(radio_identities) != 20:
                raise FitValidationPanelError("each profile must contain 20 validation radio rows")
            usage = Counter(row.radio_rank for row in profile_entries)
            if sorted(usage.values()) != [4] * 15 + [5] * 5:
                raise FitValidationPanelError("radio round-robin usage is unbalanced")
            for row in profile_entries:
                expected_rank = row.scene_rank % 20
                identity = (
                    row.radio_trace_id,
                    row.radio_trace_step_index,
                    row.radio_csv_row_number,
                    row.radio_row_sha256,
                )
                if row.radio_rank != expected_rank or identity != radio_identities[expected_rank]:
                    raise FitValidationPanelError("radio assignment is not canonical round robin")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "assignment_input_fields": list(self.assignment_input_fields),
            "assignment_rule": (
                "SORT_SCENES_BY_SAMPLE_EPISODE_FRAME; PER_PROFILE_SORT_RADIO_BY_"
                "TRACE_STEP_CSV_SHA; ASSIGN_RADIO_RANK=SCENE_RANK_MOD_20"
            ),
            "d1_pilot_binding_sha256": self.d1_pilot_binding_sha256,
            "entries": [row.to_canonical_dict() for row in self.entries],
            "fit_partition_schema": self.fit_partition_schema,
            "fit_partition_sha256": self.fit_partition_sha256,
            "profile_order": list(self.profile_order),
            "schema": self.schema,
            "source_bindings": dict(self.source_bindings),
            "source_bindings_sha256": self.source_bindings_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def _build_panel(partition: EmpiricalFitPartitionV1) -> FitValidationPanelManifestV1:
    if type(partition) is not EmpiricalFitPartitionV1:
        raise FitValidationPanelError("partition must be exact EmpiricalFitPartitionV1")
    partition_sha256 = partition.canonical_sha256()
    if partition_sha256 != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
        raise FitValidationPanelError("registered fit-partition hash drift")
    scenes = sorted(
        (row for row in partition.scene_assignments if row.split == FIT_VALIDATION_SPLIT),
        key=_scene_key,
    )
    radios_by_profile = {
        profile: sorted(
            (
                row
                for row in partition.radio_assignments
                if row.split == FIT_VALIDATION_SPLIT and row.network_profile == profile
            ),
            key=_radio_key,
        )
        for profile in NETWORK_PROFILE_ORDER
    }
    if len(scenes) != 85 or any(len(rows) != 20 for rows in radios_by_profile.values()):
        raise FitValidationPanelError("registered validation populations drifted")
    source_bindings = dict(partition.source_bindings)
    source_bindings_sha256 = canonical_sha256(source_bindings)
    d1_pilot_binding_sha256 = source_bindings["d1_pilot_binding_sha256"]
    entries = []
    for scene_rank, scene in enumerate(scenes):
        for profile_rank, profile in enumerate(NETWORK_PROFILE_ORDER):
            radio_rank = scene_rank % 20
            radio = radios_by_profile[profile][radio_rank]
            entries.append(
                FitValidationPanelEntryV1(
                    panel_index=len(entries),
                    scene_rank=scene_rank,
                    scene_sample_id=scene.sample_id,
                    scene_episode_id=scene.episode_id,
                    scene_frame_id=scene.frame_id,
                    scene_split=scene.split,
                    profile_rank=profile_rank,
                    network_profile=profile,
                    radio_rank=radio_rank,
                    radio_csv_row_number=radio.csv_row_number,
                    radio_trace_id=radio.trace_id,
                    radio_trace_step_index=radio.trace_step_index,
                    radio_row_sha256=radio.row_sha256,
                    radio_split=radio.split,
                    fit_partition_sha256=partition_sha256,
                    d1_pilot_binding_sha256=d1_pilot_binding_sha256,
                    source_bindings_sha256=source_bindings_sha256,
                )
            )
    return FitValidationPanelManifestV1(
        entries=tuple(entries),
        profile_order=tuple(NETWORK_PROFILE_ORDER),
        assignment_input_fields=PANEL_ASSIGNMENT_INPUT_FIELDS,
        fit_partition_schema=partition.schema,
        fit_partition_sha256=partition_sha256,
        d1_pilot_binding_sha256=d1_pilot_binding_sha256,
        source_bindings=source_bindings,
        source_bindings_sha256=source_bindings_sha256,
    )


def load_registered_fit_validation_panel() -> FitValidationPanelManifestV1:
    panel = _build_panel(load_registered_empirical_fit_partition())
    if panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
        raise FitValidationPanelError("registered fit-validation panel hash drift")
    return panel


REGISTERED_FIT_VALIDATION_PANEL_SHA256 = (
    "1946272c4060c193b3e9e57d3b1cb490b411b889af394f8c089eeddbf611b3d5"
)
