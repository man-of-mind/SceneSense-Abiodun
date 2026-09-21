"""Partition-aware sampling view over the frozen D1 empirical environment.

This module deliberately does not alter D1 evidence, reward, observation, or
action semantics.  It restricts only the two hidden sampling populations using
the separately registered, reward-blind fit partition.  The original D1
environment binding therefore remains intact; the partition and this sampling
contract are additional runner bindings.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_environment import (
    EmpiricalEnvironmentError,
    EmpiricalEnvironmentStateV1,
    EmpiricalOneStepEnvironmentV1,
    FitTrainingContextV1,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    EmpiricalFitPartitionV1,
    load_registered_empirical_fit_partition,
)
from .empirical_radio_context import (
    OaiRadioCalibrationStoreV1,
    RadioContextSamplerV1,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "PartitionedEmpiricalEnvironmentStateV1",
    "PartitionedEmpiricalOneStepEnvironmentV1",
    "load_registered_partitioned_d1_environment",
]


_STATE_SCHEMA = "splitfusion.partitioned_empirical_environment_state.v1"
_SAMPLING_SCHEMA = "splitfusion.partitioned_empirical_sampling.v1"
_ALLOWED_SPLITS = (TRAIN_SPLIT, FIT_VALIDATION_SPLIT)
_EXPECTED_SCENE_COUNTS = {TRAIN_SPLIT: 391, FIT_VALIDATION_SPLIT: 85}
_EXPECTED_RADIO_PROFILE_COUNTS = {
    TRAIN_SPLIT: {
        "FAVORABLE_STABLE": 80,
        "MID_VARIABLE": 80,
        "ADVERSE_STABLE": 79,
        "FADE_RECOVERY": 80,
    },
    FIT_VALIDATION_SPLIT: {
        "FAVORABLE_STABLE": 20,
        "MID_VARIABLE": 20,
        "ADVERSE_STABLE": 20,
        "FADE_RECOVERY": 20,
    },
}


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EmpiricalEnvironmentError(f"{name} must be a lowercase SHA-256")
    return value


def _require_split(value: object) -> str:
    if type(value) is not str or value not in _ALLOWED_SPLITS:
        raise EmpiricalEnvironmentError(
            "sampling split must be exactly 'train' or 'fit_validation'"
        )
    return value


@dataclass(frozen=True, slots=True)
class PartitionedEmpiricalEnvironmentStateV1:
    """Checkpoint identity plus the unchanged D1 environment state."""

    sampling_split: str
    fit_partition_sha256: str
    sampling_contract_sha256: str
    d1_state: EmpiricalEnvironmentStateV1
    schema: str = _STATE_SCHEMA

    def __post_init__(self) -> None:
        _require_split(self.sampling_split)
        _require_sha256(self.fit_partition_sha256, "fit_partition_sha256")
        _require_sha256(self.sampling_contract_sha256, "sampling_contract_sha256")
        if type(self.d1_state) is not EmpiricalEnvironmentStateV1:
            raise EmpiricalEnvironmentError(
                "d1_state must be exact EmpiricalEnvironmentStateV1"
            )
        if self.schema != _STATE_SCHEMA:
            raise EmpiricalEnvironmentError("partitioned environment state schema drift")


class PartitionedEmpiricalOneStepEnvironmentV1(EmpiricalOneStepEnvironmentV1):
    """D1 environment restricted to one registered fit-partition split.

    Scene sampling preserves D1's positive design weights after restriction
    and renormalisation.  The hidden network-profile prior remains exactly
    uniform, then a radio row is uniform within that profile and split.  In
    particular, the 79-row adverse training population is not accidentally
    underweighted relative to the three 80-row profile populations.
    """

    def __init__(
        self,
        *,
        sidecar,
        surface,
        network,
        radio_store: OaiRadioCalibrationStoreV1,
        contexts,
        normalization,
        freshness,
        seed: int,
        preflight,
        partition: EmpiricalFitPartitionV1,
        split: str,
    ) -> None:
        split = _require_split(split)
        if type(partition) is not EmpiricalFitPartitionV1:
            raise EmpiricalEnvironmentError(
                "partition must be exact EmpiricalFitPartitionV1"
            )
        if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
            raise EmpiricalEnvironmentError("registered fit-partition hash drift")
        if not isinstance(radio_store, OaiRadioCalibrationStoreV1):
            raise EmpiricalEnvironmentError(
                "radio_store must be OaiRadioCalibrationStoreV1"
            )

        super().__init__(
            sidecar=sidecar,
            surface=surface,
            network=network,
            radio_store=radio_store,
            contexts=contexts,
            normalization=normalization,
            freshness=freshness,
            seed=seed,
            preflight=preflight,
        )
        d1_binding_sha256 = self.binding.canonical_sha256()
        if (
            partition.source_bindings.get("d1_pilot_binding_sha256")
            != d1_binding_sha256
        ):
            raise EmpiricalEnvironmentError("fit partition/D1 pilot binding mismatch")
        if (
            partition.source_bindings.get("radio_calibration_sha256")
            != radio_store.source_sha256
        ):
            raise EmpiricalEnvironmentError(
                "fit partition/radio calibration binding mismatch"
            )
        if (
            partition.source_bindings.get("d1_eligible_fit_context_index_sha256")
            != contexts.canonical_sha256
        ):
            raise EmpiricalEnvironmentError(
                "fit partition/D1 context-index binding mismatch"
            )

        self._fit_partition = partition
        self._sampling_split = split
        self._fit_partition_sha256 = partition.canonical_sha256()
        self._split_contexts = self._join_scene_population(contexts.contexts)
        selected_radio_rows = self._join_radio_population(radio_store)
        rows_by_profile = MappingProxyType(
            {
                profile: tuple(
                    row
                    for row in selected_radio_rows
                    if row.network_profile == profile
                )
                for profile in NETWORK_PROFILE_ORDER
            }
        )
        observed_profile_counts = {
            profile: len(rows_by_profile[profile])
            for profile in NETWORK_PROFILE_ORDER
        }
        if observed_profile_counts != _EXPECTED_RADIO_PROFILE_COUNTS[split]:
            raise EmpiricalEnvironmentError(
                f"partitioned radio profile-count drift: {observed_profile_counts}"
            )

        # This private object is only a sampling view.  The full registered
        # radio store above remains the evidence/preflight/binding authority.
        sampling_view = OaiRadioCalibrationStoreV1(
            rows=selected_radio_rows,
            rows_by_profile=rows_by_profile,
            source_sha256=radio_store.source_sha256,
            source_relative_path=radio_store.source_relative_path,
            source_row_count=radio_store.source_row_count,
            joint_valid_row_count=len(selected_radio_rows),
        )
        self._radio_sampler = RadioContextSamplerV1(sampling_view, seed=seed)
        self._radio_profile_population_counts = MappingProxyType(
            observed_profile_counts
        )
        # Match D1's summation and cumulative-selection arithmetic exactly;
        # only the registered population is restricted.
        self._scene_sampling_weight_total = sum(
            context.sampling_weight for context in self._split_contexts
        )
        if (
            not math.isfinite(self._scene_sampling_weight_total)
            or self._scene_sampling_weight_total <= 0.0
        ):
            raise EmpiricalEnvironmentError(
                "partitioned scene sampling weight must be finite and positive"
            )
        self._sampling_contract_sha256 = canonical_sha256(
            {
                "d1_environment_binding_sha256": d1_binding_sha256,
                "fit_partition_sha256": self._fit_partition_sha256,
                "network_profile_sampling": "UNIFORM_OVER_REGISTERED_FOUR_PROFILES",
                "radio_row_sampling": "UNIFORM_WITHIN_PROFILE_AND_SPLIT",
                "radio_profile_population_counts": observed_profile_counts,
                "record": _SAMPLING_SCHEMA,
                "scene_population_count": len(self._split_contexts),
                "scene_sampling": (
                    "D1_SAMPLING_WEIGHT_RESTRICTED_TO_SPLIT_AND_RENORMALIZED"
                ),
                "split": split,
            }
        )

    @classmethod
    def load_registered(
        cls,
        *,
        seed: int,
        split: str,
        project_root: Optional[Path] = None,
    ) -> "PartitionedEmpiricalOneStepEnvironmentV1":
        """Load frozen D1 first, then add the registered sampling restriction."""
        split = _require_split(split)
        base = EmpiricalOneStepEnvironmentV1.load_registered(
            seed=seed, project_root=project_root
        )
        try:
            partition = load_registered_empirical_fit_partition(
                project_root=project_root
            )
            return cls(
                sidecar=base._sidecar,
                surface=base._surface,
                network=base._network,
                radio_store=base._radio_store,
                contexts=base._contexts,
                normalization=base.normalization,
                freshness=base.freshness,
                seed=seed,
                preflight=base.preflight,
                partition=partition,
                split=split,
            )
        except Exception:
            base.close()
            raise

    @property
    def sampling_split(self) -> str:
        return self._sampling_split

    @property
    def fit_partition_sha256(self) -> str:
        return self._fit_partition_sha256

    @property
    def sampling_contract_sha256(self) -> str:
        return self._sampling_contract_sha256

    @property
    def scene_population_count(self) -> int:
        return len(self._split_contexts)

    @property
    def radio_profile_population_counts(self) -> Mapping[str, int]:
        return self._radio_profile_population_counts

    @property
    def scene_sampling_weight_total(self) -> float:
        return self._scene_sampling_weight_total

    def _join_scene_population(
        self, contexts: Tuple[FitTrainingContextV1, ...]
    ) -> Tuple[FitTrainingContextV1, ...]:
        context_by_id = {context.sample_id: context for context in contexts}
        if len(context_by_id) != len(contexts):
            raise EmpiricalEnvironmentError("D1 context index duplicated a sample")
        assignment_by_id = {
            assignment.sample_id: assignment
            for assignment in self._fit_partition.scene_assignments
        }
        if set(context_by_id) != set(assignment_by_id):
            raise EmpiricalEnvironmentError(
                "fit partition scenes do not close over the D1 context index"
            )
        for sample_id, context in context_by_id.items():
            assignment = assignment_by_id[sample_id]
            if (
                assignment.episode_id != context.episode_id
                or assignment.frame_id != context.frame_id
            ):
                raise EmpiricalEnvironmentError(
                    f"fit partition scene identity mismatch for {sample_id!r}"
                )
        selected = tuple(
            context
            for context in contexts
            if assignment_by_id[context.sample_id].split == self._sampling_split
        )
        if len(selected) != _EXPECTED_SCENE_COUNTS[self._sampling_split]:
            raise EmpiricalEnvironmentError(
                f"partitioned scene-count drift: {len(selected)}"
            )
        return selected

    def _join_radio_population(self, store: OaiRadioCalibrationStoreV1):
        row_by_number = {row.csv_row_number: row for row in store.rows}
        if len(row_by_number) != len(store.rows):
            raise EmpiricalEnvironmentError("D1 radio store duplicated a CSV row")
        assignment_by_number = {
            assignment.csv_row_number: assignment
            for assignment in self._fit_partition.radio_assignments
        }
        if set(row_by_number) != set(assignment_by_number):
            raise EmpiricalEnvironmentError(
                "fit partition radio rows do not close over the D1 radio store"
            )
        for row_number, row in row_by_number.items():
            assignment = assignment_by_number[row_number]
            if (
                assignment.network_profile != row.network_profile
                or assignment.trace_id != row.trace_id
                or assignment.trace_step_index != row.trace_step_index
                or assignment.row_sha256 != row.row_sha256
            ):
                raise EmpiricalEnvironmentError(
                    f"fit partition radio identity mismatch for row {row_number}"
                )
        return tuple(
            row
            for row in store.rows
            if assignment_by_number[row.csv_row_number].split
            == self._sampling_split
        )

    def _sample_fit_context(self) -> FitTrainingContextV1:
        threshold = self._context_rng.random() * self._scene_sampling_weight_total
        cumulative = 0.0
        for context in self._split_contexts:
            cumulative += context.sampling_weight
            if threshold < cumulative:
                return context
        return self._split_contexts[-1]

    def state_dict(self) -> PartitionedEmpiricalEnvironmentStateV1:
        return PartitionedEmpiricalEnvironmentStateV1(
            sampling_split=self._sampling_split,
            fit_partition_sha256=self._fit_partition_sha256,
            sampling_contract_sha256=self._sampling_contract_sha256,
            d1_state=super().state_dict(),
        )

    def load_state_dict(
        self, state: PartitionedEmpiricalEnvironmentStateV1
    ) -> None:
        if type(state) is not PartitionedEmpiricalEnvironmentStateV1:
            raise EmpiricalEnvironmentError(
                "state must be exact PartitionedEmpiricalEnvironmentStateV1"
            )
        if state.sampling_split != self._sampling_split:
            raise EmpiricalEnvironmentError("checkpoint sampling-split mismatch")
        if state.fit_partition_sha256 != self._fit_partition_sha256:
            raise EmpiricalEnvironmentError("checkpoint fit-partition mismatch")
        if state.sampling_contract_sha256 != self._sampling_contract_sha256:
            raise EmpiricalEnvironmentError("checkpoint sampling-contract mismatch")
        super().load_state_dict(state.d1_state)


def load_registered_partitioned_d1_environment(
    *,
    seed: int,
    split: str,
    project_root: Optional[Path] = None,
) -> PartitionedEmpiricalOneStepEnvironmentV1:
    return PartitionedEmpiricalOneStepEnvironmentV1.load_registered(
        seed=seed, split=split, project_root=project_root
    )
