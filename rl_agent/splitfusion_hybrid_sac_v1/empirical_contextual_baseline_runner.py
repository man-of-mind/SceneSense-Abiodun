"""Preliminary train-split Hybrid-SAC baseline runner.

This is an exploratory numerical baseline over the registered D1 *train*
partition.  It deliberately reuses the audited synchronous smoke-runner path
and terminal target ``y = reward``.  Its replay linkage is established by this
trusted in-process runner, not by a provenance-complete external attestation;
therefore its artifacts are initial training diagnostics, not publication or
final scientific evidence.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalOneStepEnvironmentV1,
)
from .empirical_contextual_smoke_runner import (
    EmpiricalContextualSmokeRunnerV1,
    EmpiricalSmokeConfigV1,
    EmpiricalSmokeSummaryV1,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "BASELINE_CHECKPOINT_INTERVAL_UPDATES",
    "BASELINE_PROVENANCE_LABEL",
    "BASELINE_SCOPE_DISCLOSURE",
    "EmpiricalContextualBaselineRunnerV1",
    "PHASE_LABEL",
    "REGISTERED_BASELINE_CONFIG",
]


PHASE_LABEL = "D2C_PRELIMINARY_TRAIN_SPLIT_HYBRID_SAC_BASELINE_V1"
BASELINE_PROVENANCE_LABEL = (
    "TRUSTED_SYNCHRONOUS_RUNNER_NUMERICAL_BOUNDARY_"
    "NOT_PROVENANCE_COMPLETE_NOT_FINAL_SCIENTIFIC_EVIDENCE"
)
BASELINE_SCOPE_DISCLOSURE = (
    "Exploratory initial training on only the registered reward-blind train "
    "partition. No validation, checkpoint selection, comparator, convergence, "
    "generalization, deployment, or provenance-completeness claim is made."
)
BASELINE_RUNNER_SCHEMA = "splitfusion.empirical_contextual_baseline_runner.v1"
BASELINE_CHECKPOINT_INTERVAL_UPDATES = 500
_BASELINE_COLLECTION_NAMESPACE = uuid.UUID(
    "a8d692a2-cdf9-5488-87ba-240ef06e2c3f"
)


REGISTERED_BASELINE_CONFIG = EmpiricalSmokeConfigV1(
    seeds=(17, 29, 43),
    warmup_transitions=1024,
    batch_size=256,
    collect_per_update=4,
    update_count=5000,
    replay_capacity=32768,
    cpu_threads=1,
    scope=PHASE_LABEL,
)


class EmpiricalContextualBaselineRunnerV1(EmpiricalContextualSmokeRunnerV1):
    """Train-partition specialization of the audited mechanics runner."""

    def __init__(
        self,
        *,
        seed: int,
        config: EmpiricalSmokeConfigV1 = REGISTERED_BASELINE_CONFIG,
        project_root: Optional[Path] = None,
    ) -> None:
        if config.scope != PHASE_LABEL:
            raise ValueError("baseline config must carry the baseline scope label")
        super().__init__(seed=seed, config=config, project_root=project_root)
        base = self.environment
        try:
            partition = load_registered_empirical_fit_partition(
                project_root=project_root
            )
            partitioned = PartitionedEmpiricalOneStepEnvironmentV1(
                sidecar=base._sidecar,
                surface=base._surface,
                network=base._network,
                radio_store=base._radio_store,
                contexts=base._contexts,
                normalization=base.normalization,
                freshness=base.freshness,
                seed=self._stream_seeds["environment"],
                preflight=base.preflight,
                partition=partition,
                split=TRAIN_SPLIT,
            )
        except Exception:
            base.close()
            self._closed = True
            raise

        # The partitioned view owns the same read-only surface.  Do not close
        # the superseded base view: doing so would close the shared database.
        self.environment = partitioned
        self._baseline_binding_document = self._make_baseline_binding_document()
        self._runner_binding_sha256 = canonical_sha256(
            self._baseline_binding_document
        )
        self._collection_session_uuid = str(
            uuid.uuid5(
                _BASELINE_COLLECTION_NAMESPACE,
                f"{self._runner_binding_sha256}:{self.seed}",
            )
        )

    def _make_baseline_binding_document(self) -> Dict[str, Any]:
        return {
            "base_mechanics_binding": super()._binding_document(),
            "fit_partition_sha256": self.environment.fit_partition_sha256,
            "provenance_label": BASELINE_PROVENANCE_LABEL,
            "runner_schema": BASELINE_RUNNER_SCHEMA,
            "sampling_contract_sha256": (
                self.environment.sampling_contract_sha256
            ),
            "sampling_split": self.environment.sampling_split,
            "scope_disclosure": BASELINE_SCOPE_DISCLOSURE,
        }

    @property
    def baseline_binding_document(self) -> Dict[str, Any]:
        return dict(self._baseline_binding_document)

    def summary(self) -> EmpiricalSmokeSummaryV1:
        base = super().summary()
        return replace(
            base,
            phase_label=PHASE_LABEL,
            scope_disclosure=BASELINE_SCOPE_DISCLOSURE,
        )

    def require_registered_train_binding(self) -> None:
        if self.environment.sampling_split != TRAIN_SPLIT:
            raise RuntimeError("baseline environment is not train-only")
        if (
            self.environment.fit_partition_sha256
            != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        ):
            raise RuntimeError("baseline fit-partition binding drift")
        if self._runner_binding_sha256 != canonical_sha256(
            self._baseline_binding_document
        ):
            raise RuntimeError("baseline runner binding drift")

