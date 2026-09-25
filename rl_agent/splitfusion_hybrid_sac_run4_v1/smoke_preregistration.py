"""Prospective, fail-closed preregistration for the Run-4 SAC smoke run.

This module freezes the 500-update smoke configuration and its continuation
questions *before* any smoke training is run.  It contains no runner, reads no
evidence, initializes no accelerator and cannot manufacture a diagnostic
panel.  Production assessment stays impossible until a future composite
verifier binds real, disjoint validation-calibration cells, fit-validation
scenes, an accepted sequential-kernel identity and a pre-launch runtime
budget.

The learning gates compare the policy with a fit-selected fixed comparator and
with an exact fixed-panel oracle.  Paired confidence intervals must exclude
zero: an arbitrarily small numerical improvement is not enough.  A declared
95% dominant-mode ceiling is an engineering collapse sentinel, not a claim
that healthy policies must use every mode.  Passing remains only a bounded
continuation decision, never a convergence or deployment claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_MAX,
)

__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "FROZEN_CONFIG",
    "FROZEN_GATE_SPECS",
    "PREREGISTRATION_SHA256",
    "PAIRED_INTERVAL_SPEC_SHA256",
    "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256",
    "SENSITIVITY_FEATURES",
    "SmokePreregistrationError",
    "PanelBindingError",
    "DiagnosticError",
    "ContinuationRefused",
    "GateClassification",
    "EvidenceClass",
    "SmokeConfigV1",
    "GateSpecV1",
    "DiagnosticPanelManifestV1",
    "CompositeVerifierEvidenceV1",
    "SensitivityDiagnosticV1",
    "CheckpointDiagnosticV1",
    "SmokeDiagnosticsV1",
    "GateResultV1",
    "SmokeAssessmentV1",
    "bind_verified_panel",
    "assess_verified_smoke",
]


SCHEMA_ID = "splitfusion.run4.smoke_preregistration.v1"
SCHEMA_VERSION = 1

# A reviewed future integration must replace ``None`` with the exact digest of
# the composite-verifier manifest.  Until then no caller can obtain a
# production panel binding, even with plausible-looking hashes and identities.
REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256: Optional[str] = None

SENSITIVITY_FEATURES: Tuple[str, ...] = (
    "prior_ul_mcs_index",
    "rlc_backlog_bytes",
    "scene_si",
    "scene_p40",
    "previous_action",
    "previous_outcome",
)


class SmokePreregistrationError(ValueError):
    """Base class for malformed preregistration records."""


class PanelBindingError(SmokePreregistrationError):
    """A diagnostic panel was not issued by the registered verifier."""


class DiagnosticError(SmokePreregistrationError):
    """A smoke diagnostic record is malformed or bound to another panel."""


class ContinuationRefused(RuntimeError):
    """The smoke record does not authorize a longer or additional-seed run."""


class GateClassification(str, Enum):
    """How a continuation gate may be interpreted."""

    INTEGRITY_ACCEPTANCE = "INTEGRITY_ACCEPTANCE"
    HYPOTHESIS_DIAGNOSTIC = "HYPOTHESIS_DIAGNOSTIC"


class EvidenceClass(str, Enum):
    """Production evidence and test mechanics are deliberately disjoint."""

    VERIFIED_COMPOSITE = "VERIFIED_COMPOSITE"
    TEST_ONLY_SYNTHETIC = "TEST_ONLY_SYNTHETIC"


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SmokePreregistrationError(
            "record is not canonical-JSON encodable"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SmokePreregistrationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SmokePreregistrationError(f"{name} must be a non-empty str")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SmokePreregistrationError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def _positive_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SmokePreregistrationError(
            f"{name} must be an exact positive finite float"
        )
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticError(f"{name} must be finite")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SmokePreregistrationError(f"{name} must be an exact bool")
    return value


def _string_tuple(
    values: object, name: str, *, minimum_count: int = 1
) -> Tuple[str, ...]:
    if type(values) is not tuple or len(values) < minimum_count:
        raise SmokePreregistrationError(
            f"{name} must be a tuple with at least {minimum_count} item(s)"
        )
    for value in values:
        _text(value, name)
    if len(set(values)) != len(values):
        raise SmokePreregistrationError(f"{name} contains duplicate identities")
    return values


class _CanonicalRecord:
    __slots__ = ()
    RECORD_TYPE = "abstract"

    def _payload(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            **self._payload(),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_canonical_dict())

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class SmokeConfigV1(_CanonicalRecord):
    """Frozen mechanics for the prospective smoke and later continuations."""

    gamma_per_tensor: float
    alpha_d: float
    alpha_c: float
    actor_learning_rate: float
    critic_learning_rate: float
    polyak_tau: float
    batch_size: int
    replay_capacity: int
    warmup_mode_count: int
    warmup_q_bin_count: int
    warmup_samples_per_mode_q_bin: int
    warmup_decision_count: int
    environment_transitions_per_update: int
    torch_intraop_threads: int
    seed_order: Tuple[int, ...]
    initial_smoke_seed: int
    checkpoint_updates: Tuple[int, ...]
    smoke_stop_update: int
    paired_confidence_level: float
    paired_bootstrap_resamples: int
    paired_bootstrap_seed: int
    max_dominant_mode_fraction: float

    RECORD_TYPE = "run4_smoke_config_v1"

    def __post_init__(self) -> None:
        for name in (
            "gamma_per_tensor",
            "alpha_d",
            "alpha_c",
            "actor_learning_rate",
            "critic_learning_rate",
            "polyak_tau",
        ):
            _positive_float(getattr(self, name), name)
        if not 0.0 < self.gamma_per_tensor <= 1.0:
            raise SmokePreregistrationError("gamma_per_tensor must lie in (0,1]")
        if not 0.0 < self.polyak_tau <= 1.0:
            raise SmokePreregistrationError("polyak_tau must lie in (0,1]")
        for name in (
            "paired_confidence_level",
            "max_dominant_mode_fraction",
        ):
            _positive_float(getattr(self, name), name)
        if not 0.0 < self.paired_confidence_level < 1.0:
            raise SmokePreregistrationError(
                "paired_confidence_level must lie in (0,1)"
            )
        if not 0.0 < self.max_dominant_mode_fraction < 1.0:
            raise SmokePreregistrationError(
                "max_dominant_mode_fraction must lie in (0,1)"
            )
        for name in (
            "batch_size",
            "replay_capacity",
            "warmup_mode_count",
            "warmup_q_bin_count",
            "warmup_samples_per_mode_q_bin",
            "warmup_decision_count",
            "environment_transitions_per_update",
            "torch_intraop_threads",
            "smoke_stop_update",
            "paired_bootstrap_resamples",
        ):
            _exact_int(getattr(self, name), name, minimum=1)
        _exact_int(self.paired_bootstrap_seed, "paired_bootstrap_seed")
        if self.warmup_mode_count != EXPECTED_MODE_COUNT:
            raise SmokePreregistrationError(
                f"warmup must cover all {EXPECTED_MODE_COUNT} modes"
            )
        derived_warmup = (
            self.warmup_mode_count
            * self.warmup_q_bin_count
            * self.warmup_samples_per_mode_q_bin
        )
        if self.warmup_decision_count != derived_warmup:
            raise SmokePreregistrationError(
                "warmup_decision_count does not equal mode x q-bin x repetition"
            )
        if type(self.seed_order) is not tuple or len(self.seed_order) != 3:
            raise SmokePreregistrationError("seed_order must contain three seeds")
        for seed in self.seed_order:
            _exact_int(seed, "seed_order item")
        if len(set(self.seed_order)) != len(self.seed_order):
            raise SmokePreregistrationError("seed_order must be unique")
        if self.initial_smoke_seed != self.seed_order[0]:
            raise SmokePreregistrationError(
                "only the first seed may run before the 500-update gate"
            )
        if type(self.checkpoint_updates) is not tuple:
            raise SmokePreregistrationError("checkpoint_updates must be a tuple")
        if self.checkpoint_updates != tuple(sorted(set(self.checkpoint_updates))):
            raise SmokePreregistrationError(
                "checkpoint_updates must be unique and increasing"
            )
        if self.checkpoint_updates != (0, 100, 250, 500, 1500, 10000):
            raise SmokePreregistrationError(
                "Run-4 checkpoints must be 0/100/250/500/1500/10000"
            )
        if self.smoke_stop_update != 500:
            raise SmokePreregistrationError("the prospective smoke must stop at 500")
        if self.replay_capacity < self.batch_size:
            raise SmokePreregistrationError(
                "replay_capacity cannot be smaller than batch_size"
            )

    @property
    def smoke_checkpoint_updates(self) -> Tuple[int, ...]:
        return tuple(
            update for update in self.checkpoint_updates
            if update <= self.smoke_stop_update
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "actor_learning_rate": self.actor_learning_rate,
            "alpha_c": self.alpha_c,
            "alpha_d": self.alpha_d,
            "batch_size": self.batch_size,
            "checkpoint_updates": list(self.checkpoint_updates),
            "critic_learning_rate": self.critic_learning_rate,
            "discount_rule": (
                "replay stores contract-derived gamma_per_tensor**duration; "
                "trainer consumes emitted discount verbatim and never recomputes"
            ),
            "environment_transitions_per_update": (
                self.environment_transitions_per_update
            ),
            "gamma_per_tensor": self.gamma_per_tensor,
            "initial_smoke_seed": self.initial_smoke_seed,
            "max_dominant_mode_fraction": self.max_dominant_mode_fraction,
            "paired_bootstrap_resamples": self.paired_bootstrap_resamples,
            "paired_bootstrap_seed": self.paired_bootstrap_seed,
            "paired_confidence_level": self.paired_confidence_level,
            "polyak_tau": self.polyak_tau,
            "replay_capacity": self.replay_capacity,
            "seed_order": list(self.seed_order),
            "seed_rule": (
                "seed 17 stops at update 500; seeds 29 and 43 and updates "
                "beyond 500 require an authorized continuation decision"
            ),
            "smoke_stop_update": self.smoke_stop_update,
            "torch_intraop_threads": self.torch_intraop_threads,
            "warmup_decision_count": self.warmup_decision_count,
            "warmup_mode_count": self.warmup_mode_count,
            "warmup_q_bin_count": self.warmup_q_bin_count,
            "warmup_samples_per_mode_q_bin": (
                self.warmup_samples_per_mode_q_bin
            ),
        }


@dataclass(frozen=True, slots=True)
class GateSpecV1(_CanonicalRecord):
    gate_id: str
    classification: GateClassification
    rule: str

    RECORD_TYPE = "run4_smoke_gate_spec_v1"

    def __post_init__(self) -> None:
        _text(self.gate_id, "gate_id")
        if type(self.classification) is not GateClassification:
            raise SmokePreregistrationError(
                "classification must be exactly GateClassification"
            )
        _text(self.rule, "rule")

    def _payload(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "gate_id": self.gate_id,
            "rule": self.rule,
        }


FROZEN_CONFIG = SmokeConfigV1(
    gamma_per_tensor=0.99,
    alpha_d=0.05,
    alpha_c=0.02,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    polyak_tau=0.005,
    batch_size=256,
    replay_capacity=65536,
    warmup_mode_count=12,
    warmup_q_bin_count=6,
    warmup_samples_per_mode_q_bin=4,
    warmup_decision_count=288,
    environment_transitions_per_update=4,
    torch_intraop_threads=4,
    seed_order=(17, 29, 43),
    initial_smoke_seed=17,
    checkpoint_updates=(0, 100, 250, 500, 1500, 10000),
    smoke_stop_update=500,
    paired_confidence_level=0.95,
    paired_bootstrap_resamples=10_000,
    paired_bootstrap_seed=17,
    max_dominant_mode_fraction=0.95,
)


PAIRED_INTERVAL_SPEC_SHA256 = _canonical_sha256(
    {
        "confidence_level": FROZEN_CONFIG.paired_confidence_level,
        "method": "cluster_percentile_bootstrap_of_paired_context_differences",
        "replicates": FROZEN_CONFIG.paired_bootstrap_resamples,
        "resampling_unit": "verifier_bound_context_group_id",
        "seed": FROZEN_CONFIG.paired_bootstrap_seed,
    }
)


FROZEN_GATE_SPECS: Tuple[GateSpecV1, ...] = (
    GateSpecV1(
        "warmup_mode_q_coverage",
        GateClassification.INTEGRITY_ACCEPTANCE,
        "every one of 12 modes x 6 q bins is observed, including bin 5",
    ),
    GateSpecV1(
        "feedback_outcome_coverage",
        GateClassification.INTEGRITY_ACCEPTANCE,
        "the smoke contains at least one registered success and one failure",
    ),
    GateSpecV1(
        "finite_numerics",
        GateClassification.INTEGRITY_ACCEPTANCE,
        "all parameters, optimizer states, losses, targets and diagnostics are finite",
    ),
    GateSpecV1(
        "policy_reward_beats_fixed_comparator",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "at update 500, mean policy reward exceeds the fit-selected fixed comparator and the paired 95% confidence interval excludes zero",
    ),
    GateSpecV1(
        "oracle_gap_reduction",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "the update-0 to update-500 reduction in exact-oracle reward gap is positive and its paired 95% confidence interval excludes zero",
    ),
    GateSpecV1(
        "no_near_total_mode_collapse",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "the update-500 dominant deterministic mode fraction is at most the preregistered 0.95 engineering ceiling",
    ),
    GateSpecV1(
        "checkpoint_resume_bit_identity",
        GateClassification.INTEGRITY_ACCEPTANCE,
        "resume from update 250 is bit-identical to the uninterrupted update-500 path",
    ),
    GateSpecV1(
        "runtime_within_predeclared_bound",
        GateClassification.INTEGRITY_ACCEPTANCE,
        "recorded smoke wall time does not exceed the verifier-bound pre-launch budget",
    ),
)

if len({item.gate_id for item in FROZEN_GATE_SPECS}) != len(FROZEN_GATE_SPECS):
    raise RuntimeError("duplicate frozen continuation gate")


PREREGISTRATION_SHA256 = _canonical_sha256(
    {
        "config": FROZEN_CONFIG.to_canonical_dict(),
        "gates": [item.to_canonical_dict() for item in FROZEN_GATE_SPECS],
        "paired_interval_spec_sha256": PAIRED_INTERVAL_SPEC_SHA256,
        "record_type": "run4_smoke_preregistration_bundle_v1",
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class DiagnosticPanelManifestV1(_CanonicalRecord):
    """Identities a future verifier must supply; this class reads no rows."""

    validation_calibration_cell_ids: Tuple[str, ...]
    validation_calibration_cells_sha256: str
    fit_validation_scene_ids: Tuple[str, ...]
    fit_validation_scenes_sha256: str
    ordered_context_ids: Tuple[str, ...]
    ordered_context_rows_sha256: str
    ordered_context_group_ids: Tuple[str, ...]
    ordered_context_groups_sha256: str
    queue_kernel_binding_sha256: str
    widest_support_mode_ids: Tuple[int, ...]
    fixed_comparator_mode_id: int
    fixed_comparator_q_e4: int
    fixed_comparator_fit_selection_sha256: str
    exact_oracle_evaluator_sha256: str
    predeclared_runtime_limit_seconds: float
    runtime_budget_registration_sha256: str

    RECORD_TYPE = "run4_smoke_diagnostic_panel_manifest_v1"

    def __post_init__(self) -> None:
        _string_tuple(
            self.validation_calibration_cell_ids,
            "validation_calibration_cell_ids",
        )
        _string_tuple(self.fit_validation_scene_ids, "fit_validation_scene_ids")
        _string_tuple(self.ordered_context_ids, "ordered_context_ids")
        if (
            type(self.ordered_context_group_ids) is not tuple
            or len(self.ordered_context_group_ids) != len(self.ordered_context_ids)
        ):
            raise SmokePreregistrationError(
                "ordered_context_group_ids must align one-for-one with contexts"
            )
        for group_id in self.ordered_context_group_ids:
            _text(group_id, "ordered_context_group_id")
        if len(set(self.ordered_context_group_ids)) < 2:
            raise SmokePreregistrationError(
                "paired inference requires at least two independent context groups"
            )
        for name in (
            "validation_calibration_cells_sha256",
            "fit_validation_scenes_sha256",
            "ordered_context_rows_sha256",
            "ordered_context_groups_sha256",
            "queue_kernel_binding_sha256",
            "fixed_comparator_fit_selection_sha256",
            "exact_oracle_evaluator_sha256",
            "runtime_budget_registration_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            type(self.widest_support_mode_ids) is not tuple
            or not self.widest_support_mode_ids
        ):
            raise SmokePreregistrationError(
                "widest_support_mode_ids must be a non-empty tuple"
            )
        for mode_id in self.widest_support_mode_ids:
            if type(mode_id) is not int or not 0 <= mode_id < EXPECTED_MODE_COUNT:
                raise SmokePreregistrationError(
                    "widest-support mode is outside the registered mode range"
                )
        if len(set(self.widest_support_mode_ids)) != len(
            self.widest_support_mode_ids
        ):
            raise SmokePreregistrationError(
                "widest_support_mode_ids contains duplicates"
            )
        if (
            type(self.fixed_comparator_mode_id) is not int
            or not 0 <= self.fixed_comparator_mode_id < EXPECTED_MODE_COUNT
        ):
            raise SmokePreregistrationError(
                "fixed_comparator_mode_id is outside the registered mode range"
            )
        if (
            type(self.fixed_comparator_q_e4) is not int
            or not 0 <= self.fixed_comparator_q_e4 <= Q_E4_MAX
        ):
            raise SmokePreregistrationError(
                "fixed_comparator_q_e4 is outside the executable wire range"
            )
        _positive_float(
            self.predeclared_runtime_limit_seconds,
            "predeclared_runtime_limit_seconds",
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "fit_validation_scene_ids": list(self.fit_validation_scene_ids),
            "fit_validation_scene_partition": "FIT_VALIDATION_ONLY",
            "fit_validation_scenes_sha256": self.fit_validation_scenes_sha256,
            "fixed_comparator_fit_selection_partition": "FIT_ONLY",
            "fixed_comparator_fit_selection_sha256": (
                self.fixed_comparator_fit_selection_sha256
            ),
            "fixed_comparator_mode_id": self.fixed_comparator_mode_id,
            "fixed_comparator_q_e4": self.fixed_comparator_q_e4,
            "exact_oracle_evaluator_sha256": self.exact_oracle_evaluator_sha256,
            "ordered_context_ids": list(self.ordered_context_ids),
            "ordered_context_group_ids": list(self.ordered_context_group_ids),
            "ordered_context_groups_sha256": self.ordered_context_groups_sha256,
            "ordered_context_rows_sha256": self.ordered_context_rows_sha256,
            "paired_interval_spec_sha256": PAIRED_INTERVAL_SPEC_SHA256,
            "predeclared_runtime_limit_seconds": (
                self.predeclared_runtime_limit_seconds
            ),
            "queue_kernel_binding_sha256": self.queue_kernel_binding_sha256,
            "runtime_budget_registration_sha256": (
                self.runtime_budget_registration_sha256
            ),
            "validation_calibration_cell_ids": list(
                self.validation_calibration_cell_ids
            ),
            "validation_calibration_cells_sha256": (
                self.validation_calibration_cells_sha256
            ),
            "validation_calibration_partition": "VALIDATION_ONLY",
            "widest_support_mode_ids": list(self.widest_support_mode_ids),
        }


@dataclass(frozen=True, slots=True)
class CompositeVerifierEvidenceV1(_CanonicalRecord):
    """Claims checked by the future composite verifier, not by this module."""

    panel: DiagnosticPanelManifestV1
    composite_verifier_manifest_sha256: str
    calibration_cells_disjoint_from_fit: bool
    scenes_disjoint_from_fit: bool
    ordered_contexts_deterministic: bool
    observed_row_count: int
    fabricated_row_count: int
    kernel_binding_verified: bool
    identities_and_digests_verified: bool
    fixed_comparator_fit_only_verified: bool
    exact_oracle_verified: bool
    paired_interval_spec_verified: bool
    evidence_class: EvidenceClass = EvidenceClass.VERIFIED_COMPOSITE

    RECORD_TYPE = "run4_smoke_composite_verifier_evidence_v1"

    def __post_init__(self) -> None:
        if type(self.panel) is not DiagnosticPanelManifestV1:
            raise PanelBindingError("panel must be an exact panel manifest")
        _digest(
            self.composite_verifier_manifest_sha256,
            "composite_verifier_manifest_sha256",
        )
        for name in (
            "calibration_cells_disjoint_from_fit",
            "scenes_disjoint_from_fit",
            "ordered_contexts_deterministic",
            "kernel_binding_verified",
            "identities_and_digests_verified",
            "fixed_comparator_fit_only_verified",
            "exact_oracle_verified",
            "paired_interval_spec_verified",
        ):
            _strict_bool(getattr(self, name), name)
        _exact_int(self.observed_row_count, "observed_row_count", minimum=1)
        _exact_int(self.fabricated_row_count, "fabricated_row_count")
        if self.observed_row_count != len(self.panel.ordered_context_ids):
            raise PanelBindingError(
                "observed_row_count must equal the ordered context count"
            )
        if type(self.evidence_class) is not EvidenceClass:
            raise PanelBindingError("evidence_class has a foreign type")

    def require_verified_claims(self) -> None:
        if self.evidence_class is not EvidenceClass.VERIFIED_COMPOSITE:
            raise PanelBindingError("test-only evidence cannot bind production")
        if not self.calibration_cells_disjoint_from_fit:
            raise PanelBindingError("calibration validation cells overlap fit cells")
        if not self.scenes_disjoint_from_fit:
            raise PanelBindingError("fit-validation scenes overlap fit scenes")
        if not self.ordered_contexts_deterministic:
            raise PanelBindingError("diagnostic panel order is not deterministic")
        if self.fabricated_row_count != 0:
            raise PanelBindingError("fabricated diagnostic rows are forbidden")
        if not self.kernel_binding_verified:
            raise PanelBindingError("queue-kernel binding is not verified")
        if not self.identities_and_digests_verified:
            raise PanelBindingError("panel identities/digests are not verified")
        if not self.fixed_comparator_fit_only_verified:
            raise PanelBindingError(
                "fixed comparator was not verified as selected from fit only"
            )
        if not self.exact_oracle_verified:
            raise PanelBindingError("exact hybrid-action oracle is not verified")
        if not self.paired_interval_spec_verified:
            raise PanelBindingError("paired interval method is not verified")

    def _payload(self) -> Dict[str, Any]:
        return {
            "calibration_cells_disjoint_from_fit": (
                self.calibration_cells_disjoint_from_fit
            ),
            "composite_verifier_manifest_sha256": (
                self.composite_verifier_manifest_sha256
            ),
            "evidence_class": self.evidence_class.value,
            "fabricated_row_count": self.fabricated_row_count,
            "fixed_comparator_fit_only_verified": (
                self.fixed_comparator_fit_only_verified
            ),
            "exact_oracle_verified": self.exact_oracle_verified,
            "identities_and_digests_verified": (
                self.identities_and_digests_verified
            ),
            "kernel_binding_verified": self.kernel_binding_verified,
            "observed_row_count": self.observed_row_count,
            "ordered_contexts_deterministic": self.ordered_contexts_deterministic,
            "paired_interval_spec_verified": self.paired_interval_spec_verified,
            "panel": self.panel.to_canonical_dict(),
            "scenes_disjoint_from_fit": self.scenes_disjoint_from_fit,
        }


_PRODUCTION_BINDING_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _VerifiedPanelBindingV1(_CanonicalRecord):
    panel: DiagnosticPanelManifestV1
    verifier_evidence_sha256: str
    preregistration_sha256: str
    evidence_class: EvidenceClass
    _token: object

    RECORD_TYPE = "run4_smoke_verified_panel_binding_v1"

    def require_verified(self) -> None:
        if self._token is not _PRODUCTION_BINDING_TOKEN:
            raise PanelBindingError("panel binding token was not verifier-issued")
        if self.evidence_class is not EvidenceClass.VERIFIED_COMPOSITE:
            raise PanelBindingError("production panel has a test evidence class")
        _digest(self.verifier_evidence_sha256, "verifier_evidence_sha256")
        if self.preregistration_sha256 != PREREGISTRATION_SHA256:
            raise PanelBindingError("panel binds another preregistration")

    def _payload(self) -> Dict[str, Any]:
        return {
            "evidence_class": self.evidence_class.value,
            "panel_sha256": self.panel.canonical_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "verifier_evidence_sha256": self.verifier_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class _TestOnlyPanelBindingV1(_CanonicalRecord):
    panel: DiagnosticPanelManifestV1
    evidence_class: EvidenceClass = EvidenceClass.TEST_ONLY_SYNTHETIC

    RECORD_TYPE = "run4_smoke_test_only_panel_binding_v1"

    def __post_init__(self) -> None:
        if type(self.panel) is not DiagnosticPanelManifestV1:
            raise PanelBindingError("test panel has a foreign manifest type")
        if self.evidence_class is not EvidenceClass.TEST_ONLY_SYNTHETIC:
            raise PanelBindingError("test panel cannot claim production evidence")

    def _payload(self) -> Dict[str, Any]:
        return {
            "evidence_class": self.evidence_class.value,
            "panel_sha256": self.panel.canonical_sha256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }


def bind_verified_panel(
    evidence: CompositeVerifierEvidenceV1,
) -> _VerifiedPanelBindingV1:
    """Bind production diagnostics only after a reviewed manifest is pinned."""

    if type(evidence) is not CompositeVerifierEvidenceV1:
        raise PanelBindingError(
            "evidence must be an exact CompositeVerifierEvidenceV1"
        )
    evidence.require_verified_claims()
    registered = REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256
    if registered is None:
        raise PanelBindingError(
            "production smoke remains fail-closed: no composite verifier "
            "manifest is registered"
        )
    _digest(registered, "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256")
    if evidence.composite_verifier_manifest_sha256 != registered:
        raise PanelBindingError("composite verifier manifest digest mismatch")
    return _VerifiedPanelBindingV1(
        panel=evidence.panel,
        verifier_evidence_sha256=evidence.canonical_sha256,
        preregistration_sha256=PREREGISTRATION_SHA256,
        evidence_class=EvidenceClass.VERIFIED_COMPOSITE,
        _token=_PRODUCTION_BINDING_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class SensitivityDiagnosticV1(_CanonicalRecord):
    """One controlled-pair sensitivity result with uncertainty reported."""

    feature_name: str
    controlled_pair_count: int
    nonzero_response_count: int
    effect_estimate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    only_registered_feature_changed: bool
    report_sha256: str

    RECORD_TYPE = "run4_smoke_sensitivity_diagnostic_v1"

    def __post_init__(self) -> None:
        if self.feature_name not in SENSITIVITY_FEATURES:
            raise DiagnosticError(
                f"unregistered sensitivity feature {self.feature_name!r}"
            )
        _exact_int(self.controlled_pair_count, "controlled_pair_count", minimum=1)
        _exact_int(self.nonzero_response_count, "nonzero_response_count")
        if self.nonzero_response_count > self.controlled_pair_count:
            raise DiagnosticError("nonzero responses exceed controlled pairs")
        estimate = _finite(self.effect_estimate, "effect_estimate")
        lower = _finite(
            self.confidence_interval_lower, "confidence_interval_lower"
        )
        upper = _finite(
            self.confidence_interval_upper, "confidence_interval_upper"
        )
        confidence = _finite(self.confidence_level, "confidence_level")
        if lower > upper or not lower <= estimate <= upper:
            raise DiagnosticError(
                "effect estimate must lie inside its ordered confidence interval"
            )
        if not 0.0 < confidence < 1.0:
            raise DiagnosticError("confidence_level must lie in (0,1)")
        _strict_bool(
            self.only_registered_feature_changed,
            "only_registered_feature_changed",
        )
        _digest(self.report_sha256, "report_sha256")

    @property
    def supports_nonzero_response(self) -> bool:
        interval_excludes_zero = (
            self.confidence_interval_lower > 0.0
            or self.confidence_interval_upper < 0.0
        )
        return (
            self.only_registered_feature_changed
            and self.nonzero_response_count > 0
            and self.effect_estimate != 0.0
            and interval_excludes_zero
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "confidence_level": self.confidence_level,
            "controlled_pair_count": self.controlled_pair_count,
            "effect_estimate": self.effect_estimate,
            "feature_name": self.feature_name,
            "nonzero_response_count": self.nonzero_response_count,
            "only_registered_feature_changed": (
                self.only_registered_feature_changed
            ),
            "report_sha256": self.report_sha256,
        }


def _metric_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


@dataclass(frozen=True, slots=True)
class CheckpointDiagnosticV1(_CanonicalRecord):
    """One deterministic-policy evaluation on the fixed held panel.

    Regret decomposition is defined on the same per-context reward table:
    ``total = discrete_mode + continuous_q``.  The exact oracle enumerates the
    registered hybrid action support; it is not selected from the validation
    panel.  ``q`` statistics use executed ``q_e4 / 10000`` values.
    """

    update: int
    panel_context_count: int
    expected_policy_reward: float
    fixed_comparator_reward: float
    oracle_reward: float
    oracle_gap: float
    deadline_miss_rate: float
    fixed_comparator_deadline_miss_rate: float
    mean_action_qperc: float
    successful_feedback_latency_p50_ms: float
    successful_feedback_latency_p95_ms: float
    critic_rank_correlation: float
    deterministic_mode_counts: Tuple[int, ...]
    q_mean: float
    q_std: float
    q_support_boundary_hit_rate: float
    total_regret: float
    discrete_mode_regret: float
    continuous_q_regret: float
    context_rows_sha256: str

    RECORD_TYPE = "run4_smoke_checkpoint_diagnostic_v1"

    def __post_init__(self) -> None:
        _exact_int(self.update, "update")
        if self.update not in FROZEN_CONFIG.smoke_checkpoint_updates:
            raise DiagnosticError("checkpoint update is outside 0/100/250/500")
        _exact_int(self.panel_context_count, "panel_context_count", minimum=1)
        scalar_names = (
            "expected_policy_reward",
            "fixed_comparator_reward",
            "oracle_reward",
            "oracle_gap",
            "deadline_miss_rate",
            "fixed_comparator_deadline_miss_rate",
            "mean_action_qperc",
            "successful_feedback_latency_p50_ms",
            "successful_feedback_latency_p95_ms",
            "critic_rank_correlation",
            "q_mean",
            "q_std",
            "q_support_boundary_hit_rate",
            "total_regret",
            "discrete_mode_regret",
            "continuous_q_regret",
        )
        for name in scalar_names:
            _finite(getattr(self, name), name)
        for name in (
            "deadline_miss_rate",
            "fixed_comparator_deadline_miss_rate",
            "mean_action_qperc",
            "q_support_boundary_hit_rate",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise DiagnosticError(f"{name} must lie in [0,1]")
        if not -1.0 <= self.critic_rank_correlation <= 1.0:
            raise DiagnosticError("critic_rank_correlation must lie in [-1,1]")
        if (
            self.successful_feedback_latency_p50_ms < 0.0
            or self.successful_feedback_latency_p95_ms
            < self.successful_feedback_latency_p50_ms
        ):
            raise DiagnosticError("latency percentiles must satisfy 0 <= P50 <= P95")
        if not 0.0 <= self.q_mean <= Q_MAX:
            raise DiagnosticError("q_mean is outside the executable q range")
        if not 0.0 <= self.q_std <= Q_MAX / 2.0:
            raise DiagnosticError("q_std exceeds the bound for q in [0,Q_MAX]")
        for name in (
            "oracle_gap",
            "total_regret",
            "discrete_mode_regret",
            "continuous_q_regret",
        ):
            if getattr(self, name) < 0.0:
                raise DiagnosticError(f"{name} cannot be negative")
        if self.expected_policy_reward > self.oracle_reward + 1e-12:
            raise DiagnosticError("policy reward cannot exceed exact-oracle reward")
        if self.fixed_comparator_reward > self.oracle_reward + 1e-12:
            raise DiagnosticError("fixed comparator cannot exceed exact-oracle reward")
        if not _metric_close(
            self.oracle_gap,
            self.oracle_reward - self.expected_policy_reward,
        ):
            raise DiagnosticError("oracle_gap contradicts oracle minus policy reward")
        if not _metric_close(self.total_regret, self.oracle_gap):
            raise DiagnosticError("total_regret must equal exact-oracle reward gap")
        if not _metric_close(
            self.total_regret,
            self.discrete_mode_regret + self.continuous_q_regret,
        ):
            raise DiagnosticError(
                "discrete and continuous regret must add to total regret"
            )
        if (
            type(self.deterministic_mode_counts) is not tuple
            or len(self.deterministic_mode_counts) != EXPECTED_MODE_COUNT
        ):
            raise DiagnosticError(
                f"deterministic_mode_counts must contain {EXPECTED_MODE_COUNT} values"
            )
        for count in self.deterministic_mode_counts:
            _exact_int(count, "deterministic mode count")
        if sum(self.deterministic_mode_counts) != self.panel_context_count:
            raise DiagnosticError("deterministic mode counts do not cover the panel")
        _digest(self.context_rows_sha256, "context_rows_sha256")

    @property
    def dominant_mode_fraction(self) -> float:
        return max(self.deterministic_mode_counts) / self.panel_context_count

    @property
    def mode_entropy_nats(self) -> float:
        probabilities = (
            count / self.panel_context_count
            for count in self.deterministic_mode_counts
            if count > 0
        )
        return -sum(probability * math.log(probability) for probability in probabilities)

    def _payload(self) -> Dict[str, Any]:
        return {
            "context_rows_sha256": self.context_rows_sha256,
            "continuous_q_regret": self.continuous_q_regret,
            "critic_rank_correlation": self.critic_rank_correlation,
            "deadline_miss_rate": self.deadline_miss_rate,
            "deadline_miss_rate_population": "all ordered panel contexts",
            "deterministic_mode_counts": list(self.deterministic_mode_counts),
            "discrete_mode_regret": self.discrete_mode_regret,
            "dominant_mode_fraction": self.dominant_mode_fraction,
            "expected_policy_reward": self.expected_policy_reward,
            "fixed_comparator_deadline_miss_rate": (
                self.fixed_comparator_deadline_miss_rate
            ),
            "fixed_comparator_reward": self.fixed_comparator_reward,
            "mean_action_qperc": self.mean_action_qperc,
            "mean_action_qperc_population": "all ordered panel contexts",
            "mode_entropy_nats": self.mode_entropy_nats,
            "oracle_gap": self.oracle_gap,
            "oracle_reward": self.oracle_reward,
            "q_mean": self.q_mean,
            "q_statistic_units": "executed_q_e4_div_10000",
            "q_std": self.q_std,
            "q_support_boundary_definition": (
                "executed q_e4 equals either registered support endpoint for its mode"
            ),
            "q_support_boundary_hit_rate": self.q_support_boundary_hit_rate,
            "regret_decomposition_definition": (
                "continuous_q = best-q reward within policy-selected mode minus "
                "policy reward; discrete_mode = exact-oracle reward minus that "
                "best-within-selected-mode reward"
            ),
            "reward_statistic": "arithmetic mean over exact ordered panel contexts",
            "successful_feedback_latency_p50_ms": (
                self.successful_feedback_latency_p50_ms
            ),
            "successful_feedback_latency_p95_ms": (
                self.successful_feedback_latency_p95_ms
            ),
            "successful_feedback_latency_population": (
                "registered-success contexts only; deadline_miss_rate reports the "
                "full population and must be shown alongside these percentiles"
            ),
            "total_regret": self.total_regret,
            "update": self.update,
        }


@dataclass(frozen=True, slots=True)
class SmokeDiagnosticsV1(_CanonicalRecord):
    """Complete fixed-panel metrics required at the 500-update stop."""

    preregistration_sha256: str
    panel_sha256: str
    seed: int
    checkpoint_diagnostics: Tuple[CheckpointDiagnosticV1, ...]
    observed_mode_q_bins: Tuple[Tuple[int, int], ...]
    registered_success_count: int
    registered_failure_count: int
    all_numerics_finite: bool
    policy_minus_fixed_ci_lower: float
    policy_minus_fixed_ci_upper: float
    oracle_gap_reduction_ci_lower: float
    oracle_gap_reduction_ci_upper: float
    paired_confidence_level: float
    paired_interval_spec_sha256: str
    sensitivity_diagnostics: Tuple[SensitivityDiagnosticV1, ...]
    checkpoint_resume_from_update: int
    checkpoint_resume_bit_identical: bool
    runtime_seconds: float
    checkpoint_manifest_sha256: str
    diagnostic_report_sha256: str

    RECORD_TYPE = "run4_smoke_diagnostics_v1"

    def __post_init__(self) -> None:
        _digest(self.preregistration_sha256, "preregistration_sha256")
        _digest(self.panel_sha256, "panel_sha256")
        _exact_int(self.seed, "seed")
        if (
            type(self.checkpoint_diagnostics) is not tuple
            or any(type(item) is not CheckpointDiagnosticV1 for item in self.checkpoint_diagnostics)
            or tuple(item.update for item in self.checkpoint_diagnostics)
            != FROZEN_CONFIG.smoke_checkpoint_updates
        ):
            raise DiagnosticError(
                "checkpoint diagnostics must contain ordered updates 0/100/250/500"
            )
        contexts = {item.panel_context_count for item in self.checkpoint_diagnostics}
        row_digests = {item.context_rows_sha256 for item in self.checkpoint_diagnostics}
        if len(contexts) != 1 or len(row_digests) != 1:
            raise DiagnosticError("all checkpoints must evaluate the same exact panel rows")
        fixed_rewards = {item.fixed_comparator_reward for item in self.checkpoint_diagnostics}
        fixed_misses = {
            item.fixed_comparator_deadline_miss_rate
            for item in self.checkpoint_diagnostics
        }
        oracle_rewards = {item.oracle_reward for item in self.checkpoint_diagnostics}
        if len(fixed_rewards) != 1 or len(fixed_misses) != 1 or len(oracle_rewards) != 1:
            raise DiagnosticError(
                "fixed comparator and exact oracle must be checkpoint-invariant"
            )
        if type(self.observed_mode_q_bins) is not tuple:
            raise DiagnosticError("observed_mode_q_bins must be a tuple")
        for pair in self.observed_mode_q_bins:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not int
                or type(pair[1]) is not int
                or not 0 <= pair[0] < FROZEN_CONFIG.warmup_mode_count
                or not 0 <= pair[1] < FROZEN_CONFIG.warmup_q_bin_count
            ):
                raise DiagnosticError("invalid observed mode/q-bin identity")
        if len(set(self.observed_mode_q_bins)) != len(self.observed_mode_q_bins):
            raise DiagnosticError("duplicate observed mode/q-bin identity")
        _exact_int(self.registered_success_count, "registered_success_count")
        _exact_int(self.registered_failure_count, "registered_failure_count")
        _strict_bool(self.all_numerics_finite, "all_numerics_finite")
        for name in (
            "policy_minus_fixed_ci_lower",
            "policy_minus_fixed_ci_upper",
            "oracle_gap_reduction_ci_lower",
            "oracle_gap_reduction_ci_upper",
            "paired_confidence_level",
            "runtime_seconds",
        ):
            _finite(getattr(self, name), name)
        if self.policy_minus_fixed_ci_lower > self.policy_minus_fixed_ci_upper:
            raise DiagnosticError("policy-minus-fixed confidence interval is reversed")
        if self.oracle_gap_reduction_ci_lower > self.oracle_gap_reduction_ci_upper:
            raise DiagnosticError("oracle-gap-reduction confidence interval is reversed")
        if self.paired_confidence_level != FROZEN_CONFIG.paired_confidence_level:
            raise DiagnosticError("paired confidence level differs from preregistration")
        _digest(self.paired_interval_spec_sha256, "paired_interval_spec_sha256")
        if self.paired_interval_spec_sha256 != PAIRED_INTERVAL_SPEC_SHA256:
            raise DiagnosticError("paired interval method differs from preregistration")
        update0 = self.checkpoint(0)
        update500 = self.checkpoint(500)
        fixed_advantage = (
            update500.expected_policy_reward - update500.fixed_comparator_reward
        )
        oracle_gap_reduction = update0.oracle_gap - update500.oracle_gap
        if not (
            self.policy_minus_fixed_ci_lower
            <= fixed_advantage
            <= self.policy_minus_fixed_ci_upper
        ):
            raise DiagnosticError(
                "policy-minus-fixed estimate lies outside its confidence interval"
            )
        if not (
            self.oracle_gap_reduction_ci_lower
            <= oracle_gap_reduction
            <= self.oracle_gap_reduction_ci_upper
        ):
            raise DiagnosticError(
                "oracle-gap-reduction estimate lies outside its confidence interval"
            )
        if self.runtime_seconds < 0.0:
            raise DiagnosticError("runtime_seconds cannot be negative")
        if (
            type(self.sensitivity_diagnostics) is not tuple
            or len(self.sensitivity_diagnostics) != len(SENSITIVITY_FEATURES)
            or any(
                type(item) is not SensitivityDiagnosticV1
                for item in self.sensitivity_diagnostics
            )
            or {item.feature_name for item in self.sensitivity_diagnostics}
            != set(SENSITIVITY_FEATURES)
        ):
            raise DiagnosticError(
                "sensitivity diagnostics must contain every registered feature once"
            )
        _exact_int(
            self.checkpoint_resume_from_update,
            "checkpoint_resume_from_update",
        )
        if self.checkpoint_resume_from_update != 250:
            raise DiagnosticError("resume identity test must restart at update 250")
        _strict_bool(
            self.checkpoint_resume_bit_identical,
            "checkpoint_resume_bit_identical",
        )
        _digest(self.checkpoint_manifest_sha256, "checkpoint_manifest_sha256")
        _digest(self.diagnostic_report_sha256, "diagnostic_report_sha256")

    @property
    def panel_context_count(self) -> int:
        return self.checkpoint_diagnostics[0].panel_context_count

    @property
    def evaluated_checkpoint_updates(self) -> Tuple[int, ...]:
        return tuple(item.update for item in self.checkpoint_diagnostics)

    def checkpoint(self, update: int) -> CheckpointDiagnosticV1:
        for item in self.checkpoint_diagnostics:
            if item.update == update:
                return item
        raise DiagnosticError(f"checkpoint {update} is not present")

    def _payload(self) -> Dict[str, Any]:
        return {
            "all_numerics_finite": self.all_numerics_finite,
            "checkpoint_diagnostics": [
                item.to_canonical_dict() for item in self.checkpoint_diagnostics
            ],
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_resume_bit_identical": self.checkpoint_resume_bit_identical,
            "checkpoint_resume_from_update": self.checkpoint_resume_from_update,
            "diagnostic_report_sha256": self.diagnostic_report_sha256,
            "evaluated_checkpoint_updates": list(self.evaluated_checkpoint_updates),
            "observed_mode_q_bins": [list(pair) for pair in self.observed_mode_q_bins],
            "oracle_gap_reduction_ci_lower": self.oracle_gap_reduction_ci_lower,
            "oracle_gap_reduction_ci_upper": self.oracle_gap_reduction_ci_upper,
            "paired_confidence_level": self.paired_confidence_level,
            "paired_interval_spec_sha256": self.paired_interval_spec_sha256,
            "panel_context_count": self.panel_context_count,
            "panel_sha256": self.panel_sha256,
            "policy_minus_fixed_ci_lower": self.policy_minus_fixed_ci_lower,
            "policy_minus_fixed_ci_upper": self.policy_minus_fixed_ci_upper,
            "preregistration_sha256": self.preregistration_sha256,
            "registered_failure_count": self.registered_failure_count,
            "registered_success_count": self.registered_success_count,
            "runtime_seconds": self.runtime_seconds,
            "seed": self.seed,
            "sensitivity_diagnostics": [
                item.to_canonical_dict()
                for item in sorted(
                    self.sensitivity_diagnostics,
                    key=lambda item: item.feature_name,
                )
            ],
            "sensitivity_interpretation": (
                "diagnostic_only; never a continuation gate"
            ),
        }


@dataclass(frozen=True, slots=True)
class GateResultV1(_CanonicalRecord):
    gate_id: str
    classification: GateClassification
    passed: bool
    explanation: str

    RECORD_TYPE = "run4_smoke_gate_result_v1"

    def __post_init__(self) -> None:
        _text(self.gate_id, "gate_id")
        if type(self.classification) is not GateClassification:
            raise DiagnosticError("gate classification has a foreign type")
        _strict_bool(self.passed, "passed")
        _text(self.explanation, "explanation")

    def _payload(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.value,
            "explanation": self.explanation,
            "gate_id": self.gate_id,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class SmokeAssessmentV1(_CanonicalRecord):
    evidence_class: EvidenceClass
    diagnostics_sha256: str
    gate_results: Tuple[GateResultV1, ...]
    all_gates_pass: bool
    continuation_authorized: bool

    RECORD_TYPE = "run4_smoke_assessment_v1"

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not EvidenceClass:
            raise DiagnosticError("assessment evidence class has a foreign type")
        _digest(self.diagnostics_sha256, "diagnostics_sha256")
        if (
            type(self.gate_results) is not tuple
            or len(self.gate_results) != len(FROZEN_GATE_SPECS)
            or any(type(item) is not GateResultV1 for item in self.gate_results)
        ):
            raise DiagnosticError("assessment lacks the exact frozen gate set")
        if tuple(item.gate_id for item in self.gate_results) != tuple(
            item.gate_id for item in FROZEN_GATE_SPECS
        ):
            raise DiagnosticError("assessment gate order differs from preregistration")
        _strict_bool(self.all_gates_pass, "all_gates_pass")
        _strict_bool(self.continuation_authorized, "continuation_authorized")
        if self.all_gates_pass != all(item.passed for item in self.gate_results):
            raise DiagnosticError("all_gates_pass contradicts gate results")
        expected_authorized = (
            self.evidence_class is EvidenceClass.VERIFIED_COMPOSITE
            and self.all_gates_pass
        )
        if self.continuation_authorized != expected_authorized:
            raise DiagnosticError(
                "only a verified, all-pass assessment can authorize continuation"
            )

    def require_continuation_authorized(self) -> None:
        if not self.continuation_authorized:
            failed = [item.gate_id for item in self.gate_results if not item.passed]
            raise ContinuationRefused(
                "Run-4 continuation is not authorized; failed gates="
                f"{failed}, evidence_class={self.evidence_class.value}"
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "all_gates_pass": self.all_gates_pass,
            "continuation_authorized": self.continuation_authorized,
            "diagnostics_sha256": self.diagnostics_sha256,
            "evidence_class": self.evidence_class.value,
            "gate_results": [item.to_canonical_dict() for item in self.gate_results],
            "interpretation": (
                "bounded continuation decision only; not convergence, test-set, "
                "deployment-readiness or paper-performance evidence"
            ),
        }


def _gate_results(
    panel: DiagnosticPanelManifestV1,
    diagnostics: SmokeDiagnosticsV1,
) -> Tuple[GateResultV1, ...]:
    if diagnostics.preregistration_sha256 != PREREGISTRATION_SHA256:
        raise DiagnosticError("diagnostics bind another preregistration")
    if diagnostics.panel_sha256 != panel.canonical_sha256:
        raise DiagnosticError("diagnostics bind another diagnostic panel")
    if diagnostics.seed != FROZEN_CONFIG.initial_smoke_seed:
        raise DiagnosticError("pre-continuation smoke must use seed 17")
    if diagnostics.panel_context_count != len(panel.ordered_context_ids):
        raise DiagnosticError("diagnostic context count differs from panel identity")

    expected_bins = {
        (mode_id, q_bin)
        for mode_id in range(FROZEN_CONFIG.warmup_mode_count)
        for q_bin in range(FROZEN_CONFIG.warmup_q_bin_count)
    }
    observed_bins = set(diagnostics.observed_mode_q_bins)
    coverage = observed_bins == expected_bins
    successes_and_failures = (
        diagnostics.registered_success_count > 0
        and diagnostics.registered_failure_count > 0
    )
    update0 = diagnostics.checkpoint(0)
    update500 = diagnostics.checkpoint(500)
    policy_beats_fixed = (
        update500.expected_policy_reward > update500.fixed_comparator_reward
        and diagnostics.policy_minus_fixed_ci_lower > 0.0
    )
    oracle_gap_reduced = (
        update500.oracle_gap < update0.oracle_gap
        and diagnostics.oracle_gap_reduction_ci_lower > 0.0
    )
    no_near_total_mode_collapse = (
        update500.dominant_mode_fraction
        <= FROZEN_CONFIG.max_dominant_mode_fraction
    )
    runtime_pass = (
        diagnostics.runtime_seconds <= panel.predeclared_runtime_limit_seconds
    )

    values = (
        coverage,
        successes_and_failures,
        diagnostics.all_numerics_finite,
        policy_beats_fixed,
        oracle_gap_reduced,
        no_near_total_mode_collapse,
        diagnostics.checkpoint_resume_bit_identical,
        runtime_pass,
    )
    explanations = (
        f"observed {len(observed_bins)}/{len(expected_bins)} mode/q bins; "
        "highest bin is included by exact-set equality",
        f"success={diagnostics.registered_success_count}, "
        f"failure={diagnostics.registered_failure_count}",
        f"all_numerics_finite={diagnostics.all_numerics_finite}",
        "update-500 policy minus fit-selected fixed comparator="
        f"{update500.expected_policy_reward - update500.fixed_comparator_reward}; "
        f"paired {diagnostics.paired_confidence_level:.0%} CI="
        f"[{diagnostics.policy_minus_fixed_ci_lower}, "
        f"{diagnostics.policy_minus_fixed_ci_upper}]",
        "exact-oracle gap update 0 -> 500: "
        f"{update0.oracle_gap} -> {update500.oracle_gap}; reduction paired "
        f"{diagnostics.paired_confidence_level:.0%} CI="
        f"[{diagnostics.oracle_gap_reduction_ci_lower}, "
        f"{diagnostics.oracle_gap_reduction_ci_upper}]",
        "update-500 dominant-mode fraction="
        f"{update500.dominant_mode_fraction} <= preregistered "
        f"{FROZEN_CONFIG.max_dominant_mode_fraction}; mode entropy="
        f"{update500.mode_entropy_nats} nats",
        "resume at update 250 is bit-identical="
        f"{diagnostics.checkpoint_resume_bit_identical}",
        f"runtime={diagnostics.runtime_seconds}s <= preregistered "
        f"{panel.predeclared_runtime_limit_seconds}s",
    )
    return tuple(
        GateResultV1(
            gate_id=spec.gate_id,
            classification=spec.classification,
            passed=bool(passed),
            explanation=explanation,
        )
        for spec, passed, explanation in zip(
            FROZEN_GATE_SPECS, values, explanations
        )
    )


def assess_verified_smoke(
    binding: _VerifiedPanelBindingV1,
    diagnostics: SmokeDiagnosticsV1,
) -> SmokeAssessmentV1:
    """Assess a real smoke only through the verifier-issued binding."""

    if type(binding) is not _VerifiedPanelBindingV1:
        raise PanelBindingError(
            "production assessment requires the exact private verified binding"
        )
    binding.require_verified()
    if type(diagnostics) is not SmokeDiagnosticsV1:
        raise DiagnosticError("diagnostics must be an exact SmokeDiagnosticsV1")
    results = _gate_results(binding.panel, diagnostics)
    passed = all(item.passed for item in results)
    return SmokeAssessmentV1(
        evidence_class=EvidenceClass.VERIFIED_COMPOSITE,
        diagnostics_sha256=diagnostics.canonical_sha256,
        gate_results=results,
        all_gates_pass=passed,
        continuation_authorized=passed,
    )


def _assess_test_only_smoke(
    binding: _TestOnlyPanelBindingV1,
    diagnostics: SmokeDiagnosticsV1,
) -> SmokeAssessmentV1:
    """Exercise gate mechanics without ever issuing continuation authority."""

    if type(binding) is not _TestOnlyPanelBindingV1:
        raise PanelBindingError("test assessment requires the exact test binding")
    if type(diagnostics) is not SmokeDiagnosticsV1:
        raise DiagnosticError("diagnostics must be an exact SmokeDiagnosticsV1")
    results = _gate_results(binding.panel, diagnostics)
    return SmokeAssessmentV1(
        evidence_class=EvidenceClass.TEST_ONLY_SYNTHETIC,
        diagnostics_sha256=diagnostics.canonical_sha256,
        gate_results=results,
        all_gates_pass=all(item.passed for item in results),
        continuation_authorized=False,
    )
