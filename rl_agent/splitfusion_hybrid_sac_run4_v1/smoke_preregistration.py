"""Prospective, fail-closed preregistration for the Run-4 SAC smoke run.

This module freezes the 500-update smoke configuration and its continuation
questions *before* any smoke training is run.  It contains no runner, reads no
evidence, initializes no accelerator and cannot manufacture a diagnostic
panel.  Production assessment stays impossible until a future composite
verifier binds real, disjoint validation-calibration cells, fit-validation
scenes, an accepted sequential-kernel identity and a pre-launch runtime
budget.

The directional learning gates intentionally have no invented effect-size
threshold.  They ask only whether a metric moved in the preregistered direction
on the fixed panel.  Such gates are labelled ``HYPOTHESIS_DIAGNOSTIC`` and must
be accompanied by confidence/report artifacts; passing them is a bounded
continuation decision, not a convergence or deployment claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
)

__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "FROZEN_CONFIG",
    "FROZEN_GATE_SPECS",
    "PREREGISTRATION_SHA256",
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
            "batch_size",
            "replay_capacity",
            "warmup_mode_count",
            "warmup_q_bin_count",
            "warmup_samples_per_mode_q_bin",
            "warmup_decision_count",
            "environment_transitions_per_update",
            "torch_intraop_threads",
            "smoke_stop_update",
        ):
            _exact_int(getattr(self, name), name, minimum=1)
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
        "critic_rank_direction",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "fixed-panel critic rank correlation at update 500 exceeds update 0",
    ),
    GateSpecV1(
        "action_regret_direction",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "fixed-panel action regret at update 500 is below update 0",
    ),
    GateSpecV1(
        "no_widest_support_mode_pinning",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "no widest-support mode is the deterministic choice for every panel context",
    ),
    GateSpecV1(
        "continuous_q_regret_direction",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "fixed-panel continuous-q regret at update 500 is below update 0",
    ),
    GateSpecV1(
        "controlled_state_sensitivity",
        GateClassification.HYPOTHESIS_DIAGNOSTIC,
        "controlled pairs show a nonzero, confidence-reported response to each registered feature",
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
    queue_kernel_binding_sha256: str
    widest_support_mode_ids: Tuple[int, ...]
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
        for name in (
            "validation_calibration_cells_sha256",
            "fit_validation_scenes_sha256",
            "ordered_context_rows_sha256",
            "queue_kernel_binding_sha256",
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
        _positive_float(
            self.predeclared_runtime_limit_seconds,
            "predeclared_runtime_limit_seconds",
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "fit_validation_scene_ids": list(self.fit_validation_scene_ids),
            "fit_validation_scene_partition": "FIT_VALIDATION_ONLY",
            "fit_validation_scenes_sha256": self.fit_validation_scenes_sha256,
            "ordered_context_ids": list(self.ordered_context_ids),
            "ordered_context_rows_sha256": self.ordered_context_rows_sha256,
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
            "identities_and_digests_verified": (
                self.identities_and_digests_verified
            ),
            "kernel_binding_verified": self.kernel_binding_verified,
            "observed_row_count": self.observed_row_count,
            "ordered_contexts_deterministic": self.ordered_contexts_deterministic,
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


@dataclass(frozen=True, slots=True)
class SmokeDiagnosticsV1(_CanonicalRecord):
    """Metrics required at the prospective 500-update stop."""

    preregistration_sha256: str
    panel_sha256: str
    seed: int
    evaluated_checkpoint_updates: Tuple[int, ...]
    observed_mode_q_bins: Tuple[Tuple[int, int], ...]
    registered_success_count: int
    registered_failure_count: int
    all_numerics_finite: bool
    critic_rank_correlation_update0: float
    critic_rank_correlation_update500: float
    action_regret_update0: float
    action_regret_update500: float
    continuous_q_regret_update0: float
    continuous_q_regret_update500: float
    deterministic_mode_counts: Tuple[int, ...]
    panel_context_count: int
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
        if type(self.evaluated_checkpoint_updates) is not tuple:
            raise DiagnosticError("evaluated checkpoints must be a tuple")
        if self.evaluated_checkpoint_updates != FROZEN_CONFIG.smoke_checkpoint_updates:
            raise DiagnosticError(
                "smoke diagnostics must contain checkpoints 0/100/250/500"
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
            "critic_rank_correlation_update0",
            "critic_rank_correlation_update500",
            "action_regret_update0",
            "action_regret_update500",
            "continuous_q_regret_update0",
            "continuous_q_regret_update500",
            "runtime_seconds",
        ):
            _finite(getattr(self, name), name)
        for name in (
            "critic_rank_correlation_update0",
            "critic_rank_correlation_update500",
        ):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise DiagnosticError(f"{name} must lie in [-1,1]")
        for name in (
            "action_regret_update0",
            "action_regret_update500",
            "continuous_q_regret_update0",
            "continuous_q_regret_update500",
            "runtime_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise DiagnosticError(f"{name} cannot be negative")
        if (
            type(self.deterministic_mode_counts) is not tuple
            or len(self.deterministic_mode_counts) != EXPECTED_MODE_COUNT
        ):
            raise DiagnosticError(
                f"deterministic_mode_counts must contain {EXPECTED_MODE_COUNT} values"
            )
        for count in self.deterministic_mode_counts:
            _exact_int(count, "deterministic mode count")
        _exact_int(self.panel_context_count, "panel_context_count", minimum=1)
        if sum(self.deterministic_mode_counts) != self.panel_context_count:
            raise DiagnosticError("deterministic mode counts do not cover the panel")
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

    def _payload(self) -> Dict[str, Any]:
        return {
            "action_regret_update0": self.action_regret_update0,
            "action_regret_update500": self.action_regret_update500,
            "all_numerics_finite": self.all_numerics_finite,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_resume_bit_identical": (
                self.checkpoint_resume_bit_identical
            ),
            "checkpoint_resume_from_update": self.checkpoint_resume_from_update,
            "continuous_q_regret_update0": self.continuous_q_regret_update0,
            "continuous_q_regret_update500": self.continuous_q_regret_update500,
            "critic_rank_correlation_update0": (
                self.critic_rank_correlation_update0
            ),
            "critic_rank_correlation_update500": (
                self.critic_rank_correlation_update500
            ),
            "deterministic_mode_counts": list(self.deterministic_mode_counts),
            "diagnostic_report_sha256": self.diagnostic_report_sha256,
            "evaluated_checkpoint_updates": list(
                self.evaluated_checkpoint_updates
            ),
            "observed_mode_q_bins": [
                list(pair) for pair in self.observed_mode_q_bins
            ],
            "panel_context_count": self.panel_context_count,
            "panel_sha256": self.panel_sha256,
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
    rank_improved = (
        diagnostics.critic_rank_correlation_update500
        > diagnostics.critic_rank_correlation_update0
    )
    action_regret_improved = (
        diagnostics.action_regret_update500 < diagnostics.action_regret_update0
    )
    widest_pinned = any(
        diagnostics.deterministic_mode_counts[mode_id]
        == diagnostics.panel_context_count
        for mode_id in panel.widest_support_mode_ids
    )
    continuous_improved = (
        diagnostics.continuous_q_regret_update500
        < diagnostics.continuous_q_regret_update0
    )
    sensitivity_pass = all(
        item.supports_nonzero_response
        for item in diagnostics.sensitivity_diagnostics
    )
    runtime_pass = (
        diagnostics.runtime_seconds <= panel.predeclared_runtime_limit_seconds
    )

    values = (
        coverage,
        successes_and_failures,
        diagnostics.all_numerics_finite,
        rank_improved,
        action_regret_improved,
        not widest_pinned,
        continuous_improved,
        sensitivity_pass,
        diagnostics.checkpoint_resume_bit_identical,
        runtime_pass,
    )
    explanations = (
        f"observed {len(observed_bins)}/{len(expected_bins)} mode/q bins; "
        "highest bin is included by exact-set equality",
        f"success={diagnostics.registered_success_count}, "
        f"failure={diagnostics.registered_failure_count}",
        f"all_numerics_finite={diagnostics.all_numerics_finite}",
        "Spearman rank direction: "
        f"{diagnostics.critic_rank_correlation_update0} -> "
        f"{diagnostics.critic_rank_correlation_update500}",
        "action regret direction: "
        f"{diagnostics.action_regret_update0} -> "
        f"{diagnostics.action_regret_update500}",
        "widest-support deterministic counts="
        + repr(
            {
                mode_id: diagnostics.deterministic_mode_counts[mode_id]
                for mode_id in panel.widest_support_mode_ids
            }
        )
        + f" / {diagnostics.panel_context_count}",
        "continuous-q regret direction: "
        f"{diagnostics.continuous_q_regret_update0} -> "
        f"{diagnostics.continuous_q_regret_update500}",
        "controlled sensitivity passed for "
        f"{sum(item.supports_nonzero_response for item in diagnostics.sensitivity_diagnostics)}"
        f"/{len(SENSITIVITY_FEATURES)} registered inputs",
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

