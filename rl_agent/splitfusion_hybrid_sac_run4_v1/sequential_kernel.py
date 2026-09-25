"""Evidence-gated sequential radio/queue mechanics for Run 4.

This module is intentionally a *reducer*, not a fitted radio model.  A future
fitter must produce :class:`EmpiricalStepPredictionV1` records from the
reviewed corrected-v2 12-cell evidence.  The reducer then verifies exact
action/payload identity, support, calibration partition, latency provenance
and causal sequence before it advances the backlog/MCS state.

No coefficient, profile label, default latency or stochastic channel process
is defined here.  In particular, the stale ``analysis_v1.json`` and its
``decisions.csv`` are structurally ineligible: the provenance contract requires
an accepted generation-2 analysis and a separately pinned corrected decision
table.  Production authorization remains fail-closed until the reviewed
prerequisite digest is registered below.

The 170-ms reward clock begins at action open.  Successful feedback latency is
the exact sum of six contiguous per-frame intervals: UE action path, feature
uplink, edge pre-model work, model-tail inference, post-model feedback
preparation and feedback downlink.  The boundaries are first feature-datagram
send, complete edge reassembly, model dispatch, model-ready, feedback send and
UE feedback receipt.  Sensor callback/preparation precedes this clock;
post-feedback map service follows it.  Percentiles are never added to
manufacture an end-to-end sample.

Importing this module performs no I/O and launches no runtime component.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import quality_adapter
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

__all__ = [
    "SequentialKernelError",
    "EvidenceBindingError",
    "CorrectedEvidenceUnavailable",
    "SupportViolation",
    "SequenceViolation",
    "PredictionViolation",
    "CheckpointError",
    "KernelCalibrationPartition",
    "KernelTerminalKind",
    "KernelAuthorizationClass",
    "IntegerObservationV1",
    "RadioQueueStateV1",
    "NumericSupportV1",
    "FitValidationSplitV1",
    "LatencySupportV1",
    "KernelSupportV1",
    "KernelProvenanceBindingV1",
    "KernelVerifierPrerequisitesV1",
    "KernelAuthorizationV1",
    "verify_kernel_prerequisites",
    "KernelDecisionInputV1",
    "PredictionPayloadV1",
    "PredictionModelInputV1",
    "PredictionRequestV1",
    "FeedbackLatencyBreakdownV1",
    "PredictedIntegerObservationV1",
    "EmpiricalModelForecastV1",
    "EmpiricalStepPredictionV1",
    "KernelStepResultV1",
    "KernelCheckpointV1",
    "Run4SequentialRadioQueueKernelV1",
    "REGISTERED_KERNEL_PREREQUISITES_SHA256",
    "TIMEOUT_RESOLUTION_ELAPSED_NS",
]


SCHEMA_ID = "splitfusion_run4_sequential_radio_queue_kernel_v2"
SCHEMA_VERSION = 2
CORRECTED_ANALYSIS_GENERATION = 2
ACCEPTED_ANALYSIS_VERDICT = "ACCEPTED_FOR_RUN4_SEQUENTIAL_KERNEL"

# A reviewed change must set this to the exact canonical digest of the final
# corrected-v2 verifier prerequisites.  ``None`` is deliberate: callers cannot
# turn the current structural implementation into calibrated training evidence
# by supplying plausible-looking hashes.
REGISTERED_KERNEL_PREREQUISITES_SHA256: Optional[str] = None

# SUCCESS uses the inclusive interval ``elapsed <= REWARD_DEADLINE_NS``.  The
# first representable timeout instant is therefore exactly one nanosecond
# later.  A timeout prediction is required to close at this instant rather
# than waiting for an eventually delivered ACK.
TIMEOUT_RESOLUTION_ELAPSED_NS = contract.REWARD_DEADLINE_NS + 1


class _PredictionRequestAttestation:
    """Module-private proof that a request came from a validated decision."""

    __slots__ = ("binding", "nonce")

    def __init__(self, binding: str, nonce: object) -> None:
        self.binding = binding
        self.nonce = nonce


_PREDICTION_REQUEST_NONCE = object()


def _valid_prediction_request_attestation(
    attestation: object, binding: str
) -> bool:
    return (
        type(attestation) is _PredictionRequestAttestation
        and attestation.nonce is _PREDICTION_REQUEST_NONCE
        and attestation.binding == binding
    )


class SequentialKernelError(ValueError):
    """Base class for pure sequential-kernel contract violations."""


class EvidenceBindingError(SequentialKernelError):
    """Evidence identities or verifier claims are inconsistent."""


class CorrectedEvidenceUnavailable(EvidenceBindingError):
    """The reviewed corrected-v2 calibration has not been registered."""


class SupportViolation(SequentialKernelError):
    """A current state, action payload or prediction leaves fitted support."""


class SequenceViolation(SequentialKernelError):
    """A decision/prediction does not continue the exact causal sequence."""


class PredictionViolation(SequentialKernelError):
    """A caller-bound empirical prediction is malformed or contradictory."""


class CheckpointError(SequentialKernelError):
    """A checkpoint is malformed or belongs to another kernel binding."""


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise EvidenceBindingError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SequentialKernelError(f"{name} must be a non-empty str")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SequentialKernelError(
            f"{name} must be an exact int >= {minimum}, got {value!r}"
        )
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequentialKernelError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise SequentialKernelError(f"{name} must be finite")
    return result


def _strict_true(value: object, name: str) -> None:
    if type(value) is not bool or value is not True:
        raise EvidenceBindingError(f"{name} must be exactly True")


def _canonical_uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SequenceViolation(f"{name} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SequenceViolation(f"{name} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise SequenceViolation(f"{name} must be canonical lowercase UUID")
    return value


def _record(record_type: str, value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "record_type": record_type,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        **value,
    }


class KernelCalibrationPartition(str, Enum):
    """Which disjoint calibration cells may support a prediction."""

    FIT = "FIT"
    VALIDATION = "VALIDATION"


class KernelTerminalKind(str, Enum):
    """Terminal fact produced by the later fitted empirical model."""

    DELIVERED_FEEDBACK = "DELIVERED_FEEDBACK"
    REGISTERED_DELIVERY_FAILURE = "REGISTERED_DELIVERY_FAILURE"
    REGISTERED_SERVICE_FAILURE = "REGISTERED_SERVICE_FAILURE"
    TIMEOUT = "TIMEOUT"


class KernelAuthorizationClass(str, Enum):
    """Production authorization and test mechanics can never be conflated."""

    CORRECTED_V2_EMPIRICAL = "CORRECTED_V2_EMPIRICAL"
    TEST_ONLY_CALLER_BOUND = "TEST_ONLY_CALLER_BOUND"


@dataclass(frozen=True, slots=True)
class IntegerObservationV1:
    """An integer observation that preserves missingness separately from zero."""

    value: Optional[int]
    missing_reason: Optional[str]
    source_decision_seq: int
    provenance_sha256: str

    def __post_init__(self) -> None:
        _exact_int(self.source_decision_seq, "source_decision_seq")
        _digest(self.provenance_sha256, "provenance_sha256")
        if self.value is None:
            _text(self.missing_reason, "missing_reason")
        else:
            _exact_int(self.value, "value")
            if self.missing_reason is not None:
                raise SequentialKernelError(
                    "a present observation cannot carry missing_reason"
                )

    @property
    def present(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_reason": self.missing_reason,
            "provenance_sha256": self.provenance_sha256,
            "source_decision_seq": self.source_decision_seq,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RadioQueueStateV1:
    """Sequential state used by the hidden empirical radio/queue kernel."""

    session_uuid: str
    ue_id: str
    decision_seq: int
    prior_ul_mcs: IntegerObservationV1
    pre_enqueue_backlog_bytes: IntegerObservationV1

    def __post_init__(self) -> None:
        _canonical_uuid(self.session_uuid, "session_uuid")
        _text(self.ue_id, "ue_id")
        _exact_int(self.decision_seq, "decision_seq")
        if type(self.prior_ul_mcs) is not IntegerObservationV1:
            raise SequentialKernelError(
                "prior_ul_mcs must be exactly IntegerObservationV1"
            )
        if type(self.pre_enqueue_backlog_bytes) is not IntegerObservationV1:
            raise SequentialKernelError(
                "pre_enqueue_backlog_bytes must be exactly IntegerObservationV1"
            )
        for name, observation in (
            ("prior_ul_mcs", self.prior_ul_mcs),
            ("pre_enqueue_backlog_bytes", self.pre_enqueue_backlog_bytes),
        ):
            if observation.source_decision_seq > self.decision_seq:
                raise SequenceViolation(f"{name} comes from a future decision")
        if self.prior_ul_mcs.value is not None and not (
            contract.UL_MCS_INDEX_MIN
            <= self.prior_ul_mcs.value
            <= contract.UL_MCS_INDEX_MAX
        ):
            raise SupportViolation("prior UL MCS escaped the registered wire range")

    @property
    def actor_ready(self) -> bool:
        return (
            self.prior_ul_mcs.present
            and self.pre_enqueue_backlog_bytes.present
        )

    def require_actor_ready(self) -> None:
        if not self.actor_ready:
            raise SupportViolation(
                "missing MCS/backlog requires external fallback; missing values "
                "must never be replaced by zero"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_seq": self.decision_seq,
            "pre_enqueue_backlog_bytes": self.pre_enqueue_backlog_bytes.to_dict(),
            "prior_ul_mcs": self.prior_ul_mcs.to_dict(),
            "session_uuid": self.session_uuid,
            "ue_id": self.ue_id,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("radio_queue_state_v1", self.to_dict()))


@dataclass(frozen=True, slots=True)
class NumericSupportV1:
    """Closed fitted-support interval; this is evidence, not clipping policy."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        lower = _finite(self.minimum, "minimum")
        upper = _finite(self.maximum, "maximum")
        if lower > upper:
            raise SupportViolation("support minimum cannot exceed maximum")
        object.__setattr__(self, "minimum", lower)
        object.__setattr__(self, "maximum", upper)

    def require(self, value: object, name: str) -> float:
        checked = _finite(value, name)
        if not self.minimum <= checked <= self.maximum:
            raise SupportViolation(
                f"{name}={checked} is outside fitted support "
                f"[{self.minimum}, {self.maximum}]"
            )
        return checked

    def to_dict(self) -> Dict[str, float]:
        return {"maximum": self.maximum, "minimum": self.minimum}


@dataclass(frozen=True, slots=True)
class FitValidationSplitV1:
    """Immutable cell-level calibration split, invisible to the actor."""

    fit_cell_ids: Tuple[str, ...]
    validation_cell_ids: Tuple[str, ...]
    assignment_evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.fit_cell_ids) is not tuple or not self.fit_cell_ids:
            raise EvidenceBindingError("fit_cell_ids must be a nonempty tuple")
        if type(self.validation_cell_ids) is not tuple or not self.validation_cell_ids:
            raise EvidenceBindingError(
                "validation_cell_ids must be a nonempty tuple"
            )
        for name, values in (
            ("fit_cell_ids", self.fit_cell_ids),
            ("validation_cell_ids", self.validation_cell_ids),
        ):
            if any(not isinstance(value, str) or value == "" for value in values):
                raise EvidenceBindingError(f"{name} contains an invalid cell ID")
            if len(set(values)) != len(values):
                raise EvidenceBindingError(f"{name} contains duplicate cells")
        if set(self.fit_cell_ids) & set(self.validation_cell_ids):
            raise EvidenceBindingError("fit and validation calibration cells overlap")
        _digest(self.assignment_evidence_sha256, "assignment_evidence_sha256")

    def cells(self, partition: KernelCalibrationPartition) -> Tuple[str, ...]:
        if partition is KernelCalibrationPartition.FIT:
            return self.fit_cell_ids
        if partition is KernelCalibrationPartition.VALIDATION:
            return self.validation_cell_ids
        raise EvidenceBindingError("partition must be KernelCalibrationPartition")

    def require_cell(
        self, partition: KernelCalibrationPartition, cell_id: str
    ) -> None:
        if cell_id not in self.cells(partition):
            raise SupportViolation(
                f"cell {cell_id!r} is not in the {partition.value} calibration split"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_evidence_sha256": self.assignment_evidence_sha256,
            "fit_cell_ids": list(self.fit_cell_ids),
            "validation_cell_ids": list(self.validation_cell_ids),
        }


@dataclass(frozen=True, slots=True)
class LatencySupportV1:
    """Supports for six contiguous action-open-to-feedback intervals.

    The boundaries, in order, are action open, first feature-datagram send,
    complete edge reassembly, model dispatch, model-ready, compact-feedback
    socket send and UE feedback receipt. Sensor preparation precedes these
    intervals and map service follows them.
    """

    ue_action_path_ns: NumericSupportV1
    feature_uplink_ns: NumericSupportV1
    edge_pre_model_ns: NumericSupportV1
    model_tail_ns: NumericSupportV1
    post_model_feedback_preparation_ns: NumericSupportV1
    feedback_downlink_ns: NumericSupportV1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not NumericSupportV1:
                raise SupportViolation(f"{name} must be NumericSupportV1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name).to_dict()
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class KernelSupportV1:
    """The complete support on which a fitted kernel may be queried."""

    support_id: str
    support_version: int
    evidence_sha256: str
    per_tensor_payload_bytes: NumericSupportV1
    pre_enqueue_backlog_bytes: NumericSupportV1
    observed_prior_ul_mcs: Tuple[int, ...]
    latency: LatencySupportV1
    maximum_hold_tensors: int
    calibration_split: FitValidationSplitV1

    def __post_init__(self) -> None:
        _text(self.support_id, "support_id")
        _exact_int(self.support_version, "support_version", minimum=1)
        _digest(self.evidence_sha256, "evidence_sha256")
        if type(self.per_tensor_payload_bytes) is not NumericSupportV1:
            raise SupportViolation("per_tensor_payload_bytes must be NumericSupportV1")
        if type(self.pre_enqueue_backlog_bytes) is not NumericSupportV1:
            raise SupportViolation(
                "pre_enqueue_backlog_bytes must be NumericSupportV1"
            )
        if type(self.observed_prior_ul_mcs) is not tuple or not (
            self.observed_prior_ul_mcs
        ):
            raise SupportViolation("observed_prior_ul_mcs must be nonempty")
        if tuple(sorted(set(self.observed_prior_ul_mcs))) != (
            self.observed_prior_ul_mcs
        ):
            raise SupportViolation(
                "observed_prior_ul_mcs must be unique and ascending"
            )
        if any(
            type(value) is not int
            or not contract.UL_MCS_INDEX_MIN
            <= value
            <= contract.UL_MCS_INDEX_MAX
            for value in self.observed_prior_ul_mcs
        ):
            raise SupportViolation("observed_prior_ul_mcs contains an invalid index")
        if type(self.latency) is not LatencySupportV1:
            raise SupportViolation("latency must be LatencySupportV1")
        if (
            _exact_int(
                self.maximum_hold_tensors,
                "maximum_hold_tensors",
                minimum=contract.MINIMUM_HOLD_TENSORS,
            )
            != self.maximum_hold_tensors
        ):
            raise SupportViolation("invalid maximum_hold_tensors")
        if type(self.calibration_split) is not FitValidationSplitV1:
            raise SupportViolation("calibration_split must be FitValidationSplitV1")

    def require_state(self, state: RadioQueueStateV1) -> None:
        if type(state) is not RadioQueueStateV1:
            raise SupportViolation("state must be exactly RadioQueueStateV1")
        state.require_actor_ready()
        mcs = state.prior_ul_mcs.value
        backlog = state.pre_enqueue_backlog_bytes.value
        if mcs not in self.observed_prior_ul_mcs:
            raise SupportViolation(f"prior UL MCS {mcs} is outside fitted support")
        self.pre_enqueue_backlog_bytes.require(backlog, "pre_enqueue_backlog_bytes")

    def require_next_state_if_present(self, state: RadioQueueStateV1) -> None:
        if type(state) is not RadioQueueStateV1:
            raise SupportViolation("next state must be exactly RadioQueueStateV1")
        if state.prior_ul_mcs.value is not None and (
            state.prior_ul_mcs.value not in self.observed_prior_ul_mcs
        ):
            raise SupportViolation("predicted next MCS is outside fitted support")
        if state.pre_enqueue_backlog_bytes.value is not None:
            self.pre_enqueue_backlog_bytes.require(
                state.pre_enqueue_backlog_bytes.value,
                "predicted_next_backlog_bytes",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_split": self.calibration_split.to_dict(),
            "evidence_sha256": self.evidence_sha256,
            "latency": self.latency.to_dict(),
            "maximum_hold_tensors": self.maximum_hold_tensors,
            "observed_prior_ul_mcs": list(self.observed_prior_ul_mcs),
            "per_tensor_payload_bytes": self.per_tensor_payload_bytes.to_dict(),
            "pre_enqueue_backlog_bytes": self.pre_enqueue_backlog_bytes.to_dict(),
            "support_id": self.support_id,
            "support_version": self.support_version,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("kernel_support_v1", self.to_dict()))


@dataclass(frozen=True, slots=True)
class KernelProvenanceBindingV1:
    """Transitive provenance for the later fitted sequential kernel."""

    binding_id: str
    binding_version: int
    raw_campaign_manifest_sha256: str
    raw_decisions_sha256: str
    corrected_analysis_generation: int
    corrected_analysis_v2_sha256: str
    corrected_decisions_v2_sha256: str
    corrected_analysis_verdict: str
    feature_uplink_latency_evidence_sha256: str
    queue_transition_fit_sha256: str
    ue_action_path_latency_evidence_sha256: str
    edge_pre_model_latency_evidence_sha256: str
    model_tail_latency_evidence_sha256: str
    post_model_feedback_preparation_latency_evidence_sha256: str
    feedback_downlink_latency_evidence_sha256: str
    quality_feedback_report_sha256: str
    quality_feedback_manifest_sha256: str
    quality_adapter_binding_sha256: str
    held_provider_binding_sha256: str
    actor_feature_schema_sha256: str
    empirical_scaling_sha256: str
    freshness_policy_sha256: str
    support_sha256: str

    def __post_init__(self) -> None:
        _text(self.binding_id, "binding_id")
        _exact_int(self.binding_version, "binding_version", minimum=1)
        for name in (
            "raw_campaign_manifest_sha256",
            "raw_decisions_sha256",
            "corrected_analysis_v2_sha256",
            "corrected_decisions_v2_sha256",
            "feature_uplink_latency_evidence_sha256",
            "queue_transition_fit_sha256",
            "ue_action_path_latency_evidence_sha256",
            "edge_pre_model_latency_evidence_sha256",
            "model_tail_latency_evidence_sha256",
            "post_model_feedback_preparation_latency_evidence_sha256",
            "feedback_downlink_latency_evidence_sha256",
            "quality_feedback_report_sha256",
            "quality_feedback_manifest_sha256",
            "quality_adapter_binding_sha256",
            "held_provider_binding_sha256",
            "actor_feature_schema_sha256",
            "empirical_scaling_sha256",
            "freshness_policy_sha256",
            "support_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.corrected_analysis_generation != CORRECTED_ANALYSIS_GENERATION:
            raise EvidenceBindingError(
                "only corrected generation-2 analysis is eligible; "
                "analysis_v1/decisions.csv is explicitly rejected"
            )
        if self.corrected_analysis_verdict != ACCEPTED_ANALYSIS_VERDICT:
            raise EvidenceBindingError(
                "corrected analysis was not accepted for the sequential kernel"
            )
        if self.actor_feature_schema_sha256 != contract.FEATURE_SCHEMA_SHA256:
            raise EvidenceBindingError("actor feature schema binding drifted")

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("kernel_provenance_binding_v1", self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class KernelVerifierPrerequisitesV1:
    """All facts a later reviewer must attest before production issuance."""

    provenance: KernelProvenanceBindingV1
    support: KernelSupportV1
    fit_report_sha256: str
    validation_report_sha256: str
    fitted_model_sha256: str
    fit_validation_disjoint: bool
    sequential_queue_state_validated: bool
    per_tensor_payload_support_covered: bool
    full_feedback_frame_join_validated: bool
    missingness_preserved: bool
    actor_profile_label_absent: bool
    empirical_scaling_fit_only: bool
    freshness_policy_validated: bool
    prior_action_outcome_chain_validated: bool
    actor_state_group_variation_validated: bool
    fit_prediction_count: int
    validation_prediction_count: int

    def __post_init__(self) -> None:
        if type(self.provenance) is not KernelProvenanceBindingV1:
            raise EvidenceBindingError(
                "provenance must be KernelProvenanceBindingV1"
            )
        if type(self.support) is not KernelSupportV1:
            raise EvidenceBindingError("support must be KernelSupportV1")
        if self.provenance.support_sha256 != self.support.canonical_sha256:
            raise EvidenceBindingError("provenance/support digest mismatch")
        for name in (
            "fit_report_sha256",
            "validation_report_sha256",
            "fitted_model_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in (
            "fit_validation_disjoint",
            "sequential_queue_state_validated",
            "per_tensor_payload_support_covered",
            "full_feedback_frame_join_validated",
            "missingness_preserved",
            "actor_profile_label_absent",
            "empirical_scaling_fit_only",
            "freshness_policy_validated",
            "prior_action_outcome_chain_validated",
            "actor_state_group_variation_validated",
        ):
            _strict_true(getattr(self, name), name)
        _exact_int(self.fit_prediction_count, "fit_prediction_count", minimum=1)
        _exact_int(
            self.validation_prediction_count,
            "validation_prediction_count",
            minimum=1,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_profile_label_absent": self.actor_profile_label_absent,
            "actor_state_group_variation_validated": (
                self.actor_state_group_variation_validated
            ),
            "empirical_scaling_fit_only": self.empirical_scaling_fit_only,
            "fit_prediction_count": self.fit_prediction_count,
            "fit_report_sha256": self.fit_report_sha256,
            "fit_validation_disjoint": self.fit_validation_disjoint,
            "fitted_model_sha256": self.fitted_model_sha256,
            "full_feedback_frame_join_validated": (
                self.full_feedback_frame_join_validated
            ),
            "freshness_policy_validated": self.freshness_policy_validated,
            "missingness_preserved": self.missingness_preserved,
            "per_tensor_payload_support_covered": (
                self.per_tensor_payload_support_covered
            ),
            "provenance": self.provenance.to_dict(),
            "prior_action_outcome_chain_validated": (
                self.prior_action_outcome_chain_validated
            ),
            "sequential_queue_state_validated": (
                self.sequential_queue_state_validated
            ),
            "support": self.support.to_dict(),
            "validation_prediction_count": self.validation_prediction_count,
            "validation_report_sha256": self.validation_report_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("kernel_verifier_prerequisites_v1", self.to_dict())
        )


def _make_authorization_gate() -> Tuple[Any, Any]:
    sentinel = object()

    def issue(binding: str, authorization_class: KernelAuthorizationClass) -> Any:
        return (sentinel, binding, authorization_class)

    def valid(
        token: Any,
        binding: str,
        authorization_class: KernelAuthorizationClass,
    ) -> bool:
        return (
            type(token) is tuple
            and len(token) == 3
            and token[0] is sentinel
            and token[1] == binding
            and token[2] is authorization_class
        )

    return issue, valid


_issue_authorization, _valid_authorization = _make_authorization_gate()


@dataclass(frozen=True, slots=True)
class KernelAuthorizationV1:
    prerequisites_sha256: str
    provenance_sha256: str
    support_sha256: str
    fitted_model_sha256: str
    authorization_class: KernelAuthorizationClass
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "prerequisites_sha256",
            "provenance_sha256",
            "support_sha256",
            "fitted_model_sha256",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.authorization_class, KernelAuthorizationClass):
            raise EvidenceBindingError(
                "authorization_class must be KernelAuthorizationClass"
            )
        if self._attestation is not None and not _valid_authorization(
            self._attestation,
            self.prerequisites_sha256,
            self.authorization_class,
        ):
            raise EvidenceBindingError("kernel authorization attestation is invalid")

    @property
    def verified(self) -> bool:
        return _valid_authorization(
            self._attestation,
            self.prerequisites_sha256,
            self.authorization_class,
        )

    @property
    def replay_eligible(self) -> bool:
        return (
            self.verified
            and self.authorization_class
            is KernelAuthorizationClass.CORRECTED_V2_EMPIRICAL
        )

    def require_verified(self) -> None:
        if not self.verified:
            raise EvidenceBindingError(
                "kernel authorization must be issued by the module verifier"
            )


def _authorization(
    prerequisites: KernelVerifierPrerequisitesV1,
    authorization_class: KernelAuthorizationClass,
) -> KernelAuthorizationV1:
    candidate = KernelAuthorizationV1(
        prerequisites_sha256=prerequisites.canonical_sha256,
        provenance_sha256=prerequisites.provenance.canonical_sha256,
        support_sha256=prerequisites.support.canonical_sha256,
        fitted_model_sha256=prerequisites.fitted_model_sha256,
        authorization_class=authorization_class,
    )
    return KernelAuthorizationV1(
        prerequisites_sha256=candidate.prerequisites_sha256,
        provenance_sha256=candidate.provenance_sha256,
        support_sha256=candidate.support_sha256,
        fitted_model_sha256=candidate.fitted_model_sha256,
        authorization_class=candidate.authorization_class,
        _attestation=_issue_authorization(
            candidate.prerequisites_sha256, authorization_class
        ),
    )


def verify_kernel_prerequisites(
    prerequisites: KernelVerifierPrerequisitesV1,
) -> KernelAuthorizationV1:
    """Issue production authority only for the registered reviewed v2 digest."""

    if type(prerequisites) is not KernelVerifierPrerequisitesV1:
        raise EvidenceBindingError(
            "prerequisites must be exactly KernelVerifierPrerequisitesV1"
        )
    if REGISTERED_KERNEL_PREREQUISITES_SHA256 is None:
        raise CorrectedEvidenceUnavailable(
            "corrected-v2 calibration is not registered; production kernel "
            "authorization remains fail-closed"
        )
    if prerequisites.canonical_sha256 != REGISTERED_KERNEL_PREREQUISITES_SHA256:
        raise EvidenceBindingError(
            "prerequisite digest differs from the reviewed registered binding"
        )
    return _authorization(
        prerequisites, KernelAuthorizationClass.CORRECTED_V2_EMPIRICAL
    )


def _issue_test_only_authorization(
    prerequisites: KernelVerifierPrerequisitesV1,
) -> KernelAuthorizationV1:
    """Private structural-test issuer; its output is never replay eligible."""

    if type(prerequisites) is not KernelVerifierPrerequisitesV1:
        raise EvidenceBindingError("invalid test prerequisites")
    return _authorization(
        prerequisites, KernelAuthorizationClass.TEST_ONLY_CALLER_BOUND
    )


@dataclass(frozen=True, slots=True)
class KernelDecisionInputV1:
    """Exact action plus reward and held payloads for one decision cycle."""

    identity: contract.DecisionIdentityV1
    current_radio_state_sha256: str
    action: ExecutedActionIdentity
    reward_tensor: quality_adapter.RewardTensorResultV1
    held_tensors: Tuple[quality_adapter.HeldTensorResultV1, ...]
    action_open_timestamp_ns: int
    clock_domain: str
    calibration_partition: KernelCalibrationPartition

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.DecisionIdentityV1:
            raise SequenceViolation("identity must be exactly DecisionIdentityV1")
        _digest(self.current_radio_state_sha256, "current_radio_state_sha256")
        if type(self.action) is not ExecutedActionIdentity:
            raise SequenceViolation("action must be exactly ExecutedActionIdentity")
        self.action.require_reconciled()
        if type(self.reward_tensor) is not quality_adapter.RewardTensorResultV1:
            raise PredictionViolation(
                "reward_tensor must be exactly RewardTensorResultV1"
            )
        if type(self.held_tensors) is not tuple or not self.held_tensors:
            raise PredictionViolation("held_tensors must be a nonempty tuple")
        if any(
            type(item) is not quality_adapter.HeldTensorResultV1
            for item in self.held_tensors
        ):
            raise PredictionViolation(
                "held_tensors contains a foreign record type"
            )
        _exact_int(self.action_open_timestamp_ns, "action_open_timestamp_ns")
        _text(self.clock_domain, "clock_domain")
        if not isinstance(self.calibration_partition, KernelCalibrationPartition):
            raise PredictionViolation(
                "calibration_partition must be KernelCalibrationPartition"
            )
        if self.reward_tensor.action != self.action or any(
            item.action != self.action for item in self.held_tensors
        ):
            raise PredictionViolation(
                "the exact action must govern reward and every held tensor"
            )
        # ActionHoldV1 independently proves minimum hold and reward flags.
        contract.ActionHoldV1(
            identity=self.identity,
            action=self.action,
            tensors=(
                self.reward_tensor.tensor,
                *(item.tensor for item in self.held_tensors),
            ),
        )

    @property
    def hold(self) -> contract.ActionHoldV1:
        return contract.ActionHoldV1(
            identity=self.identity,
            action=self.action,
            tensors=(
                self.reward_tensor.tensor,
                *(item.tensor for item in self.held_tensors),
            ),
        )

    @property
    def prediction_request_sha256(self) -> str:
        """Digest of only the causal inputs exposed to the fitted model.

        This digest intentionally does not depend on ``q_perc``, the policy
        scene, CARLA sample/frame identities, or either quality/held selection
        record.  Payload values and their evidence *classes* are sufficient
        for the radio/queue prediction; provenance-bearing scene identities
        stay on the decision side of the boundary.
        """

        reward_payload = PredictionPayloadV1.from_tensor(
            self.reward_tensor.tensor
        )
        held_payloads = tuple(
            PredictionPayloadV1.from_tensor(item.tensor)
            for item in self.held_tensors
        )
        return canonical_sha256(
            _record(
                "prediction_request_v1",
                PredictionRequestV1._payload_from_parts(
                    identity=self.identity,
                    current_radio_state_sha256=(
                        self.current_radio_state_sha256
                    ),
                    action=self.action,
                    reward_payload=reward_payload,
                    held_payloads=held_payloads,
                    calibration_partition=self.calibration_partition,
                ),
            )
        )

    def to_prediction_request(
        self, current_radio_state: RadioQueueStateV1
    ) -> "PredictionRequestV1":
        """Issue the only request record a prediction provider may receive."""

        return PredictionRequestV1.from_decision(
            decision=self, current_radio_state=current_radio_state
        )

    def to_dict(self) -> Dict[str, Any]:
        hold = self.hold
        return {
            "action_open_timestamp_ns": self.action_open_timestamp_ns,
            "action_sha256": self.action.canonical_sha256(),
            "calibration_partition": self.calibration_partition.value,
            "clock_domain": self.clock_domain,
            "current_radio_state_sha256": self.current_radio_state_sha256,
            "held_estimate_sha256": [
                item.estimate.canonical_sha256 for item in self.held_tensors
            ],
            "hold_sha256": hold.canonical_sha256(),
            "identity": self.identity.to_canonical_dict(),
            "q_perc": self.reward_tensor.q_perc,
            "reward_tensor_evidence_sha256": (
                self.reward_tensor.evidence.canonical_sha256
            ),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("kernel_decision_input_v1", self.to_dict()))


@dataclass(frozen=True, slots=True)
class PredictionPayloadV1:
    """One ordered payload descriptor with all scene identity removed."""

    offered_payload_bytes: float
    payload_evidence_class: contract.PayloadEvidenceClass

    def __post_init__(self) -> None:
        payload = _finite(self.offered_payload_bytes, "offered_payload_bytes")
        if payload < 0.0:
            raise PredictionViolation("offered_payload_bytes must be >= 0")
        if not isinstance(
            self.payload_evidence_class, contract.PayloadEvidenceClass
        ):
            raise PredictionViolation(
                "payload_evidence_class must be PayloadEvidenceClass"
            )
        if (
            self.payload_evidence_class
            is contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            and type(self.offered_payload_bytes) is not int
        ):
            raise PredictionViolation(
                "measured prediction payload must remain an exact int"
            )

    @classmethod
    def from_tensor(cls, tensor: contract.HoldTensorV1) -> "PredictionPayloadV1":
        if type(tensor) is not contract.HoldTensorV1:
            raise PredictionViolation("prediction payload source must be HoldTensorV1")
        return cls(
            offered_payload_bytes=tensor.offered_payload_bytes,
            payload_evidence_class=tensor.payload_evidence_class,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "offered_payload_bytes": self.offered_payload_bytes,
            "payload_evidence_class": self.payload_evidence_class.value,
        }


@dataclass(frozen=True, slots=True)
class PredictionModelInputV1:
    """Feature-only input exposed to the empirical transition model.

    Session/UE/decision identities, source timestamps and calibration split
    labels are deliberately absent.  They remain in the attested request
    envelope for joining, but the fitted model can only consume the current
    measured values, executed action and ordered offered payloads.
    """

    prior_ul_mcs: int
    pre_enqueue_backlog_bytes: int
    action: ExecutedActionIdentity
    reward_payload: PredictionPayloadV1
    held_payloads: Tuple[PredictionPayloadV1, ...]

    def __post_init__(self) -> None:
        mcs = _exact_int(self.prior_ul_mcs, "prior_ul_mcs")
        if not contract.UL_MCS_INDEX_MIN <= mcs <= contract.UL_MCS_INDEX_MAX:
            raise SupportViolation("prior UL MCS escaped the registered wire range")
        _exact_int(
            self.pre_enqueue_backlog_bytes,
            "pre_enqueue_backlog_bytes",
        )
        if type(self.action) is not ExecutedActionIdentity:
            raise PredictionViolation(
                "model input action must be ExecutedActionIdentity"
            )
        self.action.require_reconciled()
        if type(self.reward_payload) is not PredictionPayloadV1:
            raise PredictionViolation(
                "model input reward_payload must be PredictionPayloadV1"
            )
        if type(self.held_payloads) is not tuple or not self.held_payloads:
            raise PredictionViolation("model input held_payloads must be nonempty")
        if any(type(item) is not PredictionPayloadV1 for item in self.held_payloads):
            raise PredictionViolation("model input contains a foreign held payload")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_sha256": self.action.canonical_sha256(),
            "held_payloads": [item.to_dict() for item in self.held_payloads],
            "pre_enqueue_backlog_bytes": self.pre_enqueue_backlog_bytes,
            "prior_ul_mcs": self.prior_ul_mcs,
            "reward_payload": self.reward_payload.to_dict(),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("prediction_model_input_v1", self.to_dict()))

    @property
    def held_offered_payload_bytes(self) -> float:
        return sum(item.offered_payload_bytes for item in self.held_payloads)


@dataclass(frozen=True, slots=True)
class PredictionRequestV1:
    """Causal, sanitized input to one empirical radio/queue prediction.

    The provider receives the exact prior UE MCS and pre-enqueue RLC backlog,
    including their observation provenance, plus the executed action and the
    ordered reward/held payload sizes.  It does *not* receive perception
    quality, the hidden channel-profile label, policy-scene values, CARLA
    sample/frame identities, or any future terminal/outcome.

    Construction is attested and is only issued by
    :meth:`KernelDecisionInputV1.to_prediction_request` after the supplied
    state is proven to be the exact state whose digest the decision carries.
    """

    identity: contract.DecisionIdentityV1
    current_radio_state: RadioQueueStateV1
    action: ExecutedActionIdentity
    reward_payload: PredictionPayloadV1
    held_payloads: Tuple[PredictionPayloadV1, ...]
    calibration_partition: KernelCalibrationPartition
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.DecisionIdentityV1:
            raise PredictionViolation("request identity must be DecisionIdentityV1")
        if type(self.current_radio_state) is not RadioQueueStateV1:
            raise PredictionViolation(
                "request current_radio_state must be RadioQueueStateV1"
            )
        if (
            self.current_radio_state.session_uuid,
            self.current_radio_state.ue_id,
            self.current_radio_state.decision_seq,
        ) != (
            self.identity.session_uuid,
            self.identity.ue_id,
            self.identity.decision_seq,
        ):
            raise SequenceViolation("request state belongs to another decision")
        if type(self.action) is not ExecutedActionIdentity:
            raise PredictionViolation("request action must be ExecutedActionIdentity")
        self.action.require_reconciled()
        if type(self.reward_payload) is not PredictionPayloadV1:
            raise PredictionViolation(
                "request reward_payload must be PredictionPayloadV1"
            )
        if type(self.held_payloads) is not tuple or not self.held_payloads:
            raise PredictionViolation("request held_payloads must be nonempty")
        if any(type(item) is not PredictionPayloadV1 for item in self.held_payloads):
            raise PredictionViolation("request contains a foreign held payload")
        if not isinstance(self.calibration_partition, KernelCalibrationPartition):
            raise PredictionViolation("request calibration partition is invalid")
        if self._attestation is not None and not (
            _valid_prediction_request_attestation(
                self._attestation, self._binding()
            )
        ):
            raise PredictionViolation(
                "prediction-request attestation does not bind to this request"
            )

    @staticmethod
    def _payload_from_parts(
        *,
        identity: contract.DecisionIdentityV1,
        current_radio_state_sha256: str,
        action: ExecutedActionIdentity,
        reward_payload: PredictionPayloadV1,
        held_payloads: Tuple[PredictionPayloadV1, ...],
        calibration_partition: KernelCalibrationPartition,
    ) -> Dict[str, Any]:
        return {
            "action_sha256": action.canonical_sha256(),
            "calibration_partition": calibration_partition.value,
            "current_radio_state_sha256": current_radio_state_sha256,
            "held_payloads": [item.to_dict() for item in held_payloads],
            "identity": identity.to_canonical_dict(),
            "reward_payload": reward_payload.to_dict(),
        }

    def _payload(self) -> Dict[str, Any]:
        return self._payload_from_parts(
            identity=self.identity,
            current_radio_state_sha256=self.current_radio_state.canonical_sha256,
            action=self.action,
            reward_payload=self.reward_payload,
            held_payloads=self.held_payloads,
            calibration_partition=self.calibration_partition,
        )

    def _binding(self) -> str:
        return canonical_sha256(_record("prediction_request_v1", self._payload()))

    @classmethod
    def from_decision(
        cls,
        *,
        decision: KernelDecisionInputV1,
        current_radio_state: RadioQueueStateV1,
    ) -> "PredictionRequestV1":
        if type(decision) is not KernelDecisionInputV1:
            raise PredictionViolation("request source must be KernelDecisionInputV1")
        if type(current_radio_state) is not RadioQueueStateV1:
            raise PredictionViolation("request state must be RadioQueueStateV1")
        if (
            current_radio_state.canonical_sha256
            != decision.current_radio_state_sha256
        ):
            raise SequenceViolation(
                "prediction request was given a different current radio/queue state"
            )
        candidate = cls(
            identity=decision.identity,
            current_radio_state=current_radio_state,
            action=decision.action,
            reward_payload=PredictionPayloadV1.from_tensor(
                decision.reward_tensor.tensor
            ),
            held_payloads=tuple(
                PredictionPayloadV1.from_tensor(item.tensor)
                for item in decision.held_tensors
            ),
            calibration_partition=decision.calibration_partition,
        )
        object.__setattr__(
            candidate,
            "_attestation",
            _PredictionRequestAttestation(
                candidate._binding(), _PREDICTION_REQUEST_NONCE
            ),
        )
        return candidate

    def require_attested(self) -> None:
        if not _valid_prediction_request_attestation(
            self._attestation, self._binding()
        ):
            raise PredictionViolation(
                "prediction request must be derived from its exact decision"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.require_attested()
        return self._payload()

    @property
    def canonical_sha256(self) -> str:
        self.require_attested()
        return self._binding()

    @property
    def model_input(self) -> PredictionModelInputV1:
        """Return the only record an empirical fitted model may consume."""

        self.require_attested()
        self.current_radio_state.require_actor_ready()
        assert self.current_radio_state.prior_ul_mcs.value is not None
        assert self.current_radio_state.pre_enqueue_backlog_bytes.value is not None
        return PredictionModelInputV1(
            prior_ul_mcs=self.current_radio_state.prior_ul_mcs.value,
            pre_enqueue_backlog_bytes=(
                self.current_radio_state.pre_enqueue_backlog_bytes.value
            ),
            action=self.action,
            reward_payload=self.reward_payload,
            held_payloads=self.held_payloads,
        )

    def bind_forecast(
        self, forecast: "EmpiricalModelForecastV1"
    ) -> "EmpiricalStepPredictionV1":
        """Join identity-free model output to this attested request envelope."""

        self.require_attested()
        if type(forecast) is not EmpiricalModelForecastV1:
            raise PredictionViolation("forecast must be EmpiricalModelForecastV1")
        if forecast.model_input_sha256 != self.model_input.canonical_sha256:
            raise PredictionViolation("forecast belongs to another model input")
        sequence = self.identity.decision_seq
        return EmpiricalStepPredictionV1(
            prediction_request_sha256=self.canonical_sha256,
            kernel_provenance_sha256=forecast.kernel_provenance_sha256,
            fitted_model_sha256=forecast.fitted_model_sha256,
            calibration_partition=self.calibration_partition,
            source_cell_id=forecast.source_cell_id,
            terminal_kind=forecast.terminal_kind,
            terminal_elapsed_ns=forecast.terminal_elapsed_ns,
            latency=forecast.latency,
            next_state=RadioQueueStateV1(
                session_uuid=self.identity.session_uuid,
                ue_id=self.identity.ue_id,
                decision_seq=sequence + 1,
                prior_ul_mcs=forecast.next_prior_ul_mcs.to_observation(sequence),
                pre_enqueue_backlog_bytes=(
                    forecast.next_pre_enqueue_backlog_bytes.to_observation(sequence)
                ),
            ),
            source_row_sha256=forecast.source_row_sha256,
        )

    @property
    def held_offered_payload_bytes(self) -> float:
        return self.model_input.held_offered_payload_bytes


@dataclass(frozen=True, slots=True)
class FeedbackLatencyBreakdownV1:
    """One exact six-stage action-open-to-feedback decomposition.

    ``edge_pre_model_ns`` spans complete edge reassembly to model dispatch. It
    includes any latest-only scheduler wait plus decompression, unpacking,
    dequantization, AE decode and input reconstruction. The distinct
    ``model_tail_ns`` interval ends when model output is ready.

    ``post_model_feedback_preparation_ns`` then spans model-ready to the actual
    compact-feedback socket send call. It includes required post-processing,
    p025 filtering, serialization, evaluator/ground-truth wait and scoring,
    feedback encoding, and local send preparation. ``feedback_downlink_ns``
    covers only that send call to UE receipt.
    """

    ue_action_path_ns: int
    feature_uplink_ns: int
    edge_pre_model_ns: int
    model_tail_ns: int
    post_model_feedback_preparation_ns: int
    feedback_downlink_ns: int
    ue_action_path_evidence_sha256: str
    feature_uplink_evidence_sha256: str
    edge_pre_model_evidence_sha256: str
    model_tail_evidence_sha256: str
    post_model_feedback_preparation_evidence_sha256: str
    feedback_downlink_evidence_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "ue_action_path_ns",
            "feature_uplink_ns",
            "edge_pre_model_ns",
            "model_tail_ns",
            "post_model_feedback_preparation_ns",
            "feedback_downlink_ns",
        ):
            _exact_int(getattr(self, name), name)
        for name in (
            "ue_action_path_evidence_sha256",
            "feature_uplink_evidence_sha256",
            "edge_pre_model_evidence_sha256",
            "model_tail_evidence_sha256",
            "post_model_feedback_preparation_evidence_sha256",
            "feedback_downlink_evidence_sha256",
        ):
            _digest(getattr(self, name), name)

    @property
    def transport_ns(self) -> int:
        return self.feature_uplink_ns + self.feedback_downlink_ns

    @property
    def non_network_ns(self) -> int:
        return (
            self.ue_action_path_ns
            + self.edge_pre_model_ns
            + self.model_tail_ns
            + self.post_model_feedback_preparation_ns
        )

    @property
    def full_feedback_ns(self) -> int:
        return self.transport_ns + self.non_network_ns

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        } | {
            "full_feedback_ns": self.full_feedback_ns,
            "non_network_ns": self.non_network_ns,
            "transport_ns": self.transport_ns,
        }


@dataclass(frozen=True, slots=True)
class PredictedIntegerObservationV1:
    """Identity-free next integer value returned by the fitted model."""

    value: Optional[int]
    missing_reason: Optional[str]
    provenance_sha256: str

    def __post_init__(self) -> None:
        _digest(self.provenance_sha256, "provenance_sha256")
        if self.value is None:
            _text(self.missing_reason, "missing_reason")
        else:
            _exact_int(self.value, "value")
            if self.missing_reason is not None:
                raise PredictionViolation(
                    "a present predicted observation cannot carry missing_reason"
                )

    def to_observation(self, source_decision_seq: int) -> IntegerObservationV1:
        _exact_int(source_decision_seq, "source_decision_seq")
        return IntegerObservationV1(
            value=self.value,
            missing_reason=self.missing_reason,
            source_decision_seq=source_decision_seq,
            provenance_sha256=self.provenance_sha256,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_reason": self.missing_reason,
            "provenance_sha256": self.provenance_sha256,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalModelForecastV1:
    """Identity-free empirical output, joined by an attested request."""

    model_input_sha256: str
    kernel_provenance_sha256: str
    fitted_model_sha256: str
    source_cell_id: str
    terminal_kind: KernelTerminalKind
    terminal_elapsed_ns: int
    latency: Optional[FeedbackLatencyBreakdownV1]
    next_prior_ul_mcs: PredictedIntegerObservationV1
    next_pre_enqueue_backlog_bytes: PredictedIntegerObservationV1
    source_row_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "model_input_sha256",
            "kernel_provenance_sha256",
            "fitted_model_sha256",
            "source_row_sha256",
        ):
            _digest(getattr(self, name), name)
        _text(self.source_cell_id, "source_cell_id")
        if not isinstance(self.terminal_kind, KernelTerminalKind):
            raise PredictionViolation("terminal_kind must be KernelTerminalKind")
        elapsed = _exact_int(
            self.terminal_elapsed_ns, "terminal_elapsed_ns", minimum=1
        )
        if type(self.next_prior_ul_mcs) is not PredictedIntegerObservationV1:
            raise PredictionViolation("next_prior_ul_mcs has a foreign type")
        if type(self.next_pre_enqueue_backlog_bytes) is not (
            PredictedIntegerObservationV1
        ):
            raise PredictionViolation(
                "next_pre_enqueue_backlog_bytes has a foreign type"
            )
        if self.terminal_kind is KernelTerminalKind.DELIVERED_FEEDBACK:
            if type(self.latency) is not FeedbackLatencyBreakdownV1:
                raise PredictionViolation(
                    "delivered feedback requires an exact latency breakdown"
                )
            if elapsed != self.latency.full_feedback_ns:
                raise PredictionViolation(
                    "terminal elapsed time must equal the frame-level latency sum"
                )
            if elapsed > contract.REWARD_DEADLINE_NS:
                raise PredictionViolation(
                    "DELIVERED_FEEDBACK after the inclusive 170-ms deadline "
                    "is late-orphan evidence, not a cycle terminal"
                )
        elif self.latency is not None:
            raise PredictionViolation(
                "failed/timeout forecasts must not fabricate a complete "
                "feedback latency breakdown"
            )
        if self.terminal_kind is KernelTerminalKind.TIMEOUT and (
            elapsed != TIMEOUT_RESOLUTION_ELAPSED_NS
        ):
            raise PredictionViolation(
                "TIMEOUT must close at the first nanosecond strictly after "
                "the inclusive 170-ms deadline"
            )
        if (
            self.terminal_kind is not KernelTerminalKind.TIMEOUT
            and elapsed > contract.REWARD_DEADLINE_NS
        ):
            raise PredictionViolation(
                "a non-timeout terminal cannot arrive after timeout closure"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted_model_sha256": self.fitted_model_sha256,
            "kernel_provenance_sha256": self.kernel_provenance_sha256,
            "latency": None if self.latency is None else self.latency.to_dict(),
            "model_input_sha256": self.model_input_sha256,
            "next_pre_enqueue_backlog_bytes": (
                self.next_pre_enqueue_backlog_bytes.to_dict()
            ),
            "next_prior_ul_mcs": self.next_prior_ul_mcs.to_dict(),
            "source_cell_id": self.source_cell_id,
            "source_row_sha256": self.source_row_sha256,
            "terminal_elapsed_ns": self.terminal_elapsed_ns,
            "terminal_kind": self.terminal_kind.value,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("empirical_model_forecast_v1", self.to_dict()))


@dataclass(frozen=True, slots=True)
class EmpiricalStepPredictionV1:
    """Caller-bound output of the future fitted model; never actor input."""

    prediction_request_sha256: str
    kernel_provenance_sha256: str
    fitted_model_sha256: str
    calibration_partition: KernelCalibrationPartition
    source_cell_id: str
    terminal_kind: KernelTerminalKind
    terminal_elapsed_ns: int
    latency: Optional[FeedbackLatencyBreakdownV1]
    next_state: RadioQueueStateV1
    source_row_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "prediction_request_sha256",
            "kernel_provenance_sha256",
            "fitted_model_sha256",
            "source_row_sha256",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.calibration_partition, KernelCalibrationPartition):
            raise PredictionViolation("invalid calibration_partition")
        _text(self.source_cell_id, "source_cell_id")
        if not isinstance(self.terminal_kind, KernelTerminalKind):
            raise PredictionViolation("terminal_kind must be KernelTerminalKind")
        elapsed = _exact_int(
            self.terminal_elapsed_ns, "terminal_elapsed_ns", minimum=1
        )
        if type(self.next_state) is not RadioQueueStateV1:
            raise PredictionViolation("next_state must be RadioQueueStateV1")
        if self.terminal_kind is KernelTerminalKind.DELIVERED_FEEDBACK:
            if type(self.latency) is not FeedbackLatencyBreakdownV1:
                raise PredictionViolation(
                    "delivered feedback requires an exact latency breakdown"
                )
            if elapsed != self.latency.full_feedback_ns:
                raise PredictionViolation(
                    "terminal elapsed time must equal the frame-level latency sum"
                )
            if elapsed > contract.REWARD_DEADLINE_NS:
                raise PredictionViolation(
                    "DELIVERED_FEEDBACK after the inclusive 170-ms deadline "
                    "is late-orphan evidence, not a cycle terminal"
                )
        elif self.latency is not None:
            raise PredictionViolation(
                "failed/timeout predictions must not fabricate a complete "
                "feedback latency breakdown"
            )
        if self.terminal_kind is KernelTerminalKind.TIMEOUT and (
            elapsed != TIMEOUT_RESOLUTION_ELAPSED_NS
        ):
            raise PredictionViolation(
                "TIMEOUT must close at the first nanosecond strictly after "
                "the inclusive 170-ms deadline"
            )
        if (
            self.terminal_kind is not KernelTerminalKind.TIMEOUT
            and elapsed > contract.REWARD_DEADLINE_NS
        ):
            raise PredictionViolation(
                "a non-timeout terminal cannot arrive after timeout closure"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_partition": self.calibration_partition.value,
            "prediction_request_sha256": self.prediction_request_sha256,
            "fitted_model_sha256": self.fitted_model_sha256,
            "kernel_provenance_sha256": self.kernel_provenance_sha256,
            "latency": None if self.latency is None else self.latency.to_dict(),
            "next_state": self.next_state.to_dict(),
            "source_cell_id": self.source_cell_id,
            "source_row_sha256": self.source_row_sha256,
            "terminal_elapsed_ns": self.terminal_elapsed_ns,
            "terminal_kind": self.terminal_kind.value,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("empirical_step_prediction_v1", self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class KernelStepResultV1:
    """Validated terminal cycle evidence returned by the pure reducer."""

    calibration_partition: KernelCalibrationPartition
    authorization_class: KernelAuthorizationClass
    hold: contract.ActionHoldV1
    reward_event: contract.RewardEventV1
    latency: Optional[FeedbackLatencyBreakdownV1]
    next_radio_state: RadioQueueStateV1
    cycle_end_timestamp_ns: int
    prediction_sha256: str
    provenance_sha256: str
    support_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_partition, KernelCalibrationPartition):
            raise PredictionViolation("invalid result calibration partition")
        if not isinstance(self.authorization_class, KernelAuthorizationClass):
            raise PredictionViolation("invalid result authorization class")
        if type(self.hold) is not contract.ActionHoldV1:
            raise PredictionViolation("result hold must be ActionHoldV1")
        if type(self.reward_event) is not contract.RewardEventV1:
            raise PredictionViolation("result reward_event must be RewardEventV1")
        if self.latency is not None and type(self.latency) is not (
            FeedbackLatencyBreakdownV1
        ):
            raise PredictionViolation("result latency has a foreign type")
        if type(self.next_radio_state) is not RadioQueueStateV1:
            raise PredictionViolation("result next_radio_state has a foreign type")
        _exact_int(self.cycle_end_timestamp_ns, "cycle_end_timestamp_ns")
        for name in (
            "prediction_sha256",
            "provenance_sha256",
            "support_sha256",
        ):
            _digest(getattr(self, name), name)

    @property
    def replay_export_allowed(self) -> bool:
        return (
            self.authorization_class
            is KernelAuthorizationClass.CORRECTED_V2_EMPIRICAL
        )

    def to_environment_result(
        self,
        *,
        episode_boundary: contract.EpisodeBoundary = contract.EpisodeBoundary.CONTINUES,
    ) -> Any:
        """Return the existing environment envelope without changing evidence."""

        from rl_agent.splitfusion_hybrid_sac_run4_v1.environment import (
            KernelCycleResultV1,
        )

        return KernelCycleResultV1(
            hold=self.hold,
            reward_event=self.reward_event,
            cycle_end_timestamp_ns=self.cycle_end_timestamp_ns,
            episode_boundary=episode_boundary,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorization_class": self.authorization_class.value,
            "calibration_partition": self.calibration_partition.value,
            "cycle_end_timestamp_ns": self.cycle_end_timestamp_ns,
            "hold_sha256": self.hold.canonical_sha256(),
            "latency": None if self.latency is None else self.latency.to_dict(),
            "next_radio_state": self.next_radio_state.to_dict(),
            "prediction_sha256": self.prediction_sha256,
            "provenance_sha256": self.provenance_sha256,
            "reward_event_sha256": self.reward_event.canonical_sha256(),
            "support_sha256": self.support_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("kernel_step_result_v1", self.to_dict()))


@dataclass(frozen=True, slots=True)
class KernelCheckpointV1:
    """Exact continuation state; no model/RNG state is silently omitted."""

    prerequisites_sha256: str
    provenance_sha256: str
    support_sha256: str
    fitted_model_sha256: str
    authorization_class: KernelAuthorizationClass
    calibration_partition: KernelCalibrationPartition
    current_state: RadioQueueStateV1
    completed_steps: int
    last_decision_input_sha256: Optional[str]
    last_step_result_sha256: Optional[str]

    def __post_init__(self) -> None:
        for name in (
            "prerequisites_sha256",
            "provenance_sha256",
            "support_sha256",
            "fitted_model_sha256",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.authorization_class, KernelAuthorizationClass):
            raise CheckpointError("authorization_class is invalid")
        if not isinstance(self.calibration_partition, KernelCalibrationPartition):
            raise CheckpointError("calibration_partition is invalid")
        if type(self.current_state) is not RadioQueueStateV1:
            raise CheckpointError("current_state must be RadioQueueStateV1")
        _exact_int(self.completed_steps, "completed_steps")
        if self.completed_steps == 0:
            if (
                self.last_decision_input_sha256 is not None
                or self.last_step_result_sha256 is not None
            ):
                raise CheckpointError("zero-step checkpoint cannot carry last hashes")
        else:
            _digest(
                self.last_decision_input_sha256,
                "last_decision_input_sha256",
            )
            _digest(self.last_step_result_sha256, "last_step_result_sha256")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorization_class": self.authorization_class.value,
            "calibration_partition": self.calibration_partition.value,
            "completed_steps": self.completed_steps,
            "current_state": self.current_state.to_dict(),
            "fitted_model_sha256": self.fitted_model_sha256,
            "last_decision_input_sha256": self.last_decision_input_sha256,
            "last_step_result_sha256": self.last_step_result_sha256,
            "prerequisites_sha256": self.prerequisites_sha256,
            "provenance_sha256": self.provenance_sha256,
            "support_sha256": self.support_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("kernel_checkpoint_v1", self.to_dict()))


class Run4SequentialRadioQueueKernelV1:
    """Transactional reducer around a future caller-bound empirical fitter."""

    def __init__(
        self,
        *,
        prerequisites: KernelVerifierPrerequisitesV1,
        authorization: KernelAuthorizationV1,
        calibration_partition: KernelCalibrationPartition,
        initial_state: RadioQueueStateV1,
    ) -> None:
        if type(prerequisites) is not KernelVerifierPrerequisitesV1:
            raise EvidenceBindingError("invalid verifier prerequisites")
        if type(authorization) is not KernelAuthorizationV1:
            raise EvidenceBindingError("invalid kernel authorization")
        authorization.require_verified()
        if authorization.prerequisites_sha256 != prerequisites.canonical_sha256:
            raise EvidenceBindingError("authorization/prerequisites mismatch")
        if authorization.provenance_sha256 != prerequisites.provenance.canonical_sha256:
            raise EvidenceBindingError("authorization/provenance mismatch")
        if authorization.support_sha256 != prerequisites.support.canonical_sha256:
            raise EvidenceBindingError("authorization/support mismatch")
        if authorization.fitted_model_sha256 != prerequisites.fitted_model_sha256:
            raise EvidenceBindingError("authorization/fitted-model mismatch")
        if not isinstance(calibration_partition, KernelCalibrationPartition):
            raise EvidenceBindingError("invalid calibration partition")
        if type(initial_state) is not RadioQueueStateV1:
            raise SequenceViolation("initial_state must be RadioQueueStateV1")
        prerequisites.support.require_state(initial_state)
        self._prerequisites = prerequisites
        self._authorization = authorization
        self._partition = calibration_partition
        self._state = initial_state
        self._completed_steps = 0
        self._last_decision_input_sha256: Optional[str] = None
        self._last_step_result_sha256: Optional[str] = None

    @property
    def current_state(self) -> RadioQueueStateV1:
        return self._state

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def replay_export_allowed(self) -> bool:
        return self._authorization.replay_eligible

    def _validate_decision(self, decision: KernelDecisionInputV1) -> None:
        if type(decision) is not KernelDecisionInputV1:
            raise SequenceViolation("decision must be KernelDecisionInputV1")
        expected = (
            self._state.session_uuid,
            self._state.ue_id,
            self._state.decision_seq,
        )
        observed = (
            decision.identity.session_uuid,
            decision.identity.ue_id,
            decision.identity.decision_seq,
        )
        if observed != expected:
            raise SequenceViolation(
                f"decision identity {observed!r} does not continue {expected!r}"
            )
        if decision.current_radio_state_sha256 != self._state.canonical_sha256:
            raise SequenceViolation(
                "decision was constructed from a different current radio/queue state"
            )
        if decision.calibration_partition is not self._partition:
            raise SequenceViolation("decision calibration partition drifted")
        support = self._prerequisites.support
        support.require_state(self._state)
        if decision.hold.duration > support.maximum_hold_tensors:
            raise SupportViolation("action hold exceeds calibrated duration support")
        for tensor in decision.hold.tensors:
            support.per_tensor_payload_bytes.require(
                tensor.offered_payload_bytes, "offered_payload_bytes"
            )
        binding = self._prerequisites.provenance
        if (
            decision.reward_tensor.evidence.adapter_binding_sha256
            != binding.quality_adapter_binding_sha256
        ):
            raise EvidenceBindingError("reward tensor quality-adapter binding drifted")
        if any(
            item.estimate.provider_binding_sha256
            != binding.held_provider_binding_sha256
            for item in decision.held_tensors
        ):
            raise EvidenceBindingError("held payload provider binding drifted")

    def _validate_prediction(
        self,
        decision: KernelDecisionInputV1,
        prediction: EmpiricalStepPredictionV1,
    ) -> None:
        if type(prediction) is not EmpiricalStepPredictionV1:
            raise PredictionViolation("prediction must be EmpiricalStepPredictionV1")
        if (
            prediction.prediction_request_sha256
            != decision.prediction_request_sha256
        ):
            raise PredictionViolation(
                "prediction belongs to another causal prediction request"
            )
        if prediction.kernel_provenance_sha256 != (
            self._prerequisites.provenance.canonical_sha256
        ):
            raise PredictionViolation("prediction provenance binding drifted")
        if prediction.fitted_model_sha256 != (
            self._prerequisites.fitted_model_sha256
        ):
            raise PredictionViolation("prediction fitted-model binding drifted")
        if prediction.calibration_partition is not self._partition:
            raise PredictionViolation("prediction calibration partition drifted")
        support = self._prerequisites.support
        support.calibration_split.require_cell(
            self._partition, prediction.source_cell_id
        )
        next_state = prediction.next_state
        if (
            next_state.session_uuid,
            next_state.ue_id,
            next_state.decision_seq,
        ) != (
            self._state.session_uuid,
            self._state.ue_id,
            self._state.decision_seq + 1,
        ):
            raise SequenceViolation("prediction did not produce the exact successor")
        support.require_next_state_if_present(next_state)
        latency = prediction.latency
        if latency is not None:
            for name in (
                "ue_action_path_ns",
                "feature_uplink_ns",
                "edge_pre_model_ns",
                "model_tail_ns",
                "post_model_feedback_preparation_ns",
                "feedback_downlink_ns",
            ):
                getattr(support.latency, name).require(
                    getattr(latency, name), name
                )
            binding = self._prerequisites.provenance
            if latency.ue_action_path_evidence_sha256 != (
                binding.ue_action_path_latency_evidence_sha256
            ):
                raise EvidenceBindingError("UE action-path evidence drifted")
            if latency.feature_uplink_evidence_sha256 != (
                binding.feature_uplink_latency_evidence_sha256
            ):
                raise EvidenceBindingError("feature-uplink evidence drifted")
            if latency.edge_pre_model_evidence_sha256 != (
                binding.edge_pre_model_latency_evidence_sha256
            ):
                raise EvidenceBindingError("edge-pre-model evidence drifted")
            if latency.model_tail_evidence_sha256 != (
                binding.model_tail_latency_evidence_sha256
            ):
                raise EvidenceBindingError("model-tail evidence drifted")
            if latency.post_model_feedback_preparation_evidence_sha256 != (
                binding.post_model_feedback_preparation_latency_evidence_sha256
            ):
                raise EvidenceBindingError(
                    "post-model-feedback-preparation evidence drifted"
                )
            if latency.feedback_downlink_evidence_sha256 != (
                binding.feedback_downlink_latency_evidence_sha256
            ):
                raise EvidenceBindingError("feedback-downlink evidence drifted")

    @staticmethod
    def _reward_kind(terminal: KernelTerminalKind) -> contract.RewardEventKind:
        return {
            KernelTerminalKind.DELIVERED_FEEDBACK: (
                contract.RewardEventKind.DELIVERED_SUCCESS
            ),
            KernelTerminalKind.REGISTERED_DELIVERY_FAILURE: (
                contract.RewardEventKind.REGISTERED_DELIVERY_FAILURE
            ),
            KernelTerminalKind.REGISTERED_SERVICE_FAILURE: (
                contract.RewardEventKind.REGISTERED_SERVICE_FAILURE
            ),
            KernelTerminalKind.TIMEOUT: contract.RewardEventKind.TIMEOUT,
        }[terminal]

    def advance(
        self,
        *,
        decision: KernelDecisionInputV1,
        prediction: EmpiricalStepPredictionV1,
    ) -> KernelStepResultV1:
        """Validate, resolve and atomically advance exactly one decision."""

        self._validate_decision(decision)
        self._validate_prediction(decision, prediction)
        resolution_ns = (
            decision.action_open_timestamp_ns + prediction.terminal_elapsed_ns
        )
        delivered = (
            prediction.terminal_kind is KernelTerminalKind.DELIVERED_FEEDBACK
        )
        event = contract.RewardEventV1(
            identity=decision.identity,
            action=decision.action,
            kind=self._reward_kind(prediction.terminal_kind),
            action_open_timestamp_ns=decision.action_open_timestamp_ns,
            resolution_timestamp_ns=resolution_ns,
            clock_domain=decision.clock_domain,
            source="RUN4_CORRECTED_V2_SEQUENTIAL_KERNEL",
            q_perc=(decision.reward_tensor.q_perc if delivered else None),
        )
        result = KernelStepResultV1(
            calibration_partition=self._partition,
            authorization_class=self._authorization.authorization_class,
            hold=decision.hold,
            reward_event=event,
            latency=prediction.latency,
            next_radio_state=prediction.next_state,
            cycle_end_timestamp_ns=resolution_ns,
            prediction_sha256=prediction.canonical_sha256,
            provenance_sha256=self._prerequisites.provenance.canonical_sha256,
            support_sha256=self._prerequisites.support.canonical_sha256,
        )
        # No mutation occurs before the result has passed every constructor and
        # support/provenance check above.
        self._state = prediction.next_state
        self._completed_steps += 1
        self._last_decision_input_sha256 = decision.canonical_sha256
        self._last_step_result_sha256 = result.canonical_sha256
        return result

    def checkpoint(self) -> KernelCheckpointV1:
        return KernelCheckpointV1(
            prerequisites_sha256=self._prerequisites.canonical_sha256,
            provenance_sha256=self._prerequisites.provenance.canonical_sha256,
            support_sha256=self._prerequisites.support.canonical_sha256,
            fitted_model_sha256=self._prerequisites.fitted_model_sha256,
            authorization_class=self._authorization.authorization_class,
            calibration_partition=self._partition,
            current_state=self._state,
            completed_steps=self._completed_steps,
            last_decision_input_sha256=self._last_decision_input_sha256,
            last_step_result_sha256=self._last_step_result_sha256,
        )

    @classmethod
    def restore(
        cls,
        *,
        checkpoint: KernelCheckpointV1,
        prerequisites: KernelVerifierPrerequisitesV1,
        authorization: KernelAuthorizationV1,
    ) -> "Run4SequentialRadioQueueKernelV1":
        if type(checkpoint) is not KernelCheckpointV1:
            raise CheckpointError("checkpoint must be KernelCheckpointV1")
        if type(prerequisites) is not KernelVerifierPrerequisitesV1:
            raise CheckpointError("prerequisites must be KernelVerifierPrerequisitesV1")
        if type(authorization) is not KernelAuthorizationV1:
            raise CheckpointError("authorization must be KernelAuthorizationV1")
        authorization.require_verified()
        expected = (
            prerequisites.canonical_sha256,
            prerequisites.provenance.canonical_sha256,
            prerequisites.support.canonical_sha256,
            prerequisites.fitted_model_sha256,
            authorization.authorization_class,
        )
        observed = (
            checkpoint.prerequisites_sha256,
            checkpoint.provenance_sha256,
            checkpoint.support_sha256,
            checkpoint.fitted_model_sha256,
            checkpoint.authorization_class,
        )
        if observed != expected:
            raise CheckpointError("checkpoint binding differs from this kernel")
        restored = cls(
            prerequisites=prerequisites,
            authorization=authorization,
            calibration_partition=checkpoint.calibration_partition,
            initial_state=checkpoint.current_state,
        )
        restored._completed_steps = checkpoint.completed_steps
        restored._last_decision_input_sha256 = (
            checkpoint.last_decision_input_sha256
        )
        restored._last_step_result_sha256 = checkpoint.last_step_result_sha256
        if restored.checkpoint().canonical_sha256 != checkpoint.canonical_sha256:
            raise CheckpointError("checkpoint did not restore exactly")
        return restored
