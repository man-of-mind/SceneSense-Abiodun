"""Offline-only evidence boundary for the Run-4 modeled composite.

This module does not make a modeled queue/latency sample empirical.  It gives
that sample a separate, explicit evidence class and keeps it outside the
existing ``CALIBRATED_EMPIRICAL`` environment and production replay gates.

The boundary is intentionally narrow:

* every source component is declared as a measured source, a fit-derived
  model, or a deterministic contract transform;
* per-cycle support is explicit, including the distinct
  ``PROFILE_TRANSFER_UNVALIDATED`` and ``MODE_TRANSFER_UNVALIDATED`` cases;
* the successful-feedback clock is the v3 authoritative integer-nanosecond
  action-open -> feedback-receipt total derived from ordered endpoints; and
* a modeled feedback arrival later than the inclusive 170-ms deadline is
  projected to the registered timeout closure.  Its actual late-arrival total
  is retained only as audit metadata and can never become successful latency.

The issued envelope has an explicit ``export_for_offline_training`` method.
It deliberately has no production replay export: ``export_for_replay`` always
raises, and the existing calibrated environment/replay boundaries reject the
envelope by exact type.  Importing this module performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import CATALOG_SHA256
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

__all__ = [
    "BindingError",
    "ComponentEvidenceDisclosureV1",
    "ComponentEvidenceNature",
    "ComponentRole",
    "EVIDENCE_CLASS",
    "FeedbackEndpointPairV1",
    "LatencyProjectionV1",
    "MODELED_COMPOSITE_SCHEMA_ID",
    "MODELED_COMPOSITE_SCHEMA_VERSION",
    "ModeTransferStatus",
    "ModeledCompositeBindingV1",
    "ModeledCompositeContractError",
    "ModeledCompositeEvidenceClass",
    "ModeledCompositeSupportUseV1",
    "ModeledCompositeOfflineTransitionV1",
    "ModeledCompositeTrainingEnvelopeV1",
    "ModeledCompositeTrainingIssuerV1",
    "ProductionEvidenceRejected",
    "ProfileTransferStatus",
    "SupportError",
]


MODELED_COMPOSITE_SCHEMA_ID = "splitfusion.run4.modeled_composite_training.v1"
MODELED_COMPOSITE_SCHEMA_VERSION = 1


class ModeledCompositeContractError(ValueError):
    """Base class for the offline modeled-composite evidence boundary."""


class BindingError(ModeledCompositeContractError):
    """A source, schema or evidence-class binding is inconsistent."""


class SupportError(ModeledCompositeContractError):
    """A modeled cycle lacks explicit fit support or transfer disclosure."""


class ProductionEvidenceRejected(ModeledCompositeContractError):
    """Modeled-composite evidence was offered to a production export seam."""


class ModeledCompositeEvidenceClass(str, Enum):
    MODELED_COMPOSITE_TRAINING = "MODELED_COMPOSITE_TRAINING"


EVIDENCE_CLASS = ModeledCompositeEvidenceClass.MODELED_COMPOSITE_TRAINING


class ComponentRole(str, Enum):
    SCENE_QUALITY_PAYLOAD = "SCENE_QUALITY_PAYLOAD"
    RADIO_QUEUE_DYNAMICS = "RADIO_QUEUE_DYNAMICS"
    UL_MCS_TRANSITION = "UL_MCS_TRANSITION"
    ACTOR_INFERENCE_QUANTIZATION_DISPATCH = (
        "ACTOR_INFERENCE_QUANTIZATION_DISPATCH"
    )
    ACTION_OPEN_TO_FEEDBACK_TOTAL = "ACTION_OPEN_TO_FEEDBACK_TOTAL"


class ComponentEvidenceNature(str, Enum):
    MEASURED_SOURCE = "MEASURED_SOURCE"
    FIT_DERIVED_MODEL = "FIT_DERIVED_MODEL"
    DETERMINISTIC_CONTRACT_TRANSFORM = "DETERMINISTIC_CONTRACT_TRANSFORM"


class ProfileTransferStatus(str, Enum):
    PROFILE_WITHIN_DIRECT_FIT_SUPPORT = "PROFILE_WITHIN_DIRECT_FIT_SUPPORT"
    PROFILE_TRANSFER_UNVALIDATED = "PROFILE_TRANSFER_UNVALIDATED"


class ModeTransferStatus(str, Enum):
    MODE_WITHIN_DIRECT_FIT_SUPPORT = "MODE_WITHIN_DIRECT_FIT_SUPPORT"
    MODE_TRANSFER_UNVALIDATED = "MODE_TRANSFER_UNVALIDATED"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise BindingError(f"{name} must be a non-empty str")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise BindingError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModeledCompositeContractError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def _strict_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise BindingError(f"{name} must be exactly {expected}")


def _record(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"record": kind, **payload}


@dataclass(frozen=True, slots=True)
class ComponentEvidenceDisclosureV1:
    """One source component; real measurements remain source evidence only."""

    role: ComponentRole
    nature: ComponentEvidenceNature
    source_evidence_sha256: str
    fit_support_sha256: str
    source_scope: str
    fit_partition_only: bool = True
    validation_rows_consumed: bool = False
    measured_runtime_output: bool = False

    def __post_init__(self) -> None:
        if type(self.role) is not ComponentRole:
            raise BindingError("role must be exactly ComponentRole")
        if type(self.nature) is not ComponentEvidenceNature:
            raise BindingError("nature must be exactly ComponentEvidenceNature")
        _digest(self.source_evidence_sha256, "source_evidence_sha256")
        _digest(self.fit_support_sha256, "fit_support_sha256")
        _text(self.source_scope, "source_scope")
        _strict_bool(self.fit_partition_only, "fit_partition_only", True)
        _strict_bool(self.validation_rows_consumed, "validation_rows_consumed", False)
        _strict_bool(self.measured_runtime_output, "measured_runtime_output", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_partition_only": self.fit_partition_only,
            "fit_support_sha256": self.fit_support_sha256,
            "measured_runtime_output": self.measured_runtime_output,
            "nature": self.nature.value,
            "role": self.role.value,
            "source_evidence_sha256": self.source_evidence_sha256,
            "source_scope": self.source_scope,
            "validation_rows_consumed": self.validation_rows_consumed,
        }


_REQUIRED_COMPONENT_ROLES = tuple(ComponentRole)


@dataclass(frozen=True, slots=True)
class ModeledCompositeBindingV1:
    """Transitive binding for an offline-only modeled training generator."""

    binding_id: str
    binding_version: int
    component_disclosures: Tuple[ComponentEvidenceDisclosureV1, ...]
    provider_implementation_sha256: str
    verifier_manifest_sha256: str
    feature_schema_sha256: str = contract.FEATURE_SCHEMA_SHA256
    reward_schema_sha256: str = contract.REWARD_SCHEMA_SHA256
    transition_schema_sha256: str = contract.TRANSITION_SCHEMA_SHA256
    catalog_sha256: str = CATALOG_SHA256
    latency_schema_id: str = sequential_kernel.SCHEMA_ID
    latency_schema_version: int = sequential_kernel.SCHEMA_VERSION
    evidence_class: ModeledCompositeEvidenceClass = EVIDENCE_CLASS
    offline_training_only: bool = True
    measured_runtime_evidence: bool = False
    calibrated_empirical_evidence: bool = False
    production_authorized: bool = False
    deployment_claim_allowed: bool = False

    def __post_init__(self) -> None:
        _text(self.binding_id, "binding_id")
        _exact_int(self.binding_version, "binding_version", minimum=1)
        if type(self.component_disclosures) is not tuple or any(
            type(item) is not ComponentEvidenceDisclosureV1
            for item in self.component_disclosures
        ):
            raise BindingError(
                "component_disclosures must be an exact tuple of disclosure records"
            )
        roles = tuple(item.role for item in self.component_disclosures)
        if len(roles) != len(_REQUIRED_COMPONENT_ROLES) or set(roles) != set(
            _REQUIRED_COMPONENT_ROLES
        ):
            raise BindingError(
                "modeled composite requires exactly one disclosure for each "
                f"component role: {_REQUIRED_COMPONENT_ROLES!r}"
            )
        for value, name in (
            (self.provider_implementation_sha256, "provider_implementation_sha256"),
            (self.verifier_manifest_sha256, "verifier_manifest_sha256"),
        ):
            _digest(value, name)
        expected = {
            "feature_schema_sha256": contract.FEATURE_SCHEMA_SHA256,
            "reward_schema_sha256": contract.REWARD_SCHEMA_SHA256,
            "transition_schema_sha256": contract.TRANSITION_SCHEMA_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "latency_schema_id": sequential_kernel.SCHEMA_ID,
            "latency_schema_version": sequential_kernel.SCHEMA_VERSION,
            "evidence_class": EVIDENCE_CLASS,
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise BindingError(f"{name} must be {wanted!r}")
        if self.latency_schema_version != 3:
            raise BindingError(
                "MODELED_COMPOSITE_TRAINING requires the v3 authoritative "
                "action-open-to-feedback latency contract"
            )
        _strict_bool(self.offline_training_only, "offline_training_only", True)
        _strict_bool(
            self.measured_runtime_evidence, "measured_runtime_evidence", False
        )
        _strict_bool(
            self.calibrated_empirical_evidence,
            "calibrated_empirical_evidence",
            False,
        )
        _strict_bool(self.production_authorized, "production_authorized", False)
        _strict_bool(
            self.deployment_claim_allowed, "deployment_claim_allowed", False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "binding_version": self.binding_version,
            "calibrated_empirical_evidence": self.calibrated_empirical_evidence,
            "catalog_sha256": self.catalog_sha256,
            "component_disclosures": [
                item.to_dict() for item in self.component_disclosures
            ],
            "deployment_claim_allowed": self.deployment_claim_allowed,
            "evidence_class": self.evidence_class.value,
            "feature_schema_sha256": self.feature_schema_sha256,
            "latency_schema_id": self.latency_schema_id,
            "latency_schema_version": self.latency_schema_version,
            "measured_runtime_evidence": self.measured_runtime_evidence,
            "offline_training_only": self.offline_training_only,
            "production_authorized": self.production_authorized,
            "provider_implementation_sha256": self.provider_implementation_sha256,
            "reward_schema_sha256": self.reward_schema_sha256,
            "transition_schema_sha256": self.transition_schema_sha256,
            "verifier_manifest_sha256": self.verifier_manifest_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("splitfusion_run4_modeled_composite_binding_v1", self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class ModeledCompositeSupportUseV1:
    """Per-cycle fit support and transfer disclosure.

    Profile and mode transfer are intentionally separate.  A target outside a
    direct source scope must carry the corresponding UNVALIDATED status and a
    widened uncertainty flag.  It cannot silently inherit the source claim.
    """

    target_profile_label: str
    target_mode_id: int
    source_profile_labels: Tuple[str, ...]
    source_mode_ids: Tuple[int, ...]
    profile_transfer_status: ProfileTransferStatus
    mode_transfer_status: ModeTransferStatus
    payload_in_fit_support: bool
    backlog_in_fit_support: bool
    mcs_in_fit_support: bool
    quality_in_fit_support: bool
    total_latency_residual_in_fit_support: bool
    widened_uncertainty_applied: bool
    support_evidence_sha256: str

    def __post_init__(self) -> None:
        _text(self.target_profile_label, "target_profile_label")
        _exact_int(self.target_mode_id, "target_mode_id")
        if not 0 <= self.target_mode_id < 12:
            raise SupportError("target_mode_id must lie in [0, 11]")
        if type(self.source_profile_labels) is not tuple or not (
            self.source_profile_labels
        ) or any(not isinstance(value, str) or not value for value in self.source_profile_labels):
            raise SupportError("source_profile_labels must be a nonempty tuple")
        if len(set(self.source_profile_labels)) != len(self.source_profile_labels):
            raise SupportError("source_profile_labels contains duplicates")
        if type(self.source_mode_ids) is not tuple or not self.source_mode_ids:
            raise SupportError("source_mode_ids must be a nonempty tuple")
        if len(set(self.source_mode_ids)) != len(self.source_mode_ids) or any(
            type(value) is not int or not 0 <= value < 12
            for value in self.source_mode_ids
        ):
            raise SupportError("source_mode_ids must be unique exact ids in [0, 11]")
        if type(self.profile_transfer_status) is not ProfileTransferStatus:
            raise SupportError("profile_transfer_status has a foreign type")
        if type(self.mode_transfer_status) is not ModeTransferStatus:
            raise SupportError("mode_transfer_status has a foreign type")

        profile_direct = self.target_profile_label in self.source_profile_labels
        expected_profile = (
            ProfileTransferStatus.PROFILE_WITHIN_DIRECT_FIT_SUPPORT
            if profile_direct
            else ProfileTransferStatus.PROFILE_TRANSFER_UNVALIDATED
        )
        if self.profile_transfer_status is not expected_profile:
            raise SupportError(
                "profile transfer status contradicts source/target profile scope"
            )
        mode_direct = self.target_mode_id in self.source_mode_ids
        expected_mode = (
            ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT
            if mode_direct
            else ModeTransferStatus.MODE_TRANSFER_UNVALIDATED
        )
        if self.mode_transfer_status is not expected_mode:
            raise SupportError(
                "mode transfer status contradicts source/target mode scope"
            )

        for name in (
            "payload_in_fit_support",
            "backlog_in_fit_support",
            "mcs_in_fit_support",
            "quality_in_fit_support",
            "total_latency_residual_in_fit_support",
        ):
            if getattr(self, name) is not True:
                raise SupportError(f"{name} must be exactly True")
        transfer_unvalidated = not (profile_direct and mode_direct)
        if self.widened_uncertainty_applied is not transfer_unvalidated:
            raise SupportError(
                "widened_uncertainty_applied must be true exactly when profile "
                "or mode transfer is unvalidated"
            )
        _digest(self.support_evidence_sha256, "support_evidence_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backlog_in_fit_support": self.backlog_in_fit_support,
            "mcs_in_fit_support": self.mcs_in_fit_support,
            "mode_transfer_status": self.mode_transfer_status.value,
            "payload_in_fit_support": self.payload_in_fit_support,
            "profile_transfer_status": self.profile_transfer_status.value,
            "quality_in_fit_support": self.quality_in_fit_support,
            "source_mode_ids": list(self.source_mode_ids),
            "source_profile_labels": list(self.source_profile_labels),
            "support_evidence_sha256": self.support_evidence_sha256,
            "target_mode_id": self.target_mode_id,
            "target_profile_label": self.target_profile_label,
            "total_latency_residual_in_fit_support": (
                self.total_latency_residual_in_fit_support
            ),
            "widened_uncertainty_applied": self.widened_uncertainty_applied,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("splitfusion_run4_modeled_support_use_v1", self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class FeedbackEndpointPairV1:
    """Ordered modeled endpoints defining one authoritative feedback total.

    The retained direct action-50 probe executed a fixed action. Its source
    total therefore begins after policy inference and action materialization,
    whereas the Run-4 action-open boundary begins before those operations. A
    positive, evidence-bound actor inference + quantization/dispatch delay is
    mandatory here. This makes a zero-cost actor impossible to assume.
    """

    action_open_timestamp_ns: int
    feedback_received_timestamp_ns: int
    clock_domain: str
    source_row_sha256: str
    fixed_action_source_total_ns: int
    fixed_action_source_total_evidence_sha256: str
    actor_inference_ns: int
    actor_inference_evidence_sha256: str
    quantization_dispatch_ns: int
    quantization_dispatch_evidence_sha256: str
    actor_delay_augmentation_applied: bool = True

    def __post_init__(self) -> None:
        opened = _exact_int(
            self.action_open_timestamp_ns, "action_open_timestamp_ns"
        )
        received = _exact_int(
            self.feedback_received_timestamp_ns,
            "feedback_received_timestamp_ns",
            minimum=1,
        )
        if received <= opened:
            raise ModeledCompositeContractError(
                "feedback_received_timestamp_ns must be strictly after action open"
            )
        _text(self.clock_domain, "clock_domain")
        _digest(self.source_row_sha256, "source_row_sha256")
        source_total = _exact_int(
            self.fixed_action_source_total_ns,
            "fixed_action_source_total_ns",
            minimum=1,
        )
        actor_inference = _exact_int(
            self.actor_inference_ns, "actor_inference_ns"
        )
        dispatch = _exact_int(
            self.quantization_dispatch_ns, "quantization_dispatch_ns"
        )
        for value, name in (
            (
                self.fixed_action_source_total_evidence_sha256,
                "fixed_action_source_total_evidence_sha256",
            ),
            (
                self.actor_inference_evidence_sha256,
                "actor_inference_evidence_sha256",
            ),
            (
                self.quantization_dispatch_evidence_sha256,
                "quantization_dispatch_evidence_sha256",
            ),
        ):
            _digest(value, name)
        _strict_bool(
            self.actor_delay_augmentation_applied,
            "actor_delay_augmentation_applied",
            True,
        )
        if actor_inference + dispatch <= 0:
            raise ModeledCompositeContractError(
                "fixed-action source total requires positive measured/modeled "
                "actor inference or quantization/dispatch augmentation"
            )
        if received - opened != source_total + actor_inference + dispatch:
            raise ModeledCompositeContractError(
                "ordered endpoint total must equal fixed-action source total plus "
                "actor inference and quantization/dispatch augmentation"
            )

    @property
    def action_open_to_feedback_ns(self) -> int:
        return self.feedback_received_timestamp_ns - self.action_open_timestamp_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_open_timestamp_ns": self.action_open_timestamp_ns,
            "actor_delay_augmentation_applied": (
                self.actor_delay_augmentation_applied
            ),
            "actor_inference_evidence_sha256": (
                self.actor_inference_evidence_sha256
            ),
            "actor_inference_ns": self.actor_inference_ns,
            "clock_domain": self.clock_domain,
            "feedback_received_timestamp_ns": self.feedback_received_timestamp_ns,
            "fixed_action_source_total_evidence_sha256": (
                self.fixed_action_source_total_evidence_sha256
            ),
            "fixed_action_source_total_ns": self.fixed_action_source_total_ns,
            "quantization_dispatch_evidence_sha256": (
                self.quantization_dispatch_evidence_sha256
            ),
            "quantization_dispatch_ns": self.quantization_dispatch_ns,
            "source_row_sha256": self.source_row_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("splitfusion_run4_modeled_feedback_endpoints_v1", self.to_dict())
        )


@dataclass(frozen=True, slots=True)
class LatencyProjectionV1:
    """Deadline-safe projection of ordered endpoints into the v3 contract."""

    endpoints: FeedbackEndpointPairV1
    terminal_kind: sequential_kernel.KernelTerminalKind
    terminal_elapsed_ns: int
    latency: Optional[sequential_kernel.FeedbackLatencyBreakdownV1]
    late_orphan_action_open_to_feedback_ns: Optional[int]

    @classmethod
    def from_ordered_endpoints(
        cls, endpoints: FeedbackEndpointPairV1
    ) -> "LatencyProjectionV1":
        if type(endpoints) is not FeedbackEndpointPairV1:
            raise ModeledCompositeContractError(
                "endpoints must be exactly FeedbackEndpointPairV1"
            )
        total = endpoints.action_open_to_feedback_ns
        if total <= contract.REWARD_DEADLINE_NS:
            latency = sequential_kernel.FeedbackLatencyBreakdownV1(
                action_open_to_feedback_ns=total,
                action_open_to_feedback_evidence_sha256=(
                    endpoints.canonical_sha256
                ),
                ue_action_path_ns=None,
                feature_uplink_ns=None,
                edge_pre_model_ns=None,
                model_tail_ns=None,
                post_model_feedback_preparation_ns=None,
                feedback_downlink_ns=None,
                ue_action_path_evidence_sha256=None,
                feature_uplink_evidence_sha256=None,
                edge_pre_model_evidence_sha256=None,
                model_tail_evidence_sha256=None,
                post_model_feedback_preparation_evidence_sha256=None,
                feedback_downlink_evidence_sha256=None,
            )
            return cls(
                endpoints=endpoints,
                terminal_kind=(
                    sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK
                ),
                terminal_elapsed_ns=total,
                latency=latency,
                late_orphan_action_open_to_feedback_ns=None,
            )
        return cls(
            endpoints=endpoints,
            terminal_kind=sequential_kernel.KernelTerminalKind.TIMEOUT,
            terminal_elapsed_ns=sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS,
            latency=None,
            late_orphan_action_open_to_feedback_ns=total,
        )

    def __post_init__(self) -> None:
        if type(self.endpoints) is not FeedbackEndpointPairV1:
            raise ModeledCompositeContractError(
                "endpoints must be exactly FeedbackEndpointPairV1"
            )
        if type(self.terminal_kind) is not sequential_kernel.KernelTerminalKind:
            raise ModeledCompositeContractError("terminal_kind has a foreign type")
        elapsed = _exact_int(
            self.terminal_elapsed_ns, "terminal_elapsed_ns", minimum=1
        )
        total = self.endpoints.action_open_to_feedback_ns
        if total <= contract.REWARD_DEADLINE_NS:
            if self.terminal_kind is not (
                sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK
            ):
                raise ModeledCompositeContractError(
                    "on-time ordered endpoints must project to DELIVERED_FEEDBACK"
                )
            if elapsed != total:
                raise ModeledCompositeContractError(
                    "delivered terminal elapsed must equal endpoint-derived total"
                )
            if type(self.latency) is not (
                sequential_kernel.FeedbackLatencyBreakdownV1
            ):
                raise ModeledCompositeContractError(
                    "on-time feedback requires the v3 authoritative latency record"
                )
            if self.latency.action_open_to_feedback_ns != total or (
                self.latency.action_open_to_feedback_evidence_sha256
                != self.endpoints.canonical_sha256
            ):
                raise ModeledCompositeContractError(
                    "v3 latency differs from ordered endpoints"
                )
            if self.latency.has_diagnostic_breakdown:
                raise ModeledCompositeContractError(
                    "modeled total cannot fabricate a diagnostic stage breakdown"
                )
            if self.late_orphan_action_open_to_feedback_ns is not None:
                raise ModeledCompositeContractError(
                    "on-time feedback cannot carry a late-orphan total"
                )
        else:
            if self.terminal_kind is not sequential_kernel.KernelTerminalKind.TIMEOUT:
                raise ModeledCompositeContractError(
                    "feedback later than 170 ms must project to TIMEOUT"
                )
            if elapsed != sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS:
                raise ModeledCompositeContractError(
                    "late feedback must close at the first nanosecond after the "
                    "inclusive deadline"
                )
            if self.latency is not None:
                raise ModeledCompositeContractError(
                    "late feedback cannot carry successful-feedback latency"
                )
            if self.late_orphan_action_open_to_feedback_ns != total:
                raise ModeledCompositeContractError(
                    "late-orphan audit total must equal the endpoint-derived total"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoints": self.endpoints.to_dict(),
            "late_orphan_action_open_to_feedback_ns": (
                self.late_orphan_action_open_to_feedback_ns
            ),
            "latency": None if self.latency is None else self.latency.to_dict(),
            "terminal_elapsed_ns": self.terminal_elapsed_ns,
            "terminal_kind": self.terminal_kind.value,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _record("splitfusion_run4_modeled_latency_projection_v1", self.to_dict())
        )


def _make_attestation_gate():
    sentinel = object()

    def issue(binding: str) -> tuple[object, str]:
        return sentinel, binding

    def valid(token: object, binding: str) -> bool:
        return (
            type(token) is tuple
            and len(token) == 2
            and token[0] is sentinel
            and token[1] == binding
        )

    return issue, valid


_issue_envelope, _valid_envelope = _make_attestation_gate()
_issue_offline_export, _valid_offline_export = _make_attestation_gate()

@dataclass(frozen=True, slots=True)
class ModeledCompositeOfflineTransitionV1:
    """Non-bare modeled export accepted only by a future modeled replay seam.

    The raw transition remains private. Existing replay accepts only the exact
    bare SemiMarkovTransitionV2 type and therefore rejects this record.
    """

    evidence_class: ModeledCompositeEvidenceClass
    modeled_binding_sha256: str
    source_envelope_sha256: str
    support_use_sha256: str
    latency_projection_sha256: str
    transition_sha256: str
    _transition: contract.SemiMarkovTransitionV2 = field(
        repr=False, compare=False
    )
    _attestation: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.evidence_class is not EVIDENCE_CLASS:
            raise BindingError(
                "offline transition must remain MODELED_COMPOSITE_TRAINING"
            )
        for value, name in (
            (self.modeled_binding_sha256, "modeled_binding_sha256"),
            (self.source_envelope_sha256, "source_envelope_sha256"),
            (self.support_use_sha256, "support_use_sha256"),
            (self.latency_projection_sha256, "latency_projection_sha256"),
            (self.transition_sha256, "transition_sha256"),
        ):
            _digest(value, name)
        if type(self._transition) is not contract.SemiMarkovTransitionV2:
            raise ModeledCompositeContractError(
                "offline export requires exactly SemiMarkovTransitionV2 internally"
            )
        self._transition.require_attested()
        if self._transition.canonical_sha256() != self.transition_sha256:
            raise ModeledCompositeContractError(
                "offline export transition digest differs"
            )
        if self._attestation is not None and not self.is_attested:
            raise ModeledCompositeContractError(
                "offline transition attestation is invalid"
            )

    def _binding(self) -> str:
        self._transition.require_attested()
        return canonical_sha256(
            _record(
                "splitfusion_run4_modeled_offline_transition_v1",
                {
                    "evidence_class": self.evidence_class.value,
                    "latency_projection_sha256": self.latency_projection_sha256,
                    "modeled_binding_sha256": self.modeled_binding_sha256,
                    "source_envelope_sha256": self.source_envelope_sha256,
                    "support_use_sha256": self.support_use_sha256,
                    "transition_sha256": self._transition.canonical_sha256(),
                },
            )
        )

    @property
    def is_attested(self) -> bool:
        try:
            return _valid_offline_export(self._attestation, self._binding())
        except (contract.TransitionError, ModeledCompositeContractError):
            return False

    def require_attested(self) -> None:
        if not self.is_attested:
            raise ModeledCompositeContractError(
                "modeled offline transition is absent, forged, or stale"
            )

    def _sealed_transition_for_modeled_replay(
        self, expected_modeled_binding_sha256: str
    ) -> contract.SemiMarkovTransitionV2:
        """Package-private hand-off to the dedicated modeled replay only.

        This is deliberately not a general export.  It first revalidates the
        wrapper and enclosed transition, then requires the caller's exact
        modeled-composite binding digest.  Production replay continues to see
        only the outer wrapper and rejects it by exact type; callers have no
        public method that turns modeled evidence into a bare transition.
        """

        self.require_attested()
        _digest(
            expected_modeled_binding_sha256,
            "expected_modeled_binding_sha256",
        )
        if expected_modeled_binding_sha256 != self.modeled_binding_sha256:
            raise BindingError(
                "modeled replay binding differs from the sealed transition"
            )
        self._transition.require_attested()
        if self._transition.canonical_sha256() != self.transition_sha256:
            raise ModeledCompositeContractError(
                "sealed transition digest changed after offline export"
            )
        return self._transition

    @property
    def terminal(self) -> contract.RewardTerminal:
        self.require_attested()
        return self._transition.reward_resolution.terminal

    @property
    def reward(self) -> float:
        self.require_attested()
        return self._transition.reward

    @property
    def q_perc(self) -> Optional[float]:
        self.require_attested()
        return self._transition.reward_resolution.q_perc

    @property
    def latency_ms(self) -> Optional[float]:
        self.require_attested()
        return self._transition.reward_resolution.latency_ms

    @property
    def gamma(self) -> float:
        self.require_attested()
        return self._transition.gamma

    @property
    def freshness_policy_sha256(self) -> str:
        self.require_attested()
        return self._transition.state.freshness_policy_sha256

    @property
    def empirical_scaling_sha256(self) -> str:
        self.require_attested()
        return self._transition.state_features.empirical_scaling_sha256

    @property
    def replay_export_allowed(self) -> bool:
        return False

    def export_for_replay(self) -> contract.SemiMarkovTransitionV2:
        raise ProductionEvidenceRejected(
            "a modeled offline transition cannot enter production replay"
        )

    def to_audit_dict(self) -> dict[str, Any]:
        self.require_attested()
        return {
            "evidence_class": self.evidence_class.value,
            "latency_projection_sha256": self.latency_projection_sha256,
            "modeled_binding_sha256": self.modeled_binding_sha256,
            "offline_training_only": True,
            "production_authorized": False,
            "source_envelope_sha256": self.source_envelope_sha256,
            "support_use_sha256": self.support_use_sha256,
            "transition_sha256": self.transition_sha256,
        }



@dataclass(frozen=True, slots=True)
class ModeledCompositeTrainingEnvelopeV1:
    """Attested offline envelope; never a calibrated/production cycle."""

    evidence_class: ModeledCompositeEvidenceClass
    modeled_binding_sha256: str
    support_use: ModeledCompositeSupportUseV1
    latency_projection: LatencyProjectionV1
    transition_sha256: str
    _transition: contract.SemiMarkovTransitionV2 = field(
        repr=False, compare=False
    )
    _attestation: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.evidence_class is not EVIDENCE_CLASS:
            raise BindingError(
                "modeled envelope must remain MODELED_COMPOSITE_TRAINING"
            )
        _digest(self.modeled_binding_sha256, "modeled_binding_sha256")
        if type(self.support_use) is not ModeledCompositeSupportUseV1:
            raise SupportError("support_use has a foreign type")
        if type(self.latency_projection) is not LatencyProjectionV1:
            raise ModeledCompositeContractError(
                "latency_projection has a foreign type"
            )
        _digest(self.transition_sha256, "transition_sha256")
        if type(self._transition) is not contract.SemiMarkovTransitionV2:
            raise ModeledCompositeContractError(
                "envelope requires exactly SemiMarkovTransitionV2"
            )
        self._transition.require_attested()
        if self._transition.canonical_sha256() != self.transition_sha256:
            raise ModeledCompositeContractError(
                "transition digest differs from the enclosed transition"
            )
        if self._attestation is not None and not self.is_attested:
            raise ModeledCompositeContractError("offline envelope attestation is invalid")

    def _binding(self) -> str:
        self._transition.require_attested()
        return canonical_sha256(
            _record(
                "splitfusion_run4_modeled_composite_training_envelope_v1",
                {
                    "evidence_class": self.evidence_class.value,
                    "latency_projection_sha256": (
                        self.latency_projection.canonical_sha256
                    ),
                    "modeled_binding_sha256": self.modeled_binding_sha256,
                    "support_use_sha256": self.support_use.canonical_sha256,
                    "transition_sha256": self._transition.canonical_sha256(),
                },
            )
        )

    @property
    def is_attested(self) -> bool:
        try:
            return _valid_envelope(self._attestation, self._binding())
        except (contract.TransitionError, ModeledCompositeContractError):
            return False

    @property
    def offline_training_export_allowed(self) -> bool:
        return self.is_attested

    @property
    def replay_export_allowed(self) -> bool:
        """Production-compatible name: modeled evidence is always refused."""

        return False

    def export_for_offline_training(self) -> ModeledCompositeOfflineTransitionV1:
        """Return a typed modeled export; never a bare replay transition."""

        if not self.is_attested:
            raise ModeledCompositeContractError(
                "modeled-composite envelope is absent, forged, or stale"
            )
        self._transition.require_attested()
        candidate = ModeledCompositeOfflineTransitionV1(
            evidence_class=EVIDENCE_CLASS,
            modeled_binding_sha256=self.modeled_binding_sha256,
            source_envelope_sha256=self._binding(),
            support_use_sha256=self.support_use.canonical_sha256,
            latency_projection_sha256=self.latency_projection.canonical_sha256,
            transition_sha256=self._transition.canonical_sha256(),
            _transition=self._transition,
        )
        return replace(
            candidate,
            _attestation=_issue_offline_export(candidate._binding()),
        )

    def export_for_replay(self) -> contract.SemiMarkovTransitionV2:
        raise ProductionEvidenceRejected(
            "MODELED_COMPOSITE_TRAINING cannot be exported as calibrated or "
            "production replay evidence; use the dedicated offline-training seam"
        )

    def to_audit_dict(self) -> dict[str, Any]:
        if not self.is_attested:
            raise ModeledCompositeContractError("cannot audit an unattested envelope")
        return {
            "deployment_claim_allowed": False,
            "evidence_class": self.evidence_class.value,
            "latency_projection": self.latency_projection.to_dict(),
            "measured_runtime_evidence": False,
            "modeled_binding_sha256": self.modeled_binding_sha256,
            "offline_training_only": True,
            "production_authorized": False,
            "support_use": self.support_use.to_dict(),
            "transition_sha256": self.transition_sha256,
        }


class ModeledCompositeTrainingIssuerV1:
    """Issue offline envelopes after joining the v3 latency and transition."""

    def __init__(self, binding: ModeledCompositeBindingV1) -> None:
        if type(binding) is not ModeledCompositeBindingV1:
            raise BindingError("binding must be exactly ModeledCompositeBindingV1")
        self._binding = binding

    @property
    def binding(self) -> ModeledCompositeBindingV1:
        return self._binding

    def issue(
        self,
        *,
        transition: contract.SemiMarkovTransitionV2,
        support_use: ModeledCompositeSupportUseV1,
        latency_projection: LatencyProjectionV1,
    ) -> ModeledCompositeTrainingEnvelopeV1:
        if type(transition) is not contract.SemiMarkovTransitionV2:
            raise ModeledCompositeContractError(
                "transition must be exactly SemiMarkovTransitionV2"
            )
        transition.require_attested()
        if type(support_use) is not ModeledCompositeSupportUseV1:
            raise SupportError("support_use has a foreign type")
        if type(latency_projection) is not LatencyProjectionV1:
            raise ModeledCompositeContractError(
                "latency_projection has a foreign type"
            )
        if transition.action.mode_id != support_use.target_mode_id:
            raise SupportError("support target mode differs from executed action")

        opened = transition.state.boundary.action_open_timestamp_ns
        endpoints = latency_projection.endpoints
        if opened != endpoints.action_open_timestamp_ns:
            raise ModeledCompositeContractError(
                "latency endpoints do not start at this transition's action open"
            )
        if transition.state.boundary.clock_domain != endpoints.clock_domain:
            raise ModeledCompositeContractError(
                "latency endpoints use a different clock domain"
            )
        resolution = transition.reward_resolution
        resolution.require_attested()
        if latency_projection.terminal_kind is (
            sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK
        ):
            if resolution.terminal is not contract.RewardTerminal.SUCCESS:
                raise ModeledCompositeContractError(
                    "on-time v3 latency requires a SUCCESS transition"
                )
            if resolution.resolution_timestamp_ns != (
                endpoints.feedback_received_timestamp_ns
            ):
                raise ModeledCompositeContractError(
                    "success resolution is not the ordered feedback endpoint"
                )
            expected_ms = endpoints.action_open_to_feedback_ns / 1_000_000.0
            if resolution.latency_ms != expected_ms:
                raise ModeledCompositeContractError(
                    "success reward latency differs from authoritative total"
                )
        else:
            if latency_projection.terminal_kind is not (
                sequential_kernel.KernelTerminalKind.TIMEOUT
            ):
                raise ModeledCompositeContractError(
                    "modeled composite supports only delivered feedback or timeout"
                )
            if resolution.terminal is not contract.RewardTerminal.TIMEOUT:
                raise ModeledCompositeContractError(
                    "late ordered endpoints must use the timeout reward path"
                )
            expected_resolution = opened + (
                sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS
            )
            if resolution.resolution_timestamp_ns != expected_resolution:
                raise ModeledCompositeContractError(
                    "timeout transition did not close at deadline + 1 ns"
                )
            if resolution.q_perc is not None or resolution.latency_ms is not None:
                raise ModeledCompositeContractError(
                    "timeout transition cannot retain successful quality/latency"
                )
        if transition.cycle_end_timestamp_ns < resolution.resolution_timestamp_ns:
            raise ModeledCompositeContractError(
                "transition cycle ends before terminal closure"
            )

        candidate = ModeledCompositeTrainingEnvelopeV1(
            evidence_class=EVIDENCE_CLASS,
            modeled_binding_sha256=self._binding.canonical_sha256,
            support_use=support_use,
            latency_projection=latency_projection,
            transition_sha256=transition.canonical_sha256(),
            _transition=transition,
        )
        return replace(candidate, _attestation=_issue_envelope(candidate._binding()))
