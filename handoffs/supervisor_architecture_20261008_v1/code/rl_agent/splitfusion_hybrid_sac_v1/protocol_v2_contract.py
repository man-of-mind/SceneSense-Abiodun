"""Fail-closed protocol-v2 records for exact dynamic-action rewards.

This module is deliberately a contract layer only.  It performs no filesystem,
CARLA, CUDA, radio or network work.  It closes four identity/provenance gaps
left by the anchor-only v1 quality protocol:

* :class:`DynamicFeatureRequestV2` carries the complete, reconciled executed
  action and tensor transaction, including an off-anchor ``q_e4``;
* :class:`CarlaSourceManifestV2` is issued only by a factory which hashes the
  complete required source-artifact set before an ACK can be built;
* :class:`QualityAckV2` binds the request, reward-feedback identity, exact
  quality document and source manifest without substituting a nearby anchor;
* :class:`TimeoutReconciliationV1` records a conservative post-run verdict for
  a timed-out controller ticket without automatically blaming the policy.

The source-manifest attestation is an in-process construction boundary, not a
cryptographic signature or a Python security sandbox.  A future reviewed live
producer adapter will receive the private capability and call the private
``_authenticate_from_reviewed_producer`` seam with the raw immutable bytes it
read.  The important contract property is that ``producer_status`` is derived
by that seam and is not a caller field.  Until that adapter exists, exact
positive ACKs remain explicitly not learning-ready.

Every dataclass is frozen and canonical serialization revalidates all nested
records.  Hashes use the repository's established canonical JSON definition.
Importing this module reads no files and initializes no runtime service.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .reward_ticket_controller import (
    CompletedTicket,
    FeedbackTerminalStatus,
    RewardFeedbackMessage,
    RewardTicketControllerError,
    TerminalClass,
)
from .state_reward_transition_contract import (
    LocalizationCombiner,
    RewardSpecV1,
)
from .transaction_identity import (
    ExecutedActionIdentity,
    RewardFeedbackIdentity,
    TensorTransmissionEnvelope,
    canonical_json_bytes,
    canonical_sha256,
)

__all__ = [
    "ProtocolV2ContractError",
    "SourceAuthenticationError",
    "QualityAckV2Error",
    "TimeoutReconciliationError",
    "DYNAMIC_FEATURE_REQUEST_SCHEMA_ID",
    "DYNAMIC_FEATURE_REQUEST_SCHEMA_VERSION",
    "DYNAMIC_FEATURE_REQUEST_SCHEMA_SHA256",
    "DYNAMIC_FEATURE_REQUEST_SCHEMA_DESCRIPTOR",
    "CARLA_SOURCE_MANIFEST_SCHEMA_ID",
    "CARLA_SOURCE_MANIFEST_SCHEMA_VERSION",
    "CARLA_SOURCE_MANIFEST_SCHEMA_SHA256",
    "CARLA_SOURCE_MANIFEST_SCHEMA_DESCRIPTOR",
    "QUALITY_ACK_V2_SCHEMA_ID",
    "QUALITY_ACK_V2_SCHEMA_VERSION",
    "QUALITY_ACK_V2_SCHEMA_SHA256",
    "QUALITY_ACK_V2_SCHEMA_DESCRIPTOR",
    "TIMEOUT_RECONCILIATION_SCHEMA_ID",
    "TIMEOUT_RECONCILIATION_SCHEMA_VERSION",
    "TIMEOUT_RECONCILIATION_SCHEMA_SHA256",
    "TIMEOUT_RECONCILIATION_SCHEMA_DESCRIPTOR",
    "CarlaSourceArtifactRole",
    "REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES",
    "CarlaSourceAuthenticationStatus",
    "ServiceFailureStage",
    "TimeoutVerdict",
    "ReconciliationLearningDisposition",
    "DynamicFeatureRequestV2",
    "ExactPerceptionQualityV2",
    "SourceArtifactDigestV2",
    "CarlaSourceManifestV2",
    "QualityAckV2",
    "QualityAckControllerMessageV2",
    "ServiceTerminalEvidenceV2",
    "TimeoutReconciliationV1",
]


# ---------------------------------------------------------------------------
# Errors and scalar validators
# ---------------------------------------------------------------------------


class ProtocolV2ContractError(Exception):
    """Base class for a protocol-v2 contract violation."""


class SourceAuthenticationError(ProtocolV2ContractError):
    """CARLA source evidence is incomplete, inconsistent or unattested."""


class QualityAckV2Error(ProtocolV2ContractError):
    """A quality ACK contradicts its request, identity or source evidence."""


class TimeoutReconciliationError(ProtocolV2ContractError):
    """A timeout verdict is unsupported or lacks the evidence it requires."""


def _non_empty_str(value: Any, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolV2ContractError(
            f"{field_name} must be a non-empty exact str, got {value!r}"
        )
    return value


def _canonical_uuid(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise ProtocolV2ContractError(
            f"{field_name} must be a canonical UUID str, got {value!r}"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProtocolV2ContractError(
            f"{field_name} is not a parsable UUID: {value!r}"
        ) from exc
    if str(parsed) != value:
        raise ProtocolV2ContractError(
            f"{field_name} must be canonical lowercase hyphenated UUID text; "
            f"expected {str(parsed)!r}, got {value!r}"
        )
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolV2ContractError(
            f"{field_name} must be an exact non-negative int, got {value!r}"
        )
    return value


def _positive_int(value: Any, field_name: str) -> int:
    _non_negative_int(value, field_name)
    if value == 0:
        raise ProtocolV2ContractError(f"{field_name} must be > 0")
    return value


def _sha256_hex(value: Any, field_name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProtocolV2ContractError(
            f"{field_name} must be 64 lowercase hexadecimal characters, "
            f"got {value!r}"
        )
    return value


def _optional_sha256(value: Optional[str], field_name: str) -> Optional[str]:
    if value is not None:
        _sha256_hex(value, field_name)
    return value


def _finite_between(
    value: Any,
    field_name: str,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolV2ContractError(
            f"{field_name} must be a finite real number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ProtocolV2ContractError(
            f"{field_name} must be finite in [{lower}, {upper}], got {value!r}"
        )
    return result


def _optional_finite_between(
    value: Optional[float],
    field_name: str,
    lower: float,
    upper: float,
) -> Optional[float]:
    if value is not None:
        _finite_between(value, field_name, lower, upper)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Typed protocol values and immutable schema descriptors
# ---------------------------------------------------------------------------


class CarlaSourceArtifactRole(Enum):
    """Exact source artifacts required before an exact-quality ACK is legal."""

    FEATURE_REQUEST_CANONICAL_JSON = "FEATURE_REQUEST_CANONICAL_JSON"
    CARLA_ACTOR_SNAPSHOT_JSON = "CARLA_ACTOR_SNAPSHOT_JSON"
    CARLA_SEMANTIC_LABEL = "CARLA_SEMANTIC_LABEL"
    PREDICTION_SEGMENTATION_LABEL = "PREDICTION_SEGMENTATION_LABEL"
    PREDICTION_OBJECT_RECORDS_JSON = "PREDICTION_OBJECT_RECORDS_JSON"
    SEGMENTATION_ELIGIBILITY_MASK = "SEGMENTATION_ELIGIBILITY_MASK"
    ACTOR_PROJECTION_ELIGIBILITY_JSON = "ACTOR_PROJECTION_ELIGIBILITY_JSON"
    LOCALIZATION_MATCH_LEDGER_JSON = "LOCALIZATION_MATCH_LEDGER_JSON"
    SEGMENTATION_SUFFICIENT_COUNTS_JSON = (
        "SEGMENTATION_SUFFICIENT_COUNTS_JSON"
    )
    CAMERA_CALIBRATION_JSON = "CAMERA_CALIBRATION_JSON"
    ELIGIBILITY_CONTRACT_JSON = "ELIGIBILITY_CONTRACT_JSON"
    QUALITY_EVALUATION_CANONICAL_JSON = "QUALITY_EVALUATION_CANONICAL_JSON"
    PRODUCER_SOURCE_BYTES = "PRODUCER_SOURCE_BYTES"


REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES: Tuple[CarlaSourceArtifactRole, ...] = tuple(
    CarlaSourceArtifactRole
)


class CarlaSourceAuthenticationStatus(Enum):
    """Factory-derived status; it is intentionally not a constructor field."""

    REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2 = (
        "REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2"
    )


class ServiceFailureStage(Enum):
    """Action-controlled stages whose proven failure may receive a penalty."""

    FEATURE_TRANSFER = "FEATURE_TRANSFER"
    FEATURE_REASSEMBLY = "FEATURE_REASSEMBLY"
    DEQUANTIZE_OR_DECOMPRESS = "DEQUANTIZE_OR_DECOMPRESS"
    MODEL_TAIL_INFERENCE = "MODEL_TAIL_INFERENCE"


class TimeoutVerdict(Enum):
    """The complete conservative timeout-adjudication table."""

    ACTION_PATH_SERVICE_FAILURE_NEGATIVE = (
        "ACTION_PATH_SERVICE_FAILURE_NEGATIVE"
    )
    FEEDBACK_ONLY_LOSS_CENSORED = "FEEDBACK_ONLY_LOSS_CENSORED"
    UE_CONTROL_INFRASTRUCTURE_FAULT_EXCLUDED = (
        "UE_CONTROL_INFRASTRUCTURE_FAULT_EXCLUDED"
    )
    EVALUATOR_INFRASTRUCTURE_FAULT_EXCLUDED = (
        "EVALUATOR_INFRASTRUCTURE_FAULT_EXCLUDED"
    )
    LATE_EXACT_QUALITY_CENSORED = "LATE_EXACT_QUALITY_CENSORED"
    UNRESOLVED_CENSORING = "UNRESOLVED_CENSORING"


class ReconciliationLearningDisposition(Enum):
    """Whether a reconciled timeout may affect policy learning."""

    INCLUDED_REGISTERED_NEGATIVE_SERVICE_REWARD = (
        "INCLUDED_REGISTERED_NEGATIVE_SERVICE_REWARD"
    )
    CENSORED_NO_POLICY_PENALTY = "CENSORED_NO_POLICY_PENALTY"
    EXCLUDED_INFRASTRUCTURE_FAULT = "EXCLUDED_INFRASTRUCTURE_FAULT"


_VERDICT_DISPOSITION = MappingProxyType(
    {
        TimeoutVerdict.ACTION_PATH_SERVICE_FAILURE_NEGATIVE: (
            ReconciliationLearningDisposition.
            INCLUDED_REGISTERED_NEGATIVE_SERVICE_REWARD
        ),
        TimeoutVerdict.FEEDBACK_ONLY_LOSS_CENSORED: (
            ReconciliationLearningDisposition.CENSORED_NO_POLICY_PENALTY
        ),
        TimeoutVerdict.UE_CONTROL_INFRASTRUCTURE_FAULT_EXCLUDED: (
            ReconciliationLearningDisposition.EXCLUDED_INFRASTRUCTURE_FAULT
        ),
        TimeoutVerdict.EVALUATOR_INFRASTRUCTURE_FAULT_EXCLUDED: (
            ReconciliationLearningDisposition.EXCLUDED_INFRASTRUCTURE_FAULT
        ),
        TimeoutVerdict.LATE_EXACT_QUALITY_CENSORED: (
            ReconciliationLearningDisposition.CENSORED_NO_POLICY_PENALTY
        ),
        TimeoutVerdict.UNRESOLVED_CENSORING: (
            ReconciliationLearningDisposition.CENSORED_NO_POLICY_PENALTY
        ),
    }
)


DYNAMIC_FEATURE_REQUEST_SCHEMA_ID = "splitfusion.dynamic_feature_request.v2"
DYNAMIC_FEATURE_REQUEST_SCHEMA_VERSION = 2
DYNAMIC_FEATURE_REQUEST_SCHEMA_DESCRIPTOR = _freeze(
    {
        "schema_id": DYNAMIC_FEATURE_REQUEST_SCHEMA_ID,
        "schema_version": DYNAMIC_FEATURE_REQUEST_SCHEMA_VERSION,
        "identity": (
            "TensorTransmissionEnvelope + controller_lineage_uuid + "
            "policy_decision_trace_sha256"
        ),
        "action_rule": (
            "complete reconciled ExecutedActionIdentity; off-anchor actions "
            "retain exact q_e4 and null anchor identifiers"
        ),
    }
)
DYNAMIC_FEATURE_REQUEST_SCHEMA_SHA256 = canonical_sha256(
    DYNAMIC_FEATURE_REQUEST_SCHEMA_DESCRIPTOR
)

CARLA_SOURCE_MANIFEST_SCHEMA_ID = "splitfusion.carla_source_manifest.v2"
CARLA_SOURCE_MANIFEST_SCHEMA_VERSION = 2
CARLA_SOURCE_MANIFEST_SCHEMA_DESCRIPTOR = _freeze(
    {
        "schema_id": CARLA_SOURCE_MANIFEST_SCHEMA_ID,
        "schema_version": CARLA_SOURCE_MANIFEST_SCHEMA_VERSION,
        "required_artifact_roles": tuple(
            role.value for role in REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES
        ),
        "authentication": (
            "factory hashes immutable raw bytes before ACK construction; "
            "status is factory-derived and non-caller-selectable; capability "
            "is not yet integrated into the live producer; remote "
            "cryptographic trust is out of scope"
        ),
        "ground_truth": "CARLA_GT_EXACT_PRIVILEGED_NON_DEPLOYABLE",
    }
)
CARLA_SOURCE_MANIFEST_SCHEMA_SHA256 = canonical_sha256(
    CARLA_SOURCE_MANIFEST_SCHEMA_DESCRIPTOR
)

QUALITY_ACK_V2_SCHEMA_ID = "splitfusion.privileged_quality_ack.v2"
QUALITY_ACK_V2_SCHEMA_VERSION = 2
QUALITY_ACK_V2_SCHEMA_DESCRIPTOR = _freeze(
    {
        "schema_id": QUALITY_ACK_V2_SCHEMA_ID,
        "schema_version": QUALITY_ACK_V2_SCHEMA_VERSION,
        "identity": "RewardFeedbackIdentity + request SHA-256",
        "terminals": (FeedbackTerminalStatus.REWARD_FINAL.value,),
        "failure_terminal": (
            "fail-closed unsupported until a reviewed typed failure-evidence "
            "producer and learning boundary exist"
        ),
        "exact_quality": (
            "capability-verified CARLA GT; privileged; non-deployable; not "
            "learning-ready until live producer integration"
        ),
    }
)
QUALITY_ACK_V2_SCHEMA_SHA256 = canonical_sha256(QUALITY_ACK_V2_SCHEMA_DESCRIPTOR)

TIMEOUT_RECONCILIATION_SCHEMA_ID = "splitfusion.timeout_reconciliation.v1"
TIMEOUT_RECONCILIATION_SCHEMA_VERSION = 1
TIMEOUT_RECONCILIATION_SCHEMA_DESCRIPTOR = _freeze(
    {
        "schema_id": TIMEOUT_RECONCILIATION_SCHEMA_ID,
        "schema_version": TIMEOUT_RECONCILIATION_SCHEMA_VERSION,
        "ticket": "controller-attested CompletedTicket/FEEDBACK_TIMEOUT",
        "verdicts": tuple(verdict.value for verdict in TimeoutVerdict),
        "default": "UNRESOLVED_CENSORING; no policy penalty",
    }
)
TIMEOUT_RECONCILIATION_SCHEMA_SHA256 = canonical_sha256(
    TIMEOUT_RECONCILIATION_SCHEMA_DESCRIPTOR
)


def _schema_fields(schema_id: str, version: int, digest: str) -> Dict[str, Any]:
    return {
        "schema_id": schema_id,
        "schema_sha256": digest,
        "schema_version": version,
    }


# ---------------------------------------------------------------------------
# Dynamic feature request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DynamicFeatureRequestV2:
    """One dynamic-q feature transmission and its policy commitment."""

    run_id: str
    cell_id: str
    stream_id: str
    controller_lineage_uuid: str
    capture_timestamp_ns: int
    policy_decision_trace_sha256: str
    envelope: TensorTransmissionEnvelope

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name in ("run_id", "cell_id", "stream_id"):
            _non_empty_str(getattr(self, field_name), field_name)
        _canonical_uuid(self.controller_lineage_uuid, "controller_lineage_uuid")
        _positive_int(self.capture_timestamp_ns, "capture_timestamp_ns")
        _sha256_hex(
            self.policy_decision_trace_sha256,
            "policy_decision_trace_sha256",
        )
        if type(self.envelope) is not TensorTransmissionEnvelope:
            raise ProtocolV2ContractError(
                "envelope must be an exact TensorTransmissionEnvelope"
            )
        self.envelope.action.require_reconciled()
        # Serializing the full action is also the authoritative anchor/off-anchor
        # reconciliation check; no action_id-only lookup exists in this schema.
        self.envelope.action.to_canonical_dict()

    @property
    def transaction(self):
        return self.envelope.transaction

    @property
    def action(self) -> ExecutedActionIdentity:
        return self.envelope.action

    @property
    def reward_requested(self) -> bool:
        return self.envelope.reward_requested

    def expected_feedback_identity(self) -> RewardFeedbackIdentity:
        if not self.reward_requested:
            raise ProtocolV2ContractError(
                "a feature request with reward_requested=false has no quality "
                "feedback identity"
            )
        tx = self.transaction
        return RewardFeedbackIdentity(
            session_uuid=tx.session_uuid,
            decision_seq=tx.decision_seq,
            reward_tensor_seq=tx.tensor_seq,
            carla_frame_id=tx.carla_frame_id,
            action=self.action,
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = _schema_fields(
            DYNAMIC_FEATURE_REQUEST_SCHEMA_ID,
            DYNAMIC_FEATURE_REQUEST_SCHEMA_VERSION,
            DYNAMIC_FEATURE_REQUEST_SCHEMA_SHA256,
        )
        payload.update(
            {
                "capture_timestamp_ns": self.capture_timestamp_ns,
                "cell_id": self.cell_id,
                "controller_lineage_uuid": self.controller_lineage_uuid,
                "envelope": self.envelope.to_canonical_dict(),
                "policy_decision_trace_sha256": (
                    self.policy_decision_trace_sha256
                ),
                "record": "dynamic_feature_request_v2",
                "run_id": self.run_id,
                "stream_id": self.stream_id,
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    @property
    def quality_obligation_sha256(self) -> str:
        """The transmission-time quality obligation for a reward request.

        Protocol v2 needs no retrospective identity reconstruction: the exact
        request wire digest is itself the obligation.  Non-reward tensors have
        no quality obligation.
        """
        if not self.reward_requested:
            raise ProtocolV2ContractError(
                "reward_requested=false has no quality obligation"
            )
        return self.canonical_sha256()


# ---------------------------------------------------------------------------
# Exact quality and authenticated CARLA source manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactPerceptionQualityV2:
    """Compact exact per-frame quality carried by a successful v2 ACK.

    ``None`` retains the existing undefined-class convention; it never means
    zero.  The reviewed evaluator is responsible for deriving these values
    from the artifacts which the manifest hashes.
    """

    seg_vehicle_iou: Optional[float]
    seg_person_iou: Optional[float]
    vehicle_recall: Optional[float]
    person_recall: Optional[float]
    vehicle_xy_error_m: Optional[float]
    person_xy_error_m: Optional[float]
    q_seg: Optional[float]
    q_loc: float
    q_perc: float
    reward_spec: RewardSpecV1

    def __post_init__(self) -> None:
        # Canonical numeric representation: accepted ints and floats become
        # floats once, so 0 and 0.0 can never hash as different evidence.
        for field_name in (
            "seg_vehicle_iou",
            "seg_person_iou",
            "vehicle_recall",
            "person_recall",
            "vehicle_xy_error_m",
            "person_xy_error_m",
            "q_seg",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, float(value))
        object.__setattr__(self, "q_loc", float(self.q_loc))
        object.__setattr__(self, "q_perc", float(self.q_perc))
        self.validate()

    def validate(self) -> None:
        for field_name in (
            "seg_vehicle_iou",
            "seg_person_iou",
            "vehicle_recall",
            "person_recall",
            "q_seg",
        ):
            _optional_finite_between(
                getattr(self, field_name), field_name, 0.0, 1.0
            )
        for field_name in ("q_loc", "q_perc"):
            _finite_between(getattr(self, field_name), field_name, 0.0, 1.0)
        for field_name in ("vehicle_xy_error_m", "person_xy_error_m"):
            value = getattr(self, field_name)
            if value is not None:
                _finite_between(value, field_name, 0.0, float("inf"))
        if type(self.reward_spec) is not RewardSpecV1:
            raise SourceAuthenticationError(
                "reward_spec must be an exact RewardSpecV1"
            )

    @property
    def reward_spec_sha256(self) -> str:
        return self.reward_spec.canonical_sha256()

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "person_recall": (
                None if self.person_recall is None else float(self.person_recall)
            ),
            "person_xy_error_m": (
                None
                if self.person_xy_error_m is None
                else float(self.person_xy_error_m)
            ),
            "q_loc": float(self.q_loc),
            "q_perc": float(self.q_perc),
            "q_seg": None if self.q_seg is None else float(self.q_seg),
            "record": "exact_perception_quality_v2",
            "reward_spec": self.reward_spec.to_canonical_dict(),
            "reward_spec_sha256": self.reward_spec_sha256,
            "seg_person_iou": (
                None if self.seg_person_iou is None else float(self.seg_person_iou)
            ),
            "seg_vehicle_iou": (
                None if self.seg_vehicle_iou is None else float(self.seg_vehicle_iou)
            ),
            "vehicle_recall": (
                None if self.vehicle_recall is None else float(self.vehicle_recall)
            ),
            "vehicle_xy_error_m": (
                None
                if self.vehicle_xy_error_m is None
                else float(self.vehicle_xy_error_m)
            ),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class SourceArtifactDigestV2:
    """Hash and exact byte count of one source artifact."""

    role: CarlaSourceArtifactRole
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        if type(self.role) is not CarlaSourceArtifactRole:
            raise SourceAuthenticationError(
                "role must be a CarlaSourceArtifactRole"
            )
        _sha256_hex(self.sha256, f"artifact {self.role.value} sha256")
        _positive_int(self.byte_count, f"artifact {self.role.value} byte_count")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "role": self.role.value,
            "sha256": self.sha256,
        }


def _make_source_manifest_attestation():
    sentinel = object()

    def issue(binding: str) -> Tuple[object, str]:
        return sentinel, binding

    def valid(token: Any, binding: str) -> bool:
        return (
            type(token) is tuple
            and len(token) == 2
            and token[0] is sentinel
            and token[1] == binding
        )

    return issue, valid


_issue_source_manifest, _valid_source_manifest = _make_source_manifest_attestation()


def _make_reviewed_producer_capability():
    """Create a sealed in-process capability issuer/checker.

    Python is not a cryptographic isolation boundary.  The closure prevents
    ordinary callers from constructing a capability or selecting an
    authentication status; remote signatures/PKI are explicitly out of scope.
    """
    sentinel = object()

    def issue() -> Tuple[object, str]:
        return sentinel, "reviewed-carla-producer-v2"

    def valid(value: Any) -> bool:
        return (
            type(value) is tuple
            and len(value) == 2
            and value[0] is sentinel
            and value[1] == "reviewed-carla-producer-v2"
        )

    return issue, valid


_issue_reviewed_producer, _valid_reviewed_producer = (
    _make_reviewed_producer_capability()
)
# Private integration capability.  A later reviewed producer adapter receives
# this object inside this package; it is not exported and cannot be constructed
# from public fields.  Tests deliberately exercise this boundary through the
# private package seam, just as existing contract tests exercise attestations.
_REVIEWED_CARLA_PRODUCER_CAPABILITY_V2 = _issue_reviewed_producer()
_REVIEWED_PRODUCER_SOURCE_BYTES_V2 = (
    b"splitfusion-reviewed-carla-artifact-producer-v2"
)


def _json_object(raw: bytes, role: CarlaSourceArtifactRole) -> Dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAuthenticationError(
            f"{role.value} must be canonical UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise SourceAuthenticationError(f"{role.value} must decode to an object")
    if canonical_json_bytes(value) != raw:
        raise SourceAuthenticationError(
            f"{role.value} is not in canonical JSON form"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: Tuple[str, ...], label: str) -> None:
    if set(value) != set(expected):
        raise SourceAuthenticationError(
            f"{label} fields mismatch: expected={sorted(expected)}, "
            f"got={sorted(value)}"
        )


def _class_segmentation_counts(document: Mapping[str, Any], name: str) -> Dict[str, int]:
    value = document.get(name)
    if type(value) is not dict:
        raise SourceAuthenticationError(f"segmentation counts lack {name}")
    fields = ("gt_pixels", "intersection_pixels", "pred_pixels", "union_pixels")
    _exact_keys(value, fields, f"segmentation {name}")
    result: Dict[str, int] = {}
    for field_name in fields:
        item = value[field_name]
        _non_negative_int(item, f"segmentation {name}.{field_name}")
        result[field_name] = item
    if result["intersection_pixels"] > min(
        result["gt_pixels"], result["pred_pixels"]
    ):
        raise SourceAuthenticationError(
            f"segmentation {name} intersection exceeds one of its masks"
        )
    expected_union = (
        result["gt_pixels"]
        + result["pred_pixels"]
        - result["intersection_pixels"]
    )
    if result["union_pixels"] != expected_union:
        raise SourceAuthenticationError(
            f"segmentation {name} union is not gt + pred - intersection"
        )
    return result


def _class_localization_ledger(document: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = document.get(name)
    if type(value) is not dict:
        raise SourceAuthenticationError(f"localization ledger lacks {name}")
    fields = (
        "eligible_actor_ids",
        "eligible_gt_instances",
        "fn",
        "matched_xy_errors_m",
        "tp",
    )
    _exact_keys(value, fields, f"localization {name}")
    eligible = value["eligible_gt_instances"]
    tp = value["tp"]
    fn = value["fn"]
    for item, field_name in (
        (eligible, "eligible_gt_instances"),
        (tp, "tp"),
        (fn, "fn"),
    ):
        _non_negative_int(item, f"localization {name}.{field_name}")
    if tp + fn != eligible:
        raise SourceAuthenticationError(
            f"localization {name} requires tp + fn == eligible_gt_instances"
        )
    actor_ids = value["eligible_actor_ids"]
    if (
        type(actor_ids) is not list
        or any(type(actor_id) is not int or actor_id < 0 for actor_id in actor_ids)
        or len(actor_ids) != len(set(actor_ids))
        or len(actor_ids) != eligible
    ):
        raise SourceAuthenticationError(
            f"localization {name} eligible_actor_ids must be unique, "
            "non-negative and exact-count matched"
        )
    errors = value["matched_xy_errors_m"]
    if type(errors) is not list or len(errors) != tp:
        raise SourceAuthenticationError(
            f"localization {name} must carry one XY error per true positive"
        )
    normalized_errors = tuple(
        _finite_between(error, f"localization {name} error", 0.0, float("inf"))
        for error in errors
    )
    return {
        "eligible": eligible,
        "tp": tp,
        "fn": fn,
        "errors": normalized_errors,
        "eligible_actor_ids": tuple(actor_ids),
    }


def _require_artifact_digest_binding(
    document: Mapping[str, Any],
    field_name: str,
    raw_artifact: bytes,
    label: str,
) -> None:
    """Require a summary document to name the exact raw artifact it summarizes."""

    declared = document.get(field_name)
    _sha256_hex(declared, f"{label}.{field_name}")
    actual = hashlib.sha256(raw_artifact).hexdigest()
    if declared != actual:
        raise SourceAuthenticationError(
            f"{label}.{field_name} does not bind the supplied raw artifact: "
            f"declared={declared}, actual={actual}"
        )


def _weighted_geometric(terms: Tuple[Tuple[float, float], ...]) -> float:
    if any(value <= 0.0 for _, value in terms):
        return 0.0
    total = sum(weight for weight, _ in terms)
    return math.exp(sum(weight * math.log(value) for weight, value in terms) / total)


def _derive_exact_quality(
    reward_spec: RewardSpecV1,
    segmentation_document: Mapping[str, Any],
    localization_document: Mapping[str, Any],
) -> ExactPerceptionQualityV2:
    if type(reward_spec) is not RewardSpecV1:
        raise SourceAuthenticationError("reward_spec must be an exact RewardSpecV1")
    _exact_keys(
        segmentation_document,
        (
            "camera_calibration_sha256",
            "carla_frame_id",
            "carla_semantic_label_sha256",
            "eligibility_contract_sha256",
            "person",
            "prediction_segmentation_label_sha256",
            "record",
            "segmentation_eligibility_mask_sha256",
            "vehicle",
        ),
        "segmentation sufficient-count document",
    )
    if segmentation_document["record"] != "segmentation_sufficient_counts_v2":
        raise SourceAuthenticationError("wrong segmentation-count record tag")
    _exact_keys(
        localization_document,
        (
            "actor_projection_eligibility_sha256",
            "actor_snapshot_sha256",
            "camera_calibration_sha256",
            "carla_frame_id",
            "eligibility_contract_sha256",
            "person",
            "prediction_object_records_sha256",
            "record",
            "vehicle",
        ),
        "localization match-ledger document",
    )
    if localization_document["record"] != "localization_match_ledger_v2":
        raise SourceAuthenticationError("wrong localization-ledger record tag")

    seg_values: Dict[str, Optional[float]] = {}
    seg_terms = []
    for name in ("person", "vehicle"):
        counts = _class_segmentation_counts(segmentation_document, name)
        union = counts["union_pixels"]
        iou = None if union == 0 else counts["intersection_pixels"] / union
        seg_values[name] = iou
        if iou is not None:
            normalized = min(iou / reward_spec.seg_reference_for(name), 1.0)
            seg_terms.append((reward_spec.seg_weight_for(name), normalized))
    q_seg = None if not seg_terms else _weighted_geometric(tuple(seg_terms))

    recalls: Dict[str, Optional[float]] = {}
    errors: Dict[str, Optional[float]] = {}
    loc_terms = []
    for name in ("person", "vehicle"):
        ledger = _class_localization_ledger(localization_document, name)
        if ledger["eligible"] == 0:
            recalls[name] = None
            errors[name] = None
            continue
        recall = ledger["tp"] / ledger["eligible"]
        recalls[name] = recall
        if ledger["tp"] == 0:
            errors[name] = None
            utility = 0.0
        else:
            error = float(statistics.median(ledger["errors"]))
            errors[name] = error
            utility = math.sqrt(
                recall * math.exp(-error / reward_spec.tau_for(name))
            )
        loc_terms.append((reward_spec.loc_weight_for(name), utility))
    if not loc_terms:
        raise SourceAuthenticationError(
            "exact perception quality is undefined: neither class has eligible GT"
        )
    if reward_spec.localization_combiner is LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN:
        q_loc = _weighted_geometric(tuple(loc_terms))
    else:
        total = sum(weight for weight, _ in loc_terms)
        q_loc = sum(weight * value for weight, value in loc_terms) / total
    modulation = (
        1.0
        if q_seg is None
        else (1.0 - reward_spec.segmentation_modulation_beta)
        + reward_spec.segmentation_modulation_beta * q_seg
    )
    return ExactPerceptionQualityV2(
        seg_vehicle_iou=seg_values["vehicle"],
        seg_person_iou=seg_values["person"],
        vehicle_recall=recalls["vehicle"],
        person_recall=recalls["person"],
        vehicle_xy_error_m=errors["vehicle"],
        person_xy_error_m=errors["person"],
        q_seg=q_seg,
        q_loc=q_loc,
        q_perc=q_loc * modulation,
        reward_spec=reward_spec,
    )


@dataclass(frozen=True, slots=True)
class CarlaSourceManifestV2:
    """Factory-authenticated manifest of every exact-quality source artifact."""

    request: DynamicFeatureRequestV2
    quality: ExactPerceptionQualityV2
    artifacts: Tuple[SourceArtifactDigestV2, ...]
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_shape()
        if self._attestation is not None and not self.is_authenticated:
            raise SourceAuthenticationError(
                "source-manifest attestation does not bind its serialized fields"
            )

    def _validate_shape(self) -> None:
        if type(self.request) is not DynamicFeatureRequestV2:
            raise SourceAuthenticationError(
                "request must be an exact DynamicFeatureRequestV2"
            )
        self.request.validate()
        if not self.request.reward_requested:
            raise SourceAuthenticationError(
                "exact CARLA quality may be produced only for a request whose "
                "reward_requested flag is true"
            )
        if type(self.quality) is not ExactPerceptionQualityV2:
            raise SourceAuthenticationError(
                "quality must be an exact ExactPerceptionQualityV2"
            )
        self.quality.validate()
        if type(self.artifacts) is not tuple:
            raise SourceAuthenticationError("artifacts must be an immutable tuple")
        if any(type(item) is not SourceArtifactDigestV2 for item in self.artifacts):
            raise SourceAuthenticationError(
                "every artifacts member must be a SourceArtifactDigestV2"
            )
        roles = tuple(item.role for item in self.artifacts)
        expected = tuple(sorted(REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES, key=lambda r: r.value))
        if roles != expected:
            missing = sorted(
                role.value for role in set(expected).difference(roles)
            )
            foreign_or_duplicate = [role.value for role in roles if roles.count(role) != 1 or role not in expected]
            raise SourceAuthenticationError(
                "source manifest must contain each required artifact role "
                f"exactly once; missing={missing}, foreign_or_duplicate="
                f"{foreign_or_duplicate}"
            )

    @classmethod
    def _authenticate_from_reviewed_producer(
        cls,
        *,
        producer_capability: object,
        request: DynamicFeatureRequestV2,
        reward_spec: RewardSpecV1,
        raw_artifacts: Mapping[CarlaSourceArtifactRole, bytes],
    ) -> "CarlaSourceManifestV2":
        """Private reviewed-producer seam; not a public self-authorization API.

        The feature-request and quality artifacts must be the exact canonical
        bytes of their typed records.  Every other required role must also be
        present and non-empty.  Unknown roles, string role aliases and mutable
        byte buffers are rejected.
        """
        if not _valid_reviewed_producer(producer_capability):
            raise SourceAuthenticationError(
                "source authentication requires the sealed reviewed-producer "
                "capability; callers cannot self-authorize artifact bytes"
            )
        if type(request) is not DynamicFeatureRequestV2:
            raise SourceAuthenticationError(
                "request must be an exact DynamicFeatureRequestV2"
            )
        request.validate()
        if not request.reward_requested:
            raise SourceAuthenticationError(
                "cannot authenticate exact quality for reward_requested=false"
            )
        if type(reward_spec) is not RewardSpecV1:
            raise SourceAuthenticationError("reward_spec must be an exact RewardSpecV1")
        if not isinstance(raw_artifacts, Mapping):
            raise SourceAuthenticationError("raw_artifacts must be a mapping")
        supplied_roles = set(raw_artifacts.keys())
        required_roles = set(REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES)
        if supplied_roles != required_roles:
            missing = sorted(role.value for role in required_roles - supplied_roles)
            foreign = sorted(repr(role) for role in supplied_roles - required_roles)
            raise SourceAuthenticationError(
                f"raw source-artifact set mismatch: missing={missing}, "
                f"foreign={foreign}"
            )
        copied: Dict[CarlaSourceArtifactRole, bytes] = {}
        for role in REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES:
            raw = raw_artifacts[role]
            if type(raw) is not bytes or not raw:
                raise SourceAuthenticationError(
                    f"raw artifact {role.value} must be non-empty immutable bytes"
                )
            copied[role] = bytes(raw)
        if copied[CarlaSourceArtifactRole.FEATURE_REQUEST_CANONICAL_JSON] != request.canonical_bytes():
            raise SourceAuthenticationError(
                "feature-request artifact bytes do not match the typed request"
            )
        if copied[CarlaSourceArtifactRole.PRODUCER_SOURCE_BYTES] != _REVIEWED_PRODUCER_SOURCE_BYTES_V2:
            raise SourceAuthenticationError(
                "producer source artifact is not the reviewed producer binding"
            )

        # Semantic source validation.  JSON blobs are canonical, record tags
        # and frame identity are exact, projection counts reconcile with the
        # localization ledger, and q_perc is recomputed rather than trusted.
        actor_snapshot = _json_object(
            copied[CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON],
            CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON,
        )
        _exact_keys(actor_snapshot, ("actors", "carla_frame_id", "record"), "actor snapshot")
        if type(actor_snapshot["carla_frame_id"]) is not int:
            raise SourceAuthenticationError(
                "actor snapshot carla_frame_id must be an exact int"
            )
        if actor_snapshot["record"] != "carla_actor_snapshot_v2" or actor_snapshot["carla_frame_id"] != request.transaction.carla_frame_id or type(actor_snapshot["actors"]) is not list:
            raise SourceAuthenticationError("actor snapshot record/frame/actors mismatch")
        actor_ids = []
        for actor in actor_snapshot["actors"]:
            if type(actor) is not dict or type(actor.get("actor_id")) is not int or actor["actor_id"] < 0:
                raise SourceAuthenticationError(
                    "actor snapshot entries require non-negative integer actor_id"
                )
            actor_ids.append(actor["actor_id"])
        if len(actor_ids) != len(set(actor_ids)):
            raise SourceAuthenticationError("actor snapshot actor IDs must be unique")
        projection = _json_object(
            copied[CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON],
            CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON,
        )
        _exact_keys(projection, ("carla_frame_id", "eligible_person_actor_ids", "eligible_vehicle_actor_ids", "record"), "projection eligibility")
        if type(projection["carla_frame_id"]) is not int:
            raise SourceAuthenticationError(
                "projection eligibility carla_frame_id must be an exact int"
            )
        if projection["record"] != "actor_projection_eligibility_v2" or projection["carla_frame_id"] != request.transaction.carla_frame_id:
            raise SourceAuthenticationError("projection eligibility record/frame mismatch")
        for key in ("eligible_person_actor_ids", "eligible_vehicle_actor_ids"):
            values = projection[key]
            if type(values) is not list or any(type(item) is not int or item < 0 for item in values) or len(values) != len(set(values)):
                raise SourceAuthenticationError(f"{key} must be unique non-negative actor IDs")
        for role, record_tag in (
            (CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON, "camera_calibration_v2"),
            (CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON, "eligibility_contract_v2"),
        ):
            document = _json_object(copied[role], role)
            if document.get("record") != record_tag:
                raise SourceAuthenticationError(f"{role.value} has wrong record tag")

        prediction_objects = _json_object(
            copied[CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON],
            CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON,
        )
        _exact_keys(
            prediction_objects,
            ("carla_frame_id", "objects", "record"),
            "prediction object records",
        )
        if type(prediction_objects["carla_frame_id"]) is not int:
            raise SourceAuthenticationError(
                "prediction object records carla_frame_id must be an exact int"
            )
        if (
            prediction_objects["record"] != "prediction_object_records_v2"
            or prediction_objects["carla_frame_id"]
            != request.transaction.carla_frame_id
            or type(prediction_objects["objects"]) is not list
        ):
            raise SourceAuthenticationError(
                "prediction object records record/frame/objects mismatch"
            )

        segmentation_document = _json_object(
            copied[CarlaSourceArtifactRole.SEGMENTATION_SUFFICIENT_COUNTS_JSON],
            CarlaSourceArtifactRole.SEGMENTATION_SUFFICIENT_COUNTS_JSON,
        )
        localization_document = _json_object(
            copied[CarlaSourceArtifactRole.LOCALIZATION_MATCH_LEDGER_JSON],
            CarlaSourceArtifactRole.LOCALIZATION_MATCH_LEDGER_JSON,
        )
        frame_id = request.transaction.carla_frame_id
        _non_negative_int(
            segmentation_document.get("carla_frame_id"),
            "segmentation sufficient-count document.carla_frame_id",
        )
        _non_negative_int(
            localization_document.get("carla_frame_id"),
            "localization match-ledger document.carla_frame_id",
        )
        if (
            segmentation_document.get("carla_frame_id") != frame_id
            or localization_document.get("carla_frame_id") != frame_id
        ):
            raise SourceAuthenticationError(
                "quality summaries belong to a different CARLA frame"
            )

        for document, label, bindings in (
            (
                segmentation_document,
                "segmentation sufficient-count document",
                (
                    (
                        "carla_semantic_label_sha256",
                        CarlaSourceArtifactRole.CARLA_SEMANTIC_LABEL,
                    ),
                    (
                        "prediction_segmentation_label_sha256",
                        CarlaSourceArtifactRole.PREDICTION_SEGMENTATION_LABEL,
                    ),
                    (
                        "segmentation_eligibility_mask_sha256",
                        CarlaSourceArtifactRole.SEGMENTATION_ELIGIBILITY_MASK,
                    ),
                    (
                        "camera_calibration_sha256",
                        CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON,
                    ),
                    (
                        "eligibility_contract_sha256",
                        CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON,
                    ),
                ),
            ),
            (
                localization_document,
                "localization match-ledger document",
                (
                    (
                        "prediction_object_records_sha256",
                        CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON,
                    ),
                    (
                        "actor_snapshot_sha256",
                        CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON,
                    ),
                    (
                        "actor_projection_eligibility_sha256",
                        CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON,
                    ),
                    (
                        "camera_calibration_sha256",
                        CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON,
                    ),
                    (
                        "eligibility_contract_sha256",
                        CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON,
                    ),
                ),
            ),
        ):
            for field_name, role in bindings:
                _require_artifact_digest_binding(
                    document, field_name, copied[role], label
                )
        quality = _derive_exact_quality(
            reward_spec, segmentation_document, localization_document
        )
        for name in ("person", "vehicle"):
            ledger = _class_localization_ledger(localization_document, name)
            actor_ids = projection[f"eligible_{name}_actor_ids"]
            if tuple(actor_ids) != ledger["eligible_actor_ids"]:
                raise SourceAuthenticationError(
                    f"{name} eligible actor set does not reconcile with localization ledger"
                )
            if any(actor_id not in set(
                actor["actor_id"] for actor in actor_snapshot["actors"]
            ) for actor_id in actor_ids):
                raise SourceAuthenticationError(
                    f"{name} eligible actor is absent from actor snapshot"
                )
        if copied[CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON] != quality.canonical_bytes():
            raise SourceAuthenticationError(
                "quality-evaluation artifact does not equal the verifier's "
                "recomputed quality document"
            )
        entries = tuple(
            SourceArtifactDigestV2(
                role=role,
                sha256=hashlib.sha256(copied[role]).hexdigest(),
                byte_count=len(copied[role]),
            )
            for role in sorted(REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES, key=lambda r: r.value)
        )
        record = cls(request=request, quality=quality, artifacts=entries)
        return replace(record, _attestation=_issue_source_manifest(record._binding()))

    @property
    def producer_status(self) -> CarlaSourceAuthenticationStatus:
        return (
            CarlaSourceAuthenticationStatus.
            REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2
        )

    @property
    def learning_ready(self) -> bool:
        """False until the reviewed capability is wired into the live producer."""
        return False

    @property
    def privileged(self) -> bool:
        return True

    @property
    def deployable(self) -> bool:
        return False

    @property
    def is_authenticated(self) -> bool:
        return _valid_source_manifest(self._attestation, self._binding())

    def require_authenticated(self) -> None:
        self._validate_shape()
        if not self.is_authenticated:
            raise SourceAuthenticationError(
                "CARLA source manifest was not issued by the protocol-v2 raw-"
                "artifact authentication factory"
            )

    def artifact(self, role: CarlaSourceArtifactRole) -> SourceArtifactDigestV2:
        if type(role) is not CarlaSourceArtifactRole:
            raise SourceAuthenticationError("artifact role must be typed")
        for item in self.artifacts:
            if item.role is role:
                return item
        raise SourceAuthenticationError(f"manifest lacks {role.value}")

    def _serialized_fields(self) -> Dict[str, Any]:
        payload = _schema_fields(
            CARLA_SOURCE_MANIFEST_SCHEMA_ID,
            CARLA_SOURCE_MANIFEST_SCHEMA_VERSION,
            CARLA_SOURCE_MANIFEST_SCHEMA_SHA256,
        )
        payload.update(
            {
                "artifacts": [item.to_canonical_dict() for item in self.artifacts],
                "deployable": self.deployable,
                "ground_truth_source": "CARLA_GT_EXACT",
                "learning_ready": self.learning_ready,
                "privileged": self.privileged,
                "producer_status": self.producer_status.value,
                "quality": self.quality.to_canonical_dict(),
                "quality_obligation_sha256": (
                    self.request.quality_obligation_sha256
                ),
                "record": "carla_source_manifest_v2",
                "request_sha256": self.request.canonical_sha256(),
            }
        )
        return payload

    def _binding(self) -> str:
        return canonical_sha256(self._serialized_fields())

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_authenticated()
        return self._serialized_fields()

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# ---------------------------------------------------------------------------
# Quality ACK v2
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityAckV2:
    """Exact dynamic-action quality/failure terminal bound to one request."""

    request: DynamicFeatureRequestV2
    feedback_identity: RewardFeedbackIdentity
    terminal_status: FeedbackTerminalStatus
    edge_tail_completed_ns: int
    edge_ack_emitted_ns: int
    source_manifest: Optional[CarlaSourceManifestV2] = None
    failure_evidence_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.request) is not DynamicFeatureRequestV2:
            raise QualityAckV2Error("request must be an exact DynamicFeatureRequestV2")
        self.request.validate()
        if not self.request.reward_requested:
            raise QualityAckV2Error(
                "a request with reward_requested=false must never receive a "
                "quality/failure terminal ACK"
            )
        if type(self.feedback_identity) is not RewardFeedbackIdentity:
            raise QualityAckV2Error(
                "feedback_identity must be an exact RewardFeedbackIdentity"
            )
        expected = self.request.expected_feedback_identity()
        if self.feedback_identity.to_canonical_dict() != expected.to_canonical_dict():
            raise QualityAckV2Error(
                "feedback identity does not exactly match request transaction/"
                "action; anchor substitution, frame reassignment and decision "
                "reuse are forbidden"
            )
        if type(self.terminal_status) is not FeedbackTerminalStatus:
            raise QualityAckV2Error(
                "terminal_status must be a FeedbackTerminalStatus"
            )
        _non_negative_int(self.edge_tail_completed_ns, "edge_tail_completed_ns")
        _non_negative_int(self.edge_ack_emitted_ns, "edge_ack_emitted_ns")
        if self.edge_ack_emitted_ns < self.edge_tail_completed_ns:
            raise QualityAckV2Error(
                "edge_ack_emitted_ns cannot precede edge_tail_completed_ns"
            )
        _optional_sha256(
            self.failure_evidence_sha256, "failure_evidence_sha256"
        )

        if self.terminal_status is FeedbackTerminalStatus.REWARD_FINAL:
            if type(self.source_manifest) is not CarlaSourceManifestV2:
                raise QualityAckV2Error(
                    "REWARD_FINAL requires an authenticated CarlaSourceManifestV2"
                )
            self.source_manifest.require_authenticated()
            if self.source_manifest.request.canonical_sha256() != self.request.canonical_sha256():
                raise QualityAckV2Error(
                    "source manifest belongs to a different feature request"
                )
            quality_artifact = self.source_manifest.artifact(
                CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON
            )
            if quality_artifact.sha256 != self.source_manifest.quality.canonical_sha256():
                raise QualityAckV2Error(
                    "quality document digest disagrees with authenticated source "
                    "manifest"
                )
            if quality_artifact.byte_count != len(
                self.source_manifest.quality.canonical_bytes()
            ):
                raise QualityAckV2Error(
                    "quality document byte count disagrees with source manifest"
                )
            if self.failure_evidence_sha256 is not None:
                raise QualityAckV2Error(
                    "REWARD_FINAL cannot also carry failure evidence"
                )
        else:
            raise QualityAckV2Error(
                "protocol v2 currently supports only REWARD_FINAL.  "
                "ACTION_PATH_FAILURE is fail-closed until a reviewed typed "
                "failure-evidence producer and its learning boundary are "
                "implemented; an opaque failure digest is not evidence"
            )

    @property
    def privileged(self) -> bool:
        return self.terminal_status is FeedbackTerminalStatus.REWARD_FINAL

    @property
    def deployable(self) -> bool:
        return not self.privileged

    @property
    def learning_ready(self) -> bool:
        if self.source_manifest is None:
            return False
        return self.source_manifest.learning_ready

    @property
    def source_authentication_status(
        self,
    ) -> Optional[CarlaSourceAuthenticationStatus]:
        if self.source_manifest is None:
            return None
        return self.source_manifest.producer_status

    @property
    def quality(self) -> Optional[ExactPerceptionQualityV2]:
        """Factory-derived exact quality, never a caller-selected ACK field."""
        if self.source_manifest is None:
            return None
        return self.source_manifest.quality

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = _schema_fields(
            QUALITY_ACK_V2_SCHEMA_ID,
            QUALITY_ACK_V2_SCHEMA_VERSION,
            QUALITY_ACK_V2_SCHEMA_SHA256,
        )
        payload.update(
            {
                "deployable": self.deployable,
                "edge_ack_emitted_ns": self.edge_ack_emitted_ns,
                "edge_tail_completed_ns": self.edge_tail_completed_ns,
                "failure_evidence_sha256": self.failure_evidence_sha256,
                "feedback_identity": self.feedback_identity.to_canonical_dict(),
                "learning_ready": self.learning_ready,
                "privileged": self.privileged,
                "quality": (
                    None if self.quality is None else self.quality.to_canonical_dict()
                ),
                "quality_obligation_sha256": (
                    self.request.quality_obligation_sha256
                ),
                "record": "quality_ack_v2",
                "request_sha256": self.request.canonical_sha256(),
                "source_authentication_status": (
                    None
                    if self.source_authentication_status is None
                    else self.source_authentication_status.value
                ),
                "source_manifest_sha256": (
                    None
                    if self.source_manifest is None
                    else self.source_manifest.canonical_sha256()
                ),
                "terminal_status": self.terminal_status.value,
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class QualityAckControllerMessageV2(RewardFeedbackMessage):
    """Controller-compatible carrier whose digest is the raw v2 ACK digest.

    The existing controller remains unchanged.  Because it accepts
    :class:`RewardFeedbackMessage` subclasses and obtains the duplicate key via
    ``canonical_sha256()``, this carrier makes the controller persist the exact
    raw :class:`QualityAckV2` digest in ``accepted_feedback_sha256``.  Quality
    changes therefore cannot collapse to the identity/status-only v1 message.
    """

    raw_ack: QualityAckV2

    def __post_init__(self) -> None:
        self._require_carrier_consistent()

    def _require_carrier_consistent(self) -> None:
        """Revalidate every controller-visible field against the raw ACK.

        The controller accepts :class:`RewardFeedbackMessage` subclasses and
        reads their inherited identity/status-derived properties after hashing
        them.  Rechecking at every one of those boundaries prevents deliberate
        ``object.__setattr__`` tampering from making the controller consume
        fields which differ from the exact raw ACK whose digest it persists.
        """
        super(QualityAckControllerMessageV2, self).__post_init__()
        if type(self.raw_ack) is not QualityAckV2:
            raise QualityAckV2Error("raw_ack must be an exact QualityAckV2")
        self.raw_ack.validate()
        if self.identity.to_canonical_dict() != self.raw_ack.feedback_identity.to_canonical_dict():
            raise QualityAckV2Error(
                "controller carrier identity disagrees with raw quality ACK"
            )
        if self.terminal_status is not self.raw_ack.terminal_status:
            raise QualityAckV2Error(
                "controller carrier terminal status disagrees with raw quality ACK"
            )

    @property
    def session_uuid(self) -> str:
        self._require_carrier_consistent()
        return self.raw_ack.feedback_identity.session_uuid

    @property
    def decision_seq(self) -> int:
        self._require_carrier_consistent()
        return self.raw_ack.feedback_identity.decision_seq

    @property
    def reward_tensor_seq(self) -> int:
        self._require_carrier_consistent()
        return self.raw_ack.feedback_identity.reward_tensor_seq

    @property
    def carla_frame_id(self) -> int:
        self._require_carrier_consistent()
        return self.raw_ack.feedback_identity.carla_frame_id

    @property
    def action(self) -> ExecutedActionIdentity:
        self._require_carrier_consistent()
        return self.raw_ack.feedback_identity.action

    @property
    def terminal_class(self) -> TerminalClass:
        self._require_carrier_consistent()
        return super(QualityAckControllerMessageV2, self).terminal_class

    @classmethod
    def from_raw_ack(cls, raw_ack: QualityAckV2) -> "QualityAckControllerMessageV2":
        if type(raw_ack) is not QualityAckV2:
            raise QualityAckV2Error("raw_ack must be an exact QualityAckV2")
        raw_ack.validate()
        return cls(
            identity=raw_ack.feedback_identity,
            terminal_status=raw_ack.terminal_status,
            raw_ack=raw_ack,
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self._require_carrier_consistent()
        return self.raw_ack.to_canonical_dict()

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def _make_reviewed_service_terminal_capability():
    sentinel = object()

    def issue() -> Tuple[object, str]:
        return sentinel, "reviewed-service-terminal-verifier-v2"

    def valid(value: Any) -> bool:
        return (
            type(value) is tuple
            and len(value) == 2
            and value[0] is sentinel
            and value[1] == "reviewed-service-terminal-verifier-v2"
        )

    return issue, valid


_issue_service_terminal_capability, _valid_service_terminal_capability = (
    _make_reviewed_service_terminal_capability()
)
_REVIEWED_SERVICE_TERMINAL_CAPABILITY_V2 = (
    _issue_service_terminal_capability()
)


def _make_service_terminal_attestation():
    sentinel = object()

    def issue(binding: str) -> Tuple[object, str]:
        return sentinel, binding

    def valid(value: Any, binding: str) -> bool:
        return (
            type(value) is tuple
            and len(value) == 2
            and value[0] is sentinel
            and value[1] == binding
        )

    return issue, valid


_issue_service_terminal, _valid_service_terminal = (
    _make_service_terminal_attestation()
)


@dataclass(frozen=True, slots=True)
class ServiceTerminalEvidenceV2:
    """Reviewed proof that the selected action failed before reward service.

    An opaque caller-selected digest is insufficient.  The verifier consumes a
    canonical terminal document naming the exact request, timed-out ticket,
    action-controlled stage and failure code, and then issues a field-bound
    attestation.  Remote cryptographic authentication remains out of scope.
    """

    completed_ticket: CompletedTicket
    request: DynamicFeatureRequestV2
    stage: ServiceFailureStage
    failure_code: str
    detected_at_ns: int
    terminal_document_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_shape()
        if self._attestation is not None and not self.is_verified:
            raise TimeoutReconciliationError(
                "service-terminal attestation does not bind its record"
            )

    def _validate_shape(self) -> None:
        if type(self.completed_ticket) is not CompletedTicket:
            raise TimeoutReconciliationError(
                "service evidence requires an exact CompletedTicket"
            )
        try:
            self.completed_ticket.require_lineage_attested()
        except RewardTicketControllerError as exc:
            raise TimeoutReconciliationError(
                "service evidence ticket lacks controller lineage attestation"
            ) from exc
        if self.completed_ticket.terminal_class is not TerminalClass.FEEDBACK_TIMEOUT:
            raise TimeoutReconciliationError(
                "post-run service evidence applies only to FEEDBACK_TIMEOUT"
            )
        if type(self.request) is not DynamicFeatureRequestV2:
            raise TimeoutReconciliationError(
                "service evidence requires an exact DynamicFeatureRequestV2"
            )
        self.request.validate()
        if type(self.stage) is not ServiceFailureStage:
            raise TimeoutReconciliationError("stage must be a ServiceFailureStage")
        _non_empty_str(self.failure_code, "failure_code")
        _non_negative_int(self.detected_at_ns, "detected_at_ns")
        _sha256_hex(self.terminal_document_sha256, "terminal_document_sha256")
        self._assert_request_ticket_identity()

    def _assert_request_ticket_identity(self) -> None:
        ticket = self.completed_ticket
        tx = self.request.transaction
        if (
            tx.session_uuid != ticket.session_uuid
            or tx.decision_seq != ticket.decision_seq
            or tx.tensor_seq != ticket.reward_tensor_seq
            or tx.carla_frame_id != ticket.reward_carla_frame_id
            or self.request.action != ticket.action
            or self.request.controller_lineage_uuid
            != ticket.controller_lineage_uuid
            or self.request.policy_decision_trace_sha256
            != ticket.policy_decision_trace_sha256
        ):
            raise TimeoutReconciliationError(
                "service-terminal request does not identify the exact original "
                "ticket/action/lineage/policy trace"
            )

    @classmethod
    def _verify_from_reviewed_terminal(
        cls,
        *,
        verifier_capability: object,
        completed_ticket: CompletedTicket,
        request: DynamicFeatureRequestV2,
        stage: ServiceFailureStage,
        failure_code: str,
        detected_at_ns: int,
        terminal_document: bytes,
    ) -> "ServiceTerminalEvidenceV2":
        if not _valid_service_terminal_capability(verifier_capability):
            raise TimeoutReconciliationError(
                "service-negative evidence requires the sealed reviewed "
                "service-terminal verifier capability"
            )
        if type(terminal_document) is not bytes or not terminal_document:
            raise TimeoutReconciliationError(
                "terminal_document must be non-empty immutable bytes"
            )
        provisional = cls(
            completed_ticket=completed_ticket,
            request=request,
            stage=stage,
            failure_code=failure_code,
            detected_at_ns=detected_at_ns,
            terminal_document_sha256=hashlib.sha256(terminal_document).hexdigest(),
        )
        expected_document = {
            "completed_ticket_sha256": completed_ticket.canonical_sha256(),
            "detected_at_ns": detected_at_ns,
            "failure_code": failure_code,
            "record": "service_terminal_evidence_v2",
            "request_sha256": request.canonical_sha256(),
            "stage": stage.value,
        }
        if terminal_document != canonical_json_bytes(expected_document):
            raise TimeoutReconciliationError(
                "service terminal document does not semantically bind the "
                "ticket/request/stage/failure"
            )
        return replace(
            provisional,
            _attestation=_issue_service_terminal(provisional._binding()),
        )

    @property
    def is_verified(self) -> bool:
        return _valid_service_terminal(self._attestation, self._binding())

    def require_verified(self) -> None:
        self._validate_shape()
        if not self.is_verified:
            raise TimeoutReconciliationError(
                "service terminal evidence was not issued by the reviewed verifier"
            )

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "completed_ticket_sha256": self.completed_ticket.canonical_sha256(),
            "detected_at_ns": self.detected_at_ns,
            "failure_code": self.failure_code,
            "record": "service_terminal_evidence_v2",
            "request_sha256": self.request.canonical_sha256(),
            "stage": self.stage.value,
            "terminal_document_sha256": self.terminal_document_sha256,
        }

    def _binding(self) -> str:
        return canonical_sha256(self._serialized_fields())

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_verified()
        return self._serialized_fields()

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# ---------------------------------------------------------------------------
# Conservative timeout reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeoutReconciliationV1:
    """Post-run adjudication of one controller-attested feedback timeout.

    Only a proven action-path service failure yields a policy-negative learning
    disposition.  Feedback-only loss and unresolved evidence are censored;
    infrastructure/evaluator failures are excluded.  A late exact ACK remains
    diagnostic and censored unless a future *versioned* policy changes that
    rule.  It can never be attached to a newer ticket.
    """

    completed_ticket: CompletedTicket
    verdict: TimeoutVerdict
    reconciler_source_sha256: str
    reconciled_at_wall_ns: int
    service_terminal_evidence: Optional[ServiceTerminalEvidenceV2] = None
    edge_quality_ack_sha256: Optional[str] = None
    edge_send_evidence_sha256: Optional[str] = None
    ue_receive_ledger_sha256: Optional[str] = None
    packet_capture_evidence_sha256: Optional[str] = None
    infrastructure_evidence_sha256: Optional[str] = None
    late_ack: Optional[QualityAckV2] = None
    late_received_ns: Optional[int] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.completed_ticket) is not CompletedTicket:
            raise TimeoutReconciliationError(
                "completed_ticket must be an exact CompletedTicket"
            )
        try:
            self.completed_ticket.require_lineage_attested()
        except RewardTicketControllerError as exc:
            raise TimeoutReconciliationError(
                "completed ticket lacks controller-issued lineage attestation"
            ) from exc
        if self.completed_ticket.terminal_class is not TerminalClass.FEEDBACK_TIMEOUT:
            raise TimeoutReconciliationError(
                "timeout reconciliation requires a FEEDBACK_TIMEOUT ticket"
            )
        if type(self.verdict) is not TimeoutVerdict:
            raise TimeoutReconciliationError(
                "verdict must be a supported TimeoutVerdict"
            )
        _sha256_hex(self.reconciler_source_sha256, "reconciler_source_sha256")
        _non_negative_int(self.reconciled_at_wall_ns, "reconciled_at_wall_ns")
        for field_name in (
            "edge_quality_ack_sha256",
            "edge_send_evidence_sha256",
            "ue_receive_ledger_sha256",
            "packet_capture_evidence_sha256",
            "infrastructure_evidence_sha256",
        ):
            _optional_sha256(getattr(self, field_name), field_name)
        if self.late_received_ns is not None:
            _non_negative_int(self.late_received_ns, "late_received_ns")

        if self.verdict is TimeoutVerdict.ACTION_PATH_SERVICE_FAILURE_NEGATIVE:
            if type(self.service_terminal_evidence) is not ServiceTerminalEvidenceV2:
                raise TimeoutReconciliationError(
                    "penalizing service verdict requires typed, reviewed "
                    "ServiceTerminalEvidenceV2; an enum plus opaque digest is "
                    "not sufficient"
                )
            self.service_terminal_evidence.require_verified()
            if (
                self.service_terminal_evidence.completed_ticket.canonical_sha256()
                != self.completed_ticket.canonical_sha256()
            ):
                raise TimeoutReconciliationError(
                    "service-terminal evidence belongs to a different ticket"
                )
            self._forbid_late_ack()
        elif self.verdict is TimeoutVerdict.FEEDBACK_ONLY_LOSS_CENSORED:
            self._forbid_service_terminal()
            self._require_digest(
                self.edge_quality_ack_sha256, "edge_quality_ack_sha256"
            )
            self._require_digest(
                self.edge_send_evidence_sha256, "edge_send_evidence_sha256"
            )
            self._forbid_late_ack()
        elif self.verdict is TimeoutVerdict.UE_CONTROL_INFRASTRUCTURE_FAULT_EXCLUDED:
            self._forbid_service_terminal()
            for value, field_name in (
                (self.edge_quality_ack_sha256, "edge_quality_ack_sha256"),
                (self.edge_send_evidence_sha256, "edge_send_evidence_sha256"),
                (self.ue_receive_ledger_sha256, "ue_receive_ledger_sha256"),
                (
                    self.packet_capture_evidence_sha256,
                    "packet_capture_evidence_sha256",
                ),
            ):
                self._require_digest(value, field_name)
            self._forbid_late_ack()
        elif self.verdict is TimeoutVerdict.EVALUATOR_INFRASTRUCTURE_FAULT_EXCLUDED:
            self._forbid_service_terminal()
            self._require_digest(
                self.infrastructure_evidence_sha256,
                "infrastructure_evidence_sha256",
            )
            self._forbid_late_ack()
        elif self.verdict is TimeoutVerdict.LATE_EXACT_QUALITY_CENSORED:
            self._forbid_service_terminal()
            self._validate_late_ack()
        elif self.verdict is TimeoutVerdict.UNRESOLVED_CENSORING:
            self._forbid_service_terminal()
            self._forbid_late_ack()
        else:  # pragma: no cover - exact Enum above makes this defensive only
            raise TimeoutReconciliationError(
                f"unsupported timeout verdict {self.verdict!r}"
            )

    @staticmethod
    def _require_digest(value: Optional[str], field_name: str) -> None:
        if value is None:
            raise TimeoutReconciliationError(
                f"timeout verdict requires {field_name}"
            )

    def _forbid_late_ack(self) -> None:
        if self.late_ack is not None or self.late_received_ns is not None:
            raise TimeoutReconciliationError(
                "late_ack/late_received_ns are legal only for "
                "LATE_EXACT_QUALITY_CENSORED"
            )

    def _forbid_service_terminal(self) -> None:
        if self.service_terminal_evidence is not None:
            raise TimeoutReconciliationError(
                "reviewed service-terminal evidence is legal only for "
                "ACTION_PATH_SERVICE_FAILURE_NEGATIVE"
            )

    def _validate_late_ack(self) -> None:
        if type(self.late_ack) is not QualityAckV2:
            raise TimeoutReconciliationError(
                "LATE_EXACT_QUALITY_CENSORED requires a QualityAckV2"
            )
        self.late_ack.validate()
        if self.late_ack.terminal_status is not FeedbackTerminalStatus.REWARD_FINAL:
            raise TimeoutReconciliationError(
                "late exact quality must be a REWARD_FINAL ACK"
            )
        if self.late_received_ns is None:
            raise TimeoutReconciliationError(
                "late exact quality requires the UE-local late_received_ns"
            )
        if self.late_received_ns <= self.completed_ticket.deadline_ns:
            raise TimeoutReconciliationError(
                "late_received_ns must be strictly after the original ticket "
                "deadline"
            )
        expected = RewardFeedbackIdentity(
            session_uuid=self.completed_ticket.session_uuid,
            decision_seq=self.completed_ticket.decision_seq,
            reward_tensor_seq=self.completed_ticket.reward_tensor_seq,
            carla_frame_id=self.completed_ticket.reward_carla_frame_id,
            action=self.completed_ticket.action,
        )
        if self.late_ack.feedback_identity.to_canonical_dict() != expected.to_canonical_dict():
            raise TimeoutReconciliationError(
                "late ACK does not belong to this exact timed-out ticket; late "
                "feedback may never be reused for another decision"
            )
        if (
            self.late_ack.request.controller_lineage_uuid
            != self.completed_ticket.controller_lineage_uuid
        ):
            raise TimeoutReconciliationError(
                "late ACK request belongs to a different controller lineage"
            )
        if (
            self.late_ack.request.policy_decision_trace_sha256
            != self.completed_ticket.policy_decision_trace_sha256
        ):
            raise TimeoutReconciliationError(
                "late ACK request does not bind the timed-out ticket's exact "
                "pre-execution policy-decision trace"
            )
        digest = self.late_ack.canonical_sha256()
        if self.edge_quality_ack_sha256 is not None and self.edge_quality_ack_sha256 != digest:
            raise TimeoutReconciliationError(
                "edge_quality_ack_sha256 disagrees with the supplied late ACK"
            )

    @property
    def learning_disposition(self) -> ReconciliationLearningDisposition:
        return _VERDICT_DISPOSITION[self.verdict]

    @property
    def penalizes_policy(self) -> bool:
        return (
            self.learning_disposition
            is ReconciliationLearningDisposition.
            INCLUDED_REGISTERED_NEGATIVE_SERVICE_REWARD
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = _schema_fields(
            TIMEOUT_RECONCILIATION_SCHEMA_ID,
            TIMEOUT_RECONCILIATION_SCHEMA_VERSION,
            TIMEOUT_RECONCILIATION_SCHEMA_SHA256,
        )
        payload.update(
            {
                "completed_ticket": self.completed_ticket.to_canonical_dict(),
                "completed_ticket_sha256": (
                    self.completed_ticket.canonical_sha256()
                ),
                "edge_quality_ack_sha256": (
                    self.late_ack.canonical_sha256()
                    if self.late_ack is not None
                    else self.edge_quality_ack_sha256
                ),
                "edge_send_evidence_sha256": self.edge_send_evidence_sha256,
                "service_terminal_evidence": (
                    None
                    if self.service_terminal_evidence is None
                    else self.service_terminal_evidence.to_canonical_dict()
                ),
                "infrastructure_evidence_sha256": (
                    self.infrastructure_evidence_sha256
                ),
                "late_ack": (
                    None if self.late_ack is None else self.late_ack.to_canonical_dict()
                ),
                "late_received_ns": self.late_received_ns,
                "learning_disposition": self.learning_disposition.value,
                "packet_capture_evidence_sha256": (
                    self.packet_capture_evidence_sha256
                ),
                "penalizes_policy": self.penalizes_policy,
                "reconciled_at_wall_ns": self.reconciled_at_wall_ns,
                "reconciler_source_sha256": self.reconciler_source_sha256,
                "record": "timeout_reconciliation_v1",
                "ue_receive_ledger_sha256": self.ue_receive_ledger_sha256,
                "verdict": self.verdict.value,
            }
        )
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())
