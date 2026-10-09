"""Versioned causal-state, reward-measurement and replay-transition contract.

Phase 4a (repaired through 4a.2) of the SplitFusion conditional Hybrid-SAC
foundation.  Pure, deterministic, in-memory contract code over the four frozen
dependencies.  It defines *what a transition is* and *how its reward is
measured*; it does not learn, store, sample, simulate or communicate.

Deliberately **not** implemented: Gym/Gymnasium environment, simulator or
interpolator, replay-buffer storage, actor/critic/SAC networks, recurrent
memory, optimizer or training loop, UDP/OAI/CARLA/Docker/CUDA integration, live
runtime, protocol-v2 wire transport, and the ``LOCAL``/``SKIP`` top-level modes.
The action is exactly ``(joint mode among 12, continuous q)`` and the initial
policy is feed-forward.

Forgery resistance
------------------
Every derived record -- including eligibility/quality, episode-start and
previous-outcome records, policy features/traces, latency/outcome and replay --
carries a private construction attestation bound to its serialized fields.  The
attestation is issued only by the validating factory that recomputed the value
from frozen sources, and it is refused when any field differs.  A directly
constructed or mutated record is therefore **unattested**: it cannot serialize
or enter a transition.  This guards against mistaken construction, not a
determined attacker; Python offers no true privacy.

:meth:`ReplayTransitionV1.revalidate` additionally recomputes the outcome and
both policy feature vectors from their full frozen sources, so a stored
transition can be re-proved rather than trusted.

Ground-truth boundary
---------------------
The registered quality formula is defined over ``CARLA_GT_EXACT``: a
**privileged, non-deployable** oracle that may exist only inside the
training/testbed instrument.  Phase 4a.2 does not yet have an authenticated
producer for those per-frame inputs.  Its v2 support path is therefore a
self-consistent formula fixture and is rejected from learning/replay.  Because
the eventual causal state would carry the *previous* decision's exact quality,
the whole policy observation schema is marked
``POLICY_OBSERVATION_DEPLOYABILITY = "SIMULATOR_TESTBED_ONLY"``.  Nothing here
implies physical deployability or that the current fixture is ground truth.

Quality-ACK binding: anchor-only, document-verified, never snapped
-----------------------------------------------------------------
``sf_priv_quality_ack.v1`` identifies a decision by ``action_id`` *and*
``profile_id`` and validates ``0 <= action_id < 72``, so it can only describe
one of the 72 **registered anchors**.  Hybrid SAC emits arbitrary continuous
``q``, whose identity legitimately carries ``action_id=None``.

Therefore:

* :class:`QualityAckObligationV1` binds the transport identity, frame identity
  and complete executed action.  In the current contract it is a caller-built
  identity assertion, not an authenticated proof that the commitment existed
  before ACK arrival; this is another reason exact positive rewards remain
  blocked pending the reviewed producer/protocol-v2 path.
* :meth:`QualityAckBindingV1.from_ack_document` takes the **presented raw ACK
  mapping**, calls the *real* protocol validator, computes the raw ACK
  SHA-256 itself, extracts ``dh``, retains all seven v1 identity fields, and
  cross-checks the ACK against both the obligation and the
  :class:`~.reward_ticket_controller.CompletedTicket`.  This proves content
  consistency, not receipt/storage provenance; a caller-supplied opaque hash
  is never accepted as proof of anything.
* An off-anchor action raises :class:`OffAnchorQualityAckError` and can **not**
  produce a learning-eligible exact-quality transition until an
  identity-bearing protocol-v2 carrier exists.  It is never snapped to a
  nearest anchor.
* Evidence carries an explicit :class:`EvidenceKind` and
  :class:`EvidenceGranularity`, so aggregate 72-action campaign evidence can
  never masquerade as a per-frame causal reward.

Detection-metric boundary
-------------------------
The v1 ACK carries ``tp``/``fn`` but **no false positives**.  Recall is
derivable; precision, F1 and average precision are not, and nothing here
derives or implies them.

Feature status
--------------
``camera_si`` and ``radar_p40`` are **candidate** state features, retained for
being cheap, causal and measurable -- not because they are established
predictors.  Their predictive value still requires the registered ablation and
SHAP validation.

Importing this module performs no filesystem, network, CUDA, CARLA or OAI
access.  The real quality-protocol module is loaded **lazily**, on first ACK
binding, because it resolves its own dependency through an absolute
``rl_agent.*`` import rooted outside this package.
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .action_contract import (
    CATALOG_SCHEMA,
    CATALOG_SHA256,
    EXECUTION_MODE,
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
    round_half_up_q_e4,
)
from .scene_descriptors import (
    P40_HORIZON_M,
    SCHEMA_ID as SCENE_SCHEMA_ID,
    SCHEMA_SHA256 as SCENE_SCHEMA_SHA256,
    SCHEMA_VERSION as SCENE_SCHEMA_VERSION,
    SceneDescriptorSample,
)
from .transaction_identity import (
    ACTION_IDENTITY_SCHEMA_ID,
    ACTION_IDENTITY_SCHEMA_SHA256,
    ActionHoldManifest,
    ExecutedActionIdentity,
    MINIMUM_HOLD_TENSORS,
    SCHEMA_ID as TRANSACTION_SCHEMA_ID,
    SCHEMA_SHA256 as TRANSACTION_SCHEMA_SHA256,
    SCHEMA_VERSION as TRANSACTION_SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from .reward_ticket_controller import (
    B_REWARD_DEADLINE_NS,
    CONTROLLER_SCHEMA_ID,
    CONTROLLER_SCHEMA_SHA256,
    CONTROLLER_SCHEMA_VERSION,
    CompletedTicket,
    ControllerGenesisProof,
    K_MIN_TENSORS,
    RewardTicketControllerError,
    TERMINAL_LEARNING_DISPOSITION,
    TerminalClass,
)

__all__ = [
    # errors
    "StateRewardContractError",
    "CausalStateError",
    "StaleTelemetryError",
    "ClockDomainError",
    "NormalizationSpecError",
    "QualityContractError",
    "OffAnchorQualityAckError",
    "UndefinedClassSupportError",
    "InsufficientQualitySupportError",
    "EvidenceGranularityError",
    "QualityProducerUnavailableError",
    "QualityProtocolBindingError",
    "QualityAckReuseError",
    "RewardSpecError",
    "AdjudicationError",
    "TransitionIdentityError",
    "UnattestedRecordError",
    # schema
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_DESCRIPTOR",
    "SCHEMA_SHA256",
    "POLICY_FEATURE_ORDER",
    "POLICY_FEATURE_COUNT",
    "PREVIOUS_TERMINAL_ORDER",
    "PREVIOUS_TERMINAL_FEATURE_CODES",
    "FORBIDDEN_POLICY_FEATURE_SUBSTRINGS",
    "POLICY_OBSERVATION_DEPLOYABILITY",
    "assert_policy_features_exclude_forbidden_fields",
    # quality-protocol binding
    "QUALITY_PROTOCOL_CONTRACT",
    "QUALITY_PROTOCOL_CONTRACT_SHA256",
    "QUALITY_ACK_SCHEMA",
    "QUALITY_ACK_FAILURE_SCHEMA",
    "QUALITY_ACK_PROTOCOL_VERSION",
    "QUALITY_ACK_SOURCE",
    "QUALITY_ACK_IDENTITY_FIELDS",
    "QUALITY_ACK_TIMING_FIELDS",
    "QUALITY_ACK_QUALITY_FIELDS",
    "QUALITY_ACK_ANCHOR_ACTION_COUNT",
    "QUALITY_ACK_IS_ANCHOR_ONLY",
    "QUALITY_OBLIGATION_PRECOMMIT_AUTHENTICATED",
    "QUALITY_ACK_TIMING_CLOCK_DOMAIN",
    "QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE",
    "QUALITY_ACK_DERIVABLE_DETECTION_METRICS",
    "QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS",
    "QUALITY_DETAIL_SCHEMA",
    "QUALITY_DETAIL_SUPPORT_KEY",
    "QUALITY_DETAIL_SUPPORT_SCHEMA",
    "QUALITY_DETAIL_SUPPORT_FIELDS",
    "PROTOCOL_V2_REQUIREMENT",
    "verify_quality_protocol_binding",
    # clocks / latency
    "ClockDomain",
    "REWARD_LATENCY_CLOCK_DOMAIN",
    "FORBIDDEN_REWARD_LATENCY_SOURCES",
    "OPTIMIZATION_CONTRACT_COSTS",
    "DIAGNOSTIC_ONLY_SIGNALS",
    # candidate features
    "CANDIDATE_STATE_FEATURES",
    "CANDIDATE_FEATURE_VALIDATION_REQUIRED",
    # A: evidence
    "EvidenceKind",
    "EvidenceGranularity",
    "GroundTruthSource",
    "QualityProducerStatus",
    "EvaluationEligibilityV1",
    "EvaluationEligibilityResultV1",
    "QualityAckUseRegistryV1",
    "QualityAckObligationV1",
    "QualityAckBindingV1",
    "QualityEvidenceV1",
    # B: quality
    "QualityComponentsV1",
    "LocalizationCombiner",
    "QualityEvaluationV1",
    # C: reward
    "RewardSpecV1",
    "LatencyMeasurementV1",
    "ConstraintCostsV1",
    "DiagnosticSignalsV1",
    "SwitchPenaltyV1",
    # D: state
    "SnrMetric",
    "LinkDirection",
    "BsrScope",
    "RadioEvidencePath",
    "RadioSourceWall",
    "SnrSource",
    "McsSource",
    "BsrSource",
    "BsrReportType",
    "RadioMissingReason",
    "RadioFillPolicy",
    "RadioEventProvenanceV1",
    "RadioPolicyAvailabilityV1",
    "BsrReportV1",
    "SceneObservationV1",
    "RadioObservationV1",
    "RadioDiagnosticObservationV1",
    "EpisodeStartProofV1",
    "PreviousOutcomeV1",
    "CausalStateV1",
    "StateNormalizationSpecV1",
    "StateFreshnessPolicyV1",
    "PolicyFeatureVectorV1",
    "build_policy_features",
    # E: outcome / adjudication
    "Adjudication",
    "AdjudicationRecordV1",
    "LearningEligibility",
    "DecisionOutcomeV1",
    "evaluate_completed_decision",
    # F: provenance / transition
    "PolicyDecisionTraceV1",
    "ReplayTransitionV1",
    "build_replay_transition",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class StateRewardContractError(ValueError):
    """Base class for every Phase-4a contract violation."""


class CausalStateError(StateRewardContractError):
    """A causal-state field is missing, malformed, out of range or acausal."""


class StaleTelemetryError(CausalStateError):
    """A required observation is older than its registered freshness bound.

    A runtime condition for the external registered-fallback guard, not a
    numeric value: nothing is substituted for the stale measurement.
    """


class ClockDomainError(CausalStateError):
    """Two timestamps were combined across different clock domains."""


class NormalizationSpecError(StateRewardContractError):
    """A normalization or freshness specification lacks explicit provenance."""


class QualityContractError(StateRewardContractError):
    """A quality component is malformed or contradicts its validity mask."""


class OffAnchorQualityAckError(QualityContractError):
    """An off-anchor continuous ``q`` was offered to the anchor-only v1 ACK.

    ``sf_priv_quality_ack.v1`` requires ``action_id`` and ``profile_id`` and
    validates ``0 <= action_id < 72``, so it can only describe one of the 72
    registered anchors.  An arbitrary continuous ``q`` has no such identity.
    Raised instead of snapping to a nearest anchor -- which would attribute one
    action's measured quality to a different action -- and instead of
    fabricating an identity the wire format cannot carry.  Off-anchor actions
    remain un-evaluable for exact per-frame quality until the protocol-v2
    carrier described by :data:`PROTOCOL_V2_REQUIREMENT` exists.
    """


class UndefinedClassSupportError(QualityContractError):
    """A component claims validity without the ground-truth support it needs."""


class InsufficientQualitySupportError(QualityContractError):
    """Too few valid components exist to compute the registered quality."""


class EvidenceGranularityError(QualityContractError):
    """Aggregate evidence was offered where per-frame causal evidence is required."""


class QualityProducerUnavailableError(QualityContractError):
    """A self-consistent fixture lacks source-authenticated producer evidence.

    Hashes over caller-supplied summaries prove only that those summaries were
    not altered after hashing.  They do not prove that actor eligibility,
    masks, matches or errors came from CARLA and the registered evaluator.
    Until the producer derives and manifests them from the frozen raw source
    artifacts, the record may test quality mathematics but may not enter
    learning or replay.
    """


class QualityProtocolBindingError(QualityContractError):
    """The real quality-protocol module disagrees with the declared binding."""


class QualityAckReuseError(QualityContractError):
    """One v1 wire ACK was offered for a different decision binding.

    Protocol v1 does not carry ``session_uuid``/``decision_seq``/
    ``reward_tensor_seq`` or the complete executed-action identity.  A
    process-scoped registry is therefore required as a fail-closed bridge: a
    digest may be revalidated idempotently for the *same* obligation/ticket,
    but can never be rebound to a different one.  Protocol v2 remains the
    durable, cross-process solution.
    """


class RewardSpecError(StateRewardContractError):
    """A reward-specification value is missing or outside its declared domain."""


class AdjudicationError(StateRewardContractError):
    """An adjudication record is absent, misbound, reused or inconsistent."""


class TransitionIdentityError(StateRewardContractError):
    """A replay transition violates an exact-identity or causality invariant."""


class UnattestedRecordError(StateRewardContractError):
    """A derived record was not produced by its validating factory.

    Raised when a record that must be derived -- a quality evaluation, latency
    measurement, ACK binding, previous-outcome summary, decision outcome or
    replay transition -- is used or serialized without an attestation bound to
    its own serialized fields.  Build these through their factories; a directly
    constructed or mutated copy is refused.
    """


# --------------------------------------------------------------------------- #
# Local helpers: fail closed, never normalize
# --------------------------------------------------------------------------- #


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze a literal into read-only mappings and tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _exact_int(value: Any, name: str, error: type = CausalStateError) -> int:
    """Validate an exact Python ``int``; ``bool`` is rejected as a distinct type."""
    if isinstance(value, bool):
        raise error(f"{name} must be an int, not a bool: {value!r}")
    if type(value) is not int:
        raise error(
            f"{name} must be an exact int, got {type(value).__name__}: {value!r}"
        )
    return value


def _non_negative_int(value: Any, name: str, error: type = CausalStateError) -> int:
    _exact_int(value, name, error)
    if value < 0:
        raise error(f"{name} must be >= 0, got {value}")
    return value


def _positive_int(value: Any, name: str, error: type = CausalStateError) -> int:
    _exact_int(value, name, error)
    if value <= 0:
        raise error(f"{name} must be > 0, got {value}")
    return value


def _finite_float(value: Any, name: str, error: type = CausalStateError) -> float:
    """Validate a finite real scalar; ``bool`` and non-numerics are rejected."""
    if isinstance(value, bool):
        raise error(f"{name} must be a real scalar, not a bool: {value!r}")
    if not isinstance(value, (int, float)):
        raise error(
            f"{name} must be a finite real scalar, got "
            f"{type(value).__name__}: {value!r}"
        )
    as_float = float(value)
    if not math.isfinite(as_float):
        raise error(f"{name} must be finite, got {value!r}")
    return as_float


def _finite_in(
    value: Any,
    name: str,
    low: float,
    high: float,
    error: type = CausalStateError,
) -> float:
    as_float = _finite_float(value, name, error)
    if not low <= as_float <= high:
        raise error(f"{name} must lie in [{low}, {high}], got {as_float!r}")
    return as_float


def _exact_bool(value: Any, name: str, error: type = CausalStateError) -> bool:
    if type(value) is not bool:
        raise error(
            f"{name} must be a bool, got {type(value).__name__}: {value!r}"
        )
    return value


def _non_empty_str(value: Any, name: str, error: type = CausalStateError) -> str:
    if not isinstance(value, str) or value == "":
        raise error(
            f"{name} must be a non-empty str, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _sha256_hex(value: Any, name: str, error: type = CausalStateError) -> str:
    if not isinstance(value, str):
        raise error(
            f"{name} must be a str digest, got {type(value).__name__}: {value!r}"
        )
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise error(f"{name} must be 64 lowercase hex characters, got {value!r}")
    return value


def _canonical_uuid(value: Any, error: type = CausalStateError) -> str:
    if not isinstance(value, str):
        raise error(
            f"session_uuid must be a str, got {type(value).__name__}: {value!r}"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise error(f"session_uuid is not a parsable UUID: {value!r}") from exc
    if str(parsed) != value:
        raise error(
            f"session_uuid must be the canonical lowercase hyphenated form "
            f"{str(parsed)!r}, got {value!r}"
        )
    return value


def _frozen_str_mapping(
    value: Any,
    name: str,
    error: type = StateRewardContractError,
) -> Mapping[str, str]:
    """Validate and freeze a non-empty ``str -> str`` provenance mapping."""
    if not isinstance(value, Mapping) or not value:
        raise error(
            f"{name} must be a non-empty mapping of str to str, got "
            f"{type(value).__name__}: {value!r}"
        )
    for key, item in value.items():
        _non_empty_str(key, f"{name} key", error)
        _non_empty_str(item, f"{name}[{key!r}]", error)
    return MappingProxyType(dict(value))


# --------------------------------------------------------------------------- #
# Construction attestations: only a validating factory may issue one
# --------------------------------------------------------------------------- #


def _make_attestation_gate(label: str) -> Tuple[Callable, Callable]:
    """Build an issuer/checker pair over a closure-held sentinel.

    An attestation is ``(sentinel, binding)`` where ``binding`` is the canonical
    SHA-256 of every serialized field of the record it was issued for.  It is
    therefore not transferable: copying it onto a record whose serialized fields
    differ -- by :func:`dataclasses.replace` or by lifting another record's
    token -- leaves the recomputed binding mismatched and the attestation is
    refused.  The sentinel never becomes a module attribute, so an attestation
    cannot be produced by ordinary construction.  This is a guard against
    mistaken construction, not a security boundary.
    """
    sentinel = object()

    def issue(binding: str) -> Tuple[Any, str]:
        return (sentinel, binding)

    def is_valid(token: Any, binding: str) -> bool:
        return (
            type(token) is tuple
            and len(token) == 2
            and token[0] is sentinel
            and token[1] == binding
        )

    return issue, is_valid


_issue_ack, _valid_ack = _make_attestation_gate("quality_ack_binding")
_issue_eligibility_result, _valid_eligibility_result = _make_attestation_gate(
    "evaluation_eligibility_result"
)
_issue_components, _valid_components = _make_attestation_gate(
    "quality_components"
)
_issue_quality, _valid_quality = _make_attestation_gate("quality_evaluation")
_issue_latency, _valid_latency = _make_attestation_gate("latency_measurement")
_issue_episode_start, _valid_episode_start = _make_attestation_gate(
    "episode_start_proof"
)
_issue_previous, _valid_previous = _make_attestation_gate("previous_outcome")
_issue_radio, _valid_radio = _make_attestation_gate("radio_observation")
_issue_features, _valid_features = _make_attestation_gate("policy_feature_vector")
_issue_policy_trace, _valid_policy_trace = _make_attestation_gate(
    "policy_decision_trace"
)
_issue_outcome, _valid_outcome = _make_attestation_gate("decision_outcome")
_issue_transition, _valid_transition = _make_attestation_gate("replay_transition")


class _Attested:
    """Mixin giving a frozen dataclass an unforgeable derivation attestation."""

    __slots__ = ()

    def _serialized_fields(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _binding(self) -> str:
        """The value an attestation is bound to: a hash of every serialized field."""
        return canonical_sha256(self._serialized_fields())

    @property
    def _checker(self) -> Callable:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def is_attested(self) -> bool:
        """True when this record carries an attestation bound to its own fields."""
        return self._checker(
            getattr(self, "_attestation", None), self._binding()
        )

    def require_attested(self) -> None:
        """Fail closed unless a validating factory produced this exact record."""
        if not self.is_attested:
            raise UnattestedRecordError(
                f"{type(self).__name__} carries no construction attestation "
                f"bound to its own serialized fields, so it was not produced by "
                f"its validating factory (or was mutated afterwards).  Derived "
                f"records must be built through their factory so their values "
                f"are recomputed from frozen sources rather than asserted"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping; fails closed on an unattested record."""
        self.require_attested()
        return self._serialized_fields()

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Declared binding to the privileged quality-ACK wire contract
# --------------------------------------------------------------------------- #
#
# These mirror rl_agent/splitfusion_quality_feedback_probe_v1/protocol.py.  They
# are declared literals so this module's *import* stays side-effect free: that
# module resolves its own dependency through an absolute ``rl_agent.*`` import
# rooted outside this package, so importing it eagerly would require mutating
# ``sys.path`` at import time.  The real module is loaded lazily on first ACK
# binding and every literal below is then verified against it by
# :func:`verify_quality_protocol_binding`, which raises on any drift.

QUALITY_ACK_SCHEMA: str = "sf_priv_quality_ack.v1"
QUALITY_ACK_FAILURE_SCHEMA: str = "sf_priv_quality_fail.v1"
QUALITY_ACK_PROTOCOL_VERSION: int = 1
QUALITY_ACK_SOURCE: str = "EDGE_CARLA_GT"
QUALITY_ACK_MAX_WIRE_BYTES: int = 1200

#: All seven v1 ACK identity fields, retained in full by every binding.
QUALITY_ACK_IDENTITY_FIELDS: Tuple[str, ...] = (
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "action_id",
    "profile_id",
    "capture_timestamp_ns",
)

#: Every ACK timing field is a *wall* clock reading; none may be used for
#: reward latency.  See :data:`REWARD_LATENCY_CLOCK_DOMAIN`.
QUALITY_ACK_TIMING_FIELDS: Tuple[str, ...] = (
    "model_ready_wall_ns",
    "final_prediction_ready_wall_ns",
    "gt_ready_wall_ns",
    "evaluation_enqueued_wall_ns",
    "evaluation_started_wall_ns",
    "evaluation_completed_wall_ns",
    "ack_emit_start_wall_ns",
)

QUALITY_ACK_QUALITY_FIELDS: Tuple[str, ...] = (
    "seg_vehicle_iou",
    "seg_person_iou",
    "seg_miou_3class",
    "vehicle_recall",
    "vehicle_xy_error_m",
    "vehicle_footprint_iou",
    "person_recall",
    "person_xy_error_m",
    "person_footprint_iou",
    "gt_vehicle_pixels",
    "gt_person_pixels",
    "vehicle_tp",
    "vehicle_fn",
    "person_tp",
    "person_fn",
)

#: The v1 ACK requires these anchor fields and validates ``0 <= action_id < 72``.
QUALITY_ACK_REQUIRED_ANCHOR_FIELDS: Tuple[str, ...] = ("action_id", "profile_id")
QUALITY_ACK_ANCHOR_ACTION_COUNT: int = 72
QUALITY_ACK_IS_ANCHOR_ONLY: bool = True
QUALITY_OBLIGATION_PRECOMMIT_AUTHENTICATED: bool = False
QUALITY_ACK_TIMING_CLOCK_DOMAIN: str = "WALL"

#: The v1 ACK carries tp and fn but no false positives.
QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE: bool = False
QUALITY_ACK_DERIVABLE_DETECTION_METRICS: Tuple[str, ...] = ("recall",)
QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS: Tuple[str, ...] = (
    "precision",
    "f1",
    "average_precision",
    "comprehensive_detection_quality",
)

#: Fields the v1 ACK does NOT carry but the quality formulation needs.  They
#: must come from the edge-retained detailed evidence row, which the ACK
#: hash-binds through ``dh``.  Recorded so the gap is explicit: a predicted-mask
#: pixel count is what distinguishes an excluded empty-empty segmentation class
#: from a penalized false-positive one, and the eligible-GT counts are what
#: stop mere GT presence from creating a miss penalty.
QUALITY_ACK_MISSING_REQUIRED_FIELDS: Tuple[str, ...] = (
    "pred_vehicle_pixels",
    "pred_person_pixels",
    "vehicle_eligible_gt_instances",
    "person_eligible_gt_instances",
    "evaluation_eligibility_contract_sha256",
)

#: The existing v1 detail schema is retained on the edge, but historical/live
#: rows do not contain enough information to prove the per-UE eligibility mask
#: or even distinguish an empty predicted mask from an absent class.  Phase
#: 4a.2 therefore requires this versioned, hash-bound extension *inside the
#: actual detail document*.  Legacy detail rows without it fail closed; no
#: counts are supplied out-of-band.
QUALITY_DETAIL_SCHEMA: str = "splitfusion_privileged_quality_detail.v1"
QUALITY_DETAIL_SUPPORT_KEY: str = "phase4a2_reward_support"
QUALITY_DETAIL_SUPPORT_SCHEMA: str = (
    "splitfusion_phase4a2_reward_support.v2"
)
QUALITY_DETAIL_SUPPORT_FIELDS: Tuple[str, ...] = (
    "schema",
    "ue_id",
    "session_uuid",
    "decision_seq",
    "reward_tensor_seq",
    "executed_action_sha256",
    "frame_id",
    "capture_timestamp_ns",
    "eligibility_contract_sha256",
    "gt_actor_rows",
    "gt_actor_snapshot_sha256",
    "segmentation_depth_m",
    "segmentation_gt_vehicle_indices",
    "segmentation_gt_person_indices",
    "segmentation_pred_vehicle_indices",
    "segmentation_pred_person_indices",
    "gt_segmentation_label_sha256",
    "prediction_segmentation_label_sha256",
    "segmentation_eligibility_mask_sha256",
    "eligible_vehicle_actor_ids",
    "eligible_person_actor_ids",
    "eligible_vehicle_actor_ids_sha256",
    "eligible_person_actor_ids_sha256",
    "vehicle_localization_matches",
    "person_localization_matches",
    "unmatched_vehicle_gt_actor_ids",
    "unmatched_person_gt_actor_ids",
    "unmatched_vehicle_prediction_indices",
    "unmatched_person_prediction_indices",
    "seg_vehicle_gt_pixels",
    "seg_person_gt_pixels",
    "seg_vehicle_pred_pixels",
    "seg_person_pred_pixels",
    "seg_vehicle_intersection_pixels",
    "seg_person_intersection_pixels",
    "seg_vehicle_union_pixels",
    "seg_person_union_pixels",
)

# Exact semantics hashed into every eligibility specification.  These values
# are intentionally explicit rather than hidden inside a producer: changing
# any one of them creates a different contract hash and therefore a different
# learning population.
ELIGIBILITY_VISIBILITY_ALGORITHM_VERSION: str = (
    "phase4a2_actor_geometry_visibility_v1"
)
ELIGIBILITY_VISIBILITY_THRESHOLD: float = 0.65
ELIGIBILITY_MIN_PROJECTED_SUPPORT_PIXELS: int = 1
SEGMENTATION_DOMAIN_RULE: str = (
    "CAMERA_IMAGE_INTERSECT_FINITE_POSITIVE_DEPTH_LE_MAX_RANGE_V1"
)

#: The complete declared quality-protocol contract, hashed into this schema.
QUALITY_PROTOCOL_CONTRACT: Mapping[str, Any] = _deep_freeze(
    {
        "module": (
            "rl_agent.splitfusion_quality_feedback_probe_v1.protocol"
        ),
        "evaluated_ack_schema": QUALITY_ACK_SCHEMA,
        "failed_ack_schema": QUALITY_ACK_FAILURE_SCHEMA,
        "protocol_version": QUALITY_ACK_PROTOCOL_VERSION,
        "source": QUALITY_ACK_SOURCE,
        "max_wire_bytes": QUALITY_ACK_MAX_WIRE_BYTES,
        "identity_fields": QUALITY_ACK_IDENTITY_FIELDS,
        "timing_fields": QUALITY_ACK_TIMING_FIELDS,
        "timing_clock_domain": QUALITY_ACK_TIMING_CLOCK_DOMAIN,
        "quality_fields": QUALITY_ACK_QUALITY_FIELDS,
        "required_anchor_fields": QUALITY_ACK_REQUIRED_ANCHOR_FIELDS,
        "anchor_action_count": QUALITY_ACK_ANCHOR_ACTION_COUNT,
        "anchor_only": QUALITY_ACK_IS_ANCHOR_ONLY,
        "privileged_flag": "pg=True",
        "deployable_flag": "dp=False",
        "detail_digest_field": "dh",
        "false_positive_counts_available": (
            QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE
        ),
        "derivable_detection_metrics": QUALITY_ACK_DERIVABLE_DETECTION_METRICS,
        "underivable_detection_metrics": (
            QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS
        ),
        "missing_required_fields": QUALITY_ACK_MISSING_REQUIRED_FIELDS,
    }
)

QUALITY_PROTOCOL_CONTRACT_SHA256: str = canonical_sha256(QUALITY_PROTOCOL_CONTRACT)

PROTOCOL_V2_REQUIREMENT: str = (
    "a protocol-v2 quality ACK must carry the full ExecutedActionIdentity or "
    "its canonical SHA-256, because an arbitrary continuous q has no action_id "
    "or profile_id and must never be snapped to a nearest anchor; it must also "
    "bind the raw quality-ACK digest into the controller-accepted feedback and "
    "authenticate when the quality outcome became policy-visible. Until that "
    "carrier and the reviewed source producer exist, no exact-positive quality "
    "outcome may enter learning/replay and an off-anchor action is not scored"
)

_PROTOCOL_CACHE: Dict[str, Any] = {}


def _load_quality_protocol() -> Any:
    """Lazily import the real quality-protocol module, once.

    The import is deferred out of module import so importing this contract has
    no filesystem side effect.  ``protocol.py`` resolves its own dependency
    through an absolute ``rl_agent.*`` import, so this inserts the package root
    on ``sys.path`` for the duration of the import if it is not already there.
    """
    module = _PROTOCOL_CACHE.get("module")
    if module is not None:
        return module
    import importlib
    import sys
    from pathlib import Path

    package_root = str(Path(__file__).resolve().parents[2])
    inserted = False
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
        inserted = True
    try:
        module = importlib.import_module(
            "rl_agent.splitfusion_quality_feedback_probe_v1.protocol"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise QualityProtocolBindingError(
            f"the real quality-protocol module could not be loaded, so no ACK "
            f"can be verified against it: {exc!r}"
        ) from exc
    finally:
        if inserted and package_root in sys.path:
            sys.path.remove(package_root)
    _PROTOCOL_CACHE["module"] = module
    return module


def verify_quality_protocol_binding() -> Mapping[str, Any]:
    """Assert the real protocol module still matches every declared literal.

    Called on the first ACK binding, so no ACK can be accepted under a drifted
    protocol, and callable directly by a test.

    Raises:
        QualityProtocolBindingError: on any disagreement.
    """
    if _PROTOCOL_CACHE.get("verified"):
        return QUALITY_PROTOCOL_CONTRACT
    module = _load_quality_protocol()
    expectations = (
        ("QUALITY_EVALUATED_ACK_SCHEMA", QUALITY_ACK_SCHEMA),
        ("QUALITY_EVALUATION_FAILED_ACK_SCHEMA", QUALITY_ACK_FAILURE_SCHEMA),
        ("PROTOCOL_VERSION", QUALITY_ACK_PROTOCOL_VERSION),
        ("SOURCE", QUALITY_ACK_SOURCE),
        ("MAX_WIRE_BYTES", QUALITY_ACK_MAX_WIRE_BYTES),
        ("IDENTITY_FIELDS", QUALITY_ACK_IDENTITY_FIELDS),
        ("TIMING_FIELDS", QUALITY_ACK_TIMING_FIELDS),
        ("QUALITY_FIELDS", QUALITY_ACK_QUALITY_FIELDS),
    )
    for name, expected in expectations:
        actual = getattr(module, name, None)
        if isinstance(expected, tuple):
            actual = tuple(actual) if actual is not None else None
        if actual != expected:
            raise QualityProtocolBindingError(
                f"quality-protocol drift: {name} is {actual!r} in the real "
                f"module but this contract is bound to {expected!r}.  The "
                f"Phase-4a schema hash {SCHEMA_SHA256} covers the declared "
                f"binding, so the two must be reconciled deliberately"
            )
    for required in ("validate", "digest", "identity_dict", "quality_dict"):
        if not callable(getattr(module, required, None)):
            raise QualityProtocolBindingError(
                f"the real quality-protocol module has no callable "
                f"{required!r}, so an ACK document cannot be verified"
            )
    if not all(
        field.endswith("_wall_ns") for field in module.TIMING_FIELDS
    ):  # pragma: no cover - guarded by the literal comparison above
        raise QualityProtocolBindingError(
            "an ACK timing field is no longer a wall-clock reading"
        )
    _PROTOCOL_CACHE["verified"] = True
    return QUALITY_PROTOCOL_CONTRACT


# --------------------------------------------------------------------------- #
# Clock domain, latency and cost contracts
# --------------------------------------------------------------------------- #


class ClockDomain(Enum):
    """The single clock domain every Phase-4a timestamp lives in.

    One domain, declared once: the UE-local monotonic source the reward-ticket
    controller is driven by.  Mixing a wall clock into any derived duration is a
    contract violation, not a rounding concern.
    """

    UE_LOCAL_MONOTONIC = "UE_LOCAL_MONOTONIC"


REWARD_LATENCY_CLOCK_DOMAIN: str = ClockDomain.UE_LOCAL_MONOTONIC.value

FORBIDDEN_REWARD_LATENCY_SOURCES: Tuple[str, ...] = (
    "map_install_ack_latency",
    "quality_ack_wall_timings",
    "wall_clock",
    "mixed_wall_and_monotonic",
)

#: The costs that are part of the optimization contract.
OPTIMIZATION_CONTRACT_COSTS: Tuple[str, ...] = (
    "c_deadline",
    "c_authoritative_failure",
)

#: Signals retained for diagnosis only and explicitly **outside** the
#: optimization contract.  ``c_latency_excess`` is here because the controller
#: refuses feedback after the deadline: on every feedback-resolved ticket
#: ``normalized_latency <= 1``, so the excess is identically zero and carries no
#: online information, and on a censored timeout it is unmeasurable without a
#: receipt.  It is retained as an honest diagnostic rather than presented as a
#: useful constraint.
DIAGNOSTIC_ONLY_SIGNALS: Tuple[str, ...] = ("c_latency_excess",)

#: SI and P40 are candidates pending ablation/SHAP validation.
CANDIDATE_STATE_FEATURES: Tuple[str, ...] = ("camera_si", "radar_p40")
CANDIDATE_FEATURE_VALIDATION_REQUIRED: str = (
    "camera_si and radar_p40 are candidate state features retained for being "
    "cheap, causal and measurable; their predictive value still requires the "
    "registered ablation and SHAP validation and is NOT established here"
)

#: The causal state carries the previous decision's exact CARLA ground-truth
#: quality, which no deployed system can observe.
POLICY_OBSERVATION_DEPLOYABILITY: str = "SIMULATOR_TESTBED_ONLY"


# --------------------------------------------------------------------------- #
# Evidence kind, granularity, scope and eligibility
# --------------------------------------------------------------------------- #


class EvidenceKind(Enum):
    """What kind of measurement an evidence record represents."""

    #: One exact per-frame ACK for one decision's reward-requested tensor.
    PER_FRAME_CAUSAL_ACK = "PER_FRAME_CAUSAL_ACK"
    #: A profile-level aggregate from the 288-cell / 72-action campaigns.
    AGGREGATE_PROFILE_CAMPAIGN = "AGGREGATE_PROFILE_CAMPAIGN"


class EvidenceGranularity(Enum):
    """The support an evidence record is valid over."""

    SINGLE_FRAME = "SINGLE_FRAME"
    PROFILE_AGGREGATE = "PROFILE_AGGREGATE"


#: Only this exact pair may produce a per-frame causal reward.  Aggregate
#: 72-action validation evidence is real evidence about payload, delivery and
#: profile-level quality, but it is an action/profile average rather than a
#: causal per-decision transition, so it can never stand in for one.
CAUSAL_REWARD_EVIDENCE: Tuple[EvidenceKind, EvidenceGranularity] = (
    EvidenceKind.PER_FRAME_CAUSAL_ACK,
    EvidenceGranularity.SINGLE_FRAME,
)


class RewardScope(Enum):
    """Which reward this measurement belongs to.

    ``PER_UE_PERCEPTION`` is the only scope this phase implements: what *this*
    UE's own split inference resolved, judged against what *this* UE could
    legitimately have perceived.  ``COOPERATIVE_MAP_COVERAGE`` -- whether the
    shared map ended up covering an object, possibly thanks to another agent --
    is a different quantity with a different eligibility set and a different
    credit assignment.  It is declared here so the two can never be silently
    summed, and it is **not implemented**.
    """

    PER_UE_PERCEPTION = "PER_UE_PERCEPTION"
    COOPERATIVE_MAP_COVERAGE = "COOPERATIVE_MAP_COVERAGE"


class VisibilityRule(Enum):
    """How a ground-truth object is judged perceivable by this UE."""

    #: Ray-cast line of sight from the UE sensor origin to the object.
    LINE_OF_SIGHT_VISIBILITY = "LINE_OF_SIGHT_VISIBILITY"
    #: CARLA actor-visible-object test (the AVO rule).
    AVO_ACTOR_VISIBLE_OBJECT = "AVO_ACTOR_VISIBLE_OBJECT"


class GroundTruthSource(Enum):
    """The origin of a quality measurement, with its deployability marked.

    ``CARLA_GT_EXACT`` is the only source this phase admits: a **privileged,
    non-deployable** simulator oracle.  Keeping it typed means the
    non-deployability travels with every record that carries it.
    """

    CARLA_GT_EXACT = "CARLA_GT_EXACT"

    @property
    def privileged(self) -> bool:
        return True

    @property
    def deployable(self) -> bool:
        """Always False: no deployed system observes exact per-frame accuracy."""
        return False


class QualityProducerStatus(Enum):
    """Whether the evidence was derived by an authenticated source producer.

    Phase 4a.2 currently has only the contract-fixture path.  It verifies
    internal arithmetic and binding, but all raw rows/arrays are still supplied
    by the caller and the live v1 producer emits none of them.  A future value
    may be added only together with a reviewed producer that reads the frozen
    CARLA snapshot/calibration/depth/semantic/prediction artifacts itself and
    persists their manifest before ACK emission.
    """

    CONTRACT_FIXTURE_UNVERIFIED_SOURCE = (
        "CONTRACT_FIXTURE_UNVERIFIED_SOURCE"
    )


@dataclass(frozen=True, slots=True)
class EvaluationEligibilityV1:
    """The hash-bound, UE-specific rule defining what this UE could perceive.

    **Ground-truth presence alone must never create a miss penalty.**  An object
    that exists in the simulator but lies beyond this UE's range, outside its
    field of view, or is occluded under the declared visibility rule was never
    this UE's to detect.  Charging a miss for it would penalize the policy for
    geometry it does not control, and would make the reward depend on scene
    population rather than on the action.

    So the recall denominator is the **eligible** GT set, not the raw GT set.  A
    class whose eligible count is zero is *undefined and excluded*, even when
    raw GT pixels or instances are present.

    The rule is identified by ``eligibility_contract_sha256`` so a transition
    always records which eligibility contract produced its masks; two runs under
    different range/FoV/visibility rules are then never pooled by accident.

    ``reward_scope`` is pinned to :attr:`RewardScope.PER_UE_PERCEPTION`: this is
    a per-UE perception judgement, deliberately separate from any later
    cooperative-map coverage reward, whose eligibility set and credit
    assignment differ.
    """

    ue_id: str
    eligibility_contract_id: str
    eligibility_contract_sha256: str
    max_range_m: float
    fov_deg: float
    visibility_rule: VisibilityRule
    segmentation_eligibility_masked: bool
    reward_scope: RewardScope = RewardScope.PER_UE_PERCEPTION

    @staticmethod
    def contract_document(
        *,
        eligibility_contract_id: str,
        max_range_m: float,
        fov_deg: float,
        visibility_rule: VisibilityRule,
        segmentation_eligibility_masked: bool,
        reward_scope: RewardScope = RewardScope.PER_UE_PERCEPTION,
    ) -> Dict[str, Any]:
        """Canonical eligibility-spec document whose hash is authoritative.

        The digest is derived from the actual range/FoV/visibility semantics;
        accepting an unrelated caller-declared 64-hex string would provide no
        evidence that the named contract is the one used by the evaluator.
        ``ue_id`` is intentionally not part of the reusable *specification*;
        the per-frame result below binds the concrete UE separately.
        """
        return {
            "class_visibility_rules": {
                "person": {
                    "minimum_projected_support_pixels": (
                        ELIGIBILITY_MIN_PROJECTED_SUPPORT_PIXELS
                    ),
                    "rule": visibility_rule.value,
                    "threshold": ELIGIBILITY_VISIBILITY_THRESHOLD,
                },
                "vehicle": {
                    "minimum_projected_support_pixels": (
                        ELIGIBILITY_MIN_PROJECTED_SUPPORT_PIXELS
                    ),
                    "rule": visibility_rule.value,
                    "threshold": ELIGIBILITY_VISIBILITY_THRESHOLD,
                },
            },
            "eligibility_contract_id": eligibility_contract_id,
            "fov_deg": float(fov_deg),
            "max_range_m": float(max_range_m),
            "record": "evaluation_eligibility_contract_v1",
            "reward_scope": reward_scope.value,
            "segmentation_domain_rule": SEGMENTATION_DOMAIN_RULE,
            "segmentation_eligibility_masked": (
                segmentation_eligibility_masked
            ),
            "visibility_algorithm_version": (
                ELIGIBILITY_VISIBILITY_ALGORITHM_VERSION
            ),
            "visibility_rule": visibility_rule.value,
        }

    @classmethod
    def from_spec(
        cls,
        *,
        ue_id: str,
        eligibility_contract_id: str,
        max_range_m: float,
        fov_deg: float,
        visibility_rule: VisibilityRule,
        segmentation_eligibility_masked: bool,
        reward_scope: RewardScope = RewardScope.PER_UE_PERCEPTION,
    ) -> "EvaluationEligibilityV1":
        """Construct the rule with a digest recomputed from its semantics."""
        document = cls.contract_document(
            eligibility_contract_id=eligibility_contract_id,
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            visibility_rule=visibility_rule,
            segmentation_eligibility_masked=segmentation_eligibility_masked,
            reward_scope=reward_scope,
        )
        return cls(
            ue_id=ue_id,
            eligibility_contract_id=eligibility_contract_id,
            eligibility_contract_sha256=canonical_sha256(document),
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            visibility_rule=visibility_rule,
            segmentation_eligibility_masked=(
                segmentation_eligibility_masked
            ),
            reward_scope=reward_scope,
        )

    def __post_init__(self) -> None:
        E = QualityContractError
        _non_empty_str(self.ue_id, "ue_id", E)
        _non_empty_str(self.eligibility_contract_id, "eligibility_contract_id", E)
        _sha256_hex(
            self.eligibility_contract_sha256, "eligibility_contract_sha256", E
        )
        max_range = _finite_float(self.max_range_m, "max_range_m", E)
        if max_range <= 0.0:
            raise E(f"max_range_m must be > 0, got {max_range}")
        fov = _finite_float(self.fov_deg, "fov_deg", E)
        if not 0.0 < fov <= 360.0:
            raise E(f"fov_deg must lie in (0, 360], got {fov}")
        if not isinstance(self.visibility_rule, VisibilityRule):
            raise E(
                f"visibility_rule must be a VisibilityRule (line-of-sight or "
                f"AVO), got {type(self.visibility_rule).__name__}: "
                f"{self.visibility_rule!r}"
            )
        _exact_bool(
            self.segmentation_eligibility_masked,
            "segmentation_eligibility_masked",
            E,
        )
        if not self.segmentation_eligibility_masked:
            raise E(
                "segmentation IoU must be computed over the eligibility-masked "
                "ground truth: a pixel-level score against unmasked GT would "
                "charge this UE for regions outside its own range, field of "
                "view or visibility, which is exactly the miss penalty that "
                "ground-truth presence alone must not create"
            )
        if not isinstance(self.reward_scope, RewardScope):
            raise E(
                f"reward_scope must be a RewardScope, got "
                f"{type(self.reward_scope).__name__}"
            )
        if self.reward_scope is not RewardScope.PER_UE_PERCEPTION:
            raise E(
                f"this contract measures {RewardScope.PER_UE_PERCEPTION.value} "
                f"only; {self.reward_scope.value} is a different quantity with "
                f"a different eligibility set and credit assignment and is not "
                f"implemented here.  The two rewards must never be summed"
            )
        expected_sha = canonical_sha256(
            self.contract_document(
                eligibility_contract_id=self.eligibility_contract_id,
                max_range_m=self.max_range_m,
                fov_deg=self.fov_deg,
                visibility_rule=self.visibility_rule,
                segmentation_eligibility_masked=(
                    self.segmentation_eligibility_masked
                ),
                reward_scope=self.reward_scope,
            )
        )
        if self.eligibility_contract_sha256 != expected_sha:
            raise E(
                "eligibility_contract_sha256 is disconnected from the "
                "declared range/FoV/visibility semantics: expected "
                f"{expected_sha}, got {self.eligibility_contract_sha256}.  "
                "Construct the rule with EvaluationEligibilityV1.from_spec()"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "contract_document": self.contract_document(
                eligibility_contract_id=self.eligibility_contract_id,
                max_range_m=self.max_range_m,
                fov_deg=self.fov_deg,
                visibility_rule=self.visibility_rule,
                segmentation_eligibility_masked=(
                    self.segmentation_eligibility_masked
                ),
                reward_scope=self.reward_scope,
            ),
            "eligibility_contract_id": self.eligibility_contract_id,
            "eligibility_contract_sha256": self.eligibility_contract_sha256,
            "fov_deg": float(self.fov_deg),
            "gt_presence_alone_penalizes": False,
            "max_range_m": float(self.max_range_m),
            "record": "evaluation_eligibility_v1",
            "reward_scope": self.reward_scope.value,
            "segmentation_eligibility_masked": (
                self.segmentation_eligibility_masked
            ),
            "ue_id": self.ue_id,
            "visibility_rule": self.visibility_rule.value,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class EvaluationEligibilityResultV1(_Attested):
    """Attested per-frame result of applying one eligibility contract.

    This is evidence about a *particular UE and frame*, not merely a declared
    rule.  It is issued only while verifying the presented digest-bound detail
    mapping whose digest is carried by the ACK.  The result binds the source
    GT snapshots, the exact eligible actor sets, the segmentation-eligibility
    mask and the sufficient pixel statistics used to reproduce each IoU.

    Historical ``splitfusion_privileged_quality_detail.v1`` rows without the
    :data:`QUALITY_DETAIL_SUPPORT_KEY` extension cannot produce this record.
    They remain scientifically useful historical evidence, but are explicitly
    unsupported as exact Phase-4a.2 learning rewards.
    """

    eligibility: EvaluationEligibilityV1
    run_id: str
    cell_id: str
    stream_id: str
    frame_id: int
    capture_timestamp_ns: int
    session_uuid: str
    decision_seq: int
    reward_tensor_seq: int
    executed_action_sha256: str
    gt_actor_snapshot_sha256: str
    gt_segmentation_label_sha256: str
    prediction_segmentation_label_sha256: str
    segmentation_eligibility_mask_sha256: str
    eligible_vehicle_actor_ids: Tuple[int, ...]
    eligible_person_actor_ids: Tuple[int, ...]
    eligible_vehicle_actor_ids_sha256: str
    eligible_person_actor_ids_sha256: str
    seg_vehicle_gt_pixels: int
    seg_person_gt_pixels: int
    seg_vehicle_pred_pixels: int
    seg_person_pred_pixels: int
    seg_vehicle_intersection_pixels: int
    seg_person_intersection_pixels: int
    seg_vehicle_union_pixels: int
    seg_person_union_pixels: int
    detailed_evidence_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.eligibility, EvaluationEligibilityV1):
            raise E(
                "eligibility must be a validated EvaluationEligibilityV1"
            )
        for name in ("run_id", "cell_id", "stream_id"):
            _non_empty_str(getattr(self, name), name, E)
        _non_negative_int(self.frame_id, "frame_id", E)
        _positive_int(self.capture_timestamp_ns, "capture_timestamp_ns", E)
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.decision_seq, "decision_seq", E)
        _non_negative_int(self.reward_tensor_seq, "reward_tensor_seq", E)
        for name in (
            "executed_action_sha256",
            "gt_actor_snapshot_sha256",
            "gt_segmentation_label_sha256",
            "prediction_segmentation_label_sha256",
            "segmentation_eligibility_mask_sha256",
            "eligible_vehicle_actor_ids_sha256",
            "eligible_person_actor_ids_sha256",
            "detailed_evidence_sha256",
        ):
            _sha256_hex(getattr(self, name), name, E)
        for name in (
            "eligible_vehicle_actor_ids",
            "eligible_person_actor_ids",
        ):
            actor_ids = getattr(self, name)
            if type(actor_ids) is not tuple:
                raise E(f"{name} must be a canonical tuple")
            for index, actor_id in enumerate(actor_ids):
                _non_negative_int(actor_id, f"{name}[{index}]", E)
            if actor_ids != tuple(sorted(set(actor_ids))):
                raise E(f"{name} must be sorted and duplicate-free")
        for name in (
            "seg_vehicle_gt_pixels",
            "seg_person_gt_pixels",
            "seg_vehicle_pred_pixels",
            "seg_person_pred_pixels",
            "seg_vehicle_intersection_pixels",
            "seg_person_intersection_pixels",
            "seg_vehicle_union_pixels",
            "seg_person_union_pixels",
        ):
            _non_negative_int(getattr(self, name), name, E)
        if self._attestation is not None and not _valid_eligibility_result(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the eligibility result's attestation is not bound to its "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_eligibility_result

    @staticmethod
    def _actor_set_document(
        class_name: str, actor_ids: Tuple[int, ...]
    ) -> Dict[str, Any]:
        return {
            "actor_ids": list(actor_ids),
            "class_name": class_name,
            "record": "eligible_actor_set_v1",
        }

    @classmethod
    def _from_verified_support(
        cls,
        support: Mapping[str, Any],
        *,
        eligibility: EvaluationEligibilityV1,
        obligation: "QualityAckObligationV1",
        detailed_evidence_sha256: str,
        ack_identity: Mapping[str, Any],
        ack_quality: Mapping[str, Any],
    ) -> "EvaluationEligibilityResultV1":
        """Parse support only after its enclosing detail digest was verified."""
        E = QualityContractError
        if not isinstance(support, Mapping):
            raise E(
                f"{QUALITY_DETAIL_SUPPORT_KEY} must be a mapping; legacy v1 "
                "detail rows without the Phase-4a.2 extension are unsupported"
            )
        if set(support) != set(QUALITY_DETAIL_SUPPORT_FIELDS):
            missing = sorted(set(QUALITY_DETAIL_SUPPORT_FIELDS) - set(support))
            extra = sorted(set(support) - set(QUALITY_DETAIL_SUPPORT_FIELDS))
            raise E(
                f"{QUALITY_DETAIL_SUPPORT_KEY} layout drift: missing={missing}, "
                f"extra={extra}; current v1 details without complete proof "
                "cannot be used as learning rewards"
            )
        if support["schema"] != QUALITY_DETAIL_SUPPORT_SCHEMA:
            raise E(
                f"quality-detail support schema must be "
                f"{QUALITY_DETAIL_SUPPORT_SCHEMA!r}, got "
                f"{support['schema']!r}"
            )

        expected_values = (
            ("ue_id", eligibility.ue_id),
            ("session_uuid", obligation.session_uuid),
            ("decision_seq", obligation.decision_seq),
            ("reward_tensor_seq", obligation.reward_tensor_seq),
            ("executed_action_sha256", obligation.executed_action_sha256),
            ("frame_id", obligation.frame_id),
            ("capture_timestamp_ns", obligation.capture_timestamp_ns),
            (
                "eligibility_contract_sha256",
                eligibility.eligibility_contract_sha256,
            ),
        )
        for name, expected in expected_values:
            if support[name] != expected:
                raise E(
                    f"quality-detail support {name}={support[name]!r} does "
                    f"not match the pre-existing obligation/spec value "
                    f"{expected!r}"
                )
        for name in QUALITY_ACK_IDENTITY_FIELDS:
            detail_value = support.get(name)
            if name in ("frame_id", "capture_timestamp_ns"):
                if int(ack_identity[name]) != int(detail_value):
                    raise E(
                        f"quality-detail support {name}={detail_value!r} does "
                        f"not match ACK {ack_identity[name]!r}"
                    )

        # Recompute the eligible actor sets from the actual hash-bound actor
        # rows.  Merely checking a digest over a caller-supplied list is not a
        # derivation: the same-cardinality set could otherwise be replaced by
        # arbitrary actor IDs and re-hashed.  The raw rows are canonicalized,
        # hashed and then evaluated under the exact registered geometry and
        # visibility semantics.
        raw_actor_rows = support["gt_actor_rows"]
        if not isinstance(raw_actor_rows, list):
            raise E("gt_actor_rows must be a JSON list")
        canonical_actor_rows = []
        seen_actor_ids = set()
        derived_actor_sets: Dict[str, list] = {"vehicle": [], "person": []}
        for index, raw_row in enumerate(raw_actor_rows):
            if not isinstance(raw_row, Mapping):
                raise E(f"gt_actor_rows[{index}] must be a mapping")
            required = {
                "actor_id",
                "class_name",
                "range_m",
                "bearing_deg",
                "line_of_sight_visible",
                "visibility_score",
                "projected_support_pixels",
            }
            if set(raw_row) != required:
                raise E(
                    f"gt_actor_rows[{index}] layout drift: "
                    f"missing={sorted(required - set(raw_row))}, "
                    f"extra={sorted(set(raw_row) - required)}"
                )
            actor_id = _non_negative_int(
                raw_row["actor_id"], f"gt_actor_rows[{index}].actor_id", E
            )
            if actor_id in seen_actor_ids:
                raise E(f"duplicate gt actor_id {actor_id}")
            seen_actor_ids.add(actor_id)
            class_name = raw_row["class_name"]
            if class_name not in derived_actor_sets:
                raise E(
                    f"gt_actor_rows[{index}].class_name must be vehicle or "
                    f"person, got {class_name!r}"
                )
            range_m = _finite_float(
                raw_row["range_m"], f"gt_actor_rows[{index}].range_m", E
            )
            if range_m < 0.0:
                raise E(f"gt_actor_rows[{index}].range_m must be >= 0")
            bearing_deg = _finite_float(
                raw_row["bearing_deg"],
                f"gt_actor_rows[{index}].bearing_deg",
                E,
            )
            line_of_sight_visible = _exact_bool(
                raw_row["line_of_sight_visible"],
                f"gt_actor_rows[{index}].line_of_sight_visible",
                E,
            )
            visibility_score = _finite_in(
                raw_row["visibility_score"],
                f"gt_actor_rows[{index}].visibility_score",
                0.0,
                1.0,
                E,
            )
            projected_support_pixels = _non_negative_int(
                raw_row["projected_support_pixels"],
                f"gt_actor_rows[{index}].projected_support_pixels",
                E,
            )
            canonical_row = {
                "actor_id": actor_id,
                "bearing_deg": bearing_deg,
                "class_name": class_name,
                "line_of_sight_visible": line_of_sight_visible,
                "projected_support_pixels": projected_support_pixels,
                "range_m": range_m,
                "visibility_score": visibility_score,
            }
            canonical_actor_rows.append(canonical_row)

            geometry_eligible = (
                range_m <= float(eligibility.max_range_m)
                and abs(bearing_deg) <= float(eligibility.fov_deg) / 2.0
                and projected_support_pixels
                >= ELIGIBILITY_MIN_PROJECTED_SUPPORT_PIXELS
            )
            if eligibility.visibility_rule is VisibilityRule.LINE_OF_SIGHT_VISIBILITY:
                visible = line_of_sight_visible
            else:
                visible = visibility_score >= ELIGIBILITY_VISIBILITY_THRESHOLD
            if geometry_eligible and visible:
                derived_actor_sets[class_name].append(actor_id)

        canonical_actor_rows.sort(key=lambda row: int(row["actor_id"]))
        actor_snapshot_document = {
            "actors": canonical_actor_rows,
            "record": "carla_gt_actor_snapshot_v1",
        }
        expected_actor_snapshot_sha = canonical_sha256(actor_snapshot_document)
        _sha256_hex(
            support["gt_actor_snapshot_sha256"],
            "gt_actor_snapshot_sha256",
            E,
        )
        if support["gt_actor_snapshot_sha256"] != expected_actor_snapshot_sha:
            raise E(
                "gt_actor_snapshot_sha256 does not match the canonical raw "
                "actor rows"
            )

        actor_sets: Dict[str, Tuple[int, ...]] = {}
        actor_digests: Dict[str, str] = {}
        for class_name in ("vehicle", "person"):
            ids = tuple(sorted(derived_actor_sets[class_name]))
            list_name = f"eligible_{class_name}_actor_ids"
            raw_ids = support[list_name]
            if not isinstance(raw_ids, list):
                raise E(f"{list_name} must be a JSON list")
            claimed_ids = tuple(
                _non_negative_int(value, f"{list_name}[{index}]", E)
                for index, value in enumerate(raw_ids)
            )
            if claimed_ids != ids:
                raise E(
                    f"{list_name} was not derived from gt_actor_rows under "
                    f"the registered range/FoV/visibility rule: expected "
                    f"{ids}, got {claimed_ids}"
                )
            digest = canonical_sha256(cls._actor_set_document(class_name, ids))
            digest_name = f"eligible_{class_name}_actor_ids_sha256"
            _sha256_hex(support[digest_name], digest_name, E)
            if support[digest_name] != digest:
                raise E(
                    f"{digest_name} does not match the derived actor-id set: "
                    f"expected {digest}, got {support[digest_name]}"
                )
            actor_sets[class_name] = ids
            actor_digests[class_name] = digest

        # Preserve and verify the exact localization match ledger.  Aggregate
        # TP/FN/error values alone cannot show *which* eligible actor was found,
        # and therefore cannot support later scientific audit or detect an
        # actor-identity substitution with unchanged cardinality.
        for class_name in ("vehicle", "person"):
            field_name = f"{class_name}_localization_matches"
            raw_matches = support[field_name]
            if not isinstance(raw_matches, list):
                raise E(f"{field_name} must be a JSON list")
            matched_actor_ids = []
            matched_prediction_indices = []
            xy_errors = []
            footprint_ious = []
            for index, raw_match in enumerate(raw_matches):
                if not isinstance(raw_match, Mapping):
                    raise E(f"{field_name}[{index}] must be a mapping")
                required = {
                    "prediction_index",
                    "gt_actor_id",
                    "xy_error_m",
                    "footprint_iou",
                }
                if set(raw_match) != required:
                    raise E(
                        f"{field_name}[{index}] layout drift: "
                        f"missing={sorted(required - set(raw_match))}, "
                        f"extra={sorted(set(raw_match) - required)}"
                    )
                prediction_index = _non_negative_int(
                    raw_match["prediction_index"],
                    f"{field_name}[{index}].prediction_index",
                    E,
                )
                actor_id = _non_negative_int(
                    raw_match["gt_actor_id"],
                    f"{field_name}[{index}].gt_actor_id",
                    E,
                )
                xy_error = _finite_float(
                    raw_match["xy_error_m"],
                    f"{field_name}[{index}].xy_error_m",
                    E,
                )
                if xy_error < 0.0:
                    raise E(f"{field_name}[{index}].xy_error_m must be >= 0")
                footprint_iou = _finite_in(
                    raw_match["footprint_iou"],
                    f"{field_name}[{index}].footprint_iou",
                    0.0,
                    1.0,
                    E,
                )
                if actor_id not in actor_sets[class_name]:
                    raise E(
                        f"{field_name}[{index}] matches ineligible/unknown "
                        f"{class_name} actor {actor_id}"
                    )
                matched_actor_ids.append(actor_id)
                matched_prediction_indices.append(prediction_index)
                xy_errors.append(xy_error)
                footprint_ious.append(footprint_iou)
            if len(set(matched_actor_ids)) != len(matched_actor_ids):
                raise E(f"{field_name} matches one GT actor more than once")
            if len(set(matched_prediction_indices)) != len(
                matched_prediction_indices
            ):
                raise E(f"{field_name} reuses one prediction more than once")

            unmatched_gt_name = f"unmatched_{class_name}_gt_actor_ids"
            unmatched_gt = support[unmatched_gt_name]
            if not isinstance(unmatched_gt, list):
                raise E(f"{unmatched_gt_name} must be a JSON list")
            unmatched_gt_ids = tuple(
                _non_negative_int(value, f"{unmatched_gt_name}[{index}]", E)
                for index, value in enumerate(unmatched_gt)
            )
            expected_unmatched_gt = tuple(
                sorted(set(actor_sets[class_name]) - set(matched_actor_ids))
            )
            if unmatched_gt_ids != expected_unmatched_gt:
                raise E(
                    f"{unmatched_gt_name} must equal eligible minus matched: "
                    f"expected {expected_unmatched_gt}, got {unmatched_gt_ids}"
                )

            unmatched_pred_name = (
                f"unmatched_{class_name}_prediction_indices"
            )
            unmatched_pred = support[unmatched_pred_name]
            if not isinstance(unmatched_pred, list):
                raise E(f"{unmatched_pred_name} must be a JSON list")
            unmatched_pred_indices = tuple(
                _non_negative_int(
                    value, f"{unmatched_pred_name}[{index}]", E
                )
                for index, value in enumerate(unmatched_pred)
            )
            if unmatched_pred_indices != tuple(
                sorted(set(unmatched_pred_indices))
            ):
                raise E(
                    f"{unmatched_pred_name} must be sorted and duplicate-free"
                )
            if set(unmatched_pred_indices) & set(matched_prediction_indices):
                raise E(
                    f"{unmatched_pred_name} overlaps matched predictions"
                )

            tp = _non_negative_int(
                ack_quality[f"{class_name}_tp"], f"{class_name}_tp", E
            )
            fn = _non_negative_int(
                ack_quality[f"{class_name}_fn"], f"{class_name}_fn", E
            )
            if tp != len(raw_matches) or fn != len(unmatched_gt_ids):
                raise E(
                    f"ACK {class_name} TP/FN ({tp}/{fn}) do not match the "
                    f"verified match ledger ({len(raw_matches)}/"
                    f"{len(unmatched_gt_ids)})"
                )
            ack_xy = ack_quality[f"{class_name}_xy_error_m"]
            ack_footprint = ack_quality[f"{class_name}_footprint_iou"]
            if raw_matches:
                expected_xy = sum(xy_errors) / len(xy_errors)
                expected_footprint = sum(footprint_ious) / len(footprint_ious)
                if ack_xy is None or not math.isclose(
                    float(ack_xy), expected_xy, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise E(
                        f"ACK {class_name}_xy_error_m={ack_xy!r} disagrees "
                        f"with match-ledger mean {expected_xy}"
                    )
                if ack_footprint is None or not math.isclose(
                    float(ack_footprint),
                    expected_footprint,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise E(
                        f"ACK {class_name}_footprint_iou="
                        f"{ack_footprint!r} disagrees with match-ledger mean "
                        f"{expected_footprint}"
                    )
            elif ack_xy is not None or ack_footprint is not None:
                raise E(
                    f"{class_name} has no matched prediction, so aggregate "
                    "localization error/IoU must be absent"
                )

        # Recompute the segmentation evaluation domain and sufficient IoU
        # statistics from raw per-pixel evidence.  The same finite, positive,
        # in-range depth domain is applied to prediction and GT, so an in-domain
        # false positive remains a penalty while out-of-domain pixels never
        # enter either side of the comparison.
        raw_depth = support["segmentation_depth_m"]
        if not isinstance(raw_depth, list) or not raw_depth:
            raise E("segmentation_depth_m must be a non-empty JSON list")
        depth_values = []
        domain_indices = set()
        for index, value in enumerate(raw_depth):
            if value is None:
                depth_values.append(None)
                continue
            depth = _finite_float(value, f"segmentation_depth_m[{index}]", E)
            depth_values.append(depth)
            if 0.0 < depth <= float(eligibility.max_range_m):
                domain_indices.add(index)

        def _indices(field_name: str) -> Tuple[int, ...]:
            raw = support[field_name]
            if not isinstance(raw, list):
                raise E(f"{field_name} must be a JSON list")
            values = tuple(
                _non_negative_int(value, f"{field_name}[{index}]", E)
                for index, value in enumerate(raw)
            )
            if values != tuple(sorted(set(values))):
                raise E(f"{field_name} must be sorted and duplicate-free")
            if values and values[-1] >= len(depth_values):
                raise E(
                    f"{field_name} index {values[-1]} exceeds depth support "
                    f"length {len(depth_values)}"
                )
            return values

        raw_gt = {
            class_name: _indices(f"segmentation_gt_{class_name}_indices")
            for class_name in ("vehicle", "person")
        }
        raw_pred = {
            class_name: _indices(f"segmentation_pred_{class_name}_indices")
            for class_name in ("vehicle", "person")
        }
        if set(raw_gt["vehicle"]) & set(raw_gt["person"]):
            raise E("GT vehicle/person semantic masks overlap")
        if set(raw_pred["vehicle"]) & set(raw_pred["person"]):
            raise E("predicted vehicle/person semantic masks overlap")

        gt_label_document = {
            "person_indices": list(raw_gt["person"]),
            "record": "semantic_gt_class_support_v1",
            "vehicle_indices": list(raw_gt["vehicle"]),
        }
        pred_label_document = {
            "person_indices": list(raw_pred["person"]),
            "record": "semantic_prediction_class_support_v1",
            "vehicle_indices": list(raw_pred["vehicle"]),
        }
        domain_document = {
            "domain_indices": sorted(domain_indices),
            "max_range_m": float(eligibility.max_range_m),
            "record": "segmentation_eligibility_domain_v1",
            "rule": SEGMENTATION_DOMAIN_RULE,
        }
        expected_source_hashes = {
            "gt_segmentation_label_sha256": canonical_sha256(gt_label_document),
            "prediction_segmentation_label_sha256": canonical_sha256(
                pred_label_document
            ),
            "segmentation_eligibility_mask_sha256": canonical_sha256(
                domain_document
            ),
        }
        for name, expected in expected_source_hashes.items():
            _sha256_hex(support[name], name, E)
            if support[name] != expected:
                raise E(f"{name} does not match its raw per-pixel evidence")

        pixels: Dict[str, int] = {}
        for class_name in ("vehicle", "person"):
            gt_set = set(raw_gt[class_name]) & domain_indices
            pred_set = set(raw_pred[class_name]) & domain_indices
            derived = {
                "gt": len(gt_set),
                "pred": len(pred_set),
                "intersection": len(gt_set & pred_set),
                "union": len(gt_set | pred_set),
            }
            for part, expected in derived.items():
                name = f"seg_{class_name}_{part}_pixels"
                claimed = _non_negative_int(support[name], name, E)
                if claimed != expected:
                    raise E(
                        f"{name}={claimed} was not recomputed from the raw "
                        f"GT/prediction masks and depth domain; expected "
                        f"{expected}"
                    )
                pixels[name] = expected
            gt = derived["gt"]
            pred = derived["pred"]
            intersection = derived["intersection"]
            union = derived["union"]
            if int(ack_quality[f"gt_{class_name}_pixels"]) != gt:
                raise E(
                    f"ACK gt_{class_name}_pixels does not match the masked "
                    f"detail count {gt}"
                )
            ack_iou = ack_quality[f"seg_{class_name}_iou"]
            expected_iou: Optional[float] = (
                None if union == 0 else float(intersection) / float(union)
            )
            if expected_iou is None:
                if ack_iou not in (None, 0, 0.0):
                    raise E(
                        f"{class_name} masks are both empty but ACK IoU is "
                        f"{ack_iou!r}"
                    )
            elif ack_iou is None or not math.isclose(
                float(ack_iou), expected_iou, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise E(
                    f"ACK seg_{class_name}_iou={ack_iou!r} cannot be "
                    f"reproduced from intersection/union "
                    f"{intersection}/{union}={expected_iou}"
                )

            eligible_count = len(actor_sets[class_name])
            tp = _non_negative_int(
                ack_quality[f"{class_name}_tp"], f"{class_name}_tp", E
            )
            fn = _non_negative_int(
                ack_quality[f"{class_name}_fn"], f"{class_name}_fn", E
            )
            if tp + fn != eligible_count:
                raise E(
                    f"ACK {class_name}_tp + {class_name}_fn = {tp + fn}, "
                    f"but the verified eligible actor set contains "
                    f"{eligible_count}; recall cannot be attributed to this "
                    "eligibility result"
                )
            ack_recall = ack_quality[f"{class_name}_recall"]
            expected_recall: Optional[float] = (
                None if eligible_count == 0 else float(tp) / eligible_count
            )
            if expected_recall is None:
                if ack_recall is not None:
                    raise E(
                        f"{class_name} has no eligible objects, so recall must "
                        f"be undefined; ACK carries {ack_recall!r}"
                    )
            elif ack_recall is None or not math.isclose(
                float(ack_recall),
                expected_recall,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise E(
                    f"ACK {class_name}_recall={ack_recall!r} disagrees with "
                    f"verified tp/eligible={tp}/{eligible_count}="
                    f"{expected_recall}"
                )

        record = cls(
            eligibility=eligibility,
            run_id=str(ack_identity["run_id"]),
            cell_id=str(ack_identity["cell_id"]),
            stream_id=str(ack_identity["stream_id"]),
            frame_id=int(support["frame_id"]),
            capture_timestamp_ns=int(support["capture_timestamp_ns"]),
            session_uuid=str(support["session_uuid"]),
            decision_seq=int(support["decision_seq"]),
            reward_tensor_seq=int(support["reward_tensor_seq"]),
            executed_action_sha256=str(support["executed_action_sha256"]),
            gt_actor_snapshot_sha256=str(support["gt_actor_snapshot_sha256"]),
            gt_segmentation_label_sha256=str(
                support["gt_segmentation_label_sha256"]
            ),
            prediction_segmentation_label_sha256=str(
                support["prediction_segmentation_label_sha256"]
            ),
            segmentation_eligibility_mask_sha256=str(
                support["segmentation_eligibility_mask_sha256"]
            ),
            eligible_vehicle_actor_ids=actor_sets["vehicle"],
            eligible_person_actor_ids=actor_sets["person"],
            eligible_vehicle_actor_ids_sha256=actor_digests["vehicle"],
            eligible_person_actor_ids_sha256=actor_digests["person"],
            seg_vehicle_gt_pixels=pixels["seg_vehicle_gt_pixels"],
            seg_person_gt_pixels=pixels["seg_person_gt_pixels"],
            seg_vehicle_pred_pixels=pixels["seg_vehicle_pred_pixels"],
            seg_person_pred_pixels=pixels["seg_person_pred_pixels"],
            seg_vehicle_intersection_pixels=pixels[
                "seg_vehicle_intersection_pixels"
            ],
            seg_person_intersection_pixels=pixels[
                "seg_person_intersection_pixels"
            ],
            seg_vehicle_union_pixels=pixels["seg_vehicle_union_pixels"],
            seg_person_union_pixels=pixels["seg_person_union_pixels"],
            detailed_evidence_sha256=detailed_evidence_sha256,
        )
        return replace(
            record,
            _attestation=_issue_eligibility_result(record._binding()),
        )

    @property
    def vehicle_eligible_gt_instances(self) -> int:
        return len(self.eligible_vehicle_actor_ids)

    @property
    def person_eligible_gt_instances(self) -> int:
        return len(self.eligible_person_actor_ids)

    @property
    def producer_status(self) -> QualityProducerStatus:
        """Current support is self-consistent, but not source-authenticated."""
        return QualityProducerStatus.CONTRACT_FIXTURE_UNVERIFIED_SOURCE

    @property
    def learning_ready(self) -> bool:
        """False until a reviewed raw-artifact producer issues the record."""
        return False

    def require_learning_ready(self) -> None:
        if not self.learning_ready:
            raise QualityProducerUnavailableError(
                "the Phase-4a.2 quality support is a self-consistent contract "
                "fixture, not source-authenticated learning evidence.  The "
                "current live v1 producer strips actor/projection provenance "
                "and persists aggregate scores only.  Upgrade the producer to "
                "derive and manifest actor eligibility, depth-masked GT and "
                "prediction masks, and the localization match ledger from "
                "frozen CARLA source artifacts before admitting this reward"
            )

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "cell_id": self.cell_id,
            "decision_seq": self.decision_seq,
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "eligibility": self.eligibility.to_canonical_dict(),
            "eligible_person_actor_ids": list(
                self.eligible_person_actor_ids
            ),
            "eligible_person_actor_ids_sha256": (
                self.eligible_person_actor_ids_sha256
            ),
            "eligible_vehicle_actor_ids": list(
                self.eligible_vehicle_actor_ids
            ),
            "eligible_vehicle_actor_ids_sha256": (
                self.eligible_vehicle_actor_ids_sha256
            ),
            "executed_action_sha256": self.executed_action_sha256,
            "frame_id": self.frame_id,
            "gt_actor_snapshot_sha256": self.gt_actor_snapshot_sha256,
            "gt_segmentation_label_sha256": (
                self.gt_segmentation_label_sha256
            ),
            "prediction_segmentation_label_sha256": (
                self.prediction_segmentation_label_sha256
            ),
            "producer_status": self.producer_status.value,
            "learning_ready": self.learning_ready,
            "record": "evaluation_eligibility_result_v1",
            "reward_tensor_seq": self.reward_tensor_seq,
            "run_id": self.run_id,
            "seg_person_gt_pixels": self.seg_person_gt_pixels,
            "seg_person_intersection_pixels": (
                self.seg_person_intersection_pixels
            ),
            "seg_person_pred_pixels": self.seg_person_pred_pixels,
            "seg_person_union_pixels": self.seg_person_union_pixels,
            "seg_vehicle_gt_pixels": self.seg_vehicle_gt_pixels,
            "seg_vehicle_intersection_pixels": (
                self.seg_vehicle_intersection_pixels
            ),
            "seg_vehicle_pred_pixels": self.seg_vehicle_pred_pixels,
            "seg_vehicle_union_pixels": self.seg_vehicle_union_pixels,
            "segmentation_eligibility_mask_sha256": (
                self.segmentation_eligibility_mask_sha256
            ),
            "session_uuid": self.session_uuid,
            "stream_id": self.stream_id,
        }


class QualityAckUseRegistryV1:
    """Process-wide bridge for the identity fields missing from a v1 ACK.

    V1 omits the decision identity, so durable replay requires protocol v2.
    This registry enforces a one-to-one mapping inside one verifier process:
    one raw wire digest cannot be rebound to another obligation/ticket, and one
    completed ticket cannot accept two conflicting raw ACK documents (including
    a conflict hidden behind a caller-selected obligation).
    Revalidating the exact same pair is idempotent; every different binding
    fails closed.

    The verifier uses one private module-owned instance.  A caller cannot pass
    a fresh registry to erase an earlier claim.  That closes the in-process
    ambiguity, while still making no cross-process durability claim; protocol
    v2 is required for that.
    """

    __slots__ = ("registry_id", "_claims", "_bindings", "_lock")

    def __init__(self, registry_id: str) -> None:
        self.registry_id = _non_empty_str(
            registry_id, "registry_id", QualityContractError
        )
        self._claims: Dict[str, Tuple[str, str]] = {}
        # Ticket identity is the reverse-map key.  Keying this by the
        # obligation as well would let a caller mint a second obligation with
        # a different run/cell label and thereby attach a second ACK to the
        # same completed decision.
        self._bindings: Dict[str, Tuple[str, str]] = {}
        self._lock = threading.Lock()

    def claim(
        self,
        raw_quality_ack_sha256: str,
        *,
        obligation_sha256: str,
        completed_ticket_sha256: str,
    ) -> None:
        E = QualityContractError
        raw = _sha256_hex(raw_quality_ack_sha256, "raw ACK digest", E)
        obligation = _sha256_hex(
            obligation_sha256, "obligation_sha256", E
        )
        ticket = _sha256_hex(
            completed_ticket_sha256, "completed_ticket_sha256", E
        )
        claim = (obligation, ticket)
        with self._lock:
            existing_claim = self._claims.get(raw)
            if existing_claim is not None and existing_claim != claim:
                raise QualityAckReuseError(
                    f"v1 ACK digest {raw} is already bound in registry "
                    f"{self.registry_id!r} to obligation/ticket "
                    f"{existing_claim}; "
                    f"refused different binding {claim}.  V1 lacks durable "
                    "decision identity; protocol v2 is required across "
                    "processes"
                )
            existing_binding = self._bindings.get(ticket)
            binding = (raw, obligation)
            if existing_binding is not None and existing_binding != binding:
                raise QualityAckReuseError(
                    f"completed ticket {ticket} is already bound in registry "
                    f"{self.registry_id!r} to ACK/obligation "
                    f"{existing_binding}; refused conflicting second "
                    f"ACK/obligation {binding}"
                )
            self._claims[raw] = claim
            self._bindings[ticket] = binding

    def claim_count(self) -> int:
        """Number of distinct wire ACK documents claimed in this registry."""
        with self._lock:
            return len(self._claims)


# One verifier process has exactly one v1-ACK namespace.  Keeping this object
# private is essential: accepting a caller-created registry would let the same
# identity-poor ACK be rebound simply by supplying an empty registry.
_PROCESS_QUALITY_ACK_USE_REGISTRY = QualityAckUseRegistryV1(
    "splitfusion-phase4a2-process-v1"
)


# --------------------------------------------------------------------------- #
# A. Retrospective identity obligation and document-verified ACK binding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class QualityAckObligationV1:
    """Identity assertion for the reward-requested tensor and expected ACK.

    It binds the transport identity (``run_id``/``cell_id``/
    ``stream_id``), the frame identity (``frame_id`` and
    ``capture_timestamp_ns``), the decision identity (``session_uuid``,
    ``decision_seq``, ``reward_tensor_seq``) and the **complete** executed
    action.

    Phase 4a.2 has no durable transmission-time issuer or attestation for this
    object.  :meth:`for_reward_tensor` reconstructs it from a completed ticket,
    so this record proves cross-field agreement but does **not** prove that the
    commitment predated ACK arrival.  Exact positive rewards remain blocked;
    the real producer/protocol-v2 integration must persist an authenticated
    obligation when the reward envelope is actually transmitted.

    The identity assertion may describe an off-anchor action.  What fails
    closed is *binding a v1 ACK to it*, because the v1 carrier cannot name an
    off-anchor action; the assertion itself is not proof that transmission
    occurred.
    """

    run_id: str
    cell_id: str
    stream_id: str
    frame_id: int
    capture_timestamp_ns: int
    session_uuid: str
    decision_seq: int
    reward_tensor_seq: int
    executed_action: ExecutedActionIdentity

    def __post_init__(self) -> None:
        E = QualityContractError
        for name in ("run_id", "cell_id", "stream_id"):
            _non_empty_str(getattr(self, name), name, E)
        _non_negative_int(self.frame_id, "frame_id", E)
        # The v1 protocol validates capture_timestamp_ns > 0.
        _positive_int(self.capture_timestamp_ns, "capture_timestamp_ns", E)
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.decision_seq, "decision_seq", E)
        _non_negative_int(self.reward_tensor_seq, "reward_tensor_seq", E)
        if not isinstance(self.executed_action, ExecutedActionIdentity):
            raise E(
                f"executed_action must be a Phase-2 ExecutedActionIdentity, "
                f"got {type(self.executed_action).__name__}"
            )
        self.executed_action.require_reconciled()

    @classmethod
    def for_reward_tensor(
        cls,
        completed_ticket: CompletedTicket,
        *,
        run_id: str,
        cell_id: str,
        stream_id: str,
        capture_timestamp_ns: int,
    ) -> "QualityAckObligationV1":
        """Reconstruct a candidate obligation from a completed ticket.

        This convenience path does not establish transmission-time provenance;
        see the class-level fail-closed contract.
        """
        if not isinstance(completed_ticket, CompletedTicket):
            raise QualityContractError(
                f"completed_ticket must be a controller CompletedTicket, got "
                f"{type(completed_ticket).__name__}"
            )
        return cls(
            run_id=run_id,
            cell_id=cell_id,
            stream_id=stream_id,
            frame_id=completed_ticket.reward_carla_frame_id,
            capture_timestamp_ns=capture_timestamp_ns,
            session_uuid=completed_ticket.session_uuid,
            decision_seq=completed_ticket.decision_seq,
            reward_tensor_seq=completed_ticket.reward_tensor_seq,
            executed_action=completed_ticket.action,
        )

    @property
    def executed_action_sha256(self) -> str:
        """Canonical hash of the complete executed-action identity."""
        return self.executed_action.canonical_sha256()

    @property
    def is_anchor_expressible(self) -> bool:
        """True when a v1 anchor-only ACK could name this action at all."""
        return self.executed_action.action_id is not None

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "cell_id": self.cell_id,
            "decision_seq": self.decision_seq,
            "executed_action": self.executed_action.to_canonical_dict(),
            "executed_action_sha256": self.executed_action_sha256,
            "frame_id": self.frame_id,
            "is_anchor_expressible": self.is_anchor_expressible,
            "precommit_authenticated": (
                QUALITY_OBLIGATION_PRECOMMIT_AUTHENTICATED
            ),
            "record": "quality_ack_obligation_v1",
            "reward_tensor_seq": self.reward_tensor_seq,
            "run_id": self.run_id,
            "session_uuid": self.session_uuid,
            "stream_id": self.stream_id,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class QualityAckBindingV1(_Attested):
    """A verified binding to one real ``sf_priv_quality_ack.v1`` document.

    Build with :meth:`from_ack_document`, which is the only path that can issue
    the construction attestation.  That factory:

    1. verifies the declared protocol binding against the real module;
    2. calls the **real** ``protocol.validate`` on the document;
    3. computes the raw ACK SHA-256 itself with the protocol's own
       canonicalizer -- a caller-supplied hash is never accepted as proof;
    4. extracts and retains the edge-detail digest ``dh``;
    5. retains all seven v1 identity fields verbatim;
    6. cross-checks the document against the transmission obligation **and**
       against the :class:`~.reward_ticket_controller.CompletedTicket`.

    A directly constructed instance is unattested and cannot serialize.
    """

    raw_quality_ack_sha256: str
    detailed_evidence_sha256: str
    ack_schema: str
    ack_protocol_version: int
    ack_source: str
    identity_fields: Mapping[str, Any]
    quality_fields: Mapping[str, Any]
    evaluator_mode: str
    obligation_sha256: str
    completed_ticket_sha256: str
    eligibility_result: EvaluationEligibilityResultV1
    ack_use_registry_id: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = QualityContractError
        _sha256_hex(self.raw_quality_ack_sha256, "raw_quality_ack_sha256", E)
        _sha256_hex(self.detailed_evidence_sha256, "detailed_evidence_sha256", E)
        _sha256_hex(self.obligation_sha256, "obligation_sha256", E)
        _sha256_hex(
            self.completed_ticket_sha256, "completed_ticket_sha256", E
        )
        if self.ack_schema != QUALITY_ACK_SCHEMA:
            raise E(
                f"ack_schema must be the evaluated schema "
                f"{QUALITY_ACK_SCHEMA!r}, got {self.ack_schema!r}"
            )
        if self.ack_protocol_version != QUALITY_ACK_PROTOCOL_VERSION:
            raise E(
                f"ack_protocol_version must be "
                f"{QUALITY_ACK_PROTOCOL_VERSION}, got "
                f"{self.ack_protocol_version!r}"
            )
        if self.ack_source != QUALITY_ACK_SOURCE:
            raise E(f"ack_source must be {QUALITY_ACK_SOURCE!r}")
        for name, expected in (
            ("identity_fields", QUALITY_ACK_IDENTITY_FIELDS),
            ("quality_fields", QUALITY_ACK_QUALITY_FIELDS),
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise E(f"{name} must be a mapping, got {type(value).__name__}")
            if tuple(sorted(value)) != tuple(sorted(expected)):
                raise E(
                    f"{name} must carry exactly the v1 layout {list(expected)}; "
                    f"got {sorted(value)}"
                )
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        _non_empty_str(self.evaluator_mode, "evaluator_mode", E)
        _non_empty_str(self.ack_use_registry_id, "ack_use_registry_id", E)
        if not isinstance(
            self.eligibility_result, EvaluationEligibilityResultV1
        ):
            raise E(
                "eligibility_result must be an "
                "EvaluationEligibilityResultV1 parsed from the actual detail"
            )
        self.eligibility_result.require_attested()
        if (
            self.eligibility_result.detailed_evidence_sha256
            != self.detailed_evidence_sha256
        ):
            raise E(
                "eligibility_result is bound to a different detail document"
            )
        if self._attestation is not None and not _valid_ack(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the ACK binding's attestation does not match its own "
                "serialized fields, so it was copied or the record was mutated"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_ack

    # -- the only validating path ------------------------------------------ #

    @classmethod
    def from_ack_document(
        cls,
        document: Mapping[str, Any],
        *,
        detail_document: Mapping[str, Any],
        obligation: QualityAckObligationV1,
        completed_ticket: CompletedTicket,
        eligibility: EvaluationEligibilityV1,
        expected_raw_ack_sha256: Optional[str] = None,
    ) -> "QualityAckBindingV1":
        """Verify a real ACK document against its obligation and ticket.

        ``expected_raw_ack_sha256`` is *optional* and is only ever used as an
        extra cross-check: the authoritative hash is recomputed here from the
        document itself, so a caller-supplied digest can confirm agreement but
        can never substitute for the document.

        Raises:
            OffAnchorQualityAckError: if the obligation's action is off-anchor.
            QualityContractError: on any protocol, identity or hash mismatch.
        """
        verify_quality_protocol_binding()
        protocol = _load_quality_protocol()
        if not isinstance(document, Mapping):
            raise QualityContractError(
                f"document must be the ACK mapping itself, got "
                f"{type(document).__name__}; an opaque hash is never accepted "
                f"in place of the document"
            )
        if not isinstance(obligation, QualityAckObligationV1):
            raise QualityContractError(
                "obligation must be a QualityAckObligationV1 identity "
                "assertion (not transmission-time provenance), got "
                f"{type(obligation).__name__}"
            )
        if not isinstance(completed_ticket, CompletedTicket):
            raise QualityContractError(
                f"completed_ticket must be a controller CompletedTicket, got "
                f"{type(completed_ticket).__name__}"
            )
        # ``CompletedTicket`` is a public frozen dataclass, so a type check
        # alone does not prove that the controller actually closed it.
        try:
            completed_ticket.require_lineage_attested()
        except RewardTicketControllerError as exc:
            raise QualityContractError(
                "completed_ticket lacks controller-issued lineage proof"
            ) from exc
        if not isinstance(eligibility, EvaluationEligibilityV1):
            raise QualityContractError(
                "eligibility must be the pre-registered "
                "EvaluationEligibilityV1 used by the evaluator"
            )
        action = obligation.executed_action
        if action.action_id is None or action.profile_id is None:
            raise OffAnchorQualityAckError(
                f"executed action {action.canonical_mode} q_e4={action.q_e4} is "
                f"not a registered anchor, so {QUALITY_ACK_SCHEMA} cannot "
                f"identify it: that contract requires "
                f"{list(QUALITY_ACK_REQUIRED_ANCHOR_FIELDS)} and validates "
                f"0 <= action_id < {QUALITY_ACK_ANCHOR_ACTION_COUNT}.  Refused "
                f"rather than snapped to a nearest anchor.  "
                f"{PROTOCOL_V2_REQUIREMENT}"
            )

        # 1. the real validator, not a local re-implementation
        try:
            protocol.validate(document)
        except Exception as exc:
            raise QualityContractError(
                f"the ACK document failed the real "
                f"{QUALITY_ACK_SCHEMA} validator: {exc}"
            ) from exc
        schema = str(document.get("s"))
        if schema != QUALITY_ACK_SCHEMA:
            raise QualityContractError(
                f"only an evaluated {QUALITY_ACK_SCHEMA} carries scores; this "
                f"document is {schema!r}, which has no quality to bind"
            )

        # 2. the authoritative hash is recomputed from the document
        raw_sha = protocol.digest(document)
        _sha256_hex(raw_sha, "recomputed raw ACK digest", QualityContractError)
        if expected_raw_ack_sha256 is not None:
            _sha256_hex(
                expected_raw_ack_sha256,
                "expected_raw_ack_sha256",
                QualityContractError,
            )
            if expected_raw_ack_sha256 != raw_sha:
                raise QualityContractError(
                    f"the caller's expected ACK digest "
                    f"{expected_raw_ack_sha256} does not match the digest "
                    f"{raw_sha} recomputed from the document itself; the "
                    f"document is authoritative and an asserted hash is never "
                    f"accepted as proof"
                )

        identity = dict(protocol.identity_dict(document))
        quality = dict(protocol.quality_dict(document))
        if tuple(sorted(quality)) != tuple(sorted(QUALITY_ACK_QUALITY_FIELDS)):
            raise QualityContractError(
                f"the evaluated ACK must carry all "
                f"{len(QUALITY_ACK_QUALITY_FIELDS)} score fields; got "
                f"{sorted(quality)}"
            )

        # 2b. Verify the *actual* retained detail document named by ``dh``.
        # Legacy details without the Phase-4a.2 support extension deliberately
        # fail below; no caller-provided counts may fill that evidence gap.
        if not isinstance(detail_document, Mapping):
            raise QualityContractError(
                "detail_document must be the presented digest-bound mapping; "
                "the ACK's dh digest or loose support counts are not a "
                "substitute"
            )
        detail_sha = protocol.detail_digest(detail_document)
        _sha256_hex(
            detail_sha, "recomputed detail digest", QualityContractError
        )
        if detail_sha != str(document["dh"]):
            raise QualityContractError(
                f"actual detail digest {detail_sha} does not match ACK dh "
                f"{document['dh']}; score support is not bound to this ACK"
            )
        if detail_document.get("schema") != QUALITY_DETAIL_SCHEMA:
            raise QualityContractError(
                f"detail schema must be {QUALITY_DETAIL_SCHEMA!r}, got "
                f"{detail_document.get('schema')!r}"
            )
        for field_name, required in (
            ("privileged_carla_ground_truth", True),
            ("deployable_feedback", False),
            ("terminal", False),
        ):
            if detail_document.get(field_name) is not required:
                raise QualityContractError(
                    f"detail {field_name} must be {required!r}"
                )
        for name in QUALITY_ACK_IDENTITY_FIELDS:
            if detail_document.get(name) != identity[name]:
                raise QualityContractError(
                    f"detail {name}={detail_document.get(name)!r} does not "
                    f"match ACK {identity[name]!r}"
                )
        if int(detail_document.get("frozen_carla_frame_id", -1)) != int(
            identity["frame_id"]
        ):
            raise QualityContractError(
                "detail frozen_carla_frame_id does not match the ACK frame"
            )
        if detail_document.get("failure_reason") not in ("", None):
            raise QualityContractError(
                "an evaluated detail document cannot carry a failure reason"
            )
        if not isinstance(detail_document.get("timing"), Mapping) or not isinstance(
            detail_document.get("quality"), Mapping
        ):
            raise QualityContractError(
                "detail must retain its timing and nested quality mappings"
            )
        # Rebuilding the compact ACK from the actual detail proves that the
        # retained quality/timing/evaluator fields are precisely those sent on
        # the wire, not merely a different document with a matching identity.
        try:
            rebuilt_ack = protocol.build_ack(
                identity_fields={
                    name: detail_document[name]
                    for name in QUALITY_ACK_IDENTITY_FIELDS
                },
                frozen_carla_frame_id=int(
                    detail_document["frozen_carla_frame_id"]
                ),
                timing=detail_document["timing"],
                quality=detail_document["quality"],
                evaluator_mode=str(detail_document.get("evaluator_mode") or ""),
                detail_sha256=detail_sha,
            )
        except Exception as exc:
            raise QualityContractError(
                f"the retained detail cannot reproduce a valid ACK: {exc}"
            ) from exc
        if dict(rebuilt_ack) != dict(document):
            raise QualityContractError(
                "the ACK rebuilt from the retained detail differs from the "
                "wire ACK; timing, quality or evaluator provenance drifted"
            )

        # 3. against the transmission obligation
        for name, expected in (
            ("run_id", obligation.run_id),
            ("cell_id", obligation.cell_id),
            ("stream_id", obligation.stream_id),
        ):
            if str(identity[name]) != str(expected):
                raise QualityContractError(
                    f"ACK {name} {identity[name]!r} does not match the "
                    f"transmission obligation's {expected!r}; an ACK is never "
                    f"re-attributed across runs, cells or streams"
                )
        if int(identity["frame_id"]) != obligation.frame_id:
            raise QualityContractError(
                f"ACK frame_id {identity['frame_id']} does not match the "
                f"obligation's reward frame {obligation.frame_id}; one ACK can "
                f"never be reused for a different frame"
            )
        if int(identity["capture_timestamp_ns"]) != (
            obligation.capture_timestamp_ns
        ):
            raise QualityContractError(
                f"ACK capture_timestamp_ns {identity['capture_timestamp_ns']} "
                f"does not match the obligation's "
                f"{obligation.capture_timestamp_ns}"
            )
        if int(identity["action_id"]) != int(action.action_id) or str(
            identity["profile_id"]
        ) != str(action.profile_id):
            raise QualityContractError(
                f"ACK names anchor action_id={identity['action_id']} "
                f"profile_id={identity['profile_id']!r}, but the obligation "
                f"executed action_id={action.action_id} "
                f"profile_id={action.profile_id!r}; measured quality is never "
                f"re-attributed between actions"
            )

        # 4. against the controller's own completed ticket
        if obligation.session_uuid != completed_ticket.session_uuid:
            raise QualityContractError(
                f"obligation session {obligation.session_uuid} does not match "
                f"the ticket's {completed_ticket.session_uuid}"
            )
        if obligation.decision_seq != completed_ticket.decision_seq:
            raise QualityContractError(
                f"obligation decision {obligation.decision_seq} does not match "
                f"the ticket's {completed_ticket.decision_seq}; one ACK can "
                f"never be reused for a different decision"
            )
        if obligation.reward_tensor_seq != completed_ticket.reward_tensor_seq:
            raise QualityContractError(
                f"obligation reward tensor {obligation.reward_tensor_seq} is "
                f"not the ticket's reward-requested tensor "
                f"{completed_ticket.reward_tensor_seq}"
            )
        if obligation.frame_id != completed_ticket.reward_carla_frame_id:
            raise QualityContractError(
                f"obligation frame {obligation.frame_id} is not the ticket's "
                f"reward-tensor frame "
                f"{completed_ticket.reward_carla_frame_id}"
            )
        if obligation.executed_action != completed_ticket.action:
            raise QualityContractError(
                "the obligation's executed action differs from the action the "
                "hold actually executed"
            )

        eligibility_result = EvaluationEligibilityResultV1._from_verified_support(
            detail_document.get(QUALITY_DETAIL_SUPPORT_KEY),
            eligibility=eligibility,
            obligation=obligation,
            detailed_evidence_sha256=detail_sha,
            ack_identity=identity,
            ack_quality=quality,
        )

        obligation_sha = obligation.canonical_sha256()
        ticket_sha = completed_ticket.canonical_sha256()
        # Claim only after *all* validation has succeeded.  The operation is
        # idempotent for this exact binding and rejects reuse for another one.
        _PROCESS_QUALITY_ACK_USE_REGISTRY.claim(
            raw_sha,
            obligation_sha256=obligation_sha,
            completed_ticket_sha256=ticket_sha,
        )

        record = cls(
            raw_quality_ack_sha256=raw_sha,
            detailed_evidence_sha256=detail_sha,
            ack_schema=schema,
            ack_protocol_version=int(document["v"]),
            ack_source=str(document["src"]),
            identity_fields=identity,
            quality_fields=quality,
            evaluator_mode=str(document.get("m") or ""),
            obligation_sha256=obligation_sha,
            completed_ticket_sha256=ticket_sha,
            eligibility_result=eligibility_result,
            ack_use_registry_id=(
                _PROCESS_QUALITY_ACK_USE_REGISTRY.registry_id
            ),
        )
        return replace(
            record, _attestation=_issue_ack(record._binding())
        )

    # -- accessors --------------------------------------------------------- #

    @property
    def action_id(self) -> int:
        return int(self.identity_fields["action_id"])

    @property
    def profile_id(self) -> str:
        return str(self.identity_fields["profile_id"])

    @property
    def frame_id(self) -> int:
        return int(self.identity_fields["frame_id"])

    def ack_quality_field(self, name: str) -> Any:
        """Read one retained raw ACK score field by its registered name."""
        if name not in QUALITY_ACK_QUALITY_FIELDS:
            raise QualityContractError(
                f"{name!r} is not one of the registered v1 score fields"
            )
        return self.quality_fields[name]

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "ack_protocol_version": self.ack_protocol_version,
            "ack_schema": self.ack_schema,
            "ack_source": self.ack_source,
            "anchor_only": QUALITY_ACK_IS_ANCHOR_ONLY,
            "ack_use_registry_id": self.ack_use_registry_id,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "evaluator_mode": self.evaluator_mode,
            "eligibility_result": self.eligibility_result.to_canonical_dict(),
            "false_positive_counts_available": (
                QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE
            ),
            "identity_fields": dict(self.identity_fields),
            "obligation_sha256": self.obligation_sha256,
            "quality_fields": dict(self.quality_fields),
            "quality_protocol_contract_sha256": (
                QUALITY_PROTOCOL_CONTRACT_SHA256
            ),
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "record": "quality_ack_binding_v1",
            "timing_clock_domain": QUALITY_ACK_TIMING_CLOCK_DOMAIN,
        }


@dataclass(frozen=True, slots=True)
class QualityEvidenceV1:
    """Provenance of one quality measurement, keyed on the full action identity.

    The core key is ``executed_action_sha256``, the canonical hash of the whole
    :class:`~.transaction_identity.ExecutedActionIdentity`.  That is deliberate:
    it is well defined for every action, on-anchor or not, whereas the v1 ACK's
    anchor fields are not.

    ``kind`` and ``granularity`` are mandatory so aggregate campaign evidence
    can never be mistaken for a per-frame causal reward, and ``eligibility``
    records the hash-bound per-UE rule the masks were produced under.
    """

    gt_source: GroundTruthSource
    kind: EvidenceKind
    granularity: EvidenceGranularity
    eligibility: EvaluationEligibilityV1
    executed_action_sha256: str
    gt_source_detail: str
    ack_binding: Optional[QualityAckBindingV1] = None

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.gt_source, GroundTruthSource):
            raise E(
                f"gt_source must be a GroundTruthSource (the privileged, "
                f"non-deployable oracle label), got "
                f"{type(self.gt_source).__name__}: {self.gt_source!r}"
            )
        if not isinstance(self.kind, EvidenceKind):
            raise E(
                f"kind must be an EvidenceKind so aggregate evidence cannot "
                f"masquerade as per-frame evidence, got "
                f"{type(self.kind).__name__}: {self.kind!r}"
            )
        if not isinstance(self.granularity, EvidenceGranularity):
            raise E(
                f"granularity must be an EvidenceGranularity, got "
                f"{type(self.granularity).__name__}: {self.granularity!r}"
            )
        if not isinstance(self.eligibility, EvaluationEligibilityV1):
            raise E(
                f"eligibility must be an EvaluationEligibilityV1 so the "
                f"per-UE range/FoV/visibility rule is always on the record, "
                f"got {type(self.eligibility).__name__}"
            )
        _sha256_hex(self.executed_action_sha256, "executed_action_sha256", E)
        _non_empty_str(self.gt_source_detail, "gt_source_detail", E)
        if self.ack_binding is not None:
            if not isinstance(self.ack_binding, QualityAckBindingV1):
                raise E(
                    f"ack_binding must be a QualityAckBindingV1 or None, got "
                    f"{type(self.ack_binding).__name__}"
                )
            self.ack_binding.require_attested()
            if self.ack_binding.eligibility_result.eligibility != self.eligibility:
                raise E(
                    "evidence eligibility differs from the eligibility rule "
                    "actually proven by the ACK-bound detail document"
                )
            if (
                self.executed_action_sha256
                != self.ack_binding.eligibility_result.executed_action_sha256
            ):
                raise E(
                    "evidence executed_action_sha256 differs from the exact "
                    "action identity bound by the ACK detail/obligation"
                )
        # A per-frame causal ACK claim requires an actual verified document.
        if self.is_causal_per_frame and self.ack_binding is None:
            raise E(
                f"evidence claiming {EvidenceKind.PER_FRAME_CAUSAL_ACK.value} "
                f"at {EvidenceGranularity.SINGLE_FRAME.value} must carry a "
                f"verified ACK binding; an unevidenced claim of per-frame "
                f"causality is refused"
            )
        if (
            self.kind is EvidenceKind.AGGREGATE_PROFILE_CAMPAIGN
            and self.granularity is not EvidenceGranularity.PROFILE_AGGREGATE
        ):
            raise E(
                "aggregate campaign evidence is only valid at profile-aggregate "
                "granularity"
            )

    @classmethod
    def for_verified_ack(
        cls,
        ack_binding: QualityAckBindingV1,
        *,
        obligation: QualityAckObligationV1,
        eligibility: EvaluationEligibilityV1,
        gt_source_detail: str,
        gt_source: GroundTruthSource = GroundTruthSource.CARLA_GT_EXACT,
    ) -> "QualityEvidenceV1":
        """Build per-frame causal evidence from an already-verified ACK binding."""
        if not isinstance(ack_binding, QualityAckBindingV1):
            raise QualityContractError(
                f"ack_binding must be a QualityAckBindingV1, got "
                f"{type(ack_binding).__name__}"
            )
        ack_binding.require_attested()
        if not isinstance(obligation, QualityAckObligationV1):
            raise QualityContractError(
                f"obligation must be a QualityAckObligationV1, got "
                f"{type(obligation).__name__}"
            )
        if ack_binding.obligation_sha256 != obligation.canonical_sha256():
            raise QualityContractError(
                "the ACK binding was verified against a different obligation "
                "than the one supplied here"
            )
        if ack_binding.eligibility_result.eligibility != eligibility:
            raise QualityContractError(
                "the supplied eligibility rule differs from the one proven "
                "inside the ACK-bound detail document"
            )
        return cls(
            gt_source=gt_source,
            kind=EvidenceKind.PER_FRAME_CAUSAL_ACK,
            granularity=EvidenceGranularity.SINGLE_FRAME,
            eligibility=eligibility,
            executed_action_sha256=obligation.executed_action_sha256,
            gt_source_detail=gt_source_detail,
            ack_binding=ack_binding,
        )

    @property
    def is_causal_per_frame(self) -> bool:
        """True only for the exact evidence pair a per-frame reward requires."""
        return (self.kind, self.granularity) == CAUSAL_REWARD_EVIDENCE

    def require_causal_per_frame(self) -> None:
        """Fail closed unless this is per-frame causal evidence."""
        if not self.is_causal_per_frame:
            raise EvidenceGranularityError(
                f"a per-frame causal reward requires "
                f"{CAUSAL_REWARD_EVIDENCE[0].value} evidence at "
                f"{CAUSAL_REWARD_EVIDENCE[1].value} granularity; this record "
                f"is {self.kind.value} at {self.granularity.value}.  Aggregate "
                f"72-action campaign evidence is real evidence about "
                f"profile-level payload, delivery and quality, but it is an "
                f"action average rather than a causal per-decision transition "
                f"and must never stand in for one"
            )

    @property
    def learning_ready(self) -> bool:
        return (
            self.ack_binding is not None
            and self.ack_binding.eligibility_result.learning_ready
        )

    def require_learning_ready(self) -> None:
        """Require both causal granularity and source-authenticated production."""
        self.require_causal_per_frame()
        if self.ack_binding is None:  # defensive; constructor already refuses it
            raise QualityProducerUnavailableError(
                "per-frame quality evidence has no verified ACK binding"
            )
        self.ack_binding.eligibility_result.require_learning_ready()

    @property
    def raw_quality_ack_sha256(self) -> Optional[str]:
        return (
            None if self.ack_binding is None
            else self.ack_binding.raw_quality_ack_sha256
        )

    @property
    def detailed_evidence_sha256(self) -> Optional[str]:
        return (
            None if self.ack_binding is None
            else self.ack_binding.detailed_evidence_sha256
        )

    @property
    def supports_detection_f1_claim(self) -> bool:
        """Always False: the v1 ACK reports TP and FN but no false positives."""
        return QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "ack_binding": (
                None
                if self.ack_binding is None
                else self.ack_binding.to_canonical_dict()
            ),
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "eligibility": self.eligibility.to_canonical_dict(),
            "evidence_granularity": self.granularity.value,
            "evidence_kind": self.kind.value,
            "executed_action_sha256": self.executed_action_sha256,
            "gt_deployable": self.gt_source.deployable,
            "gt_privileged": self.gt_source.privileged,
            "gt_source": self.gt_source.value,
            "gt_source_detail": self.gt_source_detail,
            "is_causal_per_frame": self.is_causal_per_frame,
            "learning_ready": self.learning_ready,
            "protocol_v2_requirement": PROTOCOL_V2_REQUIREMENT,
            "quality_protocol_contract_sha256": (
                QUALITY_PROTOCOL_CONTRACT_SHA256
            ),
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "record": "quality_evidence_v1",
            "reward_scope": self.eligibility.reward_scope.value,
            "supports_detection_f1_claim": self.supports_detection_f1_claim,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# B. Raw quality components and the corrected quality formulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _ClassLocalization:
    """One class's localization inputs, judged over the eligible GT set."""

    name: str
    eligible_gt_instances: int
    tp: int
    fn: int
    xy_error_m: Optional[float]
    footprint_iou: Optional[float]

    def __post_init__(self) -> None:
        E = QualityContractError
        _non_empty_str(self.name, "class name", E)
        _non_negative_int(
            self.eligible_gt_instances, f"{self.name}_eligible_gt_instances", E
        )
        _non_negative_int(self.tp, f"{self.name}_tp", E)
        _non_negative_int(self.fn, f"{self.name}_fn", E)
        if self.tp + self.fn != self.eligible_gt_instances:
            raise E(
                f"{self.name}: tp + fn ({self.tp} + {self.fn}) must equal the "
                f"eligible ground-truth count "
                f"{self.eligible_gt_instances}, because recall is defined over "
                f"the eligible set only.  Ineligible objects -- out of range, "
                f"outside the field of view, or occluded under the declared "
                f"visibility rule -- were never this UE's to detect and must "
                f"not enter the denominator"
            )
        if self.xy_error_m is not None:
            value = _finite_float(self.xy_error_m, f"{self.name}_xy_error_m", E)
            if value < 0.0:
                raise E(f"{self.name}_xy_error_m must be >= 0, got {value}")
        if self.footprint_iou is not None:
            _finite_in(
                self.footprint_iou, f"{self.name}_footprint_iou", 0.0, 1.0, E
            )

        if self.is_defined and self.tp > 0 and self.xy_error_m is None:
            raise UndefinedClassSupportError(
                f"{self.name}: recall is {self.recall} > 0, so a finite "
                f"non-negative matched XY error is required; it is absent"
            )
        if self.is_defined and self.tp == 0 and self.xy_error_m is not None:
            raise QualityContractError(
                f"{self.name}: tp is 0, so there is no matched object and no "
                f"matched XY error can exist; got {self.xy_error_m!r}"
            )

    @property
    def is_defined(self) -> bool:
        """True when this UE had at least one eligible object to find.

        Zero eligible objects means the class is undefined for this frame and is
        excluded.  Raw ground-truth presence alone never makes it defined.
        """
        return self.eligible_gt_instances > 0

    @property
    def recall(self) -> Optional[float]:
        """``tp / eligible_gt``, or ``None`` when the class is undefined."""
        if not self.is_defined:
            return None
        return float(self.tp) / float(self.eligible_gt_instances)

    def utility(self, tau_m: float) -> Optional[float]:
        """``U_loc = sqrt(recall * exp(-e / tau))``, or ``None`` if undefined.

        A complete miss of eligible objects gives ``recall = 0`` and therefore
        ``U_loc = 0``: the class stays in the combination with zero utility
        rather than being renormalized away.
        """
        if not self.is_defined:
            return None
        recall = self.recall
        assert recall is not None
        if self.tp == 0:
            # Every eligible object was missed.  No matched error exists, and
            # the zero coverage term is what the utility must reflect.
            return 0.0
        u_xy = math.exp(-float(self.xy_error_m) / float(tau_m))
        value = math.sqrt(recall * u_xy)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualityContractError(  # pragma: no cover - defensive
                f"{self.name}: U_loc produced invalid value {value!r}"
            )
        return value

    def xy_utility(self, tau_m: float) -> Optional[float]:
        """``U_xy = exp(-e / tau)``, retained separately as a diagnostic."""
        if not self.is_defined or self.tp == 0:
            return None
        return math.exp(-float(self.xy_error_m) / float(tau_m))

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "eligible_gt_instances": self.eligible_gt_instances,
            "fn": self.fn,
            "footprint_iou": (
                None if self.footprint_iou is None else float(self.footprint_iou)
            ),
            "is_defined": self.is_defined,
            "recall": None if self.recall is None else float(self.recall),
            "tp": self.tp,
            "xy_error_m": (
                None if self.xy_error_m is None else float(self.xy_error_m)
            ),
        }


@dataclass(frozen=True, slots=True)
class _ClassSegmentation:
    """One class's segmentation inputs and its empty-empty exclusion rule."""

    name: str
    iou: Optional[float]
    gt_pixels: int
    pred_pixels: int

    def __post_init__(self) -> None:
        E = QualityContractError
        _non_empty_str(self.name, "class name", E)
        _non_negative_int(self.gt_pixels, f"gt_{self.name}_pixels", E)
        _non_negative_int(self.pred_pixels, f"pred_{self.name}_pixels", E)
        if self.is_defined:
            if self.iou is None:
                raise E(
                    f"{self.name} segmentation is defined (gt_pixels="
                    f"{self.gt_pixels}, pred_pixels={self.pred_pixels}), so an "
                    f"IoU in [0, 1] is required"
                )
            _finite_in(self.iou, f"seg_{self.name}_iou", 0.0, 1.0, E)
        elif self.iou not in (None, 0.0):
            raise E(
                f"{self.name} segmentation has both masks empty, so it is "
                f"excluded and cannot carry IoU {self.iou!r}"
            )

    @property
    def is_defined(self) -> bool:
        """Excluded **only** when both the predicted and GT masks are empty.

        So a false positive against absent (eligibility-masked) ground truth is
        defined with IoU 0 and is penalized, and a missed mask against present
        ground truth is likewise defined with IoU 0 and penalized.  Only the
        genuinely vacuous case -- nothing there and nothing predicted -- drops
        out.
        """
        return self.gt_pixels > 0 or self.pred_pixels > 0

    @property
    def exclusion_reason(self) -> Optional[str]:
        return None if self.is_defined else "both_masks_empty"

    def normalized(self, reference_iou: float) -> Optional[float]:
        """``s = clip(IoU / reference, 0, 1)`` against the frozen reference."""
        if not self.is_defined:
            return None
        value = float(self.iou) / float(reference_iou)
        return min(max(value, 0.0), 1.0)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "exclusion_reason": self.exclusion_reason,
            "gt_pixels": self.gt_pixels,
            "iou": None if self.iou is None else float(self.iou),
            "is_defined": self.is_defined,
            "pred_pixels": self.pred_pixels,
        }


@dataclass(frozen=True, slots=True)
class QualityComponentsV1(_Attested):
    """Raw per-frame quality inputs, carrying every available ACK field.

    Segmentation: per-class IoU with GT **and predicted** pixel counts.  The
    predicted count is not on the v1 wire (see
    :data:`QUALITY_ACK_MISSING_REQUIRED_FIELDS`) and must come from the
    edge-retained detailed row that the ACK hash-binds through ``dh`` -- it is
    the only thing that distinguishes an excluded empty-empty class from a
    penalized false positive.

    Localization: per-class eligible GT instances, TP, FN, matched XY error and
    footprint IoU.  Recall is computed over the **eligible** GT set from the
    hash-bound :class:`EvaluationEligibilityV1`, so ground-truth presence alone
    never creates a miss penalty.

    ``seg_miou_3class`` and the footprint IoUs are preserved as diagnostics and
    take no part in the scalar.  No precision, F1 or AP is claimed or derivable:
    the v1 evidence has no false-positive count.
    """

    evidence: QualityEvidenceV1
    vehicle_segmentation: _ClassSegmentation
    person_segmentation: _ClassSegmentation
    vehicle_localization: _ClassLocalization
    person_localization: _ClassLocalization
    seg_miou_3class: Optional[float] = None
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.evidence, QualityEvidenceV1):
            raise E(
                f"evidence must be a QualityEvidenceV1 carrying the privileged "
                f"ground-truth label, the evidence kind/granularity, the "
                f"per-UE eligibility rule and the executed-action hash, got "
                f"{type(self.evidence).__name__}"
            )
        for name, expected in (
            ("vehicle_segmentation", _ClassSegmentation),
            ("person_segmentation", _ClassSegmentation),
            ("vehicle_localization", _ClassLocalization),
            ("person_localization", _ClassLocalization),
        ):
            if not isinstance(getattr(self, name), expected):
                raise E(
                    f"{name} must be a {expected.__name__}, got "
                    f"{type(getattr(self, name)).__name__}"
                )
        if self.seg_miou_3class is not None:
            _finite_in(self.seg_miou_3class, "seg_miou_3class", 0.0, 1.0, E)
        if self._attestation is not None and not _valid_components(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the quality components' attestation is not bound to their "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_components

    # -- construction from the verified ACK plus the detailed supplement --- #

    @classmethod
    def from_ack_binding(
        cls,
        evidence: QualityEvidenceV1,
    ) -> "QualityComponentsV1":
        """Read the raw score fields out of the verified ACK binding.

        Every field absent from the v1 wire is taken from the attested
        :class:`EvaluationEligibilityResultV1` parsed out of the *actual*
        ``dh``-bound detail document.  There are deliberately no loose count
        arguments: legacy v1 details that cannot prove them fail before this
        factory is reached.
        """
        if not isinstance(evidence, QualityEvidenceV1):
            raise QualityContractError(
                f"evidence must be a QualityEvidenceV1, got "
                f"{type(evidence).__name__}"
            )
        binding = evidence.ack_binding
        if binding is None:
            raise QualityContractError(
                "raw components can only be read from a verified ACK binding; "
                "this evidence carries none"
            )
        binding.require_attested()
        support = binding.eligibility_result
        support.require_attested()
        get = binding.ack_quality_field

        def _opt_float(value: Any, name: str) -> Optional[float]:
            if value is None:
                return None
            return _finite_float(value, name, QualityContractError)

        def _req_int(value: Any, name: str) -> int:
            if value is None:
                raise QualityContractError(f"the ACK lacks {name}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise QualityContractError(
                    f"{name} must be an exact int in the ACK, got {value!r}"
                )
            return int(value)

        record = cls(
            evidence=evidence,
            vehicle_segmentation=_ClassSegmentation(
                name="vehicle",
                iou=_opt_float(get("seg_vehicle_iou"), "seg_vehicle_iou"),
                gt_pixels=_req_int(get("gt_vehicle_pixels"), "gt_vehicle_pixels"),
                pred_pixels=support.seg_vehicle_pred_pixels,
            ),
            person_segmentation=_ClassSegmentation(
                name="person",
                iou=_opt_float(get("seg_person_iou"), "seg_person_iou"),
                gt_pixels=_req_int(get("gt_person_pixels"), "gt_person_pixels"),
                pred_pixels=support.seg_person_pred_pixels,
            ),
            vehicle_localization=_ClassLocalization(
                name="vehicle",
                eligible_gt_instances=(
                    support.vehicle_eligible_gt_instances
                ),
                tp=_req_int(get("vehicle_tp"), "vehicle_tp"),
                fn=_req_int(get("vehicle_fn"), "vehicle_fn"),
                xy_error_m=_opt_float(
                    get("vehicle_xy_error_m"), "vehicle_xy_error_m"
                ),
                footprint_iou=_opt_float(
                    get("vehicle_footprint_iou"), "vehicle_footprint_iou"
                ),
            ),
            person_localization=_ClassLocalization(
                name="person",
                eligible_gt_instances=support.person_eligible_gt_instances,
                tp=_req_int(get("person_tp"), "person_tp"),
                fn=_req_int(get("person_fn"), "person_fn"),
                xy_error_m=_opt_float(
                    get("person_xy_error_m"), "person_xy_error_m"
                ),
                footprint_iou=_opt_float(
                    get("person_footprint_iou"), "person_footprint_iou"
                ),
            ),
            seg_miou_3class=_opt_float(get("seg_miou_3class"), "seg_miou_3class"),
        )
        return replace(
            record, _attestation=_issue_components(record._binding())
        )

    # -- derived masks ----------------------------------------------------- #

    @property
    def eligibility(self) -> EvaluationEligibilityV1:
        return self.evidence.eligibility

    @property
    def gt_source(self) -> GroundTruthSource:
        return self.evidence.gt_source

    @property
    def supports_detection_f1_claim(self) -> bool:
        """Always False: the v1 evidence has no false-positive counts."""
        return self.evidence.supports_detection_f1_claim

    @property
    def defined_localization_classes(self) -> Tuple[str, ...]:
        return tuple(
            component.name
            for component in (self.vehicle_localization, self.person_localization)
            if component.is_defined
        )

    @property
    def defined_segmentation_classes(self) -> Tuple[str, ...]:
        return tuple(
            component.name
            for component in (self.vehicle_segmentation, self.person_segmentation)
            if component.is_defined
        )

    @property
    def undefined_localization_classes(self) -> Tuple[str, ...]:
        """Classes with no eligible ground truth; excluded, never penalized."""
        return tuple(
            component.name
            for component in (self.vehicle_localization, self.person_localization)
            if not component.is_defined
        )

    @property
    def missed_localization_classes(self) -> Tuple[str, ...]:
        """Classes with eligible ground truth that were completely missed."""
        return tuple(
            component.name
            for component in (self.vehicle_localization, self.person_localization)
            if component.is_defined and component.tp == 0
        )

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "defined_localization_classes": list(
                self.defined_localization_classes
            ),
            "defined_segmentation_classes": list(
                self.defined_segmentation_classes
            ),
            "evidence": self.evidence.to_canonical_dict(),
            "missed_localization_classes": list(self.missed_localization_classes),
            "person_localization": self.person_localization.to_canonical_dict(),
            "person_segmentation": self.person_segmentation.to_canonical_dict(),
            "record": "quality_components_v1",
            "seg_miou_3class": (
                None if self.seg_miou_3class is None
                else float(self.seg_miou_3class)
            ),
            "supports_detection_f1_claim": self.supports_detection_f1_claim,
            "undefined_localization_classes": list(
                self.undefined_localization_classes
            ),
            "vehicle_localization": self.vehicle_localization.to_canonical_dict(),
            "vehicle_segmentation": self.vehicle_segmentation.to_canonical_dict(),
        }

class LocalizationCombiner(Enum):
    """How per-class localization utilities combine into ``Q_loc``.

    **Unresolved scientific choice, therefore explicit and required.**  The
    registered formulation fixes the *segmentation* combiner as a weighted
    geometric mean but does not fix this one, and the two options differ in
    exactly the way that matters for a safety-relevant class:

    * ``WEIGHTED_GEOMETRIC_MEAN`` -- a single completely missed eligible class
      drives ``Q_loc`` to zero, so a missed pedestrian cannot be averaged away
      by a well-localized vehicle.
    * ``WEIGHTED_ARITHMETIC_MEAN`` -- a missed class contributes zero utility at
      its own weight, keeping it in the average without collapsing the whole
      term.

    Both keep a missed eligible class *in* the combination, which is what the
    contract requires.  Neither is defaulted: the caller must choose and the
    choice is hashed into the reward spec.
    """

    WEIGHTED_GEOMETRIC_MEAN = "WEIGHTED_GEOMETRIC_MEAN"
    WEIGHTED_ARITHMETIC_MEAN = "WEIGHTED_ARITHMETIC_MEAN"


@dataclass(frozen=True, slots=True)
class QualityEvaluationV1(_Attested):
    """The derived perception quality, with every raw input preserved beside it.

    Build with :meth:`RewardSpecV1.evaluate_quality`; a directly constructed
    instance is unattested and cannot serialize or enter a transition.
    """

    components: QualityComponentsV1
    q_seg: Optional[float]
    q_loc: float
    q_perc: float
    per_class_localization_utility: Mapping[str, float]
    per_class_xy_utility: Mapping[str, Optional[float]]
    per_class_recall: Mapping[str, float]
    per_class_normalized_segmentation: Mapping[str, float]
    segmentation_weights_used: Mapping[str, float]
    localization_weights_used: Mapping[str, float]
    localization_combiner: LocalizationCombiner
    reward_spec_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.components, QualityComponentsV1):
            raise E(
                f"components must be a QualityComponentsV1, got "
                f"{type(self.components).__name__}"
            )
        if self.q_seg is not None:
            _finite_in(self.q_seg, "q_seg", 0.0, 1.0, E)
        _finite_in(self.q_loc, "q_loc", 0.0, 1.0, E)
        _finite_in(self.q_perc, "q_perc", 0.0, 1.0, E)
        if not isinstance(self.localization_combiner, LocalizationCombiner):
            raise E("localization_combiner must be a LocalizationCombiner")
        for name in (
            "per_class_localization_utility",
            "per_class_xy_utility",
            "per_class_recall",
            "per_class_normalized_segmentation",
            "segmentation_weights_used",
            "localization_weights_used",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise E(f"{name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        if not self.localization_weights_used:
            raise E(
                "localization_weights_used cannot be empty: Q_loc is the base "
                "of the perception quality and requires at least one defined "
                "eligible class"
            )
        for label, weights in (
            ("segmentation", self.segmentation_weights_used),
            ("localization", self.localization_weights_used),
        ):
            for klass, weight in weights.items():
                as_float = _finite_float(weight, f"{label} weight {klass}", E)
                if as_float <= 0.0:
                    raise E(
                        f"{label} weight for {klass!r} must be > 0, got "
                        f"{as_float}; a zero or negative weight would silently "
                        f"delete a class from the measurement"
                    )
        _sha256_hex(self.reward_spec_sha256, "reward_spec_sha256", E)
        if self._attestation is not None and not _valid_quality(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the quality evaluation's attestation does not match its own "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_quality

    @property
    def quality(self) -> float:
        """Alias for :attr:`q_perc`, the scalar the reward consumes."""
        return self.q_perc

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "components": self.components.to_canonical_dict(),
            "localization_combiner": self.localization_combiner.value,
            "localization_weights_used": dict(self.localization_weights_used),
            "per_class_localization_utility": dict(
                self.per_class_localization_utility
            ),
            "per_class_normalized_segmentation": dict(
                self.per_class_normalized_segmentation
            ),
            "per_class_recall": dict(self.per_class_recall),
            "per_class_xy_utility": dict(self.per_class_xy_utility),
            "q_loc": float(self.q_loc),
            "q_perc": float(self.q_perc),
            "q_seg": None if self.q_seg is None else float(self.q_seg),
            "record": "quality_evaluation_v1",
            "reward_spec_sha256": self.reward_spec_sha256,
            "segmentation_weights_used": dict(self.segmentation_weights_used),
        }


# --------------------------------------------------------------------------- #
# C. Reward specification, latency and separated cost signals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RewardSpecV1:
    """Every scientific reward value, all constructor supplied.

    There is **no production default** for any weight, reference, time constant,
    mixing coefficient, switch penalty or discount: each is a calibration
    hypothesis, and baking one in would present an unmeasured choice as a frozen
    contract.

    The perception quality is localization-based:

    .. code-block:: text

        U_xy,c  = exp(-e_c / tau_c)
        U_loc,c = sqrt(recall_c * U_xy,c)
        Q_loc   = combine(U_loc,c ; w_loc,c)        # combiner is explicit
        s_c     = clip(IoU_c / reference_c, 0, 1)
        Q_seg   = weighted geometric mean(s_c ; w_seg,c)
        Q_perc  = Q_loc * ((1 - beta) + beta * Q_seg)

    Segmentation therefore *modulates* rather than substitutes: ``Q_loc = 0``
    forces ``Q_perc = 0`` for any ``beta``, so strong segmentation can never
    rescue collapsed localization.  ``beta`` is supplied explicitly and is meant
    to be sensitivity-tested, not trusted.

    The scalar reward is

    .. code-block:: text

        r = w_quality * Q_perc
            - w_latency * (L / B)
            - lambda_mode * 1[mode changed]
            - lambda_q * |q_exec,t - q_exec,t-1|
    """

    spec_id: str
    spec_version: int
    # -- localization ------------------------------------------------------ #
    w_loc_person: float
    w_loc_vehicle: float
    tau_person_m: float
    tau_vehicle_m: float
    localization_combiner: LocalizationCombiner
    # -- segmentation ------------------------------------------------------ #
    w_seg_person: float
    w_seg_vehicle: float
    seg_reference_person_iou: float
    seg_reference_vehicle_iou: float
    segmentation_modulation_beta: float
    # -- scalar reward ----------------------------------------------------- #
    w_quality: float
    w_latency: float
    lambda_mode: float
    lambda_q: float
    r_registered_failure: float
    gamma_per_tensor: float
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        E = RewardSpecError
        _non_empty_str(self.spec_id, "spec_id", E)
        _positive_int(self.spec_version, "spec_version", E)
        for name in (
            "w_loc_person",
            "w_loc_vehicle",
            "tau_person_m",
            "tau_vehicle_m",
            "w_seg_person",
            "w_seg_vehicle",
            "w_quality",
        ):
            value = _finite_float(getattr(self, name), name, E)
            if value <= 0.0:
                raise E(f"{name} must be > 0, got {value}")
        if not isinstance(self.localization_combiner, LocalizationCombiner):
            raise E(
                f"localization_combiner must be an explicit "
                f"LocalizationCombiner -- the registered formulation does not "
                f"fix it, so it must be chosen deliberately -- got "
                f"{type(self.localization_combiner).__name__}: "
                f"{self.localization_combiner!r}"
            )
        for name in ("seg_reference_person_iou", "seg_reference_vehicle_iou"):
            value = _finite_float(getattr(self, name), name, E)
            if not 0.0 < value <= 1.0:
                raise E(
                    f"{name} is a frozen normalization reference IoU and must "
                    f"lie in (0, 1], got {value}"
                )
        beta = _finite_float(
            self.segmentation_modulation_beta, "segmentation_modulation_beta", E
        )
        if not 0.0 <= beta <= 1.0:
            raise E(
                f"segmentation_modulation_beta must lie in [0, 1]; it scales "
                f"the multiplicative segmentation modulation "
                f"((1 - beta) + beta * Q_seg), got {beta}"
            )
        for name in ("w_latency", "lambda_mode", "lambda_q"):
            value = _finite_float(getattr(self, name), name, E)
            if value < 0.0:
                raise E(f"{name} must be >= 0, got {value}")
        failure = _finite_float(self.r_registered_failure, "r_registered_failure", E)
        if failure > 0.0:
            raise E(
                f"r_registered_failure is the registered *negative* service "
                f"outcome and must be <= 0, got {failure}"
            )
        gamma = _finite_float(self.gamma_per_tensor, "gamma_per_tensor", E)
        if not 0.0 < gamma <= 1.0:
            raise E(f"gamma_per_tensor must lie in (0, 1], got {gamma}")
        object.__setattr__(
            self, "provenance", _frozen_str_mapping(self.provenance, "provenance", E)
        )

    # -- helpers ----------------------------------------------------------- #

    def tau_for(self, class_name: str) -> float:
        if class_name == "person":
            return float(self.tau_person_m)
        if class_name == "vehicle":
            return float(self.tau_vehicle_m)
        raise RewardSpecError(f"no tau registered for class {class_name!r}")

    def loc_weight_for(self, class_name: str) -> float:
        if class_name == "person":
            return float(self.w_loc_person)
        if class_name == "vehicle":
            return float(self.w_loc_vehicle)
        raise RewardSpecError(
            f"no localization weight registered for class {class_name!r}"
        )

    def seg_weight_for(self, class_name: str) -> float:
        if class_name == "person":
            return float(self.w_seg_person)
        if class_name == "vehicle":
            return float(self.w_seg_vehicle)
        raise RewardSpecError(
            f"no segmentation weight registered for class {class_name!r}"
        )

    def seg_reference_for(self, class_name: str) -> float:
        if class_name == "person":
            return float(self.seg_reference_person_iou)
        if class_name == "vehicle":
            return float(self.seg_reference_vehicle_iou)
        raise RewardSpecError(
            f"no segmentation reference registered for class {class_name!r}"
        )

    def normalize_localization(self, error_m: float, tau_m: float) -> float:
        """``U_xy = exp(-error_m / tau_m)``; lower error is strictly higher."""
        value = math.exp(-float(error_m) / float(tau_m))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualityContractError(  # pragma: no cover - defensive
                f"normalized localization produced invalid value {value!r}"
            )
        return value

    @staticmethod
    def _weighted_geometric_mean(terms: Mapping[str, Tuple[float, float]]) -> float:
        """``exp(sum(w ln v) / sum(w))``, with a zero term forcing zero."""
        if any(value <= 0.0 for _, value in terms.values()):
            return 0.0
        total = sum(weight for weight, _ in terms.values())
        accumulated = sum(
            weight * math.log(value) for weight, value in terms.values()
        )
        return math.exp(accumulated / total)

    @staticmethod
    def _weighted_arithmetic_mean(terms: Mapping[str, Tuple[float, float]]) -> float:
        total = sum(weight for weight, _ in terms.values())
        return sum(weight * value for weight, value in terms.values()) / total

    # -- quality ----------------------------------------------------------- #

    def evaluate_quality(
        self, components: QualityComponentsV1
    ) -> QualityEvaluationV1:
        """Compute ``Q_perc`` and attest the derivation.

        Raises:
            InsufficientQualitySupportError: if no localization class has any
                eligible ground truth, since ``Q_loc`` is the base of the
                quality and has no defined value then.
            EvidenceGranularityError: if the evidence is not per-frame causal.
        """
        if not isinstance(components, QualityComponentsV1):
            raise QualityContractError(
                f"components must be a QualityComponentsV1, got "
                f"{type(components).__name__}"
            )
        components.require_attested()
        components.evidence.require_causal_per_frame()

        # -- localization over the eligible GT set ------------------------- #
        loc_terms: Dict[str, Tuple[float, float]] = {}
        utilities: Dict[str, float] = {}
        xy_utilities: Dict[str, Optional[float]] = {}
        recalls: Dict[str, float] = {}
        for component in (
            components.person_localization,
            components.vehicle_localization,
        ):
            utility = component.utility(self.tau_for(component.name))
            if utility is None:
                # No eligible ground truth: undefined and excluded.  Raw GT
                # presence alone never puts a class here.
                continue
            weight = self.loc_weight_for(component.name)
            loc_terms[component.name] = (weight, utility)
            utilities[component.name] = utility
            recalls[component.name] = float(component.recall)
            xy_utilities[component.name] = component.xy_utility(
                self.tau_for(component.name)
            )
        if not loc_terms:
            raise InsufficientQualitySupportError(
                f"no localization class has eligible ground truth under "
                f"eligibility contract "
                f"{components.eligibility.eligibility_contract_id!r}, so Q_loc "
                f"-- the base of the perception quality -- is undefined.  This "
                f"frame carries no per-UE perception reward; it is not a "
                f"zero-quality frame"
            )
        if self.localization_combiner is (
            LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN
        ):
            q_loc = self._weighted_geometric_mean(loc_terms)
        else:
            q_loc = self._weighted_arithmetic_mean(loc_terms)

        # -- segmentation, excluded only when both masks are empty --------- #
        seg_terms: Dict[str, Tuple[float, float]] = {}
        normalized: Dict[str, float] = {}
        for component in (
            components.person_segmentation,
            components.vehicle_segmentation,
        ):
            value = component.normalized(self.seg_reference_for(component.name))
            if value is None:
                continue
            weight = self.seg_weight_for(component.name)
            seg_terms[component.name] = (weight, value)
            normalized[component.name] = value
        if seg_terms:
            q_seg: Optional[float] = self._weighted_geometric_mean(seg_terms)
            modulation = (1.0 - float(self.segmentation_modulation_beta)) + float(
                self.segmentation_modulation_beta
            ) * float(q_seg)
        else:
            # Every segmentation class was vacuous (nothing present, nothing
            # predicted).  There is no segmentation evidence to modulate with,
            # so the modulation is the identity rather than a punitive zero.
            q_seg = None
            modulation = 1.0
            seg_terms = {}
        q_perc = q_loc * modulation

        record = QualityEvaluationV1(
            components=components,
            q_seg=q_seg,
            q_loc=q_loc,
            q_perc=q_perc,
            per_class_localization_utility=utilities,
            per_class_xy_utility=xy_utilities,
            per_class_recall=recalls,
            per_class_normalized_segmentation=normalized,
            segmentation_weights_used={
                name: weight for name, (weight, _) in seg_terms.items()
            } or {"__none_defined__": 1.0},
            localization_weights_used={
                name: weight for name, (weight, _) in loc_terms.items()
            },
            localization_combiner=self.localization_combiner,
            reward_spec_sha256=self.canonical_sha256(),
        )
        return replace(
            record, _attestation=_issue_quality(record._binding())
        )

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "gamma_per_tensor": float(self.gamma_per_tensor),
            "lambda_mode": float(self.lambda_mode),
            "lambda_q": float(self.lambda_q),
            "localization_combiner": self.localization_combiner.value,
            "provenance": dict(self.provenance),
            "quality_form": (
                "Q_perc = Q_loc * ((1 - beta) + beta * Q_seg); "
                "U_loc,c = sqrt(recall_c * exp(-e_c / tau_c)); "
                "s_c = clip(IoU_c / reference_c, 0, 1)"
            ),
            "r_registered_failure": float(self.r_registered_failure),
            "record": "reward_spec_v1",
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "reward_form": (
                "r = w_quality * Q_perc - w_latency * (L / B) "
                "- lambda_mode * 1[mode changed] "
                "- lambda_q * |q_exec,t - q_exec,t-1|"
            ),
            "seg_reference_person_iou": float(self.seg_reference_person_iou),
            "seg_reference_vehicle_iou": float(self.seg_reference_vehicle_iou),
            "segmentation_modulation_beta": float(
                self.segmentation_modulation_beta
            ),
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "tau_person_m": float(self.tau_person_m),
            "tau_vehicle_m": float(self.tau_vehicle_m),
            "w_latency": float(self.w_latency),
            "w_loc_person": float(self.w_loc_person),
            "w_loc_vehicle": float(self.w_loc_vehicle),
            "w_quality": float(self.w_quality),
            "w_seg_person": float(self.w_seg_person),
            "w_seg_vehicle": float(self.w_seg_vehicle),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class LatencyMeasurementV1(_Attested):
    """Reward latency: UE-local monotonic policy decision to feedback receipt.

    Both endpoints come from the *same* injected monotonic source inside the
    :class:`~.reward_ticket_controller.CompletedTicket`.  This is deliberately
    **not** the map-install ACK latency and never mixes clock domains: every
    timing field of ``sf_priv_quality_ack.v1`` is a wall-clock reading and none
    may enter here.

    Build with :meth:`from_completed_ticket`; a directly constructed instance is
    unattested and cannot serialize.
    """

    opened_ns: int
    resolution_ns: int
    l_ns: int
    normalized_latency: float
    deadline_ns: int
    completed_ticket_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = StateRewardContractError
        _non_negative_int(self.opened_ns, "opened_ns", E)
        _non_negative_int(self.resolution_ns, "resolution_ns", E)
        _non_negative_int(self.l_ns, "l_ns", E)
        _non_negative_int(self.deadline_ns, "deadline_ns", E)
        _sha256_hex(self.completed_ticket_sha256, "completed_ticket_sha256", E)
        if self.l_ns != self.resolution_ns - self.opened_ns:
            raise E(
                f"l_ns {self.l_ns} must equal resolution_ns - opened_ns "
                f"({self.resolution_ns} - {self.opened_ns})"
            )
        expected = float(self.l_ns) / float(B_REWARD_DEADLINE_NS)
        if float(self.normalized_latency) != expected:
            raise E(
                f"normalized_latency must be exactly l_ns / "
                f"{B_REWARD_DEADLINE_NS}; expected {expected}, got "
                f"{self.normalized_latency}"
            )
        if self.deadline_ns != self.opened_ns + B_REWARD_DEADLINE_NS:
            raise E(
                f"deadline_ns must be exactly opened_ns + "
                f"{B_REWARD_DEADLINE_NS}, got {self.deadline_ns}"
            )
        if self._attestation is not None and not _valid_latency(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the latency measurement's attestation does not match its own "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_latency

    @classmethod
    def from_completed_ticket(cls, ticket: CompletedTicket) -> "LatencyMeasurementV1":
        """Derive L exactly from the frozen ticket; never accept a caller's L."""
        if not isinstance(ticket, CompletedTicket):
            raise StateRewardContractError(
                f"ticket must be a controller CompletedTicket, got "
                f"{type(ticket).__name__}"
            )
        if ticket.resolution_ns is None:
            raise StateRewardContractError(
                f"decision {ticket.decision_seq} closed as "
                f"{ticket.terminal_class.value} with no exact feedback receipt, "
                f"so no action-to-feedback latency exists; a latency must never "
                f"be imputed for a censored or excluded transition"
            )
        l_ns = ticket.resolution_ns - ticket.opened_ns
        record = cls(
            opened_ns=ticket.opened_ns,
            resolution_ns=ticket.resolution_ns,
            l_ns=l_ns,
            normalized_latency=float(l_ns) / float(B_REWARD_DEADLINE_NS),
            deadline_ns=ticket.deadline_ns,
            completed_ticket_sha256=ticket.canonical_sha256(),
        )
        return replace(
            record, _attestation=_issue_latency(record._binding())
        )

    def assert_belongs_to(self, ticket: CompletedTicket) -> None:
        """Fail closed unless this latency was derived from exactly ``ticket``."""
        if self.completed_ticket_sha256 != ticket.canonical_sha256():
            raise StateRewardContractError(
                f"this latency measurement was derived from a different ticket "
                f"({self.completed_ticket_sha256}) than the one supplied "
                f"({ticket.canonical_sha256()}); latency timestamps unrelated "
                f"to the ticket are refused"
            )

    @property
    def clock_domain(self) -> str:
        return REWARD_LATENCY_CLOCK_DOMAIN

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "clock_domain": self.clock_domain,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "deadline_ns": self.deadline_ns,
            "forbidden_sources": list(FORBIDDEN_REWARD_LATENCY_SOURCES),
            "l_ns": self.l_ns,
            "normalized_latency": float(self.normalized_latency),
            "opened_ns": self.opened_ns,
            "record": "latency_measurement_v1",
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "resolution_ns": self.resolution_ns,
        }


@dataclass(frozen=True, slots=True)
class SwitchPenaltyV1:
    """The mode-change and continuous-``q`` movement penalties.

    Both need the previous executed action.  At an episode start there is none,
    so ``applicable`` is False and both terms are zero *because no switch
    exists* -- recorded explicitly rather than as an indistinguishable zero.
    """

    applicable: bool
    mode_changed: Optional[bool]
    q_exec_delta: Optional[float]
    lambda_mode: float
    lambda_q: float
    previous_action_sha256: Optional[str]

    def __post_init__(self) -> None:
        E = RewardSpecError
        _exact_bool(self.applicable, "applicable", E)
        for name in ("lambda_mode", "lambda_q"):
            value = _finite_float(getattr(self, name), name, E)
            if value < 0.0:
                raise E(f"{name} must be >= 0, got {value}")
        if self.applicable:
            _exact_bool(self.mode_changed, "mode_changed", E)
            value = _finite_float(self.q_exec_delta, "q_exec_delta", E)
            if value < 0.0:
                raise E(f"q_exec_delta must be >= 0, got {value}")
            _sha256_hex(
                self.previous_action_sha256, "previous_action_sha256", E
            )
        else:
            for name in ("mode_changed", "q_exec_delta", "previous_action_sha256"):
                if getattr(self, name) is not None:
                    raise E(
                        f"no previous action exists, so {name} must be None "
                        f"rather than a sentinel; got {getattr(self, name)!r}"
                    )

    @classmethod
    def between(
        cls,
        previous: Optional[ExecutedActionIdentity],
        current: ExecutedActionIdentity,
        spec: RewardSpecV1,
    ) -> "SwitchPenaltyV1":
        """Derive both switch terms from the two exact executed actions."""
        if not isinstance(current, ExecutedActionIdentity):
            raise RewardSpecError(
                f"current action must be an ExecutedActionIdentity, got "
                f"{type(current).__name__}"
            )
        if previous is None:
            return cls(
                applicable=False,
                mode_changed=None,
                q_exec_delta=None,
                lambda_mode=float(spec.lambda_mode),
                lambda_q=float(spec.lambda_q),
                previous_action_sha256=None,
            )
        if not isinstance(previous, ExecutedActionIdentity):
            raise RewardSpecError(
                f"previous action must be an ExecutedActionIdentity or None, "
                f"got {type(previous).__name__}"
            )
        previous.require_reconciled()
        current.require_reconciled()
        return cls(
            applicable=True,
            mode_changed=previous.mode_id != current.mode_id,
            q_exec_delta=abs(
                float(current.q_e4) / Q_E4_SCALE - float(previous.q_e4) / Q_E4_SCALE
            ),
            lambda_mode=float(spec.lambda_mode),
            lambda_q=float(spec.lambda_q),
            previous_action_sha256=previous.canonical_sha256(),
        )

    @property
    def total(self) -> float:
        """The summed penalty this transition must subtract."""
        if not self.applicable:
            return 0.0
        mode_term = float(self.lambda_mode) if self.mode_changed else 0.0
        return mode_term + float(self.lambda_q) * float(self.q_exec_delta)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "applicable": self.applicable,
            "lambda_mode": float(self.lambda_mode),
            "lambda_q": float(self.lambda_q),
            "mode_changed": self.mode_changed,
            "previous_action_sha256": self.previous_action_sha256,
            "q_exec_delta": (
                None if self.q_exec_delta is None else float(self.q_exec_delta)
            ),
            "record": "switch_penalty_v1",
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class ConstraintCostsV1:
    """The constraint signals that are part of the optimization contract.

    Only :data:`OPTIMIZATION_CONTRACT_COSTS`.  ``None`` means *unmeasured*, not
    zero.
    """

    c_deadline: Optional[float]
    c_authoritative_failure: Optional[float]

    def __post_init__(self) -> None:
        E = StateRewardContractError
        for name in ("c_deadline", "c_authoritative_failure"):
            value = getattr(self, name)
            if value is not None:
                as_float = _finite_float(value, name, E)
                if as_float not in (0.0, 1.0):
                    raise E(f"{name} is an indicator in {{0, 1}}, got {as_float}")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "c_authoritative_failure": (
                None
                if self.c_authoritative_failure is None
                else float(self.c_authoritative_failure)
            ),
            "c_deadline": (
                None if self.c_deadline is None else float(self.c_deadline)
            ),
            "optimization_contract_costs": list(OPTIMIZATION_CONTRACT_COSTS),
            "record": "constraint_costs_v1",
        }


@dataclass(frozen=True, slots=True)
class DiagnosticSignalsV1:
    """Signals retained for diagnosis and explicitly outside the optimization.

    ``c_latency_excess`` lives here, not in :class:`ConstraintCostsV1`.  The
    controller refuses feedback after the deadline, so on every
    feedback-resolved ticket ``normalized_latency <= 1`` and the excess is
    identically zero; on a censored timeout it is unmeasurable without a
    receipt.  Either way it carries no online information, so presenting it as a
    useful constraint would be misleading.  ``informative`` states that plainly.
    """

    c_latency_excess: Optional[float]
    informative: bool = False

    def __post_init__(self) -> None:
        E = StateRewardContractError
        if self.c_latency_excess is not None:
            value = _finite_float(self.c_latency_excess, "c_latency_excess", E)
            if value < 0.0:
                raise E(f"c_latency_excess must be >= 0, got {value}")
        _exact_bool(self.informative, "informative", E)
        if self.informative:
            raise E(
                "c_latency_excess is not an informative online signal: the "
                "controller rejects post-deadline feedback, so it is "
                "identically zero whenever it is measurable at all.  It is "
                "retained as an unmeasured diagnostic and is excluded from the "
                "optimization contract"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "c_latency_excess": (
                None
                if self.c_latency_excess is None
                else float(self.c_latency_excess)
            ),
            "diagnostic_only_signals": list(DIAGNOSTIC_ONLY_SIGNALS),
            "informative": self.informative,
            "record": "diagnostic_signals_v1",
            "why_uninformative": (
                "the reward-ticket controller refuses feedback after the "
                "200 ms deadline, so normalized_latency <= 1 on every "
                "feedback-resolved ticket and the excess is structurally zero; "
                "a censored timeout has no receipt to measure it from"
            ),
        }


# --------------------------------------------------------------------------- #
# D. Typed, timestamped causal state
# --------------------------------------------------------------------------- #


class SnrMetric(Enum):
    """Exactly which SNR quantity ``achieved_snr_db`` is.

    A free-form source string could silently change the metric's meaning
    between runs -- post-equaliser SINR and a PUCCH SNR are not interchangeable
    even though both are "dB".  Typing it makes a change a schema change.
    """

    SIMULATOR_EFFECTIVE_UL_SNR_DB = "SIMULATOR_EFFECTIVE_UL_SNR_DB"
    GNB_MAC_POWER_CONTROL_NORMALIZED_PUSCH_SNR_DB = (
        "GNB_MAC_POWER_CONTROL_NORMALIZED_PUSCH_SNR_DB"
    )
    GNB_SCHEDULER_EMA_SNR_DB = "GNB_SCHEDULER_EMA_SNR_DB"
    UE_PHY_DIAGNOSTIC_UNRELIABLE_DB = "UE_PHY_DIAGNOSTIC_UNRELIABLE_DB"


class LinkDirection(Enum):
    """Which link a radio measurement describes."""

    UPLINK = "UPLINK"
    DOWNLINK = "DOWNLINK"


class BsrScope(Enum):
    """What a buffer-status report counts."""

    #: Bytes pending for one logical channel group, latest report.
    LOGICAL_CHANNEL_GROUP_LATEST = "LOGICAL_CHANNEL_GROUP_LATEST"
    #: Bytes pending summed over all logical channel groups, latest report.
    ALL_GROUPS_LATEST = "ALL_GROUPS_LATEST"


class RadioEvidencePath(Enum):
    """How radio evidence is admitted, or deliberately not admitted, to policy."""

    SIMULATOR_TESTBED_PRIVILEGED = "SIMULATOR_TESTBED_PRIVILEGED"
    UE_VISIBLE_RUNTIME = "UE_VISIBLE_RUNTIME"
    UNBOUND_COLLECTOR_DIAGNOSTIC = "UNBOUND_COLLECTOR_DIAGNOSTIC"


class RadioSourceWall(Enum):
    """The process wall on which the source event was produced.

    This is intentionally separate from collector ingest and from the later
    UE-local policy-availability measurement.
    """

    SIMULATOR_TESTBED = "SIMULATOR_TESTBED"
    UE = "UE"
    GNB = "GNB"


class SnrSource(Enum):
    """Typed origin of the radio link-quality scalar."""

    SIMULATOR_TESTBED_PRIVILEGED = "SIMULATOR_TESTBED_PRIVILEGED"
    UE_VISIBLE_MEASURED_FEEDBACK = "UE_VISIBLE_MEASURED_FEEDBACK"
    GNB_MAC_PUSCH_POWER_CONTROL = "GNB_MAC_PUSCH_POWER_CONTROL"
    GNB_MAC_UL_MCS_DECISION_EMA = "GNB_MAC_UL_MCS_DECISION_EMA"


class McsSource(Enum):
    """Typed origin of an uplink MCS observation."""

    SIMULATOR_TESTBED_PRIVILEGED = "SIMULATOR_TESTBED_PRIVILEGED"
    NRUE_MAC_DCI_GRANT = "NRUE_MAC_DCI_GRANT"
    GNB_MAC_UL_MCS_DECISION_SELECTED = "GNB_MAC_UL_MCS_DECISION_SELECTED"
    GNB_MAC_UL_MCS_DECISION_FINAL = "GNB_MAC_UL_MCS_DECISION_FINAL"
    GNB_MAC_UL_SCHEDULED_GRANT = "GNB_MAC_UL_SCHEDULED_GRANT"


class BsrSource(Enum):
    """Typed origin of an uplink buffer report."""

    SIMULATOR_TESTBED_PRIVILEGED = "SIMULATOR_TESTBED_PRIVILEGED"
    NRUE_MAC_BSR_STATUS = "NRUE_MAC_BSR_STATUS"
    NRUE_MAC_RLC_BUFFER_STATUS = "NRUE_MAC_RLC_BUFFER_STATUS"
    GNB_MAC_UL_MCS_DECISION_ESTIMATED_BUFFER = (
        "GNB_MAC_UL_MCS_DECISION_ESTIMATED_BUFFER"
    )


class BsrReportType(Enum):
    """The representation from which the eight-LCG byte vector was read."""

    SIMULATOR_VECTOR = "SIMULATOR_VECTOR"
    NR_SHORT = "NR_SHORT"
    NR_LONG = "NR_LONG"
    RLC_BUFFER_SNAPSHOT = "RLC_BUFFER_SNAPSHOT"
    GNB_SCHEDULER_ESTIMATE = "GNB_SCHEDULER_ESTIMATE"


class RadioMissingReason(Enum):
    """Frozen missing-value reasons from the UE-N1 raw-event envelope."""

    NO_MATCHING_EVENT_IN_WINDOW = "NO_MATCHING_EVENT_IN_WINDOW"
    TELEMETRY_RECORDER_NOT_READY = "TELEMETRY_RECORDER_NOT_READY"
    TRACE_GAP_OR_DROP = "TRACE_GAP_OR_DROP"
    RAN_EPOCH_OR_RNTI_JOIN_UNRESOLVED = "RAN_EPOCH_OR_RNTI_JOIN_UNRESOLVED"
    UL_OUTCOME_SOURCE_UNBOUND = "UL_OUTCOME_SOURCE_UNBOUND"
    OTHER_EXPLICIT = "OTHER_EXPLICIT"


class RadioFillPolicy(Enum):
    """The only admissible radio missing-value policy."""

    OBSERVED_ONLY_NO_ZERO_OR_FORWARD_FILL = (
        "OBSERVED_ONLY_NO_ZERO_OR_FORWARD_FILL"
    )


@dataclass(frozen=True, slots=True)
class RadioEventProvenanceV1:
    """One source event with source-wall and collector clocks kept separate."""

    source_wall: RadioSourceWall
    source_event_id: str
    source_event_index: int
    source_event_timestamp_ns: int
    collector_ingest_wall_time_ns: int
    collector_ingest_monotonic_ns: int
    ran_epoch_id: str
    control_session_id: str
    raw_event_sha256: str

    def __post_init__(self) -> None:
        E = CausalStateError
        if not isinstance(self.source_wall, RadioSourceWall):
            raise E(
                "source_wall must be a typed RadioSourceWall, got "
                f"{type(self.source_wall).__name__}: {self.source_wall!r}"
            )
        _non_empty_str(self.source_event_id, "source_event_id", E)
        _non_negative_int(self.source_event_index, "source_event_index", E)
        _non_negative_int(
            self.source_event_timestamp_ns, "source_event_timestamp_ns", E
        )
        _non_negative_int(
            self.collector_ingest_wall_time_ns,
            "collector_ingest_wall_time_ns",
            E,
        )
        _non_negative_int(
            self.collector_ingest_monotonic_ns,
            "collector_ingest_monotonic_ns",
            E,
        )
        _non_empty_str(self.ran_epoch_id, "ran_epoch_id", E)
        _non_empty_str(self.control_session_id, "control_session_id", E)
        _sha256_hex(self.raw_event_sha256, "raw_event_sha256", E)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "collector_ingest_monotonic_ns": self.collector_ingest_monotonic_ns,
            "collector_ingest_wall_time_ns": self.collector_ingest_wall_time_ns,
            "control_session_id": self.control_session_id,
            "ran_epoch_id": self.ran_epoch_id,
            "raw_event_sha256": self.raw_event_sha256,
            "record": "radio_event_provenance_v1",
            "source_event_id": self.source_event_id,
            "source_event_index": self.source_event_index,
            "source_event_timestamp_ns": self.source_event_timestamp_ns,
            "source_timestamp_clock": "CLOCK_REALTIME",
            "source_wall": self.source_wall.value,
        }


@dataclass(frozen=True, slots=True)
class RadioPolicyAvailabilityV1:
    """Shape reserved for a future verified UE-local availability carrier.

    A directly constructed instance is not evidence and cannot make runtime
    radio policy-admissible in Phase 4a.2.
    """

    feedback_path_id: str
    policy_observation_available_monotonic_ns: int
    decision_cutoff_monotonic_ns: int
    ran_epoch_id: str
    control_session_id: str
    availability_evidence_sha256: str
    measurement_source_wall: RadioSourceWall = RadioSourceWall.UE
    clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC

    def __post_init__(self) -> None:
        E = CausalStateError
        _non_empty_str(self.feedback_path_id, "feedback_path_id", E)
        _non_negative_int(
            self.policy_observation_available_monotonic_ns,
            "policy_observation_available_monotonic_ns",
            E,
        )
        _non_negative_int(
            self.decision_cutoff_monotonic_ns,
            "decision_cutoff_monotonic_ns",
            E,
        )
        if (
            self.policy_observation_available_monotonic_ns
            > self.decision_cutoff_monotonic_ns
        ):
            raise E(
                "radio evidence became UE-visible after the decision cutoff: "
                f"{self.policy_observation_available_monotonic_ns} > "
                f"{self.decision_cutoff_monotonic_ns}"
            )
        _non_empty_str(self.ran_epoch_id, "availability ran_epoch_id", E)
        _non_empty_str(
            self.control_session_id, "availability control_session_id", E
        )
        _sha256_hex(
            self.availability_evidence_sha256,
            "availability_evidence_sha256",
            E,
        )
        if self.measurement_source_wall is not RadioSourceWall.UE:
            raise E(
                "runtime policy availability must be measured at the UE, not "
                f"on {getattr(self.measurement_source_wall, 'value', self.measurement_source_wall)!r}"
            )
        if self.clock_domain is not ClockDomain.UE_LOCAL_MONOTONIC:
            raise ClockDomainError(
                "runtime policy availability must use UE_LOCAL_MONOTONIC"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "availability_evidence_sha256": self.availability_evidence_sha256,
            "clock_domain": self.clock_domain.value,
            "control_session_id": self.control_session_id,
            "decision_cutoff_monotonic_ns": self.decision_cutoff_monotonic_ns,
            "feedback_path_id": self.feedback_path_id,
            "measurement_source_wall": self.measurement_source_wall.value,
            "policy_observation_available_monotonic_ns": (
                self.policy_observation_available_monotonic_ns
            ),
            "ran_epoch_id": self.ran_epoch_id,
            "record": "radio_policy_availability_v1",
        }


@dataclass(frozen=True, slots=True)
class BsrReportV1:
    """An eight-LCG BSR/RLC vector with explicit validity and missingness.

    A valid zero is represented as ``value=0, valid=True``.  A missing entry is
    ``value=None, valid=False`` plus a typed reason.  Consequently neither zero
    fill nor forward fill can be encoded as an innocent-looking observation.
    """

    lcg_bytes: Tuple[Optional[int], ...]
    valid_mask: Tuple[bool, ...]
    missing_reasons: Tuple[Optional[RadioMissingReason], ...]
    scope: BsrScope
    logical_channel_group: int
    report_type: BsrReportType
    source: BsrSource
    measured_ns: Optional[int]
    event: Optional[RadioEventProvenanceV1]
    fill_policy: RadioFillPolicy = (
        RadioFillPolicy.OBSERVED_ONLY_NO_ZERO_OR_FORWARD_FILL
    )

    def __post_init__(self) -> None:
        E = CausalStateError
        for name, value in (
            ("lcg_bytes", self.lcg_bytes),
            ("valid_mask", self.valid_mask),
            ("missing_reasons", self.missing_reasons),
        ):
            if type(value) is not tuple or len(value) != 8:
                raise E(f"{name} must be an immutable eight-LCG tuple")
        if not isinstance(self.scope, BsrScope):
            raise E("BSR scope must be a typed BsrScope")
        lcg = _non_negative_int(
            self.logical_channel_group, "bsr logical_channel_group", E
        )
        if lcg > 7:
            raise E(f"bsr logical_channel_group must lie in [0, 7], got {lcg}")
        if not isinstance(self.report_type, BsrReportType):
            raise E("BSR report_type must be a typed BsrReportType")
        if not isinstance(self.source, BsrSource):
            raise E("BSR source must be a typed BsrSource")
        if self.fill_policy is not RadioFillPolicy.OBSERVED_ONLY_NO_ZERO_OR_FORWARD_FILL:
            raise E("radio BSR values may not be zero-filled or forward-filled")

        any_valid = False
        for index, (value, valid, reason) in enumerate(
            zip(self.lcg_bytes, self.valid_mask, self.missing_reasons)
        ):
            _exact_bool(valid, f"bsr valid_mask[{index}]", E)
            if valid:
                any_valid = True
                _non_negative_int(value, f"bsr lcg_bytes[{index}]", E)
                if reason is not None:
                    raise E(
                        f"valid BSR LCG {index} cannot carry a missing reason"
                    )
            else:
                if value is not None:
                    raise E(
                        f"missing BSR LCG {index} must be None, not {value!r}; "
                        "zero fill and forward fill are forbidden"
                    )
                if not isinstance(reason, RadioMissingReason):
                    raise E(
                        f"missing BSR LCG {index} requires a typed missing reason"
                    )
        if self.measured_ns is not None:
            _non_negative_int(self.measured_ns, "bsr measured_ns", E)
        if any_valid and (self.measured_ns is None or self.event is None):
            raise E("a BSR containing valid values requires time and event provenance")
        if self.event is not None and not isinstance(
            self.event, RadioEventProvenanceV1
        ):
            raise E("BSR event must be RadioEventProvenanceV1 or None")

        expected_wall = {
            BsrSource.SIMULATOR_TESTBED_PRIVILEGED: RadioSourceWall.SIMULATOR_TESTBED,
            BsrSource.NRUE_MAC_BSR_STATUS: RadioSourceWall.UE,
            BsrSource.NRUE_MAC_RLC_BUFFER_STATUS: RadioSourceWall.UE,
            BsrSource.GNB_MAC_UL_MCS_DECISION_ESTIMATED_BUFFER: RadioSourceWall.GNB,
        }[self.source]
        if self.event is not None and self.event.source_wall is not expected_wall:
            raise E(
                f"BSR source {self.source.value} must originate on the "
                f"{expected_wall.value} wall"
            )

    @property
    def required_indices(self) -> Tuple[int, ...]:
        if self.scope is BsrScope.ALL_GROUPS_LATEST:
            return tuple(range(8))
        return (self.logical_channel_group,)

    @property
    def complete_for_scope(self) -> bool:
        return all(self.valid_mask[index] for index in self.required_indices)

    def require_complete_for_policy(self) -> None:
        if not self.complete_for_scope:
            missing = [
                index
                for index in self.required_indices
                if not self.valid_mask[index]
            ]
            raise CausalStateError(
                f"policy-facing BSR is missing required LCGs {missing}; "
                "missing values are not zero-filled or forward-filled"
            )

    @property
    def total_bytes(self) -> int:
        self.require_complete_for_policy()
        return sum(int(self.lcg_bytes[index]) for index in self.required_indices)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "complete_for_scope": self.complete_for_scope,
            "event": None if self.event is None else self.event.to_canonical_dict(),
            "fill_policy": self.fill_policy.value,
            "lcg_bytes": list(self.lcg_bytes),
            "logical_channel_group": self.logical_channel_group,
            "measured_ns": self.measured_ns,
            "missing_reasons": [
                None if reason is None else reason.value
                for reason in self.missing_reasons
            ],
            "record": "bsr_report_v1",
            "report_type": self.report_type.value,
            "scope": self.scope.value,
            "source": self.source.value,
            "valid_mask": list(self.valid_mask),
        }


@dataclass(frozen=True, slots=True)
class SceneObservationV1:
    """One timestamped SI/P40 observation, bound to its CARLA frame and source.

    The age is *derived* from ``observed_ns - measured_ns``; a caller never
    supplies an age, so a forged zero age is unrepresentable.
    """

    sample: SceneDescriptorSample
    measured_ns: int
    carla_frame_id: int
    source_id: str
    source_sha256: str
    clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC

    def __post_init__(self) -> None:
        E = CausalStateError
        if not isinstance(self.sample, SceneDescriptorSample):
            raise E(
                f"sample must be a Phase-3 SceneDescriptorSample -- the only "
                f"admissible SI/P40 pair -- got {type(self.sample).__name__}"
            )
        _finite_float(self.sample.camera_si, "camera_si", E)
        if float(self.sample.camera_si) < 0.0:
            raise E("camera_si cannot be negative")
        _finite_in(self.sample.radar_p40, "radar_p40", 0.0, 1.0, E)
        _non_negative_int(self.measured_ns, "scene measured_ns", E)
        _non_negative_int(self.carla_frame_id, "scene carla_frame_id", E)
        _non_empty_str(self.source_id, "scene source_id", E)
        _sha256_hex(self.source_sha256, "scene source_sha256", E)
        if not isinstance(self.clock_domain, ClockDomain):
            raise ClockDomainError(
                f"clock_domain must be a ClockDomain, got "
                f"{type(self.clock_domain).__name__}"
            )

    @property
    def camera_si(self) -> float:
        return float(self.sample.camera_si)

    @property
    def radar_p40(self) -> float:
        return float(self.sample.radar_p40)

    def age_ns(self, observed_ns: int) -> int:
        """Derived age.  Fails closed if the measurement postdates observation."""
        if self.measured_ns > observed_ns:
            raise CausalStateError(
                f"scene measurement at {self.measured_ns} ns postdates the "
                f"observation instant {observed_ns} ns; a state may never "
                f"contain a future measurement"
            )
        return observed_ns - self.measured_ns

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "camera_si": self.camera_si,
            "carla_frame_id": self.carla_frame_id,
            "clock_domain": self.clock_domain.value,
            "measured_ns": self.measured_ns,
            "radar_p40": self.radar_p40,
            "record": "scene_observation_v1",
            "scene_schema_id": SCENE_SCHEMA_ID,
            "scene_schema_sha256": SCENE_SCHEMA_SHA256,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class RadioObservationV1(_Attested):
    """Complete radio evidence explicitly admitted to the causal policy.

    Phase 4a.2 admits simulator/testbed values only through the attested
    privileged factory.  Runtime source types remain modeled so a later
    protocol can be reviewed without changing meanings, but they fail closed:
    no current repository artifact proves when those values became visible to
    the UE policy.  Collector ingest is never treated as that proof.
    """

    achieved_snr_db: float
    snr_metric: SnrMetric
    snr_direction: LinkDirection
    snr_measured_ns: int
    mcs_index: int
    mcs_table_id: str
    mcs_direction: LinkDirection
    mcs_measured_ns: int
    bsr_bytes: int
    bsr_scope: BsrScope
    bsr_logical_channel_group: int
    bsr_measured_ns: int
    evidence_path: RadioEvidencePath
    snr_source: SnrSource
    snr_event: RadioEventProvenanceV1
    mcs_source: McsSource
    mcs_event: RadioEventProvenanceV1
    bsr_report: BsrReportV1
    policy_availability: Optional[RadioPolicyAvailabilityV1]
    source_id: str
    source_sha256: str
    clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = CausalStateError
        _finite_float(self.achieved_snr_db, "achieved_snr_db", E)
        if not isinstance(self.snr_metric, SnrMetric):
            raise E(
                f"snr_metric must be a typed SnrMetric so a free-form source "
                f"string cannot silently change the metric's meaning, got "
                f"{type(self.snr_metric).__name__}: {self.snr_metric!r}"
            )
        for name in ("snr_direction", "mcs_direction"):
            value = getattr(self, name)
            if not isinstance(value, LinkDirection):
                raise E(
                    f"{name} must be a LinkDirection, got "
                    f"{type(value).__name__}: {value!r}"
                )
        _non_negative_int(self.mcs_index, "mcs_index", E)
        _non_empty_str(self.mcs_table_id, "mcs_table_id", E)
        _non_negative_int(self.bsr_bytes, "bsr_bytes", E)
        if not isinstance(self.bsr_scope, BsrScope):
            raise E(
                f"bsr_scope must be a BsrScope declaring what the report "
                f"counts, got {type(self.bsr_scope).__name__}: "
                f"{self.bsr_scope!r}"
            )
        _non_negative_int(
            self.bsr_logical_channel_group, "bsr_logical_channel_group", E
        )
        if self.bsr_logical_channel_group > 7:
            raise E("bsr_logical_channel_group must lie in [0, 7]")
        for name in ("snr_measured_ns", "mcs_measured_ns", "bsr_measured_ns"):
            _non_negative_int(getattr(self, name), name, E)
        if not isinstance(self.evidence_path, RadioEvidencePath):
            raise E("evidence_path must be a typed RadioEvidencePath")
        if self.evidence_path is RadioEvidencePath.UNBOUND_COLLECTOR_DIAGNOSTIC:
            raise E(
                "unbound collector evidence is diagnostic-only and cannot be "
                "constructed as RadioObservationV1"
            )
        if self.evidence_path is RadioEvidencePath.UE_VISIBLE_RUNTIME:
            raise CausalStateError(
                "UE-visible runtime radio evidence is not yet admissible: the "
                "repository has no measured UE-visible feedback/IPC carrier "
                "whose envelope and availability can be opened and hash-"
                "verified.  Stock collector timestamps are post-action "
                "diagnostics, not policy availability.  Use "
                "RadioDiagnosticObservationV1 until that carrier exists"
            )
        if not isinstance(self.snr_source, SnrSource):
            raise E("snr_source must be a typed SnrSource")
        if not isinstance(self.snr_event, RadioEventProvenanceV1):
            raise E("snr_event must be RadioEventProvenanceV1")
        if not isinstance(self.mcs_source, McsSource):
            raise E("mcs_source must be a typed McsSource")
        if not isinstance(self.mcs_event, RadioEventProvenanceV1):
            raise E("mcs_event must be RadioEventProvenanceV1")
        if not isinstance(self.bsr_report, BsrReportV1):
            raise E("bsr_report must be a typed BsrReportV1")
        self.bsr_report.require_complete_for_policy()
        if self.bsr_report.scope is not self.bsr_scope:
            raise E("flat BSR scope disagrees with the typed BSR report")
        if (
            self.bsr_report.logical_channel_group
            != self.bsr_logical_channel_group
        ):
            raise E("flat BSR logical-channel group disagrees with its report")
        if self.bsr_report.measured_ns != self.bsr_measured_ns:
            raise E("flat BSR timestamp disagrees with its report")
        if self.bsr_report.total_bytes != self.bsr_bytes:
            raise E(
                f"flat bsr_bytes={self.bsr_bytes} disagrees with the explicit "
                f"BSR vector total {self.bsr_report.total_bytes}"
            )
        _non_empty_str(self.source_id, "radio source_id", E)
        _sha256_hex(self.source_sha256, "radio source_sha256", E)
        if not isinstance(self.clock_domain, ClockDomain):
            raise ClockDomainError(
                f"clock_domain must be a ClockDomain, got "
                f"{type(self.clock_domain).__name__}"
            )
        # The uplink is the link the split payload traverses; a downlink SNR or
        # MCS would be a different quantity and must be declared as such.
        if self.snr_direction is not LinkDirection.UPLINK:
            raise E(
                f"the split payload traverses the uplink, so the reward-facing "
                f"SNR must be an uplink measurement; got "
                f"{self.snr_direction.value}"
            )
        if self.mcs_direction is not LinkDirection.UPLINK:
            raise E(
                f"the reward-facing MCS must be the uplink MCS; got "
                f"{self.mcs_direction.value}"
            )

        expected_snr_wall = {
            SnrSource.SIMULATOR_TESTBED_PRIVILEGED: RadioSourceWall.SIMULATOR_TESTBED,
            SnrSource.UE_VISIBLE_MEASURED_FEEDBACK: RadioSourceWall.UE,
            SnrSource.GNB_MAC_PUSCH_POWER_CONTROL: RadioSourceWall.GNB,
            SnrSource.GNB_MAC_UL_MCS_DECISION_EMA: RadioSourceWall.GNB,
        }[self.snr_source]
        if self.snr_event.source_wall is not expected_snr_wall:
            raise E(
                f"SNR source {self.snr_source.value} must originate on the "
                f"{expected_snr_wall.value} wall"
            )
        expected_mcs_wall = {
            McsSource.SIMULATOR_TESTBED_PRIVILEGED: RadioSourceWall.SIMULATOR_TESTBED,
            McsSource.NRUE_MAC_DCI_GRANT: RadioSourceWall.UE,
            McsSource.GNB_MAC_UL_MCS_DECISION_SELECTED: RadioSourceWall.GNB,
            McsSource.GNB_MAC_UL_MCS_DECISION_FINAL: RadioSourceWall.GNB,
            McsSource.GNB_MAC_UL_SCHEDULED_GRANT: RadioSourceWall.GNB,
        }[self.mcs_source]
        if self.mcs_event.source_wall is not expected_mcs_wall:
            raise E(
                f"MCS source {self.mcs_source.value} must originate on the "
                f"{expected_mcs_wall.value} wall"
            )

        events = (self.snr_event, self.mcs_event, self.bsr_report.event)
        if self.bsr_report.event is None:  # guarded by a complete report
            raise E("a policy-facing BSR must carry event provenance")
        epoch_ids = {event.ran_epoch_id for event in events if event is not None}
        session_ids = {
            event.control_session_id for event in events if event is not None
        }
        if len(epoch_ids) != 1 or len(session_ids) != 1:
            raise E(
                "SNR, MCS and BSR events must share one RAN epoch and control session"
            )

        if self.evidence_path is RadioEvidencePath.SIMULATOR_TESTBED_PRIVILEGED:
            if self.policy_availability is not None:
                raise E(
                    "simulator/testbed privileged evidence does not masquerade "
                    "as a UE-visible runtime observation"
                )
            if (
                self.snr_source is not SnrSource.SIMULATOR_TESTBED_PRIVILEGED
                or self.mcs_source
                is not McsSource.SIMULATOR_TESTBED_PRIVILEGED
                or self.bsr_report.source
                is not BsrSource.SIMULATOR_TESTBED_PRIVILEGED
            ):
                raise E(
                    "the privileged simulator path accepts only explicitly "
                    "simulator/testbed radio sources"
                )
        else:  # pragma: no cover - UE_VISIBLE_RUNTIME fails closed above
            if self.snr_source is SnrSource.SIMULATOR_TESTBED_PRIVILEGED:
                raise E("runtime radio evidence cannot use a simulator SNR source")
            if self.mcs_source is McsSource.SIMULATOR_TESTBED_PRIVILEGED:
                raise E("runtime radio evidence cannot use a simulator MCS source")
            if self.bsr_report.source is BsrSource.SIMULATOR_TESTBED_PRIVILEGED:
                raise E("runtime radio evidence cannot use a simulator BSR source")
            if not isinstance(
                self.policy_availability, RadioPolicyAvailabilityV1
            ):
                raise E(
                    "UE-visible runtime radio evidence requires measured "
                    "UE-local policy availability; collector ingest is not "
                    "availability"
                )
            availability = self.policy_availability
            if availability.ran_epoch_id not in epoch_ids:
                raise E("runtime availability is for a different RAN epoch")
            if availability.control_session_id not in session_ids:
                raise E("runtime availability is for a different control session")
            latest_measurement = max(
                self.snr_measured_ns,
                self.mcs_measured_ns,
                self.bsr_measured_ns,
            )
            if (
                availability.policy_observation_available_monotonic_ns
                < latest_measurement
            ):
                raise E(
                    "policy availability predates a component measurement; "
                    "the aggregate observation was not yet available"
                )
        expected_snr_metric = {
            SnrSource.SIMULATOR_TESTBED_PRIVILEGED: (
                SnrMetric.SIMULATOR_EFFECTIVE_UL_SNR_DB
            ),
            SnrSource.UE_VISIBLE_MEASURED_FEEDBACK: (
                SnrMetric.GNB_MAC_POWER_CONTROL_NORMALIZED_PUSCH_SNR_DB
            ),
            SnrSource.GNB_MAC_PUSCH_POWER_CONTROL: (
                SnrMetric.GNB_MAC_POWER_CONTROL_NORMALIZED_PUSCH_SNR_DB
            ),
            SnrSource.GNB_MAC_UL_MCS_DECISION_EMA: (
                SnrMetric.GNB_SCHEDULER_EMA_SNR_DB
            ),
        }[self.snr_source]
        if self.snr_metric is not expected_snr_metric:
            raise E(
                f"SNR source {self.snr_source.value} carries "
                f"{expected_snr_metric.value}, not {self.snr_metric.value}"
            )
        if self._attestation is not None and not _valid_radio(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the radio-observation attestation does not match its fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_radio

    @classmethod
    def for_simulator_testbed(
        cls,
        *,
        achieved_snr_db: float,
        snr_measured_ns: int,
        mcs_index: int,
        mcs_table_id: str,
        mcs_measured_ns: int,
        bsr_bytes: int,
        bsr_scope: BsrScope,
        bsr_logical_channel_group: int,
        bsr_measured_ns: int,
        snr_event: RadioEventProvenanceV1,
        mcs_event: RadioEventProvenanceV1,
        bsr_report: BsrReportV1,
        source_id: str,
        source_sha256: str,
    ) -> "RadioObservationV1":
        """Issue the only currently supported policy-facing radio record.

        Values are explicitly privileged simulator/testbed observations.  This
        factory makes no deployability claim and cannot be used to relabel
        stock OAI collector rows as UE-visible policy inputs.
        """
        record = cls(
            achieved_snr_db=achieved_snr_db,
            snr_metric=SnrMetric.SIMULATOR_EFFECTIVE_UL_SNR_DB,
            snr_direction=LinkDirection.UPLINK,
            snr_measured_ns=snr_measured_ns,
            mcs_index=mcs_index,
            mcs_table_id=mcs_table_id,
            mcs_direction=LinkDirection.UPLINK,
            mcs_measured_ns=mcs_measured_ns,
            bsr_bytes=bsr_bytes,
            bsr_scope=bsr_scope,
            bsr_logical_channel_group=bsr_logical_channel_group,
            bsr_measured_ns=bsr_measured_ns,
            evidence_path=RadioEvidencePath.SIMULATOR_TESTBED_PRIVILEGED,
            snr_source=SnrSource.SIMULATOR_TESTBED_PRIVILEGED,
            snr_event=snr_event,
            mcs_source=McsSource.SIMULATOR_TESTBED_PRIVILEGED,
            mcs_event=mcs_event,
            bsr_report=bsr_report,
            policy_availability=None,
            source_id=source_id,
            source_sha256=source_sha256,
        )
        return replace(record, _attestation=_issue_radio(record._binding()))

    def ages_ns(self, observed_ns: int) -> Dict[str, int]:
        """Derived per-source ages; fails closed on any future measurement."""
        if (
            self.evidence_path is RadioEvidencePath.UE_VISIBLE_RUNTIME
            and self.policy_availability is not None
            and self.policy_availability.decision_cutoff_monotonic_ns
            != observed_ns
        ):
            raise CausalStateError(
                "runtime radio availability is bound to decision cutoff "
                f"{self.policy_availability.decision_cutoff_monotonic_ns}, "
                f"not this state's observation instant {observed_ns}"
            )
        ages: Dict[str, int] = {}
        for label, measured in (
            ("snr", self.snr_measured_ns),
            ("bsr", self.bsr_measured_ns),
            ("mcs", self.mcs_measured_ns),
        ):
            if measured > observed_ns:
                raise CausalStateError(
                    f"{label} measurement at {measured} ns postdates the "
                    f"observation instant {observed_ns} ns; a state may never "
                    f"contain a future measurement"
                )
            ages[label] = observed_ns - measured
        return ages

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db": float(self.achieved_snr_db),
            "bsr_bytes": self.bsr_bytes,
            "bsr_logical_channel_group": self.bsr_logical_channel_group,
            "bsr_measured_ns": self.bsr_measured_ns,
            "bsr_report": self.bsr_report.to_canonical_dict(),
            "bsr_scope": self.bsr_scope.value,
            "clock_domain": self.clock_domain.value,
            "evidence_path": self.evidence_path.value,
            "mcs_direction": self.mcs_direction.value,
            "mcs_event": self.mcs_event.to_canonical_dict(),
            "mcs_index": self.mcs_index,
            "mcs_measured_ns": self.mcs_measured_ns,
            "mcs_source": self.mcs_source.value,
            "mcs_table_id": self.mcs_table_id,
            "policy_availability": (
                None
                if self.policy_availability is None
                else self.policy_availability.to_canonical_dict()
            ),
            "record": "radio_observation_v1",
            "snr_direction": self.snr_direction.value,
            "snr_event": self.snr_event.to_canonical_dict(),
            "snr_measured_ns": self.snr_measured_ns,
            "snr_metric": self.snr_metric.value,
            "snr_source": self.snr_source.value,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class RadioDiagnosticObservationV1:
    """Typed radio collector evidence that is structurally non-causal.

    This record deliberately is not a :class:`RadioObservationV1`, so
    :class:`CausalStateV1` rejects it by type.  It retains missing values and
    their reasons without substituting zeros or prior observations.
    """

    achieved_snr_db: Optional[float]
    snr_metric: SnrMetric
    snr_direction: LinkDirection
    snr_measured_ns: Optional[int]
    snr_source: SnrSource
    snr_event: Optional[RadioEventProvenanceV1]
    mcs_index: Optional[int]
    mcs_table_id: str
    mcs_direction: LinkDirection
    mcs_measured_ns: Optional[int]
    mcs_source: McsSource
    mcs_event: Optional[RadioEventProvenanceV1]
    bsr_report: BsrReportV1
    valid_mask: Tuple[bool, bool, bool]
    missing_reasons: Tuple[
        Optional[RadioMissingReason],
        Optional[RadioMissingReason],
        Optional[RadioMissingReason],
    ]
    source_id: str
    source_sha256: str
    clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC

    FIELD_ORDER: Tuple[str, str, str] = field(
        default=("snr", "mcs", "bsr"), init=False, repr=False
    )

    def __post_init__(self) -> None:
        E = CausalStateError
        if type(self.valid_mask) is not tuple or len(self.valid_mask) != 3:
            raise E("radio diagnostic valid_mask must be (snr, mcs, bsr)")
        if type(self.missing_reasons) is not tuple or len(self.missing_reasons) != 3:
            raise E("radio diagnostic missing_reasons must be (snr, mcs, bsr)")
        for name, value in (
            ("snr_metric", self.snr_metric),
            ("snr_direction", self.snr_direction),
            ("snr_source", self.snr_source),
            ("mcs_direction", self.mcs_direction),
            ("mcs_source", self.mcs_source),
        ):
            expected = {
                "snr_metric": SnrMetric,
                "snr_direction": LinkDirection,
                "snr_source": SnrSource,
                "mcs_direction": LinkDirection,
                "mcs_source": McsSource,
            }[name]
            if not isinstance(value, expected):
                raise E(f"{name} must be typed {expected.__name__}")
        _non_empty_str(self.mcs_table_id, "diagnostic mcs_table_id", E)
        if not isinstance(self.bsr_report, BsrReportV1):
            raise E("diagnostic BSR must be a BsrReportV1")
        _non_empty_str(self.source_id, "diagnostic radio source_id", E)
        _sha256_hex(self.source_sha256, "diagnostic radio source_sha256", E)
        if self.clock_domain is not ClockDomain.UE_LOCAL_MONOTONIC:
            raise ClockDomainError(
                "diagnostic monotonic timestamps must use UE_LOCAL_MONOTONIC"
            )

        values = (self.achieved_snr_db, self.mcs_index, self.bsr_report)
        times = (self.snr_measured_ns, self.mcs_measured_ns)
        events = (self.snr_event, self.mcs_event)
        for index, (name, valid, reason) in enumerate(
            zip(self.FIELD_ORDER, self.valid_mask, self.missing_reasons)
        ):
            _exact_bool(valid, f"diagnostic valid_mask[{name}]", E)
            if valid:
                if reason is not None:
                    raise E(f"valid diagnostic {name} cannot have a missing reason")
                if name == "bsr":
                    self.bsr_report.require_complete_for_policy()
                elif values[index] is None:
                    raise E(f"valid diagnostic {name} requires a value")
            else:
                if not isinstance(reason, RadioMissingReason):
                    raise E(f"missing diagnostic {name} requires a typed reason")
                if name != "bsr" and values[index] is not None:
                    raise E(
                        f"missing diagnostic {name} must be None; zero and "
                        "forward fill are forbidden"
                    )
                if name == "bsr" and self.bsr_report.complete_for_scope:
                    raise E("a diagnostic BSR marked missing cannot be complete")
        if self.valid_mask[0]:
            _finite_float(self.achieved_snr_db, "diagnostic achieved_snr_db", E)
        if self.valid_mask[1]:
            _non_negative_int(self.mcs_index, "diagnostic mcs_index", E)
        for index, (name, measured, event) in enumerate(
            zip(("snr", "mcs"), times, events)
        ):
            if self.valid_mask[index] and (measured is None or event is None):
                raise E(f"valid diagnostic {name} requires time and event provenance")
            if measured is not None:
                _non_negative_int(measured, f"diagnostic {name}_measured_ns", E)
            if event is not None and not isinstance(event, RadioEventProvenanceV1):
                raise E(f"diagnostic {name}_event has the wrong type")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db": (
                None if self.achieved_snr_db is None else float(self.achieved_snr_db)
            ),
            "bsr_report": self.bsr_report.to_canonical_dict(),
            "causal_policy_eligible": False,
            "clock_domain": self.clock_domain.value,
            "evidence_path": RadioEvidencePath.UNBOUND_COLLECTOR_DIAGNOSTIC.value,
            "field_order": list(self.FIELD_ORDER),
            "mcs_direction": self.mcs_direction.value,
            "mcs_event": None if self.mcs_event is None else self.mcs_event.to_canonical_dict(),
            "mcs_index": self.mcs_index,
            "mcs_measured_ns": self.mcs_measured_ns,
            "mcs_source": self.mcs_source.value,
            "mcs_table_id": self.mcs_table_id,
            "missing_reasons": [
                None if reason is None else reason.value
                for reason in self.missing_reasons
            ],
            "record": "radio_diagnostic_observation_v1",
            "snr_direction": self.snr_direction.value,
            "snr_event": None if self.snr_event is None else self.snr_event.to_canonical_dict(),
            "snr_measured_ns": self.snr_measured_ns,
            "snr_metric": self.snr_metric.value,
            "snr_source": self.snr_source.value,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "valid_mask": list(self.valid_mask),
            "zero_fill_authorized": False,
            "forward_fill_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class EpisodeStartProofV1(_Attested):
    """Pre-decision proof that a state has no policy predecessor.

    ``previous=None`` is otherwise ambiguous: it could mean a genuine episode
    start or a dropped predecessor record.  The controller genesis proof is
    explicitly authorized after the first state is observed but before any
    decision opens. This wrapper binds that proof to the first proposed
    decision/state. Replay
    later checks that the completed ticket is genuinely ordinal zero in the
    same lineage; no future ticket hash enters the policy observation.
    """

    controller_genesis: ControllerGenesisProof
    source_id: str
    source_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = CausalStateError
        if not isinstance(self.controller_genesis, ControllerGenesisProof):
            raise E(
                "controller_genesis must be a ControllerGenesisProof issued "
                "by RewardTicketController"
            )
        try:
            self.controller_genesis.require_attested()
        except Exception as exc:
            raise E(
                "controller_genesis is not an attested pre-decision proof"
            ) from exc
        _non_empty_str(self.source_id, "episode-start source_id", E)
        _sha256_hex(self.source_sha256, "episode-start source_sha256", E)
        if self._attestation is not None and not _valid_episode_start(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the episode-start attestation does not match its serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_episode_start

    @property
    def session_uuid(self) -> str:
        return self.controller_genesis.session_uuid

    @property
    def controller_lineage_uuid(self) -> str:
        return self.controller_genesis.controller_lineage_uuid

    @property
    def first_decision_seq(self) -> int:
        return self.controller_genesis.first_decision_seq

    @property
    def first_tensor_seq(self) -> int:
        return self.controller_genesis.first_tensor_seq

    @property
    def first_carla_frame_id(self) -> int:
        return self.controller_genesis.first_carla_frame_id

    @property
    def episode_started_ns(self) -> int:
        return self.controller_genesis.state_observed_ns

    @classmethod
    def from_controller_genesis(
        cls,
        controller_genesis: ControllerGenesisProof,
        *,
        source_id: str,
        source_sha256: str,
    ) -> "EpisodeStartProofV1":
        """Bind a controller-issued genesis proof before the first action."""
        if not isinstance(controller_genesis, ControllerGenesisProof):
            raise CausalStateError(
                "controller_genesis must be a ControllerGenesisProof, got "
                f"{type(controller_genesis).__name__}"
            )
        try:
            controller_genesis.require_attested()
        except Exception as exc:
            raise CausalStateError(
                "controller_genesis was not issued by RewardTicketController"
            ) from exc
        record = cls(
            controller_genesis=controller_genesis,
            source_id=source_id,
            source_sha256=source_sha256,
        )
        return replace(
            record, _attestation=_issue_episode_start(record._binding())
        )

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "controller_genesis": self.controller_genesis.to_canonical_dict(),
            "controller_genesis_sha256": (
                self.controller_genesis.canonical_sha256()
            ),
            "episode_started_ns": self.episode_started_ns,
            "first_carla_frame_id": self.first_carla_frame_id,
            "first_decision_seq": self.first_decision_seq,
            "first_tensor_seq": self.first_tensor_seq,
            "record": "episode_start_proof_v1",
            "session_uuid": self.session_uuid,
            "controller_lineage_uuid": self.controller_lineage_uuid,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreviousOutcomeV1(_Attested):
    """Policy-visible closure snapshot of the previous decision.

    Build with :meth:`from_completed`, which is the only path that can issue the
    attestation.  This is deliberately *not* the later replay adjudication.  A
    timeout is frozen as ``PENDING`` at ticket closure; a later reconciliation
    may change the replay outcome but can never rewrite the historical policy
    observation.  ``available_ns`` makes that causality check explicit.
    """

    session_uuid: str
    decision_seq: int
    action: ExecutedActionIdentity
    terminal_class: TerminalClass
    eligibility: str
    completed_ticket_sha256: str
    outcome_sha256: str
    reward_spec_sha256: str
    available_ns: int
    resolution_ns: Optional[int]
    quality_normalized: Optional[float]
    latency_normalized: Optional[float]
    quality_gt_source: Optional[str]
    latency_clock_domain: Optional[str]
    controller_lineage_uuid: str
    lineage_ordinal: int
    predecessor_completed_ticket_sha256: Optional[str]
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = CausalStateError
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.decision_seq, "decision_seq", E)
        if not isinstance(self.action, ExecutedActionIdentity):
            raise E(
                f"action must be a Phase-2 ExecutedActionIdentity, got "
                f"{type(self.action).__name__}"
            )
        self.action.require_reconciled()
        if not isinstance(self.terminal_class, TerminalClass):
            raise E(
                f"terminal_class must be a controller TerminalClass, got "
                f"{type(self.terminal_class).__name__}"
            )
        _non_empty_str(self.eligibility, "eligibility", E)
        for name in (
            "completed_ticket_sha256",
            "outcome_sha256",
            "reward_spec_sha256",
        ):
            _sha256_hex(getattr(self, name), name, E)
        _non_negative_int(self.available_ns, "available_ns", E)
        _canonical_uuid(self.controller_lineage_uuid, E)
        _non_negative_int(self.lineage_ordinal, "lineage_ordinal", E)
        if self.lineage_ordinal == 0:
            if self.predecessor_completed_ticket_sha256 is not None:
                raise E(
                    "the controller genesis previous outcome cannot name a "
                    "predecessor ticket"
                )
        else:
            if self.predecessor_completed_ticket_sha256 is None:
                raise E(
                    f"previous outcome at controller lineage ordinal "
                    f"{self.lineage_ordinal} requires its predecessor hash"
                )
            _sha256_hex(
                self.predecessor_completed_ticket_sha256,
                "predecessor_completed_ticket_sha256",
                E,
            )
        if self.resolution_ns is not None:
            _non_negative_int(self.resolution_ns, "resolution_ns", E)
        if self.quality_normalized is not None:
            _finite_in(self.quality_normalized, "quality_normalized", 0.0, 1.0, E)
            if self.quality_gt_source is None:
                raise E(
                    "a previous quality must name the ground-truth source it "
                    "came from; it is privileged CARLA ground truth"
                )
        elif self.quality_gt_source is not None:
            raise E(
                "quality_gt_source is present without a quality value"
            )
        if self.latency_normalized is not None:
            value = _finite_float(self.latency_normalized, "latency_normalized", E)
            if value < 0.0:
                raise E(f"latency_normalized must be >= 0, got {value}")
            if self.latency_clock_domain != REWARD_LATENCY_CLOCK_DOMAIN:
                raise ClockDomainError(
                    f"a previous latency must declare the "
                    f"{REWARD_LATENCY_CLOCK_DOMAIN} clock domain, got "
                    f"{self.latency_clock_domain!r}"
                )
        elif self.latency_clock_domain is not None:
            raise E(
                "latency_clock_domain is present without a latency value"
            )
        # Only an exactly-resolved decision has an exact quality or latency.
        if self.terminal_class is not TerminalClass.REWARD_FINAL_EXACT and (
            self.quality_normalized is not None
        ):
            raise E(
                f"terminal class {self.terminal_class.value} yields no exact "
                f"quality; got {self.quality_normalized!r}"
            )
        if (
            self.terminal_class is TerminalClass.FEEDBACK_TIMEOUT
            and self.eligibility
            != LearningEligibility.CENSORED_PENDING_ADJUDICATION.value
        ):
            raise E(
                "a policy-visible timeout snapshot must remain PENDING at "
                "ticket closure; later adjudication is replay-only"
            )
        if self._attestation is not None and not _valid_previous(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the previous-outcome attestation does not match its own "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_previous

    @classmethod
    def from_completed(
        cls,
        completed_ticket: CompletedTicket,
        outcome: "DecisionOutcomeV1",
        reward_spec: RewardSpecV1,
    ) -> "PreviousOutcomeV1":
        """Derive the previous-decision summary from validated records."""
        if not isinstance(completed_ticket, CompletedTicket):
            raise CausalStateError(
                f"completed_ticket must be a controller CompletedTicket, got "
                f"{type(completed_ticket).__name__}"
            )
        completed_ticket.require_lineage_attested()
        if not isinstance(outcome, DecisionOutcomeV1):
            raise CausalStateError(
                f"outcome must be a DecisionOutcomeV1, got "
                f"{type(outcome).__name__}"
            )
        outcome.require_attested()
        if not isinstance(reward_spec, RewardSpecV1):
            raise RewardSpecError(
                f"reward_spec must be a RewardSpecV1, got "
                f"{type(reward_spec).__name__}"
            )
        if outcome.completed_ticket_sha256 != completed_ticket.canonical_sha256():
            raise CausalStateError(
                "the outcome was measured for a different ticket than the one "
                "supplied"
            )
        if outcome.reward_spec_sha256 != reward_spec.canonical_sha256():
            raise CausalStateError(
                "the outcome was measured under a different reward spec"
            )
        if outcome.adjudication is not None:
            raise CausalStateError(
                "a policy-visible previous snapshot cannot be constructed "
                "from a later adjudication; use the closure-time outcome"
            )
        if (
            completed_ticket.terminal_class is TerminalClass.FEEDBACK_TIMEOUT
            and outcome.eligibility
            is not LearningEligibility.CENSORED_PENDING_ADJUDICATION
        ):
            raise CausalStateError(
                "a timeout enters policy history only as the closure-time "
                "PENDING snapshot"
            )
        quality = outcome.quality
        latency = outcome.latency
        assert completed_ticket.controller_lineage_uuid is not None
        assert completed_ticket.lineage_ordinal is not None
        record = cls(
            session_uuid=completed_ticket.session_uuid,
            decision_seq=completed_ticket.decision_seq,
            action=completed_ticket.action,
            terminal_class=completed_ticket.terminal_class,
            eligibility=outcome.eligibility.value,
            completed_ticket_sha256=completed_ticket.canonical_sha256(),
            outcome_sha256=outcome.canonical_sha256(),
            reward_spec_sha256=outcome.reward_spec_sha256,
            available_ns=completed_ticket.closed_ns,
            resolution_ns=completed_ticket.resolution_ns,
            quality_normalized=None if quality is None else quality.q_perc,
            latency_normalized=(
                None if latency is None else latency.normalized_latency
            ),
            quality_gt_source=(
                None
                if quality is None
                else quality.components.evidence.gt_source.value
            ),
            latency_clock_domain=(
                None if latency is None else latency.clock_domain
            ),
            controller_lineage_uuid=(
                completed_ticket.controller_lineage_uuid
            ),
            lineage_ordinal=completed_ticket.lineage_ordinal,
            predecessor_completed_ticket_sha256=(
                completed_ticket.predecessor_completed_ticket_sha256
            ),
        )
        return replace(
            record, _attestation=_issue_previous(record._binding())
        )

    @property
    def quality_valid(self) -> bool:
        return self.quality_normalized is not None

    @property
    def latency_valid(self) -> bool:
        return self.latency_normalized is not None

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "available_ns": self.available_ns,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "controller_lineage_uuid": self.controller_lineage_uuid,
            "decision_seq": self.decision_seq,
            "eligibility": self.eligibility,
            "executed_action": self.action.to_canonical_dict(),
            "latency_clock_domain": self.latency_clock_domain,
            "latency_normalized": (
                None
                if self.latency_normalized is None
                else float(self.latency_normalized)
            ),
            "latency_valid": self.latency_valid,
            "lineage_ordinal": self.lineage_ordinal,
            "outcome_sha256": self.outcome_sha256,
            "quality_gt_source": self.quality_gt_source,
            "quality_normalized": (
                None
                if self.quality_normalized is None
                else float(self.quality_normalized)
            ),
            "quality_valid": self.quality_valid,
            "predecessor_completed_ticket_sha256": (
                self.predecessor_completed_ticket_sha256
            ),
            "record": "previous_outcome_v1",
            "resolution_ns": self.resolution_ns,
            "reward_spec_sha256": self.reward_spec_sha256,
            "session_uuid": self.session_uuid,
            "terminal_class": self.terminal_class.value,
        }


@dataclass(frozen=True, slots=True)
class CausalStateV1:
    """One immutable causal observation, complete, typed and structurally valid.

    Every measurement arrives as a timestamped observation record, so each age
    is *derived* as ``observed_ns - measured_ns`` and a caller cannot assert a
    convenient zero.  A measurement that postdates ``observed_ns`` is refused.

    Freshness is a separate, explicitly parameterized decision: see
    :class:`StateFreshnessPolicyV1`, which :func:`build_policy_features`
    requires, so a stale state can never be vectorized.

    Because the previous outcome carries privileged CARLA quality, this whole
    observation schema is :data:`POLICY_OBSERVATION_DEPLOYABILITY`.
    """

    scene: SceneObservationV1
    radio: RadioObservationV1
    session_uuid: str
    observed_ns: int
    tensor_seq: int
    carla_frame_id: int
    clock_domain: ClockDomain = ClockDomain.UE_LOCAL_MONOTONIC
    previous: Optional[PreviousOutcomeV1] = None
    episode_start: Optional[EpisodeStartProofV1] = None

    def __post_init__(self) -> None:
        E = CausalStateError
        if not isinstance(self.scene, SceneObservationV1):
            raise E(
                f"scene must be a timestamped SceneObservationV1, got "
                f"{type(self.scene).__name__}"
            )
        if not isinstance(self.radio, RadioObservationV1):
            raise E(
                f"radio must be a typed RadioObservationV1, got "
                f"{type(self.radio).__name__}"
            )
        self.radio.require_attested()
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.observed_ns, "observed_ns", E)
        _non_negative_int(self.tensor_seq, "tensor_seq", E)
        _non_negative_int(self.carla_frame_id, "carla_frame_id", E)
        if not isinstance(self.clock_domain, ClockDomain):
            raise ClockDomainError(
                f"clock_domain must be a ClockDomain, got "
                f"{type(self.clock_domain).__name__}"
            )
        for label, observation in (("scene", self.scene), ("radio", self.radio)):
            if observation.clock_domain is not self.clock_domain:
                raise ClockDomainError(
                    f"the {label} observation is in clock domain "
                    f"{observation.clock_domain.value} but the state is in "
                    f"{self.clock_domain.value}; durations are never computed "
                    f"across domains"
                )
        # Deriving the ages validates that no measurement postdates the state.
        self.scene.age_ns(self.observed_ns)
        self.radio.ages_ns(self.observed_ns)
        # The scene descriptor must belong to this state's frame.
        if self.scene.carla_frame_id != self.carla_frame_id:
            raise E(
                f"the scene observation is bound to CARLA frame "
                f"{self.scene.carla_frame_id} but the state is at frame "
                f"{self.carla_frame_id}; a descriptor is never reused across "
                f"frames"
            )
        if (self.previous is None) == (self.episode_start is None):
            raise E(
                "a causal state must carry exactly one predecessor proof: "
                "either the policy-visible previous outcome or an explicit "
                "episode-start proof"
            )
        if self.previous is not None:
            if not isinstance(self.previous, PreviousOutcomeV1):
                raise E(
                    f"previous must be a PreviousOutcomeV1 or None (explicit "
                    f"absence at episode start), got "
                    f"{type(self.previous).__name__}"
                )
            self.previous.require_attested()
            if self.previous.session_uuid != self.session_uuid:
                raise E(
                    f"the previous outcome belongs to session "
                    f"{self.previous.session_uuid}, not this state's "
                    f"{self.session_uuid}"
                )
            if self.previous.available_ns > self.observed_ns:
                raise E(
                    f"the previous outcome became available at "
                    f"{self.previous.available_ns} ns, after this state was "
                    f"observed at {self.observed_ns} ns; later adjudication "
                    f"or feedback may never leak into an earlier policy state"
                )
        else:
            start = self.episode_start
            assert start is not None
            if not isinstance(start, EpisodeStartProofV1):
                raise E(
                    "episode_start must be an EpisodeStartProofV1 when no "
                    "previous policy outcome exists"
                )
            start.require_attested()
            if start.session_uuid != self.session_uuid:
                raise E(
                    f"episode-start proof session {start.session_uuid} does "
                    f"not match state session {self.session_uuid}"
                )
            if start.first_tensor_seq != self.tensor_seq:
                raise E(
                    f"episode-start proof names first tensor "
                    f"{start.first_tensor_seq}, not state tensor "
                    f"{self.tensor_seq}"
                )
            if start.first_carla_frame_id != self.carla_frame_id:
                raise E(
                    f"episode-start proof names first CARLA frame "
                    f"{start.first_carla_frame_id}, not state frame "
                    f"{self.carla_frame_id}"
                )
            if start.episode_started_ns != self.observed_ns:
                raise E(
                    f"episode-start authorization binds policy observation "
                    f"{start.episode_started_ns} ns, not this state's exact "
                    f"observation instant {self.observed_ns} ns"
                )

    # -- derived ----------------------------------------------------------- #

    @property
    def has_previous_decision(self) -> bool:
        return self.previous is not None

    @property
    def camera_si(self) -> float:
        return self.scene.camera_si

    @property
    def radar_p40(self) -> float:
        return self.scene.radar_p40

    @property
    def measurement_ages_ns(self) -> Mapping[str, int]:
        """All four derived ages, keyed ``scene``/``snr``/``bsr``/``mcs``."""
        ages = {"scene": self.scene.age_ns(self.observed_ns)}
        ages.update(self.radio.ages_ns(self.observed_ns))
        return MappingProxyType(ages)

    @classmethod
    def metadata_field_names(cls) -> Tuple[str, ...]:
        """Fields that are join/audit metadata and never policy features."""
        return (
            "session_uuid",
            "observed_ns",
            "tensor_seq",
            "carla_frame_id",
            "source_id",
            "source_sha256",
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "carla_frame_id": self.carla_frame_id,
            "clock_domain": self.clock_domain.value,
            "deployability": POLICY_OBSERVATION_DEPLOYABILITY,
            "episode_start": (
                None
                if self.episode_start is None
                else self.episode_start.to_canonical_dict()
            ),
            "measurement_ages_ns": dict(self.measurement_ages_ns),
            "observed_ns": self.observed_ns,
            "previous": (
                None if self.previous is None else self.previous.to_canonical_dict()
            ),
            "radio": self.radio.to_canonical_dict(),
            "record": "causal_state_v1",
            "scene": self.scene.to_canonical_dict(),
            "session_uuid": self.session_uuid,
            "tensor_seq": self.tensor_seq,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# D (continued). Freshness, normalization and the frozen policy feature order
# --------------------------------------------------------------------------- #

#: One-hot width of the previous joint mode, bound to the Phase-1 catalog.
_PREVIOUS_MODE_ONEHOT_WIDTH = EXPECTED_MODE_COUNT

#: The previous decision's terminal classification, one-hot in a fixed order, so
#: a feedback timeout and an action-path failure are never the same observation.
PREVIOUS_TERMINAL_ORDER: Tuple[TerminalClass, ...] = (
    TerminalClass.REWARD_FINAL_EXACT,
    TerminalClass.ACTION_PATH_FAILURE,
    TerminalClass.FEEDBACK_TIMEOUT,
    TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
)

#: Compact feature codes for the previous terminal class.  Deliberately not the
#: raw enum values: ``REWARD_FINAL_EXACT`` would put the substring "reward" in a
#: feature name, and the forbidden-substring guard exists precisely to stop a
#: *reward* term leaking into the observation.  The previous decision's terminal
#: class is a legitimate causal observation, so it gets an unambiguous short
#: code instead.
PREVIOUS_TERMINAL_FEATURE_CODES: Mapping[TerminalClass, str] = MappingProxyType(
    {
        TerminalClass.REWARD_FINAL_EXACT: "exact",
        TerminalClass.ACTION_PATH_FAILURE: "action_path_failure",
        TerminalClass.FEEDBACK_TIMEOUT: "feedback_timeout",
        TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED: "infra_fault_excluded",
    }
)

#: The single frozen policy-feature order.  Fixed width, names and positions.
#: Every entry is a scaled observation, a normalized measurement age, a compact
#: previous-outcome code or an explicit presence/validity mask.  No identifier,
#: ground truth of the *current* frame, current-decision outcome or future
#: measurement appears.
POLICY_FEATURE_ORDER: Tuple[str, ...] = (
    "scene_camera_si_scaled",
    "scene_radar_p40",
    "radio_achieved_snr_db_scaled",
    "radio_bsr_log1p_scaled",
    "radio_mcs_index_scaled",
    "freshness_scene_normalized",
    "freshness_snr_normalized",
    "freshness_bsr_normalized",
    "freshness_mcs_normalized",
) + tuple(
    f"prev_joint_mode_onehot_{index:02d}"
    for index in range(_PREVIOUS_MODE_ONEHOT_WIDTH)
) + tuple(
    f"prev_terminal_onehot_{PREVIOUS_TERMINAL_FEATURE_CODES[terminal]}"
    for terminal in PREVIOUS_TERMINAL_ORDER
) + (
    "prev_q_normalized",
    "prev_quality_normalized",
    "prev_latency_normalized",
    "prev_present_mask",
    "prev_quality_valid_mask",
    "prev_latency_valid_mask",
)

POLICY_FEATURE_COUNT: int = len(POLICY_FEATURE_ORDER)

#: Substrings that must never appear in a policy-feature name.
FORBIDDEN_POLICY_FEATURE_SUBSTRINGS: Tuple[str, ...] = (
    "session",
    "uuid",
    "decision_seq",
    "tensor_seq",
    "frame_id",
    "carla_frame",
    "action_id",
    "profile_id",
    "ground_truth",
    "gt_",
    "iou",
    "localization_error",
    "reward",
    "next_",
    "future",
    "profile_label",
    "network_profile",
    "object_count",
    "dynamic_fraction",
    "temporal_information",
    "age_ns",
    "sha256",
    "observed_ns",
    "measured_ns",
)


def assert_policy_features_exclude_forbidden_fields() -> None:
    """Fail closed if the frozen feature order ever gains a forbidden field."""
    if len(set(POLICY_FEATURE_ORDER)) != len(POLICY_FEATURE_ORDER):
        raise StateRewardContractError(
            "POLICY_FEATURE_ORDER contains duplicate feature names"
        )
    for name in POLICY_FEATURE_ORDER:
        _non_empty_str(name, "policy feature name", StateRewardContractError)
        for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
            if forbidden in name:
                raise StateRewardContractError(
                    f"policy feature {name!r} contains the forbidden substring "
                    f"{forbidden!r}: identifiers, current-frame ground truth, "
                    f"the current decision's outcome, anchor lookups, raw "
                    f"timestamps and future telemetry are not causal policy "
                    f"inputs"
                )


assert_policy_features_exclude_forbidden_fields()


@dataclass(frozen=True, slots=True)
class StateFreshnessPolicyV1:
    """Explicit per-source maximum measurement ages, hash-bound.

    No default: every bound is an engineering decision that must be stated.  A
    state older than its bound fails closed for the external
    registered-fallback guard rather than being used or patched.  The bounds
    also serve as the denominators for the normalized age features, so a
    transition binds :meth:`canonical_sha256` -- not merely the free-form id --
    and two policies with the same name but different bounds are distinguishable.
    """

    policy_id: str
    max_scene_age_ns: int
    max_snr_age_ns: int
    max_bsr_age_ns: int
    max_mcs_age_ns: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        E = NormalizationSpecError
        _non_empty_str(self.policy_id, "policy_id", E)
        for name in (
            "max_scene_age_ns",
            "max_snr_age_ns",
            "max_bsr_age_ns",
            "max_mcs_age_ns",
        ):
            _positive_int(getattr(self, name), name, E)
        object.__setattr__(
            self,
            "provenance",
            _frozen_str_mapping(self.provenance, "provenance", E),
        )

    def bound_for(self, source: str) -> int:
        try:
            return int(getattr(self, f"max_{source}_age_ns"))
        except AttributeError as exc:
            raise NormalizationSpecError(
                f"no freshness bound registered for source {source!r}"
            ) from exc

    def assert_fresh(self, state: CausalStateV1) -> None:
        """Fail closed when any required observation is too old."""
        if not isinstance(state, CausalStateV1):
            raise CausalStateError(
                f"state must be a CausalStateV1, got {type(state).__name__}"
            )
        for source, age in state.measurement_ages_ns.items():
            bound = self.bound_for(source)
            if age > bound:
                raise StaleTelemetryError(
                    f"{source} measurement age {age} ns exceeds the registered "
                    f"bound {bound} ns of freshness policy {self.policy_id!r} "
                    f"({self.canonical_sha256()}); this state must not be used "
                    f"and the external guard must select the registered "
                    f"fallback action.  Nothing is substituted for the stale "
                    f"value"
                )

    def normalized_age(self, source: str, age_ns: int) -> float:
        """``age / bound``, clipped to [0, 1]; the bound is the denominator."""
        return min(max(float(age_ns) / float(self.bound_for(source)), 0.0), 1.0)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "max_bsr_age_ns": self.max_bsr_age_ns,
            "max_mcs_age_ns": self.max_mcs_age_ns,
            "max_scene_age_ns": self.max_scene_age_ns,
            "max_snr_age_ns": self.max_snr_age_ns,
            "policy_id": self.policy_id,
            "provenance": dict(self.provenance),
            "record": "state_freshness_policy_v1",
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class StateNormalizationSpecV1:
    """Explicitly provenanced scaling constants for the policy feature vector.

    **No fitted constant is invented here.**  Every empirical value is
    constructor supplied and the record refuses to exist without a train-split
    identifier, a positive fit population count and a fit-config SHA-256.

    ``snr_metric``, ``mcs_table_id`` and ``bsr_scope`` are part of the spec, not
    only of the state: scaling a differently-defined measurement by these
    constants would be a silent unit error, so
    :func:`build_policy_features` requires the state's semantics to match.
    """

    spec_id: str
    spec_version: int
    train_split_id: str
    fit_population_count: int
    fit_config_sha256: str
    camera_si_clip_min: float
    camera_si_clip_max: float
    achieved_snr_db_clip_min: float
    achieved_snr_db_clip_max: float
    bsr_log1p_scale: float
    snr_metric: SnrMetric
    mcs_table_id: str
    mcs_table_max_index: int
    bsr_scope: BsrScope
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        E = NormalizationSpecError
        _non_empty_str(self.spec_id, "spec_id", E)
        _positive_int(self.spec_version, "spec_version", E)
        _non_empty_str(self.train_split_id, "train_split_id", E)
        _positive_int(self.fit_population_count, "fit_population_count", E)
        _sha256_hex(self.fit_config_sha256, "fit_config_sha256", E)

        si_lo = _finite_float(self.camera_si_clip_min, "camera_si_clip_min", E)
        si_hi = _finite_float(self.camera_si_clip_max, "camera_si_clip_max", E)
        if si_lo < 0.0:
            raise E("camera_si_clip_min must be >= 0 (SI is non-negative)")
        if si_hi <= si_lo:
            raise E(
                f"camera_si_clip_max must exceed camera_si_clip_min; got "
                f"{si_hi} <= {si_lo}"
            )
        snr_lo = _finite_float(
            self.achieved_snr_db_clip_min, "achieved_snr_db_clip_min", E
        )
        snr_hi = _finite_float(
            self.achieved_snr_db_clip_max, "achieved_snr_db_clip_max", E
        )
        if snr_hi <= snr_lo:
            raise E(
                f"achieved_snr_db_clip_max must exceed "
                f"achieved_snr_db_clip_min; got {snr_hi} <= {snr_lo}"
            )
        scale = _finite_float(self.bsr_log1p_scale, "bsr_log1p_scale", E)
        if scale <= 0.0:
            raise E(f"bsr_log1p_scale must be > 0, got {scale}")
        if not isinstance(self.snr_metric, SnrMetric):
            raise E(
                f"snr_metric must be a typed SnrMetric so the spec's clip "
                f"bounds cannot be applied to a different quantity, got "
                f"{type(self.snr_metric).__name__}"
            )
        if not isinstance(self.bsr_scope, BsrScope):
            raise E(
                f"bsr_scope must be a typed BsrScope, got "
                f"{type(self.bsr_scope).__name__}"
            )
        _non_empty_str(self.mcs_table_id, "mcs_table_id", E)
        _positive_int(self.mcs_table_max_index, "mcs_table_max_index", E)
        object.__setattr__(
            self, "provenance", _frozen_str_mapping(self.provenance, "provenance", E)
        )

    # -- named scalar transformations -------------------------------------- #

    def scale_camera_si(self, camera_si: float) -> float:
        """Configured robust clip and scale of raw SI into [0, 1]."""
        lo = float(self.camera_si_clip_min)
        hi = float(self.camera_si_clip_max)
        return (min(max(float(camera_si), lo), hi) - lo) / (hi - lo)

    def scale_achieved_snr_db(self, snr_db: float) -> float:
        """Configured clip and scale of achieved SNR into [0, 1]."""
        lo = float(self.achieved_snr_db_clip_min)
        hi = float(self.achieved_snr_db_clip_max)
        return (min(max(float(snr_db), lo), hi) - lo) / (hi - lo)

    def scale_bsr_bytes(self, bsr_bytes: int) -> float:
        """``log1p`` of the buffer occupancy, then the configured scaling."""
        value = math.log1p(float(bsr_bytes)) / float(self.bsr_log1p_scale)
        return min(max(value, 0.0), 1.0)

    def scale_mcs_index(self, mcs_index: int) -> float:
        """Scale an MCS index by the explicitly supplied table maximum."""
        if mcs_index > self.mcs_table_max_index:
            raise NormalizationSpecError(
                f"mcs_index {mcs_index} exceeds the declared maximum "
                f"{self.mcs_table_max_index} of table {self.mcs_table_id!r}"
            )
        return float(mcs_index) / float(self.mcs_table_max_index)

    def assert_semantics_match(self, radio: RadioObservationV1) -> None:
        """Fail closed if the observation is not the quantity this spec scales."""
        if radio.snr_metric is not self.snr_metric:
            raise NormalizationSpecError(
                f"the state reports {radio.snr_metric.value} but this spec's "
                f"clip bounds were fitted for {self.snr_metric.value}; scaling "
                f"one metric with another's bounds is a silent unit error"
            )
        if radio.mcs_table_id != self.mcs_table_id:
            raise NormalizationSpecError(
                f"state MCS table {radio.mcs_table_id!r} does not match the "
                f"spec's {self.mcs_table_id!r}"
            )
        if radio.bsr_scope is not self.bsr_scope:
            raise NormalizationSpecError(
                f"the state reports a {radio.bsr_scope.value} BSR but the spec "
                f"was fitted for {self.bsr_scope.value}"
            )

    @staticmethod
    def normalize_previous_q(action: ExecutedActionIdentity) -> float:
        """``q_e4 / 9800`` -- the registered wire value, not an anchor lookup."""
        return float(action.q_e4) / float(Q_E4_MAX)

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db_clip_max": float(self.achieved_snr_db_clip_max),
            "achieved_snr_db_clip_min": float(self.achieved_snr_db_clip_min),
            "bsr_log1p_scale": float(self.bsr_log1p_scale),
            "bsr_scope": self.bsr_scope.value,
            "camera_si_clip_max": float(self.camera_si_clip_max),
            "camera_si_clip_min": float(self.camera_si_clip_min),
            "fit_config_sha256": self.fit_config_sha256,
            "fit_population_count": self.fit_population_count,
            "mcs_table_id": self.mcs_table_id,
            "mcs_table_max_index": self.mcs_table_max_index,
            "policy_feature_order": list(POLICY_FEATURE_ORDER),
            "provenance": dict(self.provenance),
            "record": "state_normalization_spec_v1",
            "snr_metric": self.snr_metric.value,
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "train_split_id": self.train_split_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class PolicyFeatureVectorV1(_Attested):
    """A fixed-width, deterministically ordered policy feature vector.

    Marked :data:`POLICY_OBSERVATION_DEPLOYABILITY` because it carries the
    previous decision's privileged CARLA ground-truth quality.  Nothing about
    this vector is claimed to be observable in a physical deployment.
    """

    values: Tuple[float, ...]
    source_state_sha256: str
    normalization: StateNormalizationSpecV1
    freshness: StateFreshnessPolicyV1
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = StateRewardContractError
        if not isinstance(self.values, tuple):
            raise E(f"values must be a tuple, got {type(self.values).__name__}")
        if len(self.values) != POLICY_FEATURE_COUNT:
            raise E(
                f"expected exactly {POLICY_FEATURE_COUNT} features in the "
                f"frozen order, got {len(self.values)}"
            )
        for name, value in zip(POLICY_FEATURE_ORDER, self.values):
            if type(value) is not float or not math.isfinite(value):
                raise E(
                    f"feature {name!r} must be a finite float, got "
                    f"{type(value).__name__}: {value!r}"
                )
        _sha256_hex(self.source_state_sha256, "source_state_sha256", E)
        if not isinstance(self.normalization, StateNormalizationSpecV1):
            raise E(
                "normalization must be the complete "
                "StateNormalizationSpecV1, not only an asserted hash"
            )
        if not isinstance(self.freshness, StateFreshnessPolicyV1):
            raise E(
                "freshness must be the complete StateFreshnessPolicyV1, not "
                "only an asserted hash"
            )
        if self._attestation is not None and not _valid_features(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the policy-feature attestation does not match the source "
                "state, values or complete preprocessing specifications"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_features

    @property
    def state_normalization_spec_sha256(self) -> str:
        return self.normalization.canonical_sha256()

    @property
    def freshness_policy_sha256(self) -> str:
        return self.freshness.canonical_sha256()

    @property
    def freshness_policy_id(self) -> str:
        return self.freshness.policy_id

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return POLICY_FEATURE_ORDER

    @property
    def deployability(self) -> str:
        return POLICY_OBSERVATION_DEPLOYABILITY

    def as_tuple(self) -> Tuple[float, ...]:
        """The values, in frozen order.  Tuples are already immutable."""
        return self.values

    def as_mapping(self) -> Dict[str, float]:
        """A fresh defensive ``name -> value`` copy; mutating it is harmless."""
        return dict(zip(POLICY_FEATURE_ORDER, self.values))

    def assert_binds(self, state: CausalStateV1) -> None:
        """Rebuild this vector from its full sources and compare exactly."""
        self.require_attested()
        if not isinstance(state, CausalStateV1):
            raise TransitionIdentityError(
                f"feature source must be CausalStateV1, got "
                f"{type(state).__name__}"
            )
        if self.source_state_sha256 != state.canonical_sha256():
            raise TransitionIdentityError(
                "policy features are bound to a different causal state"
            )
        rebuilt = build_policy_features(state, self.normalization, self.freshness)
        if rebuilt.canonical_sha256() != self.canonical_sha256():
            raise TransitionIdentityError(
                "policy features do not survive recomputation from the "
                "bound state and complete preprocessing specifications"
            )

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "deployability": self.deployability,
            "feature_order": list(POLICY_FEATURE_ORDER),
            "freshness_policy": self.freshness.to_canonical_dict(),
            "freshness_policy_id": self.freshness_policy_id,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "normalization": self.normalization.to_canonical_dict(),
            "record": "policy_feature_vector_v1",
            "source_state_sha256": self.source_state_sha256,
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
            "values": [float(value) for value in self.values],
        }


def build_policy_features(
    state: CausalStateV1,
    normalization: StateNormalizationSpecV1,
    freshness: StateFreshnessPolicyV1,
) -> PolicyFeatureVectorV1:
    """Map a fresh causal state into the frozen policy feature order.

    Freshness is checked first and unconditionally, so a stale observation can
    never reach a policy.  Radio semantics are checked against the normalization
    spec, so a differently-defined SNR, MCS table or BSR scope cannot be scaled
    by the wrong constants.

    When no previous decision exists, its block is written as zeros *behind
    explicit presence and validity masks*.  That is not a silent substitution:
    the masks are themselves features, so absence is representable and
    distinguishable from a real zero.
    """
    if not isinstance(state, CausalStateV1):
        raise CausalStateError(
            f"state must be a CausalStateV1, got {type(state).__name__}"
        )
    if not isinstance(normalization, StateNormalizationSpecV1):
        raise NormalizationSpecError(
            f"normalization must be a StateNormalizationSpecV1 with explicit "
            f"train-fit provenance, got {type(normalization).__name__}"
        )
    if not isinstance(freshness, StateFreshnessPolicyV1):
        raise NormalizationSpecError(
            f"freshness must be a StateFreshnessPolicyV1, got "
            f"{type(freshness).__name__}"
        )
    freshness.assert_fresh(state)
    normalization.assert_semantics_match(state.radio)

    ages = state.measurement_ages_ns
    named: Dict[str, float] = {
        "scene_camera_si_scaled": normalization.scale_camera_si(state.camera_si),
        "scene_radar_p40": float(state.radar_p40),
        "radio_achieved_snr_db_scaled": normalization.scale_achieved_snr_db(
            state.radio.achieved_snr_db
        ),
        "radio_bsr_log1p_scaled": normalization.scale_bsr_bytes(
            state.radio.bsr_bytes
        ),
        "radio_mcs_index_scaled": normalization.scale_mcs_index(
            state.radio.mcs_index
        ),
        "freshness_scene_normalized": freshness.normalized_age(
            "scene", ages["scene"]
        ),
        "freshness_snr_normalized": freshness.normalized_age("snr", ages["snr"]),
        "freshness_bsr_normalized": freshness.normalized_age("bsr", ages["bsr"]),
        "freshness_mcs_normalized": freshness.normalized_age("mcs", ages["mcs"]),
    }

    previous = state.previous
    one_hot = [0.0] * _PREVIOUS_MODE_ONEHOT_WIDTH
    if previous is not None:
        mode_id = previous.action.mode_id
        if not 0 <= mode_id < _PREVIOUS_MODE_ONEHOT_WIDTH:  # pragma: no cover
            raise CausalStateError(
                f"previous mode_id {mode_id} is outside the catalog's "
                f"{_PREVIOUS_MODE_ONEHOT_WIDTH} joint modes"
            )
        one_hot[mode_id] = 1.0
    for index, value in enumerate(one_hot):
        named[f"prev_joint_mode_onehot_{index:02d}"] = value

    for terminal in PREVIOUS_TERMINAL_ORDER:
        named[
            f"prev_terminal_onehot_"
            f"{PREVIOUS_TERMINAL_FEATURE_CODES[terminal]}"
        ] = (
            1.0
            if previous is not None and previous.terminal_class is terminal
            else 0.0
        )

    named["prev_q_normalized"] = (
        0.0
        if previous is None
        else StateNormalizationSpecV1.normalize_previous_q(previous.action)
    )
    named["prev_quality_normalized"] = (
        0.0
        if previous is None or previous.quality_normalized is None
        else float(previous.quality_normalized)
    )
    named["prev_latency_normalized"] = (
        0.0
        if previous is None or previous.latency_normalized is None
        else float(previous.latency_normalized)
    )
    named["prev_present_mask"] = 0.0 if previous is None else 1.0
    named["prev_quality_valid_mask"] = (
        1.0 if previous is not None and previous.quality_valid else 0.0
    )
    named["prev_latency_valid_mask"] = (
        1.0 if previous is not None and previous.latency_valid else 0.0
    )

    missing = set(POLICY_FEATURE_ORDER) - set(named)
    extra = set(named) - set(POLICY_FEATURE_ORDER)
    if missing or extra:  # pragma: no cover - guarded by the frozen order
        raise StateRewardContractError(
            f"feature mapping disagrees with the frozen order; missing "
            f"{sorted(missing)}, unexpected {sorted(extra)}"
        )
    record = PolicyFeatureVectorV1(
        values=tuple(float(named[name]) for name in POLICY_FEATURE_ORDER),
        source_state_sha256=state.canonical_sha256(),
        normalization=normalization,
        freshness=freshness,
    )
    return replace(record, _attestation=_issue_features(record._binding()))


# --------------------------------------------------------------------------- #
# E. Adjudication and the validated decision outcome
# --------------------------------------------------------------------------- #


class Adjudication(Enum):
    """Reserved verdict vocabulary for a future reconciliation protocol.

    Phase 4a.2 has no verified carrier for these verdicts, so
    :func:`evaluate_completed_decision` rejects every supplied adjudication and
    leaves feedback timeouts censored.  The enum and record describe the
    intended future semantics; their presence does not make them admissible.
    """

    PENDING = "PENDING"
    FEEDBACK_ONLY_LOSS = "FEEDBACK_ONLY_LOSS"
    AUTHORITATIVE_SERVICE_FAILURE = "AUTHORITATIVE_SERVICE_FAILURE"
    INFRASTRUCTURE_FAULT = "INFRASTRUCTURE_FAULT"


@dataclass(frozen=True, slots=True)
class AdjudicationRecordV1:
    """Reserved ticket-bound verdict record; not admissible in Phase 4a.2.

    Binds the session, the decision, the ticket hash and the terminal class, and
    requires ``adjudicated_ns >= ticket.closed_ns`` -- a verdict cannot predate
    the event it adjudicates.  Because the ticket hash is part of the record, an
    adjudication is **not reusable** for another ticket.  These structural
    checks are necessary but not sufficient evidence: until a reviewed carrier
    can issue this record, the reward evaluator refuses it.
    """

    verdict: Adjudication
    session_uuid: str
    decision_seq: int
    completed_ticket_sha256: str
    terminal_class: TerminalClass
    adjudicator_id: str
    evidence_sha256: str
    adjudicated_ns: int
    detail: str

    def __post_init__(self) -> None:
        E = AdjudicationError
        if not isinstance(self.verdict, Adjudication):
            raise E(
                f"verdict must be an Adjudication, got "
                f"{type(self.verdict).__name__}: {self.verdict!r}"
            )
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.decision_seq, "decision_seq", E)
        _sha256_hex(self.completed_ticket_sha256, "completed_ticket_sha256", E)
        if not isinstance(self.terminal_class, TerminalClass):
            raise E(
                f"terminal_class must be a controller TerminalClass, got "
                f"{type(self.terminal_class).__name__}"
            )
        _non_empty_str(self.adjudicator_id, "adjudicator_id", E)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", E)
        _non_negative_int(self.adjudicated_ns, "adjudicated_ns", E)
        _non_empty_str(self.detail, "detail", E)

    @classmethod
    def for_ticket(
        cls,
        completed_ticket: CompletedTicket,
        *,
        verdict: Adjudication,
        adjudicator_id: str,
        evidence_sha256: str,
        adjudicated_ns: int,
        detail: str,
    ) -> "AdjudicationRecordV1":
        """Bind a verdict to exactly one completed ticket."""
        if not isinstance(completed_ticket, CompletedTicket):
            raise AdjudicationError(
                f"completed_ticket must be a controller CompletedTicket, got "
                f"{type(completed_ticket).__name__}"
            )
        _non_negative_int(adjudicated_ns, "adjudicated_ns", AdjudicationError)
        if adjudicated_ns < completed_ticket.closed_ns:
            raise AdjudicationError(
                f"adjudicated_ns {adjudicated_ns} predates the ticket's "
                f"closure at {completed_ticket.closed_ns} ns; a reconciliation "
                f"verdict cannot precede the event it adjudicates"
            )
        return cls(
            verdict=verdict,
            session_uuid=completed_ticket.session_uuid,
            decision_seq=completed_ticket.decision_seq,
            completed_ticket_sha256=completed_ticket.canonical_sha256(),
            terminal_class=completed_ticket.terminal_class,
            adjudicator_id=adjudicator_id,
            evidence_sha256=evidence_sha256,
            adjudicated_ns=adjudicated_ns,
            detail=detail,
        )

    def assert_binds(self, completed_ticket: CompletedTicket) -> None:
        """Fail closed unless this verdict was issued for exactly this ticket."""
        if self.session_uuid != completed_ticket.session_uuid:
            raise AdjudicationError(
                f"the adjudication belongs to session {self.session_uuid}, not "
                f"{completed_ticket.session_uuid}"
            )
        if self.decision_seq != completed_ticket.decision_seq:
            raise AdjudicationError(
                f"the adjudication belongs to decision {self.decision_seq}, "
                f"not {completed_ticket.decision_seq}; a verdict is never "
                f"reused for another decision"
            )
        if self.completed_ticket_sha256 != completed_ticket.canonical_sha256():
            raise AdjudicationError(
                f"the adjudication is bound to ticket "
                f"{self.completed_ticket_sha256} but was offered for "
                f"{completed_ticket.canonical_sha256()}; an adjudication is "
                f"not reusable across tickets"
            )
        if self.terminal_class is not completed_ticket.terminal_class:
            raise AdjudicationError(
                f"the adjudication was issued for terminal class "
                f"{self.terminal_class.value}, not "
                f"{completed_ticket.terminal_class.value}"
            )
        if self.adjudicated_ns < completed_ticket.closed_ns:
            raise AdjudicationError(
                f"adjudicated_ns {self.adjudicated_ns} predates the ticket's "
                f"closure at {completed_ticket.closed_ns} ns"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "adjudicated_ns": self.adjudicated_ns,
            "adjudicator_id": self.adjudicator_id,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "decision_seq": self.decision_seq,
            "detail": self.detail,
            "evidence_sha256": self.evidence_sha256,
            "record": "adjudication_record_v1",
            "session_uuid": self.session_uuid,
            "terminal_class": self.terminal_class.value,
            "verdict": self.verdict.value,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


class LearningEligibility(Enum):
    """Whether and why a transition may contribute to learning."""

    ELIGIBLE = "ELIGIBLE"
    CENSORED_PENDING_ADJUDICATION = "CENSORED_PENDING_ADJUDICATION"
    CENSORED_FEEDBACK_ONLY_LOSS = "CENSORED_FEEDBACK_ONLY_LOSS"
    CENSORED_NO_ELIGIBLE_GROUND_TRUTH = "CENSORED_NO_ELIGIBLE_GROUND_TRUTH"
    EXCLUDED_INFRASTRUCTURE_FAULT = "EXCLUDED_INFRASTRUCTURE_FAULT"


@dataclass(frozen=True, slots=True)
class DecisionOutcomeV1(_Attested):
    """The complete measured outcome of one completed decision.

    Build with :func:`evaluate_completed_decision`; a directly constructed
    instance is unattested and cannot serialize or enter a transition.
    """

    terminal_class: TerminalClass
    eligibility: LearningEligibility
    costs: ConstraintCostsV1
    diagnostics: DiagnosticSignalsV1
    switch_penalty: SwitchPenaltyV1
    completed_ticket_sha256: str
    reward_spec_sha256: str
    #: The *raw* components this outcome was measured from, retained even when
    #: no evaluation resulted (a censored no-eligible-ground-truth frame), so
    #: the outcome can always be recomputed from its own frozen source rather
    #: than trusted.
    quality_components: Optional[QualityComponentsV1] = None
    quality: Optional[QualityEvaluationV1] = None
    latency: Optional[LatencyMeasurementV1] = None
    scalar_reward: Optional[float] = None
    adjudication: Optional[AdjudicationRecordV1] = None
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = StateRewardContractError
        if not isinstance(self.terminal_class, TerminalClass):
            raise E("terminal_class must be a controller TerminalClass")
        if not isinstance(self.eligibility, LearningEligibility):
            raise E("eligibility must be a LearningEligibility")
        if not isinstance(self.costs, ConstraintCostsV1):
            raise E("costs must be a ConstraintCostsV1")
        if not isinstance(self.diagnostics, DiagnosticSignalsV1):
            raise E("diagnostics must be a DiagnosticSignalsV1")
        if not isinstance(self.switch_penalty, SwitchPenaltyV1):
            raise E("switch_penalty must be a SwitchPenaltyV1")
        _sha256_hex(self.completed_ticket_sha256, "completed_ticket_sha256", E)
        _sha256_hex(self.reward_spec_sha256, "reward_spec_sha256", E)
        if self.quality_components is not None and not isinstance(
            self.quality_components, QualityComponentsV1
        ):
            raise E("quality_components must be a QualityComponentsV1 or None")
        if self.quality is not None:
            if not isinstance(self.quality, QualityEvaluationV1):
                raise E("quality must be a QualityEvaluationV1 or None")
            self.quality.require_attested()
            if self.quality_components is None:
                raise E(
                    "an evaluated quality must retain the raw components it "
                    "was derived from"
                )
            if (
                self.quality.components.canonical_sha256()
                != self.quality_components.canonical_sha256()
            ):
                raise E(
                    "the evaluation was derived from different raw components "
                    "than the outcome retains"
                )
        if self.latency is not None:
            if not isinstance(self.latency, LatencyMeasurementV1):
                raise E("latency must be a LatencyMeasurementV1 or None")
            self.latency.require_attested()
        if self.scalar_reward is not None:
            _finite_float(self.scalar_reward, "scalar_reward", E)
        if self.eligibility is not LearningEligibility.ELIGIBLE and (
            self.scalar_reward is not None
        ):
            raise E(
                f"eligibility {self.eligibility.value} carries no scalar "
                f"reward; got {self.scalar_reward!r}"
            )
        if self.eligibility is LearningEligibility.ELIGIBLE and (
            self.scalar_reward is None
        ):
            raise E("an eligible outcome must carry a scalar reward")
        if self.terminal_class is TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
            if self.eligibility is not (
                LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT
            ):
                raise E("an infrastructure fault must be excluded, never scored")
            if self.costs.c_authoritative_failure == 1.0:
                raise E(
                    "an infrastructure fault must never be recorded as an "
                    "authoritative service failure"
                )
        if self._attestation is not None and not _valid_outcome(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the decision outcome's attestation does not match its own "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_outcome

    @property
    def learning_eligible(self) -> bool:
        return self.eligibility is LearningEligibility.ELIGIBLE

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "adjudication": (
                None
                if self.adjudication is None
                else self.adjudication.to_canonical_dict()
            ),
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "costs": self.costs.to_canonical_dict(),
            "diagnostics": self.diagnostics.to_canonical_dict(),
            "eligibility": self.eligibility.value,
            "latency": (
                None if self.latency is None else self.latency.to_canonical_dict()
            ),
            "quality": (
                None if self.quality is None else self.quality.to_canonical_dict()
            ),
            "quality_components_sha256": (
                None
                if self.quality_components is None
                else self.quality_components.canonical_sha256()
            ),
            "record": "decision_outcome_v1",
            "reward_spec_sha256": self.reward_spec_sha256,
            "scalar_reward": (
                None if self.scalar_reward is None else float(self.scalar_reward)
            ),
            "switch_penalty": self.switch_penalty.to_canonical_dict(),
            "terminal_class": self.terminal_class.value,
        }


def evaluate_completed_decision(
    completed_ticket: CompletedTicket,
    reward_spec: RewardSpecV1,
    *,
    quality_components: Optional[QualityComponentsV1] = None,
    previous_action: Optional[ExecutedActionIdentity] = None,
    adjudication: Optional[AdjudicationRecordV1] = None,
) -> DecisionOutcomeV1:
    """Measure one completed decision exactly as the controller classified it.

    Terminal handling agrees with :mod:`.reward_ticket_controller` term for
    term.  ``L`` is always derived from the ticket's own monotonic timestamps
    and the switch penalties from the two exact executed actions; nothing is
    accepted from the caller as a finished value.
    """
    if not isinstance(completed_ticket, CompletedTicket):
        raise StateRewardContractError(
            f"completed_ticket must be a controller CompletedTicket, got "
            f"{type(completed_ticket).__name__}"
        )
    if not isinstance(reward_spec, RewardSpecV1):
        raise RewardSpecError(
            f"reward_spec must be a RewardSpecV1 with explicit values, got "
            f"{type(reward_spec).__name__}"
        )
    if quality_components is not None and not isinstance(
        quality_components, QualityComponentsV1
    ):
        raise QualityContractError(
            f"quality_components must be a QualityComponentsV1 or None, got "
            f"{type(quality_components).__name__}"
        )
    terminal = completed_ticket.terminal_class
    spec_sha = reward_spec.canonical_sha256()
    ticket_sha = completed_ticket.canonical_sha256()
    switch = SwitchPenaltyV1.between(
        previous_action, completed_ticket.action, reward_spec
    )

    if adjudication is not None:
        raise AdjudicationError(
            "Phase 4a.2 has no verified timeout-reconciliation evidence "
            "protocol.  An arbitrary adjudicator id and evidence hash are "
            "not sufficient to score a timeout.  Keep every feedback timeout "
            "censored until a separately reviewed reconciliation carrier is "
            "implemented"
        )

    def _outcome(**kwargs: Any) -> DecisionOutcomeV1:
        record = DecisionOutcomeV1(
            terminal_class=terminal,
            completed_ticket_sha256=ticket_sha,
            reward_spec_sha256=spec_sha,
            switch_penalty=switch,
            quality_components=quality_components,
            **kwargs,
        )
        return replace(record, _attestation=_issue_outcome(record._binding()))

    # -- exact feedback: structurally eligible only after source proof ------ #
    if terminal is TerminalClass.REWARD_FINAL_EXACT:
        if quality_components is None:
            raise QualityContractError(
                f"{terminal.value} requires exact quality components before "
                f"it could become learning eligible; "
                f"quality components; none were supplied for decision "
                f"{completed_ticket.decision_seq}"
            )
        binding = quality_components.evidence.ack_binding
        if binding is None:
            raise QualityContractError(
                "a learning-eligible exact-quality transition requires a "
                "verified ACK binding; none is present.  An off-anchor "
                "continuous q cannot reach this state until the protocol-v2 "
                "carrier exists"
            )
        if binding.completed_ticket_sha256 != ticket_sha:
            raise QualityContractError(
                "the quality evidence was verified against a different ticket "
                "than the one being evaluated"
            )
        # Internal consistency and hash binding are not source authenticity.
        # The current live v1 evaluator cannot produce the raw, persisted
        # artifacts needed to prove actor/mask/match derivation, so exact
        # positive rewards remain fail-closed rather than silently training on
        # caller-asserted summaries.
        quality_components.evidence.require_learning_ready()
        latency = LatencyMeasurementV1.from_completed_ticket(completed_ticket)
        try:
            quality = reward_spec.evaluate_quality(quality_components)
        except InsufficientQualitySupportError:
            # No eligible ground truth anywhere: this frame carries no per-UE
            # perception reward.  It is censored, not scored zero -- GT absence
            # is not a failure of the action.
            return _outcome(
                eligibility=(
                    LearningEligibility.CENSORED_NO_ELIGIBLE_GROUND_TRUTH
                ),
                costs=ConstraintCostsV1(
                    c_deadline=0.0, c_authoritative_failure=0.0
                ),
                diagnostics=DiagnosticSignalsV1(
                    c_latency_excess=max(0.0, latency.normalized_latency - 1.0)
                ),
                quality=None,
                latency=latency,
                scalar_reward=None,
            )
        reward = (
            float(reward_spec.w_quality) * quality.q_perc
            - float(reward_spec.w_latency) * latency.normalized_latency
            - switch.total
        )
        return _outcome(
            eligibility=LearningEligibility.ELIGIBLE,
            costs=ConstraintCostsV1(
                c_deadline=1.0 if latency.normalized_latency > 1.0 else 0.0,
                c_authoritative_failure=0.0,
            ),
            diagnostics=DiagnosticSignalsV1(
                c_latency_excess=max(0.0, latency.normalized_latency - 1.0)
            ),
            quality=quality,
            latency=latency,
            scalar_reward=reward,
        )

    if quality_components is not None:
        raise QualityContractError(
            f"{terminal.value} received no exact per-frame quality, so none may "
            f"be supplied for decision {completed_ticket.decision_seq}"
        )

    # -- proven action-path failure: registered negative, no fabricated Q -- #
    if terminal is TerminalClass.ACTION_PATH_FAILURE:
        latency = LatencyMeasurementV1.from_completed_ticket(completed_ticket)
        return _outcome(
            eligibility=LearningEligibility.ELIGIBLE,
            costs=ConstraintCostsV1(
                c_deadline=1.0 if latency.normalized_latency > 1.0 else 0.0,
                c_authoritative_failure=1.0,
            ),
            diagnostics=DiagnosticSignalsV1(
                c_latency_excess=max(0.0, latency.normalized_latency - 1.0)
            ),
            quality=None,
            latency=latency,
            scalar_reward=float(reward_spec.r_registered_failure) - switch.total,
        )

    # -- excluded instrument fault ----------------------------------------- #
    if terminal is TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
        return _outcome(
            eligibility=LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
            costs=ConstraintCostsV1(
                c_deadline=None, c_authoritative_failure=None
            ),
            diagnostics=DiagnosticSignalsV1(c_latency_excess=None),
            quality=None,
            latency=None,
            scalar_reward=None,
        )

    # -- censored deadline expiry ------------------------------------------ #
    if terminal is not TerminalClass.FEEDBACK_TIMEOUT:  # pragma: no cover
        raise StateRewardContractError(
            f"unhandled terminal class {terminal.value}"
        )
    return _outcome(
        eligibility=LearningEligibility.CENSORED_PENDING_ADJUDICATION,
        costs=ConstraintCostsV1(c_deadline=1.0, c_authoritative_failure=None),
        diagnostics=DiagnosticSignalsV1(c_latency_excess=None),
        adjudication=adjudication,
    )


# --------------------------------------------------------------------------- #
# F. Policy-decision provenance and the replay transition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PolicyDecisionTraceV1(_Attested):
    """What the actor sampled, and how it became the executed action.

    The sampled continuous ``q`` is verified into the executed ``q_e4`` with the
    registered half-up rule, so a stored transition cannot claim an execution
    that its own sample would not have produced.

    This off-policy Hybrid-SAC replay contract intentionally does *not* store
    caller-reported behaviour log-probabilities.  SAC recomputes log-probability
    under the current actor when forming its actor and critic targets; the
    behaviour density is not an input to either update.  A pair of unattested
    scalar log-probabilities would therefore add no learning information and
    could not be reproduced from this record.  A future importance-weighted
    algorithm must introduce a separately versioned, distribution-complete
    trace (categorical logits plus all bounded-continuous distribution
    parameters), rather than repurposing this record.
    """

    session_uuid: str
    decision_seq: int
    policy_feature_sha256: str
    source_state_sha256: str
    state_normalization_spec_sha256: str
    freshness_policy_sha256: str
    sampled_mode_id: int
    sampled_q: float
    executed_action: ExecutedActionIdentity
    actor_version_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = StateRewardContractError
        _canonical_uuid(self.session_uuid, E)
        _non_negative_int(self.decision_seq, "decision_seq", E)
        for name in (
            "policy_feature_sha256",
            "source_state_sha256",
            "state_normalization_spec_sha256",
            "freshness_policy_sha256",
        ):
            _sha256_hex(getattr(self, name), name, E)
        _non_negative_int(self.sampled_mode_id, "sampled_mode_id", E)
        if self.sampled_mode_id >= EXPECTED_MODE_COUNT:
            raise E(
                f"sampled_mode_id {self.sampled_mode_id} is outside the "
                f"catalog's {EXPECTED_MODE_COUNT} joint modes"
            )
        _finite_in(
            self.sampled_q,
            "sampled_q",
            float(Q_E4_MIN) / Q_E4_SCALE,
            float(Q_E4_MAX) / Q_E4_SCALE,
            E,
        )
        if not isinstance(self.executed_action, ExecutedActionIdentity):
            raise E(
                f"executed_action must be an ExecutedActionIdentity, got "
                f"{type(self.executed_action).__name__}"
            )
        self.executed_action.require_reconciled()
        _sha256_hex(self.actor_version_sha256, "actor_version_sha256", E)

        if self.sampled_mode_id != self.executed_action.mode_id:
            raise E(
                f"the actor sampled joint mode {self.sampled_mode_id} but the "
                f"executed action is mode {self.executed_action.mode_id}"
            )
        expected_q_e4 = round_half_up_q_e4(self.sampled_q)
        if expected_q_e4 != self.executed_action.q_e4:
            raise E(
                f"the sampled q {self.sampled_q!r} quantizes to q_e4="
                f"{expected_q_e4} under the registered half-up rule, but the "
                f"executed action carries q_e4={self.executed_action.q_e4}"
            )
        if self._attestation is not None and not _valid_policy_trace(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the policy-decision trace attestation does not match its "
                "state, feature, specification or action binding"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_policy_trace

    @classmethod
    def for_decision(
        cls,
        *,
        state: CausalStateV1,
        features: PolicyFeatureVectorV1,
        decision_seq: int,
        sampled_mode_id: int,
        sampled_q: float,
        executed_action: ExecutedActionIdentity,
        actor_version_sha256: str,
    ) -> "PolicyDecisionTraceV1":
        if not isinstance(state, CausalStateV1):
            raise CausalStateError(
                f"state must be CausalStateV1, got {type(state).__name__}"
            )
        if not isinstance(features, PolicyFeatureVectorV1):
            raise StateRewardContractError(
                "features must be an attested PolicyFeatureVectorV1"
            )
        features.assert_binds(state)
        record = cls(
            session_uuid=state.session_uuid,
            decision_seq=decision_seq,
            policy_feature_sha256=features.canonical_sha256(),
            source_state_sha256=state.canonical_sha256(),
            state_normalization_spec_sha256=(
                features.state_normalization_spec_sha256
            ),
            freshness_policy_sha256=features.freshness_policy_sha256,
            sampled_mode_id=sampled_mode_id,
            sampled_q=sampled_q,
            executed_action=executed_action,
            actor_version_sha256=actor_version_sha256,
        )
        return replace(
            record, _attestation=_issue_policy_trace(record._binding())
        )

    def assert_binds(
        self,
        *,
        state: CausalStateV1,
        features: PolicyFeatureVectorV1,
        decision_seq: int,
        executed_action: ExecutedActionIdentity,
    ) -> None:
        self.require_attested()
        features.assert_binds(state)
        expected = {
            "session_uuid": state.session_uuid,
            "decision_seq": decision_seq,
            "policy_feature_sha256": features.canonical_sha256(),
            "source_state_sha256": state.canonical_sha256(),
            "state_normalization_spec_sha256": (
                features.state_normalization_spec_sha256
            ),
            "freshness_policy_sha256": features.freshness_policy_sha256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise TransitionIdentityError(
                    f"policy trace {name}={getattr(self, name)!r} does not "
                    f"match the decision source {value!r}"
                )
        if self.executed_action != executed_action:
            raise TransitionIdentityError(
                "policy trace executed action does not match the ticket"
            )

    @property
    def q_exec(self) -> float:
        """The executed continuous value actually put on the wire."""
        return float(self.executed_action.q_e4) / Q_E4_SCALE

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "actor_version_sha256": self.actor_version_sha256,
            "decision_seq": self.decision_seq,
            "executed_action": self.executed_action.to_canonical_dict(),
            "executed_q_e4": self.executed_action.q_e4,
            "policy_feature_sha256": self.policy_feature_sha256,
            "q_exec": self.q_exec,
            "quantization_rule": (
                "require 0 <= q <= 0.98, then round_half_up(q * 1e4)"
            ),
            "record": "policy_decision_trace_v1",
            "sampled_mode_id": self.sampled_mode_id,
            "sampled_q": float(self.sampled_q),
            "session_uuid": self.session_uuid,
            "source_state_sha256": self.source_state_sha256,
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
            "freshness_policy_sha256": self.freshness_policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplayTransitionV1(_Attested):
    """One exactly identified SMDP transition, ready for a later replay layer.

    This is the *record*, not the storage.  ``hold_duration_tensors`` and
    ``discount_multiplier`` are derived properties over the frozen hold, never
    constructor inputs.  Build with :func:`build_replay_transition`; a directly
    constructed instance is unattested and cannot serialize.

    :meth:`revalidate` recomputes the whole outcome from the raw components plus
    the reward spec and compares it, so a stored transition can be re-proved.
    """

    state: CausalStateV1
    executed_action: ExecutedActionIdentity
    completed_ticket: CompletedTicket
    outcome: DecisionOutcomeV1
    policy_trace: PolicyDecisionTraceV1
    state_features: PolicyFeatureVectorV1
    next_state_features: Optional[PolicyFeatureVectorV1]
    reward_spec: RewardSpecV1
    state_normalization_spec_sha256: str
    freshness_policy_sha256: str
    terminated: bool
    truncated: bool
    next_state: Optional[CausalStateV1] = None
    episode_end_reason: Optional[str] = None
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        E = TransitionIdentityError
        for name, expected in (
            ("state", CausalStateV1),
            ("executed_action", ExecutedActionIdentity),
            ("completed_ticket", CompletedTicket),
            ("outcome", DecisionOutcomeV1),
            ("policy_trace", PolicyDecisionTraceV1),
            ("state_features", PolicyFeatureVectorV1),
            ("reward_spec", RewardSpecV1),
        ):
            if not isinstance(getattr(self, name), expected):
                raise E(
                    f"{name} must be a {expected.__name__}, got "
                    f"{type(getattr(self, name)).__name__}"
                )
        _sha256_hex(
            self.state_normalization_spec_sha256,
            "state_normalization_spec_sha256",
            E,
        )
        _sha256_hex(self.freshness_policy_sha256, "freshness_policy_sha256", E)
        _exact_bool(self.terminated, "terminated", E)
        _exact_bool(self.truncated, "truncated", E)

        ticket = self.completed_ticket
        ticket.require_lineage_attested()
        self.executed_action.require_reconciled()
        self.outcome.require_attested()
        self.state_features.require_attested()
        self.policy_trace.require_attested()

        # Features are values *derived from* this exact state and the complete
        # preprocessing specifications.  Hash labels alone are insufficient.
        self.state_features.assert_binds(self.state)
        if (
            self.state_normalization_spec_sha256
            != self.state_features.state_normalization_spec_sha256
        ):
            raise E(
                "transition normalization hash does not match the complete "
                "specification bound into the state feature vector"
            )
        if (
            self.freshness_policy_sha256
            != self.state_features.freshness_policy_sha256
        ):
            raise E(
                "transition freshness hash does not match the complete policy "
                "that actually accepted the state"
            )

        # -- one exact session and decision throughout --------------------- #
        if self.state.session_uuid != ticket.session_uuid:
            raise E(
                f"state session {self.state.session_uuid} does not match the "
                f"ticket's {ticket.session_uuid}"
            )
        if self.outcome.terminal_class is not ticket.terminal_class:
            raise E(
                f"outcome terminal class {self.outcome.terminal_class.value} "
                f"disagrees with the ticket's {ticket.terminal_class.value}"
            )
        if self.outcome.completed_ticket_sha256 != ticket.canonical_sha256():
            raise E(
                "the outcome was measured for a different ticket than this "
                "transition carries"
            )
        if self.outcome.reward_spec_sha256 != self.reward_spec.canonical_sha256():
            raise E(
                "the outcome was measured under a different reward spec than "
                "the transition records"
            )

        # -- the executed action equals every held tensor's action --------- #
        if self.executed_action != ticket.action:
            raise E(
                f"executed action {self.executed_action.canonical_mode} "
                f"q_e4={self.executed_action.q_e4} does not match the hold's "
                f"{ticket.action.canonical_mode} q_e4={ticket.action.q_e4}"
            )
        for member in ticket.hold.tensors:
            if member.action != self.executed_action:
                raise E(
                    f"tensor_seq {member.tensor_seq} carries a different "
                    f"executed action than the transition's"
                )
        if self.policy_trace.executed_action != self.executed_action:
            raise E(
                "the policy-decision trace records a different executed action "
                "than the hold executed"
            )
        # -- the state joins the reward-requested tensor and frame --------- #
        if self.state.tensor_seq != ticket.reward_tensor_seq:
            raise E(
                f"state tensor_seq {self.state.tensor_seq} is not the ticket's "
                f"reward-requested tensor {ticket.reward_tensor_seq}"
            )
        if self.state.carla_frame_id != ticket.reward_carla_frame_id:
            raise E(
                f"state carla_frame_id {self.state.carla_frame_id} is not the "
                f"reward tensor's frame {ticket.reward_carla_frame_id}"
            )

        # -- state causality against the ticket ---------------------------- #
        if self.state.observed_ns > ticket.opened_ns:
            raise E(
                f"the state was observed at {self.state.observed_ns} ns but the "
                f"decision opened at {ticket.opened_ns} ns; a policy decision "
                f"can only be conditioned on an observation that precedes it"
            )
        previous = self.state.previous
        if previous is not None:
            if previous.decision_seq >= ticket.decision_seq:
                raise E(
                    f"the state's previous decision {previous.decision_seq} "
                    f"must precede this decision {ticket.decision_seq}; a "
                    f"state may never carry its own or a future outcome"
                )
            if previous.controller_lineage_uuid != (
                ticket.controller_lineage_uuid
            ):
                raise E(
                    "the previous outcome belongs to a different concrete "
                    "controller episode than the current ticket"
                )
            if ticket.lineage_ordinal != previous.lineage_ordinal + 1:
                raise E(
                    f"the current ticket has controller completion ordinal "
                    f"{ticket.lineage_ordinal}, but the policy state carries "
                    f"ordinal {previous.lineage_ordinal}; exact adjacency is "
                    f"proved by completion order, not by decision_seq-1"
                )
            if ticket.predecessor_completed_ticket_sha256 != (
                previous.completed_ticket_sha256
            ):
                raise E(
                    "the current controller ticket does not name the state's "
                    "previous completed ticket as its exact predecessor"
                )
        if previous is None:
            start = self.state.episode_start
            if start is None:  # guarded by CausalStateV1; retained for audit
                raise E(
                    "a transition without a predecessor requires an explicit "
                    "episode-start proof"
                )
            if start.first_decision_seq != ticket.decision_seq:
                raise E(
                    f"episode-start proof authorizes first decision "
                    f"{start.first_decision_seq}, not ticket decision "
                    f"{ticket.decision_seq}; a missing predecessor cannot be "
                    f"silently represented as episode start"
                )
            if start.first_tensor_seq != ticket.reward_tensor_seq:
                raise E(
                    "episode-start proof does not bind the ticket's first "
                    "reward-requested tensor"
                )
            if start.first_carla_frame_id != ticket.reward_carla_frame_id:
                raise E(
                    "episode-start proof does not bind the ticket's first "
                    "CARLA frame"
                )
            if start.controller_lineage_uuid != ticket.controller_lineage_uuid:
                raise E(
                    "episode-start proof belongs to a different concrete "
                    "controller episode"
                )
            if ticket.controller_genesis_proof_sha256 != (
                start.controller_genesis.canonical_sha256()
            ):
                raise E(
                    "the completed genesis ticket does not carry the exact "
                    "pre-decision authorization used by the first policy state"
                )
            if ticket.lineage_ordinal != 0 or (
                ticket.predecessor_completed_ticket_sha256 is not None
            ):
                raise E(
                    "a transition represented as episode start is not the "
                    "controller genesis ticket (ordinal 0, no predecessor)"
                )
        self.policy_trace.assert_binds(
            state=self.state,
            features=self.state_features,
            decision_seq=ticket.decision_seq,
            executed_action=self.executed_action,
        )
        if ticket.policy_decision_trace_sha256 is None:
            raise E(
                "the controller ticket has no pre-execution policy-decision "
                "trace commitment; a trace constructed after execution is "
                "not causal replay evidence"
            )
        if ticket.policy_decision_trace_sha256 != (
            self.policy_trace.canonical_sha256()
        ):
            raise E(
                "the replay policy trace is not the exact trace committed to "
                "the controller before this decision executed"
            )

        # -- terminal/truncation bookkeeping ------------------------------- #
        if self.terminated and self.truncated:
            raise E("a transition cannot be both terminated and truncated")
        if (self.terminated or self.truncated) and not self.episode_end_reason:
            raise E(
                "a terminated or truncated transition must carry an explicit "
                "episode_end_reason"
            )
        if not (self.terminated or self.truncated) and self.episode_end_reason:
            raise E(
                f"episode_end_reason {self.episode_end_reason!r} was given for "
                f"a transition that neither terminated nor truncated"
            )
        if self.episode_end_reason is not None:
            _non_empty_str(self.episode_end_reason, "episode_end_reason", E)

        # -- the next state is causally later and never crosses a session -- #
        if self.next_state is None:
            if not (self.terminated or self.truncated):
                raise E(
                    "a non-terminal transition requires a next state; only a "
                    "terminated or truncated transition may omit it"
                )
            if self.next_state_features is not None:
                raise E(
                    "next_state_features must be absent when next_state is absent"
                )
        else:
            if not isinstance(self.next_state, CausalStateV1):
                raise E(
                    f"next_state must be a CausalStateV1 or None, got "
                    f"{type(self.next_state).__name__}"
                )
            if self.next_state.session_uuid != self.state.session_uuid:
                raise E(
                    f"next state session {self.next_state.session_uuid} "
                    f"crosses out of session {self.state.session_uuid}; a "
                    f"transition never spans two sessions"
                )
            if not isinstance(self.next_state_features, PolicyFeatureVectorV1):
                raise E(
                    "a non-terminal transition must carry the attested feature "
                    "vector derived for its bootstrap next state"
                )
            self.next_state_features.require_attested()
            self.next_state_features.assert_binds(self.next_state)
            if (
                self.next_state_features.state_normalization_spec_sha256
                != self.state_normalization_spec_sha256
            ):
                raise E(
                    "next-state features use a different normalization spec"
                )
            if (
                self.next_state_features.freshness_policy_sha256
                != self.freshness_policy_sha256
            ):
                raise E(
                    "next-state features use a different freshness policy"
                )
            if self.next_state.observed_ns < ticket.closed_ns:
                raise E(
                    f"next state was observed at {self.next_state.observed_ns} "
                    f"ns, before this decision's ticket closed at "
                    f"{ticket.closed_ns} ns; the successor observation must "
                    f"follow the decision it succeeds"
                )
            last_governed = ticket.tensor_seqs[-1]
            if self.next_state.tensor_seq <= last_governed:
                raise E(
                    f"next state tensor_seq {self.next_state.tensor_seq} must "
                    f"follow every tensor governed by this decision (last is "
                    f"{last_governed})"
                )
            successor = self.next_state.previous
            if successor is None:
                raise E(
                    f"next_state.previous is absent: the successor observation "
                    f"must carry exactly this completed decision "
                    f"{ticket.decision_seq} as its previous outcome"
                )
            if successor.completed_ticket_sha256 != ticket.canonical_sha256():
                raise E(
                    f"next_state.previous is bound to ticket "
                    f"{successor.completed_ticket_sha256}, not to this "
                    f"transition's {ticket.canonical_sha256()}; an unrelated "
                    f"or future previous outcome is refused"
                )
            if successor.decision_seq != ticket.decision_seq:
                raise E(
                    f"next_state.previous names decision "
                    f"{successor.decision_seq}, not this transition's "
                    f"{ticket.decision_seq}"
                )
            closure_outcome = evaluate_completed_decision(
                ticket,
                self.reward_spec,
                quality_components=self.outcome.quality_components,
                previous_action=(None if previous is None else previous.action),
                adjudication=None,
            )
            closure_snapshot = PreviousOutcomeV1.from_completed(
                ticket, closure_outcome, self.reward_spec
            )
            if successor.canonical_sha256() != closure_snapshot.canonical_sha256():
                raise E(
                    "next_state.previous is not the immutable closure-time "
                    "policy snapshot for this decision.  A later replay "
                    "adjudication must never rewrite historical policy state"
                )

        # -- the switch penalty must come from the state's own previous ---- #
        # The previous executed action is not a free parameter: it is the action
        # of the decision the state already carries.  Tying the two together is
        # what makes the penalty re-derivable, and it refuses a penalty computed
        # against some unrelated action.
        expected_previous = (
            None if previous is None else previous.action.canonical_sha256()
        )
        recorded_previous = self.outcome.switch_penalty.previous_action_sha256
        if recorded_previous != expected_previous:
            raise E(
                f"the switch penalty was computed against previous action "
                f"{recorded_previous}, but this state carries "
                f"{expected_previous}; the mode and q penalties must be "
                f"derived from the decision the state actually records"
            )

        # -- quality evidence is bound into the transition ----------------- #
        quality = self.outcome.quality
        if quality is not None:
            evidence = quality.components.evidence
            evidence.require_causal_per_frame()
            if evidence.executed_action_sha256 != (
                self.executed_action.canonical_sha256()
            ):
                raise E(
                    f"the quality evidence is keyed on executed-action hash "
                    f"{evidence.executed_action_sha256}, but this transition "
                    f"executed {self.executed_action.canonical_sha256()}"
                )
            binding = evidence.ack_binding
            if binding is None:
                raise E(
                    "a quality-bearing transition must bind a verified ACK; "
                    "an off-anchor action cannot be learning-eligible for "
                    "exact quality until the protocol-v2 carrier exists"
                )
            binding.require_attested()
            if binding.completed_ticket_sha256 != ticket.canonical_sha256():
                raise E(
                    "the ACK binding was verified against a different ticket"
                )
        latency = self.outcome.latency
        if latency is not None:
            latency.require_attested()
            latency.assert_belongs_to(ticket)

        if self._attestation is not None and not _valid_transition(
            self._attestation, self._binding()
        ):
            raise UnattestedRecordError(
                "the transition's attestation does not match its own "
                "serialized fields"
            )

    @property
    def _checker(self) -> Callable:
        return _valid_transition

    # -- derived identity -------------------------------------------------- #

    @property
    def session_uuid(self) -> str:
        return self.state.session_uuid

    @property
    def decision_seq(self) -> int:
        return self.completed_ticket.decision_seq

    @property
    def reward_tensor_seq(self) -> int:
        return self.completed_ticket.reward_tensor_seq

    @property
    def reward_carla_frame_id(self) -> int:
        return self.completed_ticket.reward_carla_frame_id

    @property
    def hold(self) -> ActionHoldManifest:
        return self.completed_ticket.hold

    @property
    def hold_duration_tensors(self) -> int:
        """The realized ``d``, derived from the frozen hold."""
        return self.completed_ticket.hold_duration_tensors

    @property
    def gamma_per_tensor(self) -> float:
        return float(self.reward_spec.gamma_per_tensor)

    @property
    def discount_multiplier(self) -> float:
        """``gamma_per_tensor ** d``: the SMDP discount, always derived."""
        return self.gamma_per_tensor ** self.hold_duration_tensors

    @property
    def reward_spec_sha256(self) -> str:
        return self.reward_spec.canonical_sha256()

    @property
    def scalar_reward(self) -> Optional[float]:
        return self.outcome.scalar_reward

    @property
    def costs(self) -> ConstraintCostsV1:
        return self.outcome.costs

    @property
    def diagnostics(self) -> DiagnosticSignalsV1:
        return self.outcome.diagnostics

    @property
    def quality(self) -> Optional[QualityEvaluationV1]:
        return self.outcome.quality

    @property
    def latency(self) -> Optional[LatencyMeasurementV1]:
        return self.outcome.latency

    @property
    def eligibility(self) -> LearningEligibility:
        return self.outcome.eligibility

    @property
    def terminal_class(self) -> TerminalClass:
        return self.completed_ticket.terminal_class

    @property
    def learning_eligible(self) -> bool:
        return self.outcome.learning_eligible

    @property
    def quality_components(self) -> Optional[QualityComponentsV1]:
        """The raw components this transition's outcome was measured from."""
        return self.outcome.quality_components

    @property
    def quality_evidence(self) -> Optional[QualityEvidenceV1]:
        components = self.outcome.quality_components
        return None if components is None else components.evidence

    @property
    def raw_quality_ack_sha256(self) -> Optional[str]:
        evidence = self.quality_evidence
        return None if evidence is None else evidence.raw_quality_ack_sha256

    @property
    def detailed_evidence_sha256(self) -> Optional[str]:
        evidence = self.quality_evidence
        return None if evidence is None else evidence.detailed_evidence_sha256

    @property
    def executed_action_sha256(self) -> str:
        return self.executed_action.canonical_sha256()

    # -- re-derivation ----------------------------------------------------- #

    def revalidate(self) -> DecisionOutcomeV1:
        """Recompute the outcome from frozen sources and compare it.

        Raises:
            TransitionIdentityError: if the recomputed outcome differs in any
                serialized field -- an arbitrary scalar reward, an inconsistent
                Q, a latency unrelated to the ticket, an eligible timeout
                without a bound adjudication or an infrastructure fault dressed
                as an action failure all fail here.
        """
        self.require_attested()
        self.state_features.assert_binds(self.state)
        if self.next_state is None:
            if self.next_state_features is not None:  # pragma: no cover
                raise TransitionIdentityError(
                    "next-state feature vector exists without a next state"
                )
        else:
            if self.next_state_features is None:  # pragma: no cover
                raise TransitionIdentityError(
                    "next state is missing its bootstrap feature vector"
                )
            self.next_state_features.assert_binds(self.next_state)
        self.policy_trace.assert_binds(
            state=self.state,
            features=self.state_features,
            decision_seq=self.decision_seq,
            executed_action=self.executed_action,
        )
        recomputed = evaluate_completed_decision(
            self.completed_ticket,
            self.reward_spec,
            quality_components=self.outcome.quality_components,
            previous_action=self._previous_executed_action(),
            adjudication=self.outcome.adjudication,
        )
        if recomputed.canonical_sha256() != self.outcome.canonical_sha256():
            raise TransitionIdentityError(
                f"the stored outcome does not survive re-derivation from its "
                f"own frozen sources: recomputed "
                f"{recomputed.canonical_sha256()} but the record claims "
                f"{self.outcome.canonical_sha256()}"
            )
        return recomputed

    def _previous_executed_action(self) -> Optional[ExecutedActionIdentity]:
        """The previous decision's action, if the state carries one."""
        previous = self.state.previous
        return None if previous is None else previous.action

    # -- serialization ----------------------------------------------------- #

    def _serialized_fields(self) -> Dict[str, Any]:
        return {
            "completed_ticket": self.completed_ticket.to_canonical_dict(),
            "decision_seq": self.decision_seq,
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "discount_multiplier": self.discount_multiplier,
            "episode_end_reason": self.episode_end_reason,
            "executed_action": self.executed_action.to_canonical_dict(),
            "executed_action_sha256": self.executed_action_sha256,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "gamma_per_tensor": self.gamma_per_tensor,
            "hold_duration_tensors": self.hold_duration_tensors,
            "next_state": (
                None
                if self.next_state is None
                else self.next_state.to_canonical_dict()
            ),
            "outcome": self.outcome.to_canonical_dict(),
            "policy_trace": self.policy_trace.to_canonical_dict(),
            "state_features": self.state_features.to_canonical_dict(),
            "next_state_features": (
                None
                if self.next_state_features is None
                else self.next_state_features.to_canonical_dict()
            ),
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "record": "replay_transition_v1",
            "reward_carla_frame_id": self.reward_carla_frame_id,
            "reward_latency_clock_domain": REWARD_LATENCY_CLOCK_DOMAIN,
            "reward_spec_sha256": self.reward_spec_sha256,
            "reward_tensor_seq": self.reward_tensor_seq,
            "schema_id": SCHEMA_ID,
            "schema_sha256": SCHEMA_SHA256,
            "schema_version": SCHEMA_VERSION,
            "session_uuid": self.session_uuid,
            "state": self.state.to_canonical_dict(),
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def build_replay_transition(
    *,
    state: CausalStateV1,
    next_state: Optional[CausalStateV1],
    completed_ticket: CompletedTicket,
    outcome: DecisionOutcomeV1,
    policy_trace: PolicyDecisionTraceV1,
    reward_spec: RewardSpecV1,
    normalization: StateNormalizationSpecV1,
    freshness: StateFreshnessPolicyV1,
    terminated: bool = False,
    truncated: bool = False,
    episode_end_reason: Optional[str] = None,
) -> ReplayTransitionV1:
    """Assemble a transition, deriving every derivable field, then re-prove it.

    The executed action, the spec hashes and ``gamma_per_tensor`` all come from
    the supplied frozen records rather than loose arguments, and the assembled
    transition is immediately :meth:`~ReplayTransitionV1.revalidate`-ed so a
    stored record is never merely asserted.
    """
    if not isinstance(reward_spec, RewardSpecV1):
        raise RewardSpecError(
            f"reward_spec must be a RewardSpecV1, got {type(reward_spec).__name__}"
        )
    if not isinstance(normalization, StateNormalizationSpecV1):
        raise NormalizationSpecError(
            f"normalization must be a StateNormalizationSpecV1, got "
            f"{type(normalization).__name__}"
        )
    if not isinstance(freshness, StateFreshnessPolicyV1):
        raise NormalizationSpecError(
            f"freshness must be a StateFreshnessPolicyV1, got "
            f"{type(freshness).__name__}"
        )
    if not isinstance(completed_ticket, CompletedTicket):
        raise TransitionIdentityError(
            f"completed_ticket must be a controller CompletedTicket, got "
            f"{type(completed_ticket).__name__}"
        )
    state_features = build_policy_features(state, normalization, freshness)
    next_state_features = (
        None
        if next_state is None
        else build_policy_features(next_state, normalization, freshness)
    )
    record = ReplayTransitionV1(
        state=state,
        executed_action=completed_ticket.action,
        completed_ticket=completed_ticket,
        outcome=outcome,
        policy_trace=policy_trace,
        state_features=state_features,
        next_state_features=next_state_features,
        reward_spec=reward_spec,
        state_normalization_spec_sha256=normalization.canonical_sha256(),
        freshness_policy_sha256=freshness.canonical_sha256(),
        terminated=terminated,
        truncated=truncated,
        next_state=next_state,
        episode_end_reason=episode_end_reason,
    )
    attested = replace(
        record, _attestation=_issue_transition(record._binding())
    )
    attested.revalidate()
    return attested


# --------------------------------------------------------------------------- #
# Canonical schema descriptor
# --------------------------------------------------------------------------- #

SCHEMA_ID: str = "splitfusion_hybrid_sac_state_reward_transition_v1"
SCHEMA_VERSION: int = 4

SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "revision_note": (
            "v4 (phase 4a.2 repair) makes runtime radio admission fail closed, "
            "binds a controller-issued pre-decision genesis proof plus "
            "gap-tolerant completed-ticket lineage, requires the controller "
            "to commit the exact policy-decision trace before execution, "
            "removes "
            "incomplete behaviour log-probabilities, and separates internally "
            "consistent quality fixtures from source-authenticated evidence. "
            "Exact positive rewards remain blocked from learning/replay until "
            "a reviewed producer derives them from frozen CARLA and prediction "
            "artifacts; v3 causal-state, freshness and replay revalidation "
            "invariants remain unchanged"
        ),
        "phase": (
            "pure in-memory causal-state, reward-measurement and "
            "replay-transition contract; no environment, storage, network, "
            "optimizer, protocol-v2 transport or live integration"
        ),
        "design_reference": (
            "DESIGN.md sections 3, 4, 7, 8 and 9"
        ),
        "action_space": (
            "(joint mode among 12 family-quantizer pairs, continuous q); the "
            "older four-AE fixed-quantizer recurrent sketch is superseded"
        ),
        "policy_class": "feed-forward; no recurrent memory in v1",
        "dependencies": {
            "action_catalog": {
                "schema": CATALOG_SCHEMA,
                "sha256": CATALOG_SHA256,
                "execution_mode": EXECUTION_MODE,
                "joint_mode_count": EXPECTED_MODE_COUNT,
                "q_e4_bounds": [Q_E4_MIN, Q_E4_MAX],
            },
            "executed_action_identity": {
                "schema_id": ACTION_IDENTITY_SCHEMA_ID,
                "schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
            },
            "transaction_identity": {
                "schema_id": TRANSACTION_SCHEMA_ID,
                "schema_sha256": TRANSACTION_SCHEMA_SHA256,
                "schema_version": TRANSACTION_SCHEMA_VERSION,
                "minimum_hold_tensors": MINIMUM_HOLD_TENSORS,
            },
            "scene_descriptors": {
                "schema_id": SCENE_SCHEMA_ID,
                "schema_sha256": SCENE_SCHEMA_SHA256,
                "schema_version": SCENE_SCHEMA_VERSION,
                "descriptors": ["camera_si", "radar_p40"],
                "p40_horizon_m": P40_HORIZON_M,
            },
            "reward_ticket_controller": {
                "schema_id": CONTROLLER_SCHEMA_ID,
                "schema_sha256": CONTROLLER_SCHEMA_SHA256,
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "reward_deadline_ns": B_REWARD_DEADLINE_NS,
                "minimum_hold_tensors": K_MIN_TENSORS,
            },
            "quality_protocol": {
                "contract": QUALITY_PROTOCOL_CONTRACT,
                "contract_sha256": QUALITY_PROTOCOL_CONTRACT_SHA256,
                "verification": (
                    "verify_quality_protocol_binding() loads the real module "
                    "lazily and raises on any drift; it runs before every ACK "
                    "binding"
                ),
            },
        },
        "forgery_resistance": {
            "rule": (
                "every derived record carries a private attestation bound to a "
                "hash of its own serialized fields, issued only by the "
                "validating factory that recomputed it; an unattested or "
                "mutated record cannot serialize or enter a transition"
            ),
            "attested_records": [
                "quality_ack_binding_v1",
                "evaluation_eligibility_result_v1",
                "quality_components_v1",
                "quality_evaluation_v1",
                "latency_measurement_v1",
                "episode_start_proof_v1",
                "previous_outcome_v1",
                "policy_feature_vector_v1",
                "policy_decision_trace_v1",
                "decision_outcome_v1",
                "replay_transition_v1",
            ],
            "revalidation": (
                "ReplayTransitionV1.revalidate() recomputes the outcome from "
                "the raw components plus the reward spec and compares the "
                "canonical hash, so an arbitrary scalar reward, an "
                "inconsistent Q, a latency unrelated to the ticket, a "
                "stale or source-mismatched policy vector, an eligible "
                "timeout without verified reconciliation evidence or an "
                "infrastructure fault dressed as an action failure all fail"
            ),
            "not_a_security_boundary": (
                "Python offers no true privacy; this guards against mistaken "
                "construction, not a determined attacker"
            ),
        },
        "quality_ack": {
            "schema": QUALITY_ACK_SCHEMA,
            "anchor_only": QUALITY_ACK_IS_ANCHOR_ONLY,
            "obligation": (
                "QualityAckObligationV1 binds run/cell/stream, frame_id and "
                "capture_timestamp_ns, the session/decision/reward-tensor "
                "identity and the complete executed action, but the current "
                "factory reconstructs it from a completed ticket and does not "
                "authenticate that it existed before ACK arrival"
            ),
            "obligation_precommit_authenticated": (
                QUALITY_OBLIGATION_PRECOMMIT_AUTHENTICATED
            ),
            "verification": (
                "from_ack_document calls the real protocol validator, "
                "recomputes the raw ACK SHA-256 from the document itself, "
                "opens the actual detail whose digest must equal dh, validates "
                "its versioned phase4a2 reward-support extension, retains all "
                "seven identity fields and cross-checks the obligation and "
                "CompletedTicket. This proves identity and internal arithmetic "
                "consistency only: current v2 support rows/arrays are caller-"
                "supplied fixtures, not authenticated CARLA source evidence, "
                "and therefore cannot enter learning or replay"
            ),
            "reuse_rule": (
                "one private module-owned process registry enforces a "
                "bijection between raw legacy-v1 ACK digest and exact "
                "obligation/ticket while the reverse key is the completed "
                "ticket alone: neither one ACK for two tickets nor two "
                "conflicting ACKs hidden behind different obligations for one "
                "ticket are accepted; callers cannot "
                "substitute a fresh registry. Protocol-v2 must carry the "
                "complete decision and continuous-action identity on wire for "
                "cross-process durability"
            ),
            "legacy_detail_status": (
                "legacy v1 details without the versioned reward-support "
                "extension fail closed pending a producer update or "
                "identity-bearing protocol-v2"
            ),
            "off_anchor_rule": (
                "an off-anchor continuous q raises OffAnchorQualityAckError "
                "and cannot produce a learning-eligible exact-quality "
                "transition; it is never snapped to a nearest anchor"
            ),
            "protocol_v2_requirement": PROTOCOL_V2_REQUIREMENT,
            "missing_required_fields": QUALITY_ACK_MISSING_REQUIRED_FIELDS,
            "timing_clock_domain": QUALITY_ACK_TIMING_CLOCK_DOMAIN,
            "false_positive_counts_available": (
                QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE
            ),
            "derivable_detection_metrics": (
                QUALITY_ACK_DERIVABLE_DETECTION_METRICS
            ),
            "underivable_detection_metrics": (
                QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS
            ),
        },
        "evidence": {
            "kinds": tuple(kind.value for kind in EvidenceKind),
            "granularities": tuple(g.value for g in EvidenceGranularity),
            "causal_reward_requires": [
                CAUSAL_REWARD_EVIDENCE[0].value,
                CAUSAL_REWARD_EVIDENCE[1].value,
            ],
            "aggregate_rule": (
                "aggregate 72-action / 288-cell campaign evidence is real "
                "evidence about profile-level payload, delivery and quality, "
                "but it is an action average rather than a causal per-decision "
                "transition and must never stand in for one"
            ),
        },
        "eligibility": {
            "rule": (
                "ground-truth presence alone never creates a miss penalty; "
                "recall is defined over the per-UE eligible set fixed by a "
                "hash-bound range / field-of-view / visibility-or-AVO contract"
            ),
            "visibility_rules": tuple(r.value for r in VisibilityRule),
            "reward_scopes": tuple(s.value for s in RewardScope),
            "implemented_scope": RewardScope.PER_UE_PERCEPTION.value,
            "scope_separation": (
                "per-UE perception reward is kept separate from any later "
                "cooperative-map coverage reward: different eligibility sets "
                "and different credit assignment, never summed"
            ),
            "segmentation_masking": (
                "the same depth/range eligibility domain is applied to both "
                "prediction and ground-truth masks before segmentation IoU; "
                "the contract refuses an unmasked or asymmetric claim"
            ),
            "per_frame_result": (
                "the per-frame eligibility result is attested, binds the "
                "recomputed contract hash and the exact frame/UE/detail, and "
                "is the sole source of eligible instance and mask support. "
                "In v4 this is a formula fixture with unverified source, not "
                "learning evidence"
            ),
        },
        "quality": {
            "ground_truth_source": (
                "CARLA_GT_EXACT: explicitly privileged and non-deployable, "
                "available only in the training/testbed instrument"
            ),
            "gt_privileged": True,
            "gt_deployable": False,
            "producer_status": (
                QualityProducerStatus.CONTRACT_FIXTURE_UNVERIFIED_SOURCE.value
            ),
            "learning_ready": False,
            "producer_requirement": (
                "a reviewed producer must derive actor eligibility, calibrated "
                "depth-masked GT/prediction masks and localization matching "
                "from frozen source artifacts, bind their manifest and code/"
                "configuration identities, and emit protocol-v2 full action/"
                "decision identity before exact positive rewards are admitted"
            ),
            "localization": (
                "U_xy,c = exp(-e_c / tau_c); U_loc,c = sqrt(recall_c * U_xy,c) "
                "over the eligible GT set; eligible GT with tp=0 gives "
                "recall=0 and U_loc=0 and stays in the combination"
            ),
            "localization_combiner": (
                "explicit and required: the registered formulation does not "
                "fix it, so WEIGHTED_GEOMETRIC_MEAN or "
                "WEIGHTED_ARITHMETIC_MEAN must be chosen deliberately"
            ),
            "segmentation": (
                "s_c = clip(IoU_c / reference_c, 0, 1) against frozen "
                "references supplied by the reward spec, combined with the "
                "registered weighted geometric mean"
            ),
            "segmentation_exclusion": (
                "a segmentation class is excluded only when both the "
                "predicted and the GT mask are empty, so a false positive "
                "against absent GT and a missed mask against present GT both "
                "remain valid and penalized"
            ),
            "combination": (
                "Q_perc = Q_loc * ((1 - beta) + beta * Q_seg); localization is "
                "the base, so strong segmentation can never rescue collapsed "
                "localization"
            ),
            "beta_status": (
                "segmentation_modulation_beta is supplied explicitly and is a "
                "sensitivity hypothesis, not a frozen constant"
            ),
        },
        "reward": {
            "scalar": (
                "r = w_quality * Q_perc - w_latency * (L / B) "
                "- lambda_mode * 1[mode changed] "
                "- lambda_q * |q_exec,t - q_exec,t-1|"
            ),
            "latency": (
                "L_ns = CompletedTicket.resolution_ns - "
                "CompletedTicket.opened_ns, derived from controller "
                "timestamps and never accepted from a caller"
            ),
            "latency_clock_domain": REWARD_LATENCY_CLOCK_DOMAIN,
            "clock_domains": tuple(d.value for d in ClockDomain),
            "forbidden_latency_sources": FORBIDDEN_REWARD_LATENCY_SOURCES,
            "optimization_contract_costs": OPTIMIZATION_CONTRACT_COSTS,
            "diagnostic_only_signals": DIAGNOSTIC_ONLY_SIGNALS,
            "latency_excess_status": (
                "c_latency_excess is retained as an unmeasured diagnostic and "
                "is excluded from the optimization contract: the controller "
                "rejects post-deadline feedback, so it is structurally zero "
                "whenever measurable and unmeasurable otherwise"
            ),
            "switch_penalties": (
                "derived from the two exact executed actions; at an episode "
                "start no previous action exists and the terms are recorded as "
                "inapplicable rather than as an indistinguishable zero"
            ),
            "terminal_handling": {
                TerminalClass.REWARD_FINAL_EXACT.value: (
                    "structurally eligible only after producer authentication; "
                    "exact Q and L plus a verified ACK binding are required, "
                    "but the current fixture producer fails closed and cannot "
                    "enter learning/replay"
                ),
                TerminalClass.ACTION_PATH_FAILURE.value: (
                    "registered negative failure outcome; Q is never fabricated"
                ),
                TerminalClass.FEEDBACK_TIMEOUT.value: (
                    "censored pending reconciliation.  Phase 4a.2 deliberately "
                    "does not score it because no verified timeout-"
                    "reconciliation evidence carrier exists yet"
                ),
                TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED.value: (
                    "excluded; never converted into an agent penalty"
                ),
            },
            "controller_dispositions": {
                terminal.value: disposition
                for terminal, disposition in TERMINAL_LEARNING_DISPOSITION.items()
            },
            "adjudication_binding": (
                "arbitrary adjudicator ids and evidence hashes are refused by "
                "evaluate_completed_decision; all timeouts remain censored "
                "until a separately reviewed reconciliation protocol exists"
            ),
        },
        "causal_state": {
            "rule": (
                "only information available before the new policy decision; no "
                "image or radar tensor, dynamic fraction, temporal-information "
                "descriptor, object count, current-frame ground truth, "
                "network-profile label, action_id anchor lookup or future "
                "measurement"
            ),
            "clock_domain": ClockDomain.UE_LOCAL_MONOTONIC.value,
            "ages": (
                "every age is derived as observed_ns - measured_ns from a "
                "timestamped observation record; a caller cannot assert one, "
                "and a measurement postdating the observation is refused"
            ),
            "scene_binding": (
                "the scene descriptor is bound to its CARLA frame and its "
                "source SHA-256, and must match the state's frame"
            ),
            "radio_semantics": (
                "typed SNR/MCS/BSR sources; an eight-LCG BSR vector with "
                "per-entry validity and missing reasons; source-event wall "
                "time, collector wall/monotonic ingest and UE-local policy "
                "availability retained as distinct facts; zero fill and "
                "forward fill are forbidden"
            ),
            "radio_admission": (
                "Phase 4a.2 admits only factory-attested, explicitly "
                "privileged simulator/testbed radio observations.  The stock "
                "OAI collectors have no measured UE-visible feedback/IPC "
                "availability path, so UE_VISIBLE_RUNTIME fails closed even "
                "when a caller supplies well-shaped timestamps and hashes.  "
                "Collector evidence has a separate diagnostic-only type and "
                "cannot enter CausalStateV1"
            ),
            "causality_chain": (
                "measured_ns and previous.available_ns <= state.observed_ns "
                "<= ticket.opened_ns; "
                "next_state.observed_ns >= ticket.closed_ns; "
                "next_state.tensor_seq follows every held tensor; "
                "next_state.previous is the immutable closure-time snapshot, "
                "never a later replay adjudication"
            ),
            "predecessor_proof": (
                "every state carries exactly one attested predecessor: a "
                "policy-visible previous outcome or an episode-start proof "
                "bound to a controller genesis proof issued before the first "
                "policy decision. Replay later requires the realized ticket "
                "to be ordinal zero with no predecessor. Every "
                "non-genesis transition requires the exact predecessor "
                "ticket hash and consecutive controller completion ordinal; "
                "decision_seq gaps are legitimate and never used as the "
                "adjacency test"
            ),
            "deployability": POLICY_OBSERVATION_DEPLOYABILITY,
            "deployability_reason": (
                "the observation carries the previous decision's exact "
                "privileged CARLA ground-truth quality, which no physically "
                "deployed system can observe; this schema therefore claims no "
                "deployability"
            ),
            "candidate_features": CANDIDATE_STATE_FEATURES,
            "candidate_feature_status": CANDIDATE_FEATURE_VALIDATION_REQUIRED,
            "policy_feature_order": POLICY_FEATURE_ORDER,
            "policy_feature_count": POLICY_FEATURE_COUNT,
            "forbidden_feature_substrings": FORBIDDEN_POLICY_FEATURE_SUBSTRINGS,
            "previous_terminal_order": tuple(
                terminal.value for terminal in PREVIOUS_TERMINAL_ORDER
            ),
            "previous_terminal_feature_codes": {
                terminal.value: code
                for terminal, code in PREVIOUS_TERMINAL_FEATURE_CODES.items()
            },
            "normalization": (
                "every empirical scaling value is constructor supplied and "
                "carries a train-split identifier, fit population count and "
                "fit-config SHA-256; the spec also pins the SNR metric, MCS "
                "table and BSR scope it was fitted for"
            ),
            "freshness_binding": (
                "the attested feature vector carries the complete "
                "normalization and freshness records plus its source-state "
                "hash; both current and bootstrap vectors are recomputed and "
                "freshness-checked during transition construction/revalidation"
            ),
        },
        "transition": {
            "smdp": (
                "d = completed hold tensor count; discount multiplier = "
                "gamma_per_tensor ** d; both derived, never caller supplied"
            ),
            "policy_provenance": (
                "PolicyDecisionTraceV1 carries the sampled joint mode, the "
                "sampled continuous q, the executed q_e4 and complete action "
                "identity, actor version hash, source session/decision, "
                "state/feature hash and both preprocessing-spec hashes, and "
                "verifies the sampled-to-executed quantization with the "
                "registered half-up rule.  Behaviour log-probabilities are "
                "deliberately absent: off-policy SAC recomputes current-policy "
                "densities and does not consume behaviour density.  Any future "
                "importance-weighted method requires a separately versioned, "
                "distribution-complete trace. The controller commits this "
                "trace's canonical digest at gate time before execution, and "
                "replay requires that exact digest"
            ),
            "invariants": [
                "exactly one session and one decision throughout",
                "the executed action equals every held tensor's action",
                "the state joins the ticket's reward-requested tensor and frame",
                "the state precedes the decision it conditioned",
                "the next state follows ticket closure and every held tensor",
                "next_state.previous is exactly the closure-time policy snapshot",
                "missing previous history requires an episode-start proof",
                "state and next-state features are recomputed and fresh",
                "the actor trace binds session/decision/state/features/specs",
                "the controller ticket precommits that exact actor trace "
                "before executing the action",
                "every quality-bearing transition binds a verified ACK plus "
                "raw_quality_ack_sha256 and detailed_evidence_sha256",
                "no scalar reward for a censored or excluded transition",
                "an infrastructure fault is never a failure reward",
                "the outcome survives re-derivation from its frozen sources",
                "canonical JSON serialization and SHA-256 are deterministic",
                "records are immutable or returned as defensive copies",
            ],
        },
    }
)

SCHEMA_SHA256: str = canonical_sha256(SCHEMA_DESCRIPTOR)
