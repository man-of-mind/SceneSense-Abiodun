"""Minimal, side-effect-free state/reward/transition contract for Run 4.

This module freezes only the semantics needed before a Run-4 implementation can
be wired.  It deliberately does not implement a network kernel, scheduler,
replay buffer, trainer, live adapter, or evidence collector.

The policy sees exactly 21 numeric features.  Source timestamps, availability,
validity, freshness and identity are mandatory causal metadata, but are checked
by :func:`guard_state_for_action` outside the actor and never appended to the
feature vector.  The radio feature is the latest strictly prior, UE-decoded,
round-0 granted/final UL MCS under the registered SINR-driven gNB scheduler.
It is the value the UE actually decoded after scheduler constraints, not a
fabricated UE SNR or an unobserved intermediate lookup value.  Missing
or stale measurements raise
:class:`ExternalFallbackRequired`; they are never represented by numeric zero.

Importing this module reads no files and starts no runtime component.  The only
dependencies reused from Run 1 are immutable action/identity constants and
types whose own imports are side-effect free.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ACTION_IDENTITY_SCHEMA_SHA256,
    MINIMUM_HOLD_TENSORS,
    ExecutedActionIdentity,
    canonical_json_bytes,
    canonical_sha256,
)

__all__ = [
    # errors
    "Run4ContractError",
    "MetadataError",
    "ExternalFallbackRequired",
    "ScalingError",
    "RewardError",
    "TransitionError",
    # schemas
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_DESCRIPTOR",
    "SCHEMA_SHA256",
    "FEATURE_SCHEMA_ID",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA_SHA256",
    "REWARD_SCHEMA_ID",
    "REWARD_SCHEMA_VERSION",
    "REWARD_SCHEMA_SHA256",
    "TRANSITION_SCHEMA_ID",
    "TRANSITION_SCHEMA_VERSION",
    "TRANSITION_SCHEMA_SHA256",
    # fixed semantics
    "POLICY_FEATURE_ORDER",
    "POLICY_FEATURE_COUNT",
    "FORBIDDEN_POLICY_FEATURE_TERMS",
    "REWARD_DEADLINE_MS",
    "REWARD_DEADLINE_NS",
    "REWARD_LATENCY_WEIGHT",
    "REGISTERED_FAILURE_REWARD",
    "UL_MCS_TABLE_ID",
    "UL_MCS_INDEX_MIN",
    "UL_MCS_INDEX_MAX",
    "UL_MCS_POLICY_ID",
    "UL_MCS_SELECTION_RULE_ID",
    "TRANSMIT_CADENCE_HZ",
    "TRANSMIT_PERIOD_NS",
    "TRAINING_EVIDENCE_CLASS",
    "LIVE_EVIDENCE_STATUS",
    "assert_policy_feature_schema",
    # metadata/state
    "MeasurementKind",
    "Observer",
    "LinkDirection",
    "DecisionIdentityV1",
    "SampleIdentityV1",
    "MeasurementMetadataV1",
    "ScalarObservationV1",
    "DecisionBoundaryV1",
    "PriorUlGrantObservationV1",
    "FreshnessPolicyV2",
    "EmpiricalScalingV2",
    "PreviousOutcomeV1",
    "PolicyStateV2",
    "GuardedPolicyStateV2",
    "PolicyFeatureVectorV2",
    "guard_state_for_action",
    "build_policy_features",
    # reward
    "RewardEventKind",
    "RewardTerminal",
    "RewardEventV1",
    "RewardResolutionV1",
    "resolve_reward",
    # hold/transition
    "PayloadEvidenceClass",
    "EpisodeBoundary",
    "HoldTensorV1",
    "ActionHoldV1",
    "SemiMarkovTransitionV2",
    "build_transition",
]


# ---------------------------------------------------------------------------
# Errors and pure validation helpers
# ---------------------------------------------------------------------------


class Run4ContractError(ValueError):
    """Base class for Run-4 contract violations."""


class MetadataError(Run4ContractError):
    """Required causal metadata is malformed or internally inconsistent."""


class ExternalFallbackRequired(Run4ContractError):
    """The actor must not run; the external adapter must choose its fallback."""


class ScalingError(Run4ContractError):
    """An empirical scaling binding is missing or invalid."""


class RewardError(Run4ContractError):
    """A feedback event or reward resolution violates the frozen reward."""


class TransitionError(Run4ContractError):
    """A candidate transition is not one causal semi-Markov successor."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _exact_int(value: Any, name: str, error: type[Run4ContractError]) -> int:
    if type(value) is not int:
        raise error(f"{name} must be an exact int, got {type(value).__name__}")
    return value


def _non_negative_int(
    value: Any, name: str, error: type[Run4ContractError]
) -> int:
    value = _exact_int(value, name, error)
    if value < 0:
        raise error(f"{name} must be >= 0, got {value}")
    return value


def _positive_int(value: Any, name: str, error: type[Run4ContractError]) -> int:
    value = _exact_int(value, name, error)
    if value <= 0:
        raise error(f"{name} must be > 0, got {value}")
    return value


def _finite_float(value: Any, name: str, error: type[Run4ContractError]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a finite real scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{name} must be finite, got {value!r}")
    return result


def _closed_unit(value: Any, name: str, error: type[Run4ContractError]) -> float:
    result = _finite_float(value, name, error)
    if not 0.0 <= result <= 1.0:
        raise error(f"{name} must lie in [0, 1], got {result}")
    return result


def _non_empty_str(value: Any, name: str, error: type[Run4ContractError]) -> str:
    if not isinstance(value, str) or value == "":
        raise error(f"{name} must be a non-empty str, got {value!r}")
    return value


def _sha256_hex(value: Any, name: str, error: type[Run4ContractError]) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise error(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _canonical_uuid(value: Any, name: str, error: type[Run4ContractError]) -> str:
    if not isinstance(value, str):
        raise error(f"{name} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (TypeError, AttributeError, ValueError) as exc:
        raise error(f"{name} is not a UUID: {value!r}") from exc
    if str(parsed) != value:
        raise error(f"{name} must use canonical lowercase hyphenated form")
    return value


def _strict_bool(value: Any, name: str, error: type[Run4ContractError]) -> bool:
    if type(value) is not bool:
        raise error(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _action(value: Any, name: str, error: type[Run4ContractError]) -> ExecutedActionIdentity:
    if not isinstance(value, ExecutedActionIdentity):
        raise error(f"{name} must be an ExecutedActionIdentity")
    value.require_reconciled()
    return value


def _record(record_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "contract_schema_id": SCHEMA_ID,
        "contract_schema_sha256": SCHEMA_SHA256,
        "contract_schema_version": SCHEMA_VERSION,
        "record_type": record_type,
    }
    result.update(payload)
    return result


class _CanonicalRecord:
    """Small common canonical-serialization surface for immutable records."""

    __slots__ = ()
    RECORD_TYPE = "abstract"

    def _payload(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_canonical_dict(self) -> Dict[str, Any]:
        return _record(self.RECORD_TYPE, self._payload())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# ---------------------------------------------------------------------------
# Frozen feature, reward and transition semantics
# ---------------------------------------------------------------------------


POLICY_FEATURE_ORDER: Tuple[str, ...] = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{index}_one_hot" for index in range(EXPECTED_MODE_COUNT)),
    "prev_q_normalized",
    "prev_quality_qperc",
    "prev_latency_normalized",
    "prev_present",
    "prev_success",
)
POLICY_FEATURE_COUNT = 21

# These terms are forbidden in actor feature names.  They can occur in typed
# metadata or diagnostic records, but never in POLICY_FEATURE_ORDER.
FORBIDDEN_POLICY_FEATURE_TERMS: Tuple[str, ...] = (
    "age",
    "timestamp",
    "fresh",
    "valid",
    "source",
    "identity",
    "tbs",
    "grant",
    "network_profile",
    "gnb",
    "pusch",
    "frame_id",
    "current_outcome",
    "future_outcome",
    "prev_reward",
    "fps",
)

REWARD_DEADLINE_MS = 170.0
REWARD_DEADLINE_NS = 170_000_000
REWARD_LATENCY_WEIGHT = 0.25
REGISTERED_FAILURE_REWARD = -1.0

# The local SINR-driven scheduler uses TS 38.214 table 0.  The policy observes
# the granted/final index (0..28) carried by the latest strictly prior,
# UE-decoded round-0/new-data UL DCI, after any scheduler-side constraints.
UL_MCS_TABLE_ID = 0
UL_MCS_INDEX_MIN = 0
UL_MCS_INDEX_MAX = 28
UL_MCS_POLICY_ID = "SCENESENSE_MCS_POLICY=sinr"
UL_MCS_SELECTION_RULE_ID = "LATEST_STRICTLY_PRIOR_UE_UL_DCI_ROUND0"

TRANSMIT_CADENCE_HZ = 10
TRANSMIT_PERIOD_NS = 1_000_000_000 // TRANSMIT_CADENCE_HZ
TRAINING_EVIDENCE_CLASS = "SEMI_EMPIRICAL_RUN4_TRAINING_CONTRACT_V2"
LIVE_EVIDENCE_STATUS = "NOT_ESTABLISHED_BY_THIS_CONTRACT"


def assert_policy_feature_schema() -> None:
    """Recheck the exact 21-feature allow-list and leakage deny-list."""
    if len(POLICY_FEATURE_ORDER) != POLICY_FEATURE_COUNT:
        raise Run4ContractError(
            f"feature count drift: {len(POLICY_FEATURE_ORDER)} != "
            f"{POLICY_FEATURE_COUNT}"
        )
    if len(set(POLICY_FEATURE_ORDER)) != POLICY_FEATURE_COUNT:
        raise Run4ContractError("policy feature names must be unique")
    for name in POLICY_FEATURE_ORDER:
        lowered = name.lower()
        for forbidden in FORBIDDEN_POLICY_FEATURE_TERMS:
            if forbidden in lowered:
                raise Run4ContractError(
                    f"forbidden actor leakage term {forbidden!r} in {name!r}"
                )


assert_policy_feature_schema()

FEATURE_SCHEMA_ID = "splitfusion_run4_policy_features_v2"
FEATURE_SCHEMA_VERSION = 2
FEATURE_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": FEATURE_SCHEMA_ID,
        "version": FEATURE_SCHEMA_VERSION,
        "feature_order": POLICY_FEATURE_ORDER,
        "feature_count": POLICY_FEATURE_COUNT,
        "previous_mode_count": EXPECTED_MODE_COUNT,
        "genesis": "all previous fields zero; prev_present=0",
        "previous_success": (
            "action, q_perc and latency required; prev_present=prev_success=1"
        ),
        "previous_failure_or_timeout": (
            "action required; quality and latency absent and encoded zero only "
            "under prev_present=1, prev_success=0"
        ),
        "previous_reward": (
            "not a feature: exactly derivable from previous quality, latency, "
            "success/failure and the frozen reward schema"
        ),
        "external_metadata": (
            "timestamp/source/availability/freshness/validity/identity are "
            "mandatory and guard-only; none enters the actor vector"
        ),
        "forbidden_feature_terms": FORBIDDEN_POLICY_FEATURE_TERMS,
        "scaling": (
            "camera and log1p-backlog parameters are constructor-bound "
            "empirical inputs with no production defaults; UL MCS uses the "
            "registered table-0 wire range [0,28]"
        ),
    }
)
FEATURE_SCHEMA_SHA256 = canonical_sha256(FEATURE_SCHEMA_DESCRIPTOR)

REWARD_SCHEMA_ID = "splitfusion_run4_reward_v1"
REWARD_SCHEMA_VERSION = 1
REWARD_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": REWARD_SCHEMA_ID,
        "version": REWARD_SCHEMA_VERSION,
        "clock": "action-open to feedback, one monotonic clock domain",
        "deadline_ms": REWARD_DEADLINE_MS,
        "success_boundary": "inclusive latency <= deadline",
        "success": "q_perc - 0.25 * (latency_ms / 170.0)",
        "registered_delivery_failure": REGISTERED_FAILURE_REWARD,
        "registered_service_failure": REGISTERED_FAILURE_REWARD,
        "timeout": REGISTERED_FAILURE_REWARD,
        "excluded": ("infrastructure_fault", "evaluator_fault"),
        "diagnostic_system_path": (
            "the approximately 200-ms sensor-start-to-feedback path is logged "
            "separately and is neither this reward clock nor an actor feature"
        ),
        "absent_terms": (
            "p_admit",
            "switch_penalty",
            "payload_penalty",
            "aggression_penalty",
        ),
    }
)
REWARD_SCHEMA_SHA256 = canonical_sha256(REWARD_SCHEMA_DESCRIPTOR)

TRANSITION_SCHEMA_ID = "splitfusion_run4_semi_markov_transition_v2"
TRANSITION_SCHEMA_VERSION = 2
TRANSITION_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": TRANSITION_SCHEMA_ID,
        "version": TRANSITION_SCHEMA_VERSION,
        "step": "one decision-to-feedback-or-timeout cycle",
        "advance": (
            "virtual semi-Markov advance only; training performs no literal "
            "wall-clock sleep"
        ),
        "successor": "next real policy decision, not a fixed image step",
        "action_hold": (
            "one action governs the reward tensor and every subsequently held "
            "tensor until closure"
        ),
        "minimum_transmitted_tensors": MINIMUM_HOLD_TENSORS,
        "nominal_transmit_cadence_hz": TRANSMIT_CADENCE_HZ,
        "held_tensor_rule": (
            "held tensors add their evidence-typed offered payload to queue "
            "input but request no reward; exact measured nodes and same-scene "
            "modeled interpolation are never conflated or silently rounded"
        ),
        "duration": (
            "exact positive transmitted-tensor count plus exact positive elapsed "
            "virtual time; no scene-frame adjacency is asserted"
        ),
        "discount": "caller supplies gamma and stored gamma**duration",
        "episode_boundary": (
            "CONTINUES requires the real successor; TERMINATED/TRUNCATED omit "
            "it and force bootstrap_discount=0; timeout alone is nonterminal"
        ),
        "evidence_class": TRAINING_EVIDENCE_CLASS,
        "live_deployment_evidence": LIVE_EVIDENCE_STATUS,
        "not_implemented": (
            "network kernel",
            "scheduler",
            "replay buffer",
            "trainer",
            "training",
            "live runtime",
        ),
    }
)
TRANSITION_SCHEMA_SHA256 = canonical_sha256(TRANSITION_SCHEMA_DESCRIPTOR)

SCHEMA_ID = "splitfusion_hybrid_sac_run4_contract_v2"
SCHEMA_VERSION = 2
SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "action_identity_schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
        "feature_schema": FEATURE_SCHEMA_DESCRIPTOR,
        "reward_schema": REWARD_SCHEMA_DESCRIPTOR,
        "transition_schema": TRANSITION_SCHEMA_DESCRIPTOR,
        "scope": "pure in-memory contract; no I/O or runtime launch",
    }
)
SCHEMA_SHA256 = canonical_sha256(SCHEMA_DESCRIPTOR)


# ---------------------------------------------------------------------------
# Typed causal metadata
# ---------------------------------------------------------------------------


class MeasurementKind(str, Enum):
    CAMERA_SI = "CAMERA_SI"
    RADAR_P40 = "RADAR_P40"
    UE_PRIOR_NEW_DATA_UL_MCS_INDEX = "UE_PRIOR_NEW_DATA_UL_MCS_INDEX"
    UE_DL_SNR_DB = "UE_DL_SNR_DB"
    UE_PRE_ACTION_RLC_BACKLOG_BYTES = "UE_PRE_ACTION_RLC_BACKLOG_BYTES"
    # Diagnostic-only physical measurements.  Neither can fill the causal
    # UE-decoded UL-MCS state slot.
    GNB_UL_PUSCH_SNR_DB = "GNB_UL_PUSCH_SNR_DB"


class Observer(str, Enum):
    SCENE_PIPELINE = "SCENE_PIPELINE"
    UE = "UE"
    GNB = "GNB"


class LinkDirection(str, Enum):
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class DecisionIdentityV1(_CanonicalRecord):
    session_uuid: str
    ue_id: str
    decision_seq: int

    RECORD_TYPE = "decision_identity_v1"

    def __post_init__(self) -> None:
        _canonical_uuid(self.session_uuid, "session_uuid", MetadataError)
        _non_empty_str(self.ue_id, "ue_id", MetadataError)
        _non_negative_int(self.decision_seq, "decision_seq", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "decision_seq": self.decision_seq,
            "session_uuid": self.session_uuid,
            "ue_id": self.ue_id,
        }


@dataclass(frozen=True, slots=True)
class SampleIdentityV1(_CanonicalRecord):
    session_uuid: str
    ue_id: str
    sample_seq: int

    RECORD_TYPE = "sample_identity_v1"

    def __post_init__(self) -> None:
        _canonical_uuid(self.session_uuid, "session_uuid", MetadataError)
        _non_empty_str(self.ue_id, "ue_id", MetadataError)
        _non_negative_int(self.sample_seq, "sample_seq", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "sample_seq": self.sample_seq,
            "session_uuid": self.session_uuid,
            "ue_id": self.ue_id,
        }


@dataclass(frozen=True, slots=True)
class MeasurementMetadataV1(_CanonicalRecord):
    identity: SampleIdentityV1
    kind: MeasurementKind
    observer: Observer
    link_direction: LinkDirection
    source: str
    source_timestamp_ns: int
    available_timestamp_ns: int
    clock_domain: str
    valid: bool

    RECORD_TYPE = "measurement_metadata_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SampleIdentityV1):
            raise MetadataError("identity must be SampleIdentityV1")
        if not isinstance(self.kind, MeasurementKind):
            raise MetadataError("kind must be MeasurementKind")
        if not isinstance(self.observer, Observer):
            raise MetadataError("observer must be Observer")
        if not isinstance(self.link_direction, LinkDirection):
            raise MetadataError("link_direction must be LinkDirection")
        _non_empty_str(self.source, "source", MetadataError)
        measured = _non_negative_int(
            self.source_timestamp_ns, "source_timestamp_ns", MetadataError
        )
        available = _non_negative_int(
            self.available_timestamp_ns, "available_timestamp_ns", MetadataError
        )
        if measured > available:
            raise MetadataError(
                "source_timestamp_ns must be <= available_timestamp_ns"
            )
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)
        _strict_bool(self.valid, "valid", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "available_timestamp_ns": self.available_timestamp_ns,
            "clock_domain": self.clock_domain,
            "identity": self.identity.to_canonical_dict(),
            "kind": self.kind.value,
            "link_direction": self.link_direction.value,
            "observer": self.observer.value,
            "source": self.source,
            "source_timestamp_ns": self.source_timestamp_ns,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class ScalarObservationV1(_CanonicalRecord):
    """A value plus mandatory metadata, including explicit invalidity.

    Invalid/missing observations carry ``value=None`` and a non-empty reason.
    A numeric zero is allowed only as an explicitly valid measurement (for
    example an empty RLC queue), never as a missing-value substitute.
    """

    value: Optional[float]
    metadata: MeasurementMetadataV1
    missing_reason: Optional[str]

    RECORD_TYPE = "scalar_observation_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MeasurementMetadataV1):
            raise MetadataError("metadata must be MeasurementMetadataV1")
        if self.metadata.valid:
            if self.value is None:
                raise MetadataError("a valid observation requires a value")
            _finite_float(self.value, "value", MetadataError)
            if self.missing_reason is not None:
                raise MetadataError(
                    "a valid observation cannot carry missing_reason"
                )
        else:
            if self.value is not None:
                raise MetadataError(
                    "an invalid/missing observation must use value=None; "
                    "numeric zero-fill is forbidden"
                )
            _non_empty_str(self.missing_reason, "missing_reason", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_canonical_dict(),
            "missing_reason": self.missing_reason,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class PriorUlGrantObservationV1(_CanonicalRecord):
    """UE-decoded prior new-data UL grant used by the policy.

    This wrapper makes the scheduler/table/HARQ semantics part of the hashed
    state evidence.  A valid sample must be a table-0, round-0 UL MCS index.
    A missing sample remains explicitly missing; MCS 0 is never its sentinel.
    """

    observation: ScalarObservationV1
    mcs_table: int
    harq_round: Optional[int]
    new_data_indicator: Optional[int]
    grant_identity: Optional[str]
    scheduler_policy_id: str
    selection_rule_id: str

    RECORD_TYPE = "prior_ul_grant_observation_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ScalarObservationV1):
            raise MetadataError("observation must be ScalarObservationV1")
        metadata = self.observation.metadata
        if (
            metadata.kind is not MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX
            or metadata.observer is not Observer.UE
            or metadata.link_direction is not LinkDirection.UPLINK
        ):
            raise MetadataError(
                "prior UL grant must be UE-observed uplink MCS evidence"
            )
        if type(self.mcs_table) is not int or self.mcs_table != UL_MCS_TABLE_ID:
            raise MetadataError(
                f"mcs_table must be exactly {UL_MCS_TABLE_ID}"
            )
        if self.scheduler_policy_id != UL_MCS_POLICY_ID:
            raise MetadataError(
                f"scheduler_policy_id must be exactly {UL_MCS_POLICY_ID!r}"
            )
        if self.selection_rule_id != UL_MCS_SELECTION_RULE_ID:
            raise MetadataError(
                f"selection_rule_id must be exactly "
                f"{UL_MCS_SELECTION_RULE_ID!r}"
            )
        if metadata.valid:
            if type(self.harq_round) is not int or self.harq_round != 0:
                raise MetadataError(
                    "a valid prior policy MCS must come from HARQ round 0"
                )
            if (
                type(self.new_data_indicator) is not int
                or self.new_data_indicator not in (0, 1)
            ):
                raise MetadataError(
                    "a valid prior policy MCS requires NDI exactly 0 or 1"
                )
            _non_empty_str(self.grant_identity, "grant_identity", MetadataError)
            mcs = self.observation.value
            if (
                type(mcs) is not int
                or not UL_MCS_INDEX_MIN <= mcs <= UL_MCS_INDEX_MAX
            ):
                raise MetadataError(
                    "prior UL MCS must be an exact table-0 index in [0, 28]"
                )
        elif (
            self.harq_round is not None
            or self.new_data_indicator is not None
            or self.grant_identity is not None
        ):
            raise MetadataError(
                "a missing prior UL grant must omit round, NDI and identity"
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "harq_round": self.harq_round,
            "grant_identity": self.grant_identity,
            "mcs_table": self.mcs_table,
            "new_data_indicator": self.new_data_indicator,
            "observation": self.observation.to_canonical_dict(),
            "scheduler_policy_id": self.scheduler_policy_id,
            "selection_rule_id": self.selection_rule_id,
        }


@dataclass(frozen=True, slots=True)
class DecisionBoundaryV1(_CanonicalRecord):
    identity: DecisionIdentityV1
    state_commit_timestamp_ns: int
    action_open_timestamp_ns: int
    clock_domain: str

    RECORD_TYPE = "decision_boundary_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise MetadataError("identity must be DecisionIdentityV1")
        committed = _non_negative_int(
            self.state_commit_timestamp_ns,
            "state_commit_timestamp_ns",
            MetadataError,
        )
        opened = _non_negative_int(
            self.action_open_timestamp_ns,
            "action_open_timestamp_ns",
            MetadataError,
        )
        if committed >= opened:
            raise MetadataError(
                "state_commit_timestamp_ns must be strictly before "
                "action_open_timestamp_ns"
            )
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "action_open_timestamp_ns": self.action_open_timestamp_ns,
            "clock_domain": self.clock_domain,
            "identity": self.identity.to_canonical_dict(),
            "state_commit_timestamp_ns": self.state_commit_timestamp_ns,
        }


@dataclass(frozen=True, slots=True)
class FreshnessPolicyV2(_CanonicalRecord):
    """Explicit caller binding; this module invents no production age limits."""

    policy_id: str
    policy_version: int
    evidence_sha256: str
    camera_si_max_age_ns: int
    radar_p40_max_age_ns: int
    prior_ul_mcs_max_age_ns: int
    pre_action_rlc_backlog_max_age_ns: int

    RECORD_TYPE = "freshness_policy_v2"

    def __post_init__(self) -> None:
        _non_empty_str(self.policy_id, "policy_id", MetadataError)
        _positive_int(self.policy_version, "policy_version", MetadataError)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", MetadataError)
        for name in (
            "camera_si_max_age_ns",
            "radar_p40_max_age_ns",
            "prior_ul_mcs_max_age_ns",
            "pre_action_rlc_backlog_max_age_ns",
        ):
            _positive_int(getattr(self, name), name, MetadataError)

    def max_age_ns(self, kind: MeasurementKind) -> int:
        try:
            return {
                MeasurementKind.CAMERA_SI: self.camera_si_max_age_ns,
                MeasurementKind.RADAR_P40: self.radar_p40_max_age_ns,
                MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX: (
                    self.prior_ul_mcs_max_age_ns
                ),
                MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES: (
                    self.pre_action_rlc_backlog_max_age_ns
                ),
            }[kind]
        except KeyError as exc:
            raise MetadataError(f"no actor freshness rule for {kind.value}") from exc

    def _payload(self) -> Dict[str, Any]:
        return {
            "camera_si_max_age_ns": self.camera_si_max_age_ns,
            "evidence_sha256": self.evidence_sha256,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "pre_action_rlc_backlog_max_age_ns": (
                self.pre_action_rlc_backlog_max_age_ns
            ),
            "radar_p40_max_age_ns": self.radar_p40_max_age_ns,
            "prior_ul_mcs_max_age_ns": self.prior_ul_mcs_max_age_ns,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalScalingV2(_CanonicalRecord):
    """Empirical normalization supplied by a versioned fit, with no defaults."""

    scaling_id: str
    scaling_version: int
    evidence_sha256: str
    camera_si_center: float
    camera_si_scale: float
    backlog_log1p_scale: float

    RECORD_TYPE = "empirical_scaling_v2"

    def __post_init__(self) -> None:
        _non_empty_str(self.scaling_id, "scaling_id", ScalingError)
        _positive_int(self.scaling_version, "scaling_version", ScalingError)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", ScalingError)
        _finite_float(self.camera_si_center, "camera_si_center", ScalingError)
        for name in (
            "camera_si_scale",
            "backlog_log1p_scale",
        ):
            value = _finite_float(getattr(self, name), name, ScalingError)
            if value <= 0.0:
                raise ScalingError(f"{name} must be > 0, got {value}")

    def _payload(self) -> Dict[str, Any]:
        return {
            "backlog_log1p_scale": float(self.backlog_log1p_scale),
            "camera_si_center": float(self.camera_si_center),
            "camera_si_scale": float(self.camera_si_scale),
            "evidence_sha256": self.evidence_sha256,
            "scaling_id": self.scaling_id,
            "scaling_version": self.scaling_version,
        }


# ---------------------------------------------------------------------------
# Reward resolution and previous-outcome state
# ---------------------------------------------------------------------------


class RewardEventKind(str, Enum):
    DELIVERED_SUCCESS = "DELIVERED_SUCCESS"
    REGISTERED_DELIVERY_FAILURE = "REGISTERED_DELIVERY_FAILURE"
    REGISTERED_SERVICE_FAILURE = "REGISTERED_SERVICE_FAILURE"
    TIMEOUT = "TIMEOUT"
    INFRASTRUCTURE_FAULT = "INFRASTRUCTURE_FAULT"
    EVALUATOR_FAULT = "EVALUATOR_FAULT"


class RewardTerminal(str, Enum):
    SUCCESS = "SUCCESS"
    REGISTERED_DELIVERY_FAILURE = "REGISTERED_DELIVERY_FAILURE"
    REGISTERED_SERVICE_FAILURE = "REGISTERED_SERVICE_FAILURE"
    TIMEOUT = "TIMEOUT"
    INFRASTRUCTURE_FAULT = "INFRASTRUCTURE_FAULT"
    EVALUATOR_FAULT = "EVALUATOR_FAULT"


@dataclass(frozen=True, slots=True)
class RewardEventV1(_CanonicalRecord):
    identity: DecisionIdentityV1
    action: ExecutedActionIdentity
    kind: RewardEventKind
    action_open_timestamp_ns: int
    resolution_timestamp_ns: int
    clock_domain: str
    source: str
    q_perc: Optional[float] = None

    RECORD_TYPE = "reward_event_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise RewardError("identity must be DecisionIdentityV1")
        _action(self.action, "action", RewardError)
        if not isinstance(self.kind, RewardEventKind):
            raise RewardError("kind must be RewardEventKind")
        opened = _non_negative_int(
            self.action_open_timestamp_ns,
            "action_open_timestamp_ns",
            RewardError,
        )
        resolved = _non_negative_int(
            self.resolution_timestamp_ns,
            "resolution_timestamp_ns",
            RewardError,
        )
        if resolved < opened:
            raise RewardError("resolution_timestamp_ns cannot precede action open")
        _non_empty_str(self.clock_domain, "clock_domain", RewardError)
        _non_empty_str(self.source, "source", RewardError)
        if self.kind is RewardEventKind.DELIVERED_SUCCESS:
            if self.q_perc is None:
                raise RewardError("DELIVERED_SUCCESS requires q_perc")
            _closed_unit(self.q_perc, "q_perc", RewardError)
        elif self.q_perc is not None:
            raise RewardError(
                f"{self.kind.value} must not carry q_perc; failed/excluded "
                "outcomes have no quality value"
            )
        if (
            self.kind is RewardEventKind.TIMEOUT
            and resolved <= opened + REWARD_DEADLINE_NS
        ):
            raise RewardError(
                "TIMEOUT must be resolved strictly after the inclusive deadline"
            )

    @property
    def elapsed_ns(self) -> int:
        return self.resolution_timestamp_ns - self.action_open_timestamp_ns

    def _payload(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_canonical_dict(),
            "action_open_timestamp_ns": self.action_open_timestamp_ns,
            "clock_domain": self.clock_domain,
            "identity": self.identity.to_canonical_dict(),
            "kind": self.kind.value,
            "q_perc": self.q_perc,
            "resolution_timestamp_ns": self.resolution_timestamp_ns,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_sha256": REWARD_SCHEMA_SHA256,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "source": self.source,
        }


def _make_attestation_gate() -> Tuple[Any, Any]:
    sentinel = object()

    def issue(binding: str) -> Tuple[Any, str]:
        return sentinel, binding

    def valid(token: Any, binding: str) -> bool:
        return (
            type(token) is tuple
            and len(token) == 2
            and token[0] is sentinel
            and token[1] == binding
        )

    return issue, valid


_issue_resolution, _valid_resolution = _make_attestation_gate()
_issue_guard, _valid_guard = _make_attestation_gate()
_issue_features, _valid_features = _make_attestation_gate()
_issue_transition, _valid_transition = _make_attestation_gate()


@dataclass(frozen=True, slots=True)
class RewardResolutionV1(_CanonicalRecord):
    identity: DecisionIdentityV1
    action: ExecutedActionIdentity
    terminal: RewardTerminal
    action_open_timestamp_ns: int
    resolution_timestamp_ns: int
    clock_domain: str
    learning_included: bool
    reward: Optional[float]
    q_perc: Optional[float]
    latency_ms: Optional[float]
    event_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "reward_resolution_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise RewardError("identity must be DecisionIdentityV1")
        _action(self.action, "action", RewardError)
        if not isinstance(self.terminal, RewardTerminal):
            raise RewardError("terminal must be RewardTerminal")
        opened = _non_negative_int(
            self.action_open_timestamp_ns,
            "action_open_timestamp_ns",
            RewardError,
        )
        resolved = _non_negative_int(
            self.resolution_timestamp_ns,
            "resolution_timestamp_ns",
            RewardError,
        )
        if resolved < opened:
            raise RewardError("resolution cannot precede action open")
        _non_empty_str(self.clock_domain, "clock_domain", RewardError)
        _strict_bool(self.learning_included, "learning_included", RewardError)
        _sha256_hex(self.event_sha256, "event_sha256", RewardError)

        if self.terminal is RewardTerminal.SUCCESS:
            if not self.learning_included:
                raise RewardError("SUCCESS must be included")
            if self.reward is None or self.q_perc is None or self.latency_ms is None:
                raise RewardError("SUCCESS requires reward, q_perc and latency_ms")
            _finite_float(self.reward, "reward", RewardError)
            _closed_unit(self.q_perc, "q_perc", RewardError)
            latency = _finite_float(self.latency_ms, "latency_ms", RewardError)
            if not 0.0 <= latency <= REWARD_DEADLINE_MS:
                raise RewardError("SUCCESS latency must lie in [0, 170] ms")
        elif self.terminal in (
            RewardTerminal.REGISTERED_DELIVERY_FAILURE,
            RewardTerminal.REGISTERED_SERVICE_FAILURE,
            RewardTerminal.TIMEOUT,
        ):
            if not self.learning_included:
                raise RewardError("registered failure/timeout must be included")
            if self.reward != REGISTERED_FAILURE_REWARD:
                raise RewardError("registered failure/timeout reward must be -1")
            if self.q_perc is not None or self.latency_ms is not None:
                raise RewardError(
                    "registered failure/timeout must have absent quality/latency"
                )
        else:
            if self.learning_included:
                raise RewardError("infrastructure/evaluator faults are excluded")
            if any(
                value is not None
                for value in (self.reward, self.q_perc, self.latency_ms)
            ):
                raise RewardError(
                    "excluded faults must not carry reward, quality or latency"
                )

        if self._attestation is not None and not _valid_resolution(
            self._attestation, self._binding()
        ):
            raise RewardError("reward-resolution attestation is invalid")

    def _payload(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_canonical_dict(),
            "action_open_timestamp_ns": self.action_open_timestamp_ns,
            "clock_domain": self.clock_domain,
            "event_sha256": self.event_sha256,
            "identity": self.identity.to_canonical_dict(),
            "latency_ms": self.latency_ms,
            "learning_included": self.learning_included,
            "q_perc": self.q_perc,
            "resolution_timestamp_ns": self.resolution_timestamp_ns,
            "reward": self.reward,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_sha256": REWARD_SCHEMA_SHA256,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "terminal": self.terminal.value,
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_attested(self) -> bool:
        return _valid_resolution(self._attestation, self._binding())

    def require_attested(self) -> None:
        if not self.is_attested:
            raise RewardError("resolution must be produced by resolve_reward()")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_attested()
        # dataclass(slots=True) returns a replacement class, which makes the
        # implicit __class__ cell used by zero-argument super() unsafe here.
        return _CanonicalRecord.to_canonical_dict(self)


def resolve_reward(event: RewardEventV1) -> RewardResolutionV1:
    """Resolve one event under the frozen inclusive 170-ms reward boundary."""
    if not isinstance(event, RewardEventV1):
        raise RewardError("event must be RewardEventV1")

    elapsed_ns = event.elapsed_ns
    if event.kind is RewardEventKind.DELIVERED_SUCCESS:
        if elapsed_ns <= REWARD_DEADLINE_NS:
            latency_ms = elapsed_ns / 1_000_000.0
            q_perc = float(event.q_perc)  # construction proved non-None
            terminal = RewardTerminal.SUCCESS
            included = True
            reward: Optional[float] = q_perc - REWARD_LATENCY_WEIGHT * (
                latency_ms / REWARD_DEADLINE_MS
            )
            resolution_q: Optional[float] = q_perc
            resolution_latency: Optional[float] = latency_ms
        else:
            # A delivery after the inclusive deadline cannot retroactively turn
            # a registered timeout into success.
            terminal = RewardTerminal.TIMEOUT
            included = True
            reward = REGISTERED_FAILURE_REWARD
            resolution_q = None
            resolution_latency = None
    elif event.kind is RewardEventKind.REGISTERED_DELIVERY_FAILURE:
        terminal = RewardTerminal.REGISTERED_DELIVERY_FAILURE
        included = True
        reward = REGISTERED_FAILURE_REWARD
        resolution_q = None
        resolution_latency = None
    elif event.kind is RewardEventKind.REGISTERED_SERVICE_FAILURE:
        terminal = RewardTerminal.REGISTERED_SERVICE_FAILURE
        included = True
        reward = REGISTERED_FAILURE_REWARD
        resolution_q = None
        resolution_latency = None
    elif event.kind is RewardEventKind.TIMEOUT:
        terminal = RewardTerminal.TIMEOUT
        included = True
        reward = REGISTERED_FAILURE_REWARD
        resolution_q = None
        resolution_latency = None
    elif event.kind is RewardEventKind.INFRASTRUCTURE_FAULT:
        terminal = RewardTerminal.INFRASTRUCTURE_FAULT
        included = False
        reward = None
        resolution_q = None
        resolution_latency = None
    else:
        terminal = RewardTerminal.EVALUATOR_FAULT
        included = False
        reward = None
        resolution_q = None
        resolution_latency = None

    candidate = RewardResolutionV1(
        identity=event.identity,
        action=event.action,
        terminal=terminal,
        action_open_timestamp_ns=event.action_open_timestamp_ns,
        resolution_timestamp_ns=event.resolution_timestamp_ns,
        clock_domain=event.clock_domain,
        learning_included=included,
        reward=reward,
        q_perc=resolution_q,
        latency_ms=resolution_latency,
        event_sha256=event.canonical_sha256(),
    )
    return replace(
        candidate,
        _attestation=_issue_resolution(candidate._binding()),
    )


@dataclass(frozen=True, slots=True)
class PreviousOutcomeV1(_CanonicalRecord):
    identity: DecisionIdentityV1
    action: ExecutedActionIdentity
    terminal: RewardTerminal
    q_perc: Optional[float]
    latency_ms: Optional[float]
    available_timestamp_ns: int
    clock_domain: str
    reward_resolution_sha256: str

    RECORD_TYPE = "previous_outcome_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise MetadataError("identity must be DecisionIdentityV1")
        _action(self.action, "action", MetadataError)
        if not isinstance(self.terminal, RewardTerminal):
            raise MetadataError("terminal must be RewardTerminal")
        _non_negative_int(
            self.available_timestamp_ns, "available_timestamp_ns", MetadataError
        )
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)
        _sha256_hex(
            self.reward_resolution_sha256,
            "reward_resolution_sha256",
            MetadataError,
        )
        if self.terminal is RewardTerminal.SUCCESS:
            if self.q_perc is None or self.latency_ms is None:
                raise MetadataError("previous success requires quality and latency")
            _closed_unit(self.q_perc, "q_perc", MetadataError)
            latency = _finite_float(self.latency_ms, "latency_ms", MetadataError)
            if not 0.0 <= latency <= REWARD_DEADLINE_MS:
                raise MetadataError("previous success latency is out of range")
        elif self.terminal in (
            RewardTerminal.REGISTERED_DELIVERY_FAILURE,
            RewardTerminal.REGISTERED_SERVICE_FAILURE,
            RewardTerminal.TIMEOUT,
        ):
            if self.q_perc is not None or self.latency_ms is not None:
                raise MetadataError(
                    "previous failure/timeout requires absent quality and latency"
                )
        else:
            raise MetadataError(
                "infrastructure/evaluator faults are excluded and cannot become "
                "a previous learning outcome"
            )

    @classmethod
    def from_resolution(cls, resolution: RewardResolutionV1) -> "PreviousOutcomeV1":
        if not isinstance(resolution, RewardResolutionV1):
            raise MetadataError("resolution must be RewardResolutionV1")
        resolution.require_attested()
        if not resolution.learning_included:
            raise MetadataError(
                "an excluded infrastructure/evaluator fault cannot create "
                "previous learning state"
            )
        return cls(
            identity=resolution.identity,
            action=resolution.action,
            terminal=resolution.terminal,
            q_perc=resolution.q_perc,
            latency_ms=resolution.latency_ms,
            available_timestamp_ns=resolution.resolution_timestamp_ns,
            clock_domain=resolution.clock_domain,
            reward_resolution_sha256=resolution.canonical_sha256(),
        )

    @property
    def success(self) -> bool:
        return self.terminal is RewardTerminal.SUCCESS

    @property
    def derived_reward(self) -> float:
        """Derive the prior reward; it is intentionally not an actor feature."""
        if not self.success:
            return REGISTERED_FAILURE_REWARD
        return float(self.q_perc) - REWARD_LATENCY_WEIGHT * (
            float(self.latency_ms) / REWARD_DEADLINE_MS
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_canonical_dict(),
            "available_timestamp_ns": self.available_timestamp_ns,
            "clock_domain": self.clock_domain,
            "identity": self.identity.to_canonical_dict(),
            "latency_ms": self.latency_ms,
            "q_perc": self.q_perc,
            "reward_resolution_sha256": self.reward_resolution_sha256,
            "terminal": self.terminal.value,
        }


# ---------------------------------------------------------------------------
# State guard and exact feature vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyStateV2(_CanonicalRecord):
    identity: DecisionIdentityV1
    camera_si: ScalarObservationV1
    radar_p40: ScalarObservationV1
    prior_ul_mcs: PriorUlGrantObservationV1
    pre_action_rlc_backlog: ScalarObservationV1
    previous: Optional[PreviousOutcomeV1]

    RECORD_TYPE = "policy_state_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise MetadataError("identity must be DecisionIdentityV1")
        for name in ("camera_si", "radar_p40", "pre_action_rlc_backlog"):
            if not isinstance(getattr(self, name), ScalarObservationV1):
                raise MetadataError(f"{name} must be ScalarObservationV1")
        if not isinstance(self.prior_ul_mcs, PriorUlGrantObservationV1):
            raise MetadataError(
                "prior_ul_mcs must be PriorUlGrantObservationV1"
            )
        if self.previous is not None:
            if not isinstance(self.previous, PreviousOutcomeV1):
                raise MetadataError("previous must be PreviousOutcomeV1 or None")
            previous_identity = self.previous.identity
            if (
                previous_identity.session_uuid != self.identity.session_uuid
                or previous_identity.ue_id != self.identity.ue_id
                or previous_identity.decision_seq + 1 != self.identity.decision_seq
            ):
                raise MetadataError(
                    "previous must be the immediately preceding decision from "
                    "the same session and UE"
                )

    def _payload(self) -> Dict[str, Any]:
        return {
            "camera_si": self.camera_si.to_canonical_dict(),
            "identity": self.identity.to_canonical_dict(),
            "pre_action_rlc_backlog": (
                self.pre_action_rlc_backlog.to_canonical_dict()
            ),
            "previous": (
                None if self.previous is None else self.previous.to_canonical_dict()
            ),
            "radar_p40": self.radar_p40.to_canonical_dict(),
            "prior_ul_mcs": self.prior_ul_mcs.to_canonical_dict(),
        }


_EXPECTED_MEASUREMENT_SEMANTICS: Mapping[
    str, Tuple[MeasurementKind, Observer, LinkDirection]
] = MappingProxyType(
    {
        "camera_si": (
            MeasurementKind.CAMERA_SI,
            Observer.SCENE_PIPELINE,
            LinkDirection.NOT_APPLICABLE,
        ),
        "radar_p40": (
            MeasurementKind.RADAR_P40,
            Observer.SCENE_PIPELINE,
            LinkDirection.NOT_APPLICABLE,
        ),
        "prior_ul_mcs": (
            MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
            Observer.UE,
            LinkDirection.UPLINK,
        ),
        "pre_action_rlc_backlog": (
            MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
            Observer.UE,
            LinkDirection.UPLINK,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class GuardedPolicyStateV2(_CanonicalRecord):
    state: PolicyStateV2
    boundary: DecisionBoundaryV1
    freshness_policy_sha256: str
    observation_ages_ns: Tuple[int, int, int, int]
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "guarded_policy_state_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.state, PolicyStateV2):
            raise MetadataError("state must be PolicyStateV2")
        if not isinstance(self.boundary, DecisionBoundaryV1):
            raise MetadataError("boundary must be DecisionBoundaryV1")
        _sha256_hex(
            self.freshness_policy_sha256,
            "freshness_policy_sha256",
            MetadataError,
        )
        if type(self.observation_ages_ns) is not tuple or len(
            self.observation_ages_ns
        ) != 4:
            raise MetadataError("observation_ages_ns must be a 4-tuple")
        for index, age in enumerate(self.observation_ages_ns):
            _non_negative_int(age, f"observation_ages_ns[{index}]", MetadataError)
        if self._attestation is not None and not _valid_guard(
            self._attestation, self._binding()
        ):
            raise MetadataError("state-guard attestation is invalid")

    def _payload(self) -> Dict[str, Any]:
        return {
            "boundary": self.boundary.to_canonical_dict(),
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "observation_ages_ns": list(self.observation_ages_ns),
            "state": self.state.to_canonical_dict(),
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_guarded(self) -> bool:
        return _valid_guard(self._attestation, self._binding())

    def require_guarded(self) -> None:
        if not self.is_guarded:
            raise ExternalFallbackRequired(
                "state was not admitted by guard_state_for_action()"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_guarded()
        return _CanonicalRecord.to_canonical_dict(self)


def guard_state_for_action(
    state: PolicyStateV2,
    boundary: DecisionBoundaryV1,
    freshness: FreshnessPolicyV2,
) -> GuardedPolicyStateV2:
    """Admit a causal, valid and fresh state or demand an external fallback."""
    if not isinstance(state, PolicyStateV2):
        raise MetadataError("state must be PolicyStateV2")
    if not isinstance(boundary, DecisionBoundaryV1):
        raise MetadataError("boundary must be DecisionBoundaryV1")
    if not isinstance(freshness, FreshnessPolicyV2):
        raise MetadataError("freshness must be FreshnessPolicyV2")
    if state.identity != boundary.identity:
        raise ExternalFallbackRequired("state/boundary decision identity mismatch")

    ages = []
    for slot_name in (
        "camera_si",
        "radar_p40",
        "prior_ul_mcs",
        "pre_action_rlc_backlog",
    ):
        value = getattr(state, slot_name)
        observation = (
            value.observation if slot_name == "prior_ul_mcs" else value
        )
        metadata = observation.metadata
        expected_kind, expected_observer, expected_direction = (
            _EXPECTED_MEASUREMENT_SEMANTICS[slot_name]
        )
        if not metadata.valid or observation.value is None:
            raise ExternalFallbackRequired(
                f"{slot_name} is missing/invalid; use external fallback"
            )
        if (
            metadata.kind is not expected_kind
            or metadata.observer is not expected_observer
            or metadata.link_direction is not expected_direction
        ):
            raise ExternalFallbackRequired(
                f"{slot_name} semantic mismatch: expected "
                f"{expected_kind.value}/{expected_observer.value}/"
                f"{expected_direction.value}, got {metadata.kind.value}/"
                f"{metadata.observer.value}/{metadata.link_direction.value}"
            )
        if (
            metadata.identity.session_uuid != state.identity.session_uuid
            or metadata.identity.ue_id != state.identity.ue_id
        ):
            raise ExternalFallbackRequired(
                f"{slot_name} sample identity does not match decision identity"
            )
        if metadata.clock_domain != boundary.clock_domain:
            raise ExternalFallbackRequired(
                f"{slot_name} and decision boundary use different clocks"
            )
        if metadata.available_timestamp_ns > boundary.state_commit_timestamp_ns:
            raise ExternalFallbackRequired(
                f"{slot_name} was not available when state was committed"
            )
        # boundary construction guarantees state_commit < action_open.  Together
        # these checks establish source <= available <= commit < action.
        age = boundary.action_open_timestamp_ns - metadata.source_timestamp_ns
        if age < 0:
            raise ExternalFallbackRequired(
                f"{slot_name} source timestamp is after action open"
            )
        if age > freshness.max_age_ns(expected_kind):
            raise ExternalFallbackRequired(
                f"{slot_name} is stale ({age} ns); use external fallback"
            )
        ages.append(age)

    if state.camera_si.metadata.identity != state.radar_p40.metadata.identity:
        raise ExternalFallbackRequired(
            "camera_si and radar_p40 must share the exact current scene "
            "SampleIdentity; asynchronous radio samples remain independent"
        )

    camera = float(state.camera_si.value)
    radar = float(state.radar_p40.value)
    mcs_raw = state.prior_ul_mcs.observation.value
    backlog_raw = state.pre_action_rlc_backlog.value
    if camera < 0.0:
        raise ExternalFallbackRequired("camera_si cannot be negative")
    if not 0.0 <= radar <= 1.0:
        raise ExternalFallbackRequired("radar_p40 must lie in [0, 1]")
    if type(mcs_raw) is not int or not UL_MCS_INDEX_MIN <= mcs_raw <= UL_MCS_INDEX_MAX:
        raise ExternalFallbackRequired(
            "prior_ul_mcs must be an exact table-0 MCS index in [0, 28]"
        )
    if type(backlog_raw) is not int or backlog_raw < 0:
        raise ExternalFallbackRequired(
            "pre_action_rlc_backlog must be an exact non-negative byte count"
        )

    if state.previous is not None:
        if state.previous.clock_domain != boundary.clock_domain:
            raise ExternalFallbackRequired(
                "previous outcome and state boundary use different clocks"
            )
        if state.previous.available_timestamp_ns > boundary.state_commit_timestamp_ns:
            raise ExternalFallbackRequired(
                "previous outcome was unavailable when state was committed"
            )

    candidate = GuardedPolicyStateV2(
        state=state,
        boundary=boundary,
        freshness_policy_sha256=freshness.canonical_sha256(),
        observation_ages_ns=tuple(ages),
    )
    return replace(candidate, _attestation=_issue_guard(candidate._binding()))


@dataclass(frozen=True, slots=True)
class PolicyFeatureVectorV2(_CanonicalRecord):
    values: Tuple[float, ...]
    guarded_state_sha256: str
    empirical_scaling_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "policy_feature_vector_v2"

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != POLICY_FEATURE_COUNT:
            raise ScalingError(
                f"values must be an exact {POLICY_FEATURE_COUNT}-tuple"
            )
        for index, value in enumerate(self.values):
            _finite_float(value, f"values[{index}]", ScalingError)
        _sha256_hex(
            self.guarded_state_sha256, "guarded_state_sha256", ScalingError
        )
        _sha256_hex(
            self.empirical_scaling_sha256,
            "empirical_scaling_sha256",
            ScalingError,
        )
        if self._attestation is not None and not _valid_features(
            self._attestation, self._binding()
        ):
            raise ScalingError("feature-vector attestation is invalid")

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return POLICY_FEATURE_ORDER

    def as_tuple(self) -> Tuple[float, ...]:
        self.require_attested()
        return self.values

    def as_dict(self) -> Dict[str, float]:
        self.require_attested()
        return dict(zip(POLICY_FEATURE_ORDER, self.values))

    def _payload(self) -> Dict[str, Any]:
        return {
            "empirical_scaling_sha256": self.empirical_scaling_sha256,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "guarded_state_sha256": self.guarded_state_sha256,
            "names": list(POLICY_FEATURE_ORDER),
            "values": list(self.values),
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_attested(self) -> bool:
        return _valid_features(self._attestation, self._binding())

    def require_attested(self) -> None:
        if not self.is_attested:
            raise ScalingError(
                "features must be produced by build_policy_features() after guard"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_attested()
        return _CanonicalRecord.to_canonical_dict(self)


def build_policy_features(
    guarded: GuardedPolicyStateV2,
    scaling: EmpiricalScalingV2,
) -> PolicyFeatureVectorV2:
    """Build the exact actor vector after the external freshness guard passes."""
    if not isinstance(guarded, GuardedPolicyStateV2):
        raise ScalingError("guarded must be GuardedPolicyStateV2")
    guarded.require_guarded()
    if not isinstance(scaling, EmpiricalScalingV2):
        raise ScalingError("scaling must be EmpiricalScalingV2")

    state = guarded.state
    named: Dict[str, float] = {
        "camera_si_scaled": (
            (float(state.camera_si.value) - float(scaling.camera_si_center))
            / float(scaling.camera_si_scale)
        ),
        "radar_p40": float(state.radar_p40.value),
        "prior_ul_mcs_normalized": (
            (int(state.prior_ul_mcs.observation.value) - UL_MCS_INDEX_MIN)
            / float(UL_MCS_INDEX_MAX - UL_MCS_INDEX_MIN)
        ),
        "pre_action_rlc_backlog_log1p_scaled": (
            math.log1p(int(state.pre_action_rlc_backlog.value))
            / float(scaling.backlog_log1p_scale)
        ),
    }
    for mode_id in range(EXPECTED_MODE_COUNT):
        named[f"prev_joint_mode_{mode_id}_one_hot"] = 0.0

    previous = state.previous
    if previous is None:
        named.update(
            {
                "prev_q_normalized": 0.0,
                "prev_quality_qperc": 0.0,
                "prev_latency_normalized": 0.0,
                "prev_present": 0.0,
                "prev_success": 0.0,
            }
        )
    else:
        named[f"prev_joint_mode_{previous.action.mode_id}_one_hot"] = 1.0
        named["prev_q_normalized"] = previous.action.q_e4 / float(Q_E4_MAX)
        named["prev_quality_qperc"] = (
            float(previous.q_perc) if previous.success else 0.0
        )
        named["prev_latency_normalized"] = (
            float(previous.latency_ms) / REWARD_DEADLINE_MS
            if previous.success
            else 0.0
        )
        named["prev_present"] = 1.0
        named["prev_success"] = 1.0 if previous.success else 0.0

    if set(named) != set(POLICY_FEATURE_ORDER):  # pragma: no cover - invariant
        raise ScalingError("internal feature allow-list mismatch")
    values = tuple(float(named[name]) for name in POLICY_FEATURE_ORDER)
    candidate = PolicyFeatureVectorV2(
        values=values,
        guarded_state_sha256=guarded.canonical_sha256(),
        empirical_scaling_sha256=scaling.canonical_sha256(),
    )
    return replace(candidate, _attestation=_issue_features(candidate._binding()))


# ---------------------------------------------------------------------------
# Action hold and semi-Markov transition
# ---------------------------------------------------------------------------


class PayloadEvidenceClass(str, Enum):
    """How one offered-payload value was obtained; never inferred from dtype."""

    MEASURED_EXACT_ACTION_NODE = "MEASURED_EXACT_ACTION_NODE"
    MODELED_SAME_SCENE_INTERPOLATION = "MODELED_SAME_SCENE_INTERPOLATION"


class EpisodeBoundary(str, Enum):
    """Bootstrap semantics for a decision-cycle transition."""

    CONTINUES = "CONTINUES"
    TERMINATED = "TERMINATED"
    TRUNCATED = "TRUNCATED"


@dataclass(frozen=True, slots=True)
class HoldTensorV1(_CanonicalRecord):
    tensor_seq: int
    offered_payload_bytes: float
    payload_evidence_class: PayloadEvidenceClass
    payload_provenance_sha256: str
    reward_requested: bool

    RECORD_TYPE = "hold_tensor_v1"

    def __post_init__(self) -> None:
        _non_negative_int(self.tensor_seq, "tensor_seq", TransitionError)
        payload = _finite_float(
            self.offered_payload_bytes, "offered_payload_bytes", TransitionError
        )
        if payload < 0.0:
            raise TransitionError("offered_payload_bytes must be >= 0")
        if not isinstance(self.payload_evidence_class, PayloadEvidenceClass):
            raise TransitionError(
                "payload_evidence_class must be PayloadEvidenceClass"
            )
        if (
            self.payload_evidence_class
            is PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            and type(self.offered_payload_bytes) is not int
        ):
            raise TransitionError(
                "MEASURED_EXACT_ACTION_NODE payload must be an exact int; "
                "modeled values must not be relabelled or rounded as measured"
            )
        _sha256_hex(
            self.payload_provenance_sha256,
            "payload_provenance_sha256",
            TransitionError,
        )
        _strict_bool(self.reward_requested, "reward_requested", TransitionError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "offered_payload_bytes": self.offered_payload_bytes,
            "payload_evidence_class": self.payload_evidence_class.value,
            "payload_provenance_sha256": self.payload_provenance_sha256,
            "reward_requested": self.reward_requested,
            "tensor_seq": self.tensor_seq,
        }


@dataclass(frozen=True, slots=True)
class ActionHoldV1(_CanonicalRecord):
    identity: DecisionIdentityV1
    action: ExecutedActionIdentity
    tensors: Tuple[HoldTensorV1, ...]

    RECORD_TYPE = "action_hold_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DecisionIdentityV1):
            raise TransitionError("identity must be DecisionIdentityV1")
        _action(self.action, "action", TransitionError)
        if type(self.tensors) is not tuple:
            raise TransitionError("tensors must be a tuple")
        if len(self.tensors) < MINIMUM_HOLD_TENSORS:
            raise TransitionError(
                f"a closed hold requires at least {MINIMUM_HOLD_TENSORS} "
                "transmitted tensors"
            )
        for tensor in self.tensors:
            if not isinstance(tensor, HoldTensorV1):
                raise TransitionError("every hold member must be HoldTensorV1")
        seqs = [tensor.tensor_seq for tensor in self.tensors]
        if len(seqs) != len(set(seqs)):
            raise TransitionError("tensor_seq values must be unique")
        ordered = tuple(sorted(self.tensors, key=lambda item: item.tensor_seq))
        if not ordered[0].reward_requested:
            raise TransitionError("the earliest tensor must request the reward")
        if any(tensor.reward_requested for tensor in ordered[1:]):
            raise TransitionError("held tensors must not request a reward")
        object.__setattr__(self, "tensors", ordered)

    @property
    def duration(self) -> int:
        return len(self.tensors)

    @property
    def total_offered_payload_bytes(self) -> float:
        return sum(float(tensor.offered_payload_bytes) for tensor in self.tensors)

    @property
    def held_offered_payload_bytes(self) -> float:
        return sum(
            float(tensor.offered_payload_bytes) for tensor in self.tensors[1:]
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_canonical_dict(),
            "duration": self.duration,
            "held_offered_payload_bytes": self.held_offered_payload_bytes,
            "identity": self.identity.to_canonical_dict(),
            "nominal_transmit_cadence_hz": TRANSMIT_CADENCE_HZ,
            "tensors": [tensor.to_canonical_dict() for tensor in self.tensors],
            "total_offered_payload_bytes": self.total_offered_payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class SemiMarkovTransitionV2(_CanonicalRecord):
    state: GuardedPolicyStateV2
    state_features: PolicyFeatureVectorV2
    action: ExecutedActionIdentity
    hold: ActionHoldV1
    reward_resolution: RewardResolutionV1
    next_state: Optional[GuardedPolicyStateV2]
    next_state_features: Optional[PolicyFeatureVectorV2]
    episode_boundary: EpisodeBoundary
    duration: int
    cycle_end_timestamp_ns: int
    elapsed_virtual_ns: int
    gamma: float
    discount: float
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "semi_markov_transition_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.state, GuardedPolicyStateV2):
            raise TransitionError("state must be GuardedPolicyStateV2")
        if not isinstance(self.state_features, PolicyFeatureVectorV2):
            raise TransitionError("state_features must be PolicyFeatureVectorV2")
        _action(self.action, "action", TransitionError)
        if not isinstance(self.hold, ActionHoldV1):
            raise TransitionError("hold must be ActionHoldV1")
        if not isinstance(self.reward_resolution, RewardResolutionV1):
            raise TransitionError("reward_resolution must be RewardResolutionV1")
        if not isinstance(self.episode_boundary, EpisodeBoundary):
            raise TransitionError("episode_boundary must be EpisodeBoundary")
        if self.episode_boundary is EpisodeBoundary.CONTINUES:
            if not isinstance(self.next_state, GuardedPolicyStateV2):
                raise TransitionError(
                    "CONTINUES requires next_state=GuardedPolicyStateV2"
                )
            if not isinstance(self.next_state_features, PolicyFeatureVectorV2):
                raise TransitionError(
                    "CONTINUES requires next_state_features=PolicyFeatureVectorV2"
                )
        elif self.next_state is not None or self.next_state_features is not None:
            raise TransitionError(
                "TERMINATED/TRUNCATED must omit next_state and its features"
            )
        duration = _positive_int(self.duration, "duration", TransitionError)
        if duration < MINIMUM_HOLD_TENSORS:
            raise TransitionError(
                f"duration must be >= {MINIMUM_HOLD_TENSORS} transmitted tensors"
            )
        _positive_int(
            self.cycle_end_timestamp_ns,
            "cycle_end_timestamp_ns",
            TransitionError,
        )
        _positive_int(
            self.elapsed_virtual_ns, "elapsed_virtual_ns", TransitionError
        )
        gamma = _finite_float(self.gamma, "gamma", TransitionError)
        discount = _finite_float(self.discount, "discount", TransitionError)
        if not 0.0 < gamma <= 1.0:
            raise TransitionError("gamma must lie in (0, 1]")
        if not 0.0 < discount <= 1.0:
            raise TransitionError("discount must lie in (0, 1]")
        if self._attestation is not None and not _valid_transition(
            self._attestation, self._binding()
        ):
            raise TransitionError("transition attestation is invalid")

    @property
    def reward(self) -> float:
        self.require_attested()
        return float(self.reward_resolution.reward)

    @property
    def terminated(self) -> bool:
        return self.episode_boundary is EpisodeBoundary.TERMINATED

    @property
    def truncated(self) -> bool:
        return self.episode_boundary is EpisodeBoundary.TRUNCATED

    @property
    def bootstrap_allowed(self) -> bool:
        return self.episode_boundary is EpisodeBoundary.CONTINUES

    @property
    def bootstrap_discount(self) -> float:
        """Semi-Markov discount masked to zero at either episode boundary."""
        return self.discount if self.bootstrap_allowed else 0.0

    def _payload(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_canonical_dict(),
            "bootstrap_allowed": self.bootstrap_allowed,
            "bootstrap_discount": self.bootstrap_discount,
            "cycle_end_timestamp_ns": self.cycle_end_timestamp_ns,
            "discount": self.discount,
            "duration": self.duration,
            "elapsed_virtual_ns": self.elapsed_virtual_ns,
            "episode_boundary": self.episode_boundary.value,
            "evidence_class": TRAINING_EVIDENCE_CLASS,
            "gamma": self.gamma,
            "hold": self.hold.to_canonical_dict(),
            "live_evidence_status": LIVE_EVIDENCE_STATUS,
            "next_state": (
                None
                if self.next_state is None
                else self.next_state.to_canonical_dict()
            ),
            "next_state_features": (
                None
                if self.next_state_features is None
                else self.next_state_features.to_canonical_dict()
            ),
            "reward_resolution": self.reward_resolution.to_canonical_dict(),
            "state": self.state.to_canonical_dict(),
            "state_features": self.state_features.to_canonical_dict(),
            "transition_schema_id": TRANSITION_SCHEMA_ID,
            "transition_schema_sha256": TRANSITION_SCHEMA_SHA256,
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_attested(self) -> bool:
        return _valid_transition(self._attestation, self._binding())

    def require_attested(self) -> None:
        if not self.is_attested:
            raise TransitionError("transition must be produced by build_transition()")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_attested()
        return _CanonicalRecord.to_canonical_dict(self)


def build_transition(
    *,
    state: GuardedPolicyStateV2,
    state_features: PolicyFeatureVectorV2,
    action: ExecutedActionIdentity,
    hold: ActionHoldV1,
    reward_resolution: RewardResolutionV1,
    next_state: Optional[GuardedPolicyStateV2],
    next_state_features: Optional[PolicyFeatureVectorV2],
    episode_boundary: EpisodeBoundary,
    duration: int,
    cycle_end_timestamp_ns: int,
    elapsed_virtual_ns: int,
    gamma: float,
    discount: float,
) -> SemiMarkovTransitionV2:
    """Validate and bind one real decision-to-feedback/timeout successor.

    ``discount`` is supplied by the caller and checked against
    ``gamma ** duration`` before storage.  The function intentionally makes no
    assertion that the two states' scene samples are adjacent frames.
    """
    if not isinstance(state, GuardedPolicyStateV2):
        raise TransitionError("state must be GuardedPolicyStateV2")
    state.require_guarded()
    if not isinstance(state_features, PolicyFeatureVectorV2):
        raise TransitionError("state_features must be PolicyFeatureVectorV2")
    state_features.require_attested()
    if not isinstance(episode_boundary, EpisodeBoundary):
        raise TransitionError("episode_boundary must be EpisodeBoundary")
    _action(action, "action", TransitionError)
    if not isinstance(hold, ActionHoldV1):
        raise TransitionError("hold must be ActionHoldV1")
    if not isinstance(reward_resolution, RewardResolutionV1):
        raise TransitionError("reward_resolution must be RewardResolutionV1")
    reward_resolution.require_attested()
    if not reward_resolution.learning_included:
        raise TransitionError(
            "infrastructure/evaluator faults are excluded and cannot create a "
            "learning transition"
        )

    identity = state.state.identity
    if hold.identity != identity or reward_resolution.identity != identity:
        raise TransitionError("state, hold and reward decision identities differ")
    if action != hold.action or action != reward_resolution.action:
        raise TransitionError("one exact executed action must govern hold and reward")
    if state.boundary.action_open_timestamp_ns != (
        reward_resolution.action_open_timestamp_ns
    ):
        raise TransitionError("reward latency did not start at this action open")
    if state.boundary.clock_domain != reward_resolution.clock_domain:
        raise TransitionError("state and reward use different clock domains")

    if state_features.guarded_state_sha256 != state.canonical_sha256():
        raise TransitionError("state_features are not bound to state")

    cycle_end = _positive_int(
        cycle_end_timestamp_ns, "cycle_end_timestamp_ns", TransitionError
    )
    opened = state.boundary.action_open_timestamp_ns
    if cycle_end <= opened:
        raise TransitionError("cycle end must be strictly after action open")
    if reward_resolution.resolution_timestamp_ns > cycle_end:
        raise TransitionError("cycle end cannot precede feedback/timeout closure")

    if episode_boundary is EpisodeBoundary.CONTINUES:
        if not isinstance(next_state, GuardedPolicyStateV2):
            raise TransitionError("CONTINUES requires a real next_state")
        next_state.require_guarded()
        if not isinstance(next_state_features, PolicyFeatureVectorV2):
            raise TransitionError("CONTINUES requires next_state_features")
        next_state_features.require_attested()
        if next_state.boundary.clock_domain != state.boundary.clock_domain:
            raise TransitionError("successive decisions use different clocks")
        if next_state.boundary.action_open_timestamp_ns != cycle_end:
            raise TransitionError(
                "cycle_end_timestamp_ns must equal successor action open"
            )
        next_identity = next_state.state.identity
        if (
            next_identity.session_uuid != identity.session_uuid
            or next_identity.ue_id != identity.ue_id
            or next_identity.decision_seq != identity.decision_seq + 1
        ):
            raise TransitionError(
                "next_state must be the next real decision in the same session/UE"
            )
        if reward_resolution.resolution_timestamp_ns > (
            next_state.boundary.state_commit_timestamp_ns
        ):
            raise TransitionError(
                "next state was committed before the current cycle closed"
            )
        expected_previous = PreviousOutcomeV1.from_resolution(reward_resolution)
        actual_previous = next_state.state.previous
        if actual_previous is None or actual_previous.canonical_sha256() != (
            expected_previous.canonical_sha256()
        ):
            raise TransitionError(
                "next_state.previous must be exactly the current reward outcome"
            )
        if next_state_features.guarded_state_sha256 != next_state.canonical_sha256():
            raise TransitionError("next_state_features are not bound to next_state")
    else:
        if next_state is not None or next_state_features is not None:
            raise TransitionError(
                "TERMINATED/TRUNCATED omit the successor and stop bootstrap"
            )

    exact_duration = _positive_int(duration, "duration", TransitionError)
    if exact_duration != hold.duration:
        raise TransitionError("duration must equal the exact transmitted-tensor count")
    if exact_duration < MINIMUM_HOLD_TENSORS:
        raise TransitionError(
            f"duration must be >= {MINIMUM_HOLD_TENSORS} transmitted tensors"
        )
    exact_elapsed = _positive_int(
        elapsed_virtual_ns, "elapsed_virtual_ns", TransitionError
    )
    if exact_elapsed != cycle_end - opened:
        raise TransitionError(
            "elapsed_virtual_ns must equal cycle_end minus action open"
        )
    minimum_cadence_span = (exact_duration - 1) * TRANSMIT_PERIOD_NS
    if exact_elapsed < minimum_cadence_span:
        raise TransitionError(
            "elapsed virtual time is incompatible with duration at the "
            f"{TRANSMIT_CADENCE_HZ}-Hz hold cadence"
        )
    gamma_value = _finite_float(gamma, "gamma", TransitionError)
    discount_value = _finite_float(discount, "discount", TransitionError)
    if not 0.0 < gamma_value <= 1.0:
        raise TransitionError("gamma must lie in (0, 1]")
    expected_discount = gamma_value ** exact_duration
    if not math.isclose(
        discount_value, expected_discount, rel_tol=1e-15, abs_tol=0.0
    ):
        raise TransitionError(
            "discount must be externally derived as gamma ** duration; "
            f"expected {expected_discount!r}, got {discount_value!r}"
        )

    candidate = SemiMarkovTransitionV2(
        state=state,
        state_features=state_features,
        action=action,
        hold=hold,
        reward_resolution=reward_resolution,
        next_state=next_state,
        next_state_features=next_state_features,
        episode_boundary=episode_boundary,
        duration=exact_duration,
        cycle_end_timestamp_ns=cycle_end,
        elapsed_virtual_ns=exact_elapsed,
        gamma=gamma_value,
        discount=discount_value,
    )
    return replace(
        candidate,
        _attestation=_issue_transition(candidate._binding()),
    )
