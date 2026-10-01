"""Run-5B 21-feature deployable state: Run-4B transport state plus the UL-SNR proxy.

Frozen design (2026-09-30)
--------------------------
* Positions 0-19 are the Run-4B order: the Run-4 order with the single
  ``prev_quality_qperc`` slot removed.  Q_perc is ground-truth reward evidence;
  a deployed UE never receives it, so it cannot be an actor feature.
* Position 20 (the 21st feature) is ``effective_external_ul_snr_proxy_scaled``
  with the frozen Run-5 v2 provider semantics: the active, ACKed, unclamped
  target at state commit with a matching live controller lease, scaled by
  ``(snr_db - 5.5) / 19.0`` on the closed registered support ``[5.5, 24.5]``.
* The previous outcome is a :class:`TransportPriorOutcomeV1`.  It is a genuine
  transport-only type: it has no quality field, a *successful* prior carries
  only the operational action-open-to-feedback latency, and a failure/timeout
  prior carries neither.  The Run-4 ``PreviousOutcomeV1`` (which requires
  Q_perc on success) is never used to build the actor state.
* The vector is built natively by :func:`build_run5b_policy_features`.  No
  Run-4 or Run-5 feature builder is called and no tensor is sliced.
* Reward and the 170-ms deadline are unchanged:
  ``r = Q_perc - 0.25 * (L_ms / 170.0)``, timeout/registered failure ``-1``.
  Q_perc is training reward evidence only.

Importing this module reads no files and starts no runtime component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_json_bytes,
    canonical_sha256,
)

ExternalFallbackRequired = R4.ExternalFallbackRequired
MetadataError = R4.MetadataError
ScalingError = R4.ScalingError


class Run5BContractError(ValueError):
    """A Run-5B schema or contract invariant is violated."""


# ---------------------------------------------------------------------------
# Frozen 21-feature schema
# ---------------------------------------------------------------------------

RUN4B_POLICY_FEATURE_ORDER: Tuple[str, ...] = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{index}_one_hot" for index in range(EXPECTED_MODE_COUNT)),
    "prev_q_normalized",
    "prev_latency_normalized",
    "prev_present",
    "prev_success",
)
RUN4B_POLICY_FEATURE_COUNT = 20
REMOVED_RUN4_FEATURE = "prev_quality_qperc"

SNR_FEATURE_NAME = R5V1.SNR_FEATURE_NAME
SNR_FEATURE_INDEX = 20
RUN5B_POLICY_FEATURE_ORDER: Tuple[str, ...] = (*RUN4B_POLICY_FEATURE_ORDER, SNR_FEATURE_NAME)
RUN5B_POLICY_FEATURE_COUNT = 21

PREV_MODE_SLICE = slice(4, 4 + EXPECTED_MODE_COUNT)
PREV_Q_INDEX = 16
PREV_LATENCY_INDEX = 17
PREV_PRESENT_INDEX = 18
PREV_SUCCESS_INDEX = 19
PREVIOUS_SLICE = slice(4, 20)

# Run-4's and Run-5's actor deny-lists apply unchanged, plus every spelling of
# perception quality or ground truth: none may name an actor feature.
RUN5B_EXTRA_FORBIDDEN_TERMS: Tuple[str, ...] = (
    "quality", "qperc", "q_perc", "perc", "reward", "ground_truth", "groundtruth",
)
RUN5B_FORBIDDEN_FEATURE_TERMS: Tuple[str, ...] = (
    *R5V1.RUN5_FORBIDDEN_FEATURE_TERMS, *RUN5B_EXTRA_FORBIDDEN_TERMS,
)


def assert_run5b_feature_schema() -> None:
    expected_run4b = tuple(n for n in R4.POLICY_FEATURE_ORDER if n != REMOVED_RUN4_FEATURE)
    if RUN4B_POLICY_FEATURE_ORDER != expected_run4b:
        raise Run5BContractError("Run-4B order is not Run-4 minus prev_quality_qperc")
    if len(RUN4B_POLICY_FEATURE_ORDER) != RUN4B_POLICY_FEATURE_COUNT:
        raise Run5BContractError("Run-4B feature count drift")
    if len(RUN5B_POLICY_FEATURE_ORDER) != RUN5B_POLICY_FEATURE_COUNT:
        raise Run5BContractError("Run-5B feature count drift")
    if len(set(RUN5B_POLICY_FEATURE_ORDER)) != RUN5B_POLICY_FEATURE_COUNT:
        raise Run5BContractError("Run-5B feature names must be unique")
    if RUN5B_POLICY_FEATURE_ORDER[SNR_FEATURE_INDEX] != SNR_FEATURE_NAME:
        raise Run5BContractError("SNR feature is not at position 20")
    for index, name in ((PREV_Q_INDEX, "prev_q_normalized"),
                        (PREV_LATENCY_INDEX, "prev_latency_normalized"),
                        (PREV_PRESENT_INDEX, "prev_present"),
                        (PREV_SUCCESS_INDEX, "prev_success")):
        if RUN5B_POLICY_FEATURE_ORDER[index] != name:
            raise Run5BContractError(f"{name} is not at position {index}")
    for name in RUN5B_POLICY_FEATURE_ORDER:
        lowered = name.lower()
        for forbidden in RUN5B_FORBIDDEN_FEATURE_TERMS:
            if forbidden in lowered:
                raise Run5BContractError(f"forbidden actor term {forbidden!r} in {name!r}")


assert_run5b_feature_schema()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


FEATURE_SCHEMA_ID = "splitfusion_run5b_policy_features_v1"
FEATURE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze({
    "schema_id": FEATURE_SCHEMA_ID,
    "version": FEATURE_SCHEMA_VERSION,
    "feature_order": RUN5B_POLICY_FEATURE_ORDER,
    "feature_count": RUN5B_POLICY_FEATURE_COUNT,
    "run4b_order": {
        "positions": [0, RUN4B_POLICY_FEATURE_COUNT - 1],
        "definition": "Run-4 order and numeric semantics with prev_quality_qperc removed",
        "removed_run4_feature": REMOVED_RUN4_FEATURE,
        "built": "natively by build_run5b_policy_features; no Run-4 builder, no slicing",
    },
    "previous_outcome": {
        "type": "TransportPriorOutcomeV1",
        "genesis": "all previous fields zero; prev_present=0",
        "success": "action and operational latency required; prev_present=prev_success=1",
        "failure_or_timeout": ("action required; latency absent and encoded zero only under "
                               "prev_present=1, prev_success=0"),
        "q_perc": "absent from the type; never an actor or operational-prior input",
    },
    "snr_feature": {
        "index": SNR_FEATURE_INDEX,
        "name": SNR_FEATURE_NAME,
        "label": R5V1.SNR_PROXY_LABEL,
        "provider_semantics": SNR.LEASE_SELECTION_RULE_ID,
        "run5_v2_feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256,
        "scaling": "(snr_db - 5.5) / 19.0; outside [5.5, 24.5] -> ExternalFallbackRequired",
        "clock_domain": SNR.LIVE_CLOCK_DOMAIN,
    },
    "reward": {
        "schema_sha256": R4.REWARD_SCHEMA_SHA256,
        "success": "Q_perc - 0.25 * (L_ms / 170.0)",
        "timeout_or_registered_failure": R4.REGISTERED_FAILURE_REWARD,
        "q_perc_role": "training reward evidence only",
    },
    "forbidden_feature_terms": RUN5B_FORBIDDEN_FEATURE_TERMS,
})
FEATURE_SCHEMA_SHA256 = canonical_sha256(FEATURE_SCHEMA_DESCRIPTOR)
FEATURE_ORDER_SHA256 = canonical_sha256(list(RUN5B_POLICY_FEATURE_ORDER))

for _foreign in (R4.FEATURE_SCHEMA_SHA256, R5V1.FEATURE_SCHEMA_SHA256, SNR.FEATURE_SCHEMA_SHA256):
    if FEATURE_SCHEMA_SHA256 == _foreign:  # pragma: no cover
        raise Run5BContractError("Run-5B feature schema digest collides with a prior schema")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _non_negative_int(value: Any, name: str, error: type[Exception]) -> int:
    if type(value) is not int or value < 0:
        raise error(f"{name} must be an exact int >= 0")
    return value


def _finite_float(value: Any, name: str, error: type[Exception]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a finite real scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{name} must be finite")
    return result


def _non_empty_str(value: Any, name: str, error: type[Exception]) -> str:
    if not isinstance(value, str) or value == "":
        raise error(f"{name} must be a non-empty str")
    return value


def _sha256_hex(value: Any, name: str, error: type[Exception]) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise error(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _record(record_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {"feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "record_type": record_type, **payload}


def _gate():
    sentinel = object()
    return (lambda binding: (sentinel, binding),
            lambda token, binding: type(token) is tuple and len(token) == 2
            and token[0] is sentinel and token[1] == binding)


_issue_guard, _valid_guard = _gate()
_issue_features, _valid_features = _gate()


# ---------------------------------------------------------------------------
# Transport-only prior outcome
# ---------------------------------------------------------------------------


class TransportTerminal(str, Enum):
    SUCCESS = "SUCCESS"
    REGISTERED_DELIVERY_FAILURE = "REGISTERED_DELIVERY_FAILURE"
    REGISTERED_SERVICE_FAILURE = "REGISTERED_SERVICE_FAILURE"
    TIMEOUT = "TIMEOUT"


class TransportPriorSource(str, Enum):
    # Modeled training: projection of the resolved cycle onto its transport fields.
    MODELED_TRANSPORT_PROJECTION = "MODELED_TRANSPORT_PROJECTION"
    # Live: the GT-free operational tail-output ACK received by the UE.
    OPERATIONAL_ACK = "OPERATIONAL_ACK"


@dataclass(frozen=True, slots=True)
class TransportPriorOutcomeV1:
    """The immediately preceding decision's operational outcome, without quality.

    ``operational_latency_ms`` is the action-open-to-feedback latency of a
    successful prior; it is ``None`` for a registered failure or timeout.
    There is deliberately no quality, reward or ground-truth field.
    ``evidence_sha256`` binds the transport-only evidence the prior came from.
    """

    identity: R4.DecisionIdentityV1
    action: ExecutedActionIdentity
    terminal: TransportTerminal
    operational_latency_ms: Optional[float]
    available_timestamp_ns: int
    clock_domain: str
    source: TransportPriorSource
    evidence_sha256: str

    RECORD_TYPE = "transport_prior_outcome_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, R4.DecisionIdentityV1):
            raise MetadataError("identity must be DecisionIdentityV1")
        if not isinstance(self.action, ExecutedActionIdentity):
            raise MetadataError("action must be an ExecutedActionIdentity")
        self.action.require_reconciled()
        if not isinstance(self.terminal, TransportTerminal):
            raise MetadataError("terminal must be TransportTerminal")
        if not isinstance(self.source, TransportPriorSource):
            raise MetadataError("source must be TransportPriorSource")
        _non_negative_int(self.available_timestamp_ns, "available_timestamp_ns", MetadataError)
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", MetadataError)
        if self.terminal is TransportTerminal.SUCCESS:
            if self.operational_latency_ms is None:
                raise MetadataError("a successful prior requires its operational latency")
            latency = _finite_float(self.operational_latency_ms, "operational_latency_ms",
                                    MetadataError)
            if not 0.0 <= latency <= R4.REWARD_DEADLINE_MS:
                raise MetadataError("successful prior latency must lie in [0, 170] ms")
        elif self.operational_latency_ms is not None:
            raise MetadataError("a failed/timed-out prior must have absent latency")

    @property
    def success(self) -> bool:
        return self.terminal is TransportTerminal.SUCCESS

    @classmethod
    def from_reward_resolution(cls, resolution: R4.RewardResolutionV1) -> "TransportPriorOutcomeV1":
        """Project a resolved modeled cycle onto its transport fields.

        Reads identity, action, terminal, timestamps, clock and latency only.
        ``resolution.q_perc`` is never read; two resolutions that differ only
        in Q_perc produce byte-identical priors.
        """
        if not isinstance(resolution, R4.RewardResolutionV1):
            raise MetadataError("resolution must be RewardResolutionV1")
        resolution.require_attested()
        if not resolution.learning_included:
            raise MetadataError("an excluded fault cannot become a previous outcome")
        terminal = TransportTerminal(resolution.terminal.value)
        latency = None
        if terminal is TransportTerminal.SUCCESS:
            latency = float(resolution.latency_ms)
            elapsed_ms = (resolution.resolution_timestamp_ns
                          - resolution.action_open_timestamp_ns) / 1_000_000.0
            if latency != elapsed_ms:
                raise MetadataError("operational latency differs from the resolved elapsed time")
        projection = {
            "action": resolution.action.to_canonical_dict(),
            "action_open_timestamp_ns": resolution.action_open_timestamp_ns,
            "clock_domain": resolution.clock_domain,
            "identity": resolution.identity.to_canonical_dict(),
            "operational_latency_ms": None if latency is None else latency.hex(),
            "resolution_timestamp_ns": resolution.resolution_timestamp_ns,
            "terminal": terminal.value,
        }
        return cls(identity=resolution.identity, action=resolution.action, terminal=terminal,
                   operational_latency_ms=latency,
                   available_timestamp_ns=resolution.resolution_timestamp_ns,
                   clock_domain=resolution.clock_domain,
                   source=TransportPriorSource.MODELED_TRANSPORT_PROJECTION,
                   evidence_sha256=canonical_sha256(
                       _record("transport_projection_evidence_v1", projection)))

    def to_canonical_dict(self) -> Dict[str, Any]:
        return _record(self.RECORD_TYPE, {
            "action": self.action.to_canonical_dict(),
            "available_timestamp_ns": self.available_timestamp_ns,
            "clock_domain": self.clock_domain,
            "evidence_sha256": self.evidence_sha256,
            "identity": self.identity.to_canonical_dict(),
            "operational_latency_ms": (None if self.operational_latency_ms is None
                                       else float(self.operational_latency_ms).hex()),
            "source": self.source.value,
            "terminal": self.terminal.value,
        })

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


# ---------------------------------------------------------------------------
# State, guard and native feature builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run5BPolicyStateV1:
    identity: R4.DecisionIdentityV1
    camera_si: R4.ScalarObservationV1
    radar_p40: R4.ScalarObservationV1
    prior_ul_mcs: R4.PriorUlGrantObservationV1
    pre_action_rlc_backlog: R4.ScalarObservationV1
    previous: Optional[TransportPriorOutcomeV1]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, R4.DecisionIdentityV1):
            raise MetadataError("identity must be DecisionIdentityV1")
        for name in ("camera_si", "radar_p40", "pre_action_rlc_backlog"):
            if not isinstance(getattr(self, name), R4.ScalarObservationV1):
                raise MetadataError(f"{name} must be ScalarObservationV1")
        if not isinstance(self.prior_ul_mcs, R4.PriorUlGrantObservationV1):
            raise MetadataError("prior_ul_mcs must be PriorUlGrantObservationV1")
        if (self.identity.decision_seq == 0) != (self.previous is None):
            raise MetadataError("only decision_seq 0 may omit the transport prior")
        if self.previous is not None:
            if type(self.previous) is not TransportPriorOutcomeV1:
                raise MetadataError("previous must be TransportPriorOutcomeV1; the Run-4 "
                                    "PreviousOutcomeV1 carries Q_perc and is refused")
            prior = self.previous.identity
            if (prior.session_uuid != self.identity.session_uuid
                    or prior.ue_id != self.identity.ue_id
                    or prior.decision_seq + 1 != self.identity.decision_seq):
                raise MetadataError("previous must be the immediately preceding decision")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return _record("run5b_policy_state_v1", {
            "camera_si": self.camera_si.to_canonical_dict(),
            "identity": self.identity.to_canonical_dict(),
            "pre_action_rlc_backlog": self.pre_action_rlc_backlog.to_canonical_dict(),
            "previous": None if self.previous is None else self.previous.to_canonical_dict(),
            "prior_ul_mcs": self.prior_ul_mcs.to_canonical_dict(),
            "radar_p40": self.radar_p40.to_canonical_dict(),
        })


_EXPECTED_SEMANTICS = MappingProxyType({
    "camera_si": (R4.MeasurementKind.CAMERA_SI, R4.Observer.SCENE_PIPELINE,
                  R4.LinkDirection.NOT_APPLICABLE),
    "radar_p40": (R4.MeasurementKind.RADAR_P40, R4.Observer.SCENE_PIPELINE,
                  R4.LinkDirection.NOT_APPLICABLE),
    "prior_ul_mcs": (R4.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX, R4.Observer.UE,
                     R4.LinkDirection.UPLINK),
    "pre_action_rlc_backlog": (R4.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                               R4.Observer.UE, R4.LinkDirection.UPLINK),
})


@dataclass(frozen=True, slots=True)
class GuardedRun5BStateV1:
    state: Run5BPolicyStateV1
    snr: SNR.UlSnrLeaseObservationV1
    boundary: R4.DecisionBoundaryV1
    freshness_policy_sha256: str
    lease_policy_sha256: str
    observation_ages_ns: Tuple[int, int, int, int]
    heartbeat_age_ns: int
    _attestation: Any = field(default=None, compare=False, repr=False)

    def _binding(self) -> str:
        return canonical_sha256(_record("guarded_run5b_state_v1", {
            "boundary": self.boundary.to_canonical_dict(),
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "heartbeat_age_ns": self.heartbeat_age_ns,
            "lease_policy_sha256": self.lease_policy_sha256,
            "observation_ages_ns": list(self.observation_ages_ns),
            "snr": self.snr.to_canonical_dict(),
            "state": self.state.to_canonical_dict()}))

    def require_guarded(self) -> None:
        if not _valid_guard(self._attestation, self._binding()):
            raise ExternalFallbackRequired("state was not admitted by guard_run5b_state()")

    def canonical_sha256(self) -> str:
        self.require_guarded()
        return self._binding()


def guard_run5b_state(
    state: Run5BPolicyStateV1,
    snr: Optional[SNR.UlSnrLeaseObservationV1],
    boundary: R4.DecisionBoundaryV1,
    freshness: R4.FreshnessPolicyV2,
    lease: SNR.SnrLeasePolicyV1,
) -> GuardedRun5BStateV1:
    """Admit a causal, valid, fresh Run-5B state or demand the external fallback.

    The measurement checks are the Run-4 guard's rules (validity, semantics,
    identity, clock, availability at commit, freshness, shared scene identity,
    value domains); the SNR checks are the frozen Run-5 v2 lease rule.
    """
    if type(state) is not Run5BPolicyStateV1:
        raise MetadataError("state must be Run5BPolicyStateV1")
    if not isinstance(boundary, R4.DecisionBoundaryV1):
        raise MetadataError("boundary must be DecisionBoundaryV1")
    if not isinstance(freshness, R4.FreshnessPolicyV2):
        raise MetadataError("freshness must be FreshnessPolicyV2")
    if not isinstance(lease, SNR.SnrLeasePolicyV1):
        raise MetadataError("lease must be SnrLeasePolicyV1")
    if state.identity != boundary.identity:
        raise ExternalFallbackRequired("state/boundary decision identity mismatch")
    ages = []
    for slot in ("camera_si", "radar_p40", "prior_ul_mcs", "pre_action_rlc_backlog"):
        value = getattr(state, slot)
        observation = value.observation if slot == "prior_ul_mcs" else value
        metadata = observation.metadata
        kind, observer, direction = _EXPECTED_SEMANTICS[slot]
        if not metadata.valid or observation.value is None:
            raise ExternalFallbackRequired(f"{slot} is missing/invalid; use external fallback")
        if (metadata.kind is not kind or metadata.observer is not observer
                or metadata.link_direction is not direction):
            raise ExternalFallbackRequired(f"{slot} semantic mismatch")
        if (metadata.identity.session_uuid != state.identity.session_uuid
                or metadata.identity.ue_id != state.identity.ue_id):
            raise ExternalFallbackRequired(f"{slot} sample identity does not match decision")
        if metadata.clock_domain != boundary.clock_domain:
            raise ExternalFallbackRequired(f"{slot} and decision boundary use different clocks")
        if metadata.available_timestamp_ns > boundary.state_commit_timestamp_ns:
            raise ExternalFallbackRequired(f"{slot} was not available when state was committed")
        age = boundary.action_open_timestamp_ns - metadata.source_timestamp_ns
        if age < 0:
            raise ExternalFallbackRequired(f"{slot} source timestamp is after action open")
        if age > freshness.max_age_ns(kind):
            raise ExternalFallbackRequired(f"{slot} is stale ({age} ns)")
        ages.append(age)
    if state.camera_si.metadata.identity != state.radar_p40.metadata.identity:
        raise ExternalFallbackRequired("camera_si and radar_p40 must share the scene identity")
    camera = float(state.camera_si.value)
    radar = float(state.radar_p40.value)
    mcs = state.prior_ul_mcs.observation.value
    backlog = state.pre_action_rlc_backlog.value
    if camera < 0.0:
        raise ExternalFallbackRequired("camera_si cannot be negative")
    if not 0.0 <= radar <= 1.0:
        raise ExternalFallbackRequired("radar_p40 must lie in [0, 1]")
    if type(mcs) is not int or not R4.UL_MCS_INDEX_MIN <= mcs <= R4.UL_MCS_INDEX_MAX:
        raise ExternalFallbackRequired("prior_ul_mcs must be an exact table-0 index in [0, 28]")
    if type(backlog) is not int or backlog < 0:
        raise ExternalFallbackRequired("pre_action_rlc_backlog must be an exact byte count")
    if state.previous is not None:
        if state.previous.clock_domain != boundary.clock_domain:
            raise ExternalFallbackRequired("transport prior and boundary use different clocks")
        if state.previous.available_timestamp_ns > boundary.state_commit_timestamp_ns:
            raise ExternalFallbackRequired("transport prior was unavailable at state commit")

    # Frozen Run-5 v2 SNR lease rule.
    if not isinstance(snr, SNR.UlSnrLeaseObservationV1):
        raise ExternalFallbackRequired("SNR observation is absent; use external fallback")
    if snr.kind is not R5V1.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB:
        raise ExternalFallbackRequired("SNR observation has a foreign kind")
    if not snr.valid or snr.value_db is None:
        raise ExternalFallbackRequired(f"SNR is missing/invalid ({snr.missing_reason})")
    if (snr.identity.session_uuid != boundary.identity.session_uuid
            or snr.identity.ue_id != boundary.identity.ue_id
            or snr.controller_session_uuid != boundary.identity.session_uuid):
        raise ExternalFallbackRequired("SNR identity/controller session differs from decision")
    if snr.clock_domain != boundary.clock_domain:
        raise ExternalFallbackRequired("SNR and decision boundary use different clocks")
    if snr.heartbeat_command_id != snr.active_command_id:
        raise ExternalFallbackRequired("controller lease does not name the active command")
    commit = boundary.state_commit_timestamp_ns
    if snr.effective_since_ns > commit or snr.heartbeat_ns > commit:
        raise ExternalFallbackRequired("SNR command or lease was not available at commit")
    heartbeat_age = boundary.action_open_timestamp_ns - snr.heartbeat_ns
    if heartbeat_age <= 0 or heartbeat_age > lease.max_heartbeat_age_ns:
        raise ExternalFallbackRequired(f"controller lease is stale ({heartbeat_age} ns)")
    SNR.scale_snr_db(float(snr.value_db))  # support check; raises, never clips

    candidate = GuardedRun5BStateV1(
        state=state, snr=snr, boundary=boundary,
        freshness_policy_sha256=freshness.canonical_sha256(),
        lease_policy_sha256=lease.canonical_sha256(),
        observation_ages_ns=tuple(ages), heartbeat_age_ns=heartbeat_age)
    return replace(candidate, _attestation=_issue_guard(candidate._binding()))


@dataclass(frozen=True, slots=True)
class Run5BPolicyFeatureVectorV1:
    values: Tuple[float, ...]
    guarded_state_sha256: str
    empirical_scaling_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != RUN5B_POLICY_FEATURE_COUNT:
            raise ScalingError(f"values must be an exact {RUN5B_POLICY_FEATURE_COUNT}-tuple")
        for index, value in enumerate(self.values):
            if type(value) is not float or not math.isfinite(value):
                raise ScalingError(f"values[{index}] must be a finite float")
        _sha256_hex(self.guarded_state_sha256, "guarded_state_sha256", ScalingError)
        _sha256_hex(self.empirical_scaling_sha256, "empirical_scaling_sha256", ScalingError)

    feature_names = property(lambda self: RUN5B_POLICY_FEATURE_ORDER)

    def _binding(self) -> str:
        return canonical_sha256(_record("run5b_policy_feature_vector_v1", {
            "empirical_scaling_sha256": self.empirical_scaling_sha256,
            "guarded_state_sha256": self.guarded_state_sha256,
            "names": list(RUN5B_POLICY_FEATURE_ORDER),
            "values": [v.hex() for v in self.values]}))

    def as_tuple(self) -> Tuple[float, ...]:
        if not _valid_features(self._attestation, self._binding()):
            raise ScalingError("features must come from build_run5b_policy_features()")
        return self.values

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(RUN5B_POLICY_FEATURE_ORDER, self.as_tuple()))

    def canonical_sha256(self) -> str:
        self.as_tuple()
        return self._binding()


def build_run5b_policy_features(
    guarded: GuardedRun5BStateV1, scaling: R4.EmpiricalScalingV2,
) -> Run5BPolicyFeatureVectorV1:
    """Build the exact 21-D actor vector natively from the guarded state."""
    if type(guarded) is not GuardedRun5BStateV1:
        raise ScalingError("guarded must be GuardedRun5BStateV1")
    guarded.require_guarded()
    if not isinstance(scaling, R4.EmpiricalScalingV2):
        raise ScalingError("scaling must be EmpiricalScalingV2")
    state = guarded.state
    named: Dict[str, float] = {
        "camera_si_scaled": ((float(state.camera_si.value) - float(scaling.camera_si_center))
                             / float(scaling.camera_si_scale)),
        "radar_p40": float(state.radar_p40.value),
        "prior_ul_mcs_normalized": ((int(state.prior_ul_mcs.observation.value)
                                     - R4.UL_MCS_INDEX_MIN)
                                    / float(R4.UL_MCS_INDEX_MAX - R4.UL_MCS_INDEX_MIN)),
        "pre_action_rlc_backlog_log1p_scaled": (math.log1p(int(state.pre_action_rlc_backlog.value))
                                                / float(scaling.backlog_log1p_scale)),
    }
    for mode_id in range(EXPECTED_MODE_COUNT):
        named[f"prev_joint_mode_{mode_id}_one_hot"] = 0.0
    previous = state.previous
    if previous is None:
        named.update({"prev_q_normalized": 0.0, "prev_latency_normalized": 0.0,
                      "prev_present": 0.0, "prev_success": 0.0})
    else:
        named[f"prev_joint_mode_{previous.action.mode_id}_one_hot"] = 1.0
        named["prev_q_normalized"] = previous.action.q_e4 / float(Q_E4_MAX)
        named["prev_latency_normalized"] = (
            float(previous.operational_latency_ms) / R4.REWARD_DEADLINE_MS
            if previous.success else 0.0)
        named["prev_present"] = 1.0
        named["prev_success"] = 1.0 if previous.success else 0.0
    named[SNR_FEATURE_NAME] = float(SNR.scale_snr_db(float(guarded.snr.value_db)))
    if set(named) != set(RUN5B_POLICY_FEATURE_ORDER):  # pragma: no cover - invariant
        raise ScalingError("internal Run-5B feature allow-list mismatch")
    candidate = Run5BPolicyFeatureVectorV1(
        values=tuple(float(named[name]) for name in RUN5B_POLICY_FEATURE_ORDER),
        guarded_state_sha256=guarded.canonical_sha256(),
        empirical_scaling_sha256=scaling.canonical_sha256())
    return replace(candidate, _attestation=_issue_features(candidate._binding()))
