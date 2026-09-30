"""Run-5 SNR contract v2: registered support, fixed scaling and a lease.

Supersedes only the *freshness/scaling* part of ``run5_state_contract`` (v1),
which is left byte-for-byte unchanged because the retained audit hashes it.
The 22-feature order, the Run-4 prefix rule and the actor deny-list are v1's.

Resolved contract (2026-09-29)
------------------------------
* Raw support is the registered network-profile design interval
  ``[5.5, 24.5]`` dB (``rl_agent/configs/network_profile_design_v2.json``).
* Feature 22 is ``(snr_db - 5.5) / 19.0``.  A value outside the closed
  support raises :class:`ExternalFallbackRequired`; nothing is clipped.
* Freshness is **not** command-ACK age.  A value is valid while
  (a) it is the *active effective command* (latest ACKed, unclamped, target
  bearing command at the state-commit cutoff) and (b) the controller has
  renewed a heartbeat/lease naming that same command, no older than the
  bound lease age at action-open, within the same controller session.
  A held command therefore stays valid while the controller is live.
  Both ``effective_since_ns`` (the command's ACK) and ``heartbeat_ns`` are
  preserved.
* The live adapter stamps every event with ``CLOCK_MONOTONIC_RAW`` directly.
  It has no offset, bridge or conversion API; a completed-run clock bridge
  cannot be supplied.
* Profile ID, Markov state, trace index, future SNR, the RFsim noise command
  and gNB PUSCH SNR have no field anywhere in this module.

Importing this module reads no files and starts nothing.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    canonical_json_bytes,
    canonical_sha256,
)

from . import run5_state_contract as V1

ExternalFallbackRequired = R4.ExternalFallbackRequired
MetadataError = R4.MetadataError
ScalingError = R4.ScalingError

SNR_SUPPORT_MIN_DB = 5.5
SNR_SUPPORT_MAX_DB = 24.5
SNR_SPAN_DB = 19.0
NETWORK_PROFILE_DESIGN_RELPATH = "rl_agent/configs/network_profile_design_v2.json"
NETWORK_PROFILE_DESIGN_SHA256 = (
    "056247e5731c1ae9ac281432034e1b79d1e6da24ab4a2d579a7fe2d85917e483"
)
LIVE_CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
LEASE_SELECTION_RULE_ID = (
    "ACTIVE_EFFECTIVE_COMMAND_AT_STATE_COMMIT_WITH_MATCHING_CONTROLLER_LEASE"
)

FEATURE_SCHEMA_ID = "splitfusion_run5_policy_features_v2"
FEATURE_SCHEMA_VERSION = 2
FEATURE_SCHEMA_DESCRIPTOR = {
    "schema_id": FEATURE_SCHEMA_ID,
    "version": FEATURE_SCHEMA_VERSION,
    "feature_order": list(V1.RUN5_POLICY_FEATURE_ORDER),
    "feature_count": V1.RUN5_POLICY_FEATURE_COUNT,
    "run4_prefix_feature_schema_sha256": R4.FEATURE_SCHEMA_SHA256,
    "v1_feature_schema_sha256": V1.FEATURE_SCHEMA_SHA256,
    "snr_label": V1.SNR_PROXY_LABEL,
    "snr_support_db": [SNR_SUPPORT_MIN_DB, SNR_SUPPORT_MAX_DB],
    "snr_scaling": "(snr_db - 5.5) / 19.0; outside support -> ExternalFallbackRequired",
    "support_source": {"path": NETWORK_PROFILE_DESIGN_RELPATH,
                       "sha256": NETWORK_PROFILE_DESIGN_SHA256},
    "freshness": LEASE_SELECTION_RULE_ID,
    "clock_domain": LIVE_CLOCK_DOMAIN,
    "state_only": True,
    "forbidden_feature_terms": list(V1.RUN5_FORBIDDEN_FEATURE_TERMS),
}
FEATURE_SCHEMA_SHA256 = canonical_sha256(FEATURE_SCHEMA_DESCRIPTOR)


def scale_snr_db(value_db: float) -> float:
    """Registered scaling; refuses (never clips) out-of-support values."""
    value = float(value_db)
    if not math.isfinite(value) or not SNR_SUPPORT_MIN_DB <= value <= SNR_SUPPORT_MAX_DB:
        raise ExternalFallbackRequired(
            f"SNR {value_db!r} dB is outside the registered [5.5, 24.5] support")
    return (value - SNR_SUPPORT_MIN_DB) / SNR_SPAN_DB


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise MetadataError(f"{name} must be an exact int >= 0")
    return value


def _non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise MetadataError(f"{name} must be a non-empty str")
    return value


def _record(record_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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


@dataclass(frozen=True, slots=True)
class SnrLeasePolicyV1:
    """Maximum controller-heartbeat age at action-open.  No default."""

    policy_id: str
    policy_version: int
    evidence_sha256: str
    max_heartbeat_age_ns: int

    def __post_init__(self) -> None:
        _non_empty_str(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise MetadataError("policy_version must be an exact int >= 1")
        V1._sha256_hex(self.evidence_sha256, "evidence_sha256", MetadataError)
        if type(self.max_heartbeat_age_ns) is not int or self.max_heartbeat_age_ns <= 0:
            raise MetadataError("max_heartbeat_age_ns must be an exact int > 0")

    def canonical_sha256(self) -> str:
        return canonical_sha256(_record("snr_lease_policy_v1", {
            "evidence_sha256": self.evidence_sha256,
            "max_heartbeat_age_ns": self.max_heartbeat_age_ns,
            "policy_id": self.policy_id, "policy_version": self.policy_version}))


@dataclass(frozen=True, slots=True)
class UlSnrLeaseObservationV1:
    """Source-independent SNR observation with command identity and lease."""

    identity: R4.SampleIdentityV1
    kind: V1.SnrProxyKind
    provider_id: str
    controller_session_uuid: str
    value_db: Optional[float]
    active_command_id: Optional[str]
    effective_since_ns: Optional[int]
    heartbeat_command_id: Optional[str]
    heartbeat_ns: Optional[int]
    clock_domain: str
    valid: bool
    missing_reason: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, R4.SampleIdentityV1):
            raise MetadataError("identity must be SampleIdentityV1")
        if not isinstance(self.kind, V1.SnrProxyKind):
            raise MetadataError("kind must be SnrProxyKind")
        _non_empty_str(self.provider_id, "provider_id")
        V1._canonical_uuid(self.controller_session_uuid, "controller_session_uuid")
        _non_empty_str(self.clock_domain, "clock_domain")
        if type(self.valid) is not bool:
            raise MetadataError("valid must be a bool")
        for name in ("effective_since_ns", "heartbeat_ns"):
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        if self.valid:
            if self.value_db is None or isinstance(self.value_db, bool) or not isinstance(
                    self.value_db, (int, float)) or not math.isfinite(float(self.value_db)):
                raise MetadataError("a valid SNR lease observation requires a finite value")
            for name in ("active_command_id", "heartbeat_command_id"):
                _non_empty_str(getattr(self, name), name)
            if self.effective_since_ns is None or self.heartbeat_ns is None:
                raise MetadataError("a valid observation requires effective_since and heartbeat")
            if self.effective_since_ns > self.heartbeat_ns:
                raise MetadataError("a lease heartbeat cannot precede its command's ACK")
            if self.missing_reason is not None:
                raise MetadataError("a valid observation cannot carry missing_reason")
        else:
            if self.value_db is not None:
                raise MetadataError("invalid SNR must use value_db=None; zero-fill is forbidden")
            _non_empty_str(self.missing_reason, "missing_reason")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return _record("ul_snr_lease_observation_v1", {
            "active_command_id": self.active_command_id,
            "clock_domain": self.clock_domain,
            "controller_session_uuid": self.controller_session_uuid,
            "effective_since_ns": self.effective_since_ns,
            "heartbeat_command_id": self.heartbeat_command_id,
            "heartbeat_ns": self.heartbeat_ns,
            "identity": self.identity.to_canonical_dict(),
            "kind": self.kind.value, "missing_reason": self.missing_reason,
            "provider_id": self.provider_id, "valid": self.valid,
            "value_db": self.value_db})

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class GuardedRun5StateV2:
    run4_guarded: R4.GuardedPolicyStateV2
    snr: UlSnrLeaseObservationV1
    lease_policy_sha256: str
    heartbeat_age_ns: int
    _attestation: Any = field(default=None, compare=False, repr=False)

    def _binding(self) -> str:
        return canonical_sha256(_record("guarded_run5_state_v2", {
            "heartbeat_age_ns": self.heartbeat_age_ns,
            "lease_policy_sha256": self.lease_policy_sha256,
            "run4_guarded": self.run4_guarded.to_canonical_dict(),
            "snr": self.snr.to_canonical_dict()}))

    def require_guarded(self) -> None:
        if not _valid_guard(self._attestation, self._binding()):
            raise ExternalFallbackRequired("state was not admitted by guard_run5_state_v2()")

    def canonical_sha256(self) -> str:
        self.require_guarded()
        return self._binding()


def guard_run5_state_v2(
    state: R4.PolicyStateV2,
    snr: Optional[UlSnrLeaseObservationV1],
    boundary: R4.DecisionBoundaryV1,
    run4_freshness: R4.FreshnessPolicyV2,
    lease: SnrLeasePolicyV1,
) -> GuardedRun5StateV2:
    """Unchanged Run-4 guard, then the lease rule; any failure falls back."""
    if not isinstance(lease, SnrLeasePolicyV1):
        raise MetadataError("lease must be SnrLeasePolicyV1")
    run4_guarded = R4.guard_state_for_action(state, boundary, run4_freshness)
    if not isinstance(snr, UlSnrLeaseObservationV1):
        raise ExternalFallbackRequired("SNR observation is absent; use external fallback")
    if snr.kind is not V1.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB:
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
    age = boundary.action_open_timestamp_ns - snr.heartbeat_ns
    if age <= 0 or age > lease.max_heartbeat_age_ns:
        raise ExternalFallbackRequired(f"controller lease is stale ({age} ns)")
    scale_snr_db(float(snr.value_db))  # support check; raises, never clips
    candidate = GuardedRun5StateV2(run4_guarded, snr, lease.canonical_sha256(), age)
    return replace(candidate, _attestation=_issue_guard(candidate._binding()))


@dataclass(frozen=True, slots=True)
class Run5FeatureVectorV2:
    values: Tuple[float, ...]
    run4_prefix_sha256: str
    guarded_state_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != V1.RUN5_POLICY_FEATURE_COUNT:
            raise ScalingError("values must be an exact 22-tuple")
        for index, value in enumerate(self.values):
            if type(value) is not float or not math.isfinite(value):
                raise ScalingError(f"values[{index}] must be a finite float")

    def _binding(self) -> str:
        return canonical_sha256(_record("run5_feature_vector_v2", {
            "guarded_state_sha256": self.guarded_state_sha256,
            "run4_prefix_sha256": self.run4_prefix_sha256,
            "values": [v.hex() for v in self.values]}))

    def as_tuple(self) -> Tuple[float, ...]:
        if not _valid_features(self._attestation, self._binding()):
            raise ScalingError("features must come from build_run5_features_v2()")
        return self.values

    def canonical_sha256(self) -> str:
        self.as_tuple()
        return self._binding()


def build_run5_features_v2(
    guarded: GuardedRun5StateV2, run4_scaling: R4.EmpiricalScalingV2,
) -> Run5FeatureVectorV2:
    if not isinstance(guarded, GuardedRun5StateV2):
        raise ScalingError("guarded must be GuardedRun5StateV2")
    guarded.require_guarded()
    prefix = R4.build_policy_features(guarded.run4_guarded, run4_scaling)
    values = (*prefix.as_tuple(), float(scale_snr_db(float(guarded.snr.value_db))))
    candidate = Run5FeatureVectorV2(tuple(values), prefix.canonical_sha256(),
                                    guarded.canonical_sha256())
    return replace(candidate, _attestation=_issue_features(candidate._binding()))


# ---------------------------------------------------------------------------
# Live adapter (CLOCK_MONOTONIC_RAW, no bridge)
# ---------------------------------------------------------------------------


def _raw_now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


@dataclass(frozen=True, slots=True)
class _Ack:
    command_id: str
    ack_ns: int
    target_snr_db: Optional[float]
    usable: bool
    reason: Optional[str]


class RfsimLeaseSnrAdapterV1:
    """In-process adapter the RFsim controller calls on ACK and heartbeat.

    Every timestamp is taken here from ``CLOCK_MONOTONIC_RAW`` at the call.
    The controller must renew the lease (``record_heartbeat``) every tick,
    naming its active command, including HOLD ticks that send no command.
    """

    CLOCK_DOMAIN = LIVE_CLOCK_DOMAIN
    _clock: Callable[[], int] = staticmethod(_raw_now_ns)
    kind = V1.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB

    def __init__(self, *, provider_id: str, session_uuid: str, ue_id: str,
                 capacity: int = 4096) -> None:
        self.provider_id = _non_empty_str(provider_id, "provider_id")
        R4.DecisionIdentityV1(session_uuid, ue_id, 0)
        self.session_uuid = session_uuid
        self.ue_id = ue_id
        self._acks: collections.deque[_Ack] = collections.deque(maxlen=capacity)
        self._heartbeats: collections.deque[tuple[int, str]] = collections.deque(maxlen=capacity)
        self._last_ns = -1
        self._command_ids: set[str] = set()
        self._sample_seq = 0

    def _stamp(self) -> int:
        now = int(self._clock())
        if now < self._last_ns:
            raise MetadataError("CLOCK_MONOTONIC_RAW went backwards")
        self._last_ns = now
        return now

    def record_command_ack(self, *, command_id: str, status: str, clamped: Optional[bool],
                           target_snr_db: Optional[float]) -> int:
        _non_empty_str(command_id, "command_id")
        if command_id in self._command_ids:
            raise MetadataError("duplicate RFsim command_id")
        self._command_ids.add(command_id)
        now = self._stamp()
        reason = None
        if status != "ACK":
            reason = "ACTIVE_COMMAND_ERRORED"
        elif target_snr_db is None:
            reason = "ACTIVE_COMMAND_HAS_NO_TARGET"
        elif clamped is not False:
            reason = "ACTIVE_COMMAND_CLAMPED"
        self._acks.append(_Ack(command_id, now, None if reason else float(target_snr_db),
                               reason is None, reason))
        return now

    def record_heartbeat(self, *, active_command_id: str) -> int:
        _non_empty_str(active_command_id, "active_command_id")
        now = self._stamp()
        self._heartbeats.append((now, active_command_id))
        return now

    def observe(self, boundary: R4.DecisionBoundaryV1) -> UlSnrLeaseObservationV1:
        if not isinstance(boundary, R4.DecisionBoundaryV1):
            raise MetadataError("boundary must be DecisionBoundaryV1")
        cutoff = boundary.state_commit_timestamp_ns
        ack = next((a for a in reversed(self._acks) if a.ack_ns <= cutoff), None)
        beat = next((h for h in reversed(self._heartbeats) if h[0] <= cutoff), None)

        def missing(reason: str) -> UlSnrLeaseObservationV1:
            return self._observation(None, ack, beat, reason)

        if (boundary.identity.session_uuid != self.session_uuid
                or boundary.identity.ue_id != self.ue_id):
            return missing("DECISION_IDENTITY_FOREIGN_TO_ADAPTER")
        if boundary.clock_domain != self.CLOCK_DOMAIN:
            return missing("DECISION_CLOCK_IS_NOT_CLOCK_MONOTONIC_RAW")
        if ack is None:
            return missing("NO_EFFECTIVE_COMMAND_BEFORE_CUTOFF")
        if not ack.usable:
            return missing(ack.reason)
        if beat is None:
            return missing("NO_CONTROLLER_LEASE_BEFORE_CUTOFF")
        if beat[1] != ack.command_id:
            return missing("CONTROLLER_LEASE_NAMES_ANOTHER_COMMAND")
        if beat[0] < ack.ack_ns:
            return missing("CONTROLLER_LEASE_PRECEDES_ACTIVE_COMMAND")
        return self._observation(ack.target_snr_db, ack, beat, None)

    def _observation(self, value, ack, beat, reason) -> UlSnrLeaseObservationV1:
        identity = R4.SampleIdentityV1(self.session_uuid, self.ue_id, self._sample_seq)
        self._sample_seq += 1
        return UlSnrLeaseObservationV1(
            identity=identity, kind=self.kind, provider_id=self.provider_id,
            controller_session_uuid=self.session_uuid, value_db=value,
            active_command_id=None if ack is None else ack.command_id,
            effective_since_ns=None if ack is None else ack.ack_ns,
            heartbeat_command_id=None if beat is None else beat[1],
            heartbeat_ns=None if beat is None else beat[0],
            clock_domain=self.CLOCK_DOMAIN, valid=value is not None, missing_reason=reason)
