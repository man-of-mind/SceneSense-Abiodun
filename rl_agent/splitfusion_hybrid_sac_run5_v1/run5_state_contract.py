"""Run-5 22-feature state contract: Run-4 state plus one external UL-SNR proxy.

Frozen design (2026-09-29)
--------------------------
* Positions 0-20 are the Run-4 vector, produced by the **unchanged** Run-4
  :func:`run4_contract.guard_state_for_action` and
  :func:`run4_contract.build_policy_features`.  They are copied, never
  recomputed, so they are bit-identical to Run 4 by construction.
* Position 21 (the 22nd feature) is ``effective_external_ul_snr_proxy_scaled``.
* The SNR proxy is **state only**.  Reward, action space, the 170-ms deadline,
  transition semantics, previous-outcome fields and the Hybrid-SAC
  configuration are Run 4's and are not re-declared here.
* The proxy is labelled ``SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB``.  It is the
  simulator's commanded effective uplink SNR, not a UE-measured SNR.  gNB PUSCH
  SNR is verifier-only and has no path into this module.
* Missing, invalid, stale, foreign or out-of-support SNR raises
  :class:`run4_contract.ExternalFallbackRequired`.  The actor is never called
  and no numeric zero is ever substituted.

Provider interface
------------------
:class:`UlSnrProxyProviderV1` is source-independent: every provider returns one
:class:`UlSnrProxyObservationV1` carrying raw dB, source timestamp,
availability timestamp, clock domain, validity and a missing reason.  The
guard reads only that record, so a later non-RFsim source can be bound without
touching the guard or the feature builder.

:class:`RfsimEffectiveSnrProviderV1` is the only concrete provider.  It
accepts command records that carry no profile ID, trace ID/index, Markov
state, future target or noise command, and selects the latest command whose
ACK was available at or before the state-commit instant (which is strictly
before action-open).  If that latest effective command is clamped, errored or
carries no target (for example a clean-channel restore), the observation is
invalid: an older value is never resurrected, because the channel has
already moved away from it.  A newer command that was sent but not yet ACKed
at the cutoff is ignored exactly as the frozen rule requires; the fact is
recorded as guard-only metadata so audits can count it.

Scaling, support and freshness are constructor-bound empirical inputs with no
production defaults, exactly like the Run-4 camera/backlog scaling.

Importing this module reads no files and starts no runtime component.
"""

from __future__ import annotations

import bisect
import math
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    canonical_json_bytes,
    canonical_sha256,
)

ExternalFallbackRequired = R4.ExternalFallbackRequired
MetadataError = R4.MetadataError
ScalingError = R4.ScalingError


class Run5ContractError(ValueError):
    """Base class for Run-5-only contract violations."""


# ---------------------------------------------------------------------------
# Frozen 22-feature schema
# ---------------------------------------------------------------------------

SNR_FEATURE_NAME = "effective_external_ul_snr_proxy_scaled"
SNR_FEATURE_INDEX = 21  # zero-based; "feature 22" in one-based numbering

RUN4_PREFIX_ORDER: Tuple[str, ...] = tuple(R4.POLICY_FEATURE_ORDER)
RUN4_PREFIX_COUNT = R4.POLICY_FEATURE_COUNT
RUN5_POLICY_FEATURE_ORDER: Tuple[str, ...] = (*RUN4_PREFIX_ORDER, SNR_FEATURE_NAME)
RUN5_POLICY_FEATURE_COUNT = 22

SNR_PROXY_LABEL = "SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB"
RFSIM_SELECTION_RULE_ID = (
    "LATEST_ACKED_UNCLAMPED_TARGET_SNR_EFFECTIVE_AT_OR_BEFORE_STATE_COMMIT"
)

# Run-4's deny-list applies unchanged; these are the additional Run-5 leakage
# channels named in the frozen design.
RUN5_EXTRA_FORBIDDEN_TERMS: Tuple[str, ...] = (
    "profile",
    "trace",
    "markov",
    "hidden",
    "future",
    "target",
    "noise",
    "command",
    "measured",
)
RUN5_FORBIDDEN_FEATURE_TERMS: Tuple[str, ...] = (
    *R4.FORBIDDEN_POLICY_FEATURE_TERMS,
    *RUN5_EXTRA_FORBIDDEN_TERMS,
)

# Fields a raw RFsim command log may carry that must never reach a provider
# record.  ``RfsimSnrCommandRecordV1.from_log_entry`` drops them by allow-list.
RFSIM_LOG_FIELDS_NEVER_EXPOSED: Tuple[str, ...] = (
    "profile_id",
    "step_index",
    "trace_id",
    "trace_index",
    "markov_state",
    "hidden_state",
    "commanded_noise_power_db",
    "reason",
)


def assert_run5_feature_schema() -> None:
    if len(RUN5_POLICY_FEATURE_ORDER) != RUN5_POLICY_FEATURE_COUNT:
        raise Run5ContractError("Run-5 feature count drift")
    if RUN5_POLICY_FEATURE_ORDER[:RUN4_PREFIX_COUNT] != RUN4_PREFIX_ORDER:
        raise Run5ContractError("Run-5 positions 0-20 differ from Run 4")
    if RUN5_POLICY_FEATURE_ORDER[SNR_FEATURE_INDEX] != SNR_FEATURE_NAME:
        raise Run5ContractError("SNR feature is not at position 21")
    if len(set(RUN5_POLICY_FEATURE_ORDER)) != RUN5_POLICY_FEATURE_COUNT:
        raise Run5ContractError("Run-5 feature names must be unique")
    for name in RUN5_POLICY_FEATURE_ORDER:
        lowered = name.lower()
        for forbidden in RUN5_FORBIDDEN_FEATURE_TERMS:
            if forbidden in lowered:
                raise Run5ContractError(
                    f"forbidden actor leakage term {forbidden!r} in {name!r}"
                )


assert_run5_feature_schema()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


FEATURE_SCHEMA_ID = "splitfusion_run5_policy_features_v1"
FEATURE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": FEATURE_SCHEMA_ID,
        "version": FEATURE_SCHEMA_VERSION,
        "feature_order": RUN5_POLICY_FEATURE_ORDER,
        "feature_count": RUN5_POLICY_FEATURE_COUNT,
        "run4_prefix": {
            "positions": [0, RUN4_PREFIX_COUNT - 1],
            "feature_schema_id": R4.FEATURE_SCHEMA_ID,
            "feature_schema_sha256": R4.FEATURE_SCHEMA_SHA256,
            "contract_schema_sha256": R4.SCHEMA_SHA256,
            "rule": "copied bit-for-bit from run4_contract.build_policy_features",
        },
        "snr_feature": {
            "index": SNR_FEATURE_INDEX,
            "name": SNR_FEATURE_NAME,
            "label": SNR_PROXY_LABEL,
            "not": "UE-measured SNR; gNB PUSCH SNR is verifier-only",
            "scaling": "(snr_db - center_db) / scale_db, constructor-bound",
            "support": "closed [support_min_db, support_max_db]; outside -> fallback",
            "missing": "ExternalFallbackRequired; never zero-filled",
        },
        "state_only": (
            "reward, action space, 170-ms deadline, transition semantics, "
            "previous-outcome fields and Hybrid-SAC configuration are Run 4's"
        ),
        "rfsim_selection_rule": RFSIM_SELECTION_RULE_ID,
        "forbidden_feature_terms": RUN5_FORBIDDEN_FEATURE_TERMS,
        "rfsim_fields_never_exposed": RFSIM_LOG_FIELDS_NEVER_EXPOSED,
    }
)
FEATURE_SCHEMA_SHA256 = canonical_sha256(FEATURE_SCHEMA_DESCRIPTOR)

if FEATURE_SCHEMA_SHA256 == R4.FEATURE_SCHEMA_SHA256:  # pragma: no cover
    raise Run5ContractError("Run-5 feature schema digest collides with Run 4")


# ---------------------------------------------------------------------------
# Small validation helpers (Run-5 local; Run-4 helpers are private)
# ---------------------------------------------------------------------------


def _exact_int(value: Any, name: str, error: type[Exception]) -> int:
    if type(value) is not int:
        raise error(f"{name} must be an exact int, got {type(value).__name__}")
    return value


def _non_negative_int(value: Any, name: str, error: type[Exception]) -> int:
    value = _exact_int(value, name, error)
    if value < 0:
        raise error(f"{name} must be >= 0, got {value}")
    return value


def _positive_int(value: Any, name: str, error: type[Exception]) -> int:
    value = _exact_int(value, name, error)
    if value <= 0:
        raise error(f"{name} must be > 0, got {value}")
    return value


def _finite_float(value: Any, name: str, error: type[Exception]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a finite real scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{name} must be finite, got {value!r}")
    return result


def _non_empty_str(value: Any, name: str, error: type[Exception]) -> str:
    if not isinstance(value, str) or value == "":
        raise error(f"{name} must be a non-empty str, got {value!r}")
    return value


def _sha256_hex(value: Any, name: str, error: type[Exception]) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise error(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _canonical_uuid(value: Any, name: str) -> str:
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or str(parsed) != value:
        raise MetadataError(f"{name} must be a canonical lowercase UUID")
    return value


def _record(record_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "record_type": record_type,
    }
    result.update(payload)
    return result


class _CanonicalRecord:
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


_issue_guard, _valid_guard = _make_attestation_gate()
_issue_features, _valid_features = _make_attestation_gate()


# ---------------------------------------------------------------------------
# Source-independent SNR observation and provider interface
# ---------------------------------------------------------------------------


class SnrProxyKind(str, Enum):
    SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB = SNR_PROXY_LABEL


@dataclass(frozen=True, slots=True)
class UlSnrProxyObservationV1(_CanonicalRecord):
    """Raw SNR dB plus mandatory causal metadata, including invalidity.

    A missing/invalid observation carries ``value_db=None`` and a reason.  Its
    timestamps may be ``None`` when no effective command exists at all.
    ``newer_command_in_flight`` is guard/audit metadata only.
    """

    identity: R4.SampleIdentityV1
    kind: SnrProxyKind
    provider_id: str
    selection_rule_id: str
    value_db: Optional[float]
    source_timestamp_ns: Optional[int]
    available_timestamp_ns: Optional[int]
    clock_domain: str
    valid: bool
    missing_reason: Optional[str]
    newer_command_in_flight: bool = False

    RECORD_TYPE = "ul_snr_proxy_observation_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, R4.SampleIdentityV1):
            raise MetadataError("identity must be SampleIdentityV1")
        if not isinstance(self.kind, SnrProxyKind):
            raise MetadataError("kind must be SnrProxyKind")
        _non_empty_str(self.provider_id, "provider_id", MetadataError)
        _non_empty_str(self.selection_rule_id, "selection_rule_id", MetadataError)
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)
        if type(self.valid) is not bool:
            raise MetadataError("valid must be a bool")
        if type(self.newer_command_in_flight) is not bool:
            raise MetadataError("newer_command_in_flight must be a bool")
        for name in ("source_timestamp_ns", "available_timestamp_ns"):
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name, MetadataError)
        if (
            self.source_timestamp_ns is not None
            and self.available_timestamp_ns is not None
            and self.source_timestamp_ns > self.available_timestamp_ns
        ):
            raise MetadataError("source_timestamp_ns must be <= available_timestamp_ns")
        if self.valid:
            if self.value_db is None:
                raise MetadataError("a valid SNR observation requires value_db")
            _finite_float(self.value_db, "value_db", MetadataError)
            if self.source_timestamp_ns is None or self.available_timestamp_ns is None:
                raise MetadataError("a valid SNR observation requires timestamps")
            if self.missing_reason is not None:
                raise MetadataError("a valid SNR observation cannot carry missing_reason")
        else:
            if self.value_db is not None:
                raise MetadataError(
                    "an invalid/missing SNR must use value_db=None; zero-fill is forbidden"
                )
            _non_empty_str(self.missing_reason, "missing_reason", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "available_timestamp_ns": self.available_timestamp_ns,
            "clock_domain": self.clock_domain,
            "identity": self.identity.to_canonical_dict(),
            "kind": self.kind.value,
            "missing_reason": self.missing_reason,
            "newer_command_in_flight": self.newer_command_in_flight,
            "provider_id": self.provider_id,
            "selection_rule_id": self.selection_rule_id,
            "source_timestamp_ns": self.source_timestamp_ns,
            "valid": self.valid,
            "value_db": self.value_db,
        }


class UlSnrProxyProviderV1(Protocol):
    """Any SNR source.  ``observe`` must use only information at the cutoff."""

    provider_id: str
    kind: SnrProxyKind

    def observe(self, boundary: R4.DecisionBoundaryV1) -> UlSnrProxyObservationV1:
        ...


# ---------------------------------------------------------------------------
# RFsim provider
# ---------------------------------------------------------------------------


class RfsimCommandStatus(str, Enum):
    ACK = "ACK"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class RfsimSnrCommandRecordV1:
    """One RFsim channel command as the provider may see it.

    Deliberately has no profile, trace, step, Markov, reason or noise field.
    ``target_snr_db is None`` marks a command that sets the channel without a
    target (for example the clean restore); it can end, but never supply, an
    SNR value.
    """

    session_uuid: str
    command_seq: int
    clock_domain: str
    send_timestamp_ns: int
    ack_timestamp_ns: int
    status: RfsimCommandStatus
    clamped: Optional[bool]
    target_snr_db: Optional[float]

    def __post_init__(self) -> None:
        _canonical_uuid(self.session_uuid, "session_uuid")
        _non_negative_int(self.command_seq, "command_seq", MetadataError)
        _non_empty_str(self.clock_domain, "clock_domain", MetadataError)
        sent = _non_negative_int(self.send_timestamp_ns, "send_timestamp_ns", MetadataError)
        acked = _non_negative_int(self.ack_timestamp_ns, "ack_timestamp_ns", MetadataError)
        if acked < sent:
            raise MetadataError("ACK cannot precede send")
        if not isinstance(self.status, RfsimCommandStatus):
            raise MetadataError("status must be RfsimCommandStatus")
        if self.clamped is not None and type(self.clamped) is not bool:
            raise MetadataError("clamped must be bool or None")
        if self.target_snr_db is not None:
            _finite_float(self.target_snr_db, "target_snr_db", MetadataError)
            if self.clamped is None:
                raise MetadataError("a target command must state whether it clamped")

    @classmethod
    def from_log_entry(
        cls,
        entry: Mapping[str, Any],
        *,
        session_uuid: str,
        command_seq: int,
        clock_domain: str,
    ) -> "RfsimSnrCommandRecordV1":
        """Allow-list the retained ``command_log.json`` schema.

        Only ``send_monotonic_ns``, ``ack_monotonic_ns``, ``status``,
        ``clamped`` and ``target_snr_db`` are read; every other key, including
        ``profile_id``, ``step_index``, ``reason`` and
        ``commanded_noise_power_db``, is dropped here.
        """
        status = entry.get("status")
        try:
            parsed_status = RfsimCommandStatus(status)
        except ValueError as exc:
            raise MetadataError(f"unknown RFsim command status {status!r}") from exc
        target = entry.get("target_snr_db")
        return cls(
            session_uuid=session_uuid,
            command_seq=command_seq,
            clock_domain=clock_domain,
            send_timestamp_ns=entry["send_monotonic_ns"],
            ack_timestamp_ns=entry["ack_monotonic_ns"],
            status=parsed_status,
            clamped=entry.get("clamped"),
            target_snr_db=None if target is None else float(target),
        )


class RfsimEffectiveSnrProviderV1:
    """Latest effective RFsim target-SNR command, bound to one session/UE/clock."""

    kind = SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB

    def __init__(
        self,
        *,
        provider_id: str,
        session_uuid: str,
        ue_id: str,
        clock_domain: str,
        records: Iterable[RfsimSnrCommandRecordV1] = (),
    ) -> None:
        self.provider_id = _non_empty_str(provider_id, "provider_id", MetadataError)
        R4.DecisionIdentityV1(session_uuid, ue_id, 0)
        self.session_uuid = session_uuid
        self.ue_id = ue_id
        self.clock_domain = _non_empty_str(clock_domain, "clock_domain", MetadataError)
        self._records: list[RfsimSnrCommandRecordV1] = []
        self._ack_keys: list[int] = []
        self._send_keys: list[int] = []
        self._seqs: set[int] = set()
        self._sample_seq = 0
        for record in records:
            self.ingest(record)

    def ingest(self, record: RfsimSnrCommandRecordV1) -> None:
        if not isinstance(record, RfsimSnrCommandRecordV1):
            raise MetadataError("record must be RfsimSnrCommandRecordV1")
        if record.session_uuid != self.session_uuid:
            raise MetadataError("RFsim command belongs to a foreign session")
        if record.clock_domain != self.clock_domain:
            raise MetadataError("RFsim command uses a foreign clock domain")
        if record.command_seq in self._seqs:
            raise MetadataError("duplicate RFsim command_seq")
        if self._records:
            last = self._records[-1]
            if record.command_seq <= last.command_seq:
                raise MetadataError("RFsim commands must arrive in command order")
            if record.send_timestamp_ns < last.ack_timestamp_ns:
                # The controller is strictly serial: one command at a time.
                raise MetadataError("RFsim command overlaps the previous command")
        self._records.append(record)
        self._ack_keys.append(record.ack_timestamp_ns)
        self._send_keys.append(record.send_timestamp_ns)
        self._seqs.add(record.command_seq)

    def _missing(
        self, reason: str, *, source: Optional[int] = None, in_flight: bool = False
    ) -> UlSnrProxyObservationV1:
        return self._observation(None, source, reason, in_flight)

    def _observation(
        self,
        value: Optional[float],
        source: Optional[int],
        reason: Optional[str],
        in_flight: bool,
    ) -> UlSnrProxyObservationV1:
        identity = R4.SampleIdentityV1(self.session_uuid, self.ue_id, self._sample_seq)
        self._sample_seq += 1
        return UlSnrProxyObservationV1(
            identity=identity,
            kind=self.kind,
            provider_id=self.provider_id,
            selection_rule_id=RFSIM_SELECTION_RULE_ID,
            value_db=value,
            source_timestamp_ns=source,
            available_timestamp_ns=source,
            clock_domain=self.clock_domain,
            valid=value is not None,
            missing_reason=reason,
            newer_command_in_flight=in_flight,
        )

    def observe(self, boundary: R4.DecisionBoundaryV1) -> UlSnrProxyObservationV1:
        if not isinstance(boundary, R4.DecisionBoundaryV1):
            raise MetadataError("boundary must be DecisionBoundaryV1")
        if (
            boundary.identity.session_uuid != self.session_uuid
            or boundary.identity.ue_id != self.ue_id
        ):
            return self._missing("DECISION_IDENTITY_FOREIGN_TO_PROVIDER")
        if boundary.clock_domain != self.clock_domain:
            return self._missing("DECISION_CLOCK_FOREIGN_TO_PROVIDER")
        cutoff = boundary.state_commit_timestamp_ns  # strictly < action-open
        index = bisect.bisect_right(self._ack_keys, cutoff) - 1
        in_flight = bisect.bisect_right(self._send_keys, cutoff) - 1 > index
        if index < 0:
            return self._missing("NO_EFFECTIVE_COMMAND_BEFORE_CUTOFF", in_flight=in_flight)
        latest = self._records[index]
        if latest.status is not RfsimCommandStatus.ACK:
            return self._missing(
                "LATEST_EFFECTIVE_COMMAND_ERRORED",
                source=latest.ack_timestamp_ns, in_flight=in_flight,
            )
        if latest.target_snr_db is None:
            return self._missing(
                "LATEST_EFFECTIVE_COMMAND_HAS_NO_TARGET",
                source=latest.ack_timestamp_ns, in_flight=in_flight,
            )
        if latest.clamped:
            return self._missing(
                "LATEST_EFFECTIVE_COMMAND_CLAMPED",
                source=latest.ack_timestamp_ns, in_flight=in_flight,
            )
        return self._observation(
            float(latest.target_snr_db), latest.ack_timestamp_ns, None, in_flight
        )


# ---------------------------------------------------------------------------
# Bindings (no defaults) and the Run-5 guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnrProxyFreshnessV1(_CanonicalRecord):
    policy_id: str
    policy_version: int
    evidence_sha256: str
    max_age_ns: int

    RECORD_TYPE = "snr_proxy_freshness_v1"

    def __post_init__(self) -> None:
        _non_empty_str(self.policy_id, "policy_id", MetadataError)
        _positive_int(self.policy_version, "policy_version", MetadataError)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", MetadataError)
        _positive_int(self.max_age_ns, "max_age_ns", MetadataError)

    def _payload(self) -> Dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "max_age_ns": self.max_age_ns,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class SnrProxyScalingV1(_CanonicalRecord):
    scaling_id: str
    scaling_version: int
    evidence_sha256: str
    center_db: float
    scale_db: float
    support_min_db: float
    support_max_db: float

    RECORD_TYPE = "snr_proxy_scaling_v1"

    def __post_init__(self) -> None:
        _non_empty_str(self.scaling_id, "scaling_id", ScalingError)
        _positive_int(self.scaling_version, "scaling_version", ScalingError)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", ScalingError)
        _finite_float(self.center_db, "center_db", ScalingError)
        if _finite_float(self.scale_db, "scale_db", ScalingError) <= 0.0:
            raise ScalingError("scale_db must be > 0")
        low = _finite_float(self.support_min_db, "support_min_db", ScalingError)
        high = _finite_float(self.support_max_db, "support_max_db", ScalingError)
        if not low < high:
            raise ScalingError("support_min_db must be < support_max_db")

    def _payload(self) -> Dict[str, Any]:
        return {
            "center_db": float(self.center_db),
            "evidence_sha256": self.evidence_sha256,
            "scale_db": float(self.scale_db),
            "scaling_id": self.scaling_id,
            "scaling_version": self.scaling_version,
            "support_max_db": float(self.support_max_db),
            "support_min_db": float(self.support_min_db),
        }


@dataclass(frozen=True, slots=True)
class GuardedRun5StateV1(_CanonicalRecord):
    run4_guarded: R4.GuardedPolicyStateV2
    snr: UlSnrProxyObservationV1
    snr_freshness_sha256: str
    snr_scaling_sha256: str
    snr_age_ns: int
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "guarded_run5_state_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.run4_guarded, R4.GuardedPolicyStateV2):
            raise MetadataError("run4_guarded must be GuardedPolicyStateV2")
        if not isinstance(self.snr, UlSnrProxyObservationV1):
            raise MetadataError("snr must be UlSnrProxyObservationV1")
        _sha256_hex(self.snr_freshness_sha256, "snr_freshness_sha256", MetadataError)
        _sha256_hex(self.snr_scaling_sha256, "snr_scaling_sha256", MetadataError)
        _non_negative_int(self.snr_age_ns, "snr_age_ns", MetadataError)
        if self._attestation is not None and not _valid_guard(
            self._attestation, self._binding()
        ):
            raise MetadataError("Run-5 guard attestation is invalid")

    def _payload(self) -> Dict[str, Any]:
        return {
            "run4_guarded": self.run4_guarded.to_canonical_dict(),
            "snr": self.snr.to_canonical_dict(),
            "snr_age_ns": self.snr_age_ns,
            "snr_freshness_sha256": self.snr_freshness_sha256,
            "snr_scaling_sha256": self.snr_scaling_sha256,
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_guarded(self) -> bool:
        return _valid_guard(self._attestation, self._binding())

    def require_guarded(self) -> None:
        if not self.is_guarded:
            raise ExternalFallbackRequired(
                "state was not admitted by guard_run5_state_for_action()"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_guarded()
        return _CanonicalRecord.to_canonical_dict(self)


def guard_run5_state_for_action(
    state: R4.PolicyStateV2,
    snr: UlSnrProxyObservationV1,
    boundary: R4.DecisionBoundaryV1,
    run4_freshness: R4.FreshnessPolicyV2,
    snr_freshness: SnrProxyFreshnessV1,
    snr_scaling: SnrProxyScalingV1,
) -> GuardedRun5StateV1:
    """Run the unchanged Run-4 guard, then admit the SNR proxy or fall back."""
    if not isinstance(snr_freshness, SnrProxyFreshnessV1):
        raise MetadataError("snr_freshness must be SnrProxyFreshnessV1")
    if not isinstance(snr_scaling, SnrProxyScalingV1):
        raise ScalingError("snr_scaling must be SnrProxyScalingV1")
    run4_guarded = R4.guard_state_for_action(state, boundary, run4_freshness)

    if not isinstance(snr, UlSnrProxyObservationV1):
        raise ExternalFallbackRequired("SNR observation is absent; use external fallback")
    if snr.kind is not SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB:
        raise ExternalFallbackRequired("SNR observation has a foreign kind")
    if not snr.valid or snr.value_db is None:
        raise ExternalFallbackRequired(
            f"SNR proxy is missing/invalid ({snr.missing_reason}); use external fallback"
        )
    if (
        snr.identity.session_uuid != boundary.identity.session_uuid
        or snr.identity.ue_id != boundary.identity.ue_id
    ):
        raise ExternalFallbackRequired("SNR sample identity does not match decision")
    if snr.clock_domain != boundary.clock_domain:
        raise ExternalFallbackRequired("SNR and decision boundary use different clocks")
    assert snr.available_timestamp_ns is not None and snr.source_timestamp_ns is not None
    if snr.available_timestamp_ns > boundary.state_commit_timestamp_ns:
        raise ExternalFallbackRequired("SNR was not available when state was committed")
    age = boundary.action_open_timestamp_ns - snr.source_timestamp_ns
    if age <= 0:
        raise ExternalFallbackRequired("SNR source timestamp is not before action open")
    if age > snr_freshness.max_age_ns:
        raise ExternalFallbackRequired(f"SNR proxy is stale ({age} ns)")
    value = float(snr.value_db)
    if not snr_scaling.support_min_db <= value <= snr_scaling.support_max_db:
        raise ExternalFallbackRequired(
            f"SNR proxy {value} dB is outside the bound support"
        )
    candidate = GuardedRun5StateV1(
        run4_guarded=run4_guarded,
        snr=snr,
        snr_freshness_sha256=snr_freshness.canonical_sha256(),
        snr_scaling_sha256=snr_scaling.canonical_sha256(),
        snr_age_ns=age,
    )
    return replace(candidate, _attestation=_issue_guard(candidate._binding()))


@dataclass(frozen=True, slots=True)
class Run5PolicyFeatureVectorV1(_CanonicalRecord):
    values: Tuple[float, ...]
    run4_prefix_sha256: str
    guarded_state_sha256: str
    snr_scaling_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    RECORD_TYPE = "run5_policy_feature_vector_v1"

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != RUN5_POLICY_FEATURE_COUNT:
            raise ScalingError(f"values must be an exact {RUN5_POLICY_FEATURE_COUNT}-tuple")
        for index, value in enumerate(self.values):
            if type(value) is not float or not math.isfinite(value):
                raise ScalingError(f"values[{index}] must be a finite float")
        for name in ("run4_prefix_sha256", "guarded_state_sha256", "snr_scaling_sha256"):
            _sha256_hex(getattr(self, name), name, ScalingError)
        if self._attestation is not None and not _valid_features(
            self._attestation, self._binding()
        ):
            raise ScalingError("Run-5 feature attestation is invalid")

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return RUN5_POLICY_FEATURE_ORDER

    def as_tuple(self) -> Tuple[float, ...]:
        self.require_attested()
        return self.values

    def _payload(self) -> Dict[str, Any]:
        return {
            "guarded_state_sha256": self.guarded_state_sha256,
            "names": list(RUN5_POLICY_FEATURE_ORDER),
            "run4_prefix_sha256": self.run4_prefix_sha256,
            "snr_scaling_sha256": self.snr_scaling_sha256,
            "values": [v.hex() for v in self.values],
        }

    def _binding(self) -> str:
        return canonical_sha256(_record(self.RECORD_TYPE, self._payload()))

    @property
    def is_attested(self) -> bool:
        return _valid_features(self._attestation, self._binding())

    def require_attested(self) -> None:
        if not self.is_attested:
            raise ScalingError(
                "features must be produced by build_run5_policy_features() after guard"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.require_attested()
        return _CanonicalRecord.to_canonical_dict(self)


def scale_snr_db(value_db: float, scaling: SnrProxyScalingV1) -> float:
    return (float(value_db) - float(scaling.center_db)) / float(scaling.scale_db)


def build_run5_policy_features(
    guarded: GuardedRun5StateV1,
    run4_scaling: R4.EmpiricalScalingV2,
    snr_scaling: SnrProxyScalingV1,
) -> Run5PolicyFeatureVectorV1:
    """Run-4 vector verbatim at 0-20, then the scaled SNR proxy at 21."""
    if not isinstance(guarded, GuardedRun5StateV1):
        raise ScalingError("guarded must be GuardedRun5StateV1")
    guarded.require_guarded()
    if not isinstance(snr_scaling, SnrProxyScalingV1):
        raise ScalingError("snr_scaling must be SnrProxyScalingV1")
    if snr_scaling.canonical_sha256() != guarded.snr_scaling_sha256:
        raise ScalingError("SNR scaling differs from the one the guard admitted")
    prefix = R4.build_policy_features(guarded.run4_guarded, run4_scaling)
    prefix_values = prefix.as_tuple()
    if len(prefix_values) != RUN4_PREFIX_COUNT:  # pragma: no cover - invariant
        raise ScalingError("Run-4 prefix width drifted")
    snr_value = scale_snr_db(float(guarded.snr.value_db), snr_scaling)
    values = (*prefix_values, float(snr_value))
    candidate = Run5PolicyFeatureVectorV1(
        values=tuple(values),
        run4_prefix_sha256=prefix.canonical_sha256(),
        guarded_state_sha256=guarded.canonical_sha256(),
        snr_scaling_sha256=snr_scaling.canonical_sha256(),
    )
    return replace(candidate, _attestation=_issue_features(candidate._binding()))
