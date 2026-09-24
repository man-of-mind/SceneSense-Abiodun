"""Pure causal adapter for the two Run-4 UE network-state observations.

The adapter is deliberately stateless.  It does not cache, forward-fill,
normalize, impose an age threshold, read evidence, or choose the external
fallback action.  It performs only the ordering and identity work that is
safe before the 12-cell calibration is complete:

* select the latest strictly prior UE-decoded, table-0, round-0 UL grant from
  the registered SINR-driven scheduler; and
* select the latest UE RLC backlog sample visible strictly before both the
  payload enqueue and state-commit boundaries.

No eligible sample produces an explicit invalid observation with ``None`` as
its value.  In particular, missing UL-MCS is never encoded as the valid MCS-0
index, and a measured zero-byte RLC queue remains a valid zero.

Importing this module performs no I/O and starts no runtime component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

__all__ = [
    "StateAdapterError",
    "RawEvidenceError",
    "EvidenceIdentityConflictError",
    "AmbiguousLatestEvidenceError",
    "RawUeUlDciGrantCandidateV1",
    "RawUeRlcBacklogSampleV1",
    "MISSING_PRIOR_UL_GRANT",
    "MISSING_PRE_ACTION_RLC_BACKLOG",
    "select_prior_new_data_ul_mcs",
    "select_pre_action_rlc_backlog",
]


MISSING_PRIOR_UL_GRANT = "NO_ELIGIBLE_STRICTLY_PRIOR_NEW_DATA_UL_GRANT"
MISSING_PRE_ACTION_RLC_BACKLOG = (
    "NO_ELIGIBLE_STRICTLY_PRE_ENQUEUE_RLC_BACKLOG"
)


class StateAdapterError(ValueError):
    """Base class for state-adapter evidence failures."""


class RawEvidenceError(StateAdapterError):
    """A raw candidate is malformed rather than merely ineligible."""


class EvidenceIdentityConflictError(StateAdapterError):
    """One durable evidence identity names contradictory records."""


class AmbiguousLatestEvidenceError(StateAdapterError):
    """Distinct eligible records tie at the latest source instant."""


def _exact_non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RawEvidenceError(f"{name} must be an exact int >= 0")
    return value


def _non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise RawEvidenceError(f"{name} must be a non-empty str")
    return value


def _timestamps(
    source_timestamp_ns: object,
    available_timestamp_ns: object,
) -> Tuple[int, int]:
    source = _exact_non_negative_int(
        source_timestamp_ns, "source_timestamp_ns"
    )
    available = _exact_non_negative_int(
        available_timestamp_ns, "available_timestamp_ns"
    )
    if source > available:
        raise RawEvidenceError(
            "source_timestamp_ns must be <= available_timestamp_ns"
        )
    return source, available


def _boundary(value: object) -> contract.DecisionBoundaryV1:
    if type(value) is not contract.DecisionBoundaryV1:
        raise RawEvidenceError("boundary must be exactly DecisionBoundaryV1")
    return value


@dataclass(frozen=True, slots=True)
class RawUeUlDciGrantCandidateV1:
    """One UE-decoded UL-DCI grant candidate before policy filtering."""

    identity: contract.SampleIdentityV1
    grant_identity: str
    link_direction: contract.LinkDirection
    mcs_table: int
    mcs_index: int
    harq_round: int
    new_data_indicator: int
    scheduler_policy_id: str
    source: str
    source_timestamp_ns: int
    available_timestamp_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.SampleIdentityV1:
            raise RawEvidenceError(
                "identity must be exactly SampleIdentityV1"
            )
        _non_empty_str(self.grant_identity, "grant_identity")
        if not isinstance(self.link_direction, contract.LinkDirection):
            raise RawEvidenceError(
                "link_direction must be a LinkDirection"
            )
        _exact_non_negative_int(self.mcs_table, "mcs_table")
        _exact_non_negative_int(self.mcs_index, "mcs_index")
        _exact_non_negative_int(self.harq_round, "harq_round")
        if type(self.new_data_indicator) is not int or (
            self.new_data_indicator not in (0, 1)
        ):
            raise RawEvidenceError("new_data_indicator must be exactly 0 or 1")
        _non_empty_str(self.scheduler_policy_id, "scheduler_policy_id")
        _non_empty_str(self.source, "source")
        _timestamps(self.source_timestamp_ns, self.available_timestamp_ns)
        _non_empty_str(self.clock_domain, "clock_domain")


@dataclass(frozen=True, slots=True)
class RawUeRlcBacklogSampleV1:
    """One raw UE RLC backlog sample before causal boundary filtering."""

    identity: contract.SampleIdentityV1
    backlog_bytes: int
    link_direction: contract.LinkDirection
    source: str
    source_timestamp_ns: int
    available_timestamp_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.SampleIdentityV1:
            raise RawEvidenceError(
                "identity must be exactly SampleIdentityV1"
            )
        _exact_non_negative_int(self.backlog_bytes, "backlog_bytes")
        if not isinstance(self.link_direction, contract.LinkDirection):
            raise RawEvidenceError(
                "link_direction must be a LinkDirection"
            )
        _non_empty_str(self.source, "source")
        _timestamps(self.source_timestamp_ns, self.available_timestamp_ns)
        _non_empty_str(self.clock_domain, "clock_domain")


def _same_decision_scope(
    identity: contract.SampleIdentityV1,
    boundary: contract.DecisionBoundaryV1,
) -> bool:
    return (
        identity.session_uuid == boundary.identity.session_uuid
        and identity.ue_id == boundary.identity.ue_id
    )


def _missing_metadata(
    boundary: contract.DecisionBoundaryV1,
    *,
    kind: contract.MeasurementKind,
    source: str,
) -> contract.MeasurementMetadataV1:
    """Build decision-scoped invalid metadata, not a fabricated sample."""

    return contract.MeasurementMetadataV1(
        identity=contract.SampleIdentityV1(
            session_uuid=boundary.identity.session_uuid,
            ue_id=boundary.identity.ue_id,
            sample_seq=boundary.identity.decision_seq,
        ),
        kind=kind,
        observer=contract.Observer.UE,
        link_direction=contract.LinkDirection.UPLINK,
        source=source,
        source_timestamp_ns=boundary.state_commit_timestamp_ns,
        available_timestamp_ns=boundary.state_commit_timestamp_ns,
        clock_domain=boundary.clock_domain,
        valid=False,
    )


def _missing_prior_grant(
    boundary: contract.DecisionBoundaryV1,
) -> contract.PriorUlGrantObservationV1:
    metadata = _missing_metadata(
        boundary,
        kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
        source="run4_state_adapter:prior_ul_grant_selection",
    )
    return contract.PriorUlGrantObservationV1(
        observation=contract.ScalarObservationV1(
            value=None,
            metadata=metadata,
            missing_reason=MISSING_PRIOR_UL_GRANT,
        ),
        mcs_table=contract.UL_MCS_TABLE_ID,
        harq_round=None,
        new_data_indicator=None,
        grant_identity=None,
        scheduler_policy_id=contract.UL_MCS_POLICY_ID,
        selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
    )


def _missing_backlog(
    boundary: contract.DecisionBoundaryV1,
) -> contract.ScalarObservationV1:
    metadata = _missing_metadata(
        boundary,
        kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
        source="run4_state_adapter:pre_action_rlc_selection",
    )
    return contract.ScalarObservationV1(
        value=None,
        metadata=metadata,
        missing_reason=MISSING_PRE_ACTION_RLC_BACKLOG,
    )


def _deduplicate_grants(
    candidates: Iterable[RawUeUlDciGrantCandidateV1],
    boundary: contract.DecisionBoundaryV1,
) -> Tuple[RawUeUlDciGrantCandidateV1, ...]:
    by_grant: Dict[
        Tuple[str, str, str], RawUeUlDciGrantCandidateV1
    ] = {}
    by_sample: Dict[
        Tuple[str, str, int], RawUeUlDciGrantCandidateV1
    ] = {}
    unique = []
    for candidate in candidates:
        if type(candidate) is not RawUeUlDciGrantCandidateV1:
            raise RawEvidenceError(
                "all grant candidates must be exactly "
                "RawUeUlDciGrantCandidateV1"
            )
        if not _same_decision_scope(candidate.identity, boundary):
            continue
        grant_key = (
            candidate.identity.session_uuid,
            candidate.identity.ue_id,
            candidate.grant_identity,
        )
        sample_key = (
            candidate.identity.session_uuid,
            candidate.identity.ue_id,
            candidate.identity.sample_seq,
        )
        for name, mapping, key in (
            ("grant_identity", by_grant, grant_key),
            ("sample identity", by_sample, sample_key),
        ):
            existing = mapping.get(key)
            if existing is not None and existing != candidate:
                raise EvidenceIdentityConflictError(
                    f"contradictory records share {name}: {key!r}"
                )
        if grant_key in by_grant or sample_key in by_sample:
            # Both maps can only contain this byte-equivalent immutable value;
            # contradictory reuse was rejected above.
            continue
        by_grant[grant_key] = candidate
        by_sample[sample_key] = candidate
        unique.append(candidate)
    return tuple(unique)


def select_prior_new_data_ul_mcs(
    candidates: Iterable[RawUeUlDciGrantCandidateV1],
    boundary: contract.DecisionBoundaryV1,
) -> contract.PriorUlGrantObservationV1:
    """Select the latest causal new-data UL-MCS, or return explicit missing.

    Eligibility requires a source and availability instant strictly before the
    state commit, the same session/UE/clock domain, UL direction, table 0,
    HARQ round 0, and the registered scheduler policy.  NDI can be either 0 or
    1 because NR new data is represented by an NDI *toggle*, not by NDI=1.
    """

    boundary = _boundary(boundary)
    eligible = []
    for candidate in _deduplicate_grants(candidates, boundary):
        if (
            candidate.clock_domain != boundary.clock_domain
            or candidate.link_direction is not contract.LinkDirection.UPLINK
            or candidate.mcs_table != contract.UL_MCS_TABLE_ID
            or candidate.harq_round != 0
            or candidate.scheduler_policy_id != contract.UL_MCS_POLICY_ID
            or candidate.source_timestamp_ns
            >= boundary.state_commit_timestamp_ns
            or candidate.available_timestamp_ns
            >= boundary.state_commit_timestamp_ns
        ):
            continue
        if not (
            contract.UL_MCS_INDEX_MIN
            <= candidate.mcs_index
            <= contract.UL_MCS_INDEX_MAX
        ):
            raise RawEvidenceError(
                "an otherwise eligible table-0 grant has MCS outside [0, 28]"
            )
        eligible.append(candidate)

    if not eligible:
        return _missing_prior_grant(boundary)

    latest_source = max(item.source_timestamp_ns for item in eligible)
    latest = [
        item for item in eligible if item.source_timestamp_ns == latest_source
    ]
    if len(latest) != 1:
        identities = sorted(item.grant_identity for item in latest)
        raise AmbiguousLatestEvidenceError(
            "distinct eligible UL grants share the latest source timestamp: "
            f"{identities!r}"
        )
    selected = latest[0]
    metadata = contract.MeasurementMetadataV1(
        identity=selected.identity,
        kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
        observer=contract.Observer.UE,
        link_direction=contract.LinkDirection.UPLINK,
        source=selected.source,
        source_timestamp_ns=selected.source_timestamp_ns,
        available_timestamp_ns=selected.available_timestamp_ns,
        clock_domain=selected.clock_domain,
        valid=True,
    )
    return contract.PriorUlGrantObservationV1(
        observation=contract.ScalarObservationV1(
            value=selected.mcs_index,
            metadata=metadata,
            missing_reason=None,
        ),
        mcs_table=selected.mcs_table,
        harq_round=selected.harq_round,
        new_data_indicator=selected.new_data_indicator,
        grant_identity=selected.grant_identity,
        scheduler_policy_id=selected.scheduler_policy_id,
        selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
    )


def _deduplicate_backlog(
    samples: Iterable[RawUeRlcBacklogSampleV1],
    boundary: contract.DecisionBoundaryV1,
) -> Tuple[RawUeRlcBacklogSampleV1, ...]:
    by_sample: Dict[
        Tuple[str, str, int], RawUeRlcBacklogSampleV1
    ] = {}
    unique = []
    for sample in samples:
        if type(sample) is not RawUeRlcBacklogSampleV1:
            raise RawEvidenceError(
                "all backlog samples must be exactly "
                "RawUeRlcBacklogSampleV1"
            )
        if not _same_decision_scope(sample.identity, boundary):
            continue
        key = (
            sample.identity.session_uuid,
            sample.identity.ue_id,
            sample.identity.sample_seq,
        )
        existing = by_sample.get(key)
        if existing is not None and existing != sample:
            raise EvidenceIdentityConflictError(
                f"contradictory records share sample identity: {key!r}"
            )
        if existing is not None:
            continue
        by_sample[key] = sample
        unique.append(sample)
    return tuple(unique)


def select_pre_action_rlc_backlog(
    samples: Iterable[RawUeRlcBacklogSampleV1],
    boundary: contract.DecisionBoundaryV1,
    *,
    payload_enqueue_timestamp_ns: int,
) -> contract.ScalarObservationV1:
    """Select the latest backlog visible before enqueue and state commit.

    The function is stateless: an empty or wholly ineligible input returns
    explicit missing evidence rather than carrying a value from an earlier
    decision.  It intentionally imposes no age threshold; that remains an
    externally calibrated freshness/fallback policy.
    """

    boundary = _boundary(boundary)
    enqueue = _exact_non_negative_int(
        payload_enqueue_timestamp_ns, "payload_enqueue_timestamp_ns"
    )
    cutoff = min(enqueue, boundary.state_commit_timestamp_ns)
    eligible = [
        sample
        for sample in _deduplicate_backlog(samples, boundary)
        if (
            sample.clock_domain == boundary.clock_domain
            and sample.link_direction is contract.LinkDirection.UPLINK
            and sample.source_timestamp_ns < cutoff
            and sample.available_timestamp_ns < cutoff
        )
    ]
    if not eligible:
        return _missing_backlog(boundary)

    latest_source = max(item.source_timestamp_ns for item in eligible)
    latest = [
        item for item in eligible if item.source_timestamp_ns == latest_source
    ]
    if len(latest) != 1:
        identities = sorted(item.identity.sample_seq for item in latest)
        raise AmbiguousLatestEvidenceError(
            "distinct eligible RLC samples share the latest source timestamp: "
            f"{identities!r}"
        )
    selected = latest[0]
    metadata = contract.MeasurementMetadataV1(
        identity=selected.identity,
        kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
        observer=contract.Observer.UE,
        link_direction=contract.LinkDirection.UPLINK,
        source=selected.source,
        source_timestamp_ns=selected.source_timestamp_ns,
        available_timestamp_ns=selected.available_timestamp_ns,
        clock_domain=selected.clock_domain,
        valid=True,
    )
    return contract.ScalarObservationV1(
        value=selected.backlog_bytes,
        metadata=metadata,
        missing_reason=None,
    )
