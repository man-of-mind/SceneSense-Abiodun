"""Private terminal replay for the D1 empirical contextual pilot.

This module is intentionally separate from :mod:`replay_buffer`.  D1 is a
one-step contextual problem: every supported observation/action receives one
deterministic expected-utility outcome and terminates.  Inventing a successor
state, a two-frame duration, a discount, or a bootstrap mask would turn that
problem into a fictitious trajectory.  None of those concepts exists in the
records or batches below.

Only exact D1 public record types are accepted.  The transition binds the full
observation, action, result (including the audit fields), D1 binding, and a
private collection identity.  D1 does not yet issue an unforgeable linkage
token between those public objects, so :meth:`from_d1` is a trusted-runner
numerical boundary, not a provenance-completeness claim; D2b must establish
that linkage before collection.  The collection identity is solely an in-run
duplicate/conflict key, not a production controller identity.  Lifetime
indexes survive FIFO eviction.

The tensor batch owns private CPU tensors, and every public tensor accessor
returns a clone.  It is a training-only boundary, not a durable evidence
store and not a security sandbox against deliberate underscore access.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX
from .empirical_contextual_contract import (
    FIXED_END_TO_FEEDBACK_STAGES_MS,
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    PILOT_UTILITY_SPEC,
    EmpiricalActionV1,
    EmpiricalPilotBindingV1,
    fixed_stage_latency_ms,
    require_supported_action,
)
from .empirical_contextual_environment import (
    EmpiricalOutcomeV1,
    EmpiricalPolicyObservationV1,
    EmpiricalStepAuditV1,
    EmpiricalStepResultV1,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    POLICY_OBSERVATION_DEPLOYABILITY,
    assert_policy_features_exclude_forbidden_fields,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "BindingMismatchError",
    "DuplicateTransitionError",
    "EmpiricalTerminalBatchV1",
    "EmpiricalTerminalBindingV1",
    "EmpiricalTerminalReplayV1",
    "EmpiricalTerminalTransitionV1",
    "IdentityConflictError",
    "NonFiniteInReplayDtypeError",
    "PHASE_LABEL",
    "Q_CRITIC_NORMALIZER",
    "REPLAY_SCHEMA",
    "ReplaySamplingError",
    "TerminalReplayError",
    "TransitionRejectedError",
]


PHASE_LABEL = "D1_EMPIRICAL_TERMINAL_CONTEXTUAL_TRAINING_ONLY"
REPLAY_SCHEMA = "splitfusion.empirical_terminal_contextual_replay.v1"
TRANSITION_SCHEMA = "splitfusion.empirical_terminal_contextual_transition.v1"
Q_CRITIC_NORMALIZER = Q_E4_MAX


class TerminalReplayError(Exception):
    """Base class for the private D1 terminal replay path."""


class TransitionRejectedError(TerminalReplayError):
    """A candidate is not an exact, supported, reward-bearing D1 record."""


class NonFiniteInReplayDtypeError(TransitionRejectedError):
    """A finite Python value becomes non-finite in float32 replay."""


class DuplicateTransitionError(TransitionRejectedError):
    """This exact attested collection transition was already seen."""


class IdentityConflictError(TransitionRejectedError):
    """One private collection key was offered with two different records."""


class BindingMismatchError(TransitionRejectedError):
    """A row belongs to a different frozen D1 learning problem."""


class ReplaySamplingError(TerminalReplayError):
    """A sample request or its random generator is invalid."""


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_exact_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TransitionRejectedError(f"{name} must be an exact finite float")
    return value


def _finite_in_float32(values: Sequence[float], name: str) -> None:
    converted = torch.tensor(tuple(values), dtype=torch.float32)
    finite = torch.isfinite(converted)
    if bool(finite.all()):
        return
    index = int((~finite).nonzero()[0])
    raise NonFiniteInReplayDtypeError(
        f"{name}[{index}]={values[index]!r} is finite in Python but becomes "
        f"{converted[index].item()!r} in the required float32 replay dtype"
    )


def _transition_document(
    *,
    collection_session_uuid: str,
    collection_seq: int,
    observation: EmpiricalPolicyObservationV1,
    action: EmpiricalActionV1,
    result: EmpiricalStepResultV1,
    d1_binding: EmpiricalPilotBindingV1,
) -> Dict[str, Any]:
    return {
        "action": asdict(action),
        "collection_seq": collection_seq,
        "collection_session_uuid": collection_session_uuid,
        "d1_binding": d1_binding.to_canonical_dict(),
        "observation": asdict(observation),
        "result": asdict(result),
        "schema": TRANSITION_SCHEMA,
    }


@dataclass(frozen=True, slots=True)
class EmpiricalTerminalTransitionV1:
    """One exact, terminal D1 contextual sample with full provenance."""

    collection_session_uuid: str
    collection_seq: int
    observation: EmpiricalPolicyObservationV1
    action: EmpiricalActionV1
    result: EmpiricalStepResultV1
    d1_binding: EmpiricalPilotBindingV1
    _attestation_sha256: str = field(repr=False)

    @classmethod
    def from_d1(
        cls,
        *,
        collection_session_uuid: str,
        collection_seq: int,
        observation: EmpiricalPolicyObservationV1,
        action: EmpiricalActionV1,
        result: EmpiricalStepResultV1,
        d1_binding: EmpiricalPilotBindingV1,
    ) -> "EmpiricalTerminalTransitionV1":
        document = _transition_document(
            collection_session_uuid=collection_session_uuid,
            collection_seq=collection_seq,
            observation=observation,
            action=action,
            result=result,
            d1_binding=d1_binding,
        )
        transition = cls(
            collection_session_uuid=collection_session_uuid,
            collection_seq=collection_seq,
            observation=observation,
            action=action,
            result=result,
            d1_binding=d1_binding,
            _attestation_sha256=canonical_sha256(document),
        )
        transition.revalidate()
        return transition

    @property
    def reward(self) -> float:
        value = self.result.policy.reward
        if type(value) is not float:
            raise TransitionRejectedError("D1 terminal reward is unavailable")
        return value

    @property
    def logical_key(self) -> Tuple[str, int]:
        return (self.collection_session_uuid, self.collection_seq)

    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _transition_document(
                collection_session_uuid=self.collection_session_uuid,
                collection_seq=self.collection_seq,
                observation=self.observation,
                action=self.action,
                result=self.result,
                d1_binding=self.d1_binding,
            )
        )

    def revalidate(self) -> None:
        """Re-prove the entire D1 result and its attestation."""
        try:
            parsed = uuid.UUID(self.collection_session_uuid)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TransitionRejectedError(
                "collection_session_uuid must be a canonical UUID"
            ) from exc
        if str(parsed) != self.collection_session_uuid:
            raise TransitionRejectedError(
                "collection_session_uuid must use canonical lowercase form"
            )
        if type(self.collection_seq) is not int or self.collection_seq < 0:
            raise TransitionRejectedError(
                "collection_seq must be an exact non-negative integer"
            )
        if type(self.observation) is not EmpiricalPolicyObservationV1:
            raise TransitionRejectedError(
                "observation must be an exact EmpiricalPolicyObservationV1"
            )
        if type(self.action) is not EmpiricalActionV1:
            raise TransitionRejectedError(
                "action must be an exact EmpiricalActionV1"
            )
        if type(self.result) is not EmpiricalStepResultV1:
            raise TransitionRejectedError(
                "result must be an exact EmpiricalStepResultV1"
            )
        if type(self.d1_binding) is not EmpiricalPilotBindingV1:
            raise TransitionRejectedError(
                "d1_binding must be an exact EmpiricalPilotBindingV1"
            )
        try:
            self.d1_binding.__post_init__()
        except ValueError as exc:
            raise TransitionRejectedError(
                "the full D1 binding no longer satisfies its contract"
            ) from exc

        observation = self.observation
        if observation.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise TransitionRejectedError("D1 policy feature order drift")
        if len(observation.values) != POLICY_FEATURE_COUNT:
            raise TransitionRejectedError("D1 policy observation is not 31-wide")
        for index, value in enumerate(observation.values):
            _finite_exact_float(value, f"observation.values[{index}]")
        binding_sha256 = self.d1_binding.canonical_sha256()
        if observation.environment_binding_sha256 != binding_sha256:
            raise BindingMismatchError(
                "observation environment binding does not equal the full D1 binding"
            )
        if (
            observation.normalization_spec_sha256
            != self.d1_binding.normalization_spec_sha256
            or observation.freshness_policy_sha256
            != self.d1_binding.freshness_policy_sha256
        ):
            raise BindingMismatchError(
                "observation preprocessing hashes differ from the D1 binding"
            )
        if observation.deployability != POLICY_OBSERVATION_DEPLOYABILITY:
            raise TransitionRejectedError("D1 observation deployability drift")

        supported = require_supported_action(
            self.action.mode_id, self.action.q_e4
        )
        if supported != self.action:
            raise TransitionRejectedError("D1 action validation changed its value")

        if type(self.result.policy) is not EmpiricalOutcomeV1:
            raise TransitionRejectedError("policy outcome has a foreign type")
        if type(self.result.audit) is not EmpiricalStepAuditV1:
            raise TransitionRejectedError("step audit has a foreign type")
        policy = self.result.policy
        audit = self.result.audit
        if policy.status != "MODELED_EXPECTED_UTILITY_DEFINED":
            raise TransitionRejectedError(
                f"D1 outcome status {policy.status!r} is not reward-bearing"
            )
        if policy.terminated is not True or policy.truncated is not False:
            raise TransitionRejectedError(
                "D1 empirical outcome must be terminal and not truncated"
            )
        reward = _finite_exact_float(policy.reward, "policy.reward")
        q_perc = _finite_exact_float(policy.q_perc, "policy.q_perc")
        p_reassembly = _finite_exact_float(
            policy.p_complete_reassembly_given_sent,
            "policy.p_complete_reassembly_given_sent",
        )
        p_admit = _finite_exact_float(
            policy.p_edge_admission_given_reassembled,
            "policy.p_edge_admission_given_reassembled",
        )
        p_service = _finite_exact_float(
            policy.p_edge_admission_given_sent,
            "policy.p_edge_admission_given_sent",
        )
        if not 0.0 <= q_perc <= 1.0:
            raise TransitionRejectedError("q_perc lies outside [0, 1]")
        for name, value in (
            ("p_complete_reassembly_given_sent", p_reassembly),
            ("p_edge_admission_given_reassembled", p_admit),
            ("p_edge_admission_given_sent", p_service),
        ):
            if not 0.0 <= value <= 1.0:
                raise TransitionRejectedError(f"{name} lies outside [0, 1]")
        if not math.isclose(
            p_service, p_reassembly * p_admit, rel_tol=0.0, abs_tol=1e-15
        ):
            raise TransitionRejectedError(
                "D1 edge-admission probability does not equal reassembly "
                "probability times conditional edge-admission probability"
            )

        p50 = _finite_exact_float(
            policy.conditional_feature_uplink_p50_ms, "conditional p50"
        )
        p95 = _finite_exact_float(
            policy.conditional_feature_uplink_p95_ms, "conditional p95"
        )
        p99 = _finite_exact_float(
            policy.conditional_feature_uplink_p99_ms, "conditional p99"
        )
        if not 0.0 <= p50 <= p95 <= p99:
            raise TransitionRejectedError("conditional latency quantiles are invalid")
        fixed = fixed_stage_latency_ms()
        if (
            policy.fixed_stage_latency_ms != fixed
            or policy.fixed_latency_stages_ms != FIXED_END_TO_FEEDBACK_STAGES_MS
        ):
            raise TransitionRejectedError("fixed D1 latency-stage contract drift")
        proxies = (fixed + p50, fixed + p95, fixed + p99)
        if (
            policy.latency_proxy_ms,
            policy.latency_proxy_p95_ms,
            policy.latency_proxy_p99_ms,
        ) != proxies:
            raise TransitionRejectedError("D1 latency proxies do not reconcile")
        if policy.deadline_ms != PILOT_UTILITY_SPEC.deadline_ms:
            raise TransitionRejectedError("D1 deadline drift")
        expected_misses = tuple(
            value > PILOT_UTILITY_SPEC.deadline_ms for value in proxies
        )
        if (
            policy.modeled_budget_miss,
            policy.modeled_budget_miss_p95,
            policy.modeled_budget_miss_p99,
        ) != expected_misses:
            raise TransitionRejectedError("D1 budget-miss flags do not reconcile")
        if policy.estimator != PILOT_UTILITY_SPEC.estimator:
            raise TransitionRejectedError("D1 estimator identity drift")
        expected_reward = PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=p_service,
            q_perc=q_perc,
            latency_proxy_ms=proxies[0],
        )
        if reward != expected_reward:
            raise TransitionRejectedError(
                "D1 reward does not equal the registered expected-utility formula"
            )

        if audit.executed_mode_id != self.action.mode_id or (
            audit.executed_q_e4 != self.action.q_e4
        ):
            raise TransitionRejectedError("audit action differs from executed action")
        if audit.utility_spec_sha256 != self.d1_binding.utility_spec_sha256:
            raise BindingMismatchError("audit utility hash differs from D1 binding")
        for name in (
            "sample_id",
            "episode_id",
            "hidden_network_profile",
            "hidden_trace_id",
            "surface_evidence_status",
            "network_evidence_class",
            "service_non_admission_semantics",
            "timeout_probability_status",
        ):
            owner = policy if hasattr(policy, name) else audit
            if type(getattr(owner, name)) is not str or not getattr(owner, name):
                raise TransitionRejectedError(f"{name} must be a non-empty string")
        if not _is_sha256(audit.hidden_radio_row_sha256):
            raise TransitionRejectedError("hidden radio-row hash is invalid")
        for name in (
            "frame_id",
            "hidden_radio_csv_row_number",
            "hidden_trace_step_index",
            "datagram_count",
        ):
            value = getattr(audit, name)
            if type(value) is not int or value < 0:
                raise TransitionRejectedError(f"audit {name} must be non-negative int")
        if audit.datagram_count < 1:
            raise TransitionRejectedError("audit datagram_count must be positive")
        _finite_exact_float(audit.hidden_target_snr_db, "hidden_target_snr_db")
        payload = _finite_exact_float(
            audit.total_transmitted_bytes, "total_transmitted_bytes"
        )
        if payload <= 0.0:
            raise TransitionRejectedError("total_transmitted_bytes must be positive")

        observed_digest = self.canonical_sha256()
        if not _is_sha256(self._attestation_sha256) or (
            self._attestation_sha256 != observed_digest
        ):
            raise TransitionRejectedError(
                "transition attestation no longer matches its D1 record"
            )


@dataclass(frozen=True, slots=True)
class EmpiricalTerminalBindingV1:
    """The complete homogeneous identity of one private replay problem."""

    d1_binding: EmpiricalPilotBindingV1
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int
    modeled_smoke_support_sha256: str
    float_dtype: str
    schema: str = REPLAY_SCHEMA
    evidence_class: str = PHASE_LABEL

    @classmethod
    def from_transition(
        cls, transition: EmpiricalTerminalTransitionV1
    ) -> "EmpiricalTerminalBindingV1":
        return cls(
            d1_binding=transition.d1_binding,
            policy_feature_order=tuple(transition.observation.policy_feature_order),
            policy_feature_count=len(transition.observation.values),
            modeled_smoke_support_sha256=(
                transition.d1_binding.modeled_smoke_support_sha256
            ),
            float_dtype=str(torch.float32),
        )

    def __post_init__(self) -> None:
        if type(self.d1_binding) is not EmpiricalPilotBindingV1:
            raise TerminalReplayError("binding carries a foreign D1 binding")
        try:
            self.d1_binding.__post_init__()
        except ValueError as exc:
            raise TerminalReplayError("inner D1 binding contract drift") from exc
        if self.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise TerminalReplayError("binding policy feature order drift")
        if self.policy_feature_count != POLICY_FEATURE_COUNT:
            raise TerminalReplayError("binding policy feature count drift")
        if self.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise TerminalReplayError("binding modeled support hash drift")
        if self.float_dtype != str(torch.float32):
            raise TerminalReplayError("terminal replay dtype must be torch.float32")
        if self.schema != REPLAY_SCHEMA or self.evidence_class != PHASE_LABEL:
            raise TerminalReplayError("terminal replay binding identity drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "d1_binding": self.d1_binding.to_canonical_dict(),
            "evidence_class": self.evidence_class,
            "float_dtype": self.float_dtype,
            "modeled_smoke_support_sha256": self.modeled_smoke_support_sha256,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "schema": self.schema,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def assert_matches(self, other: "EmpiricalTerminalBindingV1") -> None:
        if type(other) is not EmpiricalTerminalBindingV1:
            raise BindingMismatchError("candidate binding has a foreign type")
        for name in self.__dataclass_fields__:
            if getattr(self, name) != getattr(other, name):
                raise BindingMismatchError(
                    f"terminal replay binding differs in {name}"
                )


@dataclass(frozen=True, slots=True)
class _StoredTerminalRow:
    digest: str
    logical_key: Tuple[str, int]
    state_values: Tuple[float, ...]
    mode_id: int
    q_e4: int
    reward: float
    audit: Mapping[str, Any]


@dataclass(frozen=True, slots=True, eq=False)
class EmpiricalTerminalBatchV1:
    """Clone-isolated float32 tensors for one-step terminal learning."""

    _state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _reward: Tensor
    binding: EmpiricalTerminalBindingV1
    audit: Tuple[Mapping[str, Any], ...]
    float_dtype: torch.dtype = torch.float32

    _TENSOR_FIELDS = ("_state", "_mode_id", "_q_e4", "_reward")

    def __post_init__(self) -> None:
        if type(self.binding) is not EmpiricalTerminalBindingV1:
            raise TerminalReplayError("batch binding has a foreign type")
        # Re-run the binding's exact invariants in case a frozen object was
        # deliberately altered with object.__setattr__ after construction.
        self.binding.__post_init__()
        if self.float_dtype is not torch.float32:
            raise TerminalReplayError("terminal batch must use torch.float32")
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TerminalReplayError(f"{name} must be a torch.Tensor")
            object.__setattr__(self, name, value.detach().clone())
        size = int(self._state.shape[0]) if self._state.dim() >= 1 else 0
        if size < 1 or self._state.shape != (size, POLICY_FEATURE_COUNT):
            raise TerminalReplayError(
                f"_state must be non-empty [B, {POLICY_FEATURE_COUNT}]"
            )
        if self._state.device.type != "cpu" or self._state.dtype is not torch.float32:
            raise TerminalReplayError("_state must be CPU float32")
        if not bool(torch.isfinite(self._state).all()):
            raise TerminalReplayError("_state contains a non-finite value")
        for name, tensor in (
            ("_mode_id", self._mode_id),
            ("_q_e4", self._q_e4),
        ):
            if tensor.shape != (size,) or tensor.dtype is not torch.int64:
                raise TerminalReplayError(f"{name} must be [B] int64")
            if tensor.device.type != "cpu":
                raise TerminalReplayError(f"{name} must be on CPU")
        if self._reward.shape != (size,) or self._reward.dtype is not torch.float32:
            raise TerminalReplayError("_reward must be [B] float32")
        if self._reward.device.type != "cpu" or not bool(
            torch.isfinite(self._reward).all()
        ):
            raise TerminalReplayError("_reward must be finite on CPU")
        if bool((self._mode_id < 0).any()) or bool(
            (self._mode_id >= EXPECTED_MODE_COUNT).any()
        ):
            raise TerminalReplayError("_mode_id lies outside [0, 11]")
        bounds = torch.tensor(
            MODELED_SMOKE_SUPPORT.mode_q_e4_bounds, dtype=torch.int64
        )
        lower = bounds[:, 0].gather(0, self._mode_id)
        upper = bounds[:, 1].gather(0, self._mode_id)
        if bool(((self._q_e4 < lower) | (self._q_e4 > upper)).any()):
            raise TerminalReplayError("_q_e4 lies outside mode-specific support")
        if type(self.audit) is not tuple or len(self.audit) != size:
            raise TerminalReplayError("audit must contain exactly one mapping per row")
        if any(not isinstance(record, Mapping) for record in self.audit):
            raise TerminalReplayError("every audit row must be a mapping")
        frozen_audit = tuple(
            MappingProxyType(dict(record)) for record in tuple(self.audit)
        )
        object.__setattr__(self, "audit", frozen_audit)

    @property
    def state(self) -> Tensor:
        return self._state.clone()

    @property
    def mode_id(self) -> Tensor:
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        return self._q_e4.clone()

    @property
    def q_normalized_executed(self) -> Tensor:
        return self._q_e4.to(torch.float32) / float(Q_CRITIC_NORMALIZER)

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    @property
    def batch_size(self) -> int:
        return int(self._state.shape[0])

    def to_canonical_metadata(self) -> Dict[str, Any]:
        return {
            "audit": [dict(record) for record in self.audit],
            "batch_size": self.batch_size,
            "binding": self.binding.to_canonical_dict(),
            "evidence_class": PHASE_LABEL,
            "float_dtype": str(self.float_dtype),
            "schema": REPLAY_SCHEMA,
        }


class EmpiricalTerminalReplayV1:
    """Bounded FIFO replay over exact D1 terminal contextual samples."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise TerminalReplayError("capacity must be an exact positive integer")
        assert_policy_features_exclude_forbidden_fields()
        self._capacity = capacity
        self._rows: Deque[_StoredTerminalRow] = deque()
        self._binding: Optional[EmpiricalTerminalBindingV1] = None
        self._seen_digests: set[str] = set()
        self._seen_identities: Dict[Tuple[str, int], str] = {}
        self._accepted_count = 0
        self._evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def binding(self) -> Optional[EmpiricalTerminalBindingV1]:
        return self._binding

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def evicted_count(self) -> int:
        return self._evicted_count

    @property
    def seen_digest_count(self) -> int:
        return len(self._seen_digests)

    @property
    def seen_identity_count(self) -> int:
        return len(self._seen_identities)

    def resident_audit(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(row.audit)) for row in self._rows)

    def insert(self, transition: Any) -> None:
        row = self._validate(transition)
        candidate_binding = EmpiricalTerminalBindingV1.from_transition(transition)
        if self._binding is None:
            frozen_binding = candidate_binding
        else:
            self._binding.assert_matches(candidate_binding)
            frozen_binding = self._binding

        self._binding = frozen_binding
        self._seen_digests.add(row.digest)
        self._seen_identities[row.logical_key] = row.digest
        self._rows.append(row)
        self._accepted_count += 1
        while len(self._rows) > self._capacity:
            self._rows.popleft()
            self._evicted_count += 1

    def _validate(self, transition: Any) -> _StoredTerminalRow:
        if type(transition) is not EmpiricalTerminalTransitionV1:
            raise TransitionRejectedError(
                "terminal contextual replay accepts only an exact "
                "EmpiricalTerminalTransitionV1; production replay records, "
                "aggregates and adapters are forbidden"
            )
        transition.revalidate()
        state = tuple(transition.observation.values)
        reward = transition.reward
        _finite_in_float32(state, "observation.values")
        _finite_in_float32((reward,), "reward")
        mode_id = transition.action.mode_id
        q_e4 = transition.action.q_e4
        require_supported_action(mode_id, q_e4)
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise TransitionRejectedError("mode_id is outside the 12-mode catalog")
        normalized = q_e4 / float(Q_CRITIC_NORMALIZER)
        _finite_in_float32((normalized,), "q_normalized_executed")

        digest = transition.canonical_sha256()
        key = transition.logical_key
        known = self._seen_identities.get(key)
        if known is not None and known != digest:
            raise IdentityConflictError(
                f"private collection key {key} already binds digest {known}, "
                f"not conflicting digest {digest}"
            )
        if digest in self._seen_digests:
            raise DuplicateTransitionError(
                f"terminal contextual transition {digest} was already inserted"
            )
        audit = MappingProxyType(
            {
                "collection_seq": transition.collection_seq,
                "collection_session_uuid": transition.collection_session_uuid,
                "d1_binding_sha256": transition.d1_binding.canonical_sha256(),
                "episode_id": transition.result.audit.episode_id,
                "frame_id": transition.result.audit.frame_id,
                "hidden_network_profile": (
                    transition.result.audit.hidden_network_profile
                ),
                "hidden_radio_csv_row_number": (
                    transition.result.audit.hidden_radio_csv_row_number
                ),
                "sample_id": transition.result.audit.sample_id,
                "transition_sha256": digest,
            }
        )
        return _StoredTerminalRow(
            digest=digest,
            logical_key=key,
            state_values=state,
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            audit=audit,
        )

    @staticmethod
    def _require_generator(generator: Any) -> torch.Generator:
        if not isinstance(generator, torch.Generator):
            raise ReplaySamplingError("sample requires an explicit torch.Generator")
        if generator is torch.default_generator:
            raise ReplaySamplingError("the global default generator is forbidden")
        if generator.device.type != "cpu":
            raise ReplaySamplingError("sample requires a local CPU generator")
        return generator

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> EmpiricalTerminalBatchV1:
        if type(batch_size) is not int or batch_size < 1:
            raise ReplaySamplingError(
                "batch_size must be an exact positive integer"
            )
        self._require_generator(generator)
        if batch_size > len(self._rows):
            raise ReplaySamplingError(
                f"cannot sample {batch_size} rows from {len(self._rows)} resident rows"
            )
        if self._binding is None:
            raise ReplaySamplingError("cannot sample an unbound empty replay")
        indices = torch.randperm(len(self._rows), generator=generator)[
            :batch_size
        ].tolist()
        resident = tuple(self._rows)
        rows = tuple(resident[index] for index in indices)
        return EmpiricalTerminalBatchV1(
            _state=torch.tensor(
                [row.state_values for row in rows], dtype=torch.float32
            ),
            _mode_id=torch.tensor(
                [row.mode_id for row in rows], dtype=torch.int64
            ),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor(
                [row.reward for row in rows], dtype=torch.float32
            ),
            binding=self._binding,
            audit=tuple(row.audit for row in rows),
            float_dtype=torch.float32,
        )
