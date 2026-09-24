"""Fail-closed replay boundary for attested Run-4 transitions.

This module is intentionally narrower than a training environment.  It reads
no evidence, invents no calibration values, launches no runtime component and
does not turn aggregate measurements or synthetic fixtures into learning
records.  Its sole accepted input is an *exact*, already-attested
:class:`~.run4_contract.SemiMarkovTransitionV2`.

The buffer is constructed with one explicit :class:`ReplayBindingV1`.  That
binding pins the Run-4 contract, feature, reward, transition and action-catalog
hashes together with the caller-selected freshness, scaling, calibration and
queue-kernel evidence hashes.  The last two are environment-level evidence:
the current transition schema does not carry them, so they are frozen at the
buffer/batch boundary rather than fabricated inside a transition.

The contract-derived per-row discount is converted to float32 once and stored
verbatim.  Replay never recomputes ``gamma ** duration``.  Likewise, absent
successors at TERMINATED/TRUNCATED boundaries use a zero *tensor sentinel*
only so a batch can be rectangular; ``has_next_state`` and ``bootstrap`` are
false for those rows, and the sentinel is never represented as observed state.
CONTINUES rows must carry the real successor and the successor's encoded
previous outcome must match the current reward resolution exactly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)

from .run4_contract import (
    FEATURE_SCHEMA_SHA256,
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    REWARD_SCHEMA_SHA256,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    TRAINING_EVIDENCE_CLASS,
    TRANSITION_SCHEMA_SHA256,
    EpisodeBoundary,
    PreviousOutcomeV1,
    SemiMarkovTransitionV2,
    assert_policy_feature_schema,
)

__all__ = [
    "BindingMismatchError",
    "DEFAULT_FLOAT_DTYPE",
    "DuplicateTransitionError",
    "EvidenceEligibilityError",
    "IdentityConflictError",
    "NonFiniteInReplayDtypeError",
    "Q_CRITIC_NORMALIZER",
    "REPLAY_SCHEMA_ID",
    "ReplayBindingV1",
    "ReplayBufferError",
    "ReplayBufferV1",
    "ReplaySamplingError",
    "ReplayTensorBatchV1",
    "TransitionRejectedError",
]


REPLAY_SCHEMA_ID = "splitfusion_hybrid_sac_run4_replay_v1"
DEFAULT_FLOAT_DTYPE = torch.float32
Q_CRITIC_NORMALIZER = Q_E4_MAX


class ReplayBufferError(Exception):
    """Base class for Run-4 replay failures."""


class TransitionRejectedError(ReplayBufferError):
    """A candidate record cannot enter the replay store."""


class BindingMismatchError(TransitionRejectedError):
    """A transition/batch does not match the buffer's frozen binding."""


class NonFiniteInReplayDtypeError(TransitionRejectedError):
    """A contract-finite value is unsafe after float32 conversion."""


class DuplicateTransitionError(TransitionRejectedError):
    """The exact transition digest was already seen by this buffer."""


class IdentityConflictError(TransitionRejectedError):
    """One logical decision was offered under two different digests."""


class ReplaySamplingError(ReplayBufferError):
    """A replay sampling request is malformed or cannot be satisfied."""


class EvidenceEligibilityError(ReplayBufferError):
    """The buffer lacks a verifier-issued empirical evidence attestation."""


class _EvidenceEligibility(str, Enum):
    VERIFIED_TRAINING = "VERIFIED_TRAINING"
    TEST_ONLY_MECHANICS = "TEST_ONLY_MECHANICS"


_EVIDENCE_ATTESTATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _VerifiedEvidenceAttestationV1:
    """Opaque hand-off that only an accepted evidence verifier may issue.

    There is deliberately no issuing function in this pre-calibration module.
    The 12-cell verifier must be integrated separately after its verdict and
    must bind its accepted calibration and queue-kernel report digests.  Until
    then, production ReplayBufferV1 construction is structurally impossible.
    """

    calibration_evidence_sha256: str
    queue_kernel_evidence_sha256: str
    verifier_manifest_sha256: str
    _token: Any

    def require_verified(self) -> None:
        if self._token is not _EVIDENCE_ATTESTATION_TOKEN:
            raise EvidenceEligibilityError(
                "evidence attestation was not issued by the Run-4 verifier"
            )
        _sha256(
            self.calibration_evidence_sha256,
            "calibration_evidence_sha256",
        )
        _sha256(
            self.queue_kernel_evidence_sha256,
            "queue_kernel_evidence_sha256",
        )
        _sha256(self.verifier_manifest_sha256, "verifier_manifest_sha256")


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReplayBufferError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayBufferError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayBufferError(f"{name} must be finite, got {value!r}")
    return result


def _assert_finite_float32(values: Sequence[float], name: str) -> None:
    converted = torch.tensor(tuple(values), dtype=DEFAULT_FLOAT_DTYPE)
    finite = torch.isfinite(converted)
    if bool(finite.all()):
        return
    index = int((~finite).nonzero()[0])
    raise NonFiniteInReplayDtypeError(
        f"{name}[{index}]={values[index]!r} is finite in the contract but "
        f"converts to {converted[index].item()!r} in torch.float32"
    )


@dataclass(frozen=True, slots=True)
class ReplayBindingV1:
    """One exact Run-4 learning problem shared by a buffer and every batch.

    No field has a production default. :meth:`from_verified_evidence` fills
    the static Run-4 pins only after a verifier-issued evidence attestation is
    supplied. Direct construction remains possible for deserialization, but
    :meth:`revalidate` refuses any altered static pin or forged eligibility.
    """

    schema_id: str
    schema_version: int
    schema_sha256: str
    feature_schema_sha256: str
    reward_schema_sha256: str
    transition_schema_sha256: str
    catalog_sha256: str
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int
    gamma: float
    freshness_policy_sha256: str
    empirical_scaling_sha256: str
    calibration_evidence_sha256: str
    queue_kernel_evidence_sha256: str
    training_evidence_class: str
    evidence_eligibility: str
    verifier_manifest_sha256: str
    _evidence_attestation: Optional[_VerifiedEvidenceAttestationV1]

    @classmethod
    def from_verified_evidence(
        cls,
        *,
        gamma: float,
        freshness_policy_sha256: str,
        empirical_scaling_sha256: str,
        evidence_attestation: _VerifiedEvidenceAttestationV1,
    ) -> "ReplayBindingV1":
        if type(evidence_attestation) is not _VerifiedEvidenceAttestationV1:
            raise EvidenceEligibilityError(
                "production binding requires the exact verifier-issued "
                "evidence attestation"
            )
        evidence_attestation.require_verified()
        binding = cls(
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            schema_sha256=SCHEMA_SHA256,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            reward_schema_sha256=REWARD_SCHEMA_SHA256,
            transition_schema_sha256=TRANSITION_SCHEMA_SHA256,
            catalog_sha256=CATALOG_SHA256,
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            policy_feature_count=POLICY_FEATURE_COUNT,
            gamma=float(gamma),
            freshness_policy_sha256=freshness_policy_sha256,
            empirical_scaling_sha256=empirical_scaling_sha256,
            calibration_evidence_sha256=(
                evidence_attestation.calibration_evidence_sha256
            ),
            queue_kernel_evidence_sha256=(
                evidence_attestation.queue_kernel_evidence_sha256
            ),
            training_evidence_class=TRAINING_EVIDENCE_CLASS,
            evidence_eligibility=_EvidenceEligibility.VERIFIED_TRAINING.value,
            verifier_manifest_sha256=(
                evidence_attestation.verifier_manifest_sha256
            ),
            _evidence_attestation=evidence_attestation,
        )
        binding.revalidate()
        return binding

    @classmethod
    def _for_test_only(
        cls,
        *,
        gamma: float,
        freshness_policy_sha256: str,
        empirical_scaling_sha256: str,
        calibration_evidence_sha256: str,
        queue_kernel_evidence_sha256: str,
    ) -> "ReplayBindingV1":
        """Mechanical unit-test binding; never eligible for ReplayBufferV1."""
        binding = cls(
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            schema_sha256=SCHEMA_SHA256,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            reward_schema_sha256=REWARD_SCHEMA_SHA256,
            transition_schema_sha256=TRANSITION_SCHEMA_SHA256,
            catalog_sha256=CATALOG_SHA256,
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            policy_feature_count=POLICY_FEATURE_COUNT,
            gamma=float(gamma),
            freshness_policy_sha256=freshness_policy_sha256,
            empirical_scaling_sha256=empirical_scaling_sha256,
            calibration_evidence_sha256=calibration_evidence_sha256,
            queue_kernel_evidence_sha256=queue_kernel_evidence_sha256,
            training_evidence_class=TRAINING_EVIDENCE_CLASS,
            evidence_eligibility=_EvidenceEligibility.TEST_ONLY_MECHANICS.value,
            verifier_manifest_sha256="0" * 64,
            _evidence_attestation=None,
        )
        binding.revalidate()
        return binding

    def revalidate(self) -> None:
        expected = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": SCHEMA_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "reward_schema_sha256": REWARD_SCHEMA_SHA256,
            "transition_schema_sha256": TRANSITION_SCHEMA_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "policy_feature_order": tuple(POLICY_FEATURE_ORDER),
            "policy_feature_count": POLICY_FEATURE_COUNT,
            "training_evidence_class": TRAINING_EVIDENCE_CLASS,
        }
        for name, wanted in expected.items():
            actual = getattr(self, name)
            if actual != wanted:
                raise BindingMismatchError(
                    f"Run-4 replay requires {name}={wanted!r}, got {actual!r}"
                )
        gamma = _finite_float(self.gamma, "gamma")
        if not 0.0 < gamma <= 1.0:
            raise ReplayBufferError(f"gamma must lie in (0, 1], got {gamma}")
        for name in (
            "freshness_policy_sha256",
            "empirical_scaling_sha256",
            "calibration_evidence_sha256",
            "queue_kernel_evidence_sha256",
        ):
            _sha256(getattr(self, name), name)
        _sha256(self.verifier_manifest_sha256, "verifier_manifest_sha256")
        if self.evidence_eligibility == _EvidenceEligibility.VERIFIED_TRAINING.value:
            if type(self._evidence_attestation) is not _VerifiedEvidenceAttestationV1:
                raise EvidenceEligibilityError(
                    "VERIFIED_TRAINING binding lacks verifier attestation"
                )
            self._evidence_attestation.require_verified()
            if (
                self.calibration_evidence_sha256
                != self._evidence_attestation.calibration_evidence_sha256
                or self.queue_kernel_evidence_sha256
                != self._evidence_attestation.queue_kernel_evidence_sha256
                or self.verifier_manifest_sha256
                != self._evidence_attestation.verifier_manifest_sha256
            ):
                raise EvidenceEligibilityError(
                    "binding evidence hashes do not match verifier attestation"
                )
        elif self.evidence_eligibility == _EvidenceEligibility.TEST_ONLY_MECHANICS.value:
            if self._evidence_attestation is not None:
                raise EvidenceEligibilityError(
                    "test-only binding must not carry a production attestation"
                )
        else:
            raise EvidenceEligibilityError(
                f"unknown evidence eligibility {self.evidence_eligibility!r}"
            )

    def require_training_eligible(self) -> None:
        self.revalidate()
        if self.evidence_eligibility != _EvidenceEligibility.VERIFIED_TRAINING.value:
            raise EvidenceEligibilityError(
                "ReplayBufferV1 requires verifier-attested calibration and "
                "queue-kernel evidence; TEST_ONLY_MECHANICS cannot train"
            )

    def assert_exactly(self, other: "ReplayBindingV1") -> None:
        if type(other) is not ReplayBindingV1:
            raise BindingMismatchError(
                f"expected exact ReplayBindingV1, got {type(other).__name__}"
            )
        self.revalidate()
        other.revalidate()
        for item in fields(self):
            mine = getattr(self, item.name)
            theirs = getattr(other, item.name)
            if mine != theirs:
                raise BindingMismatchError(
                    f"replay binding differs at {item.name}: "
                    f"{mine!r} != {theirs!r}"
                )

    def assert_transition(self, transition: SemiMarkovTransitionV2) -> None:
        """Check every binding that the transition itself can authenticate."""
        self.revalidate()
        comparisons = {
            "gamma": float(transition.gamma),
            "catalog_sha256": transition.action.catalog_sha256,
            "freshness_policy_sha256": transition.state.freshness_policy_sha256,
            "empirical_scaling_sha256": (
                transition.state_features.empirical_scaling_sha256
            ),
        }
        for name, actual in comparisons.items():
            wanted = getattr(self, name)
            if actual != wanted:
                raise BindingMismatchError(
                    f"buffer binds {name}={wanted!r}, transition declares "
                    f"{actual!r}"
                )
        if transition.next_state is not None:
            if (
                transition.next_state.freshness_policy_sha256
                != self.freshness_policy_sha256
            ):
                raise BindingMismatchError(
                    "successor freshness-policy hash differs from the buffer"
                )
        if transition.next_state_features is not None:
            if (
                transition.next_state_features.empirical_scaling_sha256
                != self.empirical_scaling_sha256
            ):
                raise BindingMismatchError(
                    "successor scaling hash differs from the buffer"
                )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.revalidate()
        return {
            "calibration_evidence_sha256": self.calibration_evidence_sha256,
            "catalog_sha256": self.catalog_sha256,
            "empirical_scaling_sha256": self.empirical_scaling_sha256,
            "evidence_eligibility": self.evidence_eligibility,
            "feature_schema_sha256": self.feature_schema_sha256,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "gamma": self.gamma,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "queue_kernel_evidence_sha256": self.queue_kernel_evidence_sha256,
            "reward_schema_sha256": self.reward_schema_sha256,
            "schema_id": self.schema_id,
            "schema_sha256": self.schema_sha256,
            "schema_version": self.schema_version,
            "training_evidence_class": self.training_evidence_class,
            "transition_schema_sha256": self.transition_schema_sha256,
            "verifier_manifest_sha256": self.verifier_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class _StoredRow:
    digest: str
    logical_key: Tuple[str, str, int]
    episode_boundary: EpisodeBoundary
    executed_action_sha256: str
    state_values: Tuple[float, ...]
    next_state_values: Optional[Tuple[float, ...]]
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    discount: float
    terminated: bool
    truncated: bool

    @property
    def has_next_state(self) -> bool:
        return self.next_state_values is not None

    @property
    def bootstrap(self) -> bool:
        return (
            self.has_next_state
            and self.episode_boundary is EpisodeBoundary.CONTINUES
        )

    def audit_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "decision_seq": self.logical_key[2],
                "discount": self.discount,
                "episode_boundary": self.episode_boundary.value,
                "executed_action_sha256": self.executed_action_sha256,
                "session_uuid": self.logical_key[0],
                "transition_sha256": self.digest,
                "ue_id": self.logical_key[1],
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class ReplayTensorBatchV1:
    """Private CPU tensors with clone-returning public accessors."""

    _state: Tensor
    _next_state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _reward: Tensor
    _duration: Tensor
    _discount: Tensor
    _has_next_state: Tensor
    _bootstrap: Tensor
    _terminated: Tensor
    _truncated: Tensor
    _binding: ReplayBindingV1
    _audit: Tuple[Mapping[str, Any], ...]

    _FLOAT_FIELDS = ("_state", "_next_state", "_reward", "_discount")
    _INT_FIELDS = ("_mode_id", "_q_e4", "_duration")
    _BOOL_FIELDS = (
        "_has_next_state",
        "_bootstrap",
        "_terminated",
        "_truncated",
    )
    _TENSOR_FIELDS = _FLOAT_FIELDS + _INT_FIELDS + _BOOL_FIELDS

    def __post_init__(self) -> None:
        if type(self._binding) is not ReplayBindingV1:
            raise ReplayBufferError("batch binding must be exact ReplayBindingV1")
        self._binding.revalidate()
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise ReplayBufferError(f"{name} must be a torch.Tensor")
            if value.device.type != "cpu":
                raise ReplayBufferError(f"{name} must be a CPU tensor")
            object.__setattr__(self, name, value.detach().clone())
        for name in self._FLOAT_FIELDS:
            if getattr(self, name).dtype is not DEFAULT_FLOAT_DTYPE:
                raise ReplayBufferError(f"{name} must use torch.float32")
        for name in self._INT_FIELDS:
            if getattr(self, name).dtype is not torch.int64:
                raise ReplayBufferError(f"{name} must use torch.int64")
        for name in self._BOOL_FIELDS:
            if getattr(self, name).dtype is not torch.bool:
                raise ReplayBufferError(f"{name} must use torch.bool")
        if self._state.ndim != 2 or self._state.shape[1] != POLICY_FEATURE_COUNT:
            raise ReplayBufferError("state must have shape [B, 21]")
        if self._next_state.shape != self._state.shape:
            raise ReplayBufferError("next_state shape must equal state shape")
        size = int(self._state.shape[0])
        for name in self._TENSOR_FIELDS[2:]:
            if getattr(self, name).shape != (size,):
                raise ReplayBufferError(f"{name} must have shape [B]")
        if len(self._audit) != size:
            raise ReplayBufferError("audit row count must equal tensor batch size")
        object.__setattr__(
            self,
            "_audit",
            tuple(MappingProxyType(dict(item)) for item in self._audit),
        )

    @property
    def state(self) -> Tensor:
        return self._state.clone()

    @property
    def next_state(self) -> Tensor:
        return self._next_state.clone()

    @property
    def mode_id(self) -> Tensor:
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        return self._q_e4.clone()

    @property
    def q_normalized_executed(self) -> Tensor:
        return self._q_e4.to(DEFAULT_FLOAT_DTYPE) / float(Q_CRITIC_NORMALIZER)

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    @property
    def duration(self) -> Tensor:
        return self._duration.clone()

    @property
    def has_next_state(self) -> Tensor:
        return self._has_next_state.clone()

    @property
    def bootstrap(self) -> Tensor:
        return self._bootstrap.clone()

    @property
    def terminated(self) -> Tensor:
        return self._terminated.clone()

    @property
    def truncated(self) -> Tensor:
        return self._truncated.clone()

    @property
    def batch_size(self) -> int:
        return int(self._state.shape[0])

    @property
    def float_dtype(self) -> torch.dtype:
        return DEFAULT_FLOAT_DTYPE

    @property
    def gamma(self) -> float:
        return self._binding.gamma

    @property
    def binding(self) -> ReplayBindingV1:
        self._binding.revalidate()
        return self._binding

    @property
    def audit(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(item)) for item in self._audit)

    def discount(self) -> Tensor:
        """Return caller-derived discounts; never recompute from gamma/duration."""
        return self._discount.clone()

    def to_canonical_metadata(self) -> Dict[str, Any]:
        return {
            "audit": [dict(item) for item in self._audit],
            "batch_size": self.batch_size,
            "binding": self.binding.to_canonical_dict(),
            "float_dtype": str(DEFAULT_FLOAT_DTYPE),
            "schema": REPLAY_SCHEMA_ID,
        }


class _ReplayBufferCore:
    """Shared bounded FIFO mechanics; eligibility is set by subclasses."""

    def __init__(
        self,
        capacity: int,
        binding: ReplayBindingV1,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ReplayBufferError("capacity must be a positive exact int")
        if type(binding) is not ReplayBindingV1:
            raise ReplayBufferError("binding must be exact ReplayBindingV1")
        binding.revalidate()
        assert_policy_feature_schema()
        self._capacity = capacity
        self._binding = binding
        self._rows: Deque[_StoredRow] = deque()
        self._seen_digests: set[str] = set()
        self._seen_identities: Dict[Tuple[str, str, int], str] = {}
        self._accepted_count = 0
        self._evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def binding(self) -> ReplayBindingV1:
        self._binding.revalidate()
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

    def resident_transition_digests(self) -> Tuple[str, ...]:
        """Resident canonical digests, oldest first; records are not exported."""
        return tuple(row.digest for row in self._rows)

    def insert(self, transition: Any) -> None:
        row = self._validate(transition)
        # All checks are complete.  Mutation starts here.
        self._seen_digests.add(row.digest)
        self._seen_identities[row.logical_key] = row.digest
        self._rows.append(row)
        self._accepted_count += 1
        while len(self._rows) > self._capacity:
            self._rows.popleft()
            self._evicted_count += 1

    def _validate(self, transition: Any) -> _StoredRow:
        if type(transition) is not SemiMarkovTransitionV2:
            raise TransitionRejectedError(
                "Run-4 replay accepts only exact attested "
                f"SemiMarkovTransitionV2, got {type(transition).__name__}"
            )
        # Re-evaluate the attestation against the record's current canonical
        # fields.  Post-attestation object.__setattr__ tampering fails here.
        transition.require_attested()
        self._binding.assert_transition(transition)

        transition.state.require_guarded()
        transition.state_features.require_attested()
        if (
            transition.state_features.guarded_state_sha256
            != transition.state.canonical_sha256()
        ):
            raise TransitionRejectedError("state features are not bound to state")
        state_values = transition.state_features.as_tuple()
        self._check_features(state_values, "state_features")

        boundary = transition.episode_boundary
        if boundary is EpisodeBoundary.CONTINUES:
            if transition.next_state is None or transition.next_state_features is None:
                raise TransitionRejectedError(
                    "CONTINUES requires the real successor and its features"
                )
            transition.next_state.require_guarded()
            transition.next_state_features.require_attested()
            if (
                transition.next_state_features.guarded_state_sha256
                != transition.next_state.canonical_sha256()
            ):
                raise TransitionRejectedError(
                    "successor features are not bound to successor state"
                )
            # This independent replay check prevents the Run-2/3 failure mode
            # where a successor omitted the actual prior action outcome.
            expected_previous = PreviousOutcomeV1.from_resolution(
                transition.reward_resolution
            )
            actual_previous = transition.next_state.state.previous
            if (
                actual_previous is None
                or actual_previous.canonical_sha256()
                != expected_previous.canonical_sha256()
            ):
                raise TransitionRejectedError(
                    "successor previous outcome is not exactly the current "
                    "transition reward resolution"
                )
            next_values: Optional[Tuple[float, ...]] = (
                transition.next_state_features.as_tuple()
            )
            self._check_features(next_values, "next_state_features")
        elif boundary in (EpisodeBoundary.TERMINATED, EpisodeBoundary.TRUNCATED):
            if transition.next_state is not None or transition.next_state_features is not None:
                raise TransitionRejectedError(
                    "TERMINATED/TRUNCATED must not fabricate a successor"
                )
            next_values = None
        else:  # pragma: no cover - exact enum enforced by contract
            raise TransitionRejectedError("unknown episode boundary")

        action = transition.action
        action.require_reconciled()
        mode_id = int(action.mode_id)
        q_e4 = int(action.q_e4)
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise TransitionRejectedError("executed mode_id is out of range")
        if not Q_E4_MIN <= q_e4 <= Q_E4_MAX:
            raise TransitionRejectedError("executed q_e4 is out of range")

        reward = float(transition.reward)
        if not math.isfinite(reward):
            raise TransitionRejectedError("reward must be finite")
        duration = int(transition.duration)
        if duration < 1 or duration != transition.hold.duration:
            raise TransitionRejectedError(
                "duration must equal the exact positive hold duration"
            )
        # The contract already derived and attested this field.  Do not form
        # gamma ** duration here or anywhere in tensorization.
        discount = float(transition.discount)
        if not math.isfinite(discount) or not 0.0 < discount <= 1.0:
            raise TransitionRejectedError("discount must lie in (0, 1]")

        _assert_finite_float32(state_values, "state_features")
        if next_values is not None:
            _assert_finite_float32(next_values, "next_state_features")
        _assert_finite_float32((reward,), "reward")
        _assert_finite_float32((discount,), "discount")
        _assert_finite_float32((float(transition.gamma),), "gamma")
        _assert_finite_float32(
            (q_e4 / float(Q_CRITIC_NORMALIZER),), "q_normalized_executed"
        )
        bootstrap = next_values is not None and boundary is EpisodeBoundary.CONTINUES
        if (
            bootstrap
            and discount > 0.0
            and float(torch.tensor(discount, dtype=DEFAULT_FLOAT_DTYPE)) == 0.0
        ):
            raise NonFiniteInReplayDtypeError(
                "a bootstrap-eligible discount underflows to zero in float32"
            )

        digest = transition.canonical_sha256()
        identity = transition.state.state.identity
        logical_key = (identity.session_uuid, identity.ue_id, identity.decision_seq)
        known = self._seen_identities.get(logical_key)
        if known is not None and known != digest:
            raise IdentityConflictError(
                f"logical decision {logical_key!r} was already recorded as "
                f"{known}, not {digest}"
            )
        if digest in self._seen_digests:
            raise DuplicateTransitionError(
                f"transition {digest} was already seen by this buffer"
            )

        return _StoredRow(
            digest=digest,
            logical_key=logical_key,
            episode_boundary=boundary,
            executed_action_sha256=action.canonical_sha256(),
            state_values=tuple(state_values),
            next_state_values=(None if next_values is None else tuple(next_values)),
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            duration=duration,
            discount=discount,
            terminated=boundary is EpisodeBoundary.TERMINATED,
            truncated=boundary is EpisodeBoundary.TRUNCATED,
        )

    @staticmethod
    def _check_features(values: Sequence[float], name: str) -> None:
        if len(values) != POLICY_FEATURE_COUNT:
            raise TransitionRejectedError(
                f"{name} must contain {POLICY_FEATURE_COUNT} values"
            )
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                raise TransitionRejectedError(
                    f"{name}[{index}] ({POLICY_FEATURE_ORDER[index]}) is not finite"
                )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ReplayTensorBatchV1:
        if not isinstance(generator, torch.Generator):
            raise ReplaySamplingError(
                "sampling requires an explicit local CPU torch.Generator"
            )
        if generator is torch.default_generator:
            raise ReplaySamplingError("torch.default_generator is forbidden")
        if generator.device.type != "cpu":
            raise ReplaySamplingError("sampling requires a CPU generator")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ReplaySamplingError("batch_size must be a positive exact int")
        if batch_size > len(self._rows):
            raise ReplaySamplingError(
                f"cannot sample {batch_size} rows from {len(self._rows)}"
            )
        permutation = torch.randperm(
            len(self._rows), generator=generator, device="cpu"
        )
        rows = [self._rows[int(index)] for index in permutation[:batch_size]]
        return self._build_batch(rows)

    def _build_batch(self, rows: Sequence[_StoredRow]) -> ReplayTensorBatchV1:
        state = torch.tensor(
            [row.state_values for row in rows], dtype=DEFAULT_FLOAT_DTYPE
        )
        next_state = torch.zeros(
            (len(rows), POLICY_FEATURE_COUNT), dtype=DEFAULT_FLOAT_DTYPE
        )
        for index, row in enumerate(rows):
            if row.next_state_values is not None:
                next_state[index] = torch.tensor(
                    row.next_state_values, dtype=DEFAULT_FLOAT_DTYPE
                )
        return ReplayTensorBatchV1(
            _state=state,
            _next_state=next_state,
            _mode_id=torch.tensor(
                [row.mode_id for row in rows], dtype=torch.int64
            ),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor(
                [row.reward for row in rows], dtype=DEFAULT_FLOAT_DTYPE
            ),
            _duration=torch.tensor(
                [row.duration for row in rows], dtype=torch.int64
            ),
            _discount=torch.tensor(
                [row.discount for row in rows], dtype=DEFAULT_FLOAT_DTYPE
            ),
            _has_next_state=torch.tensor(
                [row.has_next_state for row in rows], dtype=torch.bool
            ),
            _bootstrap=torch.tensor(
                [row.bootstrap for row in rows], dtype=torch.bool
            ),
            _terminated=torch.tensor(
                [row.terminated for row in rows], dtype=torch.bool
            ),
            _truncated=torch.tensor(
                [row.truncated for row in rows], dtype=torch.bool
            ),
            _binding=self._binding,
            _audit=tuple(row.audit_record() for row in rows),
        )


class ReplayBufferV1(_ReplayBufferCore):
    """Production replay; construction requires verifier-attested evidence."""

    def __init__(self, capacity: int, binding: ReplayBindingV1) -> None:
        if type(binding) is not ReplayBindingV1:
            raise ReplayBufferError("binding must be exact ReplayBindingV1")
        binding.require_training_eligible()
        super().__init__(capacity, binding)


class _TestOnlyReplayBufferV1(_ReplayBufferCore):
    """Private mechanics harness whose batches remain labelled TEST_ONLY."""

    def __init__(self, capacity: int, binding: ReplayBindingV1) -> None:
        if type(binding) is not ReplayBindingV1:
            raise ReplayBufferError("binding must be exact ReplayBindingV1")
        binding.revalidate()
        if binding.evidence_eligibility != _EvidenceEligibility.TEST_ONLY_MECHANICS.value:
            raise EvidenceEligibilityError(
                "test harness requires explicit TEST_ONLY_MECHANICS binding"
            )
        super().__init__(capacity, binding)
