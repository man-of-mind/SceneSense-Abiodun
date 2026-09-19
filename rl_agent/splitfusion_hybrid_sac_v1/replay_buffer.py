"""Bounded deterministic replay boundary for Hybrid SAC (Phase 4b.2, part B).

``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.  This module is the *production
boundary* between the frozen causal-transition contract and the numerical
learning path.  It stores transitions and turns samples of them into tensors.
It does not train, does not define a reward, and does not decide eligibility --
every one of those judgements is read from the frozen contract.

What may enter
--------------

Exactly one thing: a :class:`ReplayTransitionV1` that

1. is of that exact type (no subclass, no duck type, no adapter);
2. survives :meth:`ReplayTransitionV1.revalidate`, which re-derives the whole
   outcome from its own frozen sources and compares it;
3. is independently ``LearningEligibility.ELIGIBLE`` **and**
   ``learning_eligible``; and
4. carries a finite scalar reward.

Those four are checked in that order, and all of them -- along with every
binding, duplicate and identity check -- complete *before* the buffer mutates
any of its own state.  A rejected insertion leaves the buffer bit-identical.

What may never enter
--------------------

The 288-cell measured aggregates and every synthetic fixture record are
**not causal transitions** and are refused by the type gate:
``MeasuredAnchorRecord``, ``ActionQualityAnchor``, ``NetworkProfileOutcome``,
``SyntheticFixtureValueRecord``, ``SyntheticRunReport`` and
``SyntheticPolicyObservation``.  A campaign-cell aggregate has no
``session_uuid``, no ``decision_seq``, no per-decision radio state and no
single terminal classification; adapting one into this buffer would be
fabricating a transition, not reusing evidence.  Nothing here interpolates the
continuous-``q`` surface between the six measured anchors.

Censored and excluded transitions are likewise refused.  A
``FEEDBACK_TIMEOUT`` is censored pending a reviewed reconciliation carrier and
an ``INFRASTRUCTURE_FAULT_EXCLUDED`` is an instrument fault; neither is an
action the policy should be scored for.  This module does not weaken that
gate, and in particular does not weaken the fail-closed exact-positive reward
path: ``REWARD_FINAL_EXACT`` remains unreachable until an authenticated
per-frame quality carrier (protocol v2) exists.

On the value of an eligible failure reward
------------------------------------------

The registered value is

    scalar_reward = r_registered_failure - switch_penalty.total

Both terms come from the reward spec and the two exact executed actions.  It
is **not** a constant: ``r_registered_failure`` is a spec parameter, and the
switch penalty is zero only when the state carries no previous action (an
episode-genesis decision).  This module never assumes a particular numeric
value -- it requires only that the reward be finite.

Duplicate and identity policy
-----------------------------

Two indexes are kept, and neither is pruned by eviction:

* a lifetime set of transition canonical digests; and
* a lifetime map from the logical key
  ``(session_uuid, controller_lineage_uuid, decision_seq)`` to the digest first
  seen under that key.

Same key and same digest is a **duplicate**.  Same key and a *different* digest
is an **identity conflict**: one decision of one controller episode cannot have
two different transition records, and silently keeping both would put the same
decision into a batch twice under contradictory outcomes.

Both indexes are in-memory and process-local.  They are **not durable dedup**:
a fresh buffer, a new process or a restart forgets everything, so this protects
one run's batch statistics, not a persisted dataset.

Mutability
----------

``ReplayTensorBatchV1`` is a frozen dataclass, but a frozen dataclass only
freezes *attribute binding* -- it cannot make a ``torch.Tensor``'s contents
read-only.  The batch therefore holds private tensors and every public
accessor returns a fresh ``clone()``.  Mutating what an accessor returns
affects only that copy.  The narrow claim is: no caller can reach the buffer's
own storage through this object.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .action_contract import CATALOG_SHA256, EXPECTED_MODE_COUNT, Q_E4_MAX, Q_E4_MIN
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    LearningEligibility,
    ReplayTransitionV1,
    assert_policy_features_exclude_forbidden_fields,
)
from .transaction_identity import MINIMUM_HOLD_TENSORS

__all__ = [
    "BindingMismatchError",
    "DEFAULT_FLOAT_DTYPE",
    "NonFiniteInReplayDtypeError",
    "SUPPORTED_FLOAT_DTYPES",
    "DuplicateTransitionError",
    "IdentityConflictError",
    "IneligibleTransitionError",
    "PHASE_LABEL",
    "Q_CRITIC_NORMALIZER",
    "REPLAY_BUFFER_SCHEMA_ID",
    "ReplayBindingV1",
    "ReplayBufferError",
    "ReplayBufferV1",
    "ReplaySamplingError",
    "ReplayTensorBatchV1",
    "TransitionRejectedError",
]


# --------------------------------------------------------------------------- #
# Frozen registered constants
# --------------------------------------------------------------------------- #

PHASE_LABEL = "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY"

REPLAY_BUFFER_SCHEMA_ID = "splitfusion_hybrid_sac_replay_buffer_v1"

#: Floating dtype of replay tensors, matching ``HybridSacModelConfig``.
DEFAULT_FLOAT_DTYPE: torch.dtype = torch.float32

#: The only floating dtypes this buffer supports.
#:
#: Restricted deliberately.  ``float16`` and ``bfloat16`` have roughly 3 and 2
#: decimal digits of mantissa and an exponent range far narrower than the
#: reward and discount values this buffer carries, and several CPU reductions
#: are not implemented for them.  Rather than silently degrade, an unsupported
#: dtype is refused until it is explicitly proven to support every required
#: CPU operation.
SUPPORTED_FLOAT_DTYPES: Tuple[torch.dtype, ...] = (torch.float32, torch.float64)

#: The critic's continuous action input is ``q_e4 / 9800`` in ``[0, 1]``.
Q_CRITIC_NORMALIZER: int = Q_E4_MAX


# --------------------------------------------------------------------------- #
# Exceptions: fail closed, never normalize
# --------------------------------------------------------------------------- #


class ReplayBufferError(Exception):
    """Base class for every replay-boundary failure."""


class TransitionRejectedError(ReplayBufferError):
    """A candidate record may not enter the production replay store."""


class IneligibleTransitionError(TransitionRejectedError):
    """The transition is censored, excluded or carries no finite reward."""


class NonFiniteInReplayDtypeError(TransitionRejectedError):
    """A contract-finite value stops being finite in the replay dtype.

    The reward contract requires a *finite* float, which ``float64`` honours
    over the full IEEE double range.  The replay tensors are ``float32`` by
    default, whose maximum magnitude is about ``3.4e38``.  A perfectly legal
    reward of ``-1e300`` therefore becomes ``-inf`` the moment it is stored,
    and would poison every loss computed from the batch.  The conversion is
    proved *before* insertion rather than discovered during training.
    """


class DuplicateTransitionError(TransitionRejectedError):
    """This exact transition digest has already been seen in this process."""


class IdentityConflictError(TransitionRejectedError):
    """One logical decision was offered under two different transition records."""


class BindingMismatchError(TransitionRejectedError):
    """The transition does not share the buffer's frozen homogeneous bindings."""


class ReplaySamplingError(ReplayBufferError):
    """A sampling request is malformed or cannot be satisfied."""


def _assert_finite_in_dtype(
    values: Sequence[float], name: str, dtype: torch.dtype
) -> None:
    """Prove every value is still finite after conversion to ``dtype``.

    Contract validity is checked in Python ``float`` (IEEE double).  Storage
    happens in the configured replay dtype, which may be narrower.  A value
    that overflows on conversion is refused here, not silently turned into an
    infinity inside a tensor.

    Raises:
        NonFiniteInReplayDtypeError: naming the first offending index.
    """
    converted = torch.tensor(tuple(values), dtype=dtype)
    finite = torch.isfinite(converted)
    if bool(finite.all()):
        return
    index = int((~finite).nonzero()[0])
    raise NonFiniteInReplayDtypeError(
        f"{name}[{index}] is finite as a contract float ({values[index]!r}) "
        f"but converts to {converted[index].item()!r} in replay dtype "
        f"{dtype}; storing it would put a non-finite value into every batch "
        f"drawn from this buffer"
    )


# --------------------------------------------------------------------------- #
# Homogeneous bindings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReplayBindingV1:
    """The preprocessing and reward identity every stored row must share.

    A batch mixing two reward specifications, two normalizations or two
    discounts is not one learning problem, so the first accepted transition
    freezes these and every later one must match exactly.

    Deliberately **absent** from this binding, because SAC is off-policy and
    must be able to learn across them: ``session_uuid``,
    ``actor_version_sha256``, ``decision_seq``, tensor/frame identifiers, the
    executed mode and ``q_e4``, and the realized hold duration.
    """

    reward_spec_sha256: str
    state_normalization_spec_sha256: str
    freshness_policy_sha256: str
    gamma_per_tensor: float
    schema_id: str
    schema_version: int
    schema_sha256: str
    catalog_sha256: str
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int

    @classmethod
    def from_transition(cls, transition: ReplayTransitionV1) -> "ReplayBindingV1":
        """Derive the binding a transition declares."""
        return cls(
            reward_spec_sha256=transition.reward_spec_sha256,
            state_normalization_spec_sha256=(
                transition.state_normalization_spec_sha256
            ),
            freshness_policy_sha256=transition.freshness_policy_sha256,
            gamma_per_tensor=float(transition.gamma_per_tensor),
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            schema_sha256=SCHEMA_SHA256,
            catalog_sha256=transition.executed_action.catalog_sha256,
            policy_feature_order=tuple(POLICY_FEATURE_ORDER),
            policy_feature_count=POLICY_FEATURE_COUNT,
        )

    def assert_matches(self, other: "ReplayBindingV1") -> None:
        """Fail closed on any binding difference, naming the exact field.

        Raises:
            BindingMismatchError: on the first differing field.
        """
        for field_name in (
            "reward_spec_sha256",
            "state_normalization_spec_sha256",
            "freshness_policy_sha256",
            "gamma_per_tensor",
            "schema_id",
            "schema_version",
            "schema_sha256",
            "catalog_sha256",
            "policy_feature_order",
            "policy_feature_count",
        ):
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            if mine != theirs:
                raise BindingMismatchError(
                    f"the buffer is bound to {field_name}={mine!r} but this "
                    f"transition declares {theirs!r}; a replay batch may not "
                    f"mix two preprocessing or reward identities"
                )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form."""
        return {
            "catalog_sha256": self.catalog_sha256,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "gamma_per_tensor": self.gamma_per_tensor,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "reward_spec_sha256": self.reward_spec_sha256,
            "schema_id": self.schema_id,
            "schema_sha256": self.schema_sha256,
            "schema_version": self.schema_version,
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
        }


# --------------------------------------------------------------------------- #
# Stored row
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _StoredRow:
    """One accepted transition plus the scalars the tensor batch needs.

    The originating transition is retained so the buffer can always be audited
    back to the exact contract record it accepted.
    """

    transition: ReplayTransitionV1
    digest: str
    logical_key: Tuple[str, str, int]
    state_values: Tuple[float, ...]
    next_state_values: Optional[Tuple[float, ...]]
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    terminated: bool
    truncated: bool

    @property
    def has_next_state(self) -> bool:
        """True when an exact successor observation was recorded."""
        return self.next_state_values is not None

    @property
    def bootstrap(self) -> bool:
        """Bootstrap iff an exact next state exists and this is not terminal.

        A truncation *may* bootstrap, but only when it carries an exact next
        state.  A true terminal never bootstraps even when a successor
        observation happens to exist: the value beyond a terminal is zero by
        definition, not merely unobserved.
        """
        return self.has_next_state and not self.terminated

    def audit_record(self) -> Mapping[str, Any]:
        """Identity metadata for auditing only; never a learning input."""
        transition = self.transition
        return MappingProxyType(
            {
                "transition_sha256": self.digest,
                "session_uuid": transition.session_uuid,
                "controller_lineage_uuid": (
                    transition.completed_ticket.controller_lineage_uuid
                ),
                "decision_seq": transition.decision_seq,
                "reward_tensor_seq": transition.reward_tensor_seq,
                "reward_carla_frame_id": transition.reward_carla_frame_id,
                "executed_action_sha256": transition.executed_action_sha256,
                "actor_version_sha256": (
                    transition.policy_trace.actor_version_sha256
                ),
                "terminal_class": transition.terminal_class.value,
                "eligibility": transition.eligibility.value,
                "discount_multiplier": transition.discount_multiplier,
            }
        )


# --------------------------------------------------------------------------- #
# Tensor batch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, eq=False)
class ReplayTensorBatchV1:
    """A tensor view of a uniform sample of stored transitions.

    Constructor inputs are defensively cloned and detached, so the batch never
    shares storage with the buffer or with whatever the caller passed in.
    Every supported public accessor then returns a fresh ``clone()``, so two
    accesses never alias and mutating what an accessor returns cannot reach
    the batch.

    The guarantee is exactly that and no more.  A frozen dataclass prevents
    rebinding an attribute; it does not make tensor contents read-only, and
    Python has no private state.  A caller that deliberately reaches the
    underscore-prefixed fields, or uses ``object.__setattr__``, can still
    mutate this object.  The claim is protection against accidental aliasing
    and ordinary mutation, **not** a sandbox against a determined caller.
    """

    _state: Tensor
    _next_state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _reward: Tensor
    _duration: Tensor
    _has_next_state: Tensor
    _bootstrap: Tensor
    _terminated: Tensor
    _truncated: Tensor
    binding: ReplayBindingV1
    audit: Tuple[Mapping[str, Any], ...]
    float_dtype: torch.dtype

    #: Tensor fields cloned on construction so the batch owns its storage.
    _TENSOR_FIELDS = (
        "_state",
        "_next_state",
        "_mode_id",
        "_q_e4",
        "_reward",
        "_duration",
        "_has_next_state",
        "_bootstrap",
        "_terminated",
        "_truncated",
    )

    def __post_init__(self) -> None:
        # Defensive copy of every constructor input: the caller keeps no
        # handle into this batch, and the batch keeps no handle into the
        # buffer's own row storage.
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise ReplayBufferError(
                    f"{name} must be a torch.Tensor, got "
                    f"{type(value).__name__}"
                )
            object.__setattr__(self, name, value.detach().clone())

    # -- learning tensors (clone-returning) -------------------------------- #

    @property
    def state(self) -> Tensor:
        """``[B, 31]`` float policy state, in the frozen feature order."""
        return self._state.clone()

    @property
    def next_state(self) -> Tensor:
        """``[B, 31]`` float successor state; zero-filled where none exists.

        A real successor vector is preserved whenever the transition recorded
        one -- including for a terminated transition, whose successor is real
        evidence even though it must not be bootstrapped.  Only a genuinely
        absent next state is zero-filled, and those rows are excluded by
        :attr:`bootstrap`.  The zero row is a sentinel, never a state.
        """
        return self._next_state.clone()

    @property
    def mode_id(self) -> Tensor:
        """``[B]`` int64 executed joint mode, in ``[0, 11]``."""
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        """``[B]`` int64 executed wire quality, in ``[0, 9800]``."""
        return self._q_e4.clone()

    @property
    def q_normalized_executed(self) -> Tensor:
        """``[B]`` float ``q_e4 / 9800``: the executed critic action.

        Derived from the retained integer ``q_e4`` on every access, so it is
        always the executed value and never a sampled request.
        """
        return self._q_e4.to(self.float_dtype) / float(Q_CRITIC_NORMALIZER)

    @property
    def reward(self) -> Tensor:
        """``[B]`` float scalar reward."""
        return self._reward.clone()

    @property
    def duration(self) -> Tensor:
        """``[B]`` int64 realized hold duration ``d``, each ``>= 2``."""
        return self._duration.clone()

    @property
    def has_next_state(self) -> Tensor:
        """``[B]`` bool: an exact successor observation exists."""
        return self._has_next_state.clone()

    @property
    def bootstrap(self) -> Tensor:
        """``[B]`` bool: ``has_next_state AND NOT terminated``."""
        return self._bootstrap.clone()

    @property
    def terminated(self) -> Tensor:
        """``[B]`` bool true terminal."""
        return self._terminated.clone()

    @property
    def truncated(self) -> Tensor:
        """``[B]`` bool truncation, kept separate from termination."""
        return self._truncated.clone()

    @property
    def batch_size(self) -> int:
        """Number of rows."""
        return int(self._state.shape[0])

    @property
    def gamma_per_tensor(self) -> float:
        """The uniform discount every stored row agreed on."""
        return self.binding.gamma_per_tensor

    def discount(self) -> Tensor:
        """``[B]`` float ``gamma ** d``, the SMDP discount per row."""
        return torch.pow(
            torch.tensor(self.gamma_per_tensor, dtype=self.float_dtype),
            self._duration.to(self.float_dtype),
        )

    def to_canonical_metadata(self) -> Dict[str, Any]:
        """Deterministic non-tensor description of this batch."""
        return {
            "audit": [dict(record) for record in self.audit],
            "batch_size": self.batch_size,
            "binding": self.binding.to_canonical_dict(),
            "evidence_class": PHASE_LABEL,
            "float_dtype": str(self.float_dtype),
            "schema": REPLAY_BUFFER_SCHEMA_ID,
        }


# --------------------------------------------------------------------------- #
# The buffer
# --------------------------------------------------------------------------- #


class ReplayBufferV1:
    """A bounded, deterministic, in-memory replay store for exact transitions.

    Not thread-safe, matching the single-event-loop controller contract.
    """

    def __init__(
        self,
        capacity: int,
        float_dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ReplayBufferError(
                f"capacity must be a positive integer, got "
                f"{type(capacity).__name__}: {capacity!r}"
            )
        if capacity < 1:
            raise ReplayBufferError(
                f"capacity must be a positive integer, got {capacity}"
            )
        if float_dtype not in SUPPORTED_FLOAT_DTYPES:
            raise ReplayBufferError(
                f"float_dtype must be one of "
                f"{[str(d) for d in SUPPORTED_FLOAT_DTYPES]}, got "
                f"{float_dtype}.  A narrower dtype is refused until it is "
                f"explicitly proven to support every required CPU operation "
                f"at the precision the reward and discount need"
            )
        # The frozen feature order is re-proved here rather than trusted, so a
        # buffer can never be built on an order that gained a forbidden field.
        assert_policy_features_exclude_forbidden_fields()

        self._capacity = int(capacity)
        self._float_dtype = float_dtype
        self._rows: Deque[_StoredRow] = deque()
        self._binding: Optional[ReplayBindingV1] = None
        #: Lifetime digest set; never pruned on eviction.
        self._seen_digests: set = set()
        #: Lifetime logical-key -> first digest map; never pruned on eviction.
        self._seen_identities: Dict[Tuple[str, str, int], str] = {}
        self._evicted_count = 0
        self._accepted_count = 0

    # -- introspection ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def capacity(self) -> int:
        """Maximum resident rows."""
        return self._capacity

    @property
    def float_dtype(self) -> torch.dtype:
        """Floating dtype of emitted tensors."""
        return self._float_dtype

    @property
    def binding(self) -> Optional[ReplayBindingV1]:
        """The frozen homogeneous binding, or ``None`` while empty."""
        return self._binding

    @property
    def seen_digest_count(self) -> int:
        """Lifetime distinct transition digests, including evicted ones."""
        return len(self._seen_digests)

    @property
    def seen_identity_count(self) -> int:
        """Lifetime distinct logical decision keys, including evicted ones."""
        return len(self._seen_identities)

    @property
    def evicted_count(self) -> int:
        """Rows dropped by capacity pressure."""
        return self._evicted_count

    @property
    def accepted_count(self) -> int:
        """Lifetime accepted insertions."""
        return self._accepted_count

    def stored_transitions(self) -> Tuple[ReplayTransitionV1, ...]:
        """Resident transitions, oldest first, for auditing."""
        return tuple(row.transition for row in self._rows)

    # -- insertion --------------------------------------------------------- #

    def insert(self, transition: Any) -> None:
        """Validate a transition completely, then store it.

        Every check below runs before any buffer state changes, so a rejected
        insertion leaves the buffer bit-identical.

        Raises:
            TransitionRejectedError: (or a subclass) on any contract,
                eligibility, binding, duplicate or identity failure.
        """
        row = self._validate(transition)
        binding = ReplayBindingV1.from_transition(row.transition)
        if self._binding is None:
            frozen_binding: ReplayBindingV1 = binding
        else:
            self._binding.assert_matches(binding)
            frozen_binding = self._binding

        # -- all validation complete; mutate from here --------------------- #
        self._binding = frozen_binding
        self._seen_digests.add(row.digest)
        self._seen_identities[row.logical_key] = row.digest
        self._rows.append(row)
        self._accepted_count += 1
        while len(self._rows) > self._capacity:
            self._rows.popleft()
            self._evicted_count += 1

    def _validate(self, transition: Any) -> _StoredRow:
        """Run every acceptance gate and return the row that would be stored."""
        # 1. Exact type.  A subclass or an adapter around an aggregate is not a
        #    causal transition, so identity of type is required, not isinstance.
        if type(transition) is not ReplayTransitionV1:
            raise TransitionRejectedError(
                f"the production replay store accepts only an exact "
                f"ReplayTransitionV1, got {type(transition).__name__}.  "
                f"Campaign-cell aggregates and synthetic fixture records are "
                f"not causal transitions and must never be adapted into replay"
            )

        # 2. Re-derive the whole outcome from frozen sources.  This also
        #    re-proves the attestation, so a mutated record fails here.
        transition.revalidate()

        # 3. Eligibility and a finite reward, required independently.
        if transition.eligibility is not LearningEligibility.ELIGIBLE:
            raise IneligibleTransitionError(
                f"transition eligibility is {transition.eligibility.value}, "
                f"not ELIGIBLE; a censored or excluded decision must not "
                f"contribute to learning"
            )
        if transition.learning_eligible is not True:
            raise IneligibleTransitionError(
                "transition.learning_eligible is not True"
            )
        reward = transition.scalar_reward
        if type(reward) is not float or not math.isfinite(reward):
            raise IneligibleTransitionError(
                f"an eligible transition must carry a finite float scalar "
                f"reward, got {reward!r}.  The registered eligible-failure "
                f"value is r_registered_failure - switch_penalty.total, which "
                f"is a spec-dependent quantity and not a fixed constant"
            )

        # 4. Re-prove that the stored features really are derived from the
        #    exact states they claim, for both the state and any next state.
        transition.state_features.assert_binds(transition.state)
        if transition.next_state is None:
            if transition.next_state_features is not None:
                raise TransitionRejectedError(
                    "next-state features exist without a next state"
                )
            next_values: Optional[Tuple[float, ...]] = None
        else:
            if transition.next_state_features is None:
                raise TransitionRejectedError(
                    "a next state was recorded without its bootstrap features"
                )
            transition.next_state_features.assert_binds(transition.next_state)
            next_values = tuple(transition.next_state_features.values)
            self._check_feature_width(next_values, "next_state_features")

        state_values = tuple(transition.state_features.values)
        self._check_feature_width(state_values, "state_features")

        # 5. Action, duration and discount consistency.
        action = transition.executed_action
        mode_id = int(action.mode_id)
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise TransitionRejectedError(
                f"executed mode_id {mode_id} is outside "
                f"[0, {EXPECTED_MODE_COUNT - 1}]"
            )
        q_e4 = int(action.q_e4)
        if not Q_E4_MIN <= q_e4 <= Q_E4_MAX:
            raise TransitionRejectedError(
                f"executed q_e4 {q_e4} is outside [{Q_E4_MIN}, {Q_E4_MAX}]"
            )
        duration = int(transition.hold_duration_tensors)
        if duration < MINIMUM_HOLD_TENSORS:
            raise TransitionRejectedError(
                f"hold duration {duration} is below the registered minimum "
                f"hold of {MINIMUM_HOLD_TENSORS} transmitted tensors"
            )
        gamma = float(transition.gamma_per_tensor)
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise TransitionRejectedError(
                f"gamma_per_tensor must lie in (0, 1], got {gamma}"
            )
        if transition.discount_multiplier != gamma ** duration:
            raise TransitionRejectedError(
                f"the record's discount_multiplier "
                f"{transition.discount_multiplier!r} is not "
                f"gamma_per_tensor ** duration ({gamma!r} ** {duration})"
            )

        # 6. Terminal bookkeeping, restated independently of the contract.
        if transition.terminated and transition.truncated:
            raise TransitionRejectedError(
                "a transition cannot be both terminated and truncated"
            )
        if next_values is None and not (
            transition.terminated or transition.truncated
        ):
            raise TransitionRejectedError(
                "a non-terminal transition requires an exact next state"
            )

        # 7. Every floating value must still be finite once converted to the
        #    configured replay dtype.  Contract validity is proved in IEEE
        #    double; storage may be narrower, and an overflow here would put a
        #    non-finite value into every batch drawn from this buffer.
        dtype = self._float_dtype
        _assert_finite_in_dtype([reward], "scalar_reward", dtype)
        _assert_finite_in_dtype(state_values, "state_features", dtype)
        if next_values is not None:
            _assert_finite_in_dtype(next_values, "next_state_features", dtype)
        discount = float(transition.discount_multiplier)
        _assert_finite_in_dtype([discount], "discount_multiplier", dtype)
        _assert_finite_in_dtype([gamma], "gamma_per_tensor", dtype)
        # The derived critic action q_e4/9800 is bounded to [0, 1] by the
        # integer range, but it is a floating replay field so it is proved
        # rather than assumed.
        _assert_finite_in_dtype(
            [q_e4 / float(Q_CRITIC_NORMALIZER)], "q_normalized_executed", dtype
        )
        # A nonzero discount that underflows to exactly zero is not an
        # overflow, but it silently erases the entire bootstrap term, so it is
        # refused on the same principle.
        if discount > 0.0 and float(torch.tensor(discount, dtype=dtype)) == 0.0:
            raise NonFiniteInReplayDtypeError(
                f"discount_multiplier {discount!r} underflows to exactly zero "
                f"in replay dtype {dtype}; the bootstrap term would be "
                f"silently erased"
            )

        digest = transition.canonical_sha256()
        logical_key = (
            transition.session_uuid,
            transition.completed_ticket.controller_lineage_uuid,
            int(transition.decision_seq),
        )

        # 8. Duplicate and identity conflict, both against lifetime indexes
        #    that eviction never prunes.
        known = self._seen_identities.get(logical_key)
        if known is not None and known != digest:
            raise IdentityConflictError(
                f"decision {logical_key[2]} of controller episode "
                f"{logical_key[1]} in session {logical_key[0]} was already "
                f"stored as transition {known}, but a different record "
                f"{digest} claims the same decision; one decision cannot have "
                f"two outcomes"
            )
        if digest in self._seen_digests:
            raise DuplicateTransitionError(
                f"transition {digest} has already been inserted in this "
                f"process.  Note this in-memory index is not durable dedup: a "
                f"new buffer or process forgets it"
            )

        return _StoredRow(
            transition=transition,
            digest=digest,
            logical_key=logical_key,
            state_values=state_values,
            next_state_values=next_values,
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            duration=duration,
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
        )

    @staticmethod
    def _check_feature_width(values: Sequence[float], name: str) -> None:
        """Require the frozen feature width and finite entries."""
        if len(values) != POLICY_FEATURE_COUNT:
            raise TransitionRejectedError(
                f"{name} must hold exactly {POLICY_FEATURE_COUNT} features, "
                f"got {len(values)}"
            )
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise TransitionRejectedError(
                    f"{name}[{index}] ({POLICY_FEATURE_ORDER[index]}) is not "
                    f"finite: {value!r}"
                )

    # -- sampling ---------------------------------------------------------- #

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ReplayTensorBatchV1:
        """Uniformly sample ``batch_size`` distinct rows into a tensor batch.

        Sampling is uniform **without replacement** and consumes only the
        supplied generator, so it cannot advance the target-q, actor-q,
        evaluation or global RNG streams.

        Args:
            batch_size: Number of distinct rows, ``1 <= batch_size <= len()``.
            generator: An explicit local CPU ``torch.Generator``.  Required;
                there is deliberately no default, because an implicit default
                would silently couple replay sampling to the global stream.
                ``torch.default_generator`` itself is refused for the same
                reason, and so is any non-CPU generator.

        Raises:
            ReplaySamplingError: on a malformed request or an empty buffer.
        """
        if not isinstance(generator, torch.Generator):
            raise ReplaySamplingError(
                f"sampling requires an explicit torch.Generator, got "
                f"{type(generator).__name__}; replay must not draw from the "
                f"global RNG stream"
            )
        # Passing the process-wide default generator would couple replay
        # sampling to the global stream by the back door: it is a real
        # torch.Generator, so the type check alone would admit it, and every
        # draw would then perturb target-q, actor-q and evaluation sampling.
        if generator is torch.default_generator:
            raise ReplaySamplingError(
                "sampling refuses torch.default_generator: replay must own a "
                "private stream, and drawing from the global generator would "
                "advance the same state the target-q, actor-q and evaluation "
                "streams depend on"
            )
        if generator.device.type != "cpu":
            raise ReplaySamplingError(
                f"sampling requires a CPU generator, got one on device "
                f"{generator.device}; this phase is CPU-only and a non-CPU "
                f"generator would not reproduce the same draw"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ReplaySamplingError(
                f"batch_size must be a positive integer, got "
                f"{type(batch_size).__name__}: {batch_size!r}"
            )
        if batch_size < 1:
            raise ReplaySamplingError(
                f"batch_size must be a positive integer, got {batch_size}"
            )
        if not self._rows:
            raise ReplaySamplingError("the replay buffer is empty")
        if batch_size > len(self._rows):
            raise ReplaySamplingError(
                f"cannot sample {batch_size} distinct rows without "
                f"replacement from {len(self._rows)} stored transitions"
            )
        if self._binding is None:  # pragma: no cover - implied by non-empty
            raise ReplaySamplingError("the replay buffer has no frozen binding")

        permutation = torch.randperm(
            len(self._rows), generator=generator, device="cpu"
        )
        chosen = [self._rows[int(index)] for index in permutation[:batch_size]]
        return self._build_batch(chosen, self._binding)

    def _build_batch(
        self, rows: Sequence[_StoredRow], binding: ReplayBindingV1
    ) -> ReplayTensorBatchV1:
        """Tensorize the chosen rows.  Absent next states become zero rows."""
        float_dtype = self._float_dtype
        state = torch.tensor(
            [row.state_values for row in rows], dtype=float_dtype
        )
        next_state = torch.zeros(
            (len(rows), POLICY_FEATURE_COUNT), dtype=float_dtype
        )
        for position, row in enumerate(rows):
            # A recorded successor is preserved verbatim even when the
            # transition terminated; only a genuinely absent one stays zero.
            if row.next_state_values is not None:
                next_state[position] = torch.tensor(
                    row.next_state_values, dtype=float_dtype
                )
        return ReplayTensorBatchV1(
            _state=state,
            _next_state=next_state,
            _mode_id=torch.tensor([row.mode_id for row in rows], dtype=torch.int64),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor([row.reward for row in rows], dtype=float_dtype),
            _duration=torch.tensor(
                [row.duration for row in rows], dtype=torch.int64
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
            binding=binding,
            audit=tuple(row.audit_record() for row in rows),
            float_dtype=float_dtype,
        )
