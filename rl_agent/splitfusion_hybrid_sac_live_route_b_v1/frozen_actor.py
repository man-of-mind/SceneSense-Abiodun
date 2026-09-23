"""Batch-1, CPU, deterministic adapter around the frozen Run-3 actor.

The live pilot evaluates a policy; it does not train one.  This module makes
that structural rather than merely intended:

* the actor is built with the same registered configuration Run-3 trained
  under -- float32 and the hash-bound ``MODELED_SMOKE_SUPPORT`` per-mode
  ``q`` intervals -- and the weights are loaded ``strict=True``;
* every parameter and buffer has ``requires_grad`` cleared, the module is put
  in ``eval()`` mode, and inference runs inside ``torch.inference_mode()``, so
  a gradient cannot be produced even by accident;
* no optimizer, replay buffer, target network or critic is constructed
  anywhere in this module;
* the decision is ``argmax`` over the 12 mode logits at that mode's
  conditional mean ``q``.  Nothing is sampled, so no RNG is consumed.

RNG neutrality is asserted, not assumed.  The Python, NumPy and Torch global
generator states are captured before the forward pass and compared afterwards;
a mismatch fails the decision rather than silently perturbing a co-resident
CARLA or collection process that shares the interpreter.

The adapter accepts a bare tuple of 31 floats.  It cannot accept an identifier,
a timestamp or a frame id because it accepts nothing else -- the policy tensor
is constructed here, from that tuple alone.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_run3_runner import (
    _hash_state,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    HybridSacModelConfig,
    ConditionalHybridActor,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
)

from . import pilot_contract as contract
from .checkpoint_loader import LoadedPilotActorWeightsV1

__all__ = [
    "FrozenActorError",
    "FrozenPilotActor",
    "PolicyProposalV1",
]


class FrozenActorError(RuntimeError):
    """The frozen actor was misconfigured, mutated, or given a bad input."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenActorError(message)


def _rng_fingerprint() -> Dict[str, Any]:
    """A comparable snapshot of every global generator this process owns."""
    fingerprint: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state().clone(),
    }
    try:  # NumPy is a hard dependency of the sensor path but not of this one.
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is present in this repo
        fingerprint["numpy"] = None
    else:
        fingerprint["numpy"] = np.random.get_state()
    return fingerprint


def _rng_unchanged(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    if before["python"] != after["python"]:
        return False
    if not torch.equal(before["torch"], after["torch"]):
        return False
    left, right = before["numpy"], after["numpy"]
    if left is None or right is None:
        return left is right
    if left[0] != right[0] or left[2:] != right[2:]:
        return False
    return bool((left[1] == right[1]).all())


# --------------------------------------------------------------------------- #
# Proposal record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PolicyProposalV1:
    """What the frozen actor proposed for one frame, and how it was produced.

    This is the *proposal*.  It is not yet an executable action: binding it to
    the catalog, the keep/drop counts and the execution bundle is
    :mod:`execution_identity`'s job, and it can still fail there.
    """

    mode_id: int
    q: float
    q_e4: int
    q_exec: float
    q_normalized_executed: float
    mode_logits: Tuple[float, ...]
    selected_logit: float
    inference_ns: int
    actor_state_sha256: str
    snapshot_content_sha256: str
    snapshot_file_sha256: str
    seed: int
    update: int
    support_q_e4_low: int
    support_q_e4_high: int
    modeled_smoke_support_sha256: str
    feature_count: int
    deterministic_rule: str
    pilot_label: str
    pilot_contract_sha256: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "actor_state_sha256": self.actor_state_sha256,
            "deterministic_rule": self.deterministic_rule,
            "feature_count": self.feature_count,
            "inference_ns": self.inference_ns,
            "mode_id": self.mode_id,
            "mode_logits": [float(value) for value in self.mode_logits],
            "modeled_smoke_support_sha256": self.modeled_smoke_support_sha256,
            "pilot_contract_sha256": self.pilot_contract_sha256,
            "pilot_label": self.pilot_label,
            "q": self.q,
            "q_e4": self.q_e4,
            "q_exec": self.q_exec,
            "q_normalized_executed": self.q_normalized_executed,
            "record": "splitfusion.live_route_b_pilot_policy_proposal.v1",
            "seed": self.seed,
            "selected_logit": self.selected_logit,
            "snapshot_content_sha256": self.snapshot_content_sha256,
            "snapshot_file_sha256": self.snapshot_file_sha256,
            "support_q_e4_high": self.support_q_e4_high,
            "support_q_e4_low": self.support_q_e4_low,
            "update": self.update,
        }

    def canonical_sha256(self) -> str:
        return contract.canonical_sha256(self.to_canonical_dict())


_DETERMINISTIC_RULE = (
    "ARGMAX_MODE_LOGIT_THEN_THAT_MODES_CONDITIONAL_MEAN_Q_"
    "MAPPED_INTO_ITS_REGISTERED_SUPPORT_AND_ROUND_HALF_UP_TO_Q_E4"
)


# --------------------------------------------------------------------------- #
# Frozen actor
# --------------------------------------------------------------------------- #


class FrozenPilotActor:
    """Read-only, batch-1, CPU inference over the pre-registered actor."""

    def __init__(self, weights: LoadedPilotActorWeightsV1) -> None:
        if type(weights) is not LoadedPilotActorWeightsV1:
            raise FrozenActorError(
                f"weights must come from the verifying loader, got "
                f"{type(weights).__name__}"
            )
        _require(
            not torch.cuda.is_initialized(),
            "CUDA is initialized before building the frozen actor",
        )
        _require(
            weights.seed == contract.PREREGISTERED_SEED
            and weights.update == contract.PREREGISTERED_UPDATE,
            "the loaded weights are not the pre-registered seed/update",
        )

        config = HybridSacModelConfig(
            dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
        )
        actor = ConditionalHybridActor(config)
        missing_unexpected = actor.load_state_dict(
            {key: value.clone() for key, value in weights.actor_state.items()},
            strict=True,
        )
        _require(
            not missing_unexpected.missing_keys
            and not missing_unexpected.unexpected_keys,
            f"actor state dict mismatch: missing "
            f"{missing_unexpected.missing_keys}, unexpected "
            f"{missing_unexpected.unexpected_keys}",
        )
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        for buffer in actor.buffers():
            buffer.requires_grad_(False)
        _require(
            actor.uses_modeled_smoke_support,
            "the pre-registered actor must carry the registered "
            "MODELED_SMOKE_SUPPORT bounds",
        )
        _require(
            actor.modeled_smoke_support_sha256 == MODELED_SMOKE_SUPPORT_SHA256,
            "actor support digest differs from the registered contract",
        )
        for name, tensor in actor.state_dict().items():
            reference = weights.actor_state[name]
            _require(
                torch.equal(tensor, reference),
                f"actor parameter {name} does not equal the loaded weights",
            )

        self._actor = actor
        self._weights = weights
        self._actor_state_sha256 = _hash_state(dict(actor.state_dict()))
        _require(
            self._actor_state_sha256 == weights.actor_state_sha256,
            "the built actor's state digest differs from the loaded artifact",
        )
        lower, upper = actor.active_q_e4_bounds()
        self._support_low = tuple(int(value) for value in lower.tolist())
        self._support_high = tuple(int(value) for value in upper.tolist())
        _require(
            not torch.cuda.is_initialized(),
            "building the frozen actor initialized CUDA",
        )

    # -- introspection ----------------------------------------------------- #

    @property
    def actor_state_sha256(self) -> str:
        return self._actor_state_sha256

    @property
    def weights(self) -> LoadedPilotActorWeightsV1:
        return self._weights

    def support_bounds(self, mode_id: int) -> Tuple[int, int]:
        """The registered inclusive executable ``q_e4`` interval of a mode."""
        _require(
            0 <= int(mode_id) < len(self._support_low),
            f"mode_id {mode_id} is outside the 12-mode catalog",
        )
        return self._support_low[int(mode_id)], self._support_high[int(mode_id)]

    def assert_frozen(self) -> None:
        """Fail closed if the module ever left evaluation or gained a gradient."""
        _require(not self._actor.training, "the frozen actor left eval() mode")
        for name, parameter in self._actor.named_parameters():
            _require(
                not parameter.requires_grad,
                f"actor parameter {name} regained requires_grad",
            )
            _require(
                parameter.grad is None,
                f"actor parameter {name} accumulated a gradient; the live "
                f"pilot performs no learning",
            )
        _require(
            _hash_state(dict(self._actor.state_dict())) == self._actor_state_sha256,
            "the frozen actor's weights changed after loading",
        )

    # -- decision ---------------------------------------------------------- #

    def decide(self, features: Sequence[float]) -> PolicyProposalV1:
        """Return the deterministic batch-1 proposal for one 31-D observation.

        Args:
            features: Exactly 31 finite real numbers in the registered
                :data:`POLICY_FEATURE_ORDER`.  A mapping is deliberately not
                accepted: ordering is the contract, and a dict invites a
                reordering bug that no shape check would catch.
        """
        _require(
            not torch.cuda.is_initialized(),
            "CUDA is initialized before a pilot decision",
        )
        self.assert_frozen()
        values = self._validated_features(features)

        before = _rng_fingerprint()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            state = torch.tensor([values], dtype=torch.float32)
            _require(
                tuple(state.shape) == (1, POLICY_FEATURE_COUNT),
                f"policy tensor shape drift: {tuple(state.shape)}",
            )
            execution = self._actor.deterministic_execution(state)
            heads = self._actor(state)
        inference_ns = time.perf_counter_ns() - started
        after = _rng_fingerprint()

        _require(
            _rng_unchanged(before, after),
            "the frozen actor mutated a global RNG stream; deterministic "
            "evaluation must consume no randomness",
        )
        _require(
            not torch.cuda.is_initialized(),
            "a pilot decision initialized CUDA",
        )

        mode_id = int(execution.mode_index.item())
        q_e4 = int(execution.q_e4.item())
        low, high = self.support_bounds(mode_id)
        _require(
            Q_E4_MIN <= q_e4 <= Q_E4_MAX,
            f"proposed q_e4 {q_e4} is outside the mechanical wire range",
        )
        _require(
            low <= q_e4 <= high,
            f"proposed q_e4 {q_e4} is outside mode {mode_id}'s registered "
            f"executable interval [{low}, {high}]",
        )
        logits = tuple(float(value) for value in heads.logits[0].tolist())
        _require(
            logits.index(max(logits)) == mode_id,
            f"the selected mode {mode_id} is not the first argmax of the "
            f"returned logits",
        )

        return PolicyProposalV1(
            mode_id=mode_id,
            q=float(execution.q.item()),
            q_e4=q_e4,
            q_exec=q_e4 / float(Q_E4_SCALE),
            q_normalized_executed=float(execution.q_normalized_executed.item()),
            mode_logits=logits,
            selected_logit=logits[mode_id],
            inference_ns=int(inference_ns),
            actor_state_sha256=self._actor_state_sha256,
            snapshot_content_sha256=self._weights.snapshot_content_sha256,
            snapshot_file_sha256=self._weights.snapshot_file_sha256,
            seed=self._weights.seed,
            update=self._weights.update,
            support_q_e4_low=low,
            support_q_e4_high=high,
            modeled_smoke_support_sha256=MODELED_SMOKE_SUPPORT_SHA256,
            feature_count=POLICY_FEATURE_COUNT,
            deterministic_rule=_DETERMINISTIC_RULE,
            pilot_label=contract.PILOT_LABEL,
            pilot_contract_sha256=contract.PILOT_CONTRACT_SHA256,
        )

    # -- input validation -------------------------------------------------- #

    @staticmethod
    def _validated_features(features: Sequence[float]) -> Tuple[float, ...]:
        if isinstance(features, (str, bytes, dict)):
            raise FrozenActorError(
                f"features must be an ordered sequence of {POLICY_FEATURE_COUNT} "
                f"floats in POLICY_FEATURE_ORDER, got {type(features).__name__}"
            )
        try:
            values = tuple(features)
        except TypeError as exc:
            raise FrozenActorError(
                f"features is not iterable: {type(features).__name__}"
            ) from exc
        if len(values) != POLICY_FEATURE_COUNT:
            raise FrozenActorError(
                f"expected exactly {POLICY_FEATURE_COUNT} features in the "
                f"frozen order, got {len(values)}"
            )
        cleaned = []
        for name, value in zip(POLICY_FEATURE_ORDER, values):
            if type(value) is not float:
                raise FrozenActorError(
                    f"feature {name!r} must be an exact float, got "
                    f"{type(value).__name__}: {value!r}"
                )
            if value != value or value in (float("inf"), float("-inf")):
                raise FrozenActorError(f"feature {name!r} is not finite: {value!r}")
            cleaned.append(value)
        return tuple(cleaned)
