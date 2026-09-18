"""Conditional Hybrid-SAC neural models (DESIGN.md sections 5 and 6).

Phase 4b.1 -- ``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.

This module implements the *mathematics* of the feed-forward conditional
Hybrid-SAC actor and twin critics, and nothing else.  It contains no replay
buffer, no training loop, no optimizer schedule, no plotting and no
CARLA/OAI/Docker/CUDA integration.  Importing it reads no evidence file and
launches no runtime: the only import-time work is building constants from the
already-frozen action and state contracts.

What is implemented
-------------------

The policy is a parameterized-action policy over 12 joint discrete modes
(4 feature families x 3 quantizers) and one continuous spatial drop fraction
``q`` conditioned on the chosen mode::

    pi(m, q | s) = pi_d(m | s) * pi_c(q | s, m)

Actor (DESIGN.md section 5)::

    z          = f_theta(s)                        shared 2x128 ReLU MLP
    pi_d(m|s)  = softmax(logits(s))                12 logits
    u_m        = mu_m(s) + sigma_m(s) * eps_m      eps_m ~ N(0, 1)
    q_m        = 0.49 * (tanh(u_m) + 1)            in [0, 0.98]

``log_std`` is clamped to ``[LOG_STD_MIN, LOG_STD_MAX]``.  The continuous log
probability carries the **complete** change-of-variables Jacobian for both the
``tanh`` squash and the ``0.49`` scale::

    dq/du            = 0.49 * (1 - tanh(u)^2)
    log pi_c(q|s,m)  = log N(u; mu_m, sigma_m)
                       - log(0.49)
                       - log(1 - tanh(u)^2)

Execution quantization uses the repository action contract verbatim::

    q_e4 = clip(round_half_up(10000 * q), 0, 9800)

Critics (DESIGN.md section 6).  Two independently parameterized networks, each
mapping ``[state, one_hot(mode, 12), q_e4 / 9800]`` to one scalar, plus two
Polyak target copies and no target actor.  Because there are only 12 discrete
choices they are enumerated **exactly** in the soft value; only the 1-D ``q``
is sampled::

    V(s') = sum_m pi_d(m|s') [ min_i Q_target_i(s', m, q'_m)
                               - alpha_d log pi_d(m|s')
                               - alpha_c log pi_c(q'_m|s', m) ]

    y     = r + (1 - done) * gamma^d * V(s')

The ``gamma^d`` duration exponent is mandatory, not cosmetic: the action-hold
contract holds one decision across ``d >= 2`` prepared frames, which makes this
a variable-duration semi-Markov problem (DESIGN.md section 3).

    J_pi  = E_s sum_m pi_d(m|s) [ alpha_d log pi_d(m|s)
                                  + alpha_c log pi_c(q_m|s, m)
                                  - min_i Q_i(s, m, q_m) ]

``alpha_d`` and ``alpha_c`` are fixed positive constants supplied by the
caller.  Automatic entropy tuning is deliberately **not** implemented in this
phase.

Two deliberate engineering decisions
------------------------------------

**Quantization delegates to the contract.**  :func:`quantize_q_e4` calls
:func:`action_contract.round_half_up_q_e4` element-wise rather than
reimplementing half-up rounding in vectorized tensor arithmetic.  This is not
stylistic.  The contract rounds through the shortest round-tripping decimal
representation, so ``q = 0.70005`` is a true decimal tie and must round *up*
to ``7001``; the natural tensor form ``floor(10000 * q + 0.5)`` yields ``7000``
even in float64, and disagrees on many more values in float32.  A second
implementation of a registered wire rule that silently disagrees on ties is
exactly the drift this repository fails closed against, so there is only one
implementation and the tensor path defers to it.

**The default dtype is float64.**  ``torch.float32`` cannot hold a five-decimal
tie such as ``0.70005`` exactly, so a float32 actor would destroy the very tie
semantics the wire contract defines.  These networks are tiny and CPU-only, so
double precision costs nothing that matters here and additionally tightens
determinism.  float32 remains selectable and is documented as tie-lossy.

Scope boundary
--------------

Nothing here may consume the 288-cell measured aggregates as replay
transitions, and nothing here interpolates between the six measured ``q``
anchors.  This module never touches that evidence at all; see
``anchor_store.py`` for why those aggregates are inadmissible as transitions.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_SCALE,
    Q_MAX,
    Q_MIN,
    round_half_up_q_e4,
)
from .state_reward_transition_contract import POLICY_FEATURE_COUNT

__all__ = [
    "ActorHeads",
    "CRITIC_INPUT_DIM",
    "ConditionalHybridActor",
    "DeterministicExecution",
    "HIDDEN_WIDTH",
    "HybridQCritic",
    "HybridSacModelConfig",
    "HybridSacModelError",
    "InvalidHyperparameterError",
    "InvalidTensorError",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "MODE_COUNT",
    "ModeConditionalSample",
    "PHASE_LABEL",
    "Q_SQUASH_SCALE",
    "STATE_DIM",
    "TwinHybridCritics",
    "actor_objective",
    "build_actor",
    "build_twin_critics",
    "critic_target",
    "mode_one_hot",
    "quantize_q_e4",
    "soft_state_value",
]


# --------------------------------------------------------------------------- #
# Frozen registered constants
# --------------------------------------------------------------------------- #

#: This phase is a synthetic smoke test; no deployable claim attaches to it.
PHASE_LABEL = "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY"

#: Policy input width, taken from the frozen 31-feature state contract.
STATE_DIM: int = POLICY_FEATURE_COUNT

#: Joint discrete modes, taken from the frozen catalog: 4 families x 3 quantizers.
MODE_COUNT: int = EXPECTED_MODE_COUNT

#: Shared-encoder width; two hidden layers of this size.
HIDDEN_WIDTH = 128

#: Number of hidden layers in the shared encoder and in each critic trunk.
HIDDEN_DEPTH = 2

#: Explicit bounded interval for the conditional log standard deviation.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

#: ``q = Q_SQUASH_SCALE * (tanh(u) + 1)`` maps R onto the registered [0, 0.98].
Q_SQUASH_SCALE = Q_MAX / 2.0

#: ``log(0.49)``, the constant part of the squash Jacobian.
_LOG_Q_SQUASH_SCALE = math.log(Q_SQUASH_SCALE)

_LOG_TWO = math.log(2.0)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)

#: Critic input width: state, mode one-hot and the normalized executed q.
CRITIC_INPUT_DIM: int = STATE_DIM + MODE_COUNT + 1


# --------------------------------------------------------------------------- #
# Exceptions: fail closed, never normalize
# --------------------------------------------------------------------------- #


class HybridSacModelError(Exception):
    """Base class for every Hybrid-SAC model failure."""


class InvalidTensorError(HybridSacModelError):
    """A tensor has the wrong shape, dtype or a non-finite entry."""


class InvalidHyperparameterError(HybridSacModelError):
    """A temperature, discount, duration or Polyak coefficient is invalid."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _check_finite(tensor: Tensor, name: str) -> Tensor:
    """Reject NaN and infinity rather than propagating them into a network."""
    if not torch.isfinite(tensor).all():
        raise InvalidTensorError(
            f"{name} contains a non-finite entry (NaN or infinity); refusing "
            f"to propagate it"
        )
    return tensor


def _check_state(state: Tensor, state_dim: int) -> Tensor:
    """Validate a policy state batch of shape ``(batch, state_dim)``."""
    if not isinstance(state, Tensor):
        raise InvalidTensorError(
            f"state must be a torch.Tensor, got {type(state).__name__}"
        )
    if state.dim() != 2:
        raise InvalidTensorError(
            f"state must be 2-D (batch, {state_dim}), got shape "
            f"{tuple(state.shape)}"
        )
    if state.shape[1] != state_dim:
        raise InvalidTensorError(
            f"state must have exactly {state_dim} features (the frozen policy "
            f"feature count), got {state.shape[1]}"
        )
    if state.shape[0] == 0:
        raise InvalidTensorError("state batch is empty")
    if not torch.is_floating_point(state):
        raise InvalidTensorError(
            f"state must be a floating-point tensor, got dtype {state.dtype}"
        )
    return _check_finite(state, "state")


def _check_positive_alpha(value: float, name: str) -> float:
    """Validate a fixed positive entropy temperature."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise InvalidHyperparameterError(
            f"{name} must be a finite positive constant, got {value!r}"
        )
    return numeric


# --------------------------------------------------------------------------- #
# Deterministic construction
# --------------------------------------------------------------------------- #


@contextmanager
def _local_torch_seed(seed: Optional[int]) -> Iterator[None]:
    """Seed the global CPU RNG for a block, then restore the previous state.

    Module initialization draws from the global generator, so reproducible
    weights require seeding it.  The prior RNG state is saved and restored so
    that building a model does not perturb the caller's stream.
    """
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


@dataclass(frozen=True)
class HybridSacModelConfig:
    """Shapes and bounds shared by the actor and the critics.

    The defaults are bound to the frozen contracts: ``state_dim`` is the
    31-feature policy vector and ``mode_count`` is the 12-mode catalog.
    """

    state_dim: int = STATE_DIM
    mode_count: int = MODE_COUNT
    hidden_width: int = HIDDEN_WIDTH
    hidden_depth: int = HIDDEN_DEPTH
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        for name in ("state_dim", "mode_count", "hidden_width", "hidden_depth"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise InvalidHyperparameterError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if not self.log_std_min < self.log_std_max:
            raise InvalidHyperparameterError(
                f"log_std bounds must satisfy min < max, got "
                f"[{self.log_std_min}, {self.log_std_max}]"
            )
        for bound in (self.log_std_min, self.log_std_max):
            if not math.isfinite(bound):
                raise InvalidHyperparameterError(
                    f"log_std bounds must be finite, got "
                    f"[{self.log_std_min}, {self.log_std_max}]"
                )
        if not torch.is_floating_point(torch.empty(0, dtype=self.dtype)):
            raise InvalidHyperparameterError(
                f"dtype must be a floating-point dtype, got {self.dtype}"
            )

    @property
    def critic_input_dim(self) -> int:
        """``state_dim + mode_count + 1``."""
        return self.state_dim + self.mode_count + 1


def _mlp_trunk(
    input_dim: int, width: int, depth: int, dtype: torch.dtype
) -> nn.Sequential:
    """Build a ``depth``-layer ReLU MLP trunk of the given width."""
    layers: list = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, width, dtype=dtype))
        layers.append(nn.ReLU())
        current = width
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Execution quantization
# --------------------------------------------------------------------------- #


def quantize_q_e4(q: Tensor) -> Tensor:
    """Apply the registered wire quantization element-wise.

    Implements ``q_e4 = clip(round_half_up(10000 * q), 0, 9800)`` by delegating
    to :func:`action_contract.round_half_up_q_e4`, which is the single
    registered implementation of that rule.  See the module docstring for why
    this is not reimplemented in vectorized tensor arithmetic.

    Args:
        q: A floating-point tensor of requested qualities, any shape.

    Returns:
        A ``torch.long`` tensor of the same shape holding ``q_e4`` values in
        ``[0, 9800]``.

    Raises:
        InvalidTensorError: if ``q`` is not a finite floating-point tensor.
    """
    if not isinstance(q, Tensor):
        raise InvalidTensorError(
            f"q must be a torch.Tensor, got {type(q).__name__}"
        )
    if not torch.is_floating_point(q):
        raise InvalidTensorError(
            f"q must be a floating-point tensor, got dtype {q.dtype}"
        )
    _check_finite(q, "q")
    flat = q.detach().reshape(-1).double().tolist()
    quantized = [round_half_up_q_e4(value) for value in flat]
    return torch.tensor(quantized, dtype=torch.long, device=q.device).reshape(
        q.shape
    )


def _straight_through_executed_normalized_q(
    q: Tensor, q_e4: Tensor
) -> Tensor:
    """Return the executed normalized ``q`` with a straight-through gradient.

    The critic must be evaluated at what the system actually transmits, which
    is the quantized ``q_e4 / 9800`` and not the unquantized request.  But
    ``round_half_up`` is a step function whose derivative is zero almost
    everywhere, so using it directly would sever the actor's gradient path to
    ``mu`` and ``log_std``.

    The straight-through estimator resolves this exactly::

        st = request + (executed - request).detach()

    The forward value is bit-identical to ``q_e4 / 9800``; the backward pass
    uses ``d(q / 0.98)/d theta``, i.e. it pretends the quantizer is the
    identity.  This is the standard and documented bias of a straight-through
    estimator: the gradient ignores a sub-``1e-4`` rounding step, which is far
    below any quality difference the reward can resolve.
    """
    normalized_request = q / Q_MAX
    normalized_executed = q_e4.to(q.dtype) / float(Q_E4_MAX)
    return normalized_request + (normalized_executed - normalized_request).detach()


def mode_one_hot(
    mode_index: Tensor, mode_count: int = MODE_COUNT, dtype: torch.dtype = torch.float64
) -> Tensor:
    """One-hot encode a ``(batch,)`` mode index tensor as ``(batch, mode_count)``."""
    if mode_index.dim() != 1:
        raise InvalidTensorError(
            f"mode_index must be 1-D (batch,), got shape {tuple(mode_index.shape)}"
        )
    if mode_index.dtype not in (torch.int32, torch.int64):
        raise InvalidTensorError(
            f"mode_index must be an integer tensor, got dtype {mode_index.dtype}"
        )
    if int(mode_index.min()) < 0 or int(mode_index.max()) >= mode_count:
        raise InvalidTensorError(
            f"mode_index entries must lie in [0, {mode_count - 1}]"
        )
    return F.one_hot(mode_index, num_classes=mode_count).to(dtype)


# --------------------------------------------------------------------------- #
# Actor outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActorHeads:
    """Raw actor head outputs for one state batch."""

    #: ``(batch, mode_count)`` categorical logits.
    logits: Tensor
    #: ``(batch, mode_count)`` conditional Gaussian means in pre-squash space.
    mean: Tensor
    #: ``(batch, mode_count)`` conditional log standard deviations, clamped.
    log_std: Tensor


@dataclass(frozen=True)
class ModeConditionalSample:
    """One reparameterized sample of ``q`` for **every** one of the 12 modes.

    Every field is ``(batch, mode_count)`` except ``log_prob_discrete`` and
    ``probs``, which are the categorical terms over the same grid.  Sampling
    all modes at once is what makes the exact 12-mode enumeration in
    :func:`soft_state_value` and :func:`actor_objective` possible.
    """

    #: Pre-squash Gaussian sample ``u_m``.
    pre_squash: Tensor
    #: Squashed continuous quality ``q_m`` in ``[0, 0.98]``.
    q: Tensor
    #: Registered wire integer ``q_e4`` in ``[0, 9800]``.
    q_e4: Tensor
    #: ``q_e4 / 9800``, the exact executed critic input (no gradient path).
    q_normalized_executed: Tensor
    #: ``q_e4 / 9800`` forward, request gradient backward.
    q_normalized_straight_through: Tensor
    #: ``log pi_c(q_m | s, m)`` including the full tanh and 0.49 Jacobian.
    log_prob_continuous: Tensor
    #: ``log pi_d(m | s)``.
    log_prob_discrete: Tensor
    #: ``pi_d(m | s)``.
    probs: Tensor

    @property
    def q_executed(self) -> Tensor:
        """The executed real quality ``q_e4 / 10000`` in ``[0, 0.98]``."""
        return self.q_e4.to(self.q.dtype) / float(Q_E4_SCALE)


@dataclass(frozen=True)
class DeterministicExecution:
    """The deterministic evaluation action: argmax mode at its conditional mean."""

    #: ``(batch,)`` argmax categorical mode.
    mode_index: Tensor
    #: ``(batch,)`` squashed conditional mean, before quantization.
    q: Tensor
    #: ``(batch,)`` registered wire integer.
    q_e4: Tensor
    #: ``(batch,)`` executed real quality ``q_e4 / 10000``.
    q_executed: Tensor
    #: ``(batch,)`` executed critic input ``q_e4 / 9800``.
    q_normalized_executed: Tensor


# --------------------------------------------------------------------------- #
# Conditional hybrid actor
# --------------------------------------------------------------------------- #


class ConditionalHybridActor(nn.Module):
    """Feed-forward conditional Hybrid-SAC actor (DESIGN.md section 5).

    One shared 2x128 ReLU encoder feeds three heads: 12 categorical logits, 12
    conditional means and 12 conditional log standard deviations.  There is one
    Gaussian head per joint mode, so the continuous policy is conditioned on
    the discrete choice rather than shared across it.
    """

    def __init__(self, config: Optional[HybridSacModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else HybridSacModelConfig()
        cfg = self.config
        self.encoder = _mlp_trunk(
            cfg.state_dim, cfg.hidden_width, cfg.hidden_depth, cfg.dtype
        )
        self.logit_head = nn.Linear(cfg.hidden_width, cfg.mode_count, dtype=cfg.dtype)
        self.mean_head = nn.Linear(cfg.hidden_width, cfg.mode_count, dtype=cfg.dtype)
        self.log_std_head = nn.Linear(
            cfg.hidden_width, cfg.mode_count, dtype=cfg.dtype
        )

    # -- forward ----------------------------------------------------------- #

    def forward(self, state: Tensor) -> ActorHeads:
        """Return the three head outputs, with ``log_std`` clamped."""
        state = _check_state(state, self.config.state_dim)
        latent = self.encoder(state)
        log_std = torch.clamp(
            self.log_std_head(latent),
            min=self.config.log_std_min,
            max=self.config.log_std_max,
        )
        return ActorHeads(
            logits=self.logit_head(latent),
            mean=self.mean_head(latent),
            log_std=log_std,
        )

    def mode_log_probs(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(log pi_d, pi_d)`` over the 12 joint modes."""
        heads = self(state)
        log_probs = F.log_softmax(heads.logits, dim=-1)
        return log_probs, log_probs.exp()

    # -- sampling ---------------------------------------------------------- #

    def sample_all_modes(
        self,
        state: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> ModeConditionalSample:
        """Reparameterized sample of ``q_m`` for every mode, with log densities.

        Draws one ``eps_m ~ N(0, 1)`` per mode, applies
        ``q_m = 0.49 * (tanh(mu_m + sigma_m * eps_m) + 1)`` and returns the
        transformed log density including the complete Jacobian.
        """
        heads = self(state)
        std = heads.log_std.exp()
        noise = torch.randn(
            heads.mean.shape,
            dtype=heads.mean.dtype,
            device=heads.mean.device,
            generator=generator,
        )
        pre_squash = heads.mean + std * noise
        return self._finish_sample(heads, pre_squash)

    def _finish_sample(
        self, heads: ActorHeads, pre_squash: Tensor
    ) -> ModeConditionalSample:
        """Squash, quantize and score a pre-squash sample for all modes."""
        q = Q_SQUASH_SCALE * (torch.tanh(pre_squash) + 1.0)
        # Guard the registered range against any floating-point overshoot; the
        # clamp is a no-op for finite pre-squash values.
        q = q.clamp(min=Q_MIN, max=Q_MAX)
        q_e4 = quantize_q_e4(q)
        log_prob_discrete = F.log_softmax(heads.logits, dim=-1)
        return ModeConditionalSample(
            pre_squash=pre_squash,
            q=q,
            q_e4=q_e4,
            q_normalized_executed=q_e4.to(q.dtype) / float(Q_E4_MAX),
            q_normalized_straight_through=(
                _straight_through_executed_normalized_q(q, q_e4)
            ),
            log_prob_continuous=continuous_log_prob(
                pre_squash, heads.mean, heads.log_std
            ),
            log_prob_discrete=log_prob_discrete,
            probs=log_prob_discrete.exp(),
        )

    # -- deterministic evaluation ------------------------------------------ #

    @torch.no_grad()
    def deterministic_execution(self, state: Tensor) -> DeterministicExecution:
        """Select the argmax mode at its conditional mean, exactly quantized.

        This is the evaluation policy: no categorical sampling and no Gaussian
        noise.  The returned ``q_e4`` is the exact registered quantization of
        the squashed conditional mean of the selected mode.
        """
        heads = self(state)
        mode_index = torch.argmax(heads.logits, dim=-1)
        selected_mean = heads.mean.gather(1, mode_index.unsqueeze(1)).squeeze(1)
        q = Q_SQUASH_SCALE * (torch.tanh(selected_mean) + 1.0)
        q = q.clamp(min=Q_MIN, max=Q_MAX)
        q_e4 = quantize_q_e4(q)
        return DeterministicExecution(
            mode_index=mode_index,
            q=q,
            q_e4=q_e4,
            q_executed=q_e4.to(q.dtype) / float(Q_E4_SCALE),
            q_normalized_executed=q_e4.to(q.dtype) / float(Q_E4_MAX),
        )


def continuous_log_prob(
    pre_squash: Tensor, mean: Tensor, log_std: Tensor
) -> Tensor:
    """Transformed log density of ``q = 0.49 * (tanh(u) + 1)``.

    Implements::

        log pi_c(q) = log N(u; mu, sigma) - log(0.49) - log(1 - tanh(u)^2)

    The Jacobian term is evaluated in the numerically stable form
    ``log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))``, which avoids the
    catastrophic cancellation of computing ``1 - tanh(u)^2`` directly for large
    ``|u|``.
    """
    standardized = (pre_squash - mean) / log_std.exp()
    log_normal = -0.5 * standardized.pow(2) - log_std - _HALF_LOG_TWO_PI
    log_tanh_jacobian = 2.0 * (
        _LOG_TWO - pre_squash - F.softplus(-2.0 * pre_squash)
    )
    return log_normal - _LOG_Q_SQUASH_SCALE - log_tanh_jacobian


# --------------------------------------------------------------------------- #
# Critics
# --------------------------------------------------------------------------- #


class HybridQCritic(nn.Module):
    """One Q network over ``[state, one_hot(mode, 12), q_e4 / 9800]``."""

    def __init__(self, config: Optional[HybridSacModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else HybridSacModelConfig()
        cfg = self.config
        self.trunk = _mlp_trunk(
            cfg.critic_input_dim, cfg.hidden_width, cfg.hidden_depth, cfg.dtype
        )
        self.value_head = nn.Linear(cfg.hidden_width, 1, dtype=cfg.dtype)

    def forward(
        self, state: Tensor, mode_onehot: Tensor, q_normalized: Tensor
    ) -> Tensor:
        """Return ``(batch,)`` scalar Q values for one mode per state."""
        state = _check_state(state, self.config.state_dim)
        if mode_onehot.shape != (state.shape[0], self.config.mode_count):
            raise InvalidTensorError(
                f"mode_onehot must have shape "
                f"({state.shape[0]}, {self.config.mode_count}), got "
                f"{tuple(mode_onehot.shape)}"
            )
        if q_normalized.shape != (state.shape[0],):
            raise InvalidTensorError(
                f"q_normalized must have shape ({state.shape[0]},), got "
                f"{tuple(q_normalized.shape)}"
            )
        _check_finite(mode_onehot, "mode_onehot")
        _check_finite(q_normalized, "q_normalized")
        features = torch.cat(
            [state, mode_onehot.to(state.dtype), q_normalized.unsqueeze(1)], dim=1
        )
        return self.value_head(self.trunk(features)).squeeze(1)

    def q_all_modes(self, state: Tensor, q_normalized: Tensor) -> Tensor:
        """Return ``(batch, mode_count)`` Q values, one per joint mode.

        ``q_normalized`` is ``(batch, mode_count)``: the mode-conditional
        executed quality for each of the 12 modes.  This is the exact
        enumeration the twin-critic target requires -- no mode is sampled.
        """
        state = _check_state(state, self.config.state_dim)
        batch = state.shape[0]
        modes = self.config.mode_count
        if q_normalized.shape != (batch, modes):
            raise InvalidTensorError(
                f"q_normalized must have shape ({batch}, {modes}), got "
                f"{tuple(q_normalized.shape)}"
            )
        _check_finite(q_normalized, "q_normalized")
        expanded_state = state.unsqueeze(1).expand(batch, modes, state.shape[1])
        identity = torch.eye(modes, dtype=state.dtype, device=state.device)
        expanded_modes = identity.unsqueeze(0).expand(batch, modes, modes)
        features = torch.cat(
            [expanded_state, expanded_modes, q_normalized.unsqueeze(2)], dim=2
        )
        flat = features.reshape(batch * modes, self.config.critic_input_dim)
        return self.value_head(self.trunk(flat)).reshape(batch, modes)


class TwinHybridCritics(nn.Module):
    """Two independently parameterized critics plus two Polyak targets.

    There is deliberately **no target actor**: the soft value enumerates the
    discrete modes under the *current* actor and only the target critics are
    Polyak-averaged.
    """

    def __init__(self, config: Optional[HybridSacModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else HybridSacModelConfig()
        self.critic_1 = HybridQCritic(self.config)
        self.critic_2 = HybridQCritic(self.config)
        # Independent initialization: a deepcopy here would make the twins
        # identical and defeat the purpose of the minimum.
        self.target_1 = deepcopy(self.critic_1)
        self.target_2 = deepcopy(self.critic_2)
        for target in (self.target_1, self.target_2):
            target.requires_grad_(False)
            for parameter in target.parameters():
                parameter.requires_grad_(False)

    def q_values(
        self, state: Tensor, mode_onehot: Tensor, q_normalized: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Online twin Q values for one mode per state."""
        return (
            self.critic_1(state, mode_onehot, q_normalized),
            self.critic_2(state, mode_onehot, q_normalized),
        )

    def min_q_all_modes(self, state: Tensor, q_normalized: Tensor) -> Tensor:
        """``min(Q1, Q2)`` over all 12 modes, from the online critics."""
        return torch.minimum(
            self.critic_1.q_all_modes(state, q_normalized),
            self.critic_2.q_all_modes(state, q_normalized),
        )

    @torch.no_grad()
    def target_min_q_all_modes(
        self, state: Tensor, q_normalized: Tensor
    ) -> Tensor:
        """``min(Q1_target, Q2_target)`` over all 12 modes."""
        return torch.minimum(
            self.target_1.q_all_modes(state, q_normalized),
            self.target_2.q_all_modes(state, q_normalized),
        )

    @torch.no_grad()
    def polyak_update(self, tau: float) -> None:
        """Soft-update both targets: ``target <- tau * online + (1 - tau) * target``.

        ``tau = 1`` copies the online critics outright; small ``tau`` tracks
        them slowly.  Buffers are copied verbatim.
        """
        numeric_tau = float(tau)
        if not math.isfinite(numeric_tau) or not 0.0 < numeric_tau <= 1.0:
            raise InvalidHyperparameterError(
                f"tau must be a finite value in (0, 1], got {tau!r}"
            )
        for online, target in (
            (self.critic_1, self.target_1),
            (self.critic_2, self.target_2),
        ):
            for online_p, target_p in zip(
                online.parameters(), target.parameters()
            ):
                target_p.mul_(1.0 - numeric_tau).add_(online_p, alpha=numeric_tau)
            for online_b, target_b in zip(online.buffers(), target.buffers()):
                target_b.copy_(online_b)


# --------------------------------------------------------------------------- #
# Losses and targets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SoftValueBreakdown:
    """Per-mode terms of the soft state value, retained for auditing."""

    #: ``(batch,)`` the enumerated soft value ``V(s')``.
    value: Tensor
    #: ``(batch, mode_count)`` categorical probabilities ``pi_d(m|s')``.
    probs: Tensor
    #: ``(batch, mode_count)`` ``min_i Q_target_i(s', m, q'_m)``.
    min_target_q: Tensor
    #: ``(batch, mode_count)`` the bracketed soft term per mode.
    per_mode_term: Tensor


def soft_state_value(
    actor: ConditionalHybridActor,
    critics: TwinHybridCritics,
    next_state: Tensor,
    alpha_d: float,
    alpha_c: float,
    generator: Optional[torch.Generator] = None,
) -> SoftValueBreakdown:
    """Exact 12-mode enumeration of the next-state soft value.

    Implements DESIGN.md section 6::

        V(s') = sum_m pi_d(m|s') [ min_i Q_target_i(s', m, q'_m)
                                   - alpha_d log pi_d(m|s')
                                   - alpha_c log pi_c(q'_m|s', m) ]

    Every one of the 12 discrete modes contributes; only the 1-D continuous
    ``q'_m`` is sampled.  The result carries no gradient: the target critics
    are frozen and the whole computation runs under ``no_grad``.
    """
    alpha_d = _check_positive_alpha(alpha_d, "alpha_d")
    alpha_c = _check_positive_alpha(alpha_c, "alpha_c")
    with torch.no_grad():
        sample = actor.sample_all_modes(next_state, generator=generator)
        min_target_q = critics.target_min_q_all_modes(
            next_state, sample.q_normalized_executed
        )
        per_mode_term = (
            min_target_q
            - alpha_d * sample.log_prob_discrete
            - alpha_c * sample.log_prob_continuous
        )
        value = (sample.probs * per_mode_term).sum(dim=-1)
    return SoftValueBreakdown(
        value=value,
        probs=sample.probs,
        min_target_q=min_target_q,
        per_mode_term=per_mode_term,
    )


def critic_target(
    reward: Tensor,
    done: Tensor,
    next_value: Tensor,
    gamma: float,
    duration: Tensor,
) -> Tensor:
    """Semi-Markov critic target ``y = r + (1 - done) * gamma^d * V(s')``.

    ``duration`` is the realized hold length ``d_t`` in prepared-frame
    intervals.  It is **not** optional: the action-hold contract holds one
    decision for at least two frames, so a transition that spans three frames
    must be discounted by ``gamma^3`` and not ``gamma``.

    Args:
        reward: ``(batch,)`` scalar reward for the transition.
        done: ``(batch,)`` terminal flag in ``{0, 1}``.
        next_value: ``(batch,)`` enumerated soft value of ``s'``.
        gamma: Scalar discount in ``(0, 1]``.
        duration: ``(batch,)`` integer-valued hold duration, each ``>= 1``.
    """
    numeric_gamma = float(gamma)
    if not math.isfinite(numeric_gamma) or not 0.0 < numeric_gamma <= 1.0:
        raise InvalidHyperparameterError(
            f"gamma must be a finite value in (0, 1], got {gamma!r}"
        )
    for tensor, name in (
        (reward, "reward"),
        (done, "done"),
        (next_value, "next_value"),
        (duration, "duration"),
    ):
        if not isinstance(tensor, Tensor) or tensor.dim() != 1:
            raise InvalidTensorError(
                f"{name} must be a 1-D (batch,) tensor, got "
                f"{tuple(tensor.shape) if isinstance(tensor, Tensor) else type(tensor)}"
            )
        _check_finite(tensor.to(torch.float64), name)
    if not (reward.shape == done.shape == next_value.shape == duration.shape):
        raise InvalidTensorError(
            f"reward, done, next_value and duration must share one batch "
            f"shape, got {tuple(reward.shape)}, {tuple(done.shape)}, "
            f"{tuple(next_value.shape)}, {tuple(duration.shape)}"
        )
    duration_float = duration.to(next_value.dtype)
    if not torch.equal(duration_float, torch.floor(duration_float)):
        raise InvalidHyperparameterError(
            "duration must be integer-valued (whole prepared-frame intervals)"
        )
    if float(duration_float.min()) < 1.0:
        raise InvalidHyperparameterError(
            "duration must be at least 1 prepared-frame interval"
        )
    done_float = done.to(next_value.dtype)
    if not torch.all((done_float == 0.0) | (done_float == 1.0)):
        raise InvalidTensorError("done must contain only 0 or 1")
    discount = torch.pow(
        torch.tensor(numeric_gamma, dtype=next_value.dtype, device=next_value.device),
        duration_float,
    )
    return reward.to(next_value.dtype) + (1.0 - done_float) * discount * next_value


@dataclass(frozen=True)
class ActorObjectiveBreakdown:
    """The actor objective and the per-mode terms it enumerated."""

    #: Scalar ``J_pi`` averaged over the batch; minimize this.
    objective: Tensor
    #: ``(batch, mode_count)`` bracketed term per mode.
    per_mode_term: Tensor
    #: ``(batch, mode_count)`` categorical probabilities ``pi_d(m|s)``.
    probs: Tensor
    #: The underlying all-mode sample, retained for auditing.
    sample: ModeConditionalSample


def actor_objective(
    actor: ConditionalHybridActor,
    critics: TwinHybridCritics,
    state: Tensor,
    alpha_d: float,
    alpha_c: float,
    generator: Optional[torch.Generator] = None,
) -> ActorObjectiveBreakdown:
    """Exact 12-mode enumeration of the actor objective.

    Implements DESIGN.md section 5/6::

        J_pi = E_s sum_m pi_d(m|s) [ alpha_d log pi_d(m|s)
                                     + alpha_c log pi_c(q_m|s, m)
                                     - min_i Q_i(s, m, q_m) ]

    All 12 modes are enumerated exactly rather than sampled, so every
    categorical branch and every conditional Gaussian head receives gradient on
    every update.  The critics are evaluated at the **executed** quantized
    quality through a straight-through estimator, so the value corresponds to
    what the system would actually transmit while ``mu`` and ``log_std`` still
    receive gradient.

    The returned objective is a quantity to **minimize**.  Critic parameters
    appear in the graph; the caller must step only the actor's parameters.
    """
    alpha_d = _check_positive_alpha(alpha_d, "alpha_d")
    alpha_c = _check_positive_alpha(alpha_c, "alpha_c")
    sample = actor.sample_all_modes(state, generator=generator)
    min_q = critics.min_q_all_modes(
        state, sample.q_normalized_straight_through
    )
    per_mode_term = (
        alpha_d * sample.log_prob_discrete
        + alpha_c * sample.log_prob_continuous
        - min_q
    )
    objective = (sample.probs * per_mode_term).sum(dim=-1).mean()
    return ActorObjectiveBreakdown(
        objective=objective,
        per_mode_term=per_mode_term,
        probs=sample.probs,
        sample=sample,
    )


# --------------------------------------------------------------------------- #
# Deterministic factories (no work happens at import time)
# --------------------------------------------------------------------------- #


def build_actor(
    config: Optional[HybridSacModelConfig] = None,
    seed: Optional[int] = None,
) -> ConditionalHybridActor:
    """Build an actor, optionally with reproducible CPU initialization."""
    with _local_torch_seed(seed):
        return ConditionalHybridActor(config)


def build_twin_critics(
    config: Optional[HybridSacModelConfig] = None,
    seed: Optional[int] = None,
) -> TwinHybridCritics:
    """Build twin critics and their targets, optionally reproducibly."""
    with _local_torch_seed(seed):
        return TwinHybridCritics(config)
