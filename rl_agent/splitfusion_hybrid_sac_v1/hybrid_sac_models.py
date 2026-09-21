"""Conditional Hybrid-SAC neural models (DESIGN.md sections 5 and 6).

Phase 4b.1 -- ``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.

This module implements the *mathematics* of the feed-forward conditional
Hybrid-SAC actor and twin critics, and nothing else.  It contains no replay
buffer, no training loop, no optimizer schedule, no plotting and no
CARLA/OAI/Docker/CUDA integration.  Importing it reads no evidence file and
launches no runtime.  Its optional modeled-smoke curriculum imports only an
immutable contract assembled from already-frozen source SHA pins; loading
evidence remains an explicit operation owned by the evidence modules.

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
    q_m        = 0.49 * (tanh(u_m) + 1)            default: [0, 0.98]

An explicit ``MODELED_SMOKE_SUPPORT`` configuration instead maps the same
normalized ``z_m = (tanh(u_m) + 1) / 2`` directly into each mode's hash-bound
inclusive curriculum interval.  It never samples globally and then projects.

``log_std`` is clamped to ``[LOG_STD_MIN, LOG_STD_MAX]``.  The continuous log
probability carries the **complete** change-of-variables Jacobian for both the
``tanh`` squash and the ``0.49`` scale::

    dq/du            = 0.49 * (1 - tanh(u)^2)
    log pi_c(q|s,m)  = log N(u; mu_m, sigma_m)
                       - log(0.49)
                       - log(1 - tanh(u)^2)

The curriculum records its continuous density as ``NORMALIZED_Z_DENSITY`` and
uses ``-log(0.5)`` instead of ``-log(0.49)``.  It intentionally does not
subtract a mode's physical interval width, so evidence-derived width
differences cannot bias categorical entropy.  The default
:func:`continuous_log_prob` remains a physical-q density.

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

**Wire quantization remains exact.**  The selected action at the execution
boundary is passed through :func:`action_contract.round_half_up_q_e4`, the
single registered decimal half-up implementation.  Batched float32 samples
inside an actor update use a vectorized implementation after promotion to
float64.  Exhaustive threshold-neighbour tests pin that fast path to the
contract for the representable float32 inputs a float32 actor can emit.  The
float64 reference path continues to delegate element-wise to the contract,
because a hand-authored decimal tie such as ``0.70005`` can otherwise expose
binary multiplication rounding.

**The default training dtype is float32.**  Decimal tie semantics belong at
the wire boundary; a stochastic neural policy emits binary floating-point
values, not authored decimal literals.  Float32 has substantially finer
resolution than the ``1e-4`` wire grid and avoids making GPU training depend
on slow FP64 arithmetic.  Float64 remains available as an audit/reference
configuration.

Scope boundary
--------------

Nothing here may consume the 288-cell measured aggregates as replay
transitions, and nothing here interpolates between the six measured ``q``
anchors.  The optional support contract carries source hashes and precomputed
bounds but reads no evidence; see ``anchor_store.py`` for why aggregates are
inadmissible as transitions.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
    Q_MAX,
    Q_MIN,
    round_half_up_q_e4,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    ModeledSmokeSupportContract,
    ModeledSmokeSupportError,
    require_registered_modeled_smoke_support,
)
from .state_reward_transition_contract import POLICY_FEATURE_COUNT
from .transaction_identity import MINIMUM_HOLD_TENSORS

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
    "NORMALIZED_Z_DENSITY",
    "PHASE_LABEL",
    "PHYSICAL_Q_DENSITY",
    "Q_SQUASH_SCALE",
    "STATE_DIM",
    "TwinHybridCritics",
    "actor_objective",
    "build_actor",
    "build_twin_critics",
    "critic_target",
    "mode_one_hot",
    "normalized_z_log_prob",
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

#: Names make the entropy coordinate part of each sample's audit surface.
PHYSICAL_Q_DENSITY = "PHYSICAL_Q_DENSITY"
NORMALIZED_Z_DENSITY = "NORMALIZED_Z_DENSITY"

_LOG_NORMALIZED_Z_SQUASH_SCALE = math.log(0.5)

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


def _check_normalized_q(q_normalized: Tensor, name: str) -> Tensor:
    """Require a finite floating critic action coordinate in ``[0, 1]``."""
    if not isinstance(q_normalized, Tensor) or not torch.is_floating_point(
        q_normalized
    ):
        raise InvalidTensorError(
            f"{name} must be a floating-point tensor, got "
            f"{getattr(q_normalized, 'dtype', type(q_normalized).__name__)}"
        )
    _check_finite(q_normalized, name)
    if bool(((q_normalized < 0.0) | (q_normalized > 1.0)).any()):
        raise InvalidTensorError(f"{name} entries must lie in [0, 1]")
    return q_normalized


def _check_one_hot(mode_onehot: Tensor, name: str) -> Tensor:
    """Reject malformed categorical critic inputs instead of normalizing them."""
    if not isinstance(mode_onehot, Tensor) or not torch.is_floating_point(
        mode_onehot
    ):
        raise InvalidTensorError(
            f"{name} must be a floating-point one-hot tensor, got "
            f"{getattr(mode_onehot, 'dtype', type(mode_onehot).__name__)}"
        )
    _check_finite(mode_onehot, name)
    if not bool(((mode_onehot == 0.0) | (mode_onehot == 1.0)).all()):
        raise InvalidTensorError(f"{name} must contain only exact 0/1 entries")
    if not bool((mode_onehot.sum(dim=1) == 1.0).all()):
        raise InvalidTensorError(f"{name} must contain exactly one active mode per row")
    return mode_onehot


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


@contextmanager
def _frozen_online_critic_parameters(
    critics: "TwinHybridCritics",
) -> Iterator[None]:
    """Freeze critic weights during the actor forward, then restore them.

    Autograd must still differentiate the critic output with respect to the
    actor's continuous action.  Detaching the Q value would destroy that path.
    Marking only the critic parameters non-trainable during this forward keeps
    ``dQ/dq`` while preventing actor backward from allocating or accumulating
    critic parameter gradients.
    """
    parameters = tuple(
        list(critics.critic_1.parameters()) + list(critics.critic_2.parameters())
    )
    prior = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, prior):
            parameter.requires_grad_(requires_grad)


@dataclass(frozen=True)
class HybridSacModelConfig:
    """Shapes and bounds shared by the actor and the critics.

    The defaults are bound to the frozen contracts: ``state_dim`` is the
    31-feature policy vector and ``mode_count`` is the 12-mode catalog.  The
    actor retains its original full-physical-q distribution unless the exact
    hash-bound ``MODELED_SMOKE_SUPPORT`` contract is supplied explicitly.
    """

    state_dim: int = STATE_DIM
    mode_count: int = MODE_COUNT
    hidden_width: int = HIDDEN_WIDTH
    hidden_depth: int = HIDDEN_DEPTH
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    dtype: torch.dtype = torch.float32
    modeled_smoke_support: Optional[ModeledSmokeSupportContract] = None

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
        if self.modeled_smoke_support is not None:
            try:
                require_registered_modeled_smoke_support(
                    self.modeled_smoke_support
                )
            except ModeledSmokeSupportError as exc:
                raise InvalidHyperparameterError(
                    "modeled_smoke_support is not the registered hash-bound "
                    "MODELED_SMOKE_SUPPORT curriculum"
                ) from exc
            if self.mode_count != MODE_COUNT:
                raise InvalidHyperparameterError(
                    "MODELED_SMOKE_SUPPORT is defined only for the frozen "
                    f"{MODE_COUNT}-mode catalog"
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
    """Apply the canonical registered wire quantization element-wise.

    This is the execution/replay-identity boundary and therefore delegates all
    values to :func:`action_contract.round_half_up_q_e4`.  The batched actor
    update uses :func:`_quantize_q_e4_training` below to avoid a device
    synchronization and Python loop for every hypothetical mode.

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
    flat = q.detach().reshape(-1).to(torch.float64).tolist()
    quantized = [round_half_up_q_e4(value) for value in flat]
    return torch.tensor(quantized, dtype=torch.long, device=q.device).reshape(q.shape)


def _quantize_q_e4_training(q: Tensor) -> Tensor:
    """Vectorized quantization for hypothetical all-mode actor samples.

    Default float32 samples are promoted *before* scaling; direct float32
    multiplication can cross a half-grid boundary.  Tests exercise the two
    neighboring float32 values around every one of the 9,800 half-grid
    thresholds against the canonical contract.  The opt-in float64 reference
    path retains the canonical implementation because authored decimal ties
    can otherwise expose binary multiplication rounding.
    """
    if q.dtype == torch.float64:
        return quantize_q_e4(q)
    if not torch.is_floating_point(q):
        raise InvalidTensorError(
            f"q must be a floating-point tensor, got dtype {q.dtype}"
        )
    _check_finite(q, "q")
    scaled = q.detach().to(torch.float64) * float(Q_E4_SCALE)
    return (
        torch.floor(scaled + 0.5)
        .clamp(min=Q_E4_MIN, max=Q_E4_MAX)
        .to(torch.long)
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
    mode_index: Tensor, mode_count: int = MODE_COUNT, dtype: torch.dtype = torch.float32
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
    if mode_index.numel() == 0:
        raise InvalidTensorError("mode_index batch is empty")
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

    Every tensor field is ``(batch, mode_count)``; the coordinate label is a
    string.  Sampling all modes at once is what makes the exact 12-mode
    enumeration in :func:`soft_state_value` and :func:`actor_objective`
    possible.
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
    #: Continuous density in the explicitly named coordinate below.
    log_prob_continuous: Tensor
    #: ``log pi_d(m | s)``.
    log_prob_discrete: Tensor
    #: ``pi_d(m | s)``.
    probs: Tensor
    #: Coordinate of ``log_prob_continuous``; default preserves legacy API.
    continuous_log_prob_coordinate: str = PHYSICAL_Q_DENSITY

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
        support = cfg.modeled_smoke_support
        self._modeled_smoke_support_expected = support is not None
        if support is not None:
            registered = require_registered_modeled_smoke_support(support)
            bounds = torch.tensor(
                registered.mode_q_e4_bounds, dtype=torch.int64
            )
            # Curriculum buffers are persistent: a bounded actor checkpoint is
            # visibly distinct from a default actor checkpoint.  Exact-value
            # load validation below prevents a checkpoint from replacing the
            # registered immutable bounds or digest.
            self.register_buffer(
                "_support_q_e4_lower", bounds[:, 0].clone(), persistent=True
            )
            self.register_buffer(
                "_support_q_e4_upper", bounds[:, 1].clone(), persistent=True
            )
            self.register_buffer(
                "_support_sha256_bytes",
                torch.tensor(
                    list(bytes.fromhex(MODELED_SMOKE_SUPPORT_SHA256)),
                    dtype=torch.uint8,
                ),
                persistent=True,
            )

    @property
    def uses_modeled_smoke_support(self) -> bool:
        """Whether this actor uses the opt-in contextual smoke curriculum."""
        names = (
            "_support_q_e4_lower",
            "_support_q_e4_upper",
            "_support_sha256_bytes",
        )
        present = tuple(
            name in self._buffers and self._buffers[name] is not None
            for name in names
        )
        if any(present) and not all(present):
            raise InvalidTensorError(
                "MODELED_SMOKE_SUPPORT buffer set is incomplete; refusing "
                "to infer default or bounded semantics"
            )
        observed = all(present)
        expected = self._modeled_smoke_support_expected
        if observed != expected:
            raise InvalidTensorError(
                "MODELED_SMOKE_SUPPORT buffer inventory differs from the "
                "actor's construction-time semantics"
            )
        if type(self.config) is not HybridSacModelConfig:
            raise InvalidHyperparameterError(
                "actor config was replaced with a foreign type"
            )
        declared = self.config.modeled_smoke_support is not None
        if declared != expected:
            raise InvalidHyperparameterError(
                "actor config support declaration differs from its "
                "construction-time semantics"
            )
        if declared:
            try:
                require_registered_modeled_smoke_support(
                    self.config.modeled_smoke_support
                )
            except ModeledSmokeSupportError as exc:
                raise InvalidHyperparameterError(
                    "actor config no longer carries registered "
                    "MODELED_SMOKE_SUPPORT"
                ) from exc
        return expected

    @property
    def continuous_density_coordinate(self) -> str:
        """Coordinate used by ``ModeConditionalSample.log_prob_continuous``."""
        if self.uses_modeled_smoke_support:
            return NORMALIZED_Z_DENSITY
        return PHYSICAL_Q_DENSITY

    @property
    def modeled_smoke_support_sha256(self) -> Optional[str]:
        """Registered support digest, or ``None`` for the default actor."""
        if not self.uses_modeled_smoke_support:
            return None
        self._validate_registered_support_buffers()
        return bytes(self._support_sha256_bytes.tolist()).hex()

    def active_q_e4_bounds(self) -> Tuple[Tensor, Tensor]:
        """Return cloned per-mode bounds used by this actor on its device."""
        if self.uses_modeled_smoke_support:
            self._validate_registered_support_buffers()
            return (
                self._support_q_e4_lower.detach().clone(),
                self._support_q_e4_upper.detach().clone(),
            )
        device = next(self.parameters()).device
        return (
            torch.full(
                (MODE_COUNT,), Q_E4_MIN, dtype=torch.int64, device=device
            ),
            torch.full(
                (MODE_COUNT,), Q_E4_MAX, dtype=torch.int64, device=device
            ),
        )

    @staticmethod
    def _expected_support_buffer_values() -> Tuple[Tensor, Tensor, Tensor]:
        """Build exact CPU values from the statically hash-bound contract."""
        bounds = torch.tensor(
            MODELED_SMOKE_SUPPORT.mode_q_e4_bounds, dtype=torch.int64
        )
        digest = torch.tensor(
            list(bytes.fromhex(MODELED_SMOKE_SUPPORT_SHA256)),
            dtype=torch.uint8,
        )
        return bounds[:, 0], bounds[:, 1], digest

    def _validate_registered_support_buffers(self) -> None:
        """Refuse any runtime mutation of the active support or its digest."""
        if not self.uses_modeled_smoke_support:
            return
        expected_values = self._expected_support_buffer_values()
        for name, expected_cpu in zip(
            (
                "_support_q_e4_lower",
                "_support_q_e4_upper",
                "_support_sha256_bytes",
            ),
            expected_values,
        ):
            observed = self._buffers.get(name)
            if (
                not isinstance(observed, Tensor)
                or observed.dtype != expected_cpu.dtype
                or observed.shape != expected_cpu.shape
                or observed.requires_grad
                or not torch.equal(
                    observed, expected_cpu.to(device=observed.device)
                )
            ):
                raise InvalidTensorError(
                    f"{name} differs from registered MODELED_SMOKE_SUPPORT; "
                    "refusing actor output"
                )

    def _preflight_support_state_dict(
        self, state_dict: Mapping[str, Any], prefix: str
    ) -> None:
        """Validate checkpoint support semantics before any tensor is copied."""
        names = (
            "_support_q_e4_lower",
            "_support_q_e4_upper",
            "_support_sha256_bytes",
        )
        expected_keys = {prefix + name for name in names}
        offered_support_keys = {
            key
            for key in state_dict
            if isinstance(key, str)
            and key.startswith(prefix + "_support_")
            and "." not in key[len(prefix) :]
        }
        if self.uses_modeled_smoke_support:
            self._validate_registered_support_buffers()
            if offered_support_keys != expected_keys:
                raise RuntimeError(
                    "checkpoint MODELED_SMOKE_SUPPORT keys differ: expected "
                    f"{sorted(expected_keys)}, got {sorted(offered_support_keys)}"
                )
            expected_values = self._expected_support_buffer_values()
            for key, expected_cpu in zip(
                (prefix + name for name in names), expected_values
            ):
                offered = state_dict[key]
                if (
                    not isinstance(offered, Tensor)
                    or offered.dtype != expected_cpu.dtype
                    or offered.shape != expected_cpu.shape
                    or not torch.equal(
                        offered, expected_cpu.to(device=offered.device)
                    )
                ):
                    raise RuntimeError(
                        f"checkpoint {key} differs from registered "
                        "MODELED_SMOKE_SUPPORT"
                    )
        elif offered_support_keys:
            raise RuntimeError(
                "default actor refuses checkpoint MODELED_SMOKE_SUPPORT keys, "
                "including when strict=False"
            )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Refuse support-semantic drift before this module or children load."""
        self._preflight_support_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _map_pre_squash_to_q(
        self, pre_squash: Tensor, mode_index: Optional[Tensor] = None
    ) -> Tensor:
        """Apply the selected policy's direct squash-to-action mapping."""
        if not self.uses_modeled_smoke_support:
            q = Q_SQUASH_SCALE * (torch.tanh(pre_squash) + 1.0)
            return q.clamp(min=Q_MIN, max=Q_MAX)

        self._validate_registered_support_buffers()
        if (
            self._support_q_e4_lower.device != pre_squash.device
            or self._support_q_e4_upper.device != pre_squash.device
        ):
            raise InvalidTensorError(
                "MODELED_SMOKE_SUPPORT buffers and actor sample must share a device"
            )
        z = 0.5 * (torch.tanh(pre_squash) + 1.0)
        lower_e4 = self._support_q_e4_lower
        upper_e4 = self._support_q_e4_upper
        if mode_index is not None:
            lower_e4 = lower_e4.gather(0, mode_index)
            upper_e4 = upper_e4.gather(0, mode_index)
        lower = lower_e4.to(dtype=pre_squash.dtype) / float(Q_E4_SCALE)
        width = (upper_e4 - lower_e4).to(
            dtype=pre_squash.dtype
        ) / float(Q_E4_SCALE)
        # This is a direct per-mode affine transform.  There is deliberately
        # no global draw followed by clamp/projection onto the support set.
        return lower + width * z

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
        q = self._map_pre_squash_to_q(pre_squash)
        q_e4 = _quantize_q_e4_training(q)
        log_prob_discrete = F.log_softmax(heads.logits, dim=-1)
        if self.uses_modeled_smoke_support:
            log_prob_continuous = normalized_z_log_prob(
                pre_squash, heads.mean, heads.log_std
            )
        else:
            log_prob_continuous = continuous_log_prob(
                pre_squash, heads.mean, heads.log_std
            )
        return ModeConditionalSample(
            pre_squash=pre_squash,
            q=q,
            q_e4=q_e4,
            q_normalized_executed=q_e4.to(q.dtype) / float(Q_E4_MAX),
            q_normalized_straight_through=(
                _straight_through_executed_normalized_q(q, q_e4)
            ),
            log_prob_continuous=log_prob_continuous,
            continuous_log_prob_coordinate=self.continuous_density_coordinate,
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
        q = self._map_pre_squash_to_q(selected_mean, mode_index=mode_index)
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


def normalized_z_log_prob(
    pre_squash: Tensor, mean: Tensor, log_std: Tensor
) -> Tensor:
    """Density in normalized ``z = (tanh(u) + 1) / 2`` coordinates.

    This coordinate is used only by the opt-in modeled-smoke curriculum.  It
    includes the Gaussian density, tanh Jacobian and ``log(0.5)`` scale, but
    intentionally excludes each mode's physical interval width.  Thus
    evidence-artifact support widths cannot introduce a spurious categorical
    entropy preference.  :func:`continuous_log_prob` retains its original
    physical-q semantics unchanged.
    """
    standardized = (pre_squash - mean) / log_std.exp()
    log_normal = -0.5 * standardized.pow(2) - log_std - _HALF_LOG_TWO_PI
    log_tanh_jacobian = 2.0 * (
        _LOG_TWO - pre_squash - F.softplus(-2.0 * pre_squash)
    )
    return (
        log_normal
        - _LOG_NORMALIZED_Z_SQUASH_SCALE
        - log_tanh_jacobian
    )


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
        _check_one_hot(mode_onehot, "mode_onehot")
        _check_normalized_q(q_normalized, "q_normalized")
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
        _check_normalized_q(q_normalized, "q_normalized")
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


@torch.no_grad()
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
        duration: ``(batch,)`` integer-valued hold duration, each at least the
            frozen action-hold minimum (currently two prepared tensors).
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
    if not torch.is_floating_point(reward):
        raise InvalidTensorError(
            f"reward must be floating point, got dtype {reward.dtype}"
        )
    if not torch.is_floating_point(next_value):
        raise InvalidTensorError(
            f"next_value must be floating point, got dtype {next_value.dtype}; "
            "integer Bellman values would truncate gamma"
        )
    expected_device = next_value.device
    for tensor, name in (
        (reward, "reward"),
        (done, "done"),
        (duration, "duration"),
    ):
        if tensor.device != expected_device:
            raise InvalidTensorError(
                f"{name} must be on {expected_device} with next_value, got "
                f"{tensor.device}"
            )
    duration_float = duration.to(next_value.dtype)
    if not torch.equal(duration_float, torch.floor(duration_float)):
        raise InvalidHyperparameterError(
            "duration must be integer-valued (whole prepared-frame intervals)"
        )
    if float(duration_float.min()) < float(MINIMUM_HOLD_TENSORS):
        raise InvalidHyperparameterError(
            f"duration must be at least the frozen {MINIMUM_HOLD_TENSORS}-tensor "
            "action hold"
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

    The returned objective is a quantity to **minimize**.  Online critic
    parameters are frozen only while this graph is built: gradients still flow
    through ``Q`` with respect to the sampled continuous action, but actor
    backward cannot populate or contaminate critic parameter gradients.
    """
    alpha_d = _check_positive_alpha(alpha_d, "alpha_d")
    alpha_c = _check_positive_alpha(alpha_c, "alpha_c")
    sample = actor.sample_all_modes(state, generator=generator)
    with _frozen_online_critic_parameters(critics):
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
