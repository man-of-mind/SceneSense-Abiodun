"""Checkpointable joint SNR/MCS channel for Run-5 modeled training.

Hidden process
--------------
The target SNR follows the registered network-profile design v2 exactly: a
three-state hidden Markov chain (ADVERSE / INTERMEDIATE / FAVORABLE) advanced
once per 100-ms tick, emitting an independent truncated normal on the
registered ``[5.5, 24.5]`` dB support.  All four registered profiles are
used in balanced blocks: the timeline is cut into ``SEGMENT_TICKS`` = 300-tick
segments (the registered capture length) and every consecutive block of four
segments contains each of FAVORABLE_STABLE, MID_VARIABLE, ADVERSE_STABLE and
FADE_RECOVERY exactly once, in a seeded order.  Only the transition matrix
changes at a segment boundary; the hidden state is continuous.

The accepted MCS kernel was fitted on MID_VARIABLE and FADE_RECOVERY.  It
conditions on (MCS, SNR) only and all four profiles share the same three
emission states and support, so FAVORABLE_STABLE and ADVERSE_STABLE use it
under an explicit ``PROFILE_TRANSFER_UNVALIDATED`` label (different dwell
structure, same SNR support).

Successor MCS
-------------
``P(MCS' | MCS, SNR_current)`` is the accepted exponential tilt of the exact
Run-4 kernel (``successor_mcs_snr_audit``).  It supports duration 2 only, so
one decision advances the SNR process by exactly two ticks and draws exactly
one successor MCS.

Causality
---------
At decision time the channel exposes only ``(current_snr_db, current_mcs)``.
Every future SNR sample is generated inside :meth:`advance`, which the
collector calls only after the action's outcome is resolved; generated tick
indices are asserted to be strictly later than the observed tick.  Profile,
hidden state, segment position and future samples are never returned by any
actor-facing method.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as P

from . import run5_snr_v2 as SNR
from . import successor_mcs_snr_audit as SA

SCHEMA_ID = "splitfusion.run5.joint_snr_mcs_channel.v1"
PACKAGE = Path(__file__).resolve().parent
WORKTREE_ROOT = PACKAGE.parents[1]
ACCEPTED_AUDIT_FILENAME = SA.AUDIT_FILENAME
ACCEPTED_AUDIT_SHA256 = "295ca76206d9dc9a5181a871c7644f9545edff384213d3e3a01c0886e4b683a6"
TRAINING_PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")
KERNEL_FIT_PROFILES = ("MID_VARIABLE", "FADE_RECOVERY")
PROFILE_KERNEL_STATUS = {
    profile: ("KERNEL_FIT_SUPPORT" if profile in KERNEL_FIT_PROFILES
              else "PROFILE_TRANSFER_UNVALIDATED") for profile in TRAINING_PROFILES}
SEGMENT_TICKS = 300
BURN_IN_TRANSITIONS = 10
TICKS_PER_TENSOR = 1
SUPPORTED_DURATION_TENSORS = 2
_NORMAL = statistics.NormalDist()


class ChannelError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChannelError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def derive_seed(master: int, label: str) -> int:
    material = json.dumps({"domain": "RUN5_JOINT_CHANNEL_V1", "label": label,
                           "master_seed": master}, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def load_design(root: Path = WORKTREE_ROOT) -> dict[str, Any]:
    data = (root / SNR.NETWORK_PROFILE_DESIGN_RELPATH).read_bytes()
    require(hashlib.sha256(data).hexdigest() == SNR.NETWORK_PROFILE_DESIGN_SHA256,
            "network profile design differs from the registered digest")
    design = json.loads(data)
    target = design["target_snr"]
    require(float(target["lower_bound_db"]) == SNR.SNR_SUPPORT_MIN_DB
            and float(target["upper_bound_db"]) == SNR.SNR_SUPPORT_MAX_DB,
            "design support differs from the Run-5 contract")
    return design


def load_accepted_kernel(evidence_root: Path) -> SA.SnrTiltedKernelV1:
    """Refit on FIT only and require equality with the committed acceptance."""
    path = PACKAGE / ACCEPTED_AUDIT_FILENAME
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ACCEPTED_AUDIT_SHA256,
            "successor-MCS acceptance artifact differs from the registered digest")
    audit = json.loads(path.read_text())
    require(audit["gate_passed"] is True, "successor-MCS kernel gate did not pass")
    _, base, kernel, _, _, _ = SA.fit_all(Path(evidence_root))
    require(kernel.document() == audit["augmentation"],
            "refitted SNR-tilted kernel differs from the accepted kernel")
    require(base.binding_sha256 == SA.REGISTERED_RUN4_KERNEL_BINDING, "Run-4 base drifted")
    return kernel


def _stationary(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix.T)
    vector = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    vector = vector / vector.sum()
    require(bool(np.all(vector >= -1e-12)), "stationary distribution is not a distribution")
    return np.clip(vector, 0.0, None) / np.clip(vector, 0.0, None).sum()


def _choice(probabilities, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, value in enumerate(probabilities):
        cumulative += float(value)
        if threshold < cumulative:
            return index
    return len(probabilities) - 1


@dataclass(frozen=True, slots=True)
class ChannelObservationV1:
    """The complete actor-facing channel surface."""

    snr_db: float
    mcs: int
    tick: int


@dataclass(frozen=True, slots=True)
class ChannelStepV1:
    current_snr_db: float
    current_mcs: int
    current_tick: int
    successor_snr_db: float
    successor_mcs: int
    successor_tick: int
    generated_ticks: tuple[int, ...]


class JointSnrMcsChannelV1:
    def __init__(self, *, kernel: SA.SnrTiltedKernelV1, design: Mapping[str, Any],
                 seed: int) -> None:
        require(type(seed) is int and seed >= 0, "seed must be an exact int >= 0")
        self._kernel = kernel
        target = design["target_snr"]
        self._means = [float(v) for v in target["state_means_db"]]
        self._sigmas = [float(v) for v in target["state_sigma_db"]]
        profiles = {p["profile_id"]: p for p in design["profiles"]}
        require(set(profiles) == set(TRAINING_PROFILES),
                "the design must register exactly the four training profiles")
        self._matrices = [np.asarray(profiles[p]["transition_matrix"], dtype=float)
                          for p in TRAINING_PROFILES]
        self._seed = seed
        self.binding_sha256 = _sha({
            "schema": SCHEMA_ID, "kernel": kernel.document(),
            "design_sha256": SNR.NETWORK_PROFILE_DESIGN_SHA256,
            "profiles": list(TRAINING_PROFILES), "profile_block": "balanced_4_segment_permutation",
            "profile_kernel_status": PROFILE_KERNEL_STATUS, "segment_ticks": SEGMENT_TICKS,
            "burn_in_transitions": BURN_IN_TRANSITIONS,
            "ticks_per_tensor": TICKS_PER_TENSOR,
            "accepted_audit_sha256": ACCEPTED_AUDIT_SHA256})
        self._reset()

    # -- hidden process -------------------------------------------------
    def _reset(self) -> None:
        self._profile_rng = random.Random(derive_seed(self._seed, "profile"))
        self._snr_rng = random.Random(derive_seed(self._seed, "snr"))
        self._mcs_rng = random.Random(derive_seed(self._seed, "mcs"))
        self._block: list[int] = []
        self._segments_started = 0
        self._profile_segments = [0] * len(TRAINING_PROFILES)
        self._profile = self._next_block_profile()
        self._segment_left = SEGMENT_TICKS
        self._state = _choice(_stationary(self._matrices[self._profile]), self._snr_rng)
        self._tick = 0
        self._snr = self._emit()
        self._mcs = P.MCS_MIN + P._draw_index(self._kernel.base.initial_counts, self._mcs_rng)
        self._transitions = 0
        self._observed_tick = -1
        self._future_violations = 0
        for _ in range(BURN_IN_TRANSITIONS):
            self._advance_internal(SUPPORTED_DURATION_TENSORS)
        self._transitions = 0

    def _emit(self) -> float:
        mean, sigma = self._means[self._state], self._sigmas[self._state]
        low = _NORMAL.cdf((SNR.SNR_SUPPORT_MIN_DB - mean) / sigma)
        high = _NORMAL.cdf((SNR.SNR_SUPPORT_MAX_DB - mean) / sigma)
        value = mean + sigma * _NORMAL.inv_cdf(low + self._snr_rng.random() * (high - low))
        require(SNR.SNR_SUPPORT_MIN_DB <= value <= SNR.SNR_SUPPORT_MAX_DB,
                "emitted SNR left the registered support")
        return float(value)

    def _next_block_profile(self) -> int:
        if not self._block:
            self._block = list(range(len(TRAINING_PROFILES)))
            self._profile_rng.shuffle(self._block)
        profile = self._block.pop(0)
        self._segments_started += 1
        self._profile_segments[profile] += 1
        return profile

    def _tick_once(self) -> int:
        self._segment_left -= 1
        if self._segment_left == 0:
            self._profile = self._next_block_profile()
            self._segment_left = SEGMENT_TICKS
        self._state = _choice(self._matrices[self._profile][self._state], self._snr_rng)
        self._tick += 1
        self._snr = self._emit()
        return self._tick

    def _advance_internal(self, duration_tensors: int) -> ChannelStepV1:
        require(duration_tensors == SUPPORTED_DURATION_TENSORS,
                "the accepted MCS kernel supports duration 2 only")
        current = (self._snr, self._mcs, self._tick)
        probabilities = self._kernel.probabilities(self._mcs, self._snr)
        successor_mcs = P.MCS_MIN + P._draw_index(probabilities, self._mcs_rng)
        generated = tuple(self._tick_once()
                          for _ in range(duration_tensors * TICKS_PER_TENSOR))
        self._mcs = successor_mcs
        self._transitions += 1
        return ChannelStepV1(current[0], current[1], current[2], self._snr, self._mcs,
                             self._tick, generated)

    # -- actor-facing ---------------------------------------------------
    def observe(self) -> ChannelObservationV1:
        self._observed_tick = self._tick
        return ChannelObservationV1(self._snr, self._mcs, self._tick)

    def advance(self, duration_tensors: int) -> ChannelStepV1:
        """Call only after the action at the observed tick has been resolved."""
        require(self._observed_tick == self._tick,
                "advance() requires the current state to have been observed")
        step = self._advance_internal(duration_tensors)
        if any(tick <= step.current_tick for tick in step.generated_ticks):
            self._future_violations += 1
        require(self._future_violations == 0, "a generated SNR tick is not in the future")
        return step

    @property
    def future_sample_violations(self) -> int:
        return self._future_violations

    def profile_balance(self) -> dict[str, int]:
        """Audit-only segment counts; never an actor input."""
        return {name: count for name, count in zip(TRAINING_PROFILES, self._profile_segments)}

    @property
    def transitions(self) -> int:
        return self._transitions

    # -- durable state --------------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID, "binding_sha256": self.binding_sha256, "seed": self._seed,
            "tick": self._tick, "observed_tick": self._observed_tick,
            "hidden_state": self._state, "hidden_profile_index": self._profile,
            "segment_left": self._segment_left, "snr_db_hex": float(self._snr).hex(),
            "profile_block_remaining": list(self._block),
            "segments_started": self._segments_started,
            "profile_segments": list(self._profile_segments),
            "mcs": self._mcs, "transitions": self._transitions,
            "future_violations": self._future_violations,
            "rng_states": {name: _state_document(getattr(self, f"_{name}_rng"))
                           for name in ("profile", "snr", "mcs")},
        }

    def checkpoint_sha256(self) -> str:
        return _sha(self.checkpoint())

    def restore(self, document: Mapping[str, Any]) -> None:
        require(document.get("schema") == SCHEMA_ID, "foreign channel checkpoint")
        require(document["binding_sha256"] == self.binding_sha256, "channel binding differs")
        require(document["seed"] == self._seed, "channel seed differs")
        for name in ("profile", "snr", "mcs"):
            getattr(self, f"_{name}_rng").setstate(_state_from_document(
                document["rng_states"][name]))
        self._tick = int(document["tick"])
        self._observed_tick = int(document["observed_tick"])
        self._state = int(document["hidden_state"])
        self._profile = int(document["hidden_profile_index"])
        self._segment_left = int(document["segment_left"])
        self._block = [int(v) for v in document["profile_block_remaining"]]
        self._segments_started = int(document["segments_started"])
        self._profile_segments = [int(v) for v in document["profile_segments"]]
        self._snr = float.fromhex(document["snr_db_hex"])
        self._mcs = int(document["mcs"])
        self._transitions = int(document["transitions"])
        self._future_violations = int(document["future_violations"])


def _state_document(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), None if gauss is None else float(gauss).hex()]


def _state_from_document(value) -> tuple:
    version, internal, gauss = value
    return (int(version), tuple(int(v) for v in internal),
            None if gauss is None else float.fromhex(gauss))
