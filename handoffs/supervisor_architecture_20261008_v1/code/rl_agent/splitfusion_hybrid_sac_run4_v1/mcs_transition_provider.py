"""Checkpointable FIT-only prior-UL-MCS dynamics for Run 4.

The accepted target-radio evidence contains finite measured sequences.  A
training run is longer than those sequences, so looping them would manufacture
periodicity.  This module instead fits a first-order duration-two Markov model
from *only* the registered FIT transitions.  Sparse rows shrink toward the
pooled ordered-MCS jump distribution; the held-out transitions are used only
by :func:`evaluate_internal_validation`.

Generated values are explicitly model-derived, not measured UE grants.  The
provider exposes only the MCS index.  Profile, SNR, timestamps and source-grant
identity never enter its output.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from . import dynamic_mcs_273prb_evidence as evidence_module

SCHEMA_ID = "splitfusion_run4_fit_mcs_markov_provider_v1"
SCHEMA_VERSION = 1
EVIDENCE_CLASS = "MODEL_DERIVED_FROM_TARGET_RADIO_FIT_MCS_TRANSITIONS"
MCS_MIN = 9
MCS_MAX = 28
STATE_COUNT = MCS_MAX - MCS_MIN + 1
DELTA_MIN = MCS_MIN - MCS_MAX
DELTA_MAX = MCS_MAX - MCS_MIN
DELTA_COUNT = DELTA_MAX - DELTA_MIN + 1
BACKOFF_STRENGTH = 1.0


class McsTransitionProviderError(ValueError):
    """The fitted model, checkpoint or requested transition is invalid."""


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise McsTransitionProviderError("value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _exact_mcs(value: object, name: str) -> int:
    if type(value) is not int or not MCS_MIN <= value <= MCS_MAX:
        raise McsTransitionProviderError(
            f"{name} must be an exact MCS index in [{MCS_MIN},{MCS_MAX}]"
        )
    return value


def _exact_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise McsTransitionProviderError(f"{name} must be an exact count >= 0")
    return value


@dataclass(frozen=True, slots=True)
class FitMcsMarkovModelV1:
    """Immutable sufficient statistics for a duration-two MCS kernel."""

    transition_counts: tuple[tuple[int, ...], ...]
    delta_counts: tuple[int, ...]
    initial_counts: tuple[int, ...]
    source_evidence_sha256: str
    backoff_strength: float = BACKOFF_STRENGTH

    def __post_init__(self) -> None:
        if len(self.transition_counts) != STATE_COUNT or any(
            len(row) != STATE_COUNT for row in self.transition_counts
        ):
            raise McsTransitionProviderError("transition matrix shape drifted")
        if len(self.delta_counts) != DELTA_COUNT:
            raise McsTransitionProviderError("delta-count support drifted")
        if len(self.initial_counts) != STATE_COUNT:
            raise McsTransitionProviderError("initial-count support drifted")
        for row_index, row in enumerate(self.transition_counts):
            for column_index, value in enumerate(row):
                _exact_count(value, f"transition_counts[{row_index}][{column_index}]")
        for index, value in enumerate(self.delta_counts):
            _exact_count(value, f"delta_counts[{index}]")
        for index, value in enumerate(self.initial_counts):
            _exact_count(value, f"initial_counts[{index}]")
        if sum(self.delta_counts) <= 0 or sum(self.initial_counts) <= 0:
            raise McsTransitionProviderError("FIT statistics are empty")
        if (
            not isinstance(self.backoff_strength, float)
            or not math.isfinite(self.backoff_strength)
            or self.backoff_strength <= 0.0
        ):
            raise McsTransitionProviderError("backoff_strength must be finite and > 0")
        if (
            not isinstance(self.source_evidence_sha256, str)
            or len(self.source_evidence_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_evidence_sha256)
        ):
            raise McsTransitionProviderError("source evidence digest is malformed")
        if sum(sum(row) for row in self.transition_counts) != sum(self.delta_counts):
            raise McsTransitionProviderError("matrix and delta counts disagree")

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256({
            "backoff_strength": self.backoff_strength,
            "delta_counts": self.delta_counts,
            "duration_tensors": 2,
            "evidence_class": EVIDENCE_CLASS,
            "initial_counts": self.initial_counts,
            "mcs_support": [MCS_MIN, MCS_MAX],
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "source_evidence_sha256": self.source_evidence_sha256,
            "transition_counts": self.transition_counts,
        })

    def weights(self, current_mcs: int) -> tuple[float, ...]:
        """Return conditional weights with ordered-jump empirical backoff."""
        current = _exact_mcs(current_mcs, "current_mcs")
        row = self.transition_counts[current - MCS_MIN]
        delta_total = sum(self.delta_counts)
        weights = []
        for successor in range(MCS_MIN, MCS_MAX + 1):
            delta = successor - current
            delta_weight = self.delta_counts[delta - DELTA_MIN] / delta_total
            weights.append(
                float(row[successor - MCS_MIN])
                + self.backoff_strength * delta_weight
            )
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise McsTransitionProviderError("conditional row has zero mass")
        return tuple(weights)

    def probabilities(self, current_mcs: int) -> tuple[float, ...]:
        weights = self.weights(current_mcs)
        total = sum(weights)
        return tuple(value / total for value in weights)

    def initial_probabilities(self) -> tuple[float, ...]:
        total = sum(self.initial_counts)
        return tuple(value / total for value in self.initial_counts)


def fit_mcs_markov_model(
    evidence: evidence_module.DynamicMcs273PrbEvidenceV1,
) -> FitMcsMarkovModelV1:
    """Fit from the attested FIT partition; validation is never consulted."""
    if type(evidence) is not evidence_module.DynamicMcs273PrbEvidenceV1:
        raise McsTransitionProviderError("evidence has the wrong exact type")
    matrix = [[0 for _ in range(STATE_COUNT)] for _ in range(STATE_COUNT)]
    deltas = [0 for _ in range(DELTA_COUNT)]
    initials = [0 for _ in range(STATE_COUNT)]
    for sequence in evidence.fit_sequences:
        for transition in sequence.transitions:
            if not transition.learning_eligible:
                raise McsTransitionProviderError("FIT evidence contains ineligible transition")
            current, successor = transition.policy_values()
            _exact_mcs(current, "FIT current MCS")
            _exact_mcs(successor, "FIT successor MCS")
            matrix[current - MCS_MIN][successor - MCS_MIN] += 1
            # Draw genesis from the empirical FIT current-state marginal, not
            # only the first sample of each hidden profile. This preserves
            # observed support without exposing profile identity.
            initials[current - MCS_MIN] += 1

            deltas[successor - current - DELTA_MIN] += 1
    return FitMcsMarkovModelV1(
        transition_counts=tuple(tuple(row) for row in matrix),
        delta_counts=tuple(deltas),
        initial_counts=tuple(initials),
        source_evidence_sha256=evidence.canonical_evidence_sha256,
    )


@dataclass(frozen=True, slots=True)
class McsProviderCheckpointV1:
    """Exact local-RNG checkpoint; no global random state is involved."""

    model_binding_sha256: str
    current_mcs: Optional[int]
    transitions_emitted: int
    rng_state: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_binding_sha256, str) or len(
            self.model_binding_sha256
        ) != 64:
            raise McsTransitionProviderError("checkpoint binding is malformed")
        if self.current_mcs is not None:
            _exact_mcs(self.current_mcs, "checkpoint current_mcs")
        _exact_count(self.transitions_emitted, "transitions_emitted")
        if type(self.rng_state) is not tuple:
            raise McsTransitionProviderError("rng_state must be an exact tuple")


@dataclass(frozen=True, slots=True)
class ModelDerivedMcsStepV1:
    current_mcs: int
    successor_mcs: int
    duration_tensors: int
    evidence_class: str = EVIDENCE_CLASS

    def __post_init__(self) -> None:
        _exact_mcs(self.current_mcs, "current_mcs")
        _exact_mcs(self.successor_mcs, "successor_mcs")
        if self.duration_tensors != 2:
            raise McsTransitionProviderError("provider supports duration_tensors=2 only")
        if self.evidence_class != EVIDENCE_CLASS:
            raise McsTransitionProviderError("evidence class drifted")


def _draw_index(weights: Sequence[float], rng: random.Random) -> int:
    total = float(sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise McsTransitionProviderError("draw weights have no finite mass")
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        if not math.isfinite(weight) or weight < 0.0:
            raise McsTransitionProviderError("draw weight is invalid")
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(weights) - 1


class FitMcsMarkovProviderV1:
    """Arbitrary-length model-derived MCS generator with exact resume."""

    def __init__(self, model: FitMcsMarkovModelV1, *, seed: int) -> None:
        if type(model) is not FitMcsMarkovModelV1:
            raise McsTransitionProviderError("model has the wrong exact type")
        if type(seed) is not int or seed < 0:
            raise McsTransitionProviderError("seed must be an exact int >= 0")
        self._model = model
        self._rng = random.Random(seed)
        self._current_mcs: Optional[int] = None
        self._transitions_emitted = 0

    @property
    def model_binding_sha256(self) -> str:
        return self._model.binding_sha256

    @property
    def current_mcs(self) -> Optional[int]:
        return self._current_mcs

    def reset(self) -> int:
        index = _draw_index(self._model.initial_counts, self._rng)
        self._current_mcs = MCS_MIN + index
        return self._current_mcs

    def step(self) -> ModelDerivedMcsStepV1:
        if self._current_mcs is None:
            raise McsTransitionProviderError("reset() is required before step()")
        current = self._current_mcs
        index = _draw_index(self._model.weights(current), self._rng)
        successor = MCS_MIN + index
        self._current_mcs = successor
        self._transitions_emitted += 1
        return ModelDerivedMcsStepV1(current, successor, 2)

    def checkpoint(self) -> McsProviderCheckpointV1:
        return McsProviderCheckpointV1(
            model_binding_sha256=self._model.binding_sha256,
            current_mcs=self._current_mcs,
            transitions_emitted=self._transitions_emitted,
            rng_state=self._rng.getstate(),
        )

    def restore(self, checkpoint: McsProviderCheckpointV1) -> None:
        if type(checkpoint) is not McsProviderCheckpointV1:
            raise McsTransitionProviderError("checkpoint has the wrong exact type")
        if checkpoint.model_binding_sha256 != self._model.binding_sha256:
            raise McsTransitionProviderError("checkpoint/model binding differs")
        probe = random.Random()
        try:
            probe.setstate(checkpoint.rng_state)
        except (TypeError, ValueError) as exc:
            raise McsTransitionProviderError("checkpoint RNG state is invalid") from exc
        self._rng.setstate(checkpoint.rng_state)
        self._current_mcs = checkpoint.current_mcs
        self._transitions_emitted = checkpoint.transitions_emitted


def evaluate_internal_validation(
    model: FitMcsMarkovModelV1,
    evidence: evidence_module.DynamicMcs273PrbEvidenceV1,
) -> dict[str, Any]:
    """Score the untouched validation split without changing the model."""
    if type(model) is not FitMcsMarkovModelV1:
        raise McsTransitionProviderError("model has the wrong exact type")
    if type(evidence) is not evidence_module.DynamicMcs273PrbEvidenceV1:
        raise McsTransitionProviderError("evidence has the wrong exact type")
    if model.source_evidence_sha256 != evidence.canonical_evidence_sha256:
        raise McsTransitionProviderError("model/evidence binding differs")
    negative_log_likelihood = 0.0
    brier = 0.0
    top1 = 0
    persistence = 0
    count = 0
    validation_current_states: set[int] = set()
    for transition in evidence.internal_validation_transitions:
        current, successor = transition.policy_values()
        probabilities = model.probabilities(current)
        observed_index = successor - MCS_MIN
        probability = probabilities[observed_index]
        if probability <= 0.0:
            raise McsTransitionProviderError("validation transition has zero probability")
        negative_log_likelihood -= math.log(probability)
        brier += sum(
            (value - (1.0 if index == observed_index else 0.0)) ** 2
            for index, value in enumerate(probabilities)
        )
        top1 += int(max(range(STATE_COUNT), key=probabilities.__getitem__) == observed_index)
        persistence += int(current == successor)
        validation_current_states.add(current)
        count += 1
    if count == 0:
        raise McsTransitionProviderError("validation split is empty")
    return {
        "brier_mean": brier / count,
        "evidence_class": EVIDENCE_CLASS,
        "fit_transitions": len(evidence.fit_transitions),
        "mean_negative_log_likelihood": negative_log_likelihood / count,
        "model_binding_sha256": model.binding_sha256,
        "persistence_accuracy": persistence / count,
        "top1_accuracy": top1 / count,
        "validation_current_states": sorted(validation_current_states),
        "validation_transitions": count,
    }
