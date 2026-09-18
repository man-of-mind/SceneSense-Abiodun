"""Deterministic test-only event environment for the Hybrid-SAC contracts.

This module is deliberately **not** a scientific simulator.  Every state and
outcome it emits is labelled ``SYNTHETIC_CONTRACT_TEST_ONLY`` and
``SYNTHETIC_FIXTURE``.  Its only purpose is to exercise the already-frozen
action and reward-ticket mechanics against a continuous analytic surface.

In particular, this module never constructs
``ReplayTransitionV1``, never claims CARLA/OAI/MEASURED provenance, and never
turns its analytic quality value into a source-authenticated positive reward.
The current exact-positive reward path remains fail-closed as required by the
production contract.

The event loop is single-threaded and serialized.  At 10 Hz it:

* admits each prepared frame through the real :class:`RewardTicketController`;
* opens at most one reward ticket;
* reuses the exact executed action while the ticket is held;
* requests reward only on the opening tensor;
* observes the inclusive 200 ms deadline and the two-tensor minimum hold; and
* records duplicate, late-orphan and timeout classifications from the real
  controller rather than reimplementing those rules.

Random-looking fixture variation uses a stateless SHA-256 counter function.
There is no module-global or Python-global pseudo-random generator state.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

from .action_contract import (
    EXPECTED_MODE_COUNT,
    Q_MAX,
    Q_MIN,
    SplitActionContract,
    default_contract,
)
from .reward_ticket_controller import (
    B_REWARD_DEADLINE_NS,
    AdmissionDisposition,
    CompletedTicket,
    FeedbackDisposition,
    FeedbackTerminalStatus,
    RewardFeedbackMessage,
    RewardTicketController,
    TerminalClass,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    PREVIOUS_TERMINAL_FEATURE_CODES,
    PREVIOUS_TERMINAL_ORDER,
)
from .transaction_identity import (
    ExecutedActionIdentity,
    RewardFeedbackIdentity,
    canonical_sha256,
)

__all__ = [
    "SyntheticContractError",
    "SYNTHETIC_EVIDENCE_CLASS",
    "SYNTHETIC_FIXTURE_LABEL",
    "FRAME_PERIOD_NS",
    "SyntheticTracePoint",
    "SyntheticPolicyObservation",
    "StateTrace",
    "SyntheticActionChoice",
    "SyntheticPolicy",
    "SyntheticDecisionContext",
    "SyntheticFeedbackScenario",
    "SyntheticOutcomePlan",
    "OutcomeProvider",
    "AnalyticStateTrace",
    "AnalyticOutcomeProvider",
    "CyclingFixturePolicy",
    "AnalyticOracleFixturePolicy",
    "SyntheticEventKind",
    "SyntheticEventRecord",
    "SyntheticFixtureValueRecord",
    "SyntheticRunReport",
    "SyntheticContractEnvironment",
    "counter_uniform",
]


SYNTHETIC_EVIDENCE_CLASS = "SYNTHETIC_CONTRACT_TEST_ONLY"
SYNTHETIC_FIXTURE_LABEL = "SYNTHETIC_FIXTURE"
FRAME_PERIOD_NS = 100_000_000


class SyntheticContractError(ValueError):
    """A test fixture violates the synthetic environment boundary."""


def _exact_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise SyntheticContractError(
            f"{name} must be an exact non-negative int, got {value!r}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SyntheticContractError(f"{name} must be finite, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SyntheticContractError(f"{name} must be finite, got {value!r}")
    return result


def _in_unit_interval(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise SyntheticContractError(f"{name} must be in [0, 1], got {result}")
    return result


def _sha256_hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SyntheticContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def counter_uniform(seed: str, stream: str, *counters: int) -> float:
    """Return a deterministic value in ``[0, 1)`` from an explicit key.

    This is a counter function, not a stateful generator: the result depends
    only on ``seed``, ``stream`` and the exact counters.  Call order and any
    external use of :mod:`random` therefore cannot change a fixture.
    """

    if not isinstance(seed, str) or not seed:
        raise SyntheticContractError("counter seed must be a non-empty string")
    if not isinstance(stream, str) or not stream:
        raise SyntheticContractError("counter stream must be a non-empty string")
    normalized = tuple(
        _exact_non_negative_int(counter, f"counter[{index}]")
        for index, counter in enumerate(counters)
    )
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {"counters": normalized, "seed": seed, "stream": stream}
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


@dataclass(frozen=True, slots=True)
class SyntheticTracePoint:
    """One causal, current-frame-only synthetic observation."""

    frame_index: int
    carla_frame_id: int
    observed_ns: int
    camera_si_normalized: float
    radar_p40: float
    achieved_snr_db: float
    bsr_bytes: int
    mcs_index: int
    evidence_class: str = SYNTHETIC_EVIDENCE_CLASS
    fixture_label: str = SYNTHETIC_FIXTURE_LABEL

    def __post_init__(self) -> None:
        _exact_non_negative_int(self.frame_index, "frame_index")
        _exact_non_negative_int(self.carla_frame_id, "carla_frame_id")
        _exact_non_negative_int(self.observed_ns, "observed_ns")
        _in_unit_interval(self.camera_si_normalized, "camera_si_normalized")
        _in_unit_interval(self.radar_p40, "radar_p40")
        _finite(self.achieved_snr_db, "achieved_snr_db")
        _exact_non_negative_int(self.bsr_bytes, "bsr_bytes")
        _exact_non_negative_int(self.mcs_index, "mcs_index")
        if self.mcs_index > 28:
            raise SyntheticContractError("mcs_index must be in [0, 28]")
        if self.evidence_class != SYNTHETIC_EVIDENCE_CLASS:
            raise SyntheticContractError(
                "synthetic trace cannot claim another evidence class"
            )
        if self.fixture_label != SYNTHETIC_FIXTURE_LABEL:
            raise SyntheticContractError(
                "synthetic trace must retain the SYNTHETIC_FIXTURE label"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db": self.achieved_snr_db,
            "bsr_bytes": self.bsr_bytes,
            "camera_si_normalized": self.camera_si_normalized,
            "carla_frame_id": self.carla_frame_id,
            "evidence_class": self.evidence_class,
            "fixture_label": self.fixture_label,
            "frame_index": self.frame_index,
            "mcs_index": self.mcs_index,
            "observed_ns": self.observed_ns,
            "radar_p40": self.radar_p40,
            "record": "synthetic_trace_point_v1",
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class SyntheticPolicyObservation:
    """Test-only policy input with exactly the frozen feature values.

    This boundary deliberately exposes only the ordered scalar features named
    by :data:`POLICY_FEATURE_ORDER`.  Frame/tensor/decision identifiers, raw
    timestamps, evidence hashes and future state are unavailable to the
    policy.  The production contract remains the authority for constructing
    an attested :class:`PolicyFeatureVectorV1`; this fixture merely exercises
    policy mechanics without manufacturing such an attestation.
    """

    values: Tuple[float, ...]
    evidence_class: str = SYNTHETIC_EVIDENCE_CLASS
    fixture_label: str = SYNTHETIC_FIXTURE_LABEL

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or len(self.values) != POLICY_FEATURE_COUNT:
            raise SyntheticContractError(
                f"synthetic policy observation requires {POLICY_FEATURE_COUNT} "
                f"ordered values"
            )
        for name, value in zip(POLICY_FEATURE_ORDER, self.values):
            _finite(value, f"policy feature {name}")
        if self.evidence_class != SYNTHETIC_EVIDENCE_CLASS:
            raise SyntheticContractError("synthetic policy observation changed evidence class")
        if self.fixture_label != SYNTHETIC_FIXTURE_LABEL:
            raise SyntheticContractError("synthetic policy observation changed fixture label")

    def as_tuple(self) -> Tuple[float, ...]:
        return self.values

    def as_mapping(self) -> Dict[str, float]:
        return dict(zip(POLICY_FEATURE_ORDER, self.values))

    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "evidence_class": self.evidence_class,
                "feature_order": list(POLICY_FEATURE_ORDER),
                "fixture_label": self.fixture_label,
                "record": "synthetic_policy_observation_v1",
                "values": list(self.values),
            }
        )


@runtime_checkable
class StateTrace(Protocol):
    """Typed boundary supplying one observation for the current frame only."""

    @property
    def trace_id(self) -> str:
        ...

    def observation(self, frame_index: int) -> SyntheticTracePoint:
        ...


@dataclass(frozen=True, slots=True)
class SyntheticActionChoice:
    """Bounded policy request before the production action adapter."""

    mode_id: int
    q: float

    def __post_init__(self) -> None:
        _exact_non_negative_int(self.mode_id, "mode_id")
        if self.mode_id >= EXPECTED_MODE_COUNT:
            raise SyntheticContractError(
                f"mode_id must be below {EXPECTED_MODE_COUNT}, got {self.mode_id}"
            )
        q = _finite(self.q, "q")
        if not Q_MIN <= q <= Q_MAX:
            raise SyntheticContractError(
                f"q must be in the actor range [{Q_MIN}, {Q_MAX}], got {q}"
            )


@runtime_checkable
class SyntheticPolicy(Protocol):
    """Test-only policy boundary; no actor, critic or training is implied."""

    def choose(self, state: SyntheticPolicyObservation) -> SyntheticActionChoice:
        ...


@dataclass(frozen=True, slots=True)
class SyntheticDecisionContext:
    """Current state/action pair exposed to an outcome fixture after execution."""

    decision_seq: int
    state: SyntheticTracePoint
    action: ExecutedActionIdentity

    def __post_init__(self) -> None:
        _exact_non_negative_int(self.decision_seq, "decision_seq")
        if not isinstance(self.state, SyntheticTracePoint):
            raise SyntheticContractError("state must be SyntheticTracePoint")
        if not isinstance(self.action, ExecutedActionIdentity):
            raise SyntheticContractError("action must be ExecutedActionIdentity")
        self.action.require_reconciled()


class SyntheticFeedbackScenario(Enum):
    """Explicit event patterns used only by contract tests."""

    ANALYTIC_TIMELY = "ANALYTIC_TIMELY"
    EARLY_BEFORE_HOLD = "EARLY_BEFORE_HOLD"
    EARLY_WITH_DUPLICATE = "EARLY_WITH_DUPLICATE"
    EXACT_DEADLINE = "EXACT_DEADLINE"
    LATE_AFTER_TIMEOUT = "LATE_AFTER_TIMEOUT"
    NO_FEEDBACK_TIMEOUT = "NO_FEEDBACK_TIMEOUT"
    ACTION_PATH_FAILURE = "ACTION_PATH_FAILURE"


@dataclass(frozen=True, slots=True)
class SyntheticOutcomePlan:
    """Test-only outcome schedule plus an analytic, non-evidentiary value."""

    fixture_quality: float
    preferred_mode_id: int
    preferred_q: float
    feedback_status: Optional[FeedbackTerminalStatus]
    feedback_delay_ns: Optional[int]
    duplicate_after_ns: Tuple[int, ...] = ()
    evidence_class: str = SYNTHETIC_EVIDENCE_CLASS
    fixture_label: str = SYNTHETIC_FIXTURE_LABEL

    def __post_init__(self) -> None:
        _in_unit_interval(self.fixture_quality, "fixture_quality")
        _exact_non_negative_int(self.preferred_mode_id, "preferred_mode_id")
        if self.preferred_mode_id >= EXPECTED_MODE_COUNT:
            raise SyntheticContractError("preferred_mode_id is out of range")
        preferred_q = _finite(self.preferred_q, "preferred_q")
        if not Q_MIN <= preferred_q <= Q_MAX:
            raise SyntheticContractError("preferred_q is outside actor bounds")
        if (self.feedback_status is None) != (self.feedback_delay_ns is None):
            raise SyntheticContractError(
                "feedback_status and feedback_delay_ns must be both present or both absent"
            )
        if self.feedback_status is not None and not isinstance(
            self.feedback_status, FeedbackTerminalStatus
        ):
            raise SyntheticContractError("invalid feedback terminal status")
        if self.feedback_delay_ns is not None:
            _exact_non_negative_int(self.feedback_delay_ns, "feedback_delay_ns")
        if not isinstance(self.duplicate_after_ns, tuple):
            raise SyntheticContractError("duplicate_after_ns must be a tuple")
        for index, delay in enumerate(self.duplicate_after_ns):
            _exact_non_negative_int(delay, f"duplicate_after_ns[{index}]")
            if delay == 0:
                raise SyntheticContractError("duplicate offsets must be positive")
        if self.feedback_status is None and self.duplicate_after_ns:
            raise SyntheticContractError("no-feedback plan cannot schedule duplicates")
        if self.evidence_class != SYNTHETIC_EVIDENCE_CLASS:
            raise SyntheticContractError("synthetic outcome cannot claim measured evidence")
        if self.fixture_label != SYNTHETIC_FIXTURE_LABEL:
            raise SyntheticContractError("synthetic outcome label was changed")


@runtime_checkable
class OutcomeProvider(Protocol):
    """Typed boundary from an executed synthetic decision to an event plan."""

    @property
    def provider_id(self) -> str:
        ...

    def plan(self, context: SyntheticDecisionContext) -> SyntheticOutcomePlan:
        ...


class AnalyticStateTrace:
    """Stateless analytic scene/channel trace with exact 10-Hz timestamps."""

    def __init__(
        self,
        seed: str = "splitfusion-synthetic-trace-v1",
        *,
        start_ns: int = 0,
        carla_frame_base: int = 100_000,
    ) -> None:
        if not isinstance(seed, str) or not seed:
            raise SyntheticContractError("trace seed must be a non-empty string")
        self._seed = seed
        self._start_ns = _exact_non_negative_int(start_ns, "start_ns")
        self._carla_frame_base = _exact_non_negative_int(
            carla_frame_base, "carla_frame_base"
        )
        self._trace_id = canonical_sha256(
            {
                "carla_frame_base": self._carla_frame_base,
                "evidence_class": SYNTHETIC_EVIDENCE_CLASS,
                "fixture_label": SYNTHETIC_FIXTURE_LABEL,
                "seed": seed,
                "start_ns": self._start_ns,
                "type": "analytic_state_trace_v1",
            }
        )

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def observation(self, frame_index: int) -> SyntheticTracePoint:
        index = _exact_non_negative_int(frame_index, "frame_index")
        # Slow analytic trends plus counter-keyed, bounded perturbations.  Each
        # term uses only the current frame index: there is no look-ahead.
        phase = 2.0 * math.pi * ((index % 48) / 48.0)
        si = 0.50 + 0.30 * math.sin(phase) + 0.08 * (
            counter_uniform(self._seed, "si", index) - 0.5
        )
        p40 = 0.42 + 0.30 * math.cos(phase * 0.75 + 0.4) + 0.08 * (
            counter_uniform(self._seed, "p40", index) - 0.5
        )
        snr = 12.0 + 10.0 * math.sin(phase * 0.5 + 1.1) + 1.5 * (
            counter_uniform(self._seed, "snr", index) - 0.5
        )
        bsr = int(
            round(
                512.0
                + 4096.0 * (1.0 - min(max((snr + 5.0) / 32.0, 0.0), 1.0))
                + 256.0 * counter_uniform(self._seed, "bsr", index)
            )
        )
        mcs = int(round(min(max((snr + 8.0) / 32.0, 0.0), 1.0) * 28.0))
        return SyntheticTracePoint(
            frame_index=index,
            carla_frame_id=self._carla_frame_base + index,
            observed_ns=self._start_ns + index * FRAME_PERIOD_NS,
            camera_si_normalized=min(max(si, 0.0), 1.0),
            radar_p40=min(max(p40, 0.0), 1.0),
            achieved_snr_db=snr,
            bsr_bytes=bsr,
            mcs_index=mcs,
        )


class AnalyticOutcomeProvider:
    """Known continuous fixture surface with optional event-scenario overrides."""

    def __init__(
        self,
        seed: str = "splitfusion-synthetic-outcome-v1",
        *,
        scenario_by_decision: Optional[
            Mapping[int, SyntheticFeedbackScenario]
        ] = None,
    ) -> None:
        if not isinstance(seed, str) or not seed:
            raise SyntheticContractError("outcome seed must be a non-empty string")
        scenarios: Dict[int, SyntheticFeedbackScenario] = {}
        for decision, scenario in dict(scenario_by_decision or {}).items():
            seq = _exact_non_negative_int(decision, "scenario decision_seq")
            if not isinstance(scenario, SyntheticFeedbackScenario):
                raise SyntheticContractError(
                    f"scenario for decision {seq} is not SyntheticFeedbackScenario"
                )
            scenarios[seq] = scenario
        self._seed = seed
        self._scenarios = MappingProxyType(scenarios)
        self._provider_id = canonical_sha256(
            {
                "evidence_class": SYNTHETIC_EVIDENCE_CLASS,
                "fixture_label": SYNTHETIC_FIXTURE_LABEL,
                "scenarios": {
                    str(key): value.value for key, value in sorted(scenarios.items())
                },
                "seed": seed,
                "type": "analytic_outcome_provider_v1",
            }
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @staticmethod
    def _preferred_from_scalars(
        *,
        camera_si_normalized: float,
        radar_p40: float,
        achieved_snr_scaled: float,
        mcs_scaled: float,
    ) -> SyntheticActionChoice:
        """Known fixture optimum using only admissible current observations."""

        # The discrete optimum spans all modes as the allowed scene/radio
        # values change.  It never uses a frame/tensor/decision identifier.
        mode_signal = (
            0.37 * camera_si_normalized
            + 0.29 * radar_p40
            + 0.19 * achieved_snr_scaled
            + 0.15 * mcs_scaled
        )
        mode_id = min(
            EXPECTED_MODE_COUNT - 1,
            int(math.floor(mode_signal * EXPECTED_MODE_COUNT)),
        )
        channel_strength = achieved_snr_scaled
        scene_complexity = 0.5 * (camera_si_normalized + radar_p40)
        q = 0.08 + 0.72 * (1.0 - channel_strength) + 0.18 * (
            1.0 - scene_complexity
        )
        return SyntheticActionChoice(mode_id=mode_id, q=min(max(q, Q_MIN), Q_MAX))

    @classmethod
    def preferred_action(cls, state: SyntheticTracePoint) -> SyntheticActionChoice:
        """Fixture optimum for the outcome model's raw synthetic state."""

        return cls._preferred_from_scalars(
            camera_si_normalized=state.camera_si_normalized,
            radar_p40=state.radar_p40,
            achieved_snr_scaled=min(
                max((state.achieved_snr_db + 8.0) / 36.0, 0.0), 1.0
            ),
            mcs_scaled=float(state.mcs_index) / 28.0,
        )

    @classmethod
    def preferred_policy_action(
        cls, state: SyntheticPolicyObservation
    ) -> SyntheticActionChoice:
        """Same fixture optimum reconstructed only from allowed features."""

        values = state.as_mapping()
        return cls._preferred_from_scalars(
            camera_si_normalized=values["scene_camera_si_scaled"],
            radar_p40=values["scene_radar_p40"],
            achieved_snr_scaled=values["radio_achieved_snr_db_scaled"],
            mcs_scaled=values["radio_mcs_index_scaled"],
        )

    def plan(self, context: SyntheticDecisionContext) -> SyntheticOutcomePlan:
        if not isinstance(context, SyntheticDecisionContext):
            raise SyntheticContractError(
                "analytic provider requires SyntheticDecisionContext"
            )
        preferred = self.preferred_action(context.state)
        q_exec = context.action.q_e4 / 10_000.0
        mode_distance = abs(context.action.mode_id - preferred.mode_id) / (
            EXPECTED_MODE_COUNT - 1
        )
        q_distance = abs(q_exec - preferred.q) / Q_MAX
        jitter = counter_uniform(
            self._seed,
            "quality",
            context.decision_seq,
            context.action.mode_id,
            context.action.q_e4,
        )
        quality = min(
            max(0.98 - 0.30 * mode_distance - 0.55 * q_distance + 0.01 * (jitter - 0.5), 0.0),
            1.0,
        )

        channel_strength = min(
            max((context.state.achieved_snr_db + 8.0) / 36.0, 0.0), 1.0
        )
        latency_ms = (
            55.0
            + 105.0 * (1.0 - channel_strength) * (1.0 - q_exec)
            + 20.0 * mode_distance
            + 5.0
            * counter_uniform(
                self._seed,
                "latency",
                context.decision_seq,
                context.action.mode_id,
                context.action.q_e4,
            )
        )
        analytic_delay = int(round(min(latency_ms, 190.0) * 1_000_000.0))

        scenario = self._scenarios.get(
            context.decision_seq, SyntheticFeedbackScenario.ANALYTIC_TIMELY
        )
        status: Optional[FeedbackTerminalStatus] = (
            FeedbackTerminalStatus.REWARD_FINAL
        )
        duplicates: Tuple[int, ...] = ()
        if scenario is SyntheticFeedbackScenario.ANALYTIC_TIMELY:
            delay: Optional[int] = analytic_delay
        elif scenario is SyntheticFeedbackScenario.EARLY_BEFORE_HOLD:
            delay = 50_000_000
        elif scenario is SyntheticFeedbackScenario.EARLY_WITH_DUPLICATE:
            delay = 50_000_000
            duplicates = (10_000_000, 120_000_000)
        elif scenario is SyntheticFeedbackScenario.EXACT_DEADLINE:
            delay = B_REWARD_DEADLINE_NS
        elif scenario is SyntheticFeedbackScenario.LATE_AFTER_TIMEOUT:
            delay = B_REWARD_DEADLINE_NS + 50_000_000
        elif scenario is SyntheticFeedbackScenario.NO_FEEDBACK_TIMEOUT:
            status = None
            delay = None
        elif scenario is SyntheticFeedbackScenario.ACTION_PATH_FAILURE:
            status = FeedbackTerminalStatus.ACTION_PATH_FAILURE
            delay = min(analytic_delay, 150_000_000)
            quality = 0.0
        else:  # pragma: no cover - exhaustive enum guard
            raise SyntheticContractError(f"unhandled scenario {scenario}")

        return SyntheticOutcomePlan(
            fixture_quality=quality,
            preferred_mode_id=preferred.mode_id,
            preferred_q=preferred.q,
            feedback_status=status,
            feedback_delay_ns=delay,
            duplicate_after_ns=duplicates,
        )


class CyclingFixturePolicy:
    """Deterministic policy fixture that reaches all 12 modes."""

    def __init__(self, seed: str = "splitfusion-cycling-policy-v1") -> None:
        if not isinstance(seed, str) or not seed:
            raise SyntheticContractError("policy seed must be a non-empty string")
        self._seed = seed

    def choose(self, state: SyntheticPolicyObservation) -> SyntheticActionChoice:
        values = state.as_mapping()
        previous = [
            values[f"prev_joint_mode_onehot_{index:02d}"]
            for index in range(EXPECTED_MODE_COUNT)
        ]
        if values["prev_present_mask"] == 0.0:
            mode_id = 0
        else:
            mode_id = (previous.index(max(previous)) + 1) % EXPECTED_MODE_COUNT
        # Continuous and deterministic, but based only on allowed scene/radio
        # features rather than a hidden decision counter or frame identifier.
        q_signal = (
            0.45 * values["scene_camera_si_scaled"]
            + 0.35 * values["scene_radar_p40"]
            + 0.20 * (1.0 - values["radio_achieved_snr_db_scaled"])
        )
        seed_offset = 0.01 * (counter_uniform(self._seed, "q-seed", 0) - 0.5)
        q = min(max(0.02 + 0.94 * q_signal + seed_offset, Q_MIN), Q_MAX)
        return SyntheticActionChoice(mode_id=mode_id, q=q)


class AnalyticOracleFixturePolicy:
    """Returns the known optimum of :class:`AnalyticOutcomeProvider`."""

    def choose(self, state: SyntheticPolicyObservation) -> SyntheticActionChoice:
        return AnalyticOutcomeProvider.preferred_policy_action(state)


class SyntheticEventKind(Enum):
    FRAME_ADMISSION = "FRAME_ADMISSION"
    FEEDBACK_RECEIPT = "FEEDBACK_RECEIPT"
    DEADLINE_OBSERVATION = "DEADLINE_OBSERVATION"


@dataclass(frozen=True, slots=True)
class SyntheticEventRecord:
    """One serialized event-loop observation."""

    event_index: int
    observed_ns: int
    kind: SyntheticEventKind
    controller_state: str
    decision_seq: Optional[int]
    tensor_seq: Optional[int]
    reward_requested: Optional[bool]
    disposition: str
    action_sha256: Optional[str]
    completed_ticket_sha256: Optional[str]

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "action_sha256": self.action_sha256,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "controller_state": self.controller_state,
            "decision_seq": self.decision_seq,
            "disposition": self.disposition,
            "event_index": self.event_index,
            "evidence_class": SYNTHETIC_EVIDENCE_CLASS,
            "fixture_label": SYNTHETIC_FIXTURE_LABEL,
            "kind": self.kind.value,
            "observed_ns": self.observed_ns,
            "record": "synthetic_event_record_v1",
            "reward_requested": self.reward_requested,
            "tensor_seq": self.tensor_seq,
        }


@dataclass(frozen=True, slots=True)
class SyntheticFixtureValueRecord:
    """Explicitly non-replay, non-measured value record for a test ticket.

    This is the synthetic transition/value carrier permitted for contract
    mechanics.  It is intentionally not shaped like ``ReplayTransitionV1``:
    there is no next policy state, scalar production reward or learning-ready
    attestation to confuse it with a causal training record.
    """

    decision_seq: int
    state_sha256: str
    action_sha256: str
    mode_id: int
    q_e4: int
    fixture_quality: float
    preferred_mode_id: int
    preferred_q: float
    terminal_class: str
    realized_duration_d: int
    feedback_latency_ns: Optional[int]
    completed_ticket_sha256: str
    evidence_class: str = SYNTHETIC_EVIDENCE_CLASS
    fixture_label: str = SYNTHETIC_FIXTURE_LABEL
    replay_transition_v1_eligible: bool = False

    def __post_init__(self) -> None:
        _exact_non_negative_int(self.decision_seq, "decision_seq")
        _sha256_hex(self.state_sha256, "state_sha256")
        _sha256_hex(self.action_sha256, "action_sha256")
        _exact_non_negative_int(self.mode_id, "mode_id")
        if self.mode_id >= EXPECTED_MODE_COUNT:
            raise SyntheticContractError("mode_id is outside the 12-mode fixture")
        _exact_non_negative_int(self.q_e4, "q_e4")
        if not 0 <= self.q_e4 <= 9800:
            raise SyntheticContractError("q_e4 is outside [0, 9800]")
        _in_unit_interval(self.fixture_quality, "fixture_quality")
        _exact_non_negative_int(self.preferred_mode_id, "preferred_mode_id")
        if self.preferred_mode_id >= EXPECTED_MODE_COUNT:
            raise SyntheticContractError("preferred_mode_id is outside the fixture")
        preferred_q = _finite(self.preferred_q, "preferred_q")
        if not Q_MIN <= preferred_q <= Q_MAX:
            raise SyntheticContractError("preferred_q is outside actor bounds")
        _exact_non_negative_int(self.realized_duration_d, "realized_duration_d")
        if self.realized_duration_d < 2:
            raise SyntheticContractError("completed fixture hold must satisfy k_min=2")
        terminal_values = {terminal.value for terminal in TerminalClass}
        if self.terminal_class not in terminal_values:
            raise SyntheticContractError("terminal_class is not a controller terminal")
        if self.feedback_latency_ns is not None:
            _exact_non_negative_int(self.feedback_latency_ns, "feedback_latency_ns")
        _sha256_hex(self.completed_ticket_sha256, "completed_ticket_sha256")
        if self.replay_transition_v1_eligible:
            raise SyntheticContractError(
                "synthetic fixture results can never be ReplayTransitionV1 eligible"
            )
        if self.evidence_class != SYNTHETIC_EVIDENCE_CLASS or (
            self.fixture_label != SYNTHETIC_FIXTURE_LABEL
        ):
            raise SyntheticContractError("synthetic result provenance was changed")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "action_sha256": self.action_sha256,
            "completed_ticket_sha256": self.completed_ticket_sha256,
            "decision_seq": self.decision_seq,
            "evidence_class": self.evidence_class,
            "feedback_latency_ns": self.feedback_latency_ns,
            "fixture_label": self.fixture_label,
            "fixture_quality": self.fixture_quality,
            "mode_id": self.mode_id,
            "preferred_mode_id": self.preferred_mode_id,
            "preferred_q": self.preferred_q,
            "q_e4": self.q_e4,
            "realized_duration_d": self.realized_duration_d,
            "record": "synthetic_contract_test_fixture_value_v1",
            "replay_transition_v1_eligible": self.replay_transition_v1_eligible,
            "state_sha256": self.state_sha256,
            "terminal_class": self.terminal_class,
        }


@dataclass(frozen=True, slots=True)
class SyntheticRunReport:
    trace_id: str
    outcome_provider_id: str
    session_uuid: str
    controller_lineage_uuid: str
    frame_count: int
    policy_invocations: int
    events: Tuple[SyntheticEventRecord, ...]
    decisions: Tuple[SyntheticFixtureValueRecord, ...]
    final_controller_state: str
    evidence_class: str = SYNTHETIC_EVIDENCE_CLASS
    fixture_label: str = SYNTHETIC_FIXTURE_LABEL

    def __post_init__(self) -> None:
        if self.evidence_class != SYNTHETIC_EVIDENCE_CLASS or (
            self.fixture_label != SYNTHETIC_FIXTURE_LABEL
        ):
            raise SyntheticContractError("synthetic report provenance was changed")
        _exact_non_negative_int(self.frame_count, "frame_count")
        _exact_non_negative_int(self.policy_invocations, "policy_invocations")
        if not isinstance(self.events, tuple) or not isinstance(self.decisions, tuple):
            raise SyntheticContractError("events and decisions must be immutable tuples")
        if any(
            self.events[index].observed_ns > self.events[index + 1].observed_ns
            for index in range(len(self.events) - 1)
        ):
            raise SyntheticContractError("serialized event report is not time ordered")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "controller_lineage_uuid": self.controller_lineage_uuid,
            "decisions": [item.to_canonical_dict() for item in self.decisions],
            "events": [item.to_canonical_dict() for item in self.events],
            "evidence_class": self.evidence_class,
            "final_controller_state": self.final_controller_state,
            "fixture_label": self.fixture_label,
            "frame_count": self.frame_count,
            "outcome_provider_id": self.outcome_provider_id,
            "policy_invocations": self.policy_invocations,
            "record": "synthetic_contract_run_report_v1",
            "session_uuid": self.session_uuid,
            "trace_id": self.trace_id,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(order=True, slots=True)
class _QueuedEvent:
    observed_ns: int
    insertion_order: int
    kind: SyntheticEventKind = field(compare=False)
    message: Optional[RewardFeedbackMessage] = field(compare=False, default=None)


class SyntheticContractEnvironment:
    """Single-threaded 10-Hz harness around the real ticket controller."""

    def __init__(
        self,
        *,
        state_trace: StateTrace,
        outcome_provider: OutcomeProvider,
        fixture_seed: str = "splitfusion-synthetic-environment-v1",
        contract: Optional[SplitActionContract] = None,
    ) -> None:
        if not isinstance(state_trace, StateTrace):
            raise SyntheticContractError("state_trace does not satisfy StateTrace")
        if not isinstance(outcome_provider, OutcomeProvider):
            raise SyntheticContractError(
                "outcome_provider does not satisfy OutcomeProvider"
            )
        if not isinstance(fixture_seed, str) or not fixture_seed:
            raise SyntheticContractError("fixture_seed must be a non-empty string")
        self._state_trace = state_trace
        self._outcome_provider = outcome_provider
        self._fixture_seed = fixture_seed
        self._contract = default_contract() if contract is None else contract
        if not isinstance(self._contract, SplitActionContract):
            raise SyntheticContractError("contract must be SplitActionContract")

    @staticmethod
    def _uuid(seed: str, label: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{seed}/{label}"))

    @staticmethod
    def _policy_observation(
        state: SyntheticTracePoint,
        previous: Optional[SyntheticFixtureValueRecord],
    ) -> SyntheticPolicyObservation:
        """Project raw fixture state onto the frozen policy-feature allowlist."""

        named: Dict[str, float] = {
            "scene_camera_si_scaled": state.camera_si_normalized,
            "scene_radar_p40": state.radar_p40,
            "radio_achieved_snr_db_scaled": min(
                max((state.achieved_snr_db + 8.0) / 36.0, 0.0), 1.0
            ),
            "radio_bsr_log1p_scaled": min(
                max(math.log1p(float(state.bsr_bytes)) / math.log1p(8192.0), 0.0),
                1.0,
            ),
            "radio_mcs_index_scaled": float(state.mcs_index) / 28.0,
            # Every analytic trace component is sampled at this exact frame.
            "freshness_scene_normalized": 0.0,
            "freshness_snr_normalized": 0.0,
            "freshness_bsr_normalized": 0.0,
            "freshness_mcs_normalized": 0.0,
        }
        for index in range(EXPECTED_MODE_COUNT):
            named[f"prev_joint_mode_onehot_{index:02d}"] = (
                1.0 if previous is not None and previous.mode_id == index else 0.0
            )
        for terminal in PREVIOUS_TERMINAL_ORDER:
            code = PREVIOUS_TERMINAL_FEATURE_CODES[terminal]
            named[f"prev_terminal_onehot_{code}"] = (
                1.0
                if previous is not None and previous.terminal_class == terminal.value
                else 0.0
            )

        exact = (
            previous is not None
            and previous.terminal_class == TerminalClass.REWARD_FINAL_EXACT.value
        )
        named["prev_q_normalized"] = (
            0.0 if previous is None else float(previous.q_e4) / 9800.0
        )
        named["prev_quality_normalized"] = (
            previous.fixture_quality if exact and previous is not None else 0.0
        )
        named["prev_latency_normalized"] = (
            float(previous.feedback_latency_ns) / float(B_REWARD_DEADLINE_NS)
            if exact
            and previous is not None
            and previous.feedback_latency_ns is not None
            else 0.0
        )
        named["prev_present_mask"] = 0.0 if previous is None else 1.0
        named["prev_quality_valid_mask"] = 1.0 if exact else 0.0
        named["prev_latency_valid_mask"] = 1.0 if exact else 0.0

        missing = set(POLICY_FEATURE_ORDER) - set(named)
        extra = set(named) - set(POLICY_FEATURE_ORDER)
        if missing or extra:
            raise SyntheticContractError(
                f"synthetic feature projection drifted; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return SyntheticPolicyObservation(
            values=tuple(float(named[name]) for name in POLICY_FEATURE_ORDER)
        )

    def run(self, *, frame_count: int, policy: SyntheticPolicy) -> SyntheticRunReport:
        """Run a finite fixture with no threads, sleeps, I/O or hidden clock."""

        count = _exact_non_negative_int(frame_count, "frame_count")
        if count == 0:
            raise SyntheticContractError("frame_count must be positive")
        if not isinstance(policy, SyntheticPolicy):
            raise SyntheticContractError("policy does not satisfy SyntheticPolicy")

        session_uuid = self._uuid(self._fixture_seed, "session")
        lineage_uuid = self._uuid(self._fixture_seed, "lineage")
        controller = RewardTicketController(
            session_uuid,
            controller_lineage_uuid=lineage_uuid,
            max_terminal_history=max(32, count),
        )

        queue: List[_QueuedEvent] = []
        insertion_order = 0
        event_records: List[SyntheticEventRecord] = []
        contexts: Dict[int, SyntheticDecisionContext] = {}
        plans: Dict[int, SyntheticOutcomePlan] = {}
        results: Dict[int, SyntheticFixtureValueRecord] = {}
        opening_actions: Dict[int, ExecutedActionIdentity] = {}
        next_decision_seq = 0
        policy_invocations = 0
        event_index = 0

        def schedule(
            observed_ns: int,
            kind: SyntheticEventKind,
            message: Optional[RewardFeedbackMessage] = None,
        ) -> None:
            nonlocal insertion_order
            heapq.heappush(
                queue,
                _QueuedEvent(
                    observed_ns=_exact_non_negative_int(observed_ns, "event time"),
                    insertion_order=insertion_order,
                    kind=kind,
                    message=message,
                ),
            )
            insertion_order += 1

        def record_completed(ticket: Optional[CompletedTicket]) -> None:
            if ticket is None or ticket.decision_seq in results:
                return
            context = contexts[ticket.decision_seq]
            plan = plans[ticket.decision_seq]
            result = SyntheticFixtureValueRecord(
                decision_seq=ticket.decision_seq,
                state_sha256=context.state.canonical_sha256(),
                action_sha256=context.action.canonical_sha256(),
                mode_id=context.action.mode_id,
                q_e4=context.action.q_e4,
                fixture_quality=plan.fixture_quality,
                preferred_mode_id=plan.preferred_mode_id,
                preferred_q=plan.preferred_q,
                terminal_class=ticket.terminal_class.value,
                realized_duration_d=ticket.hold_duration_tensors,
                feedback_latency_ns=ticket.feedback_latency_ns,
                completed_ticket_sha256=ticket.canonical_sha256(),
            )
            results[ticket.decision_seq] = result

        def append_event(
            *,
            observed_ns: int,
            kind: SyntheticEventKind,
            decision_seq: Optional[int],
            tensor_seq: Optional[int],
            reward_requested: Optional[bool],
            disposition: str,
            action: Optional[ExecutedActionIdentity],
            completed: Optional[CompletedTicket],
        ) -> None:
            nonlocal event_index
            event_records.append(
                SyntheticEventRecord(
                    event_index=event_index,
                    observed_ns=observed_ns,
                    kind=kind,
                    controller_state=controller.state.value,
                    decision_seq=decision_seq,
                    tensor_seq=tensor_seq,
                    reward_requested=reward_requested,
                    disposition=disposition,
                    action_sha256=(
                        None if action is None else action.canonical_sha256()
                    ),
                    completed_ticket_sha256=(
                        None if completed is None else completed.canonical_sha256()
                    ),
                )
            )
            event_index += 1

        def process_queued(event: _QueuedEvent) -> None:
            if event.kind is SyntheticEventKind.DEADLINE_OBSERVATION:
                status = controller.observe(event.observed_ns)
                record_completed(status.completed_ticket)
                append_event(
                    observed_ns=event.observed_ns,
                    kind=event.kind,
                    decision_seq=(
                        None
                        if status.completed_ticket is None
                        else status.completed_ticket.decision_seq
                    ),
                    tensor_seq=None,
                    reward_requested=None,
                    disposition=(
                        "NO_DEADLINE_EFFECT"
                        if status.completed_ticket is None
                        else status.completed_ticket.terminal_class.value
                    ),
                    action=(
                        None
                        if status.completed_ticket is None
                        else status.completed_ticket.action
                    ),
                    completed=status.completed_ticket,
                )
                return
            if event.kind is not SyntheticEventKind.FEEDBACK_RECEIPT or (
                event.message is None
            ):
                raise SyntheticContractError("malformed queued event")
            outcome = controller.submit_feedback(event.message, event.observed_ns)
            record_completed(outcome.completed_ticket)
            append_event(
                observed_ns=event.observed_ns,
                kind=event.kind,
                decision_seq=event.message.decision_seq,
                tensor_seq=event.message.reward_tensor_seq,
                reward_requested=None,
                disposition=outcome.disposition.value,
                action=event.message.action,
                completed=outcome.completed_ticket,
            )

        trace_start_ns: Optional[int] = None
        for frame_index in range(count):
            state = self._state_trace.observation(frame_index)
            if not isinstance(state, SyntheticTracePoint):
                raise SyntheticContractError(
                    "StateTrace.observation must return SyntheticTracePoint"
                )
            if state.frame_index != frame_index:
                raise SyntheticContractError(
                    "StateTrace returned a different frame index; no timestamp/"
                    "frame nearest-match is permitted"
                )
            if trace_start_ns is None:
                trace_start_ns = state.observed_ns
            expected_ns = trace_start_ns + frame_index * FRAME_PERIOD_NS
            if state.observed_ns != expected_ns:
                raise SyntheticContractError(
                    "StateTrace is not an exact serialized 10-Hz trace"
                )
            while queue and queue[0].observed_ns <= state.observed_ns:
                process_queued(heapq.heappop(queue))

            selected_choice: Optional[SyntheticActionChoice] = None
            selected_identity: Optional[ExecutedActionIdentity] = None

            def select_action() -> ExecutedActionIdentity:
                nonlocal selected_choice, selected_identity, policy_invocations
                policy_invocations += 1
                previous = results[max(results)] if results else None
                policy_state = self._policy_observation(state, previous)
                selected_choice = policy.choose(policy_state)
                if not isinstance(selected_choice, SyntheticActionChoice):
                    raise SyntheticContractError(
                        "SyntheticPolicy.choose must return SyntheticActionChoice"
                    )
                executable = self._contract.resolve(
                    selected_choice.mode_id, selected_choice.q
                )
                selected_identity = ExecutedActionIdentity.from_executable_action(
                    executable, self._contract
                )
                return selected_identity

            admission = controller.admit_frame(
                tensor_seq=frame_index,
                carla_frame_id=state.carla_frame_id,
                now_ns=state.observed_ns,
                next_decision_seq=next_decision_seq,
                select_action=select_action,
            )
            record_completed(admission.completed_ticket)
            append_event(
                observed_ns=state.observed_ns,
                kind=SyntheticEventKind.FRAME_ADMISSION,
                decision_seq=admission.decision_seq,
                tensor_seq=admission.tensor_seq,
                reward_requested=admission.reward_requested,
                disposition=admission.disposition.value,
                action=admission.action,
                completed=admission.completed_ticket,
            )

            if admission.disposition is AdmissionDisposition.OPENED_NEW_DECISION:
                if selected_choice is None or selected_identity is None:
                    raise SyntheticContractError("gate opened without invoking policy")
                context = SyntheticDecisionContext(
                    decision_seq=admission.decision_seq,
                    state=state,
                    action=selected_identity,
                )
                plan = self._outcome_provider.plan(context)
                if not isinstance(plan, SyntheticOutcomePlan):
                    raise SyntheticContractError(
                        "OutcomeProvider.plan must return SyntheticOutcomePlan"
                    )
                contexts[admission.decision_seq] = context
                plans[admission.decision_seq] = plan
                opening_actions[admission.decision_seq] = admission.action
                next_decision_seq += 1

                # The deadline is inclusive.  Observe one nanosecond after it;
                # feedback at exactly B remains timely and is processed first.
                schedule(
                    admission.deadline_ns + 1,
                    SyntheticEventKind.DEADLINE_OBSERVATION,
                )
                if plan.feedback_status is not None:
                    identity = RewardFeedbackIdentity(
                        session_uuid=session_uuid,
                        decision_seq=admission.decision_seq,
                        reward_tensor_seq=admission.tensor_seq,
                        carla_frame_id=admission.carla_frame_id,
                        action=admission.action,
                    )
                    message = RewardFeedbackMessage(
                        identity=identity,
                        terminal_status=plan.feedback_status,
                    )
                    assert plan.feedback_delay_ns is not None
                    receipt_ns = state.observed_ns + plan.feedback_delay_ns
                    schedule(
                        receipt_ns,
                        SyntheticEventKind.FEEDBACK_RECEIPT,
                        message,
                    )
                    for duplicate_offset in plan.duplicate_after_ns:
                        schedule(
                            receipt_ns + duplicate_offset,
                            SyntheticEventKind.FEEDBACK_RECEIPT,
                            message,
                        )
            else:
                opening = opening_actions[admission.decision_seq]
                if admission.action != opening:
                    raise SyntheticContractError(
                        "held frame did not reuse the exact executed action identity"
                    )
                if admission.reward_requested:
                    raise SyntheticContractError(
                        "a reused held action requested a second reward"
                    )

        # Drain only already-scheduled control-plane events.  No synthetic
        # frame is fabricated to rescue a final ticket that lacks k_min.
        while queue:
            process_queued(heapq.heappop(queue))

        return SyntheticRunReport(
            trace_id=self._state_trace.trace_id,
            outcome_provider_id=self._outcome_provider.provider_id,
            session_uuid=session_uuid,
            controller_lineage_uuid=lineage_uuid,
            frame_count=count,
            policy_invocations=policy_invocations,
            events=tuple(event_records),
            decisions=tuple(results[key] for key in sorted(results)),
            final_controller_state=controller.state.value,
        )
