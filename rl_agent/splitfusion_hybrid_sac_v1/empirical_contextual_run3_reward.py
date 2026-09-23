"""Run-3 realized-outcome reward and simulator-kernel contracts.

This module deliberately separates two concepts that the historical D1
expected-utility pilot combined:

* :func:`evaluate_run3_reward` scores one *realized* terminal outcome.  Its
  interface contains no delivery or admission probability.
* :func:`sample_run3_simulator_outcome` uses modeled reassembly/admission
  probabilities only as a simulator transition kernel.  Those probabilities
  remain raw audit fields and never enter the realized reward formula.

The latency sampler is an explicitly registered proxy distribution passing
through modeled P50/P95/P99 values.  It is not a reconstructed or measured
per-frame latency distribution.  This module performs no evidence I/O,
training, CUDA initialization, CARLA launch, or network operation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .empirical_quality_surface import REWARD_SPEC_FILE_SHA256
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "RUN3_EVIDENCE_CLASS",
    "RUN3_KERNEL_SPEC",
    "RUN3_KERNEL_SPEC_SHA256",
    "RUN3_REWARD_SPEC",
    "RUN3_REWARD_SPEC_SHA256",
    "QuantileLatencyProxyV1",
    "Run3CounterRngV1",
    "Run3ExpectedOutcomeV1",
    "Run3KernelSpecV1",
    "Run3RandomDrawsV1",
    "Run3RewardError",
    "Run3RewardResultV1",
    "Run3RewardSpecV1",
    "Run3SimulatedOutcomeV1",
    "Run3TerminalOutcome",
    "evaluate_run3_reward",
    "expected_run3_reward",
    "sample_run3_simulator_outcome",
]


RUN3_EVIDENCE_CLASS = (
    "MODELED_RUN3_SIMULATOR_OUTCOME_NOT_A_MEASURED_PER_FRAME_DISTRIBUTION"
)
_REWARD_SCHEMA = "splitfusion.empirical_contextual_run3_reward.v1"
_KERNEL_SCHEMA = "splitfusion.empirical_contextual_run3_kernel.v1"
_DRAWS_SCHEMA = "splitfusion.empirical_contextual_run3_random_draws.v1"
_OUTCOME_SCHEMA = "splitfusion.empirical_contextual_run3_simulated_outcome.v1"
_EXPECTED_SCHEMA = "splitfusion.empirical_contextual_run3_expected_outcome.v1"
_COUNTER_RNG_ALGORITHM = (
    "SHA256_CANONICAL_JSON_DOMAIN_SEPARATED_TOP_53_BITS_DIV_2_POW_53"
)
_RANDOM_RECORD_VALIDATION = (
    "RECOMPUTE_ALL_DOMAIN_DRAWS_FROM_MASTER_SEED_AND_DECISION_KEY"
)
_DRAW_DOMAINS: Tuple[str, ...] = (
    "REASSEMBLY",
    "EDGE_ADMISSION_GIVEN_REASSEMBLED",
    "CONDITIONAL_LATENCY_QUANTILE",
)


class Run3RewardError(ValueError):
    """A Run-3 reward, kernel, draw, or outcome invariant failed."""


def _exact_finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise Run3RewardError(f"{name} must be an exact finite float")
    return value


def _unit_float(value: object, name: str) -> float:
    result = _exact_finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise Run3RewardError(f"{name} must lie in [0, 1]")
    return result


def _probability_draw(value: object, name: str) -> float:
    result = _exact_finite_float(value, name)
    if not 0.0 <= result < 1.0:
        raise Run3RewardError(f"{name} must lie in [0, 1)")
    return result


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Run3RewardError(f"{name} must be a lowercase SHA-256")
    return value


def _counter_uniform(master_seed: int, decision_key: str, domain: str) -> float:
    """Derive one authoritative local draw from its complete identity."""

    if type(master_seed) is not int or master_seed < 0:
        raise Run3RewardError("master_seed must be a non-negative exact integer")
    if type(decision_key) is not str or not decision_key:
        raise Run3RewardError("decision_key must be a non-empty string")
    if domain not in _DRAW_DOMAINS:
        raise Run3RewardError("random draw domain is not registered")
    document = {
        "algorithm": _COUNTER_RNG_ALGORITHM,
        "decision_key": decision_key,
        "domain": domain,
        "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        "master_seed": master_seed,
        "record": "run3_counter_rng_draw_v1",
    }
    digest = hashlib.sha256(canonical_json_bytes(document)).digest()
    integer53 = int.from_bytes(digest[:8], "big") >> 11
    return float(integer53) / float(1 << 53)


class Run3TerminalOutcome(str, Enum):
    """Exhaustive terminal classes understood by the Run-3 scalar reward."""

    SUCCESS_WITHIN_DEADLINE = "SUCCESS_WITHIN_DEADLINE"
    REASSEMBLY_FAILURE = "REASSEMBLY_FAILURE"
    EDGE_ADMISSION_FAILURE = "EDGE_ADMISSION_FAILURE"
    SIMULATED_SERVICE_TIMEOUT = "SIMULATED_SERVICE_TIMEOUT"
    INFRASTRUCTURE_FAULT_EXCLUDED = "INFRASTRUCTURE_FAULT_EXCLUDED"
    EVALUATOR_FAULT_EXCLUDED = "EVALUATOR_FAULT_EXCLUDED"


_FAILURE_OUTCOMES = (
    Run3TerminalOutcome.REASSEMBLY_FAILURE,
    Run3TerminalOutcome.EDGE_ADMISSION_FAILURE,
    Run3TerminalOutcome.SIMULATED_SERVICE_TIMEOUT,
)
_EXCLUDED_OUTCOMES = (
    Run3TerminalOutcome.INFRASTRUCTURE_FAULT_EXCLUDED,
    Run3TerminalOutcome.EVALUATOR_FAULT_EXCLUDED,
)


@dataclass(frozen=True, slots=True)
class Run3RewardSpecV1:
    """Hash-bound scalar rule for one realized Run-3 terminal outcome."""

    schema: str = _REWARD_SCHEMA
    quality_component: str = "q_perc"
    quality_definition_reward_spec_sha256: str = REWARD_SPEC_FILE_SHA256
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    deadline_ms: float = 200.0
    failure_or_timeout_reward: float = -1.0
    mode_switch_weight: float = 0.0
    q_switch_weight: float = 0.0
    success_boundary: str = "INCLUSIVE_LATENCY_LE_DEADLINE"
    excluded_fault_semantics: str = "NO_SCALAR_REWARD_AND_NO_REPLAY_ROW"

    def __post_init__(self) -> None:
        if self.schema != _REWARD_SCHEMA:
            raise Run3RewardError("Run-3 reward schema drift")
        if self.quality_component != "q_perc":
            raise Run3RewardError("Run-3 reward quality component drift")
        _require_sha256(
            self.quality_definition_reward_spec_sha256,
            "quality_definition_reward_spec_sha256",
        )
        required = {
            "quality_weight": 1.0,
            "latency_weight": 0.25,
            "deadline_ms": 200.0,
            "failure_or_timeout_reward": -1.0,
            "mode_switch_weight": 0.0,
            "q_switch_weight": 0.0,
        }
        for name, expected in required.items():
            value = _exact_finite_float(getattr(self, name), name)
            if value != expected:
                raise Run3RewardError(f"{name} must equal {expected}")
        if self.success_boundary != "INCLUSIVE_LATENCY_LE_DEADLINE":
            raise Run3RewardError("Run-3 success-boundary semantics drift")
        if self.excluded_fault_semantics != "NO_SCALAR_REWARD_AND_NO_REPLAY_ROW":
            raise Run3RewardError("Run-3 excluded-fault semantics drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "excluded_fault_semantics": self.excluded_fault_semantics,
            "failure_or_timeout_reward": self.failure_or_timeout_reward,
            "latency_weight": self.latency_weight,
            "mode_switch_weight": self.mode_switch_weight,
            "q_switch_weight": self.q_switch_weight,
            "quality_component": self.quality_component,
            "quality_definition_reward_spec_sha256": (
                self.quality_definition_reward_spec_sha256
            ),
            "quality_weight": self.quality_weight,
            "reward_form": (
                "SUCCESS_AND_L_LE_200:Q_perc-0.25*(L/200);"
                "REASSEMBLY_OR_ADMISSION_FAILURE_OR_TIMEOUT:-1;"
                "INFRASTRUCTURE_OR_EVALUATOR_FAULT:EXCLUDED"
            ),
            "schema": self.schema,
            "success_boundary": self.success_boundary,
            "deadline_ms": self.deadline_ms,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


RUN3_REWARD_SPEC = Run3RewardSpecV1()
# Literal is filled from the canonical document and asserted at import so a
# future edit must be an explicit contract version change.
RUN3_REWARD_SPEC_SHA256 = "f594b204c4ca9bb47ae94d73be20200881cd7771713b17dbe3a9450209ca9841"


@dataclass(frozen=True, slots=True)
class Run3KernelSpecV1:
    """Hash-bound stochastic transition kernel, separate from the reward."""

    schema: str = _KERNEL_SCHEMA
    bernoulli_success_rule: str = "DRAW_U_LT_MODELED_PROBABILITY"
    probability_chain: str = (
        "REASSEMBLY_THEN_EDGE_ADMISSION_CONDITIONAL_ON_REASSEMBLY"
    )
    latency_conditioning: str = "ONLY_AFTER_REASSEMBLY_AND_EDGE_ADMISSION"
    latency_proxy_knots: Tuple[Tuple[float, str], ...] = (
        (0.0, "max(0,2*p50-p95)"),
        (0.5, "p50"),
        (0.95, "p95"),
        (0.99, "p99"),
        (1.0, "p99"),
    )
    interpolation: str = "PIECEWISE_LINEAR_IN_QUANTILE_COORDINATE"
    distribution_claim: str = (
        "TRANSPARENT_QUANTILE_PROXY_NOT_RECONSTRUCTED_OR_MEASURED_"
        "PER_FRAME_LATENCY_DISTRIBUTION"
    )
    random_algorithm: str = _COUNTER_RNG_ALGORITHM
    random_record_validation: str = _RANDOM_RECORD_VALIDATION
    random_domains: Tuple[str, ...] = _DRAW_DOMAINS

    def __post_init__(self) -> None:
        if self.schema != _KERNEL_SCHEMA:
            raise Run3RewardError("Run-3 kernel schema drift")
        if self.bernoulli_success_rule != "DRAW_U_LT_MODELED_PROBABILITY":
            raise Run3RewardError("Run-3 Bernoulli rule drift")
        if self.probability_chain != (
            "REASSEMBLY_THEN_EDGE_ADMISSION_CONDITIONAL_ON_REASSEMBLY"
        ):
            raise Run3RewardError("Run-3 probability-chain drift")
        if self.latency_conditioning != (
            "ONLY_AFTER_REASSEMBLY_AND_EDGE_ADMISSION"
        ):
            raise Run3RewardError("Run-3 latency conditioning drift")
        expected_knots = (
            (0.0, "max(0,2*p50-p95)"),
            (0.5, "p50"),
            (0.95, "p95"),
            (0.99, "p99"),
            (1.0, "p99"),
        )
        if self.latency_proxy_knots != expected_knots:
            raise Run3RewardError("Run-3 latency-proxy knots drift")
        if self.interpolation != "PIECEWISE_LINEAR_IN_QUANTILE_COORDINATE":
            raise Run3RewardError("Run-3 latency interpolation drift")
        if self.distribution_claim != (
            "TRANSPARENT_QUANTILE_PROXY_NOT_RECONSTRUCTED_OR_MEASURED_"
            "PER_FRAME_LATENCY_DISTRIBUTION"
        ):
            raise Run3RewardError("Run-3 distribution disclosure drift")
        if self.random_algorithm != _COUNTER_RNG_ALGORITHM:
            raise Run3RewardError("Run-3 local RNG algorithm drift")
        if self.random_record_validation != _RANDOM_RECORD_VALIDATION:
            raise Run3RewardError("Run-3 random-record validation drift")
        if self.random_domains != _DRAW_DOMAINS:
            raise Run3RewardError("Run-3 local RNG domains drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "bernoulli_success_rule": self.bernoulli_success_rule,
            "distribution_claim": self.distribution_claim,
            "interpolation": self.interpolation,
            "latency_conditioning": self.latency_conditioning,
            "latency_proxy_knots": [list(item) for item in self.latency_proxy_knots],
            "probability_chain": self.probability_chain,
            "probability_role": (
                "SIMULATOR_TRANSITION_ONLY_NOT_REWARD_INPUT_NOT_POLICY_INPUT"
            ),
            "random_algorithm": self.random_algorithm,
            "random_record_validation": self.random_record_validation,
            "random_domains": list(self.random_domains),
            "schema": self.schema,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


RUN3_KERNEL_SPEC = Run3KernelSpecV1()
RUN3_KERNEL_SPEC_SHA256 = "1435958ff4df4b0aaf68af02e4113a9b9f3b0c7953b6f73aa5b089a7d2280c02"


@dataclass(frozen=True, slots=True)
class QuantileLatencyProxyV1:
    """Conditional latency proxy passing exactly through P50/P95/P99."""

    p50_ms: float
    p95_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        p50 = _exact_finite_float(self.p50_ms, "p50_ms")
        p95 = _exact_finite_float(self.p95_ms, "p95_ms")
        p99 = _exact_finite_float(self.p99_ms, "p99_ms")
        if not 0.0 <= p50 <= p95 <= p99:
            raise Run3RewardError("latency proxy requires 0 <= p50 <= p95 <= p99")

    @property
    def lower_endpoint_ms(self) -> float:
        return max(0.0, 2.0 * self.p50_ms - self.p95_ms)

    def points(self) -> Tuple[Tuple[float, float], ...]:
        return (
            (0.0, self.lower_endpoint_ms),
            (0.5, self.p50_ms),
            (0.95, self.p95_ms),
            (0.99, self.p99_ms),
            (1.0, self.p99_ms),
        )

    def inverse_cdf(self, quantile: float) -> float:
        value = _exact_finite_float(quantile, "quantile")
        if not 0.0 <= value <= 1.0:
            raise Run3RewardError("quantile must lie in [0, 1]")
        points = self.points()
        for (u0, y0), (u1, y1) in zip(points, points[1:]):
            if value <= u1:
                if u1 == u0:  # pragma: no cover - frozen knots are distinct
                    return y1
                return y0 + ((value - u0) / (u1 - u0)) * (y1 - y0)
        return self.p99_ms  # pragma: no cover - value <= 1 is checked above

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "distribution_claim": RUN3_KERNEL_SPEC.distribution_claim,
            "lower_endpoint_ms": self.lower_endpoint_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "points": [list(item) for item in self.points()],
            "record": "run3_quantile_latency_proxy_v1",
            "run3_kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        }


@dataclass(frozen=True, slots=True)
class Run3RandomDrawsV1:
    """Three domain-separated uniforms retained verbatim for audit/replay."""

    master_seed: int
    decision_key: str
    reassembly_u: float
    admission_u: float
    latency_u: float
    random_algorithm: str = _COUNTER_RNG_ALGORITHM
    kernel_spec_sha256: str = RUN3_KERNEL_SPEC_SHA256
    schema: str = _DRAWS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise Run3RewardError("master_seed must be a non-negative exact integer")
        if type(self.decision_key) is not str or not self.decision_key:
            raise Run3RewardError("decision_key must be a non-empty string")
        _probability_draw(self.reassembly_u, "reassembly_u")
        _probability_draw(self.admission_u, "admission_u")
        _probability_draw(self.latency_u, "latency_u")
        if self.random_algorithm != _COUNTER_RNG_ALGORITHM:
            raise Run3RewardError("random draw algorithm drift")
        if self.kernel_spec_sha256 != RUN3_KERNEL_SPEC_SHA256:
            raise Run3RewardError("random draw kernel binding drift")
        if self.schema != _DRAWS_SCHEMA:
            raise Run3RewardError("random draw schema drift")
        expected = tuple(
            _counter_uniform(self.master_seed, self.decision_key, domain)
            for domain in _DRAW_DOMAINS
        )
        supplied = (self.reassembly_u, self.admission_u, self.latency_u)
        if supplied != expected:
            raise Run3RewardError(
                "random draws do not match master_seed/decision_key/domain identity"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "admission_u": self.admission_u,
            "decision_key": self.decision_key,
            "kernel_spec_sha256": self.kernel_spec_sha256,
            "latency_u": self.latency_u,
            "master_seed": self.master_seed,
            "random_algorithm": self.random_algorithm,
            "reassembly_u": self.reassembly_u,
            "schema": self.schema,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class Run3CounterRngV1:
    """Stateless local counter PRNG with independent named draw domains."""

    master_seed: int

    def __post_init__(self) -> None:
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise Run3RewardError("master_seed must be a non-negative exact integer")

    def _uniform(self, decision_key: str, domain: str) -> float:
        return _counter_uniform(self.master_seed, decision_key, domain)

    def draws(self, decision_key: str) -> Run3RandomDrawsV1:
        if type(decision_key) is not str or not decision_key:
            raise Run3RewardError("decision_key must be a non-empty string")
        return Run3RandomDrawsV1(
            master_seed=self.master_seed,
            decision_key=decision_key,
            reassembly_u=self._uniform(decision_key, _DRAW_DOMAINS[0]),
            admission_u=self._uniform(decision_key, _DRAW_DOMAINS[1]),
            latency_u=self._uniform(decision_key, _DRAW_DOMAINS[2]),
        )


@dataclass(frozen=True, slots=True)
class Run3RewardResultV1:
    """Attested scalar result; probabilities are intentionally absent."""

    terminal_outcome: Run3TerminalOutcome
    q_perc: Optional[float]
    latency_ms: Optional[float]
    scalar_reward: Optional[float]
    learning_eligible: bool
    deadline_met: Optional[bool]
    mode_switch_penalty: float
    q_switch_penalty: float
    reward_spec_sha256: str
    _attestation_sha256: str

    def _document(self) -> Dict[str, Any]:
        return {
            "deadline_met": self.deadline_met,
            "latency_ms": self.latency_ms,
            "learning_eligible": self.learning_eligible,
            "mode_switch_penalty": self.mode_switch_penalty,
            "q_perc": self.q_perc,
            "q_switch_penalty": self.q_switch_penalty,
            "record": "run3_reward_result_v1",
            "reward_spec_sha256": self.reward_spec_sha256,
            "scalar_reward": self.scalar_reward,
            "terminal_outcome": self.terminal_outcome.value,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self._document())

    def revalidate(self) -> None:
        if type(self.terminal_outcome) is not Run3TerminalOutcome:
            raise Run3RewardError("reward result terminal outcome has foreign type")
        if type(self.learning_eligible) is not bool:
            raise Run3RewardError("learning_eligible must be bool")
        if self.reward_spec_sha256 != RUN3_REWARD_SPEC_SHA256:
            raise Run3RewardError("reward result spec binding drift")
        if self.mode_switch_penalty != 0.0 or self.q_switch_penalty != 0.0:
            raise Run3RewardError("Run-3 switch penalties must remain zero")
        outcome = self.terminal_outcome
        if outcome is Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE:
            quality = _unit_float(self.q_perc, "q_perc")
            latency = _exact_finite_float(self.latency_ms, "latency_ms")
            if not 0.0 <= latency <= RUN3_REWARD_SPEC.deadline_ms:
                raise Run3RewardError("successful latency must be within deadline")
            expected = quality - 0.25 * latency / 200.0
            if (
                self.scalar_reward != expected
                or self.learning_eligible is not True
                or self.deadline_met is not True
            ):
                raise Run3RewardError("successful reward result does not reconcile")
        elif outcome in _FAILURE_OUTCOMES:
            if self.q_perc is not None or self.latency_ms is not None:
                raise Run3RewardError("failure reward must not fabricate Q or latency")
            if (
                self.scalar_reward != -1.0
                or self.learning_eligible is not True
                or self.deadline_met is not False
            ):
                raise Run3RewardError("failure reward result does not reconcile")
        elif outcome in _EXCLUDED_OUTCOMES:
            if self.q_perc is not None or self.latency_ms is not None:
                raise Run3RewardError("excluded fault must not fabricate Q or latency")
            if (
                self.scalar_reward is not None
                or self.learning_eligible is not False
                or self.deadline_met is not None
            ):
                raise Run3RewardError("excluded-fault result does not reconcile")
        else:  # pragma: no cover - enum exhaustiveness
            raise Run3RewardError("unhandled Run-3 terminal outcome")
        _require_sha256(self._attestation_sha256, "_attestation_sha256")
        if self._attestation_sha256 != self.canonical_sha256():
            raise Run3RewardError("reward result attestation mismatch")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.revalidate()
        result = self._document()
        result["attestation_sha256"] = self._attestation_sha256
        return result


def evaluate_run3_reward(
    terminal_outcome: Run3TerminalOutcome,
    q_perc: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> Run3RewardResultV1:
    """Score one realized outcome; no transition probability is accepted."""

    if type(terminal_outcome) is not Run3TerminalOutcome:
        raise Run3RewardError("terminal_outcome must be exact Run3TerminalOutcome")
    if terminal_outcome is Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE:
        quality = _unit_float(q_perc, "q_perc")
        latency = _exact_finite_float(latency_ms, "latency_ms")
        if not 0.0 <= latency <= RUN3_REWARD_SPEC.deadline_ms:
            raise Run3RewardError(
                "SUCCESS_WITHIN_DEADLINE requires inclusive latency <= 200 ms"
            )
        scalar: Optional[float] = quality - 0.25 * latency / 200.0
        eligible = True
        deadline_met: Optional[bool] = True
        stored_q: Optional[float] = quality
        stored_latency: Optional[float] = latency
    elif terminal_outcome in _FAILURE_OUTCOMES:
        if q_perc is not None or latency_ms is not None:
            raise Run3RewardError(
                "failure/timeout reward does not accept fabricated Q or latency"
            )
        scalar = -1.0
        eligible = True
        deadline_met = False
        stored_q = None
        stored_latency = None
    elif terminal_outcome in _EXCLUDED_OUTCOMES:
        if q_perc is not None or latency_ms is not None:
            raise Run3RewardError(
                "excluded infrastructure/evaluator fault accepts no Q or latency"
            )
        scalar = None
        eligible = False
        deadline_met = None
        stored_q = None
        stored_latency = None
    else:  # pragma: no cover - enum exhaustiveness
        raise Run3RewardError("unhandled Run-3 terminal outcome")
    document = {
        "deadline_met": deadline_met,
        "latency_ms": stored_latency,
        "learning_eligible": eligible,
        "mode_switch_penalty": 0.0,
        "q_perc": stored_q,
        "q_switch_penalty": 0.0,
        "record": "run3_reward_result_v1",
        "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
        "scalar_reward": scalar,
        "terminal_outcome": terminal_outcome.value,
    }
    result = Run3RewardResultV1(
        terminal_outcome=terminal_outcome,
        q_perc=stored_q,
        latency_ms=stored_latency,
        scalar_reward=scalar,
        learning_eligible=eligible,
        deadline_met=deadline_met,
        mode_switch_penalty=0.0,
        q_switch_penalty=0.0,
        reward_spec_sha256=RUN3_REWARD_SPEC_SHA256,
        _attestation_sha256=canonical_sha256(document),
    )
    result.revalidate()
    return result


@dataclass(frozen=True, slots=True)
class Run3SimulatedOutcomeV1:
    """One sampled simulator terminal with raw kernel inputs retained."""

    source_q_perc: float
    p_complete_reassembly_given_sent: float
    p_edge_admission_given_reassembled: float
    latency_proxy: QuantileLatencyProxyV1
    random_draws: Run3RandomDrawsV1
    reassembled: bool
    edge_admitted: bool
    sampled_latency_ms: Optional[float]
    terminal_outcome: Run3TerminalOutcome
    reward_result: Run3RewardResultV1
    reward_spec_sha256: str
    kernel_spec_sha256: str
    evidence_class: str
    _attestation_sha256: str
    schema: str = _OUTCOME_SCHEMA

    def _document(self) -> Dict[str, Any]:
        return {
            "edge_admitted": self.edge_admitted,
            "evidence_class": self.evidence_class,
            "kernel_spec_sha256": self.kernel_spec_sha256,
            "latency_proxy": self.latency_proxy.to_canonical_dict(),
            "p_complete_reassembly_given_sent": (
                self.p_complete_reassembly_given_sent
            ),
            "p_edge_admission_given_reassembled": (
                self.p_edge_admission_given_reassembled
            ),
            "random_draws": self.random_draws.to_canonical_dict(),
            "reassembled": self.reassembled,
            "reward_result": self.reward_result.to_canonical_dict(),
            "reward_spec_sha256": self.reward_spec_sha256,
            "sampled_latency_ms": self.sampled_latency_ms,
            "schema": self.schema,
            "source_q_perc": self.source_q_perc,
            "terminal_outcome": self.terminal_outcome.value,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self._document())

    def revalidate(self) -> None:
        quality = _unit_float(self.source_q_perc, "source_q_perc")
        p_reassembly = _unit_float(
            self.p_complete_reassembly_given_sent,
            "p_complete_reassembly_given_sent",
        )
        p_admission = _unit_float(
            self.p_edge_admission_given_reassembled,
            "p_edge_admission_given_reassembled",
        )
        if type(self.latency_proxy) is not QuantileLatencyProxyV1:
            raise Run3RewardError("latency_proxy has foreign type")
        self.latency_proxy.__post_init__()
        if type(self.random_draws) is not Run3RandomDrawsV1:
            raise Run3RewardError("random_draws has foreign type")
        self.random_draws.__post_init__()
        if type(self.reassembled) is not bool or type(self.edge_admitted) is not bool:
            raise Run3RewardError("simulator stage flags must be bool")
        if self.edge_admitted and not self.reassembled:
            raise Run3RewardError("edge admission requires successful reassembly")
        if type(self.terminal_outcome) is not Run3TerminalOutcome:
            raise Run3RewardError("simulated terminal outcome has foreign type")
        if type(self.reward_result) is not Run3RewardResultV1:
            raise Run3RewardError("reward_result has foreign type")
        self.reward_result.revalidate()
        if self.reward_spec_sha256 != RUN3_REWARD_SPEC_SHA256:
            raise Run3RewardError("simulator reward binding drift")
        if self.kernel_spec_sha256 != RUN3_KERNEL_SPEC_SHA256:
            raise Run3RewardError("simulator kernel binding drift")
        if self.evidence_class != RUN3_EVIDENCE_CLASS:
            raise Run3RewardError("simulator evidence-class drift")
        if self.schema != _OUTCOME_SCHEMA:
            raise Run3RewardError("simulator outcome schema drift")

        expected_reassembled = self.random_draws.reassembly_u < p_reassembly
        expected_admitted = (
            expected_reassembled and self.random_draws.admission_u < p_admission
        )
        if self.reassembled != expected_reassembled or (
            self.edge_admitted != expected_admitted
        ):
            raise Run3RewardError("simulated stage flags do not match raw draws")
        if not expected_reassembled:
            expected_outcome = Run3TerminalOutcome.REASSEMBLY_FAILURE
            expected_latency = None
            expected_reward = evaluate_run3_reward(expected_outcome)
        elif not expected_admitted:
            expected_outcome = Run3TerminalOutcome.EDGE_ADMISSION_FAILURE
            expected_latency = None
            expected_reward = evaluate_run3_reward(expected_outcome)
        else:
            expected_latency = self.latency_proxy.inverse_cdf(
                self.random_draws.latency_u
            )
            if expected_latency <= RUN3_REWARD_SPEC.deadline_ms:
                expected_outcome = Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE
                expected_reward = evaluate_run3_reward(
                    expected_outcome, quality, expected_latency
                )
            else:
                expected_outcome = Run3TerminalOutcome.SIMULATED_SERVICE_TIMEOUT
                expected_reward = evaluate_run3_reward(expected_outcome)
        if self.terminal_outcome is not expected_outcome:
            raise Run3RewardError("simulated terminal classification drift")
        if self.sampled_latency_ms != expected_latency:
            raise Run3RewardError("sampled latency does not match raw latency draw")
        if self.reward_result.to_canonical_dict() != expected_reward.to_canonical_dict():
            raise Run3RewardError("simulated reward does not match realized outcome")
        _require_sha256(self._attestation_sha256, "_attestation_sha256")
        if self._attestation_sha256 != self.canonical_sha256():
            raise Run3RewardError("simulated outcome attestation mismatch")

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.revalidate()
        result = self._document()
        result["attestation_sha256"] = self._attestation_sha256
        return result


def sample_run3_simulator_outcome(
    *,
    q_perc: float,
    p_complete_reassembly_given_sent: float,
    p_edge_admission_given_reassembled: float,
    latency_proxy: QuantileLatencyProxyV1,
    random_draws: Run3RandomDrawsV1,
) -> Run3SimulatedOutcomeV1:
    """Sample the transition kernel, then score only the realized terminal."""

    quality = _unit_float(q_perc, "q_perc")
    p_reassembly = _unit_float(
        p_complete_reassembly_given_sent,
        "p_complete_reassembly_given_sent",
    )
    p_admission = _unit_float(
        p_edge_admission_given_reassembled,
        "p_edge_admission_given_reassembled",
    )
    if type(latency_proxy) is not QuantileLatencyProxyV1:
        raise Run3RewardError("latency_proxy must be exact QuantileLatencyProxyV1")
    latency_proxy.__post_init__()
    if type(random_draws) is not Run3RandomDrawsV1:
        raise Run3RewardError("random_draws must be exact Run3RandomDrawsV1")
    random_draws.__post_init__()

    reassembled = random_draws.reassembly_u < p_reassembly
    admitted = reassembled and random_draws.admission_u < p_admission
    if not reassembled:
        terminal = Run3TerminalOutcome.REASSEMBLY_FAILURE
        sampled_latency = None
        reward = evaluate_run3_reward(terminal)
    elif not admitted:
        terminal = Run3TerminalOutcome.EDGE_ADMISSION_FAILURE
        sampled_latency = None
        reward = evaluate_run3_reward(terminal)
    else:
        sampled_latency = latency_proxy.inverse_cdf(random_draws.latency_u)
        if sampled_latency <= RUN3_REWARD_SPEC.deadline_ms:
            terminal = Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE
            reward = evaluate_run3_reward(terminal, quality, sampled_latency)
        else:
            terminal = Run3TerminalOutcome.SIMULATED_SERVICE_TIMEOUT
            reward = evaluate_run3_reward(terminal)

    base = Run3SimulatedOutcomeV1(
        source_q_perc=quality,
        p_complete_reassembly_given_sent=p_reassembly,
        p_edge_admission_given_reassembled=p_admission,
        latency_proxy=latency_proxy,
        random_draws=random_draws,
        reassembled=reassembled,
        edge_admitted=admitted,
        sampled_latency_ms=sampled_latency,
        terminal_outcome=terminal,
        reward_result=reward,
        reward_spec_sha256=RUN3_REWARD_SPEC_SHA256,
        kernel_spec_sha256=RUN3_KERNEL_SPEC_SHA256,
        evidence_class=RUN3_EVIDENCE_CLASS,
        _attestation_sha256="0" * 64,
    )
    result = Run3SimulatedOutcomeV1(
        source_q_perc=base.source_q_perc,
        p_complete_reassembly_given_sent=(
            base.p_complete_reassembly_given_sent
        ),
        p_edge_admission_given_reassembled=(
            base.p_edge_admission_given_reassembled
        ),
        latency_proxy=base.latency_proxy,
        random_draws=base.random_draws,
        reassembled=base.reassembled,
        edge_admitted=base.edge_admitted,
        sampled_latency_ms=base.sampled_latency_ms,
        terminal_outcome=base.terminal_outcome,
        reward_result=base.reward_result,
        reward_spec_sha256=base.reward_spec_sha256,
        kernel_spec_sha256=base.kernel_spec_sha256,
        evidence_class=base.evidence_class,
        _attestation_sha256=base.canonical_sha256(),
    )
    result.revalidate()
    return result


@dataclass(frozen=True, slots=True)
class Run3ExpectedOutcomeV1:
    """Analysis-only expectation obtained by integrating the transition kernel."""

    expected_reward: float
    p_reassembly_failure: float
    p_admission_failure: float
    p_admitted: float
    p_timeout_given_admitted: float
    p_timely_feedback: float
    conditional_admitted_expected_reward: float
    reward_spec_sha256: str = RUN3_REWARD_SPEC_SHA256
    kernel_spec_sha256: str = RUN3_KERNEL_SPEC_SHA256
    schema: str = _EXPECTED_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "expected_reward",
            "p_reassembly_failure",
            "p_admission_failure",
            "p_admitted",
            "p_timeout_given_admitted",
            "p_timely_feedback",
            "conditional_admitted_expected_reward",
        ):
            _exact_finite_float(getattr(self, name), name)
        for name in (
            "p_reassembly_failure",
            "p_admission_failure",
            "p_admitted",
            "p_timeout_given_admitted",
            "p_timely_feedback",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise Run3RewardError(f"{name} must lie in [0,1]")
        if self.reward_spec_sha256 != RUN3_REWARD_SPEC_SHA256:
            raise Run3RewardError("expected outcome reward binding drift")
        if self.kernel_spec_sha256 != RUN3_KERNEL_SPEC_SHA256:
            raise Run3RewardError("expected outcome kernel binding drift")
        if self.schema != _EXPECTED_SCHEMA:
            raise Run3RewardError("expected outcome schema drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "conditional_admitted_expected_reward": (
                self.conditional_admitted_expected_reward
            ),
            "expected_reward": self.expected_reward,
            "kernel_spec_sha256": self.kernel_spec_sha256,
            "p_admission_failure": self.p_admission_failure,
            "p_admitted": self.p_admitted,
            "p_reassembly_failure": self.p_reassembly_failure,
            "p_timely_feedback": self.p_timely_feedback,
            "p_timeout_given_admitted": self.p_timeout_given_admitted,
            "reward_spec_sha256": self.reward_spec_sha256,
            "schema": self.schema,
        }


def _integrate_conditional_latency_reward(
    q_perc: float, latency_proxy: QuantileLatencyProxyV1
) -> Tuple[float, float]:
    """Integrate reward and timeout mass over the registered quantile proxy."""

    quality = _unit_float(q_perc, "q_perc")
    points = latency_proxy.points()
    reward_integral = 0.0
    timeout_mass = 0.0
    deadline = RUN3_REWARD_SPEC.deadline_ms
    for (u0, y0), (u1, y1) in zip(points, points[1:]):
        width = u1 - u0
        if y0 > deadline:
            reward_integral -= width
            timeout_mass += width
        elif y1 <= deadline:
            mean_latency = (y0 + y1) / 2.0
            reward_integral += width * (
                quality - 0.25 * mean_latency / deadline
            )
        else:
            fraction_timely = (deadline - y0) / (y1 - y0)
            timely_width = width * fraction_timely
            timeout_width = width - timely_width
            reward_integral += timely_width * (
                quality - 0.25 * ((y0 + deadline) / 2.0) / deadline
            )
            reward_integral -= timeout_width
            timeout_mass += timeout_width
    return reward_integral, timeout_mass


def expected_run3_reward(
    *,
    q_perc: float,
    p_complete_reassembly_given_sent: float,
    p_edge_admission_given_reassembled: float,
    latency_proxy: QuantileLatencyProxyV1,
) -> Run3ExpectedOutcomeV1:
    """Integrate the simulator kernel for analysis, not runtime reward input."""

    quality = _unit_float(q_perc, "q_perc")
    p_reassembly = _unit_float(
        p_complete_reassembly_given_sent,
        "p_complete_reassembly_given_sent",
    )
    p_admission = _unit_float(
        p_edge_admission_given_reassembled,
        "p_edge_admission_given_reassembled",
    )
    if type(latency_proxy) is not QuantileLatencyProxyV1:
        raise Run3RewardError("latency_proxy must be exact QuantileLatencyProxyV1")
    latency_proxy.__post_init__()
    conditional_reward, timeout_given_admitted = (
        _integrate_conditional_latency_reward(quality, latency_proxy)
    )
    p_reassembly_failure = 1.0 - p_reassembly
    p_admission_failure = p_reassembly * (1.0 - p_admission)
    p_admitted = p_reassembly * p_admission
    p_timely = p_admitted * (1.0 - timeout_given_admitted)
    expected = (
        -p_reassembly_failure
        - p_admission_failure
        + p_admitted * conditional_reward
    )
    return Run3ExpectedOutcomeV1(
        expected_reward=expected,
        p_reassembly_failure=p_reassembly_failure,
        p_admission_failure=p_admission_failure,
        p_admitted=p_admitted,
        p_timeout_given_admitted=timeout_given_admitted,
        p_timely_feedback=p_timely,
        conditional_admitted_expected_reward=conditional_reward,
    )


if RUN3_REWARD_SPEC.canonical_sha256() != RUN3_REWARD_SPEC_SHA256:
    raise RuntimeError("registered Run-3 reward contract canonical hash drift")
if RUN3_KERNEL_SPEC.canonical_sha256() != RUN3_KERNEL_SPEC_SHA256:
    raise RuntimeError("registered Run-3 kernel contract canonical hash drift")
