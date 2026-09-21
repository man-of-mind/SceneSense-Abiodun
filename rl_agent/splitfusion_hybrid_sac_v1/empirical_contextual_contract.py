"""Frozen D1 contract for the one-step empirical contextual pilot.

This module contains constants and validation only.  In particular it does
not open evidence files, sample randomness, run a simulator, or expose the
held-scene split.  The pilot utility is an explicitly separate hypothesis
from the pinned quality-grid reward specification: that file defines
``Q_perc`` but labels its scalar reward weights inert and unapproved for RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .empirical_quality_surface import REWARD_SPEC_FILE_SHA256
from .action_contract import CATALOG_SHA256
from .modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    ModeledSmokeSupportContract,
    require_registered_modeled_smoke_support,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "D1_SCHEMA",
    "DIRECT_QUALITY_COMPONENT",
    "FIXED_END_TO_FEEDBACK_STAGES_MS",
    "PILOT_UTILITY_SPEC",
    "PILOT_UTILITY_SPEC_SHA256",
    "PROFILE_ORDER_RNG_CONTRACT_SHA256",
    "SURFACE_QUALIFICATION_REPORT_SHA256",
    "ActionSupportError",
    "EmpiricalActionV1",
    "EmpiricalPilotBindingV1",
    "PilotUtilitySpecV1",
    "fixed_stage_latency_ms",
    "require_supported_action",
]


D1_SCHEMA = "splitfusion.empirical_contextual_one_step_d1.v1"
DIRECT_QUALITY_COMPONENT = "q_perc"
SURFACE_QUALIFICATION_REPORT_SHA256 = (
    "c62df71444c9ff0f349ca389ab637a6b7a19e04fb50963119bca673ce4d11ab9"
)
PROFILE_ORDER_RNG_CONTRACT_SHA256 = canonical_sha256(
    {
        "context_stream": "SHA256_MASTER_SEED_DOMAIN_D1_CONTEXT_V1",
        "profile_order": [
            "FAVORABLE_STABLE",
            "MID_VARIABLE",
            "ADVERSE_STABLE",
            "FADE_RECOVERY",
        ],
        "profile_stream": "SHA256_MASTER_SEED_DOMAIN_D1_RADIO_PROFILE_V1",
        "record": "d1_independent_rng_stream_contract_v1",
        "row_stream": "SHA256_MASTER_SEED_DOMAIN_D1_RADIO_ROW_V1",
        "rounding_stream": "SHA256_MASTER_SEED_DOMAIN_D1_MCS_ROUNDING_V1",
        "state_format": "PYTHON_RANDOM_GETSTATE_CANONICAL_TUPLE_V1",
    }
)

# Supervisor-approved end-to-feedback stage hypothesis.  These are retained
# individually in every outcome; 113 ms is never an unexplained intercept.
FIXED_END_TO_FEEDBACK_STAGES_MS: Tuple[Tuple[str, float], ...] = (
    ("sensor_compute", 30.0),
    ("ue_action_and_send", 25.0),
    ("decompression", 10.0),
    ("model_tail", 22.0),
    ("exact_carla_reward_evaluation", 20.0),
    ("compact_ack_downlink", 6.0),
)


class ActionSupportError(ValueError):
    """An action is not an exact member of the fit-derived curriculum."""


def fixed_stage_latency_ms() -> float:
    return sum(value for _name, value in FIXED_END_TO_FEEDBACK_STAGES_MS)


@dataclass(frozen=True, slots=True)
class EmpiricalActionV1:
    """Exact integer wire action.  Construction never clips or projects q."""

    mode_id: int
    q_e4: int


def require_supported_action(
    mode_id: object,
    q_e4: object,
    *,
    support: ModeledSmokeSupportContract = MODELED_SMOKE_SUPPORT,
) -> EmpiricalActionV1:
    """Validate exact inclusive curriculum membership without modification."""
    registered = require_registered_modeled_smoke_support(support)
    if type(mode_id) is not int or not 0 <= mode_id < len(
        registered.mode_q_e4_bounds
    ):
        raise ActionSupportError(
            f"mode_id must be an exact integer in [0, 11], got {mode_id!r}"
        )
    if type(q_e4) is not int:
        raise ActionSupportError(
            f"q_e4 must be an exact integer; clipping/projection is forbidden, "
            f"got {type(q_e4).__name__}: {q_e4!r}"
        )
    lower, upper = registered.mode_q_e4_bounds[mode_id]
    if not lower <= q_e4 <= upper:
        raise ActionSupportError(
            f"(mode_id={mode_id}, q_e4={q_e4}) is outside the inclusive "
            f"fit-derived support [{lower}, {upper}]; the environment never "
            "clips or projects an action"
        )
    return EmpiricalActionV1(mode_id=mode_id, q_e4=q_e4)


@dataclass(frozen=True, slots=True)
class PilotUtilitySpecV1:
    """Separately registered D1 utility hypothesis over direct ``Q_perc``."""

    schema: str
    quality_component: str
    quality_definition_reward_spec_sha256: str
    quality_weight: float
    latency_weight: float
    deadline_ms: float
    service_non_admission_utility: float
    mode_switch_weight: float
    q_switch_weight: float
    gamma: float
    horizon: str
    estimator: str
    latency_proxy_definition: str
    budget_miss_semantics: str
    scalar_provenance_status: str
    fixed_latency_stages_ms: Tuple[Tuple[str, float], ...]
    probability_semantics: str

    def __post_init__(self) -> None:
        if self.schema != "splitfusion.empirical_contextual_pilot_utility.v1":
            raise ValueError("pilot utility schema drift")
        if self.quality_component != DIRECT_QUALITY_COMPONENT:
            raise ValueError("D1 admits direct q_perc only")
        if self.quality_definition_reward_spec_sha256 != REWARD_SPEC_FILE_SHA256:
            raise ValueError("quality-definition reward-spec binding drift")
        expected = {
            "quality_weight": 1.0,
            "latency_weight": 0.25,
            "deadline_ms": 200.0,
            "service_non_admission_utility": -1.0,
            "mode_switch_weight": 0.0,
            "q_switch_weight": 0.0,
            "gamma": 1.0,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value != required:
                raise ValueError(f"pilot utility {name} must equal {required}")
        for name in (
            "horizon",
            "estimator",
            "latency_proxy_definition",
            "budget_miss_semantics",
            "scalar_provenance_status",
            "probability_semantics",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"pilot utility {name} must be a non-empty string")
        if self.fixed_latency_stages_ms != FIXED_END_TO_FEEDBACK_STAGES_MS:
            raise ValueError("pilot utility fixed latency-stage contract drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "budget_miss_semantics": self.budget_miss_semantics,
            "deadline_ms": self.deadline_ms,
            "estimator": self.estimator,
            "fixed_latency_stages_ms": [
                [name, value] for name, value in self.fixed_latency_stages_ms
            ],
            "gamma": self.gamma,
            "horizon": self.horizon,
            "latency_proxy_definition": self.latency_proxy_definition,
            "latency_weight": self.latency_weight,
            "mode_switch_weight": self.mode_switch_weight,
            "q_switch_weight": self.q_switch_weight,
            "quality_component": self.quality_component,
            "quality_definition_reward_spec_sha256": (
                self.quality_definition_reward_spec_sha256
            ),
            "quality_weight": self.quality_weight,
            "probability_semantics": self.probability_semantics,
            "scalar_provenance_status": self.scalar_provenance_status,
            "schema": self.schema,
            "service_non_admission_utility": self.service_non_admission_utility,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def expected_utility(
        self,
        *,
        p_edge_admission_given_sent: float,
        q_perc: float,
        latency_proxy_ms: float,
    ) -> float:
        for name, value in (
            ("p_edge_admission_given_sent", p_edge_admission_given_sent),
            ("q_perc", q_perc),
            ("latency_proxy_ms", latency_proxy_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= p_edge_admission_given_sent <= 1.0:
            raise ValueError("p_edge_admission_given_sent must lie in [0, 1]")
        if not 0.0 <= q_perc <= 1.0:
            raise ValueError("q_perc must lie in [0, 1]")
        if latency_proxy_ms < 0.0:
            raise ValueError("latency_proxy_ms must be non-negative")
        admitted = self.quality_weight * q_perc - self.latency_weight * (
            latency_proxy_ms / self.deadline_ms
        )
        return p_edge_admission_given_sent * admitted + (
            1.0 - p_edge_admission_given_sent
        ) * self.service_non_admission_utility


PILOT_UTILITY_SPEC = PilotUtilitySpecV1(
    schema="splitfusion.empirical_contextual_pilot_utility.v1",
    quality_component=DIRECT_QUALITY_COMPONENT,
    quality_definition_reward_spec_sha256=REWARD_SPEC_FILE_SHA256,
    quality_weight=1.0,
    latency_weight=0.25,
    deadline_ms=200.0,
    service_non_admission_utility=-1.0,
    mode_switch_weight=0.0,
    q_switch_weight=0.0,
    gamma=1.0,
    horizon="ONE_STEP_TERMINAL",
    estimator="DETERMINISTIC_EXPECTED_UTILITY_NOT_SAMPLED_TERMINAL_REWARD",
    latency_proxy_definition=(
        "113_MS_FIXED_SIMULATOR_TRAINING_STAGES_PLUS_CONDITIONAL_"
        "FEATURE_UPLINK_RETAINED_SURVIVOR_P50"
    ),
    budget_miss_semantics=(
        "MODELED_INDICATOR_ONLY_REWARD_REMAINS_DEFINED_NO_TIMEOUT_"
        "PROBABILITY_INFERRED_FROM_P50"
    ),
    scalar_provenance_status=(
        "SEPARATE_SUPERVISOR_APPROVED_D1_PILOT_HYPOTHESIS_NOT_SCALAR_"
        "FIELDS_OF_THE_QUALITY_GRID_REWARD_SPEC"
    ),
    fixed_latency_stages_ms=FIXED_END_TO_FEEDBACK_STAGES_MS,
    probability_semantics=(
        "P_EDGE_ADMISSION_GIVEN_SENT_EQUALS_P_REASSEMBLY_GIVEN_SENT_TIMES_"
        "P_EDGE_ADMISSION_GIVEN_REASSEMBLED; NOT_END_TO_FEEDBACK_SUCCESS;_"
        "D1_CONDITIONAL_ADMITTED_UTILITY_ASSUMES_EVALUATION_AND_ACK_COMPLETE_"
        "BECAUSE_THEIR_SUCCESS_PROBABILITIES_ARE_UNAVAILABLE"
    ),
)
PILOT_UTILITY_SPEC_SHA256 = (
    "2f6d6aaf22d3995152c5ca9fb726abb07beef32a1b6f846336764ac6f18ea11b"
)
if PILOT_UTILITY_SPEC.canonical_sha256() != PILOT_UTILITY_SPEC_SHA256:
    raise RuntimeError("registered D1 pilot utility canonical hash drift")


@dataclass(frozen=True, slots=True)
class EmpiricalPilotBindingV1:
    """All inputs that can change a D1 observation or expected utility."""

    corrected_p40_binding_sha256: str
    quality_surface_binding_sha256: str
    network_surrogate_sha256: str
    radio_calibration_sha256: str
    modeled_smoke_support_sha256: str
    utility_spec_sha256: str
    normalization_spec_sha256: str
    freshness_policy_sha256: str
    eligible_fit_context_index_sha256: str
    action_catalog_sha256: str
    surface_qualification_report_sha256: str
    profile_order_rng_contract_sha256: str
    implementation_bundle_sha256: str
    schema: str = D1_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != D1_SCHEMA:
            raise ValueError("D1 binding schema drift")
        for name in (
            "corrected_p40_binding_sha256",
            "quality_surface_binding_sha256",
            "network_surrogate_sha256",
            "radio_calibration_sha256",
            "modeled_smoke_support_sha256",
            "utility_spec_sha256",
            "normalization_spec_sha256",
            "freshness_policy_sha256",
            "eligible_fit_context_index_sha256",
            "action_catalog_sha256",
            "surface_qualification_report_sha256",
            "profile_order_rng_contract_sha256",
            "implementation_bundle_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        expected = {
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "action_catalog_sha256": CATALOG_SHA256,
            "surface_qualification_report_sha256": (
                SURFACE_QUALIFICATION_REPORT_SHA256
            ),
            "profile_order_rng_contract_sha256": (
                PROFILE_ORDER_RNG_CONTRACT_SHA256
            ),
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"D1 binding {name} drift")

    def to_canonical_dict(self) -> Dict[str, str]:
        return {
            "action_catalog_sha256": self.action_catalog_sha256,
            "corrected_p40_binding_sha256": self.corrected_p40_binding_sha256,
            "eligible_fit_context_index_sha256": (
                self.eligible_fit_context_index_sha256
            ),
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "implementation_bundle_sha256": self.implementation_bundle_sha256,
            "modeled_smoke_support_sha256": self.modeled_smoke_support_sha256,
            "network_surrogate_sha256": self.network_surrogate_sha256,
            "normalization_spec_sha256": self.normalization_spec_sha256,
            "profile_order_rng_contract_sha256": (
                self.profile_order_rng_contract_sha256
            ),
            "quality_surface_binding_sha256": self.quality_surface_binding_sha256,
            "radio_calibration_sha256": self.radio_calibration_sha256,
            "schema": self.schema,
            "surface_qualification_report_sha256": (
                self.surface_qualification_report_sha256
            ),
            "utility_spec_sha256": self.utility_spec_sha256,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


assert fixed_stage_latency_ms() == 113.0
assert MODELED_SMOKE_SUPPORT_SHA256 == MODELED_SMOKE_SUPPORT.canonical_sha256()
