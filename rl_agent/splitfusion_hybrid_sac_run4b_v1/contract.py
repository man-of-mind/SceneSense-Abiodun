"""Run-4B deployable-observation state, operational prior and reward.

The actor state is exactly 20 values.  Q_perc is a training-reward input only:
the operational prior type has no quality field, so no code path can carry
it into the successor state.  Latency is the operational action-open ->
TAIL_OUTPUT_READY ACK receipt total from the shared provider.

Importing this module performs no I/O, RNG or accelerator operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)

MODE_COUNT = EXPECTED_MODE_COUNT
FEATURE_ORDER: Tuple[str, ...] = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{m}_one_hot" for m in range(MODE_COUNT)),
    "prev_q_normalized",
    "prev_operational_latency_normalized",
    "prev_present",
    "prev_operational_success",
)
FEATURE_COUNT = 20
CRITIC_INPUT_WIDTH = FEATURE_COUNT + MODE_COUNT + 1
UL_MCS_MIN, UL_MCS_MAX = 0, 28
DEADLINE_NS = 170_000_000
DEADLINE_MS = 170.0
LATENCY_WEIGHT = 0.25
FAILURE_REWARD = -1.0
GAMMA = 0.99
DURATION = 2
DISCOUNT = GAMMA ** DURATION

FORBIDDEN_FEATURE_TOKENS = (
    "qperc", "quality", "reward", "gt", "ground", "profile", "family",
    "frame", "scene", "session", "identity", "seq", "timestamp", "snr",
    "tbs", "future", "next", "action_id",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


FEATURE_SCHEMA = {
    "schema_id": "splitfusion_run4b_policy_features_v1",
    "schema_version": 1,
    "feature_count": FEATURE_COUNT,
    "feature_order": list(FEATURE_ORDER),
    "scaling": {
        "camera_si_scaled": "(si - center) / scale, FIT-catalog moments",
        "radar_p40": "frozen unitless [0,1] descriptor",
        "prior_ul_mcs_normalized": "(mcs - 0) / 28",
        "pre_action_rlc_backlog_log1p_scaled": "log1p(B) / log1p(50e6)",
        "prev_q_normalized": f"q_e4 / {Q_E4_MAX}",
        "prev_operational_latency_normalized": (
            "timely operational latency ms / 170; 0 at genesis or failure"),
    },
    "prior_semantics": {
        "genesis": "present=0, all prior fields 0",
        "timely_ack": "action/mode/q retained, latency present, success=1",
        "failure_or_timeout": ("action/mode/q retained, latency=0 (never the "
                               "170-ms censoring value), success=0, present=1"),
    },
    "excluded": ["Q_perc", "previous reward", "GT", "map installation",
                 "identifiers", "profile labels", "family labels",
                 "future information"],
}
FEATURE_SCHEMA_SHA256 = canonical_sha256(FEATURE_SCHEMA)

REWARD_SCHEMA = {
    "schema_id": "splitfusion_run4b_reward_v1",
    "timely": "Q_perc - 0.25 * (L_op_ms / 170.0), L_op <= 170 ms inclusive",
    "timeout_or_registered_failure": FAILURE_REWARD,
    "infrastructure_or_evaluator_fault": "EXCLUDED",
    "latency": "operational action-open -> TAIL_OUTPUT_READY ACK receipt",
}
REWARD_SCHEMA_SHA256 = canonical_sha256(REWARD_SCHEMA)


class ContractError(RuntimeError):
    """A Run-4B state, prior or reward invariant failed (fail closed)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _exact_int(value: Any, name: str) -> int:
    _require(type(value) is int, f"{name} must be an exact int")
    return value


class PriorKind(str, Enum):
    GENESIS = "GENESIS"
    TIMELY_ACK = "TIMELY_ACK"
    FAILURE = "FAILURE"


@dataclass(frozen=True, slots=True)
class OperationalPriorV1:
    """Previous decision as the UE observes it: action and operational ACK.

    There is deliberately no quality or reward field.
    """

    kind: PriorKind
    mode_id: Optional[int] = None
    q_e4: Optional[int] = None
    operational_latency_ns: Optional[int] = None

    def __post_init__(self) -> None:
        _require(type(self.kind) is PriorKind, "kind must be PriorKind")
        if self.kind is PriorKind.GENESIS:
            _require(self.mode_id is None and self.q_e4 is None
                     and self.operational_latency_ns is None,
                     "genesis prior carries no fields")
            return
        _exact_int(self.mode_id, "mode_id")
        _exact_int(self.q_e4, "q_e4")
        _require(0 <= self.mode_id < MODE_COUNT, "mode_id out of range")
        _require(Q_E4_MIN <= self.q_e4 <= Q_E4_MAX, "q_e4 out of range")
        if self.kind is PriorKind.TIMELY_ACK:
            _exact_int(self.operational_latency_ns, "operational_latency_ns")
            _require(0 < self.operational_latency_ns <= DEADLINE_NS,
                     "timely latency must lie in (0, 170 ms]")
        else:
            _require(self.operational_latency_ns is None,
                     "a failure carries no successful latency")

    @classmethod
    def genesis(cls) -> "OperationalPriorV1":
        return cls(PriorKind.GENESIS)

    @classmethod
    def from_outcome(cls, *, mode_id: int, q_e4: int, timely: bool,
                     operational_latency_ns: Optional[int]
                     ) -> "OperationalPriorV1":
        if timely:
            return cls(PriorKind.TIMELY_ACK, mode_id, q_e4,
                       operational_latency_ns)
        return cls(PriorKind.FAILURE, mode_id, q_e4, None)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "mode_id": self.mode_id,
                "q_e4": self.q_e4,
                "operational_latency_ns": self.operational_latency_ns}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OperationalPriorV1":
        _require(set(value) == {"kind", "mode_id", "q_e4",
                                "operational_latency_ns"},
                 "prior fields differ")
        return cls(PriorKind(value["kind"]), value["mode_id"], value["q_e4"],
                   value["operational_latency_ns"])


@dataclass(frozen=True, slots=True)
class ObservationV1:
    """Current deployable measurements available before action open."""

    camera_si: float
    radar_p40: float
    prior_ul_mcs: int
    pre_action_rlc_backlog_bytes: int

    def __post_init__(self) -> None:
        for name in ("camera_si", "radar_p40"):
            value = getattr(self, name)
            _require(type(value) is float and math.isfinite(value),
                     f"{name} must be a finite float")
        _require(0.0 <= self.radar_p40 <= 1.0, "radar_p40 outside [0,1]")
        _exact_int(self.prior_ul_mcs, "prior_ul_mcs")
        _require(UL_MCS_MIN <= self.prior_ul_mcs <= UL_MCS_MAX,
                 "prior_ul_mcs outside the table-0 wire domain")
        _exact_int(self.pre_action_rlc_backlog_bytes, "backlog")
        _require(self.pre_action_rlc_backlog_bytes >= 0, "negative backlog")


@dataclass(frozen=True, slots=True)
class ScalingV1:
    camera_si_center: float
    camera_si_scale: float
    backlog_log1p_scale: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            _require(type(value) is float and math.isfinite(value),
                     f"{item.name} must be a finite float")
        _require(self.camera_si_scale > 0 and self.backlog_log1p_scale > 0,
                 "scales must be positive")

    @property
    def sha256(self) -> str:
        return canonical_sha256({item.name: getattr(self, item.name)
                                 for item in fields(self)})


def build_features(observation: ObservationV1, prior: OperationalPriorV1,
                   scaling: ScalingV1) -> Tuple[float, ...]:
    """The exact 20-value actor vector, in FEATURE_ORDER."""
    _require(type(observation) is ObservationV1, "observation foreign type")
    _require(type(prior) is OperationalPriorV1, "prior foreign type")
    _require(type(scaling) is ScalingV1, "scaling foreign type")
    one_hot = [0.0] * MODE_COUNT
    if prior.kind is PriorKind.GENESIS:
        prev_q = prev_latency = present = success = 0.0
    else:
        one_hot[prior.mode_id] = 1.0
        prev_q = prior.q_e4 / float(Q_E4_MAX)
        present = 1.0
        if prior.kind is PriorKind.TIMELY_ACK:
            prev_latency = (prior.operational_latency_ns / 1_000_000.0
                            / DEADLINE_MS)
            success = 1.0
        else:
            prev_latency = success = 0.0
    values = (
        (observation.camera_si - scaling.camera_si_center)
        / scaling.camera_si_scale,
        observation.radar_p40,
        (observation.prior_ul_mcs - UL_MCS_MIN) / float(UL_MCS_MAX - UL_MCS_MIN),
        math.log1p(observation.pre_action_rlc_backlog_bytes)
        / scaling.backlog_log1p_scale,
        *one_hot, prev_q, prev_latency, present, success,
    )
    _require(len(values) == FEATURE_COUNT, "feature count drift")
    _require(all(math.isfinite(v) for v in values), "non-finite feature")
    return tuple(float(v) for v in values)


class RewardKind(str, Enum):
    TIMELY_SUCCESS = "TIMELY_SUCCESS"
    TIMEOUT_OR_FAILURE = "TIMEOUT_OR_FAILURE"
    EXCLUDED_FAULT = "EXCLUDED_FAULT"


@dataclass(frozen=True, slots=True)
class RewardResolutionV1:
    kind: RewardKind
    reward: Optional[float]
    q_perc: Optional[float]
    operational_latency_ns: Optional[int]

    @property
    def learning_included(self) -> bool:
        return self.kind is not RewardKind.EXCLUDED_FAULT


def resolve_reward(*, kind: RewardKind, q_perc: Optional[float] = None,
                   operational_latency_ns: Optional[int] = None
                   ) -> RewardResolutionV1:
    """Frozen reward: Q_perc - 0.25 L/170 on time, -1 otherwise."""
    _require(type(kind) is RewardKind, "kind must be RewardKind")
    if kind is RewardKind.EXCLUDED_FAULT:
        return RewardResolutionV1(kind, None, None, None)
    if kind is RewardKind.TIMEOUT_OR_FAILURE:
        return RewardResolutionV1(kind, FAILURE_REWARD, None, None)
    _require(type(q_perc) is float and math.isfinite(q_perc),
             "timely success requires a finite Q_perc")
    _exact_int(operational_latency_ns, "operational_latency_ns")
    _require(0 < operational_latency_ns <= DEADLINE_NS,
             "timely latency must lie in (0, 170 ms]")
    latency_ms = operational_latency_ns / 1_000_000.0
    reward = q_perc - LATENCY_WEIGHT * (latency_ms / DEADLINE_MS)
    return RewardResolutionV1(kind, reward, q_perc, operational_latency_ns)
