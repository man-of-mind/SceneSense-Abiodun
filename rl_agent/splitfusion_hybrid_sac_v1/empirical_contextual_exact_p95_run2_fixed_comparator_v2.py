"""Frozen train-only fixed-action comparator for exact-P95 Run-2 v2.

This comparator is deliberately frozen before any Run-2 actor outcome is
opened.  It crosses each of the 391 registered *training* scenes with the four
authored network profiles and scores every one of the 52,240 executable
``(mode_id, q_e4)`` pairs.  Each context/action reward follows the registered
binary64 operation order and is emitted exactly once to an explicit CPU
float32 tensor before deterministic float64 accumulation.

The validation partition is used only to prove sample-ID disjointness.  This
module neither imports nor opens the validation panel/evaluator, reads an actor
checkpoint, trains a policy, nor makes a live-service claim.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    PILOT_UTILITY_SPEC_SHA256,
    fixed_stage_latency_ms,
    require_supported_action,
)
from .empirical_contextual_environment import EmpiricalOneStepEnvironmentV1
from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
    base_p95_expected_utility64_v2,
    shaped_p95_expected_utility64_v2,
)
from .empirical_contextual_exact_p95_run2_replay_v2 import (
    EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
    PREREG_V2_FILE_SHA256,
    RUN2_V2_DEADLINE_MS,
    RUN2_V2_DEADLINE_PENALTY,
    TRAIN_V2_IMPLEMENTATION_SHA256,
    TRAIN_V2_SUMMARY_SHA256,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    EmpiricalFitPartitionV1,
    load_registered_empirical_fit_partition,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .offline_quality_grid.contract import Q_E4_GRID
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .transaction_identity import canonical_sha256

__all__ = [
    "COMPARATOR_SCHEMA_V2",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_CONTEXT_COUNT",
    "ExactP95Run2FixedComparatorV2Error",
    "FixedComparatorContextV2",
    "FixedComparatorDecisionV2",
    "build_train_contexts_v2",
    "enumerate_supported_actions_v2",
    "run_exact_p95_run2_fixed_comparator_v2",
    "select_fixed_action_v2",
]


COMPARATOR_SCHEMA_V2 = "splitfusion.exact_p95_run2_fixed_comparator.v2"
DECISION_SCHEMA_V2 = "splitfusion.exact_p95_run2_fixed_comparator_decision.v2"
CONTEXT_SCHEMA_V2 = "splitfusion.exact_p95_run2_fixed_comparator_context.v2"
ARTIFACT_SCHEMA_V2 = "splitfusion.exact_p95_run2_fixed_comparator_artifacts.v2"
EXPECTED_TRAIN_SCENES = 391
EXPECTED_VALIDATION_SCENES = 85
EXPECTED_CONTEXT_COUNT = EXPECTED_TRAIN_SCENES * len(NETWORK_PROFILE_ORDER)
EXPECTED_ACTION_COUNT = 52_240
EXPECTED_ACTION_CONTEXT_EVALUATIONS = EXPECTED_ACTION_COUNT * EXPECTED_CONTEXT_COUNT
SCALAR_VECTOR_ABS_TOLERANCE = 2e-9
ACCUMULATION_RULE = (
    "CONTEXT_INDEX_ASCENDING_BINARY64_VECTOR_ADD_ONE_CONTEXT_AT_A_TIME;_"
    "DIVIDE_ONCE_BY_1564_AFTER_ALL_FLOAT32_TARGETS_WIDEN_TO_BINARY64"
)
TIE_RULE = "MAX_MEAN_EMITTED_FLOAT32_THEN_MIN_MODE_ID_THEN_MIN_Q_E4"
PREREGISTRATION_FILE_SHA256 = (
    "b1b42a622ae514f469d7a4c49214abe2002739e4bedf5f1c94ef52c3d298b8ac"
)
TRAIN_PENALTY_IMPLEMENTATION_SHA256 = (
    "40975469634225c651c78a8bedcdebd12399f875dd55680c1f83c91573bd20ff"
)
TRAIN_PENALTY_SUMMARY_SHA256 = (
    "4e102b3309c1c2bf8c6e7868957285f440f091a194ff06756d1c5f49d33b8dcd"
)
if PREREG_V2_FILE_SHA256 != PREREGISTRATION_FILE_SHA256:
    raise RuntimeError("comparator/replay preregistration hash disagreement")
if TRAIN_V2_IMPLEMENTATION_SHA256 != TRAIN_PENALTY_IMPLEMENTATION_SHA256:
    raise RuntimeError("comparator/replay penalty implementation disagreement")
if TRAIN_V2_SUMMARY_SHA256 != TRAIN_PENALTY_SUMMARY_SHA256:
    raise RuntimeError("comparator/replay train-summary disagreement")


class ExactP95Run2FixedComparatorV2Error(RuntimeError):
    """The frozen comparator cannot continue without weakening a binding."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _float64_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _emit_cpu_float32_vector(values64: np.ndarray) -> np.ndarray:
    """Perform the registered one-time target emission on explicit CPU."""

    contiguous = np.ascontiguousarray(values64, dtype=np.float64)
    emitted = torch.tensor(contiguous, dtype=torch.float32, device="cpu").numpy()
    if not np.all(np.isfinite(emitted)):
        raise ExactP95Run2FixedComparatorV2Error(
            "emitted float32 target vector contains a non-finite value"
        )
    return emitted


def _emit_cpu_float32_scalar(value64: float) -> float:
    value = float(torch.tensor(float(value64), dtype=torch.float32, device="cpu"))
    if not math.isfinite(value):
        raise ExactP95Run2FixedComparatorV2Error(
            "emitted float32 scalar is non-finite"
        )
    return value


@dataclass(frozen=True, slots=True)
class FixedComparatorContextV2:
    context_index: int
    scene_rank: int
    sample_id: str
    episode_id: str
    frame_id: int
    network_profile: str
    profile_rank: int

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FixedComparatorDecisionV2:
    mode_id: int
    q_e4: int
    mean_emitted_target: float
    runner_up_mode_id: int
    runner_up_q_e4: int
    runner_up_mean_emitted_target: float
    winner_margin: float
    context_count: int
    action_count: int
    action_context_evaluations: int
    context_order_sha256: str
    schema: str = DECISION_SCHEMA_V2

    def __post_init__(self) -> None:
        require_supported_action(self.mode_id, self.q_e4)
        require_supported_action(self.runner_up_mode_id, self.runner_up_q_e4)
        if (self.mode_id, self.q_e4) == (
            self.runner_up_mode_id,
            self.runner_up_q_e4,
        ):
            raise ExactP95Run2FixedComparatorV2Error(
                "winner and runner-up must be distinct"
            )
        for name in (
            "mean_emitted_target",
            "runner_up_mean_emitted_target",
            "winner_margin",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ExactP95Run2FixedComparatorV2Error(
                    f"{name} must be an exact finite float"
                )
        if self.winner_margin != (
            self.mean_emitted_target - self.runner_up_mean_emitted_target
        ) or self.winner_margin < 0.0:
            raise ExactP95Run2FixedComparatorV2Error("winner margin drift")
        if (
            self.context_count != EXPECTED_CONTEXT_COUNT
            or self.action_count != EXPECTED_ACTION_COUNT
            or self.action_context_evaluations
            != EXPECTED_ACTION_CONTEXT_EVALUATIONS
            or self.schema != DECISION_SCHEMA_V2
        ):
            raise ExactP95Run2FixedComparatorV2Error(
                "comparator decision inventory/schema drift"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return asdict(self)


def enumerate_supported_actions_v2() -> Tuple[Tuple[int, int], ...]:
    actions = tuple(
        (mode_id, q_e4)
        for mode_id, (lower, upper) in enumerate(
            MODELED_SMOKE_MODE_Q_E4_BOUNDS
        )
        for q_e4 in range(lower, upper + 1)
    )
    if len(actions) != EXPECTED_ACTION_COUNT or len(set(actions)) != len(actions):
        raise ExactP95Run2FixedComparatorV2Error(
            "registered executable action inventory drift"
        )
    if actions != tuple(sorted(actions)):
        raise ExactP95Run2FixedComparatorV2Error(
            "executable action inventory is not canonical mode/q order"
        )
    for action in actions:
        require_supported_action(*action)
    return actions


def build_train_contexts_v2(
    partition: EmpiricalFitPartitionV1,
) -> Tuple[FixedComparatorContextV2, ...]:
    if type(partition) is not EmpiricalFitPartitionV1:
        raise ExactP95Run2FixedComparatorV2Error(
            "context builder requires exact EmpiricalFitPartitionV1"
        )
    if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
        raise ExactP95Run2FixedComparatorV2Error("fit-partition hash drift")
    train = sorted(
        (row for row in partition.scene_assignments if row.split == TRAIN_SPLIT),
        key=lambda row: (row.sample_id, row.episode_id, row.frame_id),
    )
    validation_ids = {
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    }
    train_ids = {row.sample_id for row in train}
    if (
        len(train) != EXPECTED_TRAIN_SCENES
        or len(train_ids) != EXPECTED_TRAIN_SCENES
        or len(validation_ids) != EXPECTED_VALIDATION_SCENES
        or train_ids.intersection(validation_ids)
    ):
        raise ExactP95Run2FixedComparatorV2Error(
            "train/fit-validation identity partition drift"
        )
    contexts = tuple(
        FixedComparatorContextV2(
            context_index=scene_rank * len(NETWORK_PROFILE_ORDER) + profile_rank,
            scene_rank=scene_rank,
            sample_id=scene.sample_id,
            episode_id=scene.episode_id,
            frame_id=scene.frame_id,
            network_profile=profile,
            profile_rank=profile_rank,
        )
        for scene_rank, scene in enumerate(train)
        for profile_rank, profile in enumerate(NETWORK_PROFILE_ORDER)
    )
    if (
        len(contexts) != EXPECTED_CONTEXT_COUNT
        or tuple(item.context_index for item in contexts)
        != tuple(range(EXPECTED_CONTEXT_COUNT))
    ):
        raise ExactP95Run2FixedComparatorV2Error(
            "registered context cross-product/order drift"
        )
    return contexts


def _context_order_sha256(
    contexts: Sequence[FixedComparatorContextV2],
) -> str:
    return canonical_sha256(
        {
            "contexts": [item.to_canonical_dict() for item in contexts],
            "ordering": (
                "TRAIN_SCENES_SORTED_BY_SAMPLE_ID_EPISODE_ID_FRAME_ID_THEN_"
                "REGISTERED_NETWORK_PROFILE_ORDER"
            ),
            "record": CONTEXT_SCHEMA_V2,
        }
    )


def _curve_vector(curve: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ExactP95Run2FixedComparatorV2Error(
            "curve input must be a finite one-dimensional vector"
        )
    if np.any(x < curve.raw_x_min) or np.any(x > curve.raw_x_max):
        raise ExactP95Run2FixedComparatorV2Error("curve query would extrapolate")
    centers = np.asarray([block.x_center for block in curve.blocks], dtype=np.float64)
    values = np.asarray([block.value for block in curve.blocks], dtype=np.float64)
    weights = np.asarray([block.weight for block in curve.blocks], dtype=np.float64)
    if len(centers) == 0 or not np.all(np.diff(centers) > 0.0):
        raise ExactP95Run2FixedComparatorV2Error("curve centers drifted")
    result = np.empty_like(x)
    support = np.empty_like(x)
    low = x <= centers[0]
    high = x >= centers[-1]
    middle = ~(low | high)
    result[low], support[low] = values[0], weights[0]
    result[high], support[high] = values[-1], weights[-1]
    if np.any(middle):
        xi = x[middle]
        right = np.searchsorted(centers, xi, side="left")
        left = right - 1
        fraction = (xi - centers[left]) / (centers[right] - centers[left])
        result[middle] = values[left] + fraction * (
            values[right] - values[left]
        )
        support[middle] = np.minimum(weights[left], weights[right])
    if not np.all(np.isfinite(result)) or not np.all(support > 0.0):
        raise ExactP95Run2FixedComparatorV2Error("curve vector is invalid")
    return result, support


def _local_latency_support_vector(model: Any, x: np.ndarray) -> np.ndarray:
    knots_x = np.asarray(
        [pair[0] for pair in model.latency_support_knots], dtype=np.float64
    )
    knots_n = np.asarray(
        [pair[1] for pair in model.latency_support_knots], dtype=np.float64
    )
    if len(knots_x) == 0 or not np.all(np.diff(knots_x) > 0.0):
        raise ExactP95Run2FixedComparatorV2Error(
            "latency-support knots drifted"
        )
    if np.any(x < knots_x[0]) or np.any(x > knots_x[-1]):
        raise ExactP95Run2FixedComparatorV2Error(
            "latency-support query escaped its envelope"
        )
    right = np.searchsorted(knots_x, x, side="left")
    bounded = np.minimum(right, len(knots_x) - 1)
    exact = (right < len(knots_x)) & (x == knots_x[bounded])
    result = np.empty_like(x)
    result[exact] = knots_n[right[exact]]
    other = ~exact
    if np.any(other):
        indices = right[other]
        if np.any(indices <= 0) or np.any(indices >= len(knots_x)):
            raise ExactP95Run2FixedComparatorV2Error(
                "latency-support interval was not bracketed"
            )
        result[other] = np.minimum(knots_n[indices - 1], knots_n[indices])
    return result


def _interpolate_same_frame_mode(
    surface: Any, sample_id: str, mode_id: int
) -> Dict[str, np.ndarray]:
    lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
    q = np.arange(lower, upper + 1, dtype=np.int64)
    rows = surface._rows_for(sample_id, mode_id)
    anchor_q = np.asarray([row.q_e4 for row in rows], dtype=np.int64)
    if tuple(int(value) for value in anchor_q) != Q_E4_GRID:
        raise ExactP95Run2FixedComparatorV2Error("quality q-grid drift")
    anchor_payload = np.asarray(
        [row.total_transmitted_bytes for row in rows], dtype=np.float64
    )
    components = [row.component(DIRECT_QUALITY_COMPONENT) for row in rows]
    if any(not item.valid or item.value is None for item in components):
        raise ExactP95Run2FixedComparatorV2Error(
            "q_perc interpolation endpoint is undefined"
        )
    anchor_quality = np.asarray(
        [item.value for item in components], dtype=np.float64
    )
    if not np.all(np.diff(anchor_payload) < 0.0):
        raise ExactP95Run2FixedComparatorV2Error(
            "payload anchors are not strictly decreasing"
        )

    right = np.searchsorted(anchor_q, q, side="left")
    exact = anchor_q[np.minimum(right, len(anchor_q) - 1)] == q

    def interpolate(anchors: np.ndarray) -> np.ndarray:
        result = np.empty(len(q), dtype=np.float64)
        result[exact] = anchors[right[exact]]
        modeled = ~exact
        if np.any(modeled):
            high = right[modeled]
            low = high - 1
            alpha = (q[modeled] - anchor_q[low]) / (
                anchor_q[high] - anchor_q[low]
            )
            # Same separated operation order as the scalar surface _lerp.
            delta = anchors[high] - anchors[low]
            scaled = alpha * delta
            result[modeled] = anchors[low] + scaled
        return result

    payload = interpolate(anchor_payload)
    quality = interpolate(anchor_quality)
    if (
        not np.all(np.isfinite(payload))
        or not np.all(payload > 0.0)
        or not np.all(np.isfinite(quality))
        or np.any(quality < 0.0)
        or np.any(quality > 1.0)
    ):
        raise ExactP95Run2FixedComparatorV2Error(
            "surface interpolation produced an invalid vector"
        )
    datagrams = np.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES).astype(np.int64)
    return {
        "q": q,
        "payload": payload,
        "quality": quality,
        "datagrams": datagrams,
    }


def _network_vector(
    network: Any,
    profile: str,
    payload: np.ndarray,
    datagrams: np.ndarray,
) -> Dict[str, np.ndarray]:
    model = network.profile_models.get(profile)
    if model is None:
        raise ExactP95Run2FixedComparatorV2Error("unknown network profile")
    expected = np.ceil(
        payload / network.contract.udp_payload_capacity_bytes
    ).astype(np.int64)
    if not np.array_equal(datagrams, expected):
        raise ExactP95Run2FixedComparatorV2Error("payload/datagram drift")
    if np.any(datagrams < model.datagram_min) or np.any(
        datagrams > model.datagram_max
    ):
        raise ExactP95Run2FixedComparatorV2Error(
            "datagram count escaped measured profile support"
        )
    x = np.log(payload)
    reassembly, _ = _curve_vector(model.reassembly_curve, x)
    admission, _ = _curve_vector(model.admission_curve, x)
    p_admit = np.clip(reassembly, 0.0, 1.0) * np.clip(
        admission, 0.0, 1.0
    )
    effective = _local_latency_support_vector(model, x)
    latencies: Dict[str, np.ndarray] = {}
    for name in ("p50", "p95", "p99"):
        values, support = _curve_vector(model.latency_curves[name], x)
        latencies[name] = values
        effective = np.minimum(effective, support)
    if np.any(effective < network.contract.latency_min_support):
        raise ExactP95Run2FixedComparatorV2Error(
            "enumerated action lacks qualified latency support"
        )
    fixed = fixed_stage_latency_ms()
    p50 = latencies["p50"]
    p95 = np.maximum(p50, latencies["p95"])
    p99 = np.maximum(p95, latencies["p99"])
    return {
        "p_admit": p_admit,
        "p50": fixed + p50,
        "p95": fixed + p95,
        "p99": fixed + p99,
    }


def _reward_vectors(
    *, p_admit: np.ndarray, quality: np.ndarray, latency_p95: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Keep each ufunc separated to preserve the registered scalar operation
    # order: p*(Q-0.25*(L95/200.0)) + (1.0-p)*(-1.0).
    normalized = latency_p95 / RUN2_V2_DEADLINE_MS
    weighted_latency = 0.25 * normalized
    admitted = quality - weighted_latency
    admitted_term = p_admit * admitted
    failure_term = (1.0 - p_admit) * (-1.0)
    base64 = admitted_term + failure_term
    shaped64 = base64.copy()
    infeasible = latency_p95 > RUN2_V2_DEADLINE_MS
    shaped64[infeasible] = base64[infeasible] - (
        p_admit[infeasible] * RUN2_V2_DEADLINE_PENALTY
    )
    if not np.all(np.isfinite(base64)) or not np.all(np.isfinite(shaped64)):
        raise ExactP95Run2FixedComparatorV2Error("reward vector is non-finite")
    emitted32 = _emit_cpu_float32_vector(shaped64)
    return base64, shaped64, emitted32


def select_fixed_action_v2(
    actions: Sequence[Tuple[int, int]], mean_targets: np.ndarray
) -> Tuple[int, int]:
    """Return winner and runner-up indices under the frozen stable tie rule."""

    if len(actions) != len(mean_targets) or len(actions) < 2:
        raise ExactP95Run2FixedComparatorV2Error(
            "fixed-action selection inventory mismatch"
        )
    if tuple(actions) != tuple(sorted(actions)) or not np.all(
        np.isfinite(mean_targets)
    ):
        raise ExactP95Run2FixedComparatorV2Error(
            "selection requires canonical actions and finite means"
        )
    order = sorted(
        range(len(actions)),
        key=lambda index: (
            -float(mean_targets[index]),
            actions[index][0],
            actions[index][1],
        ),
    )
    return order[0], order[1]


def _require_source_bindings(root: Path) -> Dict[str, Any]:
    prereg = (
        root
        / "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
        "20260921_exact_p95_run2_preregistration_v2/"
        "preregistration_v2.json"
    )
    train = (
        root
        / "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
        "20260921_train_exact_p95_deadline_penalty_v2"
    )
    paths = {
        "preregistration": (prereg, PREREGISTRATION_FILE_SHA256),
        "train_penalty_summary": (
            train / "summary_v2.json",
            TRAIN_PENALTY_SUMMARY_SHA256,
        ),
        "train_penalty_implementation": (
            root
            / "rl_agent/splitfusion_hybrid_sac_v1/"
            "empirical_contextual_exact_p95_deadline_penalty_v2.py",
            TRAIN_PENALTY_IMPLEMENTATION_SHA256,
        ),
    }
    observed: Dict[str, str] = {}
    for label, (path, expected) in paths.items():
        if not path.is_file():
            raise ExactP95Run2FixedComparatorV2Error(
                f"required frozen source is missing: {path}"
            )
        digest = _sha256_file(path)
        if digest != expected:
            raise ExactP95Run2FixedComparatorV2Error(
                f"{label} hash drift: expected {expected}, observed {digest}"
            )
        observed[label] = digest
    return {
        "actor_checkpoint_read_count": 0,
        "d1_pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        "fit_validation_outcome_query_count": 0,
        "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "preregistration_file_sha256": observed["preregistration"],
        "replay_reward_spec_v2_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
        "train_penalty_implementation_sha256": observed[
            "train_penalty_implementation"
        ],
        "train_penalty_spec_sha256": (
            REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        ),
        "train_penalty_summary_sha256": observed["train_penalty_summary"],
    }


def _scalar_winner_revalidation(
    *,
    environment: EmpiricalOneStepEnvironmentV1,
    contexts: Sequence[FixedComparatorContextV2],
    mode_id: int,
    q_e4: int,
) -> Tuple[Tuple[Dict[str, Any], ...], float]:
    rows = []
    accumulated = 0.0
    for context in contexts:
        query = environment._surface.query_fit_q_e4(
            context.sample_id, mode_id, q_e4
        )
        component = query.policy.component(DIRECT_QUALITY_COMPONENT)
        if not component.valid or component.value is None:
            raise ExactP95Run2FixedComparatorV2Error(
                "winner scalar q_perc is unavailable"
            )
        payload = float(query.policy.payload.total_transmitted_bytes)
        datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
        prediction = environment._prediction_session.predict(
            network_profile=context.network_profile,
            payload_bytes=payload,
            datagram_count=datagrams,
        )
        latency = prediction.conditional_retained_survivor_latency_model()
        p_admit = float(prediction.p_edge_admission_given_sent)
        latency_p95 = float(fixed_stage_latency_ms() + latency.p95_ms)
        quality = float(component.value)
        base64 = base_p95_expected_utility64_v2(
            p_admit=p_admit,
            q_perc=quality,
            latency_p95_ms=latency_p95,
        )
        shaped64 = shaped_p95_expected_utility64_v2(
            p_admit=p_admit,
            q_perc=quality,
            latency_p95_ms=latency_p95,
            deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
        )
        emitted32 = _emit_cpu_float32_scalar(shaped64)
        accumulated += float(emitted32)
        rows.append(
            {
                "base_reward64": base64,
                "base_reward64_bits_hex": _float64_bits_hex(base64),
                "context_index": context.context_index,
                "emitted_reward_float32": emitted32,
                "emitted_reward_float32_bits_hex": _float32_bits_hex(emitted32),
                "episode_id": context.episode_id,
                "frame_id": context.frame_id,
                "mode_id": mode_id,
                "network_profile": context.network_profile,
                "p_edge_admission_given_sent": p_admit,
                "payload_bytes": payload,
                "q_e4": q_e4,
                "q_perc": quality,
                "sample_id": context.sample_id,
                "shaped_reward64": shaped64,
                "shaped_reward64_bits_hex": _float64_bits_hex(shaped64),
                "latency_proxy_p95_ms": latency_p95,
            }
        )
    if len(rows) != EXPECTED_CONTEXT_COUNT:
        raise ExactP95Run2FixedComparatorV2Error(
            "winner scalar revalidation context count drift"
        )
    return tuple(rows), accumulated / float(EXPECTED_CONTEXT_COUNT)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ExactP95Run2FixedComparatorV2Error("cannot emit an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ExactP95Run2FixedComparatorV2Error("CSV field order drift")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _report_markdown(summary: Mapping[str, Any]) -> str:
    winner = summary["winner"]
    return "\n".join(
        (
            "# Exact-P95 Run-2 v2 fixed-action comparator",
            "",
            "This comparator was frozen on the registered training partition "
            "before any Run-2 actor outcome was opened.",
            "",
            f"- Winner: mode `{winner['mode_id']}`, q_e4 `{winner['q_e4']}`",
            f"- Mean emitted float32 target: `{winner['mean_emitted_target']:.12g}`",
            f"- Runner-up margin: `{winner['winner_margin']:.12g}`",
            f"- Actions enumerated: `{summary['action_count']}`",
            f"- Training contexts: `{summary['context_count']}`",
            f"- Action-context evaluations: `{summary['action_context_evaluations']}`",
            "- Winner scalar revalidation: `1564/1564`",
            "- Fit-validation outcome queries: `0`",
            "- Actor checkpoint reads: `0`",
            "",
            "The comparator is an offline modeled train-partition baseline, not "
            "a live-service or generalization claim.",
            "",
        )
    )


def run_exact_p95_run2_fixed_comparator_v2(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Exhaustively select and freeze one train-only fixed action."""

    root = _project_root() if project_root is None else Path(project_root).resolve()
    bindings = _require_source_bindings(root)
    partition = load_registered_empirical_fit_partition(project_root=root)
    contexts = build_train_contexts_v2(partition)
    actions = enumerate_supported_actions_v2()
    context_order_sha256 = _context_order_sha256(contexts)
    train_ids = {item.sample_id for item in contexts}
    validation_ids = {
        item.sample_id
        for item in partition.scene_assignments
        if item.split == FIT_VALIDATION_SPLIT
    }
    if train_ids.intersection(validation_ids):
        raise ExactP95Run2FixedComparatorV2Error(
            "validation identity leaked into comparator contexts"
        )

    reward_sums = np.zeros(EXPECTED_ACTION_COUNT, dtype=np.float64)
    action_offset = {
        mode_id: sum(
            upper - lower + 1
            for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS[:mode_id]
        )
        for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS))
    }
    environment = EmpiricalOneStepEnvironmentV1.load_registered(
        seed=0, project_root=root
    )
    try:
        scene_contexts = contexts[:: len(NETWORK_PROFILE_ORDER)]
        if len(scene_contexts) != EXPECTED_TRAIN_SCENES:
            raise ExactP95Run2FixedComparatorV2Error(
                "scene/context projection drift"
            )
        processed_mode_contexts = 0
        for scene in scene_contexts:
            for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)):
                surface = _interpolate_same_frame_mode(
                    environment._surface, scene.sample_id, mode_id
                )
                offset = action_offset[mode_id]
                width = len(surface["q"])
                for profile in NETWORK_PROFILE_ORDER:
                    network = _network_vector(
                        environment._network,
                        profile,
                        surface["payload"],
                        surface["datagrams"],
                    )
                    _base64, _shaped64, emitted32 = _reward_vectors(
                        p_admit=network["p_admit"],
                        quality=surface["quality"],
                        latency_p95=network["p95"],
                    )
                    # Deterministic context-order float64 accumulation.  Each
                    # emitted target is widened only after its one float32
                    # emission; numpy adds contexts one at a time in order.
                    reward_sums[offset : offset + width] += emitted32.astype(
                        np.float64
                    )
                    processed_mode_contexts += 1
        if processed_mode_contexts != (
            EXPECTED_CONTEXT_COUNT * len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)
        ):
            raise ExactP95Run2FixedComparatorV2Error(
                "exhaustive mode/context count drift"
            )
        mean_targets = reward_sums / float(EXPECTED_CONTEXT_COUNT)
        winner_index, runner_up_index = select_fixed_action_v2(
            actions, mean_targets
        )
        winner_mode, winner_q = actions[winner_index]
        runner_mode, runner_q = actions[runner_up_index]
        winner_rows, scalar_winner_mean = _scalar_winner_revalidation(
            environment=environment,
            contexts=contexts,
            mode_id=winner_mode,
            q_e4=winner_q,
        )
        vector_winner_mean = float(mean_targets[winner_index])
        if (
            scalar_winner_mean != vector_winner_mean
            or not math.isfinite(scalar_winner_mean)
        ):
            raise ExactP95Run2FixedComparatorV2Error(
                "winner scalar/vector emitted-target mean mismatch"
            )
    finally:
        environment.close()

    decision = FixedComparatorDecisionV2(
        mode_id=winner_mode,
        q_e4=winner_q,
        mean_emitted_target=vector_winner_mean,
        runner_up_mode_id=runner_mode,
        runner_up_q_e4=runner_q,
        runner_up_mean_emitted_target=float(mean_targets[runner_up_index]),
        winner_margin=(
            vector_winner_mean - float(mean_targets[runner_up_index])
        ),
        context_count=EXPECTED_CONTEXT_COUNT,
        action_count=EXPECTED_ACTION_COUNT,
        action_context_evaluations=EXPECTED_ACTION_CONTEXT_EVALUATIONS,
        context_order_sha256=context_order_sha256,
    )
    action_rows = tuple(
        {
            "mode_id": mode_id,
            "q_e4": q_e4,
            "mean_emitted_float32_target": float(mean_targets[index]),
            "rank": rank,
            "selected": index == winner_index,
        }
        for rank, index in enumerate(
            sorted(
                range(len(actions)),
                key=lambda item: (
                    -float(mean_targets[item]),
                    actions[item][0],
                    actions[item][1],
                ),
            ),
            1,
        )
        for mode_id, q_e4 in (actions[index],)
    )
    bindings = {
        **bindings,
        "accumulation_rule": ACCUMULATION_RULE,
        "comparator_implementation_sha256": _sha256_file(Path(__file__)),
        "context_order_sha256": context_order_sha256,
        "reward_arithmetic": (
            "BINARY64_PINNED_BASE_THEN_CONDITIONAL_P_TIMES_LAMBDA_SUBTRACTION_"
            "THEN_ONE_EXPLICIT_CPU_FLOAT32_EMISSION_PER_CONTEXT_ACTION"
        ),
        "tie_rule": TIE_RULE,
        "train_sample_id_sha256": canonical_sha256(sorted(train_ids)),
        "validation_sample_id_overlap_check_sha256": canonical_sha256(
            sorted(validation_ids)
        ),
    }
    summary: Dict[str, Any] = {
        "action_context_evaluations": EXPECTED_ACTION_CONTEXT_EVALUATIONS,
        "action_count": EXPECTED_ACTION_COUNT,
        "actor_checkpoint_read_count": 0,
        "bindings": bindings,
        "context_count": EXPECTED_CONTEXT_COUNT,
        "context_order_sha256": context_order_sha256,
        "fit_validation_outcome_query_count": 0,
        "fit_validation_sample_id_count_used_only_for_overlap_identity": (
            len(validation_ids)
        ),
        "record": COMPARATOR_SCHEMA_V2,
        "scalar_winner_revalidation_count": len(winner_rows),
        "status": "COMPLETE",
        "train_sample_id_count": len(train_ids),
        "winner": decision.to_canonical_dict(),
    }
    summary["summary_content_sha256"] = canonical_sha256(summary)
    decision_document = {
        "actor_checkpoint_read_count": 0,
        "decision": "GO_FIXED_COMPARATOR_FROZEN_BEFORE_RUN2_OUTCOME_ACCESS",
        "fit_validation_outcome_query_count": 0,
        "record": DECISION_SCHEMA_V2,
        "winner": decision.to_canonical_dict(),
    }
    decision_document["decision_content_sha256"] = canonical_sha256(
        decision_document
    )
    output_dir = Path(output_dir).resolve()
    artifacts = {
        "action_scores.csv": _csv_bytes(action_rows),
        "context_winner_revalidation.csv": _csv_bytes(winner_rows),
        "decision.json": _json_bytes(decision_document),
        "summary.json": _json_bytes(summary),
        "REPORT.md": _report_markdown(summary).encode("utf-8"),
    }
    for name, payload in artifacts.items():
        _atomic_bytes(output_dir / name, payload)
    manifest = {
        "actor_checkpoint_read_count": 0,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifacts.items()
        },
        "fit_validation_outcome_query_count": 0,
        "record": ARTIFACT_SCHEMA_V2,
        "status": "COMPLETE",
        "summary_content_sha256": summary["summary_content_sha256"],
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    _atomic_bytes(output_dir / "artifact_manifest.json", _json_bytes(manifest))
    return summary
