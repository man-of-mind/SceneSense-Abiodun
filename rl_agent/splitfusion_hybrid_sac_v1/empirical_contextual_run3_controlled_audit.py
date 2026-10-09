"""Internal controlled offline audit of the completed Run-3 actors.

This module is deliberately *not* a temporal replay and *not* live evidence.
It evaluates deterministic actors on the registered 340-entry reward-held
fit-validation panel.  Panel entries are ordered held-out contexts, but their
source CARLA frame gaps are irregular and their radio rows are assigned by a
registered round robin.  Consequently ``scene_rank`` is only a canonical panel
order and must never be read as elapsed wall-clock time or a 100-ms route.

The audit matches the one-step/genesis state distribution used by Run-3.  It
does not invent chained previous-feedback features.  It reports the learned
action, the fixed mode-11/q6095 comparator, and an exhaustive per-context
oracle over every one of the 52,240 executable ``(mode_id, q_e4)`` actions.
All rewards are the analytic expectation of the registered Run-3 transition
kernel; the historical D1 expected-utility reward is not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import fixed_stage_latency_ms, require_supported_action
from .empirical_contextual_fit_validation_evaluator import (
    FitValidationActorEvaluatorV1,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
    FitValidationPanelEntryV1,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_run3_reward import (
    RUN3_KERNEL_SPEC_SHA256,
    RUN3_REWARD_SPEC,
    RUN3_REWARD_SPEC_SHA256,
    QuantileLatencyProxyV1,
    expected_run3_reward,
)
from .empirical_contextual_run3_runner import (
    RUN3_PREFLIGHT_RESULT_SHA256,
    RUN3_REGISTERED_CONFIG,
    _hash_state,
)
from .empirical_contextual_split_oracle import (
    EXPECTED_EXACT_ACTION_COUNT_PER_SCENE,
    _curve_vector,
    _local_latency_support_vector,
    _surface_mode_vector,
)
from .hybrid_sac_models import HybridSacModelConfig, build_actor
from .modeled_smoke_support import MODELED_SMOKE_MODE_Q_E4_BOUNDS, MODELED_SMOKE_SUPPORT
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .run_empirical_contextual_run3 import load_run3_checkpoint
from .transaction_identity import canonical_sha256

__all__ = [
    "AUDIT_CLASS",
    "AUDIT_SCOPE_DISCLOSURE",
    "AUDIT_UPDATES",
    "FIXED_COMPARATOR",
    "ActionAuditV1",
    "ControlledAuditError",
    "ControlledAuditRowV1",
    "aggregate_controlled_rows",
    "aggregate_paired_progress",
    "expected_run3_reward_vector",
    "run_controlled_audit",
]


AUDIT_SCHEMA = "splitfusion.run3_internal_controlled_offline_audit.v1"
AUDIT_CLASS = "INTERNAL_CONTROLLED_OFFLINE_AUDIT_NOT_TEMPORAL_OR_LIVE"
AUDIT_SCOPE_DISCLOSURE = (
    "The 340 panel entries are ordered reward-held fit-validation contexts, "
    "not consecutive route frames or elapsed time. Source frame gaps are "
    "irregular; radio rows are deterministic round-robin assignments. The "
    "audit uses genesis previous-feedback features to match Run-3 training. "
    "It is not live deployment, final-test generalization, or an SLA claim. "
    "Rewards, event probabilities, and latency quantiles are analytic "
    "model-surrogate expectations, not realized outcomes or measured runtime "
    "latency. Quality values are frozen empirical quality-surface lookups or "
    "interpolations for each selected scene/action; this audit does not rerun "
    "the perception model."
)
AUDIT_UPDATES: Tuple[int, ...] = (0, 5_000, 10_000)
FIXED_COMPARATOR: Tuple[int, int] = (11, 6_095)
FIXED_COMPARATOR_PROVENANCE = (
    "TRAIN_SPLIT_SELECTED_MODE11_Q6095_NOT_SELECTED_ON_THIS_VALIDATION_PANEL"
)
_QUALITY_COMPONENTS: Tuple[str, ...] = (
    "seg_vehicle_iou",
    "seg_person_iou",
    "vehicle_recall",
    "person_recall",
    "vehicle_xy_error_m",
    "person_xy_error_m",
    "q_seg",
    "q_loc",
    "q_perc",
)
_SCALAR_VECTOR_TOLERANCE = 3e-9


class ControlledAuditError(RuntimeError):
    """A source binding, evaluation invariant, or output gate failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ControlledAuditError(f"{path} is not a JSON object")
    return value


def _verify_embedded_hash(document: Mapping[str, Any], field: str) -> str:
    materialized = dict(document)
    supplied = materialized.pop(field, None)
    if supplied != canonical_sha256(materialized):
        raise ControlledAuditError(f"embedded {field} does not verify")
    return str(supplied)


@dataclass(frozen=True, slots=True)
class ComponentAuditV1:
    value: Optional[float]
    valid: bool
    status: str

    def __post_init__(self) -> None:
        if type(self.valid) is not bool or type(self.status) is not str or not self.status:
            raise ControlledAuditError("quality component validity/status is malformed")
        if self.value is None:
            if self.valid:
                raise ControlledAuditError("a missing quality value cannot be valid")
        elif not self.valid or not math.isfinite(self.value):
            raise ControlledAuditError("a numeric quality value must be finite and valid")


@dataclass(frozen=True, slots=True)
class ActionAuditV1:
    mode_id: int
    q_e4: int
    requested_q: Optional[float]
    family: str
    quantizer: str
    seg_vehicle_iou: ComponentAuditV1
    seg_person_iou: ComponentAuditV1
    vehicle_recall: ComponentAuditV1
    person_recall: ComponentAuditV1
    vehicle_xy_error_m: ComponentAuditV1
    person_xy_error_m: ComponentAuditV1
    q_seg: ComponentAuditV1
    q_loc: ComponentAuditV1
    q_perc: ComponentAuditV1
    total_transmitted_bytes: float
    datagram_count: int
    p_complete_reassembly_given_sent: float
    p_edge_admission_given_reassembled: float
    p_edge_admission_given_sent: float
    conditional_feature_uplink_p50_ms: float
    conditional_feature_uplink_p95_ms: float
    conditional_feature_uplink_p99_ms: float
    fixed_stage_latency_ms: float
    total_latency_p50_ms: float
    total_latency_p95_ms: float
    total_latency_p99_ms: float
    budget_miss_p50: bool
    budget_miss_p95: bool
    budget_miss_p99: bool
    expected_reward: float
    p_reassembly_failure: float
    p_admission_failure: float
    p_admitted: float
    p_timeout_given_admitted: float
    p_timely_feedback: float
    conditional_admitted_expected_reward: float

    def __post_init__(self) -> None:
        require_supported_action(self.mode_id, self.q_e4)
        if self.requested_q is not None and not math.isfinite(self.requested_q):
            raise ControlledAuditError("requested_q is non-finite")
        if not self.q_perc.valid or self.q_perc.value is None:
            raise ControlledAuditError("q_perc must be available for an audited action")
        for name in (
            "total_transmitted_bytes",
            "p_complete_reassembly_given_sent",
            "p_edge_admission_given_reassembled",
            "p_edge_admission_given_sent",
            "conditional_feature_uplink_p50_ms",
            "conditional_feature_uplink_p95_ms",
            "conditional_feature_uplink_p99_ms",
            "fixed_stage_latency_ms",
            "total_latency_p50_ms",
            "total_latency_p95_ms",
            "total_latency_p99_ms",
            "expected_reward",
            "p_reassembly_failure",
            "p_admission_failure",
            "p_admitted",
            "p_timeout_given_admitted",
            "p_timely_feedback",
            "conditional_admitted_expected_reward",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ControlledAuditError(f"{name} is non-finite")
        if not (
            self.total_latency_p50_ms <= self.total_latency_p95_ms
            <= self.total_latency_p99_ms
        ):
            raise ControlledAuditError("latency percentiles are not ordered")
        if self.datagram_count != math.ceil(
            self.total_transmitted_bytes / UDP_PAYLOAD_CAPACITY_BYTES
        ):
            raise ControlledAuditError("payload/datagram relation drift")
        expected_misses = (
            self.total_latency_p50_ms > RUN3_REWARD_SPEC.deadline_ms,
            self.total_latency_p95_ms > RUN3_REWARD_SPEC.deadline_ms,
            self.total_latency_p99_ms > RUN3_REWARD_SPEC.deadline_ms,
        )
        if expected_misses != (
            self.budget_miss_p50,
            self.budget_miss_p95,
            self.budget_miss_p99,
        ):
            raise ControlledAuditError("200-ms deadline flags drifted")

    def flat(self, prefix: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in ("mode_id", "q_e4", "requested_q", "family", "quantizer"):
            result[f"{prefix}{name}"] = getattr(self, name)
        for name in _QUALITY_COMPONENTS:
            component: ComponentAuditV1 = getattr(self, name)
            result[f"{prefix}{name}"] = component.value
            result[f"{prefix}{name}_valid"] = component.valid
            result[f"{prefix}{name}_status"] = component.status
        skip = {"mode_id", "q_e4", "requested_q", "family", "quantizer", *_QUALITY_COMPONENTS}
        for name in self.__dataclass_fields__:
            if name not in skip:
                result[f"{prefix}{name}"] = getattr(self, name)
        return result


@dataclass(frozen=True, slots=True)
class ControlledAuditRowV1:
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_content_sha256: str
    checkpoint_file_sha256: str
    panel_index: int
    scene_rank: int
    sample_id: str
    episode_id: str
    frame_id: int
    network_profile: str
    radio_csv_row_number: int
    radio_trace_step_index: int
    camera_si: float
    radar_p40: float
    learned: ActionAuditV1
    fixed: ActionAuditV1
    oracle: ActionAuditV1
    learned_oracle_regret: float
    fixed_oracle_regret: float
    learned_minus_fixed_reward: float
    audit_class: str = AUDIT_CLASS

    def __post_init__(self) -> None:
        if self.audit_class != AUDIT_CLASS:
            raise ControlledAuditError("audit-class drift")
        for value in (
            self.learned_oracle_regret,
            self.fixed_oracle_regret,
            self.learned_minus_fixed_reward,
        ):
            if not math.isfinite(value):
                raise ControlledAuditError("non-finite comparison value")
        if self.learned_oracle_regret < -_SCALAR_VECTOR_TOLERANCE:
            raise ControlledAuditError("learned action exceeds exhaustive oracle")
        if self.fixed_oracle_regret < -_SCALAR_VECTOR_TOLERANCE:
            raise ControlledAuditError("fixed action exceeds exhaustive oracle")

    def flat(self) -> Dict[str, Any]:
        result = {
            name: getattr(self, name)
            for name in (
                "seed",
                "update_index",
                "actor_state_sha256",
                "checkpoint_content_sha256",
                "checkpoint_file_sha256",
                "panel_index",
                "scene_rank",
                "sample_id",
                "episode_id",
                "frame_id",
                "network_profile",
                "radio_csv_row_number",
                "radio_trace_step_index",
                "camera_si",
                "radar_p40",
                "learned_oracle_regret",
                "fixed_oracle_regret",
                "learned_minus_fixed_reward",
                "audit_class",
            )
        }
        result.update(self.learned.flat("learned_"))
        result.update(self.fixed.flat("fixed_"))
        result.update(self.oracle.flat("oracle_"))
        return result


def _expected_run3_vector_metrics(
    *,
    q_perc: np.ndarray,
    p_reassembly: np.ndarray,
    p_admission_given_reassembled: np.ndarray,
    latency_p50_ms: np.ndarray,
    latency_p95_ms: np.ndarray,
    latency_p99_ms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expected reward, admitted-timeout mass, and timely probability."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            q_perc,
            p_reassembly,
            p_admission_given_reassembled,
            latency_p50_ms,
            latency_p95_ms,
            latency_p99_ms,
        )
    )
    shape = arrays[0].shape
    if any(value.ndim != 1 or value.shape != shape for value in arrays):
        raise ControlledAuditError("Run-3 vector inputs must share one 1-D shape")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ControlledAuditError("Run-3 vector inputs contain non-finite values")
    quality, p_re, p_adm, p50, p95, p99 = arrays
    if (
        np.any((quality < 0.0) | (quality > 1.0))
        or np.any((p_re < 0.0) | (p_re > 1.0))
        or np.any((p_adm < 0.0) | (p_adm > 1.0))
        or np.any(p50 < 0.0)
        or np.any(p95 < p50)
        or np.any(p99 < p95)
    ):
        raise ControlledAuditError("Run-3 vector input range/order drift")

    lower = np.maximum(0.0, 2.0 * p50 - p95)
    coordinates = (lower, p50, p95, p99, p99)
    quantiles = (0.0, 0.5, 0.95, 0.99, 1.0)
    conditional_reward = np.zeros(shape, dtype=np.float64)
    timeout_given_admitted = np.zeros(shape, dtype=np.float64)
    for u0, u1, y0, y1 in zip(
        quantiles, quantiles[1:], coordinates, coordinates[1:]
    ):
        width = u1 - u0
        all_timeout = y0 > RUN3_REWARD_SPEC.deadline_ms
        all_timely = y1 <= RUN3_REWARD_SPEC.deadline_ms
        crossing = ~(all_timeout | all_timely)
        conditional_reward[all_timeout] -= width
        timeout_given_admitted[all_timeout] += width
        conditional_reward[all_timely] += width * (
            quality[all_timely]
            - RUN3_REWARD_SPEC.latency_weight
            * ((y0[all_timely] + y1[all_timely]) / 2.0)
            / RUN3_REWARD_SPEC.deadline_ms
        )
        if np.any(crossing):
            fraction = (
                (RUN3_REWARD_SPEC.deadline_ms - y0[crossing])
                / (y1[crossing] - y0[crossing])
            )
            timely_width = width * fraction
            timeout_width = width - timely_width
            conditional_reward[crossing] += timely_width * (
                quality[crossing]
                - RUN3_REWARD_SPEC.latency_weight
                * ((y0[crossing] + RUN3_REWARD_SPEC.deadline_ms) / 2.0)
                / RUN3_REWARD_SPEC.deadline_ms
            ) - timeout_width
            timeout_given_admitted[crossing] += timeout_width
    p_admitted = p_re * p_adm
    result = (
        -(1.0 - p_re)
        - p_re * (1.0 - p_adm)
        + p_admitted * conditional_reward
    )
    p_timely = p_admitted * (1.0 - timeout_given_admitted)
    if not np.all(np.isfinite(result)) or not np.all(np.isfinite(p_timely)):
        raise ControlledAuditError("Run-3 vector metrics are non-finite")
    return result, timeout_given_admitted, p_timely


def expected_run3_reward_vector(
    *,
    q_perc: np.ndarray,
    p_reassembly: np.ndarray,
    p_admission_given_reassembled: np.ndarray,
    latency_p50_ms: np.ndarray,
    latency_p95_ms: np.ndarray,
    latency_p99_ms: np.ndarray,
) -> np.ndarray:
    """Vector equivalent of :func:`expected_run3_reward` for oracle search."""

    result, _timeout, _timely = _expected_run3_vector_metrics(
        q_perc=q_perc,
        p_reassembly=p_reassembly,
        p_admission_given_reassembled=p_admission_given_reassembled,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=latency_p99_ms,
    )
    return result


def _candidate_key(
    expected_reward: float,
    p_timely_feedback: float,
    latency_p50_ms: float,
    mode_id: int,
    q_e4: int,
) -> Tuple[float, float, float, int, int]:
    """Registered Run-3 key: reward, timely, -P50, -mode, -q."""

    return (
        expected_reward,
        p_timely_feedback,
        -latency_p50_ms,
        -mode_id,
        -q_e4,
    )


def _network_vectors(network: Any, profile: str, payload: np.ndarray, datagrams: np.ndarray) -> Dict[str, np.ndarray]:
    model = network.profile_models.get(profile)
    if model is None:
        raise ControlledAuditError(f"unknown network profile {profile!r}")
    if payload.ndim != 1 or datagrams.ndim != 1 or payload.shape != datagrams.shape:
        raise ControlledAuditError(
            "oracle payload/datagram vectors must be aligned 1-D arrays"
        )
    if np.any(~np.isfinite(payload)) or np.any(payload <= 0.0):
        raise ControlledAuditError(
            "oracle payload vector must be positive and finite"
        )
    if (
        np.any(datagrams < model.datagram_min)
        or np.any(datagrams > model.datagram_max)
    ):
        raise ControlledAuditError(
            "oracle datagram count is outside calibrated support"
        )
    x = np.log(payload)
    p_reassembly, _ = _curve_vector(model.reassembly_curve, x)
    p_admission, _ = _curve_vector(model.admission_curve, x)
    p_reassembly = np.clip(p_reassembly, 0.0, 1.0)
    p_admission = np.clip(p_admission, 0.0, 1.0)
    local_support = _local_latency_support_vector(model, x)
    latency: Dict[str, np.ndarray] = {}
    effective_support = local_support.copy()
    for name in ("p50", "p95", "p99"):
        values, support = _curve_vector(model.latency_curves[name], x)
        latency[name] = values
        effective_support = np.minimum(effective_support, support)
    if np.any(effective_support < network.contract.latency_min_support):
        raise ControlledAuditError("oracle action lacks qualified latency support")
    p50 = latency["p50"]
    p95 = np.maximum(p50, latency["p95"])
    p99 = np.maximum(p95, latency["p99"])
    fixed = fixed_stage_latency_ms()
    expected_datagrams = np.ceil(
        payload / network.contract.udp_payload_capacity_bytes
    ).astype(np.int64)
    if not np.array_equal(datagrams, expected_datagrams):
        raise ControlledAuditError("oracle payload/datagram relation drift")
    return {
        "p_reassembly": p_reassembly,
        "p_admission": p_admission,
        "p50": fixed + p50,
        "p95": fixed + p95,
        "p99": fixed + p99,
    }


def _component(policy: Any, name: str) -> ComponentAuditV1:
    raw = policy.component(name)
    return ComponentAuditV1(
        value=None if raw.value is None else float(raw.value),
        valid=bool(raw.valid),
        status=str(raw.status),
    )


def _evaluate_action(
    evaluator: FitValidationActorEvaluatorV1,
    entry: FitValidationPanelEntryV1,
    mode_id: int,
    q_e4: int,
    *,
    requested_q: Optional[float],
) -> ActionAuditV1:
    require_supported_action(mode_id, q_e4)
    query = evaluator.environment._surface.query_fit_q_e4(
        entry.scene_sample_id, mode_id, q_e4
    )
    policy = query.policy
    payload = float(policy.payload.total_transmitted_bytes)
    datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
    prediction = evaluator.environment._prediction_session.predict(
        network_profile=entry.network_profile,
        payload_bytes=payload,
        datagram_count=datagrams,
    )
    uplink = prediction.conditional_retained_survivor_latency_model()
    fixed = fixed_stage_latency_ms()
    totals = (fixed + uplink.p50_ms, fixed + uplink.p95_ms, fixed + uplink.p99_ms)
    latency_proxy = QuantileLatencyProxyV1(*map(float, totals))
    q_perc = _component(policy, "q_perc")
    if q_perc.value is None or not q_perc.valid:
        raise ControlledAuditError("audited action has unavailable q_perc")
    expected = expected_run3_reward(
        q_perc=q_perc.value,
        p_complete_reassembly_given_sent=float(
            prediction.p_complete_reassembly_given_sent
        ),
        p_edge_admission_given_reassembled=float(
            prediction.p_edge_admission_given_reassembled
        ),
        latency_proxy=latency_proxy,
    )
    components = {name: _component(policy, name) for name in _QUALITY_COMPONENTS}
    return ActionAuditV1(
        mode_id=mode_id,
        q_e4=q_e4,
        requested_q=requested_q,
        family=policy.family,
        quantizer=policy.quantizer,
        **components,
        total_transmitted_bytes=payload,
        datagram_count=datagrams,
        p_complete_reassembly_given_sent=float(
            prediction.p_complete_reassembly_given_sent
        ),
        p_edge_admission_given_reassembled=float(
            prediction.p_edge_admission_given_reassembled
        ),
        p_edge_admission_given_sent=float(prediction.p_edge_admission_given_sent),
        conditional_feature_uplink_p50_ms=float(uplink.p50_ms),
        conditional_feature_uplink_p95_ms=float(uplink.p95_ms),
        conditional_feature_uplink_p99_ms=float(uplink.p99_ms),
        fixed_stage_latency_ms=float(fixed),
        total_latency_p50_ms=float(totals[0]),
        total_latency_p95_ms=float(totals[1]),
        total_latency_p99_ms=float(totals[2]),
        budget_miss_p50=totals[0] > RUN3_REWARD_SPEC.deadline_ms,
        budget_miss_p95=totals[1] > RUN3_REWARD_SPEC.deadline_ms,
        budget_miss_p99=totals[2] > RUN3_REWARD_SPEC.deadline_ms,
        expected_reward=float(expected.expected_reward),
        p_reassembly_failure=float(expected.p_reassembly_failure),
        p_admission_failure=float(expected.p_admission_failure),
        p_admitted=float(expected.p_admitted),
        p_timeout_given_admitted=float(expected.p_timeout_given_admitted),
        p_timely_feedback=float(expected.p_timely_feedback),
        conditional_admitted_expected_reward=float(
            expected.conditional_admitted_expected_reward
        ),
    )


def _oracle_action(
    evaluator: FitValidationActorEvaluatorV1,
    entry: FitValidationPanelEntryV1,
) -> ActionAuditV1:
    best_key: Optional[Tuple[float, float, float, int, int]] = None
    best_identity: Optional[Tuple[int, int]] = None
    best_vector_fields: Optional[Dict[str, float]] = None
    evaluated = 0
    for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)):
        surface = _surface_mode_vector(
            evaluator.environment._surface, entry.scene_sample_id, mode_id
        )
        if not np.all(np.diff(surface["q"]) == 1):
            raise ControlledAuditError("oracle q_e4 vector is not strictly ascending")
        network = _network_vectors(
            evaluator.environment._network,
            entry.network_profile,
            surface["payload"],
            surface["datagrams"],
        )
        reward, _timeout, p_timely = _expected_run3_vector_metrics(
            q_perc=surface["quality"],
            p_reassembly=network["p_reassembly"],
            p_admission_given_reassembled=network["p_admission"],
            latency_p50_ms=network["p50"],
            latency_p95_ms=network["p95"],
            latency_p99_ms=network["p99"],
        )
        evaluated += len(reward)
        indexes = np.flatnonzero(reward == np.max(reward))
        indexes = indexes[p_timely[indexes] == np.max(p_timely[indexes])]
        indexes = indexes[
            network["p50"][indexes] == np.min(network["p50"][indexes])
        ]
        index = int(indexes[0])  # q strictly ascends, so this is minimum q.
        q_e4 = int(surface["q"][index])
        candidate_key = _candidate_key(
            float(reward[index]),
            float(p_timely[index]),
            float(network["p50"][index]),
            mode_id,
            q_e4,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_identity = (mode_id, q_e4)
            best_vector_fields = {
                "q_perc": float(surface["quality"][index]),
                "payload": float(surface["payload"][index]),
                "p_reassembly": float(network["p_reassembly"][index]),
                "p_admission": float(network["p_admission"][index]),
                "p50": float(network["p50"][index]),
                "p95": float(network["p95"][index]),
                "p99": float(network["p99"][index]),
                "p_timely": float(p_timely[index]),
            }
    if (
        evaluated != EXPECTED_EXACT_ACTION_COUNT_PER_SCENE
        or best_key is None
        or best_identity is None
        or best_vector_fields is None
    ):
        raise ControlledAuditError("exhaustive oracle support count drift")
    scalar = _evaluate_action(
        evaluator, entry, *best_identity, requested_q=None
    )
    checks = {
        "expected_reward": (scalar.expected_reward, best_key[0]),
        "q_perc": (float(scalar.q_perc.value), best_vector_fields["q_perc"]),
        "payload": (scalar.total_transmitted_bytes, best_vector_fields["payload"]),
        "p_reassembly": (
            scalar.p_complete_reassembly_given_sent,
            best_vector_fields["p_reassembly"],
        ),
        "p_admission": (
            scalar.p_edge_admission_given_reassembled,
            best_vector_fields["p_admission"],
        ),
        "p50": (scalar.total_latency_p50_ms, best_vector_fields["p50"]),
        "p95": (scalar.total_latency_p95_ms, best_vector_fields["p95"]),
        "p99": (scalar.total_latency_p99_ms, best_vector_fields["p99"]),
        "p_timely": (scalar.p_timely_feedback, best_vector_fields["p_timely"]),
    }
    for name, (observed, expected) in checks.items():
        if not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=_SCALAR_VECTOR_TOLERANCE
        ):
            raise ControlledAuditError(
                f"oracle scalar/vector winner mismatch for {name}"
            )
    return scalar
@dataclass(frozen=True, slots=True)
class LoadedRun3ActorV1:
    actor: torch.nn.Module
    seed: int
    update_index: int
    actor_state_sha256: str
    checkpoint_content_sha256: str
    checkpoint_file_sha256: str


def _verify_fixed_comparator(project_root: Path) -> Dict[str, Any]:
    directory = project_root / (
        "experiments/splitfusion_hybrid_sac_run3_preflight_v1/"
        "20260922_train_only_reward_v1_run2"
    )
    preflight = directory / "preflight.json"
    baseline = directory / "train_only_baselines.csv"
    expected_preflight_sha = RUN3_PREFLIGHT_RESULT_SHA256
    if expected_preflight_sha != (
        "52801ce029eaac4dfde4834a2d8031429663a3efb383deed7ec1ed06d50ebab7"
    ):
        raise ControlledAuditError("compiled Run-3 preflight binding drift")
    expected_baseline_sha = (
        "fa9f37dfa4a13bb5b6656c4c11058fd0f087b36ab7b0c1693d50780a660cfda5"
    )
    if _sha256_file(preflight) != expected_preflight_sha:
        raise ControlledAuditError("train-only Run-3 preflight SHA-256 drift")
    if _sha256_file(baseline) != expected_baseline_sha:
        raise ControlledAuditError("train-only fixed-baseline CSV SHA-256 drift")
    manifest = _read_json(directory / "manifest.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("preflight_result_sha256") != expected_preflight_sha
        or manifest.get("artifact_sha256", {}).get("preflight.json")
        != expected_preflight_sha
        or manifest.get("artifact_sha256", {}).get("train_only_baselines.csv")
        != expected_baseline_sha
        or manifest.get("reward_spec_sha256") != RUN3_REWARD_SPEC_SHA256
        or manifest.get("kernel_spec_sha256") != RUN3_KERNEL_SPEC_SHA256
    ):
        raise ControlledAuditError("train-only fixed-comparator manifest drift")
    with baseline.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    selected = [row for row in rows if row.get("label") == "TRAIN_SELECTED_FIXED_ACTION"]
    if (
        len(selected) != 1
        or int(selected[0]["mode_id"]) != FIXED_COMPARATOR[0]
        or int(selected[0]["q_e4"]) != FIXED_COMPARATOR[1]
    ):
        raise ControlledAuditError("train-selected fixed action identity drift")
    return {
        "provenance": FIXED_COMPARATOR_PROVENANCE,
        "preflight_result_file_sha256": expected_preflight_sha,
        "train_only_baselines_csv_sha256": expected_baseline_sha,
        "label": "TRAIN_SELECTED_FIXED_ACTION",
        "mode_id": FIXED_COMPARATOR[0],
        "q_e4": FIXED_COMPARATOR[1],
    }


def _verify_campaign(
    campaign: Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    manifest_path = campaign / "campaign_manifest.json"
    manifest = _read_json(manifest_path)
    manifest_content_sha = _verify_embedded_hash(
        manifest, "campaign_manifest_sha256"
    )
    if (
        manifest.get("config") != RUN3_REGISTERED_CONFIG.to_canonical_dict()
        or manifest.get("config_sha256") != RUN3_REGISTERED_CONFIG.canonical_sha256()
        or manifest.get("seeds") != list(RUN3_REGISTERED_CONFIG.seeds)
    ):
        raise ControlledAuditError("Run-3 campaign manifest identity drift")
    complete_path = campaign / "campaign_complete.json"
    complete = _read_json(complete_path)
    complete_content_sha = _verify_embedded_hash(complete, "campaign_sha256")
    if (
        complete.get("config_sha256") != RUN3_REGISTERED_CONFIG.canonical_sha256()
        or complete.get("manifest_file_sha256") != _sha256_file(manifest_path)
        or complete.get("seeds") != list(RUN3_REGISTERED_CONFIG.seeds)
    ):
        raise ControlledAuditError("Run-3 completion identity drift")
    reports: Dict[int, Dict[str, Any]] = {}
    terminal_bindings: Dict[str, Any] = {}
    for seed in RUN3_REGISTERED_CONFIG.seeds:
        seed_dir = campaign / f"seed_{seed}"
        seed_manifest_path = seed_dir / "manifest.json"
        seed_manifest = _read_json(seed_manifest_path)
        _verify_embedded_hash(seed_manifest, "manifest_sha256")
        if (
            seed_manifest.get("seed") != seed
            or seed_manifest.get("config")
            != RUN3_REGISTERED_CONFIG.to_canonical_dict()
            or seed_manifest.get("config_sha256")
            != RUN3_REGISTERED_CONFIG.canonical_sha256()
        ):
            raise ControlledAuditError("Run-3 seed manifest identity drift")
        report_path = seed_dir / "report.json"
        report = _read_json(report_path)
        report_sha = _verify_embedded_hash(report, "report_sha256")
        if complete.get("seed_reports", {}).get(str(seed)) != report_sha:
            raise ControlledAuditError("campaign/seed report binding drift")
        summary = report.get("summary", {})
        if (
            report.get("fixed_endpoint_update") != 10_000
            or summary.get("seed") != seed
            or summary.get("completed_updates") != 10_000
            or summary.get("completed_training_hard_gates_passed") is not True
        ):
            raise ControlledAuditError("Run-3 seed is not a completed fixed endpoint")
        terminal_path = seed_dir / "RUN3_TRAINING_COMPLETE.json"
        terminal = _read_json(terminal_path)
        terminal_sha = _verify_embedded_hash(terminal, "terminal_sha256")
        if (
            terminal.get("seed") != seed
            or terminal.get("update") != 10_000
            or terminal.get("status") != "RUN3_FIXED_ENDPOINT_COMPLETE"
            or terminal.get("report_sha256") != report_sha
            or terminal.get("report_file_sha256") != _sha256_file(report_path)
        ):
            raise ControlledAuditError("Run-3 terminal/report binding drift")
        pointer_path = seed_dir / "latest_checkpoint.json"
        pointer = _read_json(pointer_path)
        pointer_sha = _verify_embedded_hash(pointer, "pointer_sha256")
        endpoint_path = seed_dir / str(pointer.get("path"))
        if (
            pointer.get("path") != "checkpoints/full_010000.pt"
            or pointer.get("update") != 10_000
            or pointer.get("checkpoint_file_sha256") != _sha256_file(endpoint_path)
            or report.get("artifact_hashes", {}).get("latest_checkpoint.json")
            != _sha256_file(pointer_path)
        ):
            raise ControlledAuditError("Run-3 endpoint pointer binding drift")
        endpoint = load_run3_checkpoint(endpoint_path)
        if (
            endpoint.checkpoint_sha256 != pointer.get("checkpoint_content_sha256")
            or summary.get("checkpoint_sha256") != endpoint.checkpoint_sha256
        ):
            raise ControlledAuditError("Run-3 endpoint content binding drift")
        reports[seed] = report
        terminal_bindings[str(seed)] = {
            "report_content_sha256": report_sha,
            "terminal_content_sha256": terminal_sha,
            "terminal_file_sha256": _sha256_file(terminal_path),
            "latest_pointer_content_sha256": pointer_sha,
            "latest_pointer_file_sha256": _sha256_file(pointer_path),
            "endpoint_checkpoint_content_sha256": endpoint.checkpoint_sha256,
        }
    return reports, {
        "campaign_manifest_content_sha256": manifest_content_sha,
        "campaign_manifest_file_sha256": _sha256_file(manifest_path),
        "campaign_completion_content_sha256": complete_content_sha,
        "campaign_completion_file_sha256": _sha256_file(complete_path),
        "seed_terminal_bindings": terminal_bindings,
    }
def _load_actor(
    campaign: Path,
    reports: Mapping[int, Mapping[str, Any]],
    seed: int,
    update: int,
) -> LoadedRun3ActorV1:
    if seed not in RUN3_REGISTERED_CONFIG.seeds or update not in AUDIT_UPDATES:
        raise ControlledAuditError("actor identity is outside registered audit scope")
    path = campaign / f"seed_{seed}" / "checkpoints" / f"full_{update:06d}.pt"
    file_sha = _sha256_file(path)
    relative = f"checkpoints/full_{update:06d}.pt"
    if reports[seed].get("artifact_hashes", {}).get(relative) != file_sha:
        raise ControlledAuditError("checkpoint file hash differs from final report")
    checkpoint = load_run3_checkpoint(path)
    if (
        checkpoint.seed != seed
        or checkpoint.update_count != update
        or checkpoint.config != RUN3_REGISTERED_CONFIG
    ):
        raise ControlledAuditError("Run-3 checkpoint identity drift")
    actor = build_actor(
        HybridSacModelConfig(
            dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
        ),
        seed=0,
    )
    actor.load_state_dict(checkpoint.actor_state, strict=True)
    actor.eval()
    actor_hash = _hash_state(actor.state_dict())
    if actor_hash != _hash_state(checkpoint.actor_state):
        raise ControlledAuditError("loaded actor differs from checkpoint")
    return LoadedRun3ActorV1(
        actor=actor,
        seed=seed,
        update_index=update,
        actor_state_sha256=actor_hash,
        checkpoint_content_sha256=checkpoint.checkpoint_sha256,
        checkpoint_file_sha256=file_sha,
    )


def _assert_panel_scope(evaluator: FitValidationActorEvaluatorV1) -> None:
    panel = evaluator.panel
    entries = panel.entries
    if len(entries) != 340 or tuple(row.panel_index for row in entries) != tuple(range(340)):
        raise ControlledAuditError("controlled panel index closure drift")
    scenes: Dict[Tuple[str, str, int], set[str]] = {}
    for row in entries:
        scenes.setdefault(
            (row.scene_sample_id, row.scene_episode_id, row.scene_frame_id), set()
        ).add(row.network_profile)
    if len(scenes) != 85 or any(
        profiles != set(NETWORK_PROFILE_ORDER) for profiles in scenes.values()
    ):
        raise ControlledAuditError("panel is not exactly 85 scenes x four profiles")
    if any(
        sum(row.network_profile == profile for row in entries) != 85
        for profile in NETWORK_PROFILE_ORDER
    ):
        raise ControlledAuditError("panel profile counts are not exactly 85")
    partition = load_registered_empirical_fit_partition()
    train_ids = {
        row.sample_id for row in partition.scene_assignments if row.split == TRAIN_SPLIT
    }
    validation_ids = {
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    }
    panel_ids = {row.scene_sample_id for row in entries}
    train_radio_ids = {
        row.csv_row_number
        for row in partition.radio_assignments
        if row.split == TRAIN_SPLIT
    }
    validation_radio_ids = {
        row.csv_row_number
        for row in partition.radio_assignments
        if row.split == FIT_VALIDATION_SPLIT
    }
    panel_radio_ids = {row.radio_csv_row_number for row in entries}
    if (
        train_ids & validation_ids
        or panel_ids != validation_ids
        or train_radio_ids & validation_radio_ids
        or len(validation_radio_ids) != 80
        or panel_radio_ids != validation_radio_ids
    ):
        raise ControlledAuditError("train/fit-validation identity separation drift")


def _assert_row_closure(rows: Sequence[ControlledAuditRowV1]) -> None:
    expected_profiles = set(NETWORK_PROFILE_ORDER)
    for seed in RUN3_REGISTERED_CONFIG.seeds:
        for update in AUDIT_UPDATES:
            selected = [
                row for row in rows
                if row.seed == seed and row.update_index == update
            ]
            if len(selected) != 340:
                raise ControlledAuditError("seed/checkpoint row count is not 340")
            indexes = [row.panel_index for row in selected]
            if sorted(indexes) != list(range(340)) or len(set(indexes)) != 340:
                raise ControlledAuditError("seed/checkpoint panel identity closure drift")
            profiles = {row.network_profile for row in selected}
            if profiles != expected_profiles or any(
                sum(row.network_profile == profile for row in selected) != 85
                for profile in NETWORK_PROFILE_ORDER
            ):
                raise ControlledAuditError("seed/checkpoint profile closure drift")
            scenes: Dict[Tuple[str, str, int], set[str]] = {}
            for row in selected:
                scenes.setdefault((row.sample_id, row.episode_id, row.frame_id), set()).add(
                    row.network_profile
                )
            if len(scenes) != 85 or any(
                value != expected_profiles for value in scenes.values()
            ):
                raise ControlledAuditError("seed/checkpoint scene-profile closure drift")
            identities = {
                (
                    row.actor_state_sha256,
                    row.checkpoint_content_sha256,
                    row.checkpoint_file_sha256,
                )
                for row in selected
            }
            if len(identities) != 1:
                raise ControlledAuditError("seed/checkpoint actor identity is not singular")


def _validated_regret(oracle_reward: float, action_reward: float) -> float:
    raw = oracle_reward - action_reward
    if raw < -_SCALAR_VECTOR_TOLERANCE:
        raise ControlledAuditError("an action exceeds the exhaustive oracle")
    return 0.0 if raw < 0.0 else raw

def _component_mean(rows: Sequence[ControlledAuditRowV1], name: str) -> Tuple[Optional[float], int]:
    values = []
    for row in rows:
        component: ComponentAuditV1 = getattr(row.learned, name)
        if component.valid and component.value is not None:
            values.append(component.value)
    return (None if not values else fmean(values), len(values))


def aggregate_controlled_rows(
    rows: Sequence[ControlledAuditRowV1],
    *,
    require_registered_coverage: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    """Aggregate checkpoints while preserving valid denominators per metric."""

    if require_registered_coverage:
        _assert_row_closure(rows)
    output = []
    profiles = (
        tuple(NETWORK_PROFILE_ORDER) if require_registered_coverage
        else tuple(sorted({row.network_profile for row in rows}))
    )
    identities = sorted({(row.seed, row.update_index) for row in rows})
    for seed, update in identities:
        actor_rows = [row for row in rows if row.seed == seed and row.update_index == update]
        for profile in (*profiles, "ALL_PROFILES"):
            selected = actor_rows if profile == "ALL_PROFILES" else [
                row for row in actor_rows if row.network_profile == profile
            ]
            if not selected:
                raise ControlledAuditError("aggregate group is empty")
            record: Dict[str, Any] = {
                "seed": seed,
                "update_index": update,
                "network_profile": profile,
                "row_count": len(selected),
                "expected_reward_mean": fmean(row.learned.expected_reward for row in selected),
                "oracle_expected_reward_mean": fmean(row.oracle.expected_reward for row in selected),
                "fixed_expected_reward_mean": fmean(row.fixed.expected_reward for row in selected),
                "oracle_regret_mean": fmean(row.learned_oracle_regret for row in selected),
                "oracle_regret_max": max(row.learned_oracle_regret for row in selected),
                "learned_minus_fixed_reward_mean": fmean(row.learned_minus_fixed_reward for row in selected),
                "exact_oracle_action_fraction": fmean(
                    float((row.learned.mode_id, row.learned.q_e4) == (row.oracle.mode_id, row.oracle.q_e4))
                    for row in selected
                ),
                "unique_learned_action_count": len({(row.learned.mode_id, row.learned.q_e4) for row in selected}),
                "total_latency_p50_ms_mean": fmean(row.learned.total_latency_p50_ms for row in selected),
                "total_latency_p95_ms_mean": fmean(row.learned.total_latency_p95_ms for row in selected),
                "total_latency_p99_ms_mean": fmean(row.learned.total_latency_p99_ms for row in selected),
                "budget_miss_rate_p50": fmean(float(row.learned.budget_miss_p50) for row in selected),
                "budget_miss_rate_p95": fmean(float(row.learned.budget_miss_p95) for row in selected),
                "budget_miss_rate_p99": fmean(float(row.learned.budget_miss_p99) for row in selected),
                "p_timely_feedback_mean": fmean(row.learned.p_timely_feedback for row in selected),
            }
            for name in _QUALITY_COMPONENTS:
                value, denominator = _component_mean(selected, name)
                record[f"{name}_mean"] = value
                record[f"{name}_valid_count"] = denominator
            output.append(record)
    return tuple(output)


def aggregate_paired_progress(
    rows: Sequence[ControlledAuditRowV1],
    *,
    require_registered_coverage: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    """Matched-context changes across 0->5k, 5k->10k, and 0->10k."""

    if require_registered_coverage:
        _assert_row_closure(rows)
    output = []
    profiles = (
        tuple(NETWORK_PROFILE_ORDER) if require_registered_coverage
        else tuple(sorted({row.network_profile for row in rows}))
    )
    pairs = ((0, 5_000), (5_000, 10_000), (0, 10_000))
    for seed in sorted({row.seed for row in rows}):
        for start, end in pairs:
            for profile in (*profiles, "ALL_PROFILES"):
                before = {
                    row.panel_index: row
                    for row in rows
                    if row.seed == seed and row.update_index == start
                    and (profile == "ALL_PROFILES" or row.network_profile == profile)
                }
                after = {
                    row.panel_index: row
                    for row in rows
                    if row.seed == seed and row.update_index == end
                    and (profile == "ALL_PROFILES" or row.network_profile == profile)
                }
                if not before or before.keys() != after.keys():
                    raise ControlledAuditError("paired checkpoint context identity drift")
                reward_deltas = [
                    after[key].learned.expected_reward - before[key].learned.expected_reward
                    for key in before
                ]
                regret_reductions = [
                    before[key].learned_oracle_regret - after[key].learned_oracle_regret
                    for key in before
                ]
                output.append({
                    "seed": seed,
                    "start_update": start,
                    "end_update": end,
                    "network_profile": profile,
                    "matched_context_count": len(before),
                    "expected_reward_delta_mean": fmean(reward_deltas),
                    "oracle_regret_reduction_mean": fmean(regret_reductions),
                    "improved_context_count": sum(value > 0.0 for value in reward_deltas),
                    "unchanged_context_count": sum(value == 0.0 for value in reward_deltas),
                    "worsened_context_count": sum(value < 0.0 for value in reward_deltas),
                })
    return tuple(output)


def aggregate_equal_seed_summary(
    rows: Sequence[ControlledAuditRowV1],
) -> Tuple[Dict[str, Any], ...]:
    """Macro-average per-seed results; each registered seed has equal weight."""

    per_seed = aggregate_controlled_rows(rows)
    output = []
    for update in AUDIT_UPDATES:
        for profile in (*NETWORK_PROFILE_ORDER, "ALL_PROFILES"):
            selected = [
                row for row in per_seed
                if row["update_index"] == update
                and row["network_profile"] == profile
            ]
            if len(selected) != len(RUN3_REGISTERED_CONFIG.seeds):
                raise ControlledAuditError("equal-seed summary seed closure drift")
            record: Dict[str, Any] = {
                "update_index": update,
                "network_profile": profile,
                "seed_count": len(selected),
                "contexts_per_seed": 340 if profile == "ALL_PROFILES" else 85,
                "unique_scenes_per_seed": 85,
                "scene_profile_estimand": (
                    "85_UNIQUE_SCENES_X_4_PROFILES"
                    if profile == "ALL_PROFILES" else "85_UNIQUE_SCENES"
                ),
            }
            for name in (
                "expected_reward_mean",
                "oracle_expected_reward_mean",
                "fixed_expected_reward_mean",
                "oracle_regret_mean",
                "learned_minus_fixed_reward_mean",
                "total_latency_p50_ms_mean",
                "total_latency_p95_ms_mean",
                "total_latency_p99_ms_mean",
                "budget_miss_rate_p50",
                "budget_miss_rate_p95",
                "budget_miss_rate_p99",
                "p_timely_feedback_mean",
            ):
                record[name] = fmean(float(row[name]) for row in selected)
            for component in _QUALITY_COMPONENTS:
                means = [
                    row[f"{component}_mean"] for row in selected
                    if row[f"{component}_mean"] is not None
                ]
                record[f"{component}_mean"] = (
                    None if not means else fmean(float(value) for value in means)
                )
                record[f"{component}_valid_count_total"] = sum(
                    int(row[f"{component}_valid_count"]) for row in selected
                )
            output.append(record)
    return tuple(output)

def _write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        raise ControlledAuditError(f"refusing to write empty {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ControlledAuditError(f"non-rectangular table {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_controlled_audit(
    *,
    campaign_directory: Path,
    output_directory: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the bounded CPU-only audit and write create-only evidence."""

    campaign = Path(campaign_directory).resolve(strict=True)
    destination = Path(output_directory)
    root = (
        Path(__file__).resolve().parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    reports, campaign_bindings = _verify_campaign(campaign)
    fixed_comparator_binding = _verify_fixed_comparator(root)
    cuda_before = torch.cuda.is_initialized()
    rows: list[ControlledAuditRowV1] = []
    checkpoint_bindings: list[Dict[str, Any]] = []
    oracle_by_panel: Dict[int, ActionAuditV1] = {}
    fixed_by_panel: Dict[int, ActionAuditV1] = {}
    panel_source_bindings_sha256: Optional[str] = None
    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        _assert_panel_scope(evaluator)
        if evaluator.panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
            raise ControlledAuditError("frozen validation panel digest drift")
        panel_source_bindings_sha256 = evaluator.panel.source_bindings_sha256
        for entry in evaluator.panel.entries:
            oracle_by_panel[entry.panel_index] = _oracle_action(evaluator, entry)
            fixed_by_panel[entry.panel_index] = _evaluate_action(
                evaluator, entry, *FIXED_COMPARATOR, requested_q=None
            )
        for seed in RUN3_REGISTERED_CONFIG.seeds:
            for update in AUDIT_UPDATES:
                loaded = _load_actor(campaign, reports, seed, update)
                checkpoint_bindings.append({
                    "seed": seed,
                    "update_index": update,
                    "checkpoint_file_sha256": loaded.checkpoint_file_sha256,
                    "checkpoint_content_sha256": loaded.checkpoint_content_sha256,
                    "actor_state_sha256": loaded.actor_state_sha256,
                })
                for entry in evaluator.panel.entries:
                    observation, context, _radio = evaluator._make_observation(entry)
                    state = torch.tensor([observation.values], dtype=torch.float32)
                    rng_before = torch.random.get_rng_state()
                    with torch.inference_mode():
                        execution = loaded.actor.deterministic_execution(state)
                        repeated = loaded.actor.deterministic_execution(state)
                    if (
                        tuple(execution.mode_index.shape) != (1,)
                        or not torch.equal(rng_before, torch.random.get_rng_state())
                        or any(
                            not torch.equal(
                                getattr(execution, name), getattr(repeated, name)
                            )
                            for name in ("mode_index", "q_e4", "q")
                        )
                    ):
                        raise ControlledAuditError(
                            "deterministic actor batch-1/RNG invariance failed"
                        )
                    learned = _evaluate_action(
                        evaluator,
                        entry,
                        int(execution.mode_index.item()),
                        int(execution.q_e4.item()),
                        requested_q=float(execution.q.item()),
                    )
                    oracle = oracle_by_panel[entry.panel_index]
                    fixed = fixed_by_panel[entry.panel_index]
                    rows.append(ControlledAuditRowV1(
                        seed=seed,
                        update_index=update,
                        actor_state_sha256=loaded.actor_state_sha256,
                        checkpoint_content_sha256=loaded.checkpoint_content_sha256,
                        checkpoint_file_sha256=loaded.checkpoint_file_sha256,
                        panel_index=entry.panel_index,
                        scene_rank=entry.scene_rank,
                        sample_id=context.sample_id,
                        episode_id=context.episode_id,
                        frame_id=context.frame_id,
                        network_profile=entry.network_profile,
                        radio_csv_row_number=entry.radio_csv_row_number,
                        radio_trace_step_index=entry.radio_trace_step_index,
                        camera_si=context.camera_si,
                        radar_p40=context.radar_p40,
                        learned=learned,
                        fixed=fixed,
                        oracle=oracle,
                        learned_oracle_regret=_validated_regret(
                            oracle.expected_reward, learned.expected_reward
                        ),
                        fixed_oracle_regret=_validated_regret(
                            oracle.expected_reward, fixed.expected_reward
                        ),
                        learned_minus_fixed_reward=learned.expected_reward - fixed.expected_reward,
                    ))
    expected_rows = 340 * len(RUN3_REGISTERED_CONFIG.seeds) * len(AUDIT_UPDATES)
    checkpoint_identities = {
        (row["seed"], row["update_index"]) for row in checkpoint_bindings
    }
    if (
        len(rows) != expected_rows
        or len(checkpoint_bindings) != 9
        or len(checkpoint_identities) != 9
        or panel_source_bindings_sha256 is None
    ):
        raise ControlledAuditError("controlled audit row count drift")
    _assert_row_closure(rows)
    if not cuda_before and torch.cuda.is_initialized():
        raise ControlledAuditError("CPU-only audit initialized CUDA")
    aggregate = aggregate_controlled_rows(rows)
    paired = aggregate_paired_progress(rows)
    equal_seed = aggregate_equal_seed_summary(rows)
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "per_context.csv", (row.flat() for row in rows))
    _write_csv(destination / "checkpoint_summary.csv", aggregate)
    _write_csv(destination / "paired_progress.csv", paired)
    _write_csv(destination / "equal_seed_summary.csv", equal_seed)
    output_names = (
        "per_context.csv",
        "checkpoint_summary.csv",
        "paired_progress.csv",
        "equal_seed_summary.csv",
    )
    source_path = Path(__file__).resolve()
    test_path = source_path.with_name(
        "test_empirical_contextual_run3_controlled_audit.py"
    )
    manifest = {
        "audit_class": AUDIT_CLASS,
        "audit_schema": AUDIT_SCHEMA,
        "campaign_bindings": campaign_bindings,
        "campaign_directory": str(campaign),
        "checkpoint_bindings": checkpoint_bindings,
        "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        "fit_validation_panel_source_bindings_sha256": (
            panel_source_bindings_sha256
        ),
        "fixed_comparator": fixed_comparator_binding,
        "oracle_tie_rule": (
            "MAX_EXPECTED_RUN3_REWARD; MAX_P_TIMELY_FEEDBACK; MIN_P50; "
            "MIN_MODE_ID; MIN_Q_E4"
        ),
        "panel_estimand": "85_REWARD_HELD_SCENES_X_4_PROFILES_340_CONTEXTS",
        "panel_radio_estimand": "80_REWARD_HELD_RADIO_ROWS_20_PER_PROFILE",
        "scene_rank_semantics": "CANONICAL_PANEL_ORDER_NOT_ELAPSED_TIME",
        "output_files": {
            name: _sha256_file(destination / name)
            for name in output_names
        },
        "source_code_bindings": {
            source_path.name: _sha256_file(source_path),
            test_path.name: _sha256_file(test_path),
        },
        "estimand_semantics": {
            "reward_and_event_probabilities": "ANALYTIC_RUN3_MODEL_EXPECTATIONS",
            "latency_quantiles": "PAYLOAD_NETWORK_SURROGATE_NOT_MEASURED_RUNTIME",
            "quality": "FROZEN_EMPIRICAL_SURFACE_LOOKUP_OR_INTERPOLATION",
            "realized_sampled_reward_emitted": False,
        },
        "row_count": len(rows),
        "run3_kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        "run3_reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
        "scope_disclosure": AUDIT_SCOPE_DISCLOSURE,
        "seeds": list(RUN3_REGISTERED_CONFIG.seeds),
        "status": "COMPLETE",
        "updates": list(AUDIT_UPDATES),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_controlled_audit(
        campaign_directory=args.campaign_directory,
        output_directory=args.output_directory,
        project_root=args.project_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
