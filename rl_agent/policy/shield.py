"""Shared action evaluator and safety shield for both Track A oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .catalog import Action
from .latency import LatencyProjector
from .types import Observation, QualitySnapshot, SceneObject


UNOBSERVED_ERROR_M = 1_000_000.0


@dataclass(frozen=True)
class ActionEvaluation:
    action: Action
    hard_admitted: bool
    expected_g_m: float
    risk_p95_m: float
    risk_sigma_m: float
    bound_m: float
    expected_task_utility: float
    prb_cost: float
    switch_cost: float
    expected_reward: float
    delivery_probability: float
    out_of_support: bool
    payload_provenance: str
    rate_provenance: str


@dataclass(frozen=True)
class ShieldDecision:
    selected: ActionEvaluation
    evaluations: Sequence[ActionEvaluation]
    raw_safe_action_ids: frozenset[str]
    candidate_action_ids: frozenset[str]
    hard_admitted_action_ids: frozenset[str]
    feasible: bool
    over_budget: bool
    shield_ood: bool
    degraded_tier_used: bool
    observed_vulnerable_count: int
    observed_low_confidence_vulnerable_count: int
    vulnerable_guardrail_applied: bool
    vulnerable_guardrail_unachievable: bool
    vulnerable_guardrail_removed_action_ids: frozenset[str]


def profile_quality(action: Action, reward_config: Mapping[str, object]) -> QualitySnapshot:
    weights = reward_config["task_metric_weights"]
    refs = reward_config["task_metric_references"]
    utility = (
        float(weights["miou"]) * action.miou / float(refs["miou"])
        + float(weights["pedestrian_recall"])
        * action.pedestrian_recall
        / float(refs["pedestrian_recall"])
        + float(weights["vehicle_recall"])
        * action.vehicle_recall
        / float(refs["vehicle_recall"])
    )
    return QualitySnapshot(
        profile_id=action.profile_id or "none",
        miou=action.miou,
        pedestrian_recall=action.pedestrian_recall,
        vehicle_recall=action.vehicle_recall,
        object_recall=action.object_recall,
        normalized_utility=utility,
        base_loc_m=action.base_loc_m,
    )


def _mode(action_id: str) -> str:
    return action_id.split("::", 1)[0] if action_id else "NONE"


class SharedShield:
    def __init__(self, config: Mapping[str, object], latency: LatencyProjector) -> None:
        self.config = config
        self.latency = latency
        self.dt = 1.0 / float(config["clock"]["hz"])
        self.epsilon = float(config["safety"]["epsilon_m"])
        self.pessimism = float(config["safety"]["c1_pessimism_factor"])
        self.ucb_k = float(config["safety"]["ucb_k"])
        self.delta_loc = float(config["safety"]["delta_loc_m"])
        self.capacity_multipliers = [
            float(value) for value in config["safety"]["capacity_sample_multipliers"]
        ]
        self.reward = config["reward"]
        guardrail = config["safety"].get("vulnerable_object_guardrails", {})
        self.vulnerable_guardrail_enabled = bool(guardrail.get("enabled", False))
        self.vulnerable_classes = frozenset(
            str(value).strip().lower() for value in guardrail.get("classes", ())
        )
        self.vulnerable_low_confidence = float(
            guardrail.get("low_confidence_threshold", 0.30)
        )
        self.vulnerable_max_roi_q = float(
            guardrail.get("low_confidence_max_roi_q", 0.0)
        )

    def _guardrail_context(self, observation: Observation) -> tuple[int, int]:
        vulnerable = [
            obj
            for obj in observation.objects
            if str(obj.class_name).strip().lower() in self.vulnerable_classes
        ]
        low_confidence = [
            obj for obj in vulnerable if obj.confidence < self.vulnerable_low_confidence
        ]
        return len(vulnerable), len(low_confidence)

    def _guardrail_admits(
        self,
        item: ActionEvaluation,
        vulnerable_count: int,
        low_confidence_count: int,
    ) -> bool:
        if not self.vulnerable_guardrail_enabled:
            return True
        if vulnerable_count and item.action.mode == "SKIP":
            return False
        if (
            low_confidence_count
            and item.action.mode == "SPLIT"
            and item.action.roi_q > self.vulnerable_max_roi_q + 1e-12
        ):
            return False
        return True

    def _prior_error(
        self,
        obj: SceneObject,
        observation: Observation,
        horizon_s: float,
        risk: bool,
    ) -> float:
        capture_time = observation.map_capture_times.get(obj.track_key)
        quality = observation.map_quality.get(obj.track_key)
        if capture_time is None or quality is None:
            return UNOBSERVED_ERROR_M
        age = max(0.0, observation.timestamp_s + horizon_s - capture_time)
        speed = obj.speed_mps + (1.645 * obj.speed_sigma_mps if risk else 0.0)
        return float(np.hypot(quality.base_loc_m, speed * age))

    def _delivered_error(self, obj: SceneObject, action: Action, latency_s: float, risk: bool) -> float:
        speed = obj.speed_mps + (1.645 * obj.speed_sigma_mps if risk else 0.0)
        return float(np.hypot(action.base_loc_m, speed * latency_s))

    @staticmethod
    def _aggregate(errors: Sequence[float]) -> float:
        return max(errors) if errors else 0.0

    def evaluate(
        self,
        action: Action,
        observation: Observation,
        rung_name: str,
        capture_delay_s: float,
        true_capacity_mbps: Optional[float] = None,
    ) -> ActionEvaluation:
        clairvoyant = true_capacity_mbps is not None
        mask_capacity = (
            float(true_capacity_mbps)
            if clairvoyant
            else self.pessimism * observation.estimated_capacity_mbps
        )
        hard_admitted = action.mode == "SKIP" or action.offered_mbps <= mask_capacity + 1e-12
        supported_payload = action.mode == "SKIP" or 49.0 <= action.payload_kib <= 130.0
        out_of_support = rung_name not in self.latency.surface.rungs or not supported_payload
        effective_rung = (
            rung_name if rung_name in self.latency.surface.rungs else next(iter(self.latency.surface.rungs))
        )
        if action.mode == "SKIP":
            capacities = [float(true_capacity_mbps)] if clairvoyant else [observation.estimated_capacity_mbps]
            latency_p50_s = self.dt
            latency_p95_s = self.dt
            payload_provenance = "none"
            rate_provenance = "none"
        else:
            if clairvoyant:
                capacities = [float(true_capacity_mbps)]
            else:
                capacities = [
                    max(0.1, observation.estimated_capacity_mbps * multiplier)
                    for multiplier in self.capacity_multipliers
                ]
            estimate = self.latency.estimate(action, effective_rung)
            latency_p50_s = estimate.p50_ms / 1000.0
            latency_p95_s = estimate.p95_ms / 1000.0
            payload_provenance = estimate.payload_anchor
            rate_provenance = estimate.rate_provenance

        quality = profile_quality(action, self.reward) if action.mode == "SPLIT" else None
        expected_g_values: List[float] = []
        risk_g_values: List[float] = []
        utility_values: List[float] = []
        prb_values: List[float] = []
        delivered_values: List[float] = []
        for capacity in capacities:
            delivered = action.mode == "SPLIT" and action.offered_mbps <= capacity + 1e-12
            delivered_values.append(float(delivered))
            if action.mode == "SKIP":
                horizon_p50 = horizon_p95 = self.dt
            else:
                horizon_p50 = capture_delay_s + latency_p50_s
                horizon_p95 = capture_delay_s + latency_p95_s
            expected_errors = []
            risk_errors = []
            post_utilities = []
            for obj in observation.objects:
                if delivered:
                    delivered_expected = self._delivered_error(obj, action, latency_p50_s, risk=False)
                    delivered_risk = self._delivered_error(obj, action, latency_p95_s, risk=True)
                    if obj.track_key in observation.map_capture_times:
                        prepublish_expected = self._prior_error(
                            obj, observation, horizon_p50, risk=False
                        )
                        prepublish_risk = self._prior_error(obj, observation, horizon_p95, risk=True)
                        expected_errors.append(max(delivered_expected, prepublish_expected))
                        risk_errors.append(max(delivered_risk, prepublish_risk))
                    else:
                        recovery_expected = self._delivered_error(
                            obj, action, capture_delay_s + latency_p50_s, risk=False
                        )
                        recovery_risk = self._delivered_error(
                            obj, action, capture_delay_s + latency_p95_s, risk=True
                        )
                        expected_errors.append(recovery_expected)
                        risk_errors.append(recovery_risk)
                    post_utilities.append(quality.normalized_utility if quality is not None else 0.0)
                else:
                    expected_errors.append(self._prior_error(obj, observation, horizon_p50, risk=False))
                    risk_errors.append(self._prior_error(obj, observation, horizon_p95, risk=True))
                    prior = observation.map_quality.get(obj.track_key)
                    post_utilities.append(prior.normalized_utility if prior is not None else 0.0)
            expected_g_values.append(self._aggregate(expected_errors))
            risk_g_values.append(self._aggregate(risk_errors))
            utility_values.append(float(np.mean(post_utilities)) if post_utilities else 0.0)
            prb_values.append(action.offered_mbps / max(capacity, 1e-6) if action.mode == "SPLIT" else 0.0)

        expected_g = float(np.mean(expected_g_values))
        risk_p95 = float(np.quantile(risk_g_values, 0.95))
        risk_sigma = float(np.std(risk_g_values))
        bound = risk_p95 if clairvoyant else risk_p95 + self.ucb_k * risk_sigma
        expected_task = float(np.mean(utility_values))
        prb_cost = float(np.mean(prb_values))
        switch_cost = float(_mode(observation.previous_action_id) not in {"NONE", action.mode})
        expected_reward = (
            float(self.reward["w_task"]) * expected_task
            - float(self.reward["lambda_prb"]) * prb_cost
            - float(self.reward["lambda_switch"]) * switch_cost
            - float(self.reward["w_error"]) * expected_g / self.epsilon
        )
        return ActionEvaluation(
            action=action,
            hard_admitted=hard_admitted,
            expected_g_m=expected_g,
            risk_p95_m=risk_p95,
            risk_sigma_m=risk_sigma,
            bound_m=bound,
            expected_task_utility=expected_task,
            prb_cost=prb_cost,
            switch_cost=switch_cost,
            expected_reward=expected_reward,
            delivery_probability=float(np.mean(delivered_values)),
            out_of_support=out_of_support,
            payload_provenance=payload_provenance,
            rate_provenance=rate_provenance,
        )

    def decide(
        self,
        actions: Sequence[Action],
        observation: Observation,
        rung_name: str,
        capture_delay: Callable[[Action], float],
        true_capacity_mbps: Optional[float] = None,
    ) -> ShieldDecision:
        evaluations = [
            self.evaluate(action, observation, rung_name, capture_delay(action), true_capacity_mbps)
            for action in actions
        ]
        strict_floor = bool(self.config["actions"]["strict_floor_diagnostic"])
        hard = [
            item
            for item in evaluations
            if item.hard_admitted
            and (not strict_floor or item.action.mode == "SKIP" or item.action.core_tier)
        ]
        hard_ids = frozenset(item.action.action_id for item in hard)
        vulnerable_count, low_confidence_count = self._guardrail_context(observation)
        guarded_hard = [
            item
            for item in hard
            if self._guardrail_admits(item, vulnerable_count, low_confidence_count)
        ]
        removed_ids = frozenset(
            item.action.action_id for item in hard if item not in guarded_hard
        )
        guardrail_applied = bool(removed_ids)
        guardrail_unachievable = bool(hard and not guarded_hard)
        # C1 remains dominant. If the vulnerable-object rule conflicts with the
        # entire C1-admitted set, retain C1's least-risk action and flag the
        # guardrail violation rather than silently sending over capacity.
        active_hard = guarded_hard if guarded_hard else hard
        bounded = [item for item in active_hard if not item.out_of_support]
        shield_ood = not bounded
        if shield_ood:
            fallback = min(
                active_hard,
                key=lambda item: (item.bound_m, -item.expected_reward, item.action.action_id),
            )
            return ShieldDecision(
                selected=fallback,
                evaluations=evaluations,
                raw_safe_action_ids=frozenset(),
                candidate_action_ids=frozenset({fallback.action.action_id}),
                hard_admitted_action_ids=hard_ids,
                feasible=False,
                over_budget=True,
                shield_ood=True,
                degraded_tier_used=not fallback.action.core_tier and fallback.action.mode == "SPLIT",
                observed_vulnerable_count=vulnerable_count,
                observed_low_confidence_vulnerable_count=low_confidence_count,
                vulnerable_guardrail_applied=guardrail_applied,
                vulnerable_guardrail_unachievable=guardrail_unachievable,
                vulnerable_guardrail_removed_action_ids=removed_ids,
            )
        safe = [item for item in bounded if item.bound_m <= self.epsilon]
        raw_safe_ids = frozenset(item.action.action_id for item in safe)
        feasible = bool(safe)
        if feasible:
            core_sends = [item for item in safe if item.action.mode == "SPLIT" and item.action.core_tier]
            skips = [item for item in safe if item.action.mode == "SKIP"]
            if core_sends:
                candidates = core_sends + skips
            else:
                candidates = safe
        else:
            best_bound = min(item.bound_m for item in bounded)
            candidates = [item for item in bounded if item.bound_m <= best_bound + self.delta_loc]
        selected = max(candidates, key=lambda item: (item.expected_reward, -item.bound_m, item.action.action_id))
        return ShieldDecision(
            selected=selected,
            evaluations=evaluations,
            raw_safe_action_ids=raw_safe_ids,
            candidate_action_ids=frozenset(item.action.action_id for item in candidates),
            hard_admitted_action_ids=hard_ids,
            feasible=feasible,
            over_budget=not feasible,
            shield_ood=False,
            degraded_tier_used=selected.action.mode == "SPLIT" and not selected.action.core_tier,
            observed_vulnerable_count=vulnerable_count,
            observed_low_confidence_vulnerable_count=low_confidence_count,
            vulnerable_guardrail_applied=guardrail_applied,
            vulnerable_guardrail_unachievable=guardrail_unachievable,
            vulnerable_guardrail_removed_action_ids=removed_ids,
        )
