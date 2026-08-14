"""Deployable controller ladder for the Track A SPLIT+SKIP surrogate.

Every controller in this module ranks only the candidates produced by the
shared live shield.  The controller API intentionally exposes no environment
or replay-truth handle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import Action
from .shield import ActionEvaluation, SharedShield, ShieldDecision, profile_quality
from .rdo import supported_action_profiles
from .types import Observation, QualitySnapshot


@dataclass(frozen=True)
class ControllerSelection:
    action_id: str
    diagnostics: Mapping[str, object]


class DeployableController:
    """Minimal stateful interface shared by every deployable ladder rung."""

    name = "controller"

    def reset(self, episode_id: str) -> None:
        """Reset episode-local state without discarding fitted parameters."""

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        raise NotImplementedError

    def update(self, observation: Observation, action_id: str, reward: float) -> None:
        """Consume training-only scalar feedback. Stateless controllers ignore it."""

    def state_dict(self) -> Mapping[str, object]:
        return {"controller": self.name}


def _candidate_evaluations(decision: ShieldDecision) -> Dict[str, ActionEvaluation]:
    evaluations = {
        item.action.action_id: item
        for item in decision.evaluations
        if item.action.action_id in decision.candidate_action_ids
    }
    if not evaluations:
        raise RuntimeError("shared shield returned no controller candidates")
    return evaluations


def _risk_first_fallback(candidates: Mapping[str, ActionEvaluation]) -> ActionEvaluation:
    return min(
        candidates.values(),
        key=lambda item: (item.bound_m, item.action.mode != "SKIP", item.action.action_id),
    )


class FixedActionController(DeployableController):
    """Prefer one fixed schedule; use a deterministic risk-first shield fallback."""

    name = "fixed"

    def __init__(self, action_id: str) -> None:
        self.action_id = str(action_id)

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        del observation
        candidates = _candidate_evaluations(decision)
        if self.action_id in candidates:
            selected = candidates[self.action_id]
            fallback = False
        else:
            selected = _risk_first_fallback(candidates)
            fallback = True
        return ControllerSelection(
            selected.action.action_id,
            {"fixed_requested_action_id": self.action_id, "fixed_fallback": fallback},
        )

    def state_dict(self) -> Mapping[str, object]:
        return {"controller": self.name, "action_id": self.action_id}


class RuleController(DeployableController):
    """Explicit AoI/speed/capacity thresholds with no fitted reward model."""

    name = "rule"

    def __init__(self, config: Mapping[str, object], rule_config: Mapping[str, object]) -> None:
        self.epsilon_m = float(config["safety"]["epsilon_m"])
        self.dt = 1.0 / float(config["clock"]["hz"])
        self.default_profile_id = str(rule_config["default_profile_id"])
        self.high_capacity_profile_id = str(rule_config["high_capacity_profile_id"])
        self.high_capacity_mbps = float(rule_config["high_capacity_mbps"])
        self.fresh_skip_fraction = float(rule_config["fresh_skip_fraction"])
        self.urgent_risk_fraction = float(rule_config["urgent_risk_fraction"])
        self.fast_speed_mps = float(rule_config["fast_speed_mps"])
        self.low_fps = int(rule_config["low_fps"])
        self.medium_fps = int(rule_config["medium_fps"])
        self.high_fps = int(rule_config["high_fps"])

    def _prior_risk(self, observation: Observation) -> Tuple[float, float, bool]:
        worst = 0.0
        fastest = 0.0
        unmapped = False
        for obj in observation.objects:
            object_speed = obj.speed_mps + 1.645 * obj.speed_sigma_mps
            fastest = max(fastest, object_speed)
            capture_time = observation.map_capture_times.get(obj.track_key)
            quality = observation.map_quality.get(obj.track_key)
            if capture_time is None or quality is None:
                unmapped = True
                continue
            age = max(0.0, observation.timestamp_s + self.dt - capture_time)
            worst = max(worst, float(np.hypot(quality.base_loc_m, object_speed * age)))
        return worst, fastest, unmapped

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        candidates = _candidate_evaluations(decision)
        skip = candidates.get("SKIP")
        prior_risk, fastest, unmapped = self._prior_risk(observation)
        risk_fraction = prior_risk / self.epsilon_m
        if not observation.objects or (
            not unmapped
            and skip is not None
            and risk_fraction <= self.fresh_skip_fraction
            and skip.bound_m <= self.epsilon_m
        ):
            selected = skip if skip is not None else _risk_first_fallback(candidates)
            reason = "empty_or_fresh"
            desired_fps = 0
            desired_profile = ""
        else:
            desired_fps = (
                self.high_fps
                if unmapped or risk_fraction >= self.urgent_risk_fraction or fastest >= self.fast_speed_mps
                else self.medium_fps if risk_fraction >= self.fresh_skip_fraction else self.low_fps
            )
            desired_profile = (
                self.high_capacity_profile_id
                if observation.estimated_capacity_mbps >= self.high_capacity_mbps
                else self.default_profile_id
            )
            sends = [item for item in candidates.values() if item.action.mode == "SPLIT"]
            exact_profile = [item for item in sends if item.action.profile_id == desired_profile]
            core = [item for item in sends if item.action.core_tier]
            pool = exact_profile or core or sends
            if pool:
                selected = min(
                    pool,
                    key=lambda item: (
                        abs(item.action.target_fps - desired_fps),
                        item.action.target_fps < desired_fps,
                        -item.action.target_fps,
                        item.action.payload_kib,
                        item.action.action_id,
                    ),
                )
                reason = "threshold_send"
            else:
                selected = skip if skip is not None else _risk_first_fallback(candidates)
                reason = "no_safe_send"
        return ControllerSelection(
            selected.action.action_id,
            {
                "rule_reason": reason,
                "rule_prior_risk_fraction": risk_fraction,
                "rule_fastest_mps": fastest,
                "rule_unmapped": unmapped,
                "rule_desired_fps": desired_fps,
                "rule_desired_profile_id": desired_profile,
            },
        )

    def state_dict(self) -> Mapping[str, object]:
        return {
            "controller": self.name,
            "default_profile_id": self.default_profile_id,
            "high_capacity_profile_id": self.high_capacity_profile_id,
            "high_capacity_mbps": self.high_capacity_mbps,
            "fresh_skip_fraction": self.fresh_skip_fraction,
            "urgent_risk_fraction": self.urgent_risk_fraction,
            "fast_speed_mps": self.fast_speed_mps,
            "fps": [self.low_fps, self.medium_fps, self.high_fps],
        }


class GreedyController(DeployableController):
    """The deployable observation-based one-step oracle."""

    name = "greedy"

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        del observation
        return ControllerSelection(
            decision.selected.action.action_id,
            {"greedy_expected_reward": decision.selected.expected_reward},
        )


class BudgetedEnumeratorController(DeployableController):
    """Exact finite argmax over every candidate admitted by the shared shield."""

    name = "budgeted_enumerator"

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        del observation
        return ControllerSelection(
            decision.selected.action.action_id,
            {
                "enumerator_exact_candidate_count": len(decision.candidate_action_ids),
                "enumerator_expected_reward": decision.selected.expected_reward,
            },
        )


class LambdaRDOController(DeployableController):
    """Restrict profiles to max(U-lambda*payload) supported hull points."""

    name = "lambda_rdo"

    def __init__(self, actions: Sequence[Action], reward_config: Mapping[str, object]) -> None:
        self.supported_profiles = frozenset(
            supported_action_profiles(actions, reward_config)
        )

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        del observation
        candidates = _candidate_evaluations(decision)
        supported = {
            action_id: item
            for action_id, item in candidates.items()
            if item.action.mode == "SKIP" or item.action.profile_id in self.supported_profiles
        }
        if not supported:
            selected = _risk_first_fallback(candidates)
            fallback = True
        else:
            selected = max(
                supported.values(),
                key=lambda item: (item.expected_reward, -item.bound_m, item.action.action_id),
            )
            fallback = False
        return ControllerSelection(
            selected.action.action_id,
            {
                "lambda_rdo_supported_profile_count": len(self.supported_profiles),
                "lambda_rdo_supported_profiles": "|".join(sorted(self.supported_profiles)),
                "lambda_rdo_fallback": fallback,
                "lambda_rdo_full_enumerator_action_id": decision.selected.action.action_id,
                "lambda_rdo_full_enumerator_reward_gap": (
                    decision.selected.expected_reward - selected.expected_reward
                ),
            },
        )

    def state_dict(self) -> Mapping[str, object]:
        return {
            "controller": self.name,
            "algorithm": "measured_profile_supported_hull_lookup",
            "objective": "max(profile_utility - lambda * payload_kib), lambda >= 0",
            "supported_profiles": sorted(self.supported_profiles),
        }


class AoIIndexInspiredController(DeployableController):
    """Freshness-risk reduction per PRB heuristic; not a Whittle index."""

    name = "aoi_index"

    def __init__(self, epsilon_m: float, prb_floor: float, minimum_positive_index: float) -> None:
        self.epsilon_m = float(epsilon_m)
        self.prb_floor = float(prb_floor)
        self.minimum_positive_index = float(minimum_positive_index)

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        del observation
        candidates = _candidate_evaluations(decision)
        skip_any = next(
            (item for item in decision.evaluations if item.action.mode == "SKIP"), None
        )
        skip_candidate = next(
            (item for item in candidates.values() if item.action.mode == "SKIP"), None
        )
        sends = [item for item in candidates.values() if item.action.mode == "SPLIT"]
        if skip_any is None or not sends:
            selected = skip_candidate or _risk_first_fallback(candidates)
            return ControllerSelection(
                selected.action.action_id,
                {"aoi_index_value": 0.0, "aoi_index_reason": "no_send_comparison"},
            )
        capped_skip = min(skip_any.bound_m, 2.0 * self.epsilon_m)
        scored = []
        for item in sends:
            capped_send = min(item.bound_m, 2.0 * self.epsilon_m)
            normalized_gain = max(0.0, capped_skip - capped_send) / self.epsilon_m
            index = normalized_gain / max(item.prb_cost, self.prb_floor)
            scored.append((index, item.expected_task_utility, -item.prb_cost, item.action.action_id, item))
        best = max(scored, key=lambda value: value[:4])
        if skip_candidate is not None and best[0] <= self.minimum_positive_index:
            selected = skip_candidate
            reason = "no_positive_freshness_value"
        else:
            selected = best[4]
            reason = "freshness_risk_reduction_per_prb"
        return ControllerSelection(
            selected.action.action_id,
            {
                "aoi_index_value": best[0],
                "aoi_index_reference_skip_bound_m": skip_any.bound_m,
                "aoi_index_reason": reason,
                "aoi_index_is_whittle": False,
            },
        )

    def state_dict(self) -> Mapping[str, object]:
        return {
            "controller": self.name,
            "algorithm": "aoi_index_inspired_freshness_risk_reduction_per_prb",
            "is_whittle_index": False,
            "epsilon_m": self.epsilon_m,
            "prb_floor": self.prb_floor,
            "minimum_positive_index": self.minimum_positive_index,
        }


FEATURE_NAMES = (
    "bias",
    "capacity",
    "capacity_sigma",
    "object_count",
    "max_speed",
    "max_speed_sigma",
    "unmapped_fraction",
    "max_map_aoi",
    "previous_skip",
    "previous_split",
    "scheduler_credit",
    "inflight_count",
    "next_arrival",
)


def observable_features(observation: Observation) -> np.ndarray:
    """Fixed, bounded phase-1 features; every value comes from ``s_obs``."""

    object_count = len(observation.objects)
    speeds = [obj.speed_mps for obj in observation.objects]
    speed_sigmas = [obj.speed_sigma_mps for obj in observation.objects]
    mapped_ages = [
        max(0.0, observation.timestamp_s - observation.map_capture_times[obj.track_key])
        for obj in observation.objects
        if obj.track_key in observation.map_capture_times
    ]
    unmapped = sum(obj.track_key not in observation.map_capture_times for obj in observation.objects)
    previous_mode = observation.previous_action_id.split("::", 1)[0]
    values = np.array(
        [
            1.0,
            np.clip(observation.estimated_capacity_mbps / 40.0, 0.0, 2.0),
            np.clip(observation.capacity_sigma_mbps / 10.0, 0.0, 2.0),
            np.clip(object_count / 10.0, 0.0, 2.0),
            np.clip(max(speeds, default=0.0) / 20.0, 0.0, 2.0),
            np.clip(max(speed_sigmas, default=0.0) / 5.0, 0.0, 2.0),
            unmapped / object_count if object_count else 0.0,
            np.clip(max(mapped_ages, default=0.0) / 2.0, 0.0, 2.0),
            float(previous_mode == "SKIP"),
            float(previous_mode == "SPLIT"),
            np.clip(observation.scheduler_credit, 0.0, 1.0),
            np.clip(observation.inflight_count / 10.0, 0.0, 2.0),
            np.clip((observation.next_expected_arrival_s or 0.0) / 1.0, 0.0, 2.0),
        ],
        dtype=float,
    )
    return values


class LinUCBController(DeployableController):
    """Disjoint linear contextual bandit, trained only through ``update``."""

    name = "linucb"

    def __init__(
        self,
        action_ids: Sequence[str],
        alpha: float,
        ridge: float,
        reward_clip: Sequence[float],
        seed: int,
    ) -> None:
        self.action_ids = tuple(sorted(str(value) for value in action_ids))
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.reward_clip = (float(reward_clip[0]), float(reward_clip[1]))
        self.seed = int(seed)
        dimension = len(FEATURE_NAMES)
        self.a = {action_id: np.eye(dimension) * self.ridge for action_id in self.action_ids}
        self.b = {action_id: np.zeros(dimension) for action_id in self.action_ids}
        self.counts = {action_id: 0 for action_id in self.action_ids}

    def _score(self, action_id: str, features: np.ndarray) -> Tuple[float, float, float]:
        a_inv_x = np.linalg.solve(self.a[action_id], features)
        mean = float(self.b[action_id] @ a_inv_x)
        bonus = self.alpha * float(np.sqrt(max(0.0, features @ a_inv_x)))
        digest = hashlib.sha256(f"{self.seed}:{action_id}".encode()).digest()
        tie_break = int.from_bytes(digest[:4], "big") / 2**32
        return mean + bonus + 1e-12 * tie_break, mean, bonus

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        candidates = _candidate_evaluations(decision)
        features = observable_features(observation)
        scores = {
            action_id: self._score(action_id, features)
            for action_id in candidates
            if action_id in self.a
        }
        if not scores:
            selected = _risk_first_fallback(candidates)
            return ControllerSelection(selected.action.action_id, {"bandit_unknown_catalog": True})
        action_id = max(scores, key=lambda value: (scores[value][0], value))
        score, mean, bonus = scores[action_id]
        return ControllerSelection(
            action_id,
            {
                "bandit_score": score,
                "bandit_mean": mean,
                "bandit_bonus": bonus,
                "bandit_action_updates": self.counts[action_id],
            },
        )

    def update(self, observation: Observation, action_id: str, reward: float) -> None:
        if action_id not in self.a:
            raise ValueError(f"bandit update references unknown action: {action_id}")
        features = observable_features(observation)
        clipped = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))
        self.a[action_id] += np.outer(features, features)
        self.b[action_id] += clipped * features
        self.counts[action_id] += 1

    def state_dict(self) -> Mapping[str, object]:
        return {
            "controller": self.name,
            "algorithm": "disjoint_linucb",
            "feature_names": list(FEATURE_NAMES),
            "alpha": self.alpha,
            "ridge": self.ridge,
            "reward_clip": list(self.reward_clip),
            "seed": self.seed,
            "actions": {
                action_id: {
                    "updates": self.counts[action_id],
                    "a": self.a[action_id].tolist(),
                    "b": self.b[action_id].tolist(),
                }
                for action_id in self.action_ids
            },
        }


@dataclass(frozen=True)
class _PlannedEvent:
    publish_timestamp_s: float
    capture_timestamp_s: float
    track_keys: Tuple[tuple[str, int], ...]
    quality: QualitySnapshot


@dataclass(frozen=True)
class _PlanNode:
    root_action_id: str
    observation: Observation
    pending: Tuple[_PlannedEvent, ...]
    score: float
    path: Tuple[str, ...]


class MPCController(DeployableController):
    """Short-horizon receding controller over an observable-state projection.

    The planner propagates the configured Markov channel's expected capacity,
    uses the modal rung for latency, holds observed object kinematics constant,
    and tracks only planned (not hidden environment) in-flight contributions.
    Each action at every search depth is admitted by the same ``SharedShield``
    implementation used at the live step.
    """

    name = "mpc"

    def __init__(
        self,
        config: Mapping[str, object],
        actions: Sequence[Action],
        shield: SharedShield,
        mpc_config: Mapping[str, object],
    ) -> None:
        self.config = config
        self.actions = tuple(actions)
        self.actions_by_id = {action.action_id: action for action in actions}
        self.shield = shield
        self.dt = 1.0 / float(config["clock"]["hz"])
        self.horizon_steps = int(mpc_config["horizon_steps"])
        self.discount = float(mpc_config["discount"])
        self.future_branch_width = int(mpc_config["future_branch_width"])
        self.beam_width_per_root = int(mpc_config["beam_width_per_root"])
        self.delivery_threshold = float(mpc_config["delivery_probability_threshold"])
        self.forecast_name = str(mpc_config["forecast"])
        self.existing_inflight_policy = str(mpc_config["existing_inflight_policy"])
        self.transition_matrix = config["channel"]["transition_matrix"]
        self.rungs = config["channel"]["rungs"]

    def _capture_delay(self, action: Action, observation: Observation) -> float:
        if action.mode == "SKIP":
            return self.dt
        credit = observation.scheduler_credit if observation.active_schedule_id == action.action_id else 0.0
        increment = action.target_fps * self.dt
        for ticks in range(int(float(self.config["clock"]["hz"])) + 1):
            credit += increment
            if credit >= 1.0 - 1e-12:
                return ticks * self.dt
        raise AssertionError("MPC target FPS did not schedule within one second")

    def _channel_forecast(self, rung: str) -> Tuple[str, float]:
        row = self.transition_matrix.get(rung)
        if row is None:
            return rung, float(self.rungs[rung]["capacity_mbps"])
        modal_rung = max(sorted(row), key=lambda name: float(row[name]))
        expected_capacity = sum(
            float(probability) * float(self.rungs[name]["capacity_mbps"])
            for name, probability in row.items()
        )
        return modal_rung, expected_capacity

    def _advance(
        self,
        observation: Observation,
        action: Action,
        evaluation: ActionEvaluation,
        pending: Tuple[_PlannedEvent, ...],
    ) -> Tuple[Observation, Tuple[_PlannedEvent, ...]]:
        now = observation.timestamp_s
        if action.mode == "SKIP":
            active_schedule_id: Optional[str] = None
            scheduler_credit = 0.0
            captured = False
        else:
            active_schedule_id = action.action_id
            scheduler_credit = (
                observation.scheduler_credit
                if observation.active_schedule_id == action.action_id
                else 0.0
            )
            scheduler_credit += action.target_fps * self.dt
            captured = scheduler_credit >= 1.0 - 1e-12
            if captured:
                scheduler_credit -= 1.0

        planned = list(pending)
        if captured and evaluation.delivery_probability >= self.delivery_threshold:
            latency = self.shield.latency.estimate(action, observation.observed_channel_rung)
            planned.append(
                _PlannedEvent(
                    publish_timestamp_s=now + latency.p50_ms / 1000.0,
                    capture_timestamp_s=now,
                    track_keys=tuple(obj.track_key for obj in observation.objects),
                    quality=profile_quality(action, self.config["reward"]),
                )
            )

        next_time = now + self.dt
        capture_times = dict(observation.map_capture_times)
        qualities = dict(observation.map_quality)
        remaining = []
        for event in sorted(planned, key=lambda item: item.publish_timestamp_s):
            if event.publish_timestamp_s <= next_time + 1e-12:
                for track_key in event.track_keys:
                    if event.capture_timestamp_s > capture_times.get(track_key, -np.inf):
                        capture_times[track_key] = event.capture_timestamp_s
                        qualities[track_key] = event.quality
            else:
                remaining.append(event)

        rung, forecast_capacity = self._channel_forecast(observation.observed_channel_rung)
        sigma_fraction = (
            observation.capacity_sigma_mbps / max(observation.estimated_capacity_mbps, 1e-6)
        )
        unknown_inflight = max(0, observation.inflight_count - len(pending))
        next_unknown_arrival = observation.next_expected_arrival_s
        if next_unknown_arrival is not None:
            next_unknown_arrival -= self.dt
            if next_unknown_arrival <= 1e-12:
                unknown_inflight = max(0, unknown_inflight - 1)
                next_unknown_arrival = None
        ages = [max(0.0, next_time - item.capture_timestamp_s) for item in remaining]
        arrivals = [max(0.0, item.publish_timestamp_s - next_time) for item in remaining]
        return (
            replace(
                observation,
                timestamp_s=next_time,
                estimated_capacity_mbps=forecast_capacity,
                capacity_sigma_mbps=sigma_fraction * forecast_capacity,
                observed_channel_rung=rung,
                previous_action_id=action.action_id,
                scheduler_credit=scheduler_credit,
                active_schedule_id=active_schedule_id,
                inflight_count=unknown_inflight + len(remaining),
                newest_pending_capture_age_s=max(ages) if ages else None,
                next_expected_arrival_s=min(
                    arrivals
                    + ([next_unknown_arrival] if next_unknown_arrival is not None else [])
                )
                if arrivals or next_unknown_arrival is not None
                else None,
                map_capture_times=capture_times,
                map_quality=qualities,
            ),
            tuple(remaining),
        )

    def _future_decision(self, observation: Observation) -> ShieldDecision:
        return self.shield.decide(
            self.actions,
            observation,
            observation.observed_channel_rung,
            lambda action: self._capture_delay(action, observation),
            true_capacity_mbps=None,
        )

    def select(self, observation: Observation, decision: ShieldDecision) -> ControllerSelection:
        root_candidates = _candidate_evaluations(decision)
        nodes = []
        for action_id, evaluation in root_candidates.items():
            next_observation, pending = self._advance(
                observation, evaluation.action, evaluation, tuple()
            )
            nodes.append(
                _PlanNode(
                    root_action_id=action_id,
                    observation=next_observation,
                    pending=pending,
                    score=evaluation.expected_reward,
                    path=(action_id,),
                )
            )

        for depth in range(1, self.horizon_steps):
            expanded = []
            for node in nodes:
                future = self._future_decision(node.observation)
                candidates = sorted(
                    _candidate_evaluations(future).values(),
                    key=lambda item: (item.expected_reward, -item.bound_m, item.action.action_id),
                    reverse=True,
                )[: self.future_branch_width]
                for evaluation in candidates:
                    next_observation, pending = self._advance(
                        node.observation, evaluation.action, evaluation, node.pending
                    )
                    expanded.append(
                        _PlanNode(
                            root_action_id=node.root_action_id,
                            observation=next_observation,
                            pending=pending,
                            score=node.score + self.discount**depth * evaluation.expected_reward,
                            path=node.path + (evaluation.action.action_id,),
                        )
                    )
            kept = []
            for root_action_id in root_candidates:
                group = sorted(
                    (node for node in expanded if node.root_action_id == root_action_id),
                    key=lambda node: (node.score, node.path),
                    reverse=True,
                )
                kept.extend(group[: self.beam_width_per_root])
            nodes = kept

        best = max(nodes, key=lambda node: (node.score, node.path))
        return ControllerSelection(
            best.root_action_id,
            {
                "mpc_planned_return": best.score,
                "mpc_horizon_steps": self.horizon_steps,
                "mpc_forecast": self.forecast_name,
                "mpc_planned_path": "|".join(best.path),
            },
        )

    def state_dict(self) -> Mapping[str, object]:
        return {
            "controller": self.name,
            "horizon_steps": self.horizon_steps,
            "discount": self.discount,
            "future_branch_width": self.future_branch_width,
            "beam_width_per_root": self.beam_width_per_root,
            "delivery_probability_threshold": self.delivery_threshold,
            "forecast": self.forecast_name,
            "existing_inflight_semantics": self.existing_inflight_policy,
        }


def build_controller(
    name: str,
    config: Mapping[str, object],
    actions: Sequence[Action],
    shield: SharedShield,
    ladder_config: Mapping[str, object],
) -> DeployableController:
    spec = ladder_config["controllers"]
    if name == "fixed":
        return FixedActionController(spec["fixed"]["action_id"])
    if name == "rule":
        return RuleController(config, spec["rule"])
    if name == "greedy":
        return GreedyController()
    if name == "budgeted_enumerator":
        return BudgetedEnumeratorController()
    if name == "lambda_rdo":
        return LambdaRDOController(actions, config["reward"])
    if name == "aoi_index":
        values = spec["aoi_index"]
        return AoIIndexInspiredController(
            epsilon_m=float(config["safety"]["epsilon_m"]),
            prb_floor=float(values["prb_floor"]),
            minimum_positive_index=float(values["minimum_positive_index"]),
        )
    if name == "linucb":
        values = spec["linucb"]
        return LinUCBController(
            [action.action_id for action in actions],
            alpha=float(values["alpha"]),
            ridge=float(values["ridge"]),
            reward_clip=values["reward_clip"],
            seed=int(ladder_config["seed"]),
        )
    if name == "mpc":
        return MPCController(config, actions, shield, spec["mpc"])
    raise ValueError(f"unknown deployable controller: {name}")
