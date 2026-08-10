"""Fixed-20-Hz event-driven Track A surrogate environment."""

from __future__ import annotations

import heapq
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .catalog import Action
from .channel import ChannelProcess, ChannelSurface
from .latency import LatencyProjector
from .shield import ActionEvaluation, SharedShield, profile_quality
from .types import Contribution, MapObjectState, Observation, PendingFrame, SceneFrame


class SurrogateEnv:
    def __init__(
        self,
        config: Mapping[str, object],
        frames: Sequence[SceneFrame],
        actions: Sequence[Action],
        channel: ChannelProcess,
        surface: ChannelSurface,
        seed: int,
        latency_mode: str = "sample",
    ) -> None:
        if not frames:
            raise ValueError("surrogate environment requires at least one scene frame")
        if latency_mode not in {"sample", "p50", "p95"}:
            raise ValueError("latency_mode must be sample, p50, or p95")
        self.config = config
        self.frames = list(frames)
        self.actions = list(actions)
        self.actions_by_id = {action.action_id: action for action in actions}
        self.channel = channel
        self.surface = surface
        self.latency = LatencyProjector(config, surface)
        self.shield = SharedShield(config, self.latency)
        self.rng = np.random.default_rng(seed)
        self.latency_mode = latency_mode
        self.dt = 1.0 / float(config["clock"]["hz"])
        self.step_index = 0
        self.map_state: Dict[tuple[str, int], MapObjectState] = {}
        self.pending: List[PendingFrame] = []
        self.sequence_id = 0
        self.scheduler_credit = 0.0
        self.active_schedule_id: Optional[str] = None
        self.previous_action_id = ""
        self.previous_delivery: Optional[bool] = None
        self.previous_latency_ms: Optional[float] = None
        self.last_channel_snapshot = self.channel.snapshot()
        self.counters = {
            "attempts": 0,
            "delivered": 0,
            "dropped": 0,
            "out_of_order_ignored": 0,
            "c1_estimate_miss": 0,
        }

    @property
    def done(self) -> bool:
        return self.step_index >= len(self.frames)

    @property
    def frame(self) -> SceneFrame:
        if self.done:
            raise RuntimeError("environment episode is complete")
        return self.frames[self.step_index]

    def _process_events(self, now_s: float) -> None:
        while self.pending and self.pending[0].publish_timestamp_s <= now_s + 1e-12:
            event = heapq.heappop(self.pending)
            self.previous_delivery = event.delivered
            self.previous_latency_ms = (event.publish_timestamp_s - event.capture_timestamp_s) * 1000.0
            if not event.delivered:
                continue
            self.counters["delivered"] += 1
            for track_key in event.captured_track_keys:
                state = self.map_state.setdefault(track_key, MapObjectState(track_key=track_key))
                if not state.install(event.contribution):
                    self.counters["out_of_order_ignored"] += 1

    def _visible_map(self, track_keys: Sequence[tuple[str, int]]) -> tuple[dict, dict]:
        capture_times = {}
        qualities = {}
        for key in track_keys:
            state = self.map_state.get(key)
            newest = state.newest if state is not None else None
            if newest is not None:
                capture_times[key] = newest.capture_timestamp_s
                qualities[key] = newest.quality
        return capture_times, qualities

    def observation(self, truth: bool = False) -> Observation:
        now = self.frame.timestamp_s
        self._process_events(now)
        objects = self.frame.truth_objects if truth else self.frame.observed_objects
        capture_times, qualities = self._visible_map([obj.track_key for obj in objects])
        pending_ages = [max(0.0, now - item.capture_timestamp_s) for item in self.pending]
        arrivals = [max(0.0, item.publish_timestamp_s - now) for item in self.pending]
        if truth:
            estimated_capacity = self.last_channel_snapshot.true_capacity_mbps
            sigma = 0.0
            rung = self.last_channel_snapshot.rung
        else:
            estimated_capacity = self.last_channel_snapshot.estimated_capacity_mbps
            sigma = self.last_channel_snapshot.estimate_sigma_mbps
            rung = self.last_channel_snapshot.observed_rung
        return Observation(
            timestamp_s=now,
            objects=tuple(objects),
            estimated_capacity_mbps=estimated_capacity,
            capacity_sigma_mbps=sigma,
            observed_channel_rung=rung,
            previous_action_id=self.previous_action_id,
            previous_delivery=self.previous_delivery,
            previous_latency_ms=self.previous_latency_ms,
            scheduler_credit=self.scheduler_credit,
            active_schedule_id=self.active_schedule_id,
            inflight_count=len(self.pending),
            newest_pending_capture_age_s=max(pending_ages) if pending_ages else None,
            next_expected_arrival_s=min(arrivals) if arrivals else None,
            map_capture_times=capture_times,
            map_quality=qualities,
        )

    def matched_truth_observation(self) -> Observation:
        """Hidden truth restricted to currently observable tracker keys for C2 attribution."""
        deployable = self.observation(truth=False)
        visible_keys = {obj.track_key for obj in deployable.objects}
        truth_objects = tuple(obj for obj in self.frame.truth_objects if obj.track_key in visible_keys)
        capture_times, qualities = self._visible_map([obj.track_key for obj in truth_objects])
        return Observation(
            timestamp_s=deployable.timestamp_s,
            objects=truth_objects,
            estimated_capacity_mbps=self.last_channel_snapshot.true_capacity_mbps,
            capacity_sigma_mbps=0.0,
            observed_channel_rung=self.last_channel_snapshot.rung,
            previous_action_id=deployable.previous_action_id,
            previous_delivery=deployable.previous_delivery,
            previous_latency_ms=deployable.previous_latency_ms,
            scheduler_credit=deployable.scheduler_credit,
            active_schedule_id=deployable.active_schedule_id,
            inflight_count=deployable.inflight_count,
            newest_pending_capture_age_s=deployable.newest_pending_capture_age_s,
            next_expected_arrival_s=deployable.next_expected_arrival_s,
            map_capture_times=capture_times,
            map_quality=qualities,
        )

    def matched_truth_evaluation(self, action: Action) -> ActionEvaluation:
        observation = self.matched_truth_observation()
        return self.shield.evaluate(
            action,
            observation,
            self.last_channel_snapshot.rung,
            self.time_to_next_capture(action),
            true_capacity_mbps=self.last_channel_snapshot.true_capacity_mbps,
        )

    def time_to_next_capture(self, action: Action) -> float:
        if action.mode == "SKIP":
            return self.dt
        credit = self.scheduler_credit if self.active_schedule_id == action.action_id else 0.0
        increment = action.target_fps * self.dt
        for ticks in range(int(float(self.config["clock"]["hz"])) + 1):
            credit += increment
            if credit >= 1.0 - 1e-12:
                return ticks * self.dt
        raise AssertionError("target FPS did not schedule a capture within one second")

    def shielded_decision(self):
        observation = self.observation(truth=False)
        return self.shield.decide(
            self.actions,
            observation,
            observation.observed_channel_rung,
            self.time_to_next_capture,
            true_capacity_mbps=None,
        )

    def clairvoyant_decision(self):
        observation = self.observation(truth=True)
        return self.shield.decide(
            self.actions,
            observation,
            self.last_channel_snapshot.rung,
            self.time_to_next_capture,
            true_capacity_mbps=self.last_channel_snapshot.true_capacity_mbps,
        )

    def true_evaluation(self, action: Action) -> ActionEvaluation:
        observation = self.observation(truth=True)
        return self.shield.evaluate(
            action,
            observation,
            self.last_channel_snapshot.rung,
            self.time_to_next_capture(action),
            true_capacity_mbps=self.last_channel_snapshot.true_capacity_mbps,
        )

    def _sample_latency_ms(self, p50_ms: float, p95_ms: float) -> float:
        if self.latency_mode == "p50":
            return p50_ms
        if self.latency_mode == "p95":
            return p95_ms
        if p50_ms <= 0 or p95_ms <= p50_ms:
            return max(p50_ms, p95_ms)
        sigma = np.log(p95_ms / p50_ms) / 1.6448536269514722
        return float(self.rng.lognormal(np.log(p50_ms), sigma))

    def _scheduler_attempts_capture(self, action: Action) -> bool:
        if action.mode == "SKIP":
            self.active_schedule_id = None
            self.scheduler_credit = 0.0
            return False
        if self.active_schedule_id != action.action_id:
            self.active_schedule_id = action.action_id
            self.scheduler_credit = 0.0
        self.scheduler_credit += action.target_fps * self.dt
        if self.scheduler_credit >= 1.0 - 1e-12:
            self.scheduler_credit -= 1.0
            return True
        return False

    def step(self, action: Action) -> Dict[str, object]:
        if self.done:
            raise RuntimeError("cannot step a completed episode")
        now = self.frame.timestamp_s
        self._process_events(now)
        true_eval = self.true_evaluation(action)
        captured = self._scheduler_attempts_capture(action)
        actual_delivery: Optional[bool] = None
        actual_latency_ms: Optional[float] = None
        latency_info = None
        if captured:
            self.counters["attempts"] += 1
            latency_info = self.latency.estimate(action, self.last_channel_snapshot.rung)
            actual_delivery = action.offered_mbps <= self.last_channel_snapshot.true_capacity_mbps + 1e-12
            if not actual_delivery:
                self.counters["dropped"] += 1
                if action.offered_mbps <= (
                    float(self.config["safety"]["c1_pessimism_factor"])
                    * self.last_channel_snapshot.estimated_capacity_mbps
                ):
                    self.counters["c1_estimate_miss"] += 1
            actual_latency_ms = self._sample_latency_ms(latency_info.p50_ms, latency_info.p95_ms)
            quality = profile_quality(action, self.config["reward"])
            contribution = Contribution(
                source_ue_id="phase1_ue",
                capture_timestamp_s=now,
                publish_timestamp_s=now + actual_latency_ms / 1000.0,
                confidence=1.0,
                profile_id=action.profile_id or "none",
                quality=quality,
            )
            event = PendingFrame(
                publish_timestamp_s=contribution.publish_timestamp_s,
                sequence_id=self.sequence_id,
                capture_timestamp_s=now,
                # A deployable contribution may update only objects present in
                # the prediction/tracker observation at capture time. Hidden GT
                # is retained solely for evaluation and must not seed map state.
                captured_track_keys=tuple(obj.track_key for obj in self.frame.observed_objects),
                contribution=contribution,
                action_id=action.action_id,
                delivered=bool(actual_delivery),
            )
            self.sequence_id += 1
            heapq.heappush(self.pending, event)
        c1_estimate_miss = bool(
            captured
            and actual_delivery is False
            and action.offered_mbps
            <= float(self.config["safety"]["c1_pessimism_factor"])
            * self.last_channel_snapshot.estimated_capacity_mbps
            + 1e-12
        )
        self.previous_action_id = action.action_id
        row = {
            "episode_id": self.frame.episode_id,
            "step_index": self.step_index,
            "timestamp_s": now,
            "action_id": action.action_id,
            "mode": action.mode,
            "profile_id": action.profile_id or "",
            "target_fps": action.target_fps,
            "channel_rung_true": self.last_channel_snapshot.rung,
            "channel_rung_observed": self.last_channel_snapshot.observed_rung,
            "true_capacity_mbps": self.last_channel_snapshot.true_capacity_mbps,
            "estimated_capacity_mbps": self.last_channel_snapshot.estimated_capacity_mbps,
            "offered_mbps": action.offered_mbps,
            "captured": captured,
            "actual_delivery": actual_delivery,
            "actual_latency_ms": actual_latency_ms,
            "c1_estimate_miss": c1_estimate_miss,
            "pending_count": len(self.pending),
            "scheduler_credit": self.scheduler_credit,
            "true_expected_g_m": true_eval.expected_g_m,
            "true_risk_p95_m": true_eval.risk_p95_m,
            "true_expected_reward": true_eval.expected_reward,
            "true_safe": true_eval.risk_p95_m <= float(self.config["safety"]["epsilon_m"]),
            "payload_provenance": latency_info.payload_anchor if latency_info else "none",
            "rate_provenance": latency_info.rate_provenance if latency_info else "none",
            "truth_object_count": len(self.frame.truth_objects),
            "observed_object_count": len(self.frame.observed_objects),
        }
        self.channel.advance()
        self.step_index += 1
        if not self.done:
            self.last_channel_snapshot = self.channel.snapshot()
        return row
