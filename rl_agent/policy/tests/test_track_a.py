from __future__ import annotations

import copy
import hashlib
import json
import unittest

from rl_agent.policy.catalog import Action, flatten_actions, load_profile_catalog
from rl_agent.policy.channel import ChannelProcess, ChannelSurface
from rl_agent.policy.config import REPO_ROOT, load_config
from rl_agent.policy.env import SurrogateEnv
from rl_agent.policy.latency import LatencyProjector
from rl_agent.policy.replay import discover_trace_registry, synthetic_episode
from rl_agent.policy.shield import SharedShield, UNOBSERVED_ERROR_M, profile_quality
from rl_agent.policy.types import Contribution, MapObjectState, Observation, SceneFrame, SceneObject


class TrackATestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.profiles = load_profile_catalog(cls.config["actions"]["catalog_csv"])
        cls.actions = flatten_actions(
            cls.profiles, cls.config["actions"]["fps"], cls.config["actions"]["preferred_core_kib"]
        )
        cls.surface = ChannelSurface(cls.config)
        cls.latency = LatencyProjector(cls.config, cls.surface)

    def make_env(
        self,
        frames,
        seed=1,
        rung="clear",
        multiplier=1.0,
        config=None,
        latency_mode="p50",
        latency_crn_by_tick=False,
    ):
        config = config or self.config
        actions = flatten_actions(
            self.profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"]
        )
        channel = ChannelProcess(
            config,
            self.surface,
            seed,
            fixed_rungs=[rung] * len(frames),
            fixed_capacity_multiplier=multiplier,
        )
        return SurrogateEnv(
            config,
            frames,
            actions,
            channel,
            self.surface,
            seed + 1,
            latency_mode,
            latency_crn_by_tick=latency_crn_by_tick,
        )

    def test_catalog_hash_and_action_count(self):
        self.assertEqual(len(self.profiles), 7)
        self.assertEqual(len(self.actions), 36)
        meta_path = REPO_ROOT / "rl_agent/policy/data/action_catalog.meta.json"
        metadata = json.loads(meta_path.read_text())
        catalog_path = REPO_ROOT / metadata["catalog_file"]
        self.assertEqual(hashlib.sha256(catalog_path.read_bytes()).hexdigest(), metadata["catalog_sha256"])
        source_path = REPO_ROOT / metadata["source_file"]
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), metadata["source_sha256"])

    def test_latency_90k_measured_anchor_and_tail_reconstruction(self):
        action = next(
            item
            for item in self.actions
            if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 10
        )
        expected_p50 = {"clear": 94.0, "mild": 130.0, "mid": 146.0, "strong": 175.0}
        for rung, p50 in expected_p50.items():
            estimate = self.latency.estimate(action, rung)
            self.assertAlmostEqual(estimate.p50_ms, p50, places=6)
            self.assertGreater(estimate.p95_ms, estimate.p50_ms)
            self.assertEqual(estimate.payload_anchor, "measured_90k_anchor")
            self.assertEqual(estimate.rate_provenance, "fps_projection")

    def test_replay_split_is_grouped_and_paired_test_exists(self):
        registry = discover_trace_registry(self.config)
        family_splits = {}
        for record in registry:
            family_splits.setdefault(record.scenario_family, set()).add(record.split)
        self.assertTrue(all(len(value) == 1 for value in family_splits.values()))
        self.assertTrue(any(record.split == "test" and record.prediction_path for record in registry))
        self.assertTrue(all(record.ground_truth_path.stat().st_size > 0 for record in registry))

    def test_channel_telemetry_lag_is_exactly_two_steps(self):
        config = copy.deepcopy(self.config)
        config["channel"]["telemetry_lag_steps"] = 2
        config["channel"]["estimate_noise_fraction"] = 0.0
        channel = ChannelProcess(
            config,
            self.surface,
            seed=7,
            fixed_rungs=["clear", "mild", "mid", "strong"],
            fixed_capacity_multiplier=1.0,
        )
        snapshots = []
        for _ in range(4):
            snapshots.append(channel.snapshot())
            channel.advance()
        self.assertEqual([item.observed_rung for item in snapshots], ["clear", "clear", "clear", "mild"])
        self.assertEqual(snapshots[3].estimated_capacity_mbps, 28.0)

    def test_10fps_scheduler_captures_every_other_20hz_tick(self):
        frames = synthetic_episode("scheduler", [2.0], 6)
        env = self.make_env(frames)
        action = next(
            item
            for item in env.actions
            if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 10
        )
        captures = [bool(env.step(action)["captured"]) for _ in range(6)]
        self.assertEqual(captures, [False, True, False, True, False, True])

    def test_action_change_resets_fractional_scheduler_credit(self):
        frames = synthetic_episode("reset", [2.0], 4)
        env = self.make_env(frames)
        action_5 = next(item for item in env.actions if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 5)
        action_10 = next(item for item in env.actions if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 10)
        self.assertFalse(env.step(action_5)["captured"])
        self.assertFalse(env.step(action_10)["captured"])
        self.assertTrue(env.step(action_10)["captured"])

    def test_newer_capture_wins(self):
        action = next(item for item in self.actions if item.profile_id == "ae32__uint4__roi0.0")
        quality = profile_quality(action, self.config["reward"])
        state = MapObjectState(("episode", 1))
        newer = Contribution("ue", 2.0, 2.1, 1.0, action.profile_id, quality)
        older = Contribution("ue", 1.0, 3.0, 1.0, action.profile_id, quality)
        self.assertTrue(state.install(newer))
        self.assertFalse(state.install(older))
        self.assertEqual(state.newest.capture_timestamp_s, 2.0)

    def test_hidden_truth_does_not_seed_map_contributions(self):
        obj = SceneObject(("hidden", 1), "vehicle", 0.0, 0.0, 10.0, 2.0, 0.0, 1.0)
        frames = [
            SceneFrame("hidden", 0, 0.00, (obj,), ()),
            SceneFrame("hidden", 1, 0.05, (obj,), ()),
            SceneFrame("hidden", 2, 0.10, (obj,), (obj,)),
        ]
        env = self.make_env(frames, latency_mode="p50")
        action = next(
            item
            for item in env.actions
            if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 20
        )
        self.assertTrue(env.step(action)["captured"])
        env.step(next(item for item in env.actions if item.mode == "SKIP"))
        observation = env.observation()
        self.assertNotIn(obj.track_key, observation.map_capture_times)

    def test_empty_scene_selects_free_skip(self):
        env = self.make_env(synthetic_episode("empty", [], 2))
        decision = env.shielded_decision()
        self.assertEqual(decision.selected.action.mode, "SKIP")
        self.assertEqual(decision.selected.expected_reward, 0.0)

    def test_raw_safe_set_is_not_preference_narrowed(self):
        env = self.make_env(synthetic_episode("empty_sets", [], 2))
        decision = env.shielded_decision()
        self.assertTrue(decision.candidate_action_ids.issubset(decision.raw_safe_action_ids))
        self.assertGreater(len(decision.raw_safe_action_ids), len(decision.candidate_action_ids))
        self.assertTrue(
            any(
                action_id.startswith("SPLIT::ae32__uint4__roi0.5")
                for action_id in decision.raw_safe_action_ids
            )
        )

    def test_safety_knob_sets_are_monotone_for_fixed_observation(self):
        frames = synthetic_episode("monotone", [3.0], 3)
        base_env = self.make_env(frames, multiplier=0.85)
        observation = base_env.observation()

        low_k_config = copy.deepcopy(self.config)
        low_k_config["safety"]["ucb_k"] = 0.0
        high_k_config = copy.deepcopy(self.config)
        high_k_config["safety"]["ucb_k"] = 2.0
        low_k = SharedShield(low_k_config, self.latency).decide(
            self.actions, observation, "clear", base_env.time_to_next_capture
        )
        high_k = SharedShield(high_k_config, self.latency).decide(
            self.actions, observation, "clear", base_env.time_to_next_capture
        )
        self.assertTrue(high_k.raw_safe_action_ids.issubset(low_k.raw_safe_action_ids))

        low_c1_config = copy.deepcopy(self.config)
        low_c1_config["safety"]["c1_pessimism_factor"] = 0.6
        high_c1_config = copy.deepcopy(self.config)
        high_c1_config["safety"]["c1_pessimism_factor"] = 1.0
        low_c1 = SharedShield(low_c1_config, self.latency).decide(
            self.actions, observation, "clear", base_env.time_to_next_capture
        )
        high_c1 = SharedShield(high_c1_config, self.latency).decide(
            self.actions, observation, "clear", base_env.time_to_next_capture
        )
        self.assertTrue(low_c1.hard_admitted_action_ids.issubset(high_c1.hard_admitted_action_ids))

    def test_per_tick_latency_crn_is_action_history_independent(self):
        frames = synthetic_episode("crn", [2.0], 3)
        env_every_tick = self.make_env(
            frames, seed=31, latency_mode="sample", latency_crn_by_tick=True
        )
        env_second_tick = self.make_env(
            frames, seed=31, latency_mode="sample", latency_crn_by_tick=True
        )
        action = next(
            item
            for item in env_every_tick.actions
            if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 20
        )
        skip = next(item for item in env_every_tick.actions if item.mode == "SKIP")
        env_every_tick.step(action)
        second_a = env_every_tick.step(action)
        env_second_tick.step(skip)
        second_b = env_second_tick.step(action)
        self.assertTrue(second_a["captured"] and second_b["captured"])
        self.assertAlmostEqual(second_a["actual_latency_ms"], second_b["actual_latency_ms"], places=12)

    def test_new_unobserved_object_makes_skip_unsafe(self):
        env = self.make_env(synthetic_episode("new", [4.0], 2))
        decision = env.shielded_decision()
        skip = next(item for item in decision.evaluations if item.action.mode == "SKIP")
        self.assertGreaterEqual(skip.bound_m, UNOBSERVED_ERROR_M)
        self.assertEqual(decision.selected.action.mode, "SPLIT")

    def test_drop_retains_prior_map_quality(self):
        obj = SceneObject(("drop", 1), "vehicle", 0.0, 0.0, 10.0, 3.0, 0.2, 1.0)
        prior_action = next(item for item in self.actions if item.profile_id == "ae32__uint4__roi0.0")
        prior_quality = profile_quality(prior_action, self.config["reward"])
        observation = Observation(
            timestamp_s=1.0,
            objects=(obj,),
            estimated_capacity_mbps=5.0,
            capacity_sigma_mbps=0.0,
            observed_channel_rung="strong",
            previous_action_id=prior_action.action_id,
            previous_delivery=True,
            previous_latency_ms=100.0,
            scheduler_credit=0.0,
            active_schedule_id=None,
            inflight_count=0,
            newest_pending_capture_age_s=None,
            next_expected_arrival_s=None,
            map_capture_times={obj.track_key: 0.9},
            map_quality={obj.track_key: prior_quality},
        )
        overload = next(item for item in self.actions if item.payload_kib > 128 and item.target_fps == 20)
        evaluation = SharedShield(self.config, self.latency).evaluate(
            overload, observation, "strong", 0.0, true_capacity_mbps=5.0
        )
        self.assertEqual(evaluation.delivery_probability, 0.0)
        self.assertAlmostEqual(evaluation.expected_task_utility, prior_quality.normalized_utility)

    def test_c1_estimate_miss_is_logged(self):
        env = self.make_env(synthetic_episode("miss", [2.0], 2), multiplier=0.30)
        action = next(item for item in env.actions if item.payload_kib > 128 and item.target_fps == 20)
        row = env.step(action)
        self.assertTrue(row["captured"])
        self.assertFalse(row["actual_delivery"])
        self.assertEqual(env.counters["c1_estimate_miss"], 1)

    def test_infeasible_state_uses_flagged_graceful_degradation(self):
        env = self.make_env(synthetic_episode("fast", [20.0], 3), rung="strong")
        decision = env.shielded_decision()
        self.assertFalse(decision.feasible)
        self.assertTrue(decision.over_budget)
        self.assertIn(decision.selected.action.action_id, decision.hard_admitted_action_ids)

    def test_unknown_rung_enters_ood_fallback(self):
        env = self.make_env(synthetic_episode("ood", [], 2))
        obs = env.observation()
        decision = env.shield.decide(env.actions, obs, "unknown", env.time_to_next_capture)
        self.assertTrue(decision.shield_ood)
        self.assertTrue(decision.over_budget)

    def test_129_preferred_core_marks_90_as_degraded(self):
        config = copy.deepcopy(self.config)
        config["actions"]["preferred_core_kib"] = 129
        actions = flatten_actions(self.profiles, config["actions"]["fps"], 129)
        profile_90 = [item for item in actions if item.profile_id == "ae32__uint4__roi0.0"]
        profile_129 = [item for item in actions if item.profile_id == "ae128__uint4__roi0.0"]
        self.assertTrue(all(not item.core_tier for item in profile_90))
        self.assertTrue(all(item.core_tier for item in profile_129))

    def test_strict_floor_diagnostic_never_selects_degraded_profile(self):
        config = copy.deepcopy(self.config)
        config["actions"]["preferred_core_kib"] = 129
        config["actions"]["strict_floor_diagnostic"] = True
        frames = synthetic_episode("strict", [20.0], 3)
        env = self.make_env(frames, rung="strong", config=config)
        decision = env.shielded_decision()
        self.assertTrue(decision.selected.action.mode == "SKIP" or decision.selected.action.core_tier)

    def test_tail_risk_includes_wait_for_next_scheduled_capture(self):
        obj = SceneObject(("fps", 1), "vehicle", 0.0, 0.0, 10.0, 10.0, 0.0, 1.0)
        prior_action = next(item for item in self.actions if item.profile_id == "ae32__uint4__roi0.0")
        quality = profile_quality(prior_action, self.config["reward"])
        observation = Observation(
            timestamp_s=0.10,
            objects=(obj,),
            estimated_capacity_mbps=37.0,
            capacity_sigma_mbps=0.0,
            observed_channel_rung="clear",
            previous_action_id=prior_action.action_id,
            previous_delivery=True,
            previous_latency_ms=94.0,
            scheduler_credit=0.0,
            active_schedule_id=None,
            inflight_count=0,
            newest_pending_capture_age_s=None,
            next_expected_arrival_s=None,
            map_capture_times={obj.track_key: 0.0},
            map_quality={obj.track_key: quality},
        )
        action_2 = next(
            item for item in self.actions if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 2
        )
        action_20 = next(
            item for item in self.actions if item.profile_id == "ae32__uint4__roi0.0" and item.target_fps == 20
        )
        shield = SharedShield(self.config, self.latency)
        slow = shield.evaluate(action_2, observation, "clear", 0.45, true_capacity_mbps=37.0)
        fast = shield.evaluate(action_20, observation, "clear", 0.0, true_capacity_mbps=37.0)
        self.assertGreater(slow.risk_p95_m, fast.risk_p95_m)


if __name__ == "__main__":
    unittest.main()
