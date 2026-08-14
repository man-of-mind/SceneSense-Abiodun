from __future__ import annotations

import copy
import unittest

from rl_agent.policy.catalog import flatten_actions, load_profile_catalog
from rl_agent.policy.channel import ChannelSurface
from rl_agent.policy.config import load_config
from rl_agent.policy.latency import LatencyProjector
from rl_agent.policy.shield import SharedShield, profile_quality
from rl_agent.policy.types import Observation, SceneObject


class VulnerableGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        profiles = load_profile_catalog(cls.config["actions"]["catalog_csv"])
        cls.actions = flatten_actions(
            profiles,
            cls.config["actions"]["fps"],
            cls.config["actions"]["preferred_core_kib"],
        )
        cls.quality = profile_quality(
            next(action for action in cls.actions if action.profile_id == "ae32__uint4__roi0.0"),
            cls.config["reward"],
        )

    def decide(self, *, class_name="pedestrian", confidence=0.8, capacity=37.0):
        obj = SceneObject(
            ("guardrail", 1), class_name, 0.0, 0.0, 8.0, 0.0, 0.0, confidence
        )
        observation = Observation(
            timestamp_s=1.0,
            objects=(obj,),
            estimated_capacity_mbps=capacity,
            capacity_sigma_mbps=0.0,
            observed_channel_rung="clear",
            previous_action_id="SKIP",
            previous_delivery=True,
            previous_latency_ms=0.0,
            scheduler_credit=0.0,
            active_schedule_id=None,
            inflight_count=0,
            newest_pending_capture_age_s=None,
            next_expected_arrival_s=None,
            map_capture_times={obj.track_key: 1.0},
            map_quality={obj.track_key: self.quality},
        )
        surface = ChannelSurface(self.config)
        shield = SharedShield(self.config, LatencyProjector(self.config, surface))
        return shield.decide(self.actions, observation, "clear", lambda action: 0.0)

    def test_observed_pedestrian_removes_skip(self):
        decision = self.decide(confidence=0.8)
        self.assertEqual(decision.observed_vulnerable_count, 1)
        self.assertNotIn("SKIP", decision.candidate_action_ids)
        self.assertEqual(decision.selected.action.mode, "SPLIT")
        self.assertTrue(decision.vulnerable_guardrail_applied)

    def test_low_confidence_vulnerable_object_clamps_roi(self):
        decision = self.decide(class_name="cyclist", confidence=0.20)
        self.assertEqual(decision.observed_low_confidence_vulnerable_count, 1)
        selected = decision.selected.action
        self.assertEqual(selected.mode, "SPLIT")
        self.assertEqual(selected.roi_q, 0.0)
        for item in decision.evaluations:
            if item.action.action_id in decision.raw_safe_action_ids:
                self.assertLessEqual(item.action.roi_q, 0.0)

    def test_c1_conflict_is_flagged_and_never_bypassed(self):
        decision = self.decide(confidence=0.20, capacity=0.1)
        self.assertTrue(decision.vulnerable_guardrail_unachievable)
        self.assertEqual(decision.selected.action.mode, "SKIP")
        self.assertIn(decision.selected.action.action_id, decision.hard_admitted_action_ids)

    def test_disabled_guardrail_leaves_skip_available(self):
        config = copy.deepcopy(self.config)
        config["safety"]["vulnerable_object_guardrails"]["enabled"] = False
        obj = SceneObject(("disabled", 1), "pedestrian", 0.0, 0.0, 8.0, 0.0, 0.0, 0.2)
        observation = Observation(
            1.0,
            (obj,),
            37.0,
            0.0,
            "clear",
            "SKIP",
            True,
            0.0,
            0.0,
            None,
            0,
            None,
            None,
            {obj.track_key: 1.0},
            {obj.track_key: self.quality},
        )
        surface = ChannelSurface(config)
        decision = SharedShield(config, LatencyProjector(config, surface)).decide(
            self.actions, observation, "clear", lambda action: 0.0
        )
        self.assertIn("SKIP", decision.candidate_action_ids)
        self.assertFalse(decision.vulnerable_guardrail_applied)


if __name__ == "__main__":
    unittest.main()
