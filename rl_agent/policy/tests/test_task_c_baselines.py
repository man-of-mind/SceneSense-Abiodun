from __future__ import annotations

import unittest

from rl_agent.policy.catalog import flatten_actions, load_profile_catalog
from rl_agent.policy.channel import ChannelProcess, ChannelSurface
from rl_agent.policy.config import load_controller_ladder_config
from rl_agent.policy.controllers import (
    AoIIndexInspiredController,
    BudgetedEnumeratorController,
    LambdaRDOController,
)
from rl_agent.policy.env import SurrogateEnv
from rl_agent.policy.rdo import lagrangian_dual_bound, supported_upper_hull
from rl_agent.policy.replay import synthetic_episode


class TaskCBaselineTests(unittest.TestCase):
    def test_hull_excludes_non_supported_pareto_point_and_dual_reports_gap(self):
        points = [("a", 1.0, 1.0), ("b", 2.0, 1.4), ("c", 3.0, 2.0)]
        self.assertEqual(supported_upper_hull(points), ("a", "c"))
        dual, multiplier = lagrangian_dual_bound(points, budget=2.0)
        self.assertAlmostEqual(dual, 1.5)
        self.assertAlmostEqual(multiplier, 0.5)
        self.assertAlmostEqual(dual - 1.4, 0.1)

    def test_task_c_controllers_never_bypass_shield(self):
        config = load_controller_ladder_config(
            "rl_agent/policy/configs/controller_ladder_task_c_v1.yaml"
        )
        profiles = load_profile_catalog(config["actions"]["catalog_csv"])
        actions = flatten_actions(
            profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"]
        )
        surface = ChannelSurface(config)
        env = SurrogateEnv(
            config,
            synthetic_episode("task_c", [4.0], 3),
            actions,
            ChannelProcess(config, surface, 11, fixed_rungs=["clear"] * 3),
            surface,
            12,
            latency_mode="p50",
        )
        observation = env.observation()
        decision = env.shielded_decision()
        enumerator = BudgetedEnumeratorController()
        exact = enumerator.select(observation, decision)
        self.assertEqual(exact.action_id, decision.selected.action.action_id)

        rdo = LambdaRDOController(actions, config["reward"])
        rdo_selection = rdo.select(observation, decision)
        self.assertIn(rdo_selection.action_id, decision.candidate_action_ids)
        self.assertIn("lambda_rdo_full_enumerator_reward_gap", rdo_selection.diagnostics)

        aoi = AoIIndexInspiredController(2.0, 0.01, 0.0)
        aoi_selection = aoi.select(observation, decision)
        self.assertIn(aoi_selection.action_id, decision.candidate_action_ids)
        self.assertFalse(aoi_selection.diagnostics["aoi_index_is_whittle"])


if __name__ == "__main__":
    unittest.main()
