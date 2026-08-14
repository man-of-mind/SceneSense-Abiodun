from __future__ import annotations

import copy
import unittest

import pandas as pd

from rl_agent.multiue_oai.endpoint import frame_onwire_bytes
from rl_agent.policy.catalog import Action
from rl_agent.policy.expanded_gate import (
    CONTROLLERS,
    _candidate_evaluations,
    _joint_oracle,
    compute_feasibility_frontier,
    decide_outcome,
    load_accepted_config,
    load_actions,
    load_gate_spec,
    run_common_state_group,
    verify_frozen_sources,
)
from rl_agent.policy.channel import ChannelProcess, ChannelSurface
from rl_agent.policy.env import SurrogateEnv
from rl_agent.policy.replay import synthetic_episode
from rl_agent.policy.shield import ActionEvaluation


CONFIG_PATH = "rl_agent/policy/configs/expanded_action_gate_v3.yaml"


def evaluation(action: Action, reward: float) -> ActionEvaluation:
    return ActionEvaluation(
        action=action,
        hard_admitted=True,
        expected_g_m=1.0,
        risk_p95_m=1.0,
        risk_sigma_m=0.0,
        bound_m=1.0,
        expected_task_utility=1.0,
        prb_cost=0.0,
        switch_cost=0.0,
        expected_reward=reward,
        delivery_probability=1.0,
        out_of_support=False,
        payload_provenance="test",
        rate_provenance="test",
    )


class ExpandedActionGateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_gate_spec(CONFIG_PATH)
        cls.config = load_accepted_config(cls.spec)
        cls.actions = load_actions(cls.config, cls.spec)
        cls.surface = ChannelSurface(cls.config)

    def test_frozen_sources_reward_and_action_contract(self):
        self.assertEqual(len(verify_frozen_sources(self.spec)), 6)
        self.assertEqual(self.config["reward"]["formulation_version"], 5)
        self.assertEqual(
            self.config["reward"]["task_metric_weights"],
            {"miou": 0.35, "pedestrian_recall": 0.40, "vehicle_recall": 0.25},
        )
        actions = load_actions(self.config, self.spec)
        self.assertEqual(len(actions), 36)
        self.assertEqual(sum(action.mode == "SPLIT" for action in actions), 35)
        self.assertEqual(sum(action.mode == "SKIP" for action in actions), 1)
        self.assertEqual(self.spec["evaluation"]["local_status"], "excluded_uncalibrated")
        self.assertEqual(
            self.spec["evaluation"]["oracle_truth_scope"],
            "matched_deployable_track_keys_with_true_kinematics_and_capacity",
        )
        self.assertEqual(
            self.spec["evaluation"]["evaluation_mode"],
            "common_greedy_state_counterfactual",
        )

    def test_frontier_uses_production_onwire_bytes_and_separates_rate_from_latency(self):
        frontier = compute_feasibility_frontier(self.config, self.spec)
        self.assertFalse(frontier["queue_sufficiency_claimed"].any())
        stress = frontier[frontier["payload_kib"] == 400.0]
        expected = frame_onwire_bytes(400 * 1024, 60000)
        self.assertEqual(stress["onwire_bytes"].nunique(), 1)
        self.assertEqual(int(stress["onwire_bytes"].iloc[0]), expected)
        row = frontier[
            (frontier["payload_kib"] == 90.0)
            & (frontier["ue_count"] == 2)
            & (frontier["rung"] == "strong")
            & (frontier["deadline_s"] == 0.50)
            & (frontier["target_fps"] == 5)
            & (frontier["share_envelope"] == "equal_c1")
        ].iloc[0]
        # At strong/N=2 the exact UDP overhead makes 90 KiB at 5 FPS
        # 3.68928 Mbps, just above the 3.64 Mbps equal-C1 rate budget, even
        # though its queue-free latency lower bound fits 500 ms.
        self.assertFalse(bool(row["rate_feasible"]))
        self.assertTrue(bool(row["latency_necessary_feasible"]))
        self.assertFalse(bool(row["joint_necessary_feasible"]))

    def test_joint_oracle_solves_multiple_choice_rate_budget(self):
        skip = Action("SKIP", "SKIP")
        a = Action("a", "SPLIT", target_fps=1, payload_kib=122.0703125)  # 1 Mbps
        b = Action("b", "SPLIT", target_fps=1, payload_kib=244.140625)  # 2 Mbps
        candidates = [
            [evaluation(skip, 0.0), evaluation(a, 3.0), evaluation(b, 4.0)],
            [evaluation(skip, 0.0), evaluation(a, 3.0), evaluation(b, 4.0)],
        ]
        chosen = _joint_oracle(candidates, budget_mbps=3.0)
        self.assertEqual(sum(item.action.offered_mbps for item in chosen), 3.0)
        self.assertAlmostEqual(sum(item.expected_reward for item in chosen), 7.0)

    def test_skip_remains_available_after_per_ue_degradation_filter(self):
        frames = synthetic_episode("expanded_skip", [20.0], 2)
        channel = ChannelProcess(
            self.config,
            self.surface,
            seed=9,
            fixed_rungs=["clear", "clear"],
            fixed_capacity_multiplier=1.0,
        )
        env = SurrogateEnv(
            self.config,
            frames,
            self.actions,
            channel,
            self.surface,
            seed=10,
            latency_mode="p50",
            latency_crn_by_tick=True,
        )
        _, candidates, _ = _candidate_evaluations(
            env,
            self.actions,
            truth=True,
            external_rate_budget_mbps=100.0,
        )
        self.assertIn("SKIP", {item.action.action_id for item in candidates})

    def test_common_state_oracle_is_an_instantaneous_upper_bound(self):
        episodes = [
            synthetic_episode("upper_a", [18.0], 12),
            synthetic_episode("upper_b", [19.0], 12),
        ]
        rows, summaries = run_common_state_group(
            self.config,
            self.actions,
            self.surface,
            "upper_bound",
            ["upper_a", "upper_b"],
            episodes,
            1101,
            self.spec["evaluation"]["oracle_truth_scope"],
        )
        frame = pd.DataFrame(rows)
        eligible = frame[frame["primary_eligible"]]
        totals = eligible.groupby(["group_step", "controller"])["matched_truth_reward_v5"].sum().unstack()
        self.assertTrue(
            (
                totals[CONTROLLERS[1]]
                >= totals[CONTROLLERS[0]] - 1e-12
            ).all()
        )
        by_controller = {row["controller"]: row for row in summaries}
        self.assertGreaterEqual(
            by_controller[CONTROLLERS[1]]["mean_reward_v5"],
            by_controller[CONTROLLERS[0]]["mean_reward_v5"] - 1e-12,
        )

    def test_decision_gate_requires_all_registered_checks(self):
        rows = []
        for group_id, ue_count in (("g2a", 2), ("g2b", 2), ("g4", 4)):
            for seed in (1, 2, 3):
                for controller, lift in ((CONTROLLERS[0], 0.0), (CONTROLLERS[1], 0.03)):
                    rows.append(
                        {
                            "group_id": group_id,
                            "ue_count": ue_count,
                            "channel_seed": seed,
                            "controller": controller,
                            "mean_reward_v5": 0.20 + lift,
                            "worst_ue_mean_reward_v5": 0.18 + lift,
                            "aggregate_c1_miss_fraction": 0.0,
                        }
                    )
        decision = decide_outcome(pd.DataFrame(rows), self.spec)
        self.assertEqual(
            decision["verdict"],
            self.spec["decision_gate"]["positive_result"],
        )
        invalid = pd.DataFrame(rows)
        invalid.loc[invalid["controller"] == CONTROLLERS[0], "aggregate_c1_miss_fraction"] = 0.02
        held = decide_outcome(invalid, self.spec)
        self.assertEqual(held["verdict"], self.spec["decision_gate"]["invalid_result"])

    def test_source_hash_drift_fails_closed(self):
        drifted = copy.deepcopy(self.spec)
        drifted["sources"]["action_catalog"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen source hash mismatch"):
            verify_frozen_sources(drifted)


if __name__ == "__main__":
    unittest.main()
