from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rl_agent.policy.catalog import flatten_actions, load_profile_catalog
from rl_agent.policy.channel import ChannelProcess, ChannelSurface
from rl_agent.policy.config import load_controller_ladder_config
from rl_agent.policy.controllers import (
    DeployableController,
    FixedActionController,
    GreedyController,
    LinUCBController,
    MPCController,
    RuleController,
)
from rl_agent.policy.env import SurrogateEnv
from rl_agent.policy.ladder import run_deployable_controller
from rl_agent.policy.latency import LatencyProjector
from rl_agent.policy.replay import synthetic_episode
from rl_agent.policy.run_controller_ladder import _verify_corpus_contract
from rl_agent.policy.shield import SharedShield


class UnsafeController(DeployableController):
    name = "unsafe_test"

    def select(self, observation, decision):
        del observation, decision
        from rl_agent.policy.controllers import ControllerSelection

        return ControllerSelection("not-in-safe-set", {})


class CountingGreedy(GreedyController):
    name = "counting_greedy"

    def __init__(self):
        self.updates = 0

    def update(self, observation, action_id, reward):
        del observation, action_id, reward
        self.updates += 1


class ControllerLadderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_controller_ladder_config()
        cls.ladder = cls.config["controller_ladder"]
        cls.profiles = load_profile_catalog(cls.config["actions"]["catalog_csv"])
        cls.actions = flatten_actions(
            cls.profiles,
            cls.config["actions"]["fps"],
            cls.config["actions"]["preferred_core_kib"],
        )
        cls.surface = ChannelSurface(cls.config)
        cls.shield = SharedShield(cls.config, LatencyProjector(cls.config, cls.surface))

    def make_env(self, episode_id="ladder", speeds=(3.0,), steps=8):
        frames = synthetic_episode(episode_id, speeds, steps)
        channel = ChannelProcess(
            self.config,
            self.surface,
            seed=19,
            fixed_rungs=["clear"] * steps,
            fixed_capacity_multiplier=1.0,
        )
        return SurrogateEnv(
            self.config,
            frames,
            self.actions,
            channel,
            self.surface,
            seed=20,
            latency_mode="p50",
            latency_crn_by_tick=True,
        )

    def test_ladder_config_is_pre_rl_and_requires_verified_corpus(self):
        self.assertEqual(
            self.ladder["enabled_controllers"], ["fixed", "rule", "greedy", "linucb", "mpc"]
        )
        self.assertTrue(self.ladder["require_verified_corpus"])
        self.assertNotIn("dqn", self.ladder["enabled_controllers"])
        with self.assertRaisesRegex(ValueError, "verified corrected-vehicle corpus root"):
            _verify_corpus_contract(self.config)

    def test_verified_corpus_contract_checks_pass_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch_dir = Path(temporary) / "vehicle_v2_full"
            verification_dir = batch_dir / "verification" / "stamp"
            verification_dir.mkdir(parents=True)
            batch_manifest = {"mode": "full", "status": "collection_complete_pending_verification"}
            batch_path = batch_dir / "batch_manifest.json"
            batch_path.write_text(json.dumps(batch_manifest), encoding="utf-8")
            collection_path = batch_dir / "resolved_collection_config.yaml"
            collection_path.write_text(
                yaml.safe_dump({"experiment_name": "policy_corpus_vehicle_v2"}),
                encoding="utf-8",
            )
            split_path = verification_dir / "replay_split_manifest.csv"
            split_path.write_text(
                "episode_id,scenario_family,split\ne1,family,train\n", encoding="utf-8"
            )
            verification = {
                "schema": "policy_corpus_verification.v1",
                "status": "PASS",
                "gate_failures": [],
                "batch_manifest_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                "collection_config_sha256": hashlib.sha256(
                    collection_path.read_bytes()
                ).hexdigest(),
                "artifacts": {
                    "replay_split_manifest.csv": {
                        "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()
                    }
                },
            }
            verification_path = verification_dir / "verification_manifest.json"
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            config = copy.deepcopy(self.config)
            config["replay"]["roots"] = [str(batch_dir)]
            config["replay"]["split_manifest_csv"] = str(split_path)
            config["controller_ladder"]["verification_manifest_json"] = str(
                verification_path
            )
            _verify_corpus_contract(config)
            split_path.write_text(
                "episode_id,scenario_family,split\ne1,family,test\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "split manifest does not match"):
                _verify_corpus_contract(config)

    def test_fixed_rule_and_greedy_rank_only_shared_candidates(self):
        constructors = [
            FixedActionController(self.ladder["controllers"]["fixed"]["action_id"]),
            RuleController(self.config, self.ladder["controllers"]["rule"]),
            GreedyController(),
        ]
        for index, controller in enumerate(constructors):
            env = self.make_env(f"basic_{index}")
            observation = env.observation()
            decision = env.shielded_decision()
            selection = controller.select(observation, decision)
            self.assertIn(selection.action_id, decision.candidate_action_ids)

    def test_rule_skips_empty_scene_and_sends_for_new_object(self):
        controller = RuleController(self.config, self.ladder["controllers"]["rule"])
        empty = self.make_env("rule_empty", speeds=())
        empty_selection = controller.select(empty.observation(), empty.shielded_decision())
        self.assertEqual(empty_selection.action_id, "SKIP")

        new_object = self.make_env("rule_new", speeds=(4.0,))
        send_selection = controller.select(
            new_object.observation(), new_object.shielded_decision()
        )
        selected_action = next(
            action for action in self.actions if action.action_id == send_selection.action_id
        )
        self.assertEqual(selected_action.mode, "SPLIT")

    def test_fixed_action_uses_shield_fallback_when_requested_action_is_masked(self):
        env = self.make_env("fixed_mask", speeds=(20.0,))
        controller = FixedActionController("SPLIT::ae128__uint4__roi0.0::20fps")
        observation = env.observation()
        decision = env.shielded_decision()
        selection = controller.select(observation, decision)
        self.assertIn(selection.action_id, decision.candidate_action_ids)
        self.assertTrue(selection.diagnostics["fixed_fallback"])

    def test_linucb_update_is_explicit_and_serializable(self):
        controller = LinUCBController(
            [action.action_id for action in self.actions],
            alpha=0.75,
            ridge=1.0,
            reward_clip=(-2.0, 2.0),
            seed=7,
        )
        env = self.make_env("bandit")
        observation = env.observation()
        decision = env.shielded_decision()
        selection = controller.select(observation, decision)
        controller.update(observation, selection.action_id, reward=100.0)
        state = controller.state_dict()
        self.assertEqual(state["actions"][selection.action_id]["updates"], 1)
        self.assertEqual(state["reward_clip"], [-2.0, 2.0])

    def test_mpc_uses_shared_candidates_at_root_and_future_depths(self):
        mpc_spec = copy.deepcopy(self.ladder["controllers"]["mpc"])
        mpc_spec.update({"horizon_steps": 2, "future_branch_width": 2, "beam_width_per_root": 2})
        controller = MPCController(self.config, self.actions, self.shield, mpc_spec)
        env = self.make_env("mpc", speeds=(8.0,), steps=4)
        observation = env.observation()
        decision = env.shielded_decision()
        selection = controller.select(observation, decision)
        self.assertIn(selection.action_id, decision.candidate_action_ids)
        self.assertEqual(selection.diagnostics["mpc_horizon_steps"], 2)
        self.assertEqual(
            selection.diagnostics["mpc_forecast"],
            "markov_expected_capacity_modal_latency",
        )

    def test_common_runner_updates_only_when_training(self):
        training_controller = CountingGreedy()
        training = run_deployable_controller(
            self.make_env("train_updates", steps=3), training_controller, training=True
        )
        self.assertEqual(training_controller.updates, 3)
        self.assertTrue(all(row["controller_training"] for row in training.rows))

        evaluation_controller = CountingGreedy()
        evaluation = run_deployable_controller(
            self.make_env("eval_frozen", steps=3), evaluation_controller, training=False
        )
        self.assertEqual(evaluation_controller.updates, 0)
        self.assertTrue(all(not row["controller_training"] for row in evaluation.rows))

    def test_common_runner_rejects_shield_bypass(self):
        with self.assertRaisesRegex(RuntimeError, "bypassed the shared shield"):
            run_deployable_controller(
                self.make_env("unsafe", steps=2), UnsafeController(), training=False
            )


if __name__ == "__main__":
    unittest.main()
