#!/usr/bin/env python3
"""Concrete tests for the real modeled collector.

These exercise the actual registered artifact, not a fixture.  No test
launches a radio, CARLA, OAI, Docker or any network service.
"""

from __future__ import annotations

import collections
import unittest
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import collector_v1 as CV
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2


ARTIFACT = (C2.ROOT / "rl_agent/experiments/ue_production_queue_capture_v1"
            / "20260929_model_v2b/transport_model_v2.json")


def _requests(count: int) -> list[orch.ModeledActionRequestV1]:
    schedule = orch.build_frozen_warmup_schedule()
    out = []
    for index in range(count):
        action = schedule.action_at(index)
        out.append(orch.ModeledActionRequestV1(
            decision_ordinal=index, mode_id=action.mode_id,
            q_e4=action.q_e4, source="STRATIFIED_WARMUP",
            warmup_q_bin_index=action.q_bin_index))
    return out


class CollectorTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ARTIFACT.is_file():
            raise unittest.SkipTest(f"v2 artifact missing: {ARTIFACT}")
        cls.seed_collector = CV.RealModeledTransitionCollectorV1(
            artifact_path=ARTIFACT)
        cls.shared = cls.seed_collector.shared_sources()

    def fresh(self, seed: int = C2.MODEL_SEED
              ) -> CV.RealModeledTransitionCollectorV1:
        return CV.RealModeledTransitionCollectorV1(
            artifact_path=ARTIFACT, seed=seed, shared_sources=self.shared)


class SupportRegressionTests(CollectorTestBase):
    def test_all_288_registered_actions_complete_without_out_of_support(self):
        """The registered schedule must never hit the support refusal."""
        collector = self.fresh()
        requests = _requests(288)
        self.assertEqual(len(requests), 288)
        for request in requests:
            try:
                collector.collect(request)
            except A2.OutOfSupport as error:  # pragma: no cover - regression
                self.fail(f"decision {request.decision_ordinal} "
                          f"(mode {request.mode_id}, q {request.q_e4}) "
                          f"was refused: {error}")
        self.assertEqual(collector.decision_count, 288)
        diagnostics = collector.diagnostics()
        self.assertEqual(len({d["mode_id"] for d in diagnostics}), 12)
        terminals = collections.Counter(d["terminal"] for d in diagnostics)
        self.assertGreater(terminals["SUCCESS"], 0)
        self.assertGreater(terminals["TIMEOUT"], 0)


class PersistentSessionTests(CollectorTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.collector = CV.RealModeledTransitionCollectorV1(
            artifact_path=ARTIFACT, shared_sources=cls.seed_collector
            .shared_sources())
        for request in _requests(24):
            cls.collector.collect(request)
        cls.diag = cls.collector.diagnostics()

    def test_one_session_and_monotonic_decision_seq(self) -> None:
        self.assertEqual(len({d["session_uuid"] for d in self.diag}), 1)
        self.assertEqual([d["decision_seq"] for d in self.diag],
                         list(range(len(self.diag))))

    def test_exact_previous_outcome_propagates(self) -> None:
        self.assertFalse(self.diag[0]["previous_present"])
        for index in range(1, len(self.diag)):
            current, previous = self.diag[index], self.diag[index - 1]
            self.assertTrue(current["previous_present"])
            self.assertEqual(current["previous_mode_id"], previous["mode_id"])
            self.assertEqual(current["previous_q_e4"], previous["q_e4"])
            success = previous["terminal"] == "SUCCESS"
            self.assertEqual(current["previous_success"], success)
            if success:
                self.assertAlmostEqual(
                    float(current["previous_q_perc"]),
                    float(previous["q_perc"]), places=9)
                self.assertAlmostEqual(
                    float(current["previous_latency_ms"]),
                    previous["composed_ns"] / 1e6, places=6)
            else:
                self.assertIsNone(current["previous_q_perc"])
                self.assertIsNone(current["previous_latency_ms"])

    def test_held_frame_is_an_independent_scene_and_payload(self) -> None:
        for row in self.diag:
            self.assertNotEqual(row["reward_scene_key"], row["held_scene_key"])
            self.assertEqual(
                row["ingress_bytes"],
                row["reward_frame_wire_bytes"] + row["held_frame_wire_bytes"])
        # the held payload is genuinely independent, not a duplicate
        doubled = sum(1 for row in self.diag
                      if row["ingress_bytes"] == 2 * row["reward_frame_wire_bytes"])
        self.assertEqual(doubled, 0)

    def test_queue_and_mcs_advance_exactly_once_per_decision(self) -> None:
        for index in range(1, len(self.diag)):
            self.assertEqual(self.diag[index]["pre_enqueue_backlog_bytes"],
                             self.diag[index - 1]["next_backlog_bytes"])

    def test_reward_is_never_scaled_by_probability(self) -> None:
        for row in self.diag:
            if row["terminal"] == "SUCCESS":
                expected = (row["q_perc"]
                            - 0.25 * (row["composed_ns"] / 1e6) / 170.0)
                self.assertAlmostEqual(row["reward"], expected, places=9)
            else:
                self.assertEqual(row["reward"], -1.0)


class RngIsolationTests(CollectorTestBase):
    def test_streams_are_independent_and_do_not_touch_global_state(self) -> None:
        import random as global_random
        global_random.seed(1234)
        before = global_random.random()
        collector = self.fresh()
        for request in _requests(12):
            collector.collect(request)
        global_random.seed(1234)
        self.assertEqual(global_random.random(), before)

    def test_reward_and_held_scene_streams_are_distinct(self) -> None:
        collector = self.fresh()
        for request in _requests(24):
            collector.collect(request)
        diagnostics = collector.diagnostics()
        reward = [d["reward_scene_key"] for d in diagnostics]
        held = [d["held_scene_key"] for d in diagnostics]
        self.assertNotEqual(reward, held)
        # neither stream is a shifted copy of the other
        self.assertNotEqual(reward[1:], held[:-1])
        self.assertNotEqual(held[1:], reward[:-1])

    def test_a_different_seed_changes_the_trajectory(self) -> None:
        first = self.fresh(seed=C2.MODEL_SEED)
        second = self.fresh(seed=C2.MODEL_SEED + 1)
        for request in _requests(12):
            first.collect(request)
            second.collect(request)
        self.assertNotEqual(
            [d["reward_scene_key"] for d in first.diagnostics()],
            [d["reward_scene_key"] for d in second.diagnostics()])


class CheckpointResumeTests(CollectorTestBase):
    def test_checkpoint_restores_bit_identically_and_continues(self) -> None:
        requests = _requests(20)
        original = self.fresh()
        for request in requests[:12]:
            original.collect(request)
        checkpoint = original.checkpoint()
        self.assertEqual(checkpoint.decision_count, 12)

        restored = self.fresh()
        restored.restore(checkpoint)
        self.assertEqual(restored.decision_count, 12)
        self.assertEqual(
            tuple(item.wrapper.transition_sha256
                  for item in restored.history()),
            checkpoint.transition_sha256s)
        self.assertEqual(restored.checkpoint().payload_sha256,
                         checkpoint.payload_sha256)
        self.assertEqual(restored.current_state_features(),
                         original.current_state_features())

        # the next action must produce the identical transition on both
        nxt = requests[12]
        a = original.collect(nxt)
        b = restored.collect(nxt)
        self.assertEqual(a.wrapper.transition_sha256,
                         b.wrapper.transition_sha256)
        self.assertEqual(a.state_features_sha256, b.state_features_sha256)
        self.assertEqual(a.reward, b.reward)
        self.assertEqual(original.checkpoint().payload_sha256,
                         restored.checkpoint().payload_sha256)

    def test_restore_refuses_a_foreign_checkpoint(self) -> None:
        collector = self.fresh()
        for request in _requests(4):
            collector.collect(request)
        checkpoint = collector.checkpoint()
        other = self.fresh(seed=C2.MODEL_SEED + 7)
        with self.assertRaises(CV.CollectorError):
            other.restore(checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
