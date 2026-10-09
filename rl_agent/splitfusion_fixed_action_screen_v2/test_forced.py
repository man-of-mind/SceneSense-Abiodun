"""Offline verification of the forced-action seam against the real retained engine/controller/pipeline.

Uses the profiler's composed harness (real Run4DecisionEngineV2, RewardHoldControllerV2,
Run4FramePipelineV2, Run4EdgeProcessorV2 and Run4EvaluatorV2 with CPU fakes) and the real
frozen Run-5 actor for the decision-type checks. No CARLA, OAI, GPU or network.
"""
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
from rl_agent.splitfusion_single_gpu_profiler_v1 import test_single_gpu_profiler_v1 as PT

from .forced import ForcedActionError, _ForcedActor, install_forced_action

FORCED = dict(mode_id=5, q_e4=7000, action_id=33, profile_id="split_ae128_uint4_q7000")


def rows(path):
    return [json.loads(line) for line in open(path)]


class ForcedSeamComposedTest(unittest.TestCase):
    def run_composed(self, profiled):
        directory = Path(tempfile.mkdtemp(prefix="forced_seam_"))
        log = directory / "forced" / "FORCED_ACTIONS.jsonl"
        with install_forced_action(log_path=log, forced=FORCED) as state:
            self.assertIsNot(E.Run4DecisionEngineV2.__mro__[1], object)
            out = PT.run_scenario(profiled, directory)
        return directory, log, state, out

    def check_common(self, log, state, out):
        decisions = [p for p in out["plans"] if p[0] == "POLICY_DECISION"]
        holds = [p for p in out["plans"] if p[0] == "POLICY_HOLD"]
        fallbacks = [p for p in out["plans"] if p[0] == "FALLBACK"]
        self.assertTrue(decisions and holds and fallbacks)
        for kind, mode, q, _ in decisions + holds:          # executed action drives decisions and holds
            self.assertEqual((mode, q), (5, 7000))
        for kind, mode, q, _ in fallbacks:                   # retained fallback semantics unchanged
            self.assertEqual((mode, q), (11, 9800))
        self.assertEqual(out["actor_calls"], len(decisions))  # shadow actor runs exactly once per decision
        self.assertEqual(out["counters"]["actor_calls"], out["counters"]["policy_decisions"])
        logged = rows(log)
        self.assertEqual(len(logged), len(out["plans"]))
        for row in logged:
            if row["kind"] == "POLICY_DECISION":
                self.assertEqual((row["proposed"]["proposed_mode_id"], row["proposed"]["proposed_q_e4"]), (9, 4321))
                self.assertEqual(row["actor_calls_this_frame"], 1)
            else:
                self.assertIsNone(row["proposed"])
            if row["kind"] != "FALLBACK":
                self.assertEqual((row["executed"]["mode_id"], row["executed"]["q_e4"]), (5, 7000))
                self.assertEqual(row["executed"]["action_id"], 33)
        summary = json.loads(log.with_name("FORCED_ACTION_SUMMARY.json").read_text())
        self.assertEqual(summary["counters"]["decisions"], len(decisions))
        self.assertEqual(summary["counters"]["proposal_differs"], len(decisions))
        self.assertEqual(summary["counters"]["executed_mismatch"], 0)
        # feedback matched the executed (forced) ticket: every decision resolved, none as identity mismatch
        self.assertEqual(len(out["resolutions"]), len(decisions))
        self.assertNotIn("MISMATCH", " ".join(out["classes"]).upper())

    def test_unprofiled_forced_action_drives_bookkeeping(self):
        _, log, state, out = self.run_composed(False)
        self.check_common(log, state, out)
        self.assertIs(E.Run4DecisionEngineV2.__name__, "Run4DecisionEngineV2")   # restored

    def test_profiled_span_records_genuine_proposal(self):
        directory, log, state, out = self.run_composed(True)
        self.check_common(log, state, out)
        spans = [r for r in rows(directory / "ue.profile.jsonl") if r.get("r") == "span" and r.get("stage") == "agent.actor"]
        self.assertTrue(spans)
        for span in spans:                                   # profiler sees the actor's own decision
            self.assertEqual((span["decided_mode_id"], span["decided_q_e4"]), (9, 4321))

    def test_unforced_baseline_is_identical_except_action(self):
        directory = Path(tempfile.mkdtemp(prefix="forced_base_"))
        out = PT.run_scenario(False, directory)
        _, _, _, forced = self.run_composed(False)
        self.assertEqual([p[0] for p in out["plans"]], [p[0] for p in forced["plans"]])
        self.assertEqual([p[3] for p in out["plans"]], [p[3] for p in forced["plans"]])
        self.assertEqual(out["classes"], forced["classes"])

    def test_previous_outcome_features_carry_executed_action(self):
        captured = []
        original_init = PT._DeterministicPipeline.__init__

        def init(pipe):
            original_init(pipe)
            captured.append(pipe)
        PT._DeterministicPipeline.__init__ = init
        try:
            _, _, _, out = self.run_composed(False)
        finally:
            PT._DeterministicPipeline.__init__ = original_init
        engine = captured[0].engine
        resolutions = [r for c in engine.controllers for r in c.resolutions()]
        self.assertTrue(resolutions)
        for resolution in resolutions:                      # reward events and next-state features
            self.assertEqual((resolution.action.mode_id, resolution.action.q_e4), (5, 7000))
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as RC
        previous = RC.PreviousOutcomeV1.from_resolution(resolutions[-1])
        self.assertEqual((previous.action.mode_id, previous.action.q_e4), (5, 7000))

    def test_nested_installation_refused_and_restored(self):
        directory = Path(tempfile.mkdtemp(prefix="forced_nest_"))
        original = E.Run4DecisionEngineV2
        with install_forced_action(log_path=directory / "a.jsonl", forced=FORCED):
            with self.assertRaises(ForcedActionError):
                with install_forced_action(log_path=directory / "b.jsonl", forced=FORCED):
                    pass
        self.assertIs(E.Run4DecisionEngineV2, original)


class ForcedSeamRealActorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_agent.splitfusion_run5_live_avo_v1 import test_actor_binding as TB
        from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
        cls.actor = load_registered_actor()
        kwargs, snr = TB.state_fixture()
        _, cls.vector = TB.A.build_run5_live_state(snr=snr, **kwargs)

    def test_real_decision_type_and_support(self):
        sink = []
        proposal = self.actor.act(self.vector)
        executed = _ForcedActor(self.actor, FORCED, sink).act(self.vector)
        self.assertIs(type(executed), type(proposal))
        self.assertEqual((executed.mode_id, executed.q_e4), (5, 7000))
        self.assertEqual((executed.q_support_lower_e4, executed.q_support_upper_e4),
                         (self.actor._lower[5], self.actor._upper[5]))
        self.assertEqual((executed.state_sha256, executed.heads_sha256), (proposal.state_sha256, proposal.heads_sha256))
        self.assertEqual((sink[0]["proposed_mode_id"], sink[0]["proposed_q_e4"]), (proposal.mode_id, proposal.q_e4))

    def test_forced_action_outside_actor_support_is_refused(self):
        with self.assertRaises(ForcedActionError):                 # action 71: mode 11 q 9800 > 9791
            _ForcedActor(self.actor, dict(mode_id=11, q_e4=9800), []).act(self.vector)

    def test_all_registered_screen_actions_inside_support(self):
        for action_id in (17, 33, 35, 50, 53, 65, 67):
            mode, index = divmod(action_id, 6)
            q = (0, 3000, 5000, 7000, 9000, 9800)[index]
            sink = []
            decision = _ForcedActor(self.actor, dict(mode_id=mode, q_e4=q), sink).act(self.vector)
            self.assertEqual((decision.mode_id, decision.q_e4), (mode, q))


if __name__ == "__main__":
    unittest.main()
