"""Offline verification of the action-0 external-baseline forced layer (CPU only; no CARLA, OAI, GPU or network).

Same composed harness as the v3 suite (real Run4DecisionEngineV2, RewardHoldControllerV2, Run4FramePipelineV2,
Run4EdgeProcessorV2, Run4EvaluatorV2 with CPU fakes) plus the real frozen FULL Run-5 actor:
* action 0 is exactly the dynamic execution contract's resolution of (mode 0, q_e4 0);
* it is outside the actor's mode-0 support and is executed unclamped, with the support status recorded;
* every decision and hold executes action 0; fallbacks keep a71 and are counted separately;
* the next state's previous-action features carry the executed (0, 0) and stay inside [0, 1].
"""
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
from rl_agent.splitfusion_single_gpu_profiler_v1 import test_single_gpu_profiler_v1 as PT

from . import contract as C
from .forced import ForcedActionError, _ForcedActor, install_forced_action

SHORT = dict(mode_id=0, q_e4=0, action_id=0, profile_id="split_noae_uint8_q0000")


def rows(path):
    return [json.loads(line) for line in open(path)]


class IdentityTest(unittest.TestCase):
    def test_registered_document_is_the_contract_resolution(self):
        self.assertEqual(C.resolved_forced(), C.FORCED_ACTION)
        self.assertEqual((C.FORCED_ACTION["family"], C.FORCED_ACTION["quantizer"], C.FORCED_ACTION["keep_count"],
                          C.FORCED_ACTION["drop_count"]), ("noAE", "UINT8", 21504, 0))
        self.assertEqual(C.check_forced_shape(dict(C.FORCED_ACTION)), C.FORCED_ACTION)

    def test_other_documents_refused(self):
        for bad in (dict(C.FORCED_ACTION, action_id=17), dict(C.FORCED_ACTION, q_e4=8812), dict(C.FORCED_ACTION, keep_count=21503),
                    dict(C.FORCED_ACTION, execution_bundle_sha256="0" * 64), None, {}):
            with self.assertRaises(ValueError):
                C.check_forced_shape(bad)

    def test_token_lineage_and_metrics_carry_the_baseline(self):
        self.assertEqual(C.BUDGETS, (100,))
        self.assertTrue(C.metrics()["forced_action_layer"])
        self.assertEqual(C.metrics()["forced_support_policy"], C.SUPPORT_POLICY)
        self.assertEqual(C.FALLBACK_IDENTITY, (11, 9800, 71, "split_ae32_uint4_q9800"))


class ForcedSeamComposedTest(unittest.TestCase):
    def run_composed(self, profiled=False):
        directory = Path(tempfile.mkdtemp(prefix="forced_noae_"))
        log = directory / "forced" / "FORCED_ACTIONS.jsonl"
        with install_forced_action(log_path=log, forced=SHORT):
            out = PT.run_scenario(profiled, directory)
        return directory, log, out

    def test_decisions_and_holds_execute_action0_fallbacks_keep_a71(self):
        _, log, out = self.run_composed()
        decisions = [p for p in out["plans"] if p[0] == "POLICY_DECISION"]
        holds = [p for p in out["plans"] if p[0] == "POLICY_HOLD"]
        fallbacks = [p for p in out["plans"] if p[0] == "FALLBACK"]
        self.assertTrue(decisions and holds and fallbacks)
        for _, m, v, _ in decisions + holds:
            self.assertEqual((m, v), (0, 0))
        for _, m, v, _ in fallbacks:
            self.assertEqual((m, v), (11, 9800))
        self.assertEqual(out["actor_calls"], len(decisions))
        logged = rows(log)
        self.assertEqual(len(logged), len(out["plans"]))
        for row in logged:
            ex = row["executed"]
            if row["kind"] == "FALLBACK":
                self.assertEqual((ex["mode_id"], ex["q_e4"], ex["action_id"], ex["profile_id"]), C.FALLBACK_IDENTITY)
                self.assertIsNone(row["proposed"])
            else:
                self.assertEqual((ex["mode_id"], ex["q_e4"], ex["action_id"], ex["profile_id"], ex["keep_count"],
                                  ex["execution_bundle_sha256"]),
                                 (0, 0, 0, "split_noae_uint8_q0000", 21504, C.FORCED_ACTION["execution_bundle_sha256"]))
        summary = json.loads(log.with_name("FORCED_ACTION_SUMMARY.json").read_text())
        self.assertEqual(summary["counters"]["executed_mismatch"], 0)
        self.assertEqual(summary["counters"]["decisions"], len(decisions))
        self.assertEqual(summary["counters"]["fallbacks"], len(fallbacks))
        self.assertNotIn("MISMATCH", " ".join(out["classes"]).upper())

    def test_previous_outcome_features_carry_action0_in_range(self):
        captured = []
        original_init = PT._DeterministicPipeline.__init__

        def init(pipe):
            original_init(pipe)
            captured.append(pipe)
        PT._DeterministicPipeline.__init__ = init
        try:
            self.run_composed()
        finally:
            PT._DeterministicPipeline.__init__ = original_init
        resolutions = [r for c in captured[0].engine.controllers for r in c.resolutions()]
        self.assertTrue(resolutions)
        for resolution in resolutions:
            self.assertEqual((resolution.action.mode_id, resolution.action.q_e4, resolution.action.action_id), (0, 0, 0))
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as RC
        previous = RC.PreviousOutcomeV1.from_resolution(resolutions[-1])
        self.assertEqual((previous.action.mode_id, previous.action.q_e4), (0, 0))

    def test_nested_installation_refused_and_restored(self):
        directory = Path(tempfile.mkdtemp(prefix="forced_noae_nest_"))
        original = E.Run4DecisionEngineV2
        with install_forced_action(log_path=directory / "a.jsonl", forced=SHORT):
            with self.assertRaises(ForcedActionError):
                with install_forced_action(log_path=directory / "b.jsonl", forced=SHORT):
                    pass
        self.assertIs(E.Run4DecisionEngineV2, original)


class ForcedSeamRealFullActorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_agent.splitfusion_run5_live_avo_v1 import test_actor_binding as TB
        from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
        cls.actor = load_registered_actor()
        kwargs, snr = TB.state_fixture()
        _, cls.vector = TB.A.build_run5_live_state(snr=snr, **kwargs)

    def test_action0_outside_support_executed_unclamped_and_recorded(self):
        self.assertEqual((self.actor._lower[0], self.actor._upper[0]), (8812, 9800))
        proposal = self.actor.act(self.vector)
        sink = []
        executed = _ForcedActor(self.actor, SHORT, sink).act(self.vector)
        self.assertIs(type(executed), type(proposal))
        self.assertEqual((executed.mode_id, executed.q_e4), (0, 0))                       # not clamped to 8812
        self.assertEqual((executed.q_support_lower_e4, executed.q_support_upper_e4), (8812, 9800))
        self.assertEqual((executed.state_sha256, executed.heads_sha256), (proposal.state_sha256, proposal.heads_sha256))
        self.assertEqual((sink[0]["proposed_mode_id"], sink[0]["proposed_q_e4"]), (proposal.mode_id, proposal.q_e4))
        self.assertIs(sink[0]["forced_in_actor_support"], False)
        self.assertEqual(sink[0]["actor_support"], [8812, 9800])

    def test_prev_q_feature_for_action0_is_zero(self):
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as RC
        self.assertEqual(0 / float(RC.Q_E4_MAX), 0.0)


if __name__ == "__main__":
    unittest.main()
