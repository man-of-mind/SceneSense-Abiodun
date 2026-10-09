"""Offline verification of the forced-action seam for the three registered mode-11 identities.

Same composed harness as the v2 suite (real Run4DecisionEngineV2, RewardHoldControllerV2, Run4FramePipelineV2,
Run4EdgeProcessorV2, Run4EvaluatorV2 with CPU fakes) and the real frozen Run-5 actor, run once per arm, plus the
identity rule: every executed decision/hold carries exactly the dynamic contract's resolution (catalogue id for
anchors, ``action_id=None`` for the off-anchor q8000). No CARLA, OAI, GPU or network.
"""
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
from rl_agent.splitfusion_single_gpu_profiler_v1 import test_single_gpu_profiler_v1 as PT

from . import contract as C
from .forced import ForcedActionError, _ForcedActor, install_forced_action

RESOLVED = {key: C.resolve(mode, q) for key, mode, q in C.ARMS}


def rows(path):
    return [json.loads(line) for line in open(path)]


def short(forced):
    return dict(mode_id=forced["mode_id"], q_e4=forced["q_e4"], action_id=forced["action_id"], profile_id=forced["profile_id"])


class ResolutionTest(unittest.TestCase):
    def test_three_identities_are_the_contract_resolution(self):
        a67, a69, off = (RESOLVED[k] for k, _, _ in C.ARMS)
        self.assertEqual((a67["action_id"], a67["profile_id"], a67["keep_count"]), (67, "split_ae32_uint4_q3000", 15053))
        self.assertEqual((a69["action_id"], a69["profile_id"], a69["keep_count"]), (69, "split_ae32_uint4_q7000", 6451))
        self.assertEqual((off["action_id"], off["profile_id"], off["keep_count"], off["drop_count"]), (None, None, 4301, 17203))
        self.assertEqual(off["measurement_status"], "UNMEASURED_OFF_ANCHOR")
        for value in RESOLVED.values():
            self.assertEqual(C.check_forced_shape(dict(value)), value)
            self.assertEqual((value["family"], value["quantizer"]), ("AE32", "UINT4"))

    def test_invented_or_inconsistent_identities_are_refused(self):
        off = dict(RESOLVED["arm2_m11_q8000_offanchor"])
        for bad in (dict(off, action_id=70), dict(off, profile_id="split_ae32_uint4_q8000"),
                    dict(off, measurement_status="MEASURED_ANCHOR"), dict(off, keep_count=4300),
                    dict(RESOLVED["arm1_m11_q7000_a69"], action_id=None), dict(RESOLVED["arm0_m11_q3000_a67"], action_id=68),
                    dict(off, q_e4=8100), dict(off, identity_source="CATALOGUE"), dict(off, execution_bundle_sha256="x")):
            with self.assertRaises(ValueError):
                C.check_forced_shape(bad)

    def test_entry_registration_refuses_an_invented_catalogue_id(self):
        from . import entries
        import rl_agent.splitfusion_fixed_action_screen_v3.entries as EN
        root = Path(C.__file__).resolve().parents[2]
        value = dict(id=C.ID, transmitted_budget=C.BUDGET, child_module=C.CHILD, edge_module=C.EDGE,
                     live_metric_contract=C.metrics(), exact_gt_sources=C.exact_sources(root),
                     forced_action=dict(RESOLVED["arm2_m11_q8000_offanchor"]))
        value["registration_sha256"] = C.canonical(value)
        directory = Path(tempfile.mkdtemp(prefix="v3_reg_"))
        (directory / "ok.json").write_text(json.dumps(value))
        self.assertEqual(EN.registration(directory / "ok.json", root)["forced_action"]["action_id"], None)
        bad = dict(value, forced_action=dict(value["forced_action"], action_id=70))
        bad.pop("registration_sha256")
        bad["registration_sha256"] = C.canonical(bad)
        (directory / "bad.json").write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            EN.registration(directory / "bad.json", root)
        self.assertIs(entries, EN)


class ForcedSeamComposedTest(unittest.TestCase):
    def run_composed(self, forced, profiled=False):
        directory = Path(tempfile.mkdtemp(prefix="forced_seam_v3_"))
        log = directory / "forced" / "FORCED_ACTIONS.jsonl"
        with install_forced_action(log_path=log, forced=short(forced)) as state:
            out = PT.run_scenario(profiled, directory)
        return directory, log, state, out

    def check(self, forced, log, out):
        mode, q = forced["mode_id"], forced["q_e4"]
        decisions = [p for p in out["plans"] if p[0] == "POLICY_DECISION"]
        holds = [p for p in out["plans"] if p[0] == "POLICY_HOLD"]
        fallbacks = [p for p in out["plans"] if p[0] == "FALLBACK"]
        self.assertTrue(decisions and holds and fallbacks)
        for _, m, v, _ in decisions + holds:
            self.assertEqual((m, v), (mode, q))
        for _, m, v, _ in fallbacks:
            self.assertEqual((m, v), (11, 9800))
        self.assertEqual(out["actor_calls"], len(decisions))
        logged = rows(log)
        self.assertEqual(len(logged), len(out["plans"]))
        for row in logged:
            if row["kind"] == "POLICY_DECISION":
                self.assertEqual((row["proposed"]["proposed_mode_id"], row["proposed"]["proposed_q_e4"]), (9, 4321))
                self.assertEqual(row["actor_calls_this_frame"], 1)
            else:
                self.assertIsNone(row["proposed"])
            if row["kind"] != "FALLBACK":       # exact executed identity, including off-anchor None ids
                ex = row["executed"]
                self.assertEqual((ex["mode_id"], ex["q_e4"], ex["action_id"], ex["profile_id"], ex["keep_count"],
                                  ex["execution_bundle_sha256"]),
                                 (mode, q, forced["action_id"], forced["profile_id"], forced["keep_count"],
                                  forced["execution_bundle_sha256"]))
            else:
                self.assertEqual((row["executed"]["action_id"], row["executed"]["profile_id"]), (71, "split_ae32_uint4_q9800"))
        summary = json.loads(log.with_name("FORCED_ACTION_SUMMARY.json").read_text())
        self.assertEqual(summary["counters"]["executed_mismatch"], 0)
        self.assertEqual(summary["counters"]["decisions"], len(decisions))
        self.assertEqual(len(out["resolutions"]), len(decisions))
        self.assertNotIn("MISMATCH", " ".join(out["classes"]).upper())

    def test_arm0_a67(self):
        _, log, _, out = self.run_composed(RESOLVED["arm0_m11_q3000_a67"])
        self.check(RESOLVED["arm0_m11_q3000_a67"], log, out)

    def test_arm1_a69(self):
        _, log, _, out = self.run_composed(RESOLVED["arm1_m11_q7000_a69"])
        self.check(RESOLVED["arm1_m11_q7000_a69"], log, out)

    def test_arm2_offanchor_q8000(self):
        _, log, _, out = self.run_composed(RESOLVED["arm2_m11_q8000_offanchor"])
        self.check(RESOLVED["arm2_m11_q8000_offanchor"], log, out)
        self.assertIs(E.Run4DecisionEngineV2.__name__, "Run4DecisionEngineV2")

    def test_profiled_span_records_genuine_proposal_offanchor(self):
        directory, log, _, out = self.run_composed(RESOLVED["arm2_m11_q8000_offanchor"], profiled=True)
        self.check(RESOLVED["arm2_m11_q8000_offanchor"], log, out)
        spans = [r for r in rows(directory / "ue.profile.jsonl") if r.get("r") == "span" and r.get("stage") == "agent.actor"]
        self.assertTrue(spans)
        for span in spans:
            self.assertEqual((span["decided_mode_id"], span["decided_q_e4"]), (9, 4321))

    def test_unforced_baseline_is_identical_except_action(self):
        out = PT.run_scenario(False, Path(tempfile.mkdtemp(prefix="forced_base_v3_")))
        _, _, _, forced = self.run_composed(RESOLVED["arm2_m11_q8000_offanchor"])
        self.assertEqual([p[0] for p in out["plans"]], [p[0] for p in forced["plans"]])
        self.assertEqual([p[3] for p in out["plans"]], [p[3] for p in forced["plans"]])
        self.assertEqual(out["classes"], forced["classes"])

    def test_previous_outcome_features_carry_offanchor_executed_action(self):
        captured = []
        original_init = PT._DeterministicPipeline.__init__

        def init(pipe):
            original_init(pipe)
            captured.append(pipe)
        PT._DeterministicPipeline.__init__ = init
        try:
            self.run_composed(RESOLVED["arm2_m11_q8000_offanchor"])
        finally:
            PT._DeterministicPipeline.__init__ = original_init
        resolutions = [r for c in captured[0].engine.controllers for r in c.resolutions()]
        self.assertTrue(resolutions)
        for resolution in resolutions:
            self.assertEqual((resolution.action.mode_id, resolution.action.q_e4, resolution.action.action_id), (11, 8000, None))
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as RC
        previous = RC.PreviousOutcomeV1.from_resolution(resolutions[-1])
        self.assertEqual((previous.action.mode_id, previous.action.q_e4), (11, 8000))

    def test_nested_installation_refused_and_restored(self):
        directory = Path(tempfile.mkdtemp(prefix="forced_nest_v3_"))
        original = E.Run4DecisionEngineV2
        forced = short(RESOLVED["arm1_m11_q7000_a69"])
        with install_forced_action(log_path=directory / "a.jsonl", forced=forced):
            with self.assertRaises(ForcedActionError):
                with install_forced_action(log_path=directory / "b.jsonl", forced=forced):
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

    def test_three_arms_inside_frozen_support_and_type_preserved(self):
        self.assertEqual((self.actor._lower[11], self.actor._upper[11]), (0, 9791))
        proposal = self.actor.act(self.vector)
        for value in RESOLVED.values():
            sink = []
            executed = _ForcedActor(self.actor, short(value), sink).act(self.vector)
            self.assertIs(type(executed), type(proposal))
            self.assertEqual((executed.mode_id, executed.q_e4), (value["mode_id"], value["q_e4"]))
            self.assertEqual((executed.state_sha256, executed.heads_sha256), (proposal.state_sha256, proposal.heads_sha256))
            self.assertEqual((sink[0]["proposed_mode_id"], sink[0]["proposed_q_e4"]), (proposal.mode_id, proposal.q_e4))

    def test_state_is_the_original_22d_run5_state(self):
        from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S5
        self.assertIs(type(self.vector), S5.Run5FeatureVectorV2)     # quality-aware, SNR-bearing Run-5 state
        self.assertEqual(len(self.vector.as_tuple()), 22)

    def test_outside_support_still_refused(self):
        with self.assertRaises(ForcedActionError):
            _ForcedActor(self.actor, dict(mode_id=11, q_e4=9800), []).act(self.vector)


if __name__ == "__main__":
    unittest.main()
