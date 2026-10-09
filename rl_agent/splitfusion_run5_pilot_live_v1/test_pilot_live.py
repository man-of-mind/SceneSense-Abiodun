"""Focused adapter / state / action-parity tests for the pilot live actor (CPU, no CARLA/OAI/GPU).

Run from the live worktree root (or main):
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m unittest -v rl_agent.splitfusion_run5_pilot_live_v1.test_pilot_live
"""
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as F4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S5
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A
from rl_agent.splitfusion_run5_restricted_pilot_v1 import support as SUP

from . import actor as PA
from . import contract as C


def random_states(n, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prev_mode = [0.0] * 12
        present = rng.random() < 0.9
        if present:
            prev_mode[rng.choice(SUP.ALLOWED_MODES)] = 1.0
        success = present and rng.random() < 0.7
        out.append(tuple([rng.uniform(-2.5, 2.5), rng.uniform(0, 1), rng.choice(range(0, 29)) / 28.0, rng.uniform(0, 0.6)]
                         + prev_mode + [rng.uniform(0.3, 0.98) if present else 0.0,
                                        rng.uniform(0, 0.9) if success else 0.0,
                                        rng.uniform(0.5, 1.0) if success else 0.0,
                                        1.0 if present else 0.0, 1.0 if success else 0.0, rng.uniform(0, 1)]))
    return out


class ExportTest(unittest.TestCase):
    def test_export_binds_the_selected_actor_and_audits_clean(self):
        self.assertEqual(F4.sha256_file(PA.EXPORT_DIR / "actor_state_dict.pt"), PA.WEIGHTS_SHA256)
        self.assertEqual(PA.WEIGHTS_SHA256, "ecd7ce4ae20de037fe41670a91406259fc2567f9228adfe1084a55e507c5897c")
        source = PA.ROOT / PA.SOURCE_WEIGHTS
        if source.exists():                         # main checkout: the run's own file
            self.assertEqual(F4.sha256_file(source), PA.WEIGHTS_SHA256)
        audit = PA.audit()
        self.assertEqual(audit["verdict"], "PASS", audit["checks"])
        binding = json.loads((PA.EXPORT_DIR / "ACTOR_BINDING.json").read_text())
        self.assertEqual(binding["selected"], {"arm": "CANDIDATE", "seed": 43, "update": 10000})
        self.assertIn("NOT_A_BEST_SEED_CLAIM", binding["selection_scope"])
        self.assertEqual(binding["binding"]["policy_feature_order"][17], "prev_quality_qperc")
        self.assertEqual(binding["binding"]["policy_feature_order"][21], "effective_external_ul_snr_proxy_scaled")

    def test_refuses_original_export_and_tampered_weights(self):
        tmp = Path(tempfile.mkdtemp(prefix="pilot_export_"))
        shutil.copy(PA.EXPORT_DIR / "ACTOR_BINDING.json", tmp / "ACTOR_BINDING.json")
        shutil.copy(PA.ROOT / "rl_agent/splitfusion_run5_live_avo_v1/actor_export/actor_state_dict.pt",
                    tmp / "actor_state_dict.pt")
        with self.assertRaises(PA.PilotLiveActorError):
            PA.load_export(tmp)
        state = torch.load(PA.EXPORT_DIR / "actor_state_dict.pt", weights_only=True)
        state["mean_head.bias"] = state["mean_head.bias"] + 1e-3
        torch.save(state, tmp / "actor_state_dict.pt")
        with self.assertRaises(PA.PilotLiveActorError):
            PA.load_export(tmp)

    def test_original_loader_and_wrapper_refuse_the_pilot_actor(self):
        actor = PA.load_export()
        with self.assertRaises(A.Run5LiveBindingError):
            A.FrozenRun5Actor(actor.module)              # original wrapper: foreign class
        original = A.load_source_export(PA.ROOT / "rl_agent/splitfusion_run5_live_avo_v1/actor_export")
        with self.assertRaises(Exception):
            original.module.load_state_dict(torch.load(PA.EXPORT_DIR / "actor_state_dict.pt", weights_only=True))


class ActionParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor = PA.load_export()
        cls.contract = dec.load_dynamic_execution_contract()

    def test_decisions_equal_the_masked_module_rule_and_stay_in_support(self):
        module = self.actor.module
        modes = set()
        for values in random_states(600):
            decision = self.actor.act_on_vector(values)
            with torch.inference_mode():
                ex = module.deterministic_execution(torch.tensor((values,), dtype=torch.float32))
            self.assertEqual((decision.mode_id, decision.q_e4), (int(ex.mode_index[0]), int(ex.q_e4[0])))
            self.assertTrue(SUP.in_support(decision.mode_id, decision.q_e4))
            lo, hi = SUP.BRACKETS[decision.mode_id]
            self.assertEqual((decision.q_support_lower_e4, decision.q_support_upper_e4), (lo, hi))
            self.assertEqual(decision.actor_boundary_sha256, PA.ACTOR_TREE_SHA256)
            self.assertEqual(self.actor.act_on_vector(values), decision)          # deterministic, frozen
            profile = self.contract.resolve_q_e4(decision.mode_id, decision.q_e4)  # executable by the engine
            self.assertEqual((profile.mode_id, profile.q_e4), (decision.mode_id, decision.q_e4))
            modes.add(decision.mode_id)
        self.assertTrue(modes <= set(SUP.ALLOWED_MODES))
        self.assertFalse(any(p.requires_grad for p in module.parameters()))
        self.assertFalse(module.training)

    def test_real_run5_state_vector_and_quality_feature(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import test_actor_binding as TB
        kwargs, snr = TB.state_fixture()
        _, vector = A.build_run5_live_state(snr=snr, **kwargs)
        self.assertIs(type(vector), S5.Run5FeatureVectorV2)
        values = vector.as_tuple()
        self.assertEqual(len(values), 22)
        self.assertEqual(self.actor.act(vector), self.actor.act_on_vector(values))
        with self.assertRaises(PA.PilotLiveActorError):
            self.actor.act(values)                                   # foreign same-width state refused
        with self.assertRaises(PA.PilotLiveActorError):
            self.actor.act_on_vector(values[:21])

    def test_profiler_replay_with_the_registered_loader_matches(self):
        from rl_agent.splitfusion_single_gpu_profiler_v1 import analyze_v1 as AN
        records, decisions = [], []
        for i, values in enumerate(random_states(40, seed=11)):
            d = self.actor.act_on_vector(values)
            records.append(dict(r="span", stage="agent.actor", frame_id=i, features=list(values), decided_mode_id=d.mode_id,
                                decided_q_e4=d.q_e4, state_sha256=d.state_sha256, heads_sha256=d.heads_sha256))
            decisions.append(dict(frame_id=i, kind="POLICY_DECISION", mode_id=d.mode_id, q_e4=d.q_e4))
        result = AN.actor_replay(records, decisions, loader="rl_agent.splitfusion_run5_pilot_live_v1.actor:load_registered_actor",
                                 method="act_on_vector")
        self.assertTrue(result["passed"], result)
        wrong = AN.actor_replay(records, decisions, loader="rl_agent.splitfusion_run5_live_avo_v1.audit_actor:load_registered_actor",
                                method="act_on_vector")
        self.assertFalse(wrong["passed"])                            # the replay discriminates actors


class CompositionTest(unittest.TestCase):
    def test_budgets_limits_and_lineage(self):
        self.assertEqual(C.BUDGETS, (100, 1000))
        self.assertEqual(C.limits(1000)["max_frames"], 1000)
        self.assertEqual(C.limits(100)["max_frames"], 100)
        with self.assertRaises(ValueError):
            C.limits(500)
        m = C.metrics()
        self.assertFalse(m["forced_action_layer"])
        self.assertFalse(m["online_learning"])

    def test_entry_registration_refuses_forced_action_and_foreign_limits(self):
        from . import entries
        root = Path(C.__file__).resolve().parents[2]
        base = dict(id=C.ID, transmitted_budget=100, g1_boundary_budget=100, exact_gt_limits=C.limits(100),
                    child_module=C.CHILD, edge_module=C.EDGE, live_metric_contract=C.metrics(),
                    exact_gt_sources={k: v for k, v in C.exact_sources(root).items()})
        tmp = Path(tempfile.mkdtemp(prefix="pilot_reg_"))
        for name, value in (("ok", base), ("forced", dict(base, forced_action={"mode_id": 11})),
                            ("limits", dict(base, exact_gt_limits=C.limits(1000)))):
            value = dict(value)
            value["registration_sha256"] = C.canonical(value)
            (tmp / f"{name}.json").write_text(json.dumps(value))
        self.assertEqual(entries.registration(tmp / "ok.json", root)["transmitted_budget"], 100)
        for name in ("forced", "limits"):
            with self.assertRaises(ValueError):
                entries.registration(tmp / f"{name}.json", root)


if __name__ == "__main__":
    unittest.main()
