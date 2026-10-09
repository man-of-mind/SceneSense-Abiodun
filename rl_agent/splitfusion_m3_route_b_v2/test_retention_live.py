"""Retention launcher: contract/plan binding, composed production input scopes, attempt-wide ordinal, drain, refusal.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v \
      rl_agent.splitfusion_m3_route_b_v2.test_retention_live
"""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
from rl_agent.splitfusion_live_parity_repair_v1.runtime import install_ue_perception
from rl_agent.splitfusion_ue_input_prep_v1.candidate import install_bgr2rgb_candidate
from rl_agent.splitfusion_ue_input_prep_v2 import candidate as V2
from rl_agent.splitfusion_paired_retention_v2 import fixtures as F, observer as OB, plan as RP, validate as V
from rl_agent.splitfusion_paired_retention_v2.test_observer import Guard

from . import contract as C, entries as EN, retention as R

ROOT = Path(__file__).resolve().parents[2]
S1, S2 = "00000000-0000-0000-0000-0000000000a1", "00000000-0000-0000-0000-0000000000b2"


def small_observer(planned_frames=12, target_records=3):
    plan = RP.build_plan(plan_id="launcher-test", planned_frames=planned_frames, target_records=target_records,
                         require_cuda_transfer=False)
    return R.AttemptRetentionObserver(RP.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=RP.plan_sha256(plan)))


def stream(n_decisions):
    """Policy decisions interleaved with holds; the controller session resets after 3 decisions (decision_seq -> 0)."""
    out, tensor = [], 0
    for i in range(n_decisions):
        session, seq = (S1, i) if i < 3 else (S2, i - 3)
        planned, prepared = F.make_decision(frame_id=1000 + 4 * i, tensor_seq=tensor, decision_seq=seq,
                                            session_uuid=session, q_e4=4000 + 37 * i)
        envelope, context = W.unpack_sfd4(prepared.wire)
        planned.plan = NS(tensor_seq=tensor, kind=NS(value="POLICY_DECISION"), frame_identity=envelope,
                          profile=NS(mode_id=11, q_e4=envelope.q_e4, action_id=None, profile_id=None))
        planned.verify = lambda **kwargs: None
        out.append((planned, envelope, context))
        hold = NS(reward_requested=False, plan=NS(kind=NS(value="POLICY_HOLD")), frame_id=1002 + 4 * i,
                  tensor_seq=tensor + 1)
        out.append((hold, None, None))
        tensor += 2
    return out


def drive(observer, frames, seen):
    """Run the REAL Run4FramePipelineV2.materialize on a thread named like the owned UE frame worker."""
    rng = np.random.default_rng(11)

    def work():
        for planned, envelope, context in frames:
            bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            radar = rng.random((4, 432, 768), dtype=np.float32)
            if envelope is None:            # a hold: input is prepared, nothing is a policy decision
                token = observer.begin(planned)
                observer.observe_input(B._prepare_live_input(bgr, radar, "cpu"), "cpu")
                observer.end(token, None)
                continue

            def prepare(profile, tensor, identity, _e=envelope):
                seen.append((tensor, V2.prepare_live_input_fused(bgr, radar, "cpu")))
                return NS(envelope=_e)
            pipeline = U.Run4FramePipelineV2(engine=mock.Mock(), continuous_ue=NS(prepare=prepare),
                                             provider=mock.Mock(), stream_id=context.stream_id, run_id=F.RUN_ID,
                                             cell_id=F.CELL_ID)
            pipeline.materialize(planned, frame_id=planned.frame_id, capture_wall_ns=envelope.capture_timestamp_ns,
                                 ego_pose=tuple(context.ego_world.values()),
                                 input_7ch=lambda: B._prepare_live_input(bgr, radar, "cpu"))
    thread = threading.Thread(target=work, name="route-b-split-front")
    return thread


class ContractTests(unittest.TestCase):
    def test_registered_plan_is_typical_loop_stride15(self):
        plan = C.retention_plan(C.FRAME_CEILING)
        self.assertEqual((plan["planned_policy_decisions"], plan["stride"], plan["offset"]), (1500, 15, 7))
        self.assertEqual(plan["ordinals"], list(range(7, 1500, 15)))
        self.assertEqual(plan["persistence"]["mode"], "POST_DRAIN")
        self.assertTrue(plan["require_cuda_transfer"])
        self.assertEqual(RP.canonical(plan), RP.canonical(C.retention_plan(C.FRAME_CEILING)))      # deterministic
        self.assertEqual(set(plan["independent_of"]) >= {"latency", "deadline_outcome", "quality", "action"}, True)

    def test_sources_lineage_and_registration_bind_retention(self):
        sources = C.own_sources(ROOT)
        for name in (*C.RETENTION_SOURCES, C.PACKAGE + "/retention.py", C.PACKAGE + "/entries.py"):
            self.assertIn(name, sources)
        digest = RP.plan_sha256(C.retention_plan(C.FRAME_CEILING))
        reg = C.retention_registration(digest)
        self.assertEqual((reg["plan_sha256"], reg["plan_file"], reg["directory"]),
                         (digest, C.RETENTION_PLAN_FILE, C.RETENTION_DIR))
        with self.assertRaises(ValueError):
            C.retention_registration(None)
        with self.assertRaises(ValueError):
            C.lineage(ROOT, "a" * 64, "b" * 64, C.FRAME_CEILING, "candidate", "v2", None)
        self.assertNotEqual(C.token("candidate"), "RUN5_FULL_LIVE_GC_PRODUCER_V2_VALIDATION_V1_CANDIDATE_V2_B1000_ONCE_EXECUTE_V1")
        self.assertNotEqual(C.ID, "RUN5_FULL_LIVE_GC_PRODUCER_V2_VALIDATION_V1")

    def test_load_observer_binds_digest_and_budget(self):
        plan = C.retention_plan(C.FRAME_CEILING)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            digest = RP.write_plan(out / C.RETENTION_PLAN_FILE, plan)
            reg = dict(transmitted_budget=C.FRAME_CEILING, paired_retention=C.retention_registration(digest))
            observer = R.load_observer(out, reg)
            self.assertIsInstance(observer, R.AttemptRetentionObserver)
            self.assertEqual(observer.plan_sha256, digest)
            with self.assertRaises(ValueError):
                R.load_observer(out, dict(reg, transmitted_budget=500))            # not this budget's rule
            with self.assertRaises(Exception):
                R.load_observer(out, dict(reg, paired_retention=C.retention_registration("0" * 64)))
            raw = (out / C.RETENTION_PLAN_FILE).read_bytes()
            (out / C.RETENTION_PLAN_FILE).write_bytes(raw.replace(b'"stride":15', b'"stride":14'))
            with self.assertRaises(Exception):
                R.load_observer(out, reg)                                           # tampered file


class ComposedScopeTests(unittest.TestCase):
    def test_production_scope_order_session_reset_holds_and_post_drain_archive(self):
        observer = small_observer(planned_frames=16, target_records=4)          # 8 planned decisions: ordinals 1,3,5,7
        frames, seen = stream(7), []
        with install_ue_perception(), install_bgr2rgb_candidate(), V2.install_fused_candidate(), \
                OB.install_retention(observer) as info:
            self.assertTrue(info["enabled"])
            thread = drive(observer, frames, seen)
            thread.start()
            thread.join()
        self.assertEqual(U.Run4FramePipelineV2.materialize.__qualname__, "Run4FramePipelineV2.materialize")   # restored
        self.assertEqual(observer.counters["policy_decisions_observed"], 7)     # attempt-wide, across the reset
        self.assertEqual(observer.counters["non_decision_frames"], 7)
        self.assertEqual(observer.counters["faults"], 0)
        self.assertIs(observer.owner_thread, thread)
        for tensor, baseline in seen:                                            # production tensor retained is bit-exact
            self.assertTrue(torch.equal(tensor, baseline))
        retained = {o: observer.records[o]["input"] for o in (1, 3, 5)}
        for ordinal, tensor in retained.items():
            self.assertIs(tensor, seen[ordinal][0])                              # the SAME object prepare() received
        with tempfile.TemporaryDirectory() as tmp:
            result = R.finalize_after_child(observer, Path(tmp), child_returncode=0, child_error=None)
            self.assertTrue(result["finalized"], result.get("error"))
            self.assertEqual(result["owner_thread"]["alive"], False)
            self.assertEqual(result["status_counts"], {"RETAINED_COMPLETE": 3, "NOT_REACHED": 1})
            manifest = V.verify_archive_v2(Path(tmp) / C.RETENTION_DIR, expected_manifest_sha256=result["manifest_sha256"],
                                           expected_plan_sha256=observer.plan_sha256)
            sessions = {r["join_key"]["session_uuid"] for r in manifest["records"]}
            self.assertEqual(sessions, {S1, S2})
            for record in manifest["records"]:
                array = np.load(Path(tmp) / C.RETENTION_DIR / record["input_file"])
                self.assertTrue(np.array_equal(array, seen[record["ordinal"]][1].numpy()))
            saved = json.loads((Path(tmp) / C.RETENTION_RESULT_FILE).read_bytes())
            self.assertEqual(saved["decision_path_bookkeeping"]["begin"]["n"], 14)

    def test_finalize_never_raises_or_writes_while_the_owner_is_alive(self):
        observer = small_observer()
        gate = threading.Event()
        frames, seen = stream(3), []
        with install_ue_perception(), install_bgr2rgb_candidate(), V2.install_fused_candidate(), \
                OB.install_retention(observer):
            thread = drive(observer, frames, seen)
            original = thread._target
            thread._target = lambda: (original(), gate.wait(10))
            thread.start()
            while observer.counters["policy_decisions_observed"] < 3:
                threading.Event().wait(0.01)
            with tempfile.TemporaryDirectory() as tmp:
                result = R.finalize_after_child(observer, Path(tmp), child_returncode=0, child_error=None)
                self.assertFalse(result["finalized"])
                self.assertIn("has not drained", result["error"])
                self.assertFalse((Path(tmp) / C.RETENTION_DIR).exists())
                self.assertEqual(result["child_returncode"], 0)
            gate.set()
            thread.join()

    def test_decision_path_does_no_io_copy_hash_or_sync(self):
        observer = small_observer(planned_frames=4, target_records=2)
        planned, prepared = F.make_decision(frame_id=8, tensor_seq=0, decision_seq=0, session_uuid=S1)
        tensor = F.shared_input()
        observer.start()
        with Guard():
            token = observer.begin(planned)
            observer.observe_input(tensor, "cpu")
            observer.end(token, prepared)
        self.assertEqual(observer.status[token["ordinal"]], "HELD")
        self.assertIs(observer.records[token["ordinal"]]["input"], tensor)

    def test_registration_check_refuses_missing_or_foreign_retention(self):
        source = Path(EN.__file__).read_text()
        self.assertIn("registration omits retention source", source)
        self.assertIn('install_retention(observer)', source)
        # the retention scope is the innermost context of the UE child (entered after the early-GT scope)
        self.assertLess(source.index("install_early_gt_ue(attempt_dir=attempt, registration=reg)"),
                        source.index("install_retention(observer):"))
        self.assertLess(source.index("result = CHILD.main(args)"), source.index("finalize_after_child(observer"))


if __name__ == "__main__":
    unittest.main()
