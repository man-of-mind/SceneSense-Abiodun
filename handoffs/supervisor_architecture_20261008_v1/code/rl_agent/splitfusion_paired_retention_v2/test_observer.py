"""Observer: ownership, reentrancy, duplicates, bounds, no hot-path I/O or sync, spill writer, real seam."""
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
from rl_agent.splitfusion_ue_input_prep_v2 import candidate as V2
from . import fixtures as F, observer as OB, plan as P, validate as V

SESSION = "00000000-0000-0000-0000-000000000001"


def config(**kwargs):
    kwargs.setdefault("planned_frames", 24)
    kwargs.setdefault("target_records", 3)
    plan = P.build_plan(plan_id="obs", **kwargs)
    return P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))


def decisions(n, start=0, session=SESSION):
    return [("decision", *F.make_decision(frame_id=100 + 4 * i, tensor_seq=2 * i, decision_seq=i,
                                          session_uuid=session, q_e4=4000 + i), None)
            for i in range(start, start + n)]


class Guard:
    """Patch every I/O, copy, hash and sync entry point to raise (optionally only on one thread)."""
    TARGETS = [(builtins, "open"), (io, "open"), (os, "open"), (os, "write"), (os, "fsync"),
               (Path, "open"), (Path, "write_bytes"), (np, "save"), (torch, "save"),
               (torch.cuda, "synchronize"), (torch.Tensor, "clone"), (torch.Tensor, "numpy"),
               (torch.Tensor, "cpu"), (torch.Tensor, "to"), (torch.Tensor, "copy_"),
               (torch.Tensor, "contiguous"), (hashlib, "sha256"), (json, "dumps")]

    def __init__(self, thread=None):
        self.thread, self.patches = thread, []

    def __enter__(self):
        for owner, name in self.TARGETS:
            original = getattr(owner, name)
            def guarded(*a, _o=original, _n=name, **k):
                if self.thread is None or threading.get_ident() == self.thread:
                    raise AssertionError(f"decision-path call to {_n}")
                return _o(*a, **k)
            patch = mock.patch.object(owner, name, guarded)
            patch.start()
            self.patches.append(patch)
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()


class ObserverTests(unittest.TestCase):
    def test_hot_path_has_no_io_copy_hash_or_device_sync(self):
        observer = OB.make_observer(P.RetentionConfigV2(enabled=True, plan=P.build_plan(plan_id="x", planned_frames=1000),
                                    plan_sha256=P.plan_sha256(P.build_plan(plan_id="x", planned_frames=1000))))
        events, tensor = list(F.route(999)), F.shared_input()
        with Guard():
            F.drive(observer, events, tensor)
            for i in range(50):
                observer.notify_feedback(1000 + i)
        self.assertEqual(observer.counters["faults"], 0)
        held = [r for r in observer.records.values() if observer.status[r["ordinal"]] == "HELD"]
        self.assertGreaterEqual(len(held), 95)
        self.assertTrue(all(r["input"] is tensor for r in held))           # by reference, never copied
        self.assertTrue(all(type(r["prepared"]["wire"]) is bytes for r in held))

    def test_foreign_thread_is_a_fault_not_an_exception(self):
        observer = OB.make_observer(config())
        events = decisions(3)
        token = observer.begin(events[0][1])
        observer.end(token, None)
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault("t", observer.begin(events[1][1])))
        worker.start(); worker.join()
        self.assertIsNone(result["t"])
        self.assertEqual(observer.counters["policy_decisions_observed"], 1)   # foreign call takes no ordinal
        self.assertEqual(observer.fault_details[-1]["reason"], "FOREIGN_THREAD")

    def test_reentrancy_and_duplicate_identity_are_faults(self):
        observer = OB.make_observer(config(planned_frames=8, target_records=4))   # ordinals 0..3 all sampled
        events, tensor = decisions(4), F.shared_input()
        first = observer.begin(events[0][1])
        observer.observe_input(tensor, "cuda:0")
        nested = observer.begin(events[1][1])                                     # reentrant
        self.assertIsNone(nested)
        observer.end(first, events[0][2])
        self.assertEqual(observer.status[0], "INCOMPLETE_REENTRANT")
        self.assertEqual(observer.status[1], "FAULT_REENTRANT")
        self.assertIsNone(first["input"])                                        # reference released
        dup = observer.begin(events[0][1])                                        # same identity again
        self.assertIsNone(dup)
        self.assertEqual(observer.status[2], "FAULT_DUPLICATE_IDENTITY")
        token = observer.begin(events[3][1])
        observer.observe_input(tensor, "cuda:0")
        observer.end(token, events[3][2])
        self.assertEqual(observer.status[3], "HELD")
        self.assertEqual(observer.counters["faults"], 2)

    def test_incomplete_classifications_release_references(self):
        observer = OB.make_observer(config(planned_frames=10, target_records=5))
        events, tensor = decisions(5), F.shared_input()
        cases = [
            lambda t: observer.end(t, None),                                                      # production exception
            lambda t: observer.end(t, events[1][2]),                                              # no input observed
            lambda t: (observer.observe_input(torch.zeros(1, 7, 448, 768, dtype=torch.float16), "cuda:0"),
                       observer.end(t, events[2][2])),                                            # bad tensor
            lambda t: (observer.observe_input(tensor, "cuda:0"), observer.end(t, NS(wire=b""))),  # no wire
            lambda t: (observer.observe_input(tensor, "cuda:0"), observer.observe_input(tensor, "cuda:0"),
                       observer.end(t, events[4][2])),                                            # two inputs
        ]
        for (kind, planned, prepared, _), case in zip(events, cases):
            case(observer.begin(planned))
        self.assertEqual([observer.status[i] for i in range(5)],
                         ["INCOMPLETE_PRODUCTION_EXCEPTION", "INCOMPLETE_NO_INPUT_OBSERVED",
                          "INCOMPLETE_BAD_INPUT_TENSOR", "INCOMPLETE_NO_WIRE", "INCOMPLETE_MULTIPLE_INPUTS"])
        self.assertEqual(observer.host_bytes(), 0)
        self.assertTrue(all(r["input"] is None for r in observer.records.values()))

    def test_budget_exhaustion_is_counted_never_raised(self):
        plan = P.build_plan(plan_id="b", planned_frames=24, target_records=6, max_host_bytes=2 * P.INPUT_BYTES + 4096)
        observer = OB.make_observer(P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan)))
        F.drive(observer, decisions(12), F.shared_input())
        statuses = [observer.status[o] for o in plan["ordinals"]]
        self.assertEqual(statuses.count("HELD"), 2)
        self.assertEqual(statuses.count("BUDGET_EXHAUSTED"), 4)
        self.assertLessEqual(observer.peak_host_bytes, plan["bounds"]["max_host_bytes"])
        self.assertEqual(observer.counters["policy_decisions_observed"], 12)    # run continued

    def spill_run(self, directory, *, pace_s):
        plan = P.build_plan(plan_id="spill", planned_frames=80, target_records=8, persistence="BACKGROUND_SPILL",
                            spill_queue_records=2, spill_lag_decisions=1, min_free_disk_bytes=0)
        cfg = P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))
        self.assertLess(plan["bounds"]["max_host_bytes"], 8 * P.INPUT_BYTES)    # window, not the whole sample
        observer = OB.make_observer(cfg, directory)
        observer.start()
        tensors = [torch.full(P.INPUT_SHAPE, float(i)) for i in range(40)]
        events = decisions(40)
        import time
        with Guard(thread=threading.get_ident()):          # the writer thread may do I/O; the frame worker may not
            for i, (_, planned, prepared, _) in enumerate(events):
                token = observer.begin(planned)
                observer.observe_input(tensors[i], "cuda:0")
                observer.end(token, prepared)
                if i % 5 == 1:
                    observer.notify_feedback(planned.frame_id)
                time.sleep(pace_s)                         # decision cadence (live: ~200 ms per decision)
        manifest = observer.finalize(workers=F.worker_done())
        self.assertLessEqual(manifest["host_bytes"]["peak"], plan["bounds"]["max_host_bytes"])
        self.assertEqual(manifest["host_bytes"]["at_close"], 0)
        return cfg, manifest

    def test_background_spill_keeps_ram_bounded_and_writes_off_the_frame_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "retention"
            cfg, manifest = self.spill_run(directory, pace_s=0.08)
            self.assertEqual(manifest["status_counts"], {"RETAINED_COMPLETE": 8})
            V.verify_archive_v2(directory, expected_manifest_sha256=manifest["manifest_sha256"],
                                expected_plan_sha256=cfg.plan_sha256)
            for entry in manifest["records"]:          # exact tensor bytes of that decision
                array = np.load(directory / entry["input_file"])
                self.assertEqual(float(array[0, 0, 0, 0]), float(entry["ordinal"]))

    def test_spill_backpressure_becomes_counted_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "retention"
            cfg, manifest = self.spill_run(directory, pace_s=0.0)   # no cadence: writer cannot keep up
            counts = manifest["status_counts"]
            self.assertEqual(sum(counts.values()), 8)
            self.assertEqual(set(counts) - {"RETAINED_COMPLETE", "BUDGET_EXHAUSTED"}, set())
            if counts.get("BUDGET_EXHAUSTED"):
                with self.assertRaisesRegex(V.RetentionAdmissionError, "incomplete"):
                    V.verify_archive_v2(directory, expected_manifest_sha256=manifest["manifest_sha256"],
                                        expected_plan_sha256=cfg.plan_sha256)
                V.verify_archive_v2(directory, expected_manifest_sha256=manifest["manifest_sha256"],
                                    expected_plan_sha256=cfg.plan_sha256, strict=False)

    def test_finalize_requires_drained_workers(self):
        observer = OB.make_observer(config())
        F.drive(observer, decisions(12), F.shared_input())
        with tempfile.TemporaryDirectory() as tmp:
            for workers in ([], [NS(is_alive=lambda: True)], [NS()]):
                with self.assertRaises(ValueError):
                    observer.finalize(Path(tmp) / "r", workers=workers)
            self.assertFalse((Path(tmp) / "r").exists())

    # -------------------------------------------------------------- real seam (CPU)
    def test_fused_seam_is_bit_exact_and_retains_the_production_tensor(self):
        rng = np.random.default_rng(5)
        bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        radar = rng.random((4, 432, 768), dtype=np.float32)
        seen = []
        replacement = OB.retained_fused_packing(V2.prepare_live_input_fused, lambda t, d: seen.append(t) or t)
        baseline = V2.prepare_live_input_fused(bgr, radar, "cpu")
        result = replacement(bgr, radar, "cpu")
        self.assertTrue(torch.equal(baseline, result))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].data_ptr(), result.data_ptr())
        with self.assertRaisesRegex(ValueError, "unavailable|instrumented"):
            OB.retained_fused_packing(replacement, lambda t, d: t)                  # already instrumented

    def test_install_scope_wraps_real_materialize_and_restores(self):
        plan = P.build_plan(plan_id="seam", planned_frames=2, target_records=1, require_cuda_transfer=False)
        observer = OB.make_observer(P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan)))
        planned, prepared = F.make_decision(frame_id=105, tensor_seq=0, decision_seq=0, session_uuid=SESSION)
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        envelope, context = W.unpack_sfd4(prepared.wire)
        frame_plan = NS(tensor_seq=0, kind=NS(value="POLICY_DECISION"), frame_identity=envelope,
                        profile=NS(mode_id=11, q_e4=envelope.q_e4, action_id=None, profile_id=None))
        planned.plan, planned.verify = frame_plan, (lambda **kwargs: None)
        pipeline = U.Run4FramePipelineV2(engine=mock.Mock(),
            continuous_ue=NS(prepare=lambda *args: NS(envelope=envelope)), provider=mock.Mock(),
            stream_id=context.stream_id, run_id=F.RUN_ID, cell_id=F.CELL_ID)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((4, 432, 768), dtype=np.float32)
        before_materialize = U.Run4FramePipelineV2.materialize
        with mock.patch.object(B, "_prepare_live_input", V2.prepare_live_input_fused):
            with OB.install_retention(observer) as info:
                self.assertTrue(info["enabled"])
                self.assertIsNot(B._prepare_live_input, V2.prepare_live_input_fused)
                with self.assertRaisesRegex(ValueError, "nested"):
                    with OB.install_retention(OB.make_observer(P.RetentionConfigV2(
                            enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan)))):
                        pass
                result = pipeline.materialize(planned, frame_id=105, capture_wall_ns=envelope.capture_timestamp_ns,
                    ego_pose=tuple(context.ego_world.values()),
                    input_7ch=lambda: B._prepare_live_input(bgr, radar, "cpu"))
                self.assertEqual(result.envelope, envelope)
            self.assertIs(B._prepare_live_input, V2.prepare_live_input_fused)
        self.assertIs(U.Run4FramePipelineV2.materialize, before_materialize)
        self.assertEqual(observer.status[0], "HELD")
        self.assertEqual(observer.records[0]["prepared"]["wire"], result.wire)
        self.assertEqual(observer.counters["faults"], 0)

    def test_scope_requires_the_installed_fused_candidate_and_preserves_production_errors(self):
        cfg = config(planned_frames=2, target_records=1)
        with self.assertRaisesRegex(ValueError, "fused input candidate"):
            with OB.install_retention(OB.make_observer(cfg)):
                pass
        observer = OB.make_observer(cfg)
        sentinel = RuntimeError("production failure")
        planned, _ = F.make_decision(frame_id=105, tensor_seq=0, decision_seq=0, session_uuid=SESSION)
        with mock.patch.object(B, "_prepare_live_input", V2.prepare_live_input_fused), \
                mock.patch.object(U.Run4FramePipelineV2, "materialize", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as raised:
                with OB.install_retention(observer):
                    U.Run4FramePipelineV2.materialize(NS(), planned)
            self.assertIs(raised.exception, sentinel)
        self.assertEqual(observer.status[0], "INCOMPLETE_PRODUCTION_EXCEPTION")


if __name__ == "__main__":
    unittest.main()
