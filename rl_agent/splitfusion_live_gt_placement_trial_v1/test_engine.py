"""Actual local IPC/GT invariance and cleanup tests, never a performance run."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
from rl_agent.splitfusion_live_gt_optimization_v3.benchmark_offload import without_avo_clocks
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, memfd_transport as M
from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B, contract as H, scoring_packet as P
from rl_agent.splitfusion_live_gt_handoff_v1.test_engine import clone_inputs
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from . import client as C, contract, engine as N
from rl_agent.splitfusion_live_gt_placement_v1.placement import scheduling


def setup(root, label, mode, frames=2):
    owner = root / label
    owner.mkdir()
    targets, evidence = owner / "targets", owner / "evidence"
    targets.mkdir()
    evidence.mkdir()
    broker = B.Server(targets / H.SOCKET_NAME, binding=H.binding(), max_frames=frames,
                      max_jobs=frames).start()
    try:
        child = C.Client(mode=mode, placement_cpu=4 if mode == "p4" else None, attempt_dir=owner / "worker",
            catalogue_dir=root / "catalogue", max_frames=frames, expected_sources=W.sources(),
            expected_bootstrap_sources=contract.sources(), allow_replay_clock=True,
            source_attempt_name=root.name,
            handoff_config=dict(directory=str(targets), evidence_root=str(evidence)))
    except BaseException:
        broker.close()
        raise
    return child, broker, targets, evidence


def state():
    return (scheduling(), dict(os.environ), gc.isenabled(), gc.get_threshold(),
            gc.get_freeze_count(), tuple(gc.callbacks))


class FreshWorkerTests(unittest.TestCase):
    fault_mode = "p4"
    def check_scope(self, root, label, cpu, original, requests):
        scope = json.loads((root / label / "worker/engine/PLACEMENT_SCOPE.json").read_bytes())
        self.assertTrue(scope["closed"] and scope["restored"])
        self.assertEqual(scope["requested_cpu"], cpu)
        self.assertEqual(scope["request_count"], requests)
        self.assertEqual(scope["archive_returncode"], 0)
        self.assertEqual(scope["original_scheduling"], original)
        self.assertNotIn("native_pool_startup", scope)
        self.assertFalse(scope["native_pool_preparation"])
        self.assertFalse(scope["helper_affinity_mutated"])
        for phase in ("archive_start", "archive_end"):
            self.assertTrue(all(row["affinity"] == original["affinity"]
                for row in scope[phase]["threads"].values()))
        self.assertEqual(scope["set_calls"], [scope["main_tid"]] if cpu is not None else [])
        self.assertEqual(scope["restore_calls"], [scope["main_tid"]] if cpu is not None else [])
        self.assertNotIn(scope["archive_start"]["pid"], scope["set_calls"])
        gc_scope = json.loads((root / label / "worker/engine/GC_SCOPE.json").read_bytes())
        self.assertEqual(gc_scope["mode"], "control")
        self.assertEqual(gc_scope["freeze_calls"], 0)
        self.assertTrue(gc_scope["settings_restored"])
        self.assertEqual(gc_scope["startup_collection"]["generation"], 2)
        self.assertEqual(gc_scope["startup_collection"]["stats_delta"][2]["collections"], 1)
        return scope

    def test_two_placements_keep_native_targets_avo_and_scorer_identical(self):
        import numpy as np
        before = state()
        cpus = before[0]["affinity"]
        with F.fixture() as (root, initial, radar):
            proofs = []
            for label, mode in (("control", "control"), ("p4", "p4")):
                cpu = 4 if mode == "p4" else None
                child, broker, targets, evidence = setup(root, label, mode)
                kwargs = clone_inputs(initial)
                try:
                    ready = child.ready["process_context"]
                    allowed = [cpu] if cpu is not None else before[0]["affinity"]
                    self.assertEqual(ready["scheduling"]["affinity"], allowed)
                    self.assertEqual(child.expected_context["affinity"], allowed)
                    self.assertEqual(ready["runtime_gc"]["mode"], "control")
                    self.assertEqual(ready["runtime_placement"]["trial_mode"], mode)
                    outcomes = []
                    for index in range(2):
                        if index:
                            F.advance(kwargs, replenish=True)
                        result = child.run(directory=targets, kwargs=kwargs, radar=radar,
                            sent_raw_ns=W.raw(), sync_config=F.sync_config(evidence),
                            replay_observed_raw_ns=123+index)
                        context = result["process_context"]["runtime_placement"]
                        self.assertIn(context["cpu_now"], allowed)
                        for tid, thread in context["current_threads"].items():
                            self.assertEqual(thread["affinity"], allowed if int(tid) == child.process.pid
                                             else before[0]["affinity"])
                        self.assertEqual(len(context["request_samples"]), 1)
                        row = context["request_samples"][0]
                        self.assertEqual(row["request_index"], index)
                        self.assertIn(row["before"]["cpu"], allowed)
                        self.assertIn(row["after"]["cpu"], allowed)
                        self.assertTrue(all(value >= 0 for value in row["delta"].values()))
                        packet = P.decode_packet(broker.take(result["identity"]),
                            expected_identity=result["identity"], binding=H.binding())
                        outcomes.append((result, packet))
                    closed = child.close()
                    self.assertTrue(closed["timing"]["archive_drained"])
                    self.assertEqual(closed["process_context"]["scheduling"], before[0])
                    self.assertEqual(child.expected_context, before[0])
                    native = json.loads((root / label / "worker/engine/HANDOFF_DRAIN.json").read_bytes())
                    self.assertEqual(child.handoff_close["timing"], native)
                    self.assertIn("runtime_gc_scope", closed["timing"])
                    self.assertIn("runtime_placement_scope", closed["timing"])
                    self.assertEqual(child.extended_handoff_close, closed)
                    import hashlib
                    receipt = Path(child.extended_close_record["path"]).read_bytes()
                    self.assertEqual(hashlib.sha256(receipt).hexdigest(), child.extended_close_record["sha256"])
                    self.assertEqual(json.loads(receipt)["full_close"], closed)
                    self.check_scope(root, label, cpu, before[0], 2)
                    arm = []
                    for index, (result, packet) in enumerate(outcomes):
                        native = F.T.verify_full_inputs(targets, expected_identity=result["identity"])
                        avo = json.loads(Path(result["record"]["path"]).read_bytes())
                        scored = E.evaluate_prepared(identity=dict(frame_id=result["identity"]["frame_id"],
                            session_uuid="placement-parity", decision_seq=index, tensor_seq=index, ticket_seq=index),
                            prepared=packet["prepared"], predicted_labels=np.zeros((432, 768), np.uint8),
                            predictions=[], avo=packet["avo"])
                        self.assertEqual(native.prepared.content_sha256, packet["prepared"].content_sha256)
                        arm.append(dict(content=native.prepared.content_sha256,
                            payloads=native.manifest["payloads"], provenance=native.inputs.provenance,
                            source_tree=F.tree(Path(result["record"]["complete_sources"]["directory"])),
                            avo=without_avo_clocks(avo), measurement=scored.measurement, evidence=scored.evidence))
                    proofs.append(arm)
                finally:
                    if not child.closed:
                        child.abort()
                    self.assertFalse(broker.close()["worker_alive"])
            self.assertEqual(proofs[0], proofs[1])
        self.assertEqual(state(), before)

    def test_handled_failure_drains_and_restores_before_reply(self):
        before = state()
        cpu = 4 if self.fault_mode == "p4" else None
        with F.fixture() as (root, kwargs, radar):
            child, broker, targets, evidence = setup(root, "failed", self.fault_mode, frames=1)
            kwargs["frozen_world"].population_evidence["registry_complete"] = False
            try:
                with self.assertRaises(W.WorkerError) as caught:
                    child.run(directory=targets, kwargs=kwargs, radar=radar, sent_raw_ns=W.raw(),
                        sync_config=F.sync_config(evidence), replay_observed_raw_ns=123)
                context = caught.exception.worker_result["process_context"]
                self.assertTrue(context["runtime_placement"]["restored"])
                self.assertEqual(context["scheduling"], before[0])
                self.assertTrue(child.closed and child.failed)
                self.assertTrue(child.handoff_close["timing"]["archive_drained"])
                native = json.loads((root / "failed/worker/engine/HANDOFF_DRAIN.json").read_bytes())
                self.assertEqual(child.handoff_close["timing"], native)
                self.assertTrue(child.extended_close_record)
                self.check_scope(root, "failed", cpu, before[0], 1)
            finally:
                if not child.closed:
                    child.abort()
                broker.close()
        self.assertEqual(state(), before)

    def test_malformed_snapshot_records_failure_then_restores(self):
        before = state()
        cpu = 4 if self.fault_mode == "p4" else None
        with F.fixture() as (root, kwargs, _radar):
            child, broker, targets, _evidence = setup(root, "malformed", self.fault_mode, frames=1)
            descriptor, size = M.create([b"X" * 100])
            try:
                with self.assertRaisesRegex(W.WorkerError, "snapshot magic mismatch"):
                    child._exchange(dict(command="RUN", directory=str(targets), sent_raw_ns=W.raw(),
                        expected_identity=kwargs["identity"], snapshot_bytes=size), descriptor=descriptor)
                scope = self.check_scope(root, "malformed", cpu, before[0], 1)
                self.assertEqual(scope["request_samples"][0]["outcome"], "SNAPSHOT_FAILED")
                self.assertTrue(child.closed and child.failed)
            finally:
                os.close(descriptor)
                if not child.closed:
                    child.abort()
                broker.close()
        self.assertEqual(state(), before)


class ControlFailureTests(FreshWorkerTests):
    fault_mode = "control"
    test_two_placements_keep_native_targets_avo_and_scorer_identical = None


def direct_failure_case(name):
    before = state()
    with F.fixture() as (root, _kwargs, _radar):
        targets, evidence = root / "targets", root / "evidence"
        targets.mkdir()
        evidence.mkdir()
        options = dict(attempt_dir=root / "owner", catalogue_dir=root / "catalogue", max_frames=1,
            expected_sources=W.sources(), allow_replay_clock=True, source_attempt_name=root.name,
            handoff_config=dict(directory=str(targets), evidence_root=str(evidence)),
            placement_mode="p4", placement_original_context=before[0])
        if name == "constructor":
            actual = os.sched_setaffinity
            calls = []
            def fail_once(tid, mask):
                actual(tid, mask)
                calls.append(tid)
                if len(calls) == 1:
                    raise RuntimeError("injected partial placement construction")
            with mock.patch.object(os, "sched_setaffinity", side_effect=fail_once):
                try:
                    N.Engine(**options)
                except RuntimeError as error:
                    assert "partial placement construction" in str(error)
                else:
                    raise AssertionError("constructor fault hidden")
        elif name == "drain":
            engine = N.Engine(**options)
            original = engine.archiver.close
            def fail_after_drain():
                original()
                raise OSError("injected archive close failure")
            with mock.patch.object(engine.archiver, "close", side_effect=fail_after_drain):
                try:
                    engine.close()
                except OSError as error:
                    assert "archive close failure" in str(error)
                else:
                    raise AssertionError("native archive fault hidden")
            assert engine.archiver.process.poll() == 0
        else:
            raise AssertionError("unknown case")
        scope = json.loads((root / "owner/PLACEMENT_SCOPE.json").read_bytes())
        assert scope["restored"] and scope["closed"] and scope["archive_returncode"] == 0
        assert scheduling() == before[0]
    assert state() == before


class DirectFailureTests(unittest.TestCase):
    def run_fresh(self, case):
        env = dict(os.environ, **contract.ENVIRONMENT, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-c",
            "from rl_agent.splitfusion_live_gt_placement_trial_v1.test_engine import direct_failure_case; "
            + "direct_failure_case(" + repr(case) + ")"], cwd=Path(__file__).resolve().parents[2],
            env=env, capture_output=True, text=True, timeout=50)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_partial_constructor_fault_restores_and_reaps_archive(self):
        self.run_fresh("constructor")

    def test_archive_close_fault_preserves_error_and_restores(self):
        self.run_fresh("drain")


if __name__ == "__main__":
    unittest.main()
