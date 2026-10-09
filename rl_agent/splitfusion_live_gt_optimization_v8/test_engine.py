"""Independent integration checks of the real CPU GT worker, no live services."""
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
from . import client as C, control_client as CC, contract, control_contract, engine as N


def state():
    return (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count(), tuple(gc.callbacks))


def setup_client(root, label, kind, binding, frames=2):
    owner = root/label
    owner.mkdir()
    targets, evidence = owner/"targets", owner/"evidence"
    targets.mkdir(); evidence.mkdir()
    broker = B.Server(targets/H.SOCKET_NAME, binding=H.binding(), max_frames=frames, max_jobs=frames).start()
    try:
        client = kind(attempt_dir=owner/"worker", catalogue_dir=root/"catalogue", max_frames=frames,
            expected_sources=W.sources(), expected_bootstrap_sources=binding.sources(),
            allow_replay_clock=True, source_attempt_name=root.name,
            handoff_config=dict(directory=str(targets), evidence_root=str(evidence)))
    except BaseException:
        broker.close()
        raise
    return client, broker, targets, evidence


def check_scope(test, path, mode, *, native_close_succeeded=True):
    scope = json.loads(path.read_bytes())
    test.assertEqual(scope["mode"], mode)
    test.assertTrue(scope["closed"] and scope["settings_restored"])
    test.assertTrue(scope["enabled"])
    test.assertEqual(scope["thresholds"], [700, 10, 10])
    test.assertEqual(scope["freeze_count"], 0)
    test.assertEqual(scope["unfreeze_calls"], int(mode == "freeze"))
    test.assertFalse(scope["callback_installed"])
    test.assertFalse(scope["per_request_manual_collection"])
    test.assertEqual(scope["dropped"], 0)
    test.assertEqual(scope["callback_errors"], 0)
    return scope


class FreshWorkerTests(unittest.TestCase):
    def test_two_real_fresh_workers_match_all_native_targets_avo_and_scorer(self):
        import numpy as np
        before = state()
        with F.fixture() as (root, initial, radar):
            proofs = {}
            for label, kind, binding, mode in (("control", CC.Client, control_contract, "control"),
                    ("treatment", C.Client, contract, "freeze")):
                kwargs = clone_inputs(initial)
                child, broker, targets, evidence = setup_client(root, label, kind, binding)
                results, packets = [], []
                try:
                    ready = child.ready["process_context"]["runtime_gc"]
                    self.assertEqual(ready["mode"], mode)
                    self.assertEqual(ready["worker_pid"], child.process.pid)
                    self.assertEqual(ready["freeze_calls"], int(mode == "freeze"))
                    self.assertEqual(ready["processed_at_freeze"], 0)
                    self.assertGreater(ready["initial_frozen_count"], 0) if mode == "freeze" else self.assertEqual(ready["freeze_count"], 0)
                    for index in range(2):
                        if index:
                            F.advance(kwargs, replenish=True)
                        result = child.run(directory=targets, kwargs=kwargs, radar=radar, sent_raw_ns=W.raw(),
                            sync_config=F.sync_config(evidence), replay_observed_raw_ns=123+index)
                        self.assertEqual(result["status"], "OK", result.get("error"))
                        self.assertEqual(result["processed"], index+1)
                        self.assertEqual(result["worker"]["pid"], child.process.pid)
                        gc_state = result["process_context"]["runtime_gc"]
                        self.assertEqual(gc_state["processed"], index+1)
                        self.assertEqual(gc_state["freeze_calls"], int(mode == "freeze"))
                        self.assertTrue(gc_state["enabled"])
                        packets.append(P.decode_packet(broker.take(result["identity"]),
                            expected_identity=result["identity"], binding=H.binding()))
                        results.append(result)
                    closed = child.close()
                    self.assertEqual(child.process.poll(), 0)
                    self.assertTrue(closed["timing"]["archive_drained"])
                    self.assertEqual(closed["timing"]["archive"]["completed"], 2)
                    self.assertEqual(closed["process_context"]["runtime_gc"]["freeze_count"], 0)
                    check_scope(self, root/label/"worker/engine/GC_SCOPE.json", mode)
                    proofs[label] = []
                    for index, (result, packet) in enumerate(zip(results, packets)):
                        native = F.T.verify_full_inputs(targets, expected_identity=result["identity"])
                        self.assertEqual(native.prepared.content_sha256, packet["prepared"].content_sha256)
                        avo = json.loads(Path(result["record"]["path"]).read_bytes())
                        self.assertEqual(avo, packet["avo_document"])
                        scored = E.evaluate_prepared(identity=dict(frame_id=result["identity"]["frame_id"],
                                session_uuid="gc-parity", decision_seq=index, tensor_seq=index, ticket_seq=index),
                            prepared=packet["prepared"], predicted_labels=np.zeros((432, 768), np.uint8),
                            predictions=[], avo=packet["avo"])
                        proofs[label].append(dict(content=native.prepared.content_sha256,
                            payloads=native.manifest["payloads"], provenance=native.inputs.provenance,
                            source_tree=F.tree(Path(result["record"]["complete_sources"]["directory"])),
                            avo=without_avo_clocks(avo), measurement=scored.measurement, evidence=scored.evidence))
                finally:
                    if not child.closed:
                        child.abort()
                    self.assertFalse(broker.close()["worker_alive"])
            self.assertEqual(proofs["control"], proofs["treatment"])
        self.assertEqual(state(), before)

    def test_handled_source_failure_drains_archive_restores_gc_and_preserves_error(self):
        before = state()
        with F.fixture() as (root, kwargs, radar):
            child, broker, targets, evidence = setup_client(root, "failed", C.Client, contract, frames=1)
            kwargs["frozen_world"].population_evidence["registry_complete"] = False
            try:
                with self.assertRaises(W.WorkerError) as caught:
                    child.run(directory=targets, kwargs=kwargs, radar=radar, sent_raw_ns=W.raw(),
                        sync_config=F.sync_config(evidence), replay_observed_raw_ns=123)
                self.assertEqual(caught.exception.worker_result["status"], "FAILED")
                self.assertTrue(child.closed and child.failed)
                self.assertIsNotNone(child.process.poll())
                self.assertIsNotNone(child.handoff_close)
                self.assertTrue(child.handoff_close["timing"]["archive_drained"])
                self.assertEqual(child.handoff_close["timing"]["archive"]["completed"], 1)
                check_scope(self, root/"failed/worker/engine/GC_SCOPE.json", "freeze")
                self.assertEqual(broker.stats()["admitted"], 0)
            finally:
                if not child.closed:
                    child.abort()
                broker.close()
        self.assertEqual(state(), before)

    def test_malformed_snapshot_unfreezes_and_reaps_before_client_error(self):
        before = state()
        with F.fixture() as (root, kwargs, _radar):
            child, broker, targets, evidence = setup_client(root, "malformed", C.Client, contract, frames=1)
            descriptor, size = M.create([b"X"*100])
            try:
                with self.assertRaisesRegex(W.WorkerError, "snapshot magic mismatch"):
                    child._exchange(dict(command="RUN", directory=str(targets), sent_raw_ns=W.raw(),
                        expected_identity=kwargs["identity"], snapshot_bytes=size), descriptor=descriptor)
                self.assertTrue(child.closed and child.failed)
                self.assertIsNotNone(child.process.poll())
                check_scope(self, root/"malformed/worker/engine/GC_SCOPE.json", "freeze")
                drain = json.loads((root/"malformed/worker/engine/HANDOFF_DRAIN.json").read_bytes())
                self.assertTrue(drain["archive_drained"])
                self.assertEqual(drain["archive"]["completed"], 0)
            finally:
                os.close(descriptor)
                if not child.closed:
                    child.abort()
                broker.close()
        self.assertEqual(state(), before)


def direct_failure_case(name):
    """Run each process-global GC failure case in its own fresh interpreter."""
    before = state()
    with F.fixture() as (root, _kwargs, _radar):
        targets, evidence = root/"targets", root/"evidence"
        targets.mkdir(); evidence.mkdir()
        options = dict(attempt_dir=root/"owner", catalogue_dir=root/"catalogue", max_frames=1,
            expected_sources=W.sources(), allow_replay_clock=True, source_attempt_name=root.name,
            handoff_config=dict(directory=str(targets), evidence_root=str(evidence)))
        if name == "constructor":
            from rl_agent.splitfusion_live_gt_handoff_v1 import engine as HEngine
            actual_factory, made = HEngine.ArchiveClient, []
            def record_factory(**kwargs):
                child = actual_factory(**kwargs)
                made.append(child)
                return child
            with mock.patch.object(HEngine, "ArchiveClient", side_effect=record_factory), \
                    mock.patch.object(N, "GCPolicy", side_effect=ValueError("injected policy construction failure")):
                try:
                    N.Engine(**options)
                except ValueError as exc:
                    assert str(exc) == "injected policy construction failure"
                else:
                    raise AssertionError("constructor fault was hidden")
            assert len(made) == 1 and made[0].process.poll() == 0 and made[0].closed
        elif name in ("drain", "dual_failure"):
            engine = N.Engine(**options)
            assert gc.get_freeze_count() > 0
            original = engine.archiver.close
            def drain_failure():
                original()
                raise OSError("injected archive close failure after reap")
            try:
                if name == "dual_failure":
                    # The real policy must detect the drift, restore automatic
                    # GC, and report its own failure without hiding disk failure.
                    gc.set_threshold(701, 10, 10)
                with mock.patch.object(engine.archiver, "close", side_effect=drain_failure):
                    try:
                        engine.close()
                    except OSError as exc:
                        assert str(exc) == "injected archive close failure after reap"
                        if name == "dual_failure":
                            assert exc.__cause__ is not None and "GC settings drifted" in str(exc.__cause__)
                    else:
                        raise AssertionError("archive close failure was hidden")
                assert engine.archiver.process.poll() == 0
                scope = json.loads((root/"owner/GC_SCOPE.json").read_bytes())
                assert scope["settings_restored"] and scope["closed"] and scope["freeze_count"] == 0
                assert engine.gc_scope_record["native_close_succeeded"] is False
                if name == "dual_failure":
                    assert "injected archive close failure" in scope["native_close_error"]
                    assert "GC settings drifted" in scope["cleanup_error"]
            finally:
                if not engine.gc_policy.closed:
                    engine.gc_policy.close()
                if not engine.archiver.closed:
                    engine.archiver.abort()
        else:
            raise AssertionError("unknown fresh failure case")
    assert state() == before, "parent GC settings/callback ownership not restored"


class DirectFailureTests(unittest.TestCase):
    def run_fresh(self, name):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        result = subprocess.run(["/usr/bin/python3", "-c",
            "from rl_agent.splitfusion_live_gt_optimization_v8.test_engine import direct_failure_case; direct_failure_case("+repr(name)+")"],
            cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_constructor_policy_failure_reaps_actual_archive_child(self):
        self.run_fresh("constructor")

    def test_native_close_failure_still_restores_gc_without_false_success(self):
        self.run_fresh("drain")

    def test_native_error_is_not_masked_by_simultaneous_gc_policy_error(self):
        self.run_fresh("dual_failure")


if __name__ == "__main__":
    unittest.main()
