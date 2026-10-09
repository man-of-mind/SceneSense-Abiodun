"""Real exact GT, local handoff and durable archive composition; CPU only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
from rl_agent.splitfusion_live_gt_optimization_v3.benchmark_offload import without_avo_clocks
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_optimization_v6.engine import Engine as RetainedEngine
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
from . import archive as A, broker as B, contract as C, engine as H, scoring_packet as P
from .deferred_files import resolve_records


def input_hashes(kwargs, radar):
    return dict(depth=hashlib.sha256(kwargs["depth_image"].raw_data).hexdigest(),
        semantic=hashlib.sha256(kwargs["semantic_image"].raw_data).hexdigest(),
        radar=hashlib.sha256(radar.tobytes()).hexdigest(),
        camera=hashlib.sha256(kwargs["camera_matrix"].tobytes()).hexdigest(),
        inverse=hashlib.sha256(kwargs["camera_inverse"].tobytes()).hexdigest())


def make_engine(kind, root, name, frames=2):
    return kind(attempt_dir=root/name, catalogue_dir=root/"catalogue", max_frames=frames,
        expected_sources=W.sources(), allow_replay_clock=True, source_attempt_name=root.name)


def clone_inputs(initial):
    """Copy mutable fixture state without pickling immutable CARLA geometry."""
    result = dict(initial, identity=dict(initial["identity"]),
        semantic_image=copy.copy(initial["semantic_image"]),
        depth_image=copy.copy(initial["depth_image"]),
        frozen_world=copy.copy(initial["frozen_world"]),
        camera_matrix=initial["camera_matrix"].copy(),
        camera_inverse=initial["camera_inverse"].copy())
    result["frozen_world"].population_evidence = copy.deepcopy(initial["frozen_world"].population_evidence)
    result["rgb"] = copy.deepcopy({k: v for k, v in initial["rgb"].items() if k != "_rgb_image"})
    result["rgb"]["_rgb_image"] = result["semantic_image"]
    return result


def process(engine, directory, kwargs, radar, index=0):
    return engine.process(directory=directory, kwargs=kwargs, radar=radar,
        sent_raw_ns=W.raw(), sync_config=F.sync_config(engine.attempt),
        replay_observed_raw_ns=123+index)


class EngineTests(unittest.TestCase):
    def stopped_factory(self, clients):
        original = H.ArchiveClient
        def factory(**kwargs):
            client = original(**kwargs)
            clients.append(client)
            os.kill(client.process.pid, signal.SIGSTOP)
            os.waitpid(client.process.pid, os.WUNTRACED)
            return client
        return factory

    def resume(self, clients):
        for client in clients:
            if client.process.poll() is None:
                os.kill(client.process.pid, signal.SIGCONT)

    def cleanup_engine(self, engine):
        if not engine.closed:
            try:
                engine.close()
            except Exception:
                if engine.archiver is not None and not engine.archiver.closed:
                    engine.archiver.abort()
                raise

    def test_two_frames_exact_native_targets_avo_scorer_and_truthful_pending_receipts(self):
        before_env, before_sources = dict(os.environ), W.sources()
        original_writer = GE._create_identical_or_fail
        with F.fixture() as (root, initial, radar):
            control_kwargs, candidate_kwargs = clone_inputs(initial), clone_inputs(initial)
            control_dir, candidate_dir = root/"control_edge", root/"candidate_edge"
            candidate_dir.mkdir()
            reference, packets, candidate = [], [], []
            baseline = make_engine(RetainedEngine, root, "control")
            try:
                for index in range(2):
                    if index:
                        F.advance(control_kwargs, replenish=True)
                    result = process(baseline, control_dir, control_kwargs, radar, index)
                    self.assertEqual(result["status"], "OK", result.get("error"))
                    bundle = F.T.verify_full_inputs(control_dir, expected_identity=result["identity"])
                    reference.append((result, bundle, json.loads(Path(result["record"]["path"]).read_bytes())))
            finally:
                baseline.close()
            server = B.Server(candidate_dir/C.SOCKET_NAME, binding=C.binding(), max_frames=2).start()
            engine = make_engine(H.Engine, root, "candidate")
            try:
                producer = None
                for index in range(2):
                    if index:
                        F.advance(candidate_kwargs, replenish=True)
                    before = input_hashes(candidate_kwargs, radar)
                    result = process(engine, candidate_dir, candidate_kwargs, radar, index)
                    self.assertEqual(result["status"], "OK", result.get("error"))
                    candidate.append(result)
                    packet = P.decode_packet(server.take(result["identity"]),
                        expected_identity=result["identity"], binding=C.binding())
                    packets.append(packet)
                    desc = result["record"]["complete_target"]
                    self.assertIsNone(desc["manifest_sha256"])
                    self.assertTrue(desc["archive_pending"])
                    self.assertFalse(result["record"]["archive_receipt"]["durable"])
                    self.assertEqual(result["processed"], index+1)
                    self.assertEqual(result["sync"]["accepted"], index+1)
                    self.assertEqual((result["depth_reuse"]["decodes"], result["depth_reuse"]["reuses"]), (1, 3))
                    self.assertFalse(result["depth_reuse"]["retained_after_request"])
                    self.assertEqual(input_hashes(candidate_kwargs, radar), before)
                    if producer is None:
                        producer = engine.owner._complete_producer
                    self.assertIs(engine.owner._complete_producer, producer)
                    self.assertEqual(producer.processed_frames, index+1)
                    expected = reference[index][1].prepared
                    self.assertEqual(packet["prepared"].content_sha256, expected.content_sha256)
                    action = dict(frame_id=result["identity"]["frame_id"], session_uuid="packet-test",
                                  decision_seq=index, tensor_seq=index, ticket_seq=index)
                    arguments = dict(identity=action, predicted_labels=np.zeros((432, 768), np.uint8), predictions=[])
                    original_measurement = E.evaluate_prepared(**arguments, prepared=expected,
                        avo=reference[index][2]["avo"]).measurement
                    packet_measurement = E.evaluate_prepared(**arguments, prepared=packet["prepared"],
                        avo=packet["avo"]).measurement
                    self.assertEqual(packet_measurement, original_measurement)
                self.assertEqual(producer.population, {1, 2, 3})
                closed = engine.close()
                self.assertTrue(closed["archive_drained"])
                self.assertEqual(closed["archive"]["completed"], 2)
                self.assertEqual(engine.archiver.process.poll(), 0)
                for index, actual in enumerate(candidate):
                    record, expected_bundle, expected_avo = reference[index]
                    actual_record = actual["record"]
                    self.assertTrue(actual_record["archive_receipt"]["durable"])
                    self.assertNotIn("archive_pending", actual_record["complete_target"])
                    bundle = F.T.verify_full_inputs(candidate_dir, expected_identity=actual["identity"],
                        expected_manifest_sha256=actual_record["complete_target"]["manifest_sha256"])
                    self.assertEqual(bundle.manifest["payloads"], expected_bundle.manifest["payloads"])
                    self.assertEqual(bundle.prepared.content_sha256, expected_bundle.prepared.content_sha256)
                    self.assertEqual(F.tree(Path(actual_record["complete_sources"]["directory"])),
                                     F.tree(Path(record["record"]["complete_sources"]["directory"])))
                    actual_avo = json.loads(Path(actual_record["path"]).read_bytes())
                    self.assertEqual(without_avo_clocks(actual_avo), without_avo_clocks(expected_avo))
                    self.assertEqual(actual_avo, packets[index]["avo_document"])
                    self.assertGreater(bundle.manifest["ready_raw_ns"],
                        actual_record["scoring_handoff"]["accepted_response_raw_ns"])
            finally:
                try:
                    self.cleanup_engine(engine)
                finally:
                    self.assertFalse(server.close()["worker_alive"])
        self.assertIs(GE._create_identical_or_fail, original_writer)
        self.assertEqual(W.sources(), before_sources)
        self.assertEqual(dict(os.environ), before_env)
        self.assertFalse(torch.cuda.is_initialized())

    def test_packet_is_consumable_while_real_archive_process_is_stopped(self):
        with F.fixture() as (root, kwargs, radar):
            directory = root/"edge"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=1, max_jobs=1).start()
            engine, clients = make_engine(H.Engine, root, "handoff", frames=1), []
            try:
                with mock.patch.object(H, "ArchiveClient", side_effect=self.stopped_factory(clients)):
                    result = process(engine, directory, kwargs, radar)
                self.assertEqual(result["status"], "OK", result.get("error"))
                self.assertEqual(len(clients), 1)
                self.assertEqual(clients[0].outstanding_jobs, 1)
                packet = P.decode_packet(server.take(kwargs["identity"]), expected_identity=kwargs["identity"], binding=C.binding())
                self.assertEqual(packet["prepared"].frame_id, 100)
                self.assertFalse(Path(result["record"]["path"]).exists())
                self.assertFalse((Path(result["record"]["complete_target"]["directory"])/F.T.READY_NAME).exists())
                self.assertFalse(clients[0].completions)
                self.assertIsNone(result["record"]["complete_target"]["manifest_sha256"])
                self.resume(clients)
                closed = engine.close()
                self.assertTrue(closed["archive_drained"])
                self.assertTrue(result["record"]["archive_receipt"]["durable"])
                self.assertTrue(Path(result["record"]["path"]).is_file())
            finally:
                self.resume(clients)
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()

    def test_eager_archive_startup_precedes_live_clock_requests(self):
        with F.fixture() as (root, kwargs, radar):
            directory, attempt = root/"edge", root/"handoff"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=1, max_jobs=1).start()
            engine = H.Engine(attempt_dir=attempt, catalogue_dir=root/"catalogue", max_frames=1,
                expected_sources=W.sources(), allow_replay_clock=False,
                source_attempt_name=root.name,
                handoff_config=dict(directory=str(directory), evidence_root=str(attempt)))
            try:
                self.assertIsNotNone(engine.archiver)
                self.assertIsNone(engine.archiver.process.poll())
                with mock.patch.object(H, "ArchiveClient", side_effect=AssertionError("late archive startup")):
                    result = engine.process(directory=directory, kwargs=kwargs, radar=radar,
                        sent_raw_ns=W.raw(), sync_config=F.sync_config(attempt))
                self.assertEqual(result["status"], "OK", result.get("error"))
                packet = P.decode_packet(server.take(kwargs["identity"]),
                    expected_identity=kwargs["identity"], binding=C.binding())
                self.assertEqual(packet["prepared"].frame_id, 100)
                self.assertTrue(engine.close()["archive_drained"])
            finally:
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()

    def test_resolve_records_rejects_packet_native_binding_drift_against_valid_archive(self):
        with F.fixture() as (root, kwargs, radar):
            directory = root/"edge"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=1, max_jobs=1).start()
            engine = make_engine(H.Engine, root, "handoff", frames=1)
            try:
                result = process(engine, directory, kwargs, radar)
                self.assertEqual(result["status"], "OK", result.get("error"))
                packet = P.decode_packet(server.take(kwargs["identity"]),
                    expected_identity=kwargs["identity"], binding=C.binding())
                pending = copy.deepcopy(result["record"])
                engine.close()
                completions = engine.archive_result["completions"]
                # All native bytes, target metadata, READY and completion
                # digests remain genuine and internally consistent. Only the
                # independent pending/scoring reference is forged below.
                verified = F.T.verify_full_inputs(directory, expected_identity=kwargs["identity"])
                self.assertEqual(verified.prepared.content_sha256, packet["prepared"].content_sha256)
                for section, field in (("complete_target", "input_content_sha256"),
                        ("complete_target", "prepared_content_sha256"),
                        ("scoring_handoff", "input_content_sha256"),
                        ("scoring_handoff", "prepared_content_sha256")):
                    with self.subTest(section=section, field=field):
                        bad = copy.deepcopy(pending)
                        self.assertNotEqual(bad[section][field], "0"*64)
                        bad[section][field] = "0"*64
                        with self.assertRaises(ValueError):
                            resolve_records([bad], completions)
                        self.assertFalse(bad["archive_receipt"]["durable"])
                        self.assertIsNone(bad["complete_target"]["manifest_sha256"])
                good = resolve_records([copy.deepcopy(pending)], completions)[0]
                self.assertTrue(good["archive_receipt"]["durable"])
                self.assertEqual(good["complete_target"]["manifest_sha256"], verified.manifest_sha256)
            finally:
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()

    def test_source_alignment_rejection_preserves_evidence_without_packet(self):
        original_writer = GE._create_identical_or_fail
        with F.fixture() as (root, kwargs, radar):
            directory = root/"edge"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=2).start()
            engine = make_engine(H.Engine, root, "handoff")
            # Leave V6's earlier native-depth scope valid and exercise the
            # original sync validator's independent callback evidence guard.
            kwargs["rgb"]["depth_callback"]["timestamp"] += .01
            try:
                result = process(engine, directory, kwargs, radar)
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["processed"], 0)
                self.assertEqual(result["sync"]["rejected"], 1)
                self.assertEqual(server.stats()["admitted"], 0)
                self.assertFalse(engine.depth.active or engine.depth.retained)
                self.assertIs(GE._create_identical_or_fail, original_writer)
                with self.assertRaises(W.WorkerError):
                    process(engine, directory, kwargs, radar)
                closed = engine.close()
                self.assertTrue(closed["archive_drained"])
                self.assertEqual(closed["archive"]["completed"], 1)
                rejected = json.loads((engine.attempt/"rejected_100.json").read_bytes())
                self.assertFalse(rejected["scoring_packet_admitted"])
                self.assertTrue(list((engine.attempt/"sync").glob("*.rejected.json")))
                self.assertFalse(F.T.bundle_path(directory, kwargs["identity"]).exists())
            finally:
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()

    def test_early_depth_and_source_build_failures_preserve_exact_rejected_raw_sources(self):
        for failure in ("native_depth", "source_build"):
            with self.subTest(failure=failure), F.fixture() as (root, kwargs, radar):
                directory = root/"edge"
                directory.mkdir()
                server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=1, max_jobs=1).start()
                engine = make_engine(H.Engine, root, "handoff", frames=1)
                if failure == "native_depth":
                    kwargs["depth_image"].timestamp += .01
                try:
                    if failure == "source_build":
                        with mock.patch.object(engine.producer_type, "build",
                                side_effect=ValueError("injected source build failure")):
                            result = process(engine, directory, kwargs, radar)
                        self.assertIn("injected source build failure", result["error"]["message"])
                    else:
                        result = process(engine, directory, kwargs, radar)
                    self.assertEqual(result["status"], "FAILED")
                    self.assertEqual(result["processed"], 0)
                    self.assertEqual(server.stats()["admitted"], 0)
                    self.assertFalse(engine.depth.active or engine.depth.retained)
                    self.assertTrue(engine.close()["archive_drained"])
                    rejected = json.loads((engine.attempt/"rejected_100.json").read_bytes())
                    sources = rejected["rejected_sources"]
                    self.assertFalse(sources["accepted"])
                    self.assertEqual(sources["identity"], kwargs["identity"])
                    self.assertEqual(set(sources["sources"]), {"rgb", "depth", "semantic", "radar"})
                    self.assertLessEqual(sources["captured_payload_bytes"], sources["max_payload_bytes"])
                    expected = dict(rgb=kwargs["rgb"]["_rgb_image"].raw_data,
                        depth=kwargs["depth_image"].raw_data, semantic=kwargs["semantic_image"].raw_data)
                    for role, row in sources["sources"].items():
                        payload = Path(row["path"]).read_bytes()
                        self.assertEqual(row["status"], "CAPTURED_NOT_ACCEPTED")
                        self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
                        if role == "radar":
                            np.testing.assert_array_equal(np.load(row["path"], allow_pickle=False), radar)
                            self.assertEqual(row["source_sha256"], hashlib.sha256(radar.tobytes()).hexdigest())
                        else:
                            self.assertEqual(payload, expected[role])
                            self.assertEqual(row["source_sha256"], hashlib.sha256(expected[role]).hexdigest())
                    self.assertFalse(F.T.bundle_path(directory, kwargs["identity"]).exists())
                finally:
                    try:
                        self.cleanup_engine(engine)
                    finally:
                        server.close()

    def test_rejected_snapshot_serialization_error_does_not_replace_primary_failure(self):
        with F.fixture() as (root, kwargs, radar):
            directory = root/"edge"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=1, max_jobs=1).start()
            engine = make_engine(H.Engine, root, "handoff", frames=1)
            try:
                with mock.patch.object(engine.producer_type, "build", side_effect=ValueError("primary build error")), \
                        mock.patch.object(engine, "_rejected_sources", side_effect=RuntimeError("snapshot error")):
                    result = process(engine, directory, kwargs, radar)
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["error"]["message"], "primary build error")
                self.assertTrue(engine.close()["archive_drained"])
                failed = engine.failed_batches[-1]
                self.assertIn("snapshot error", failed["rejection"]["diagnostic_preservation_error"])
                self.assertIn("primary build error", failed["original_error"])
                self.assertEqual(server.stats()["admitted"], 0)
            finally:
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()

    def test_archive_overflow_after_packet_is_explicit_failure_and_drains_admitted_jobs(self):
        with F.fixture() as (root, kwargs, radar):
            directory = root/"edge"
            directory.mkdir()
            server = B.Server(directory/C.SOCKET_NAME, binding=C.binding(), max_frames=3).start()
            engine, clients = make_engine(H.Engine, root, "handoff", frames=3), []
            try:
                with mock.patch.object(C, "MAX_JOBS", 2), \
                        mock.patch.object(H, "ArchiveClient", side_effect=self.stopped_factory(clients)):
                    for index in range(3):
                        if index:
                            F.advance(kwargs, frame=100+index)
                        result = process(engine, directory, kwargs, radar, index)
                        server.take(kwargs["identity"])
                        self.assertEqual(result["status"], "OK" if index < 2 else "FAILED", result.get("error"))
                self.assertEqual(result["processed"], 2)
                self.assertTrue(engine.failed)
                self.assertIn("credits", result["error"]["message"])
                self.assertEqual(server.stats()["admitted"], 3)
                self.assertEqual(clients[0].outstanding_jobs, 2)
                self.assertGreater(engine.failed_batches[-1]["unpreserved_bytes"], 0)
                self.resume(clients)
                with self.assertRaises(A.ArchiveError):
                    engine.close()
                self.assertEqual(engine.archive_result["completed"], 2)
                self.assertTrue(engine.archive_result["failed"])
                self.assertFalse(engine.close_result["archive_drained"])
                self.assertFalse(F.T.bundle_path(directory, kwargs["identity"]).exists())
                self.assertEqual(clients[0].process.poll(), 0)
            finally:
                self.resume(clients)
                try:
                    self.cleanup_engine(engine)
                finally:
                    server.close()


if __name__ == "__main__":
    unittest.main()
