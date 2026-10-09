"""Independent CPU-only transport, lifecycle and full-GT worker proof tests."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import worker as W
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as F
from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import sensor_metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in Path(path).rglob("*") if p.is_file()}


def advance(kwargs, frame=101, *, replenish=False):
    stamp = frame * .05
    kwargs["identity"] = dict(kwargs["identity"], frame_id=frame, capture_timestamp_ns=int(stamp * 1e9))
    kwargs["policy_timestamp_s"] = stamp
    frozen = kwargs["frozen_world"]
    frozen.frozen_frame_id, frozen.frozen_timestamp_s = frame, stamp
    population = dict(frozen.population_evidence, frame_id=frame)
    if replenish:
        actors = F.Actors((*frozen.get_actors(), F.actor(3, "vehicle.new", 25., 7.)))
        frozen.get_actors = lambda: actors
        population.update(active_population_ids=[1, 2, 3], registered_population_ids=[1, 2, 3],
            population={"registered_actors": [1, 2, 3], "ego_id": 999})
        population["population_sha256"] = G.sha(population["population"])
    frozen.population_evidence = population
    for image in (kwargs["semantic_image"], kwargs["depth_image"]):
        image.frame, image.timestamp = frame, stamp
    rgb = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
    rgb["_rgb_image"] = kwargs["semantic_image"]
    rgb["semantic_callback"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
    rgb["depth_callback"] = sensor_metadata(kwargs["depth_image"], receipt_raw_ns=1)
    rgb["radar_callback"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
    kwargs["rgb"] = rgb


@contextmanager
def fixture():
    G.bind_legacy_namespace()
    with F.SourceGeometryTests().fixture() as (root, supplied):
        actors = F.Actors((F.actor(1, "walker.pedestrian.0001", 20., -3.),
                           F.actor(2, "vehicle.test", 20., 4.)))
        supplied["frozen_world"].get_actors = lambda: actors
        population = dict(supplied["dynamic_snapshot_evidence"], registered_population_ids=[1, 2],
            population={"registered_actors": [1, 2], "ego_id": 999})
        population["population_sha256"] = G.sha(population["population"])
        supplied["frozen_world"].population_evidence = population
        supplied["rgb"]["_rgb_image"] = supplied["semantic_image"]
        supplied["rgb"]["semantic_callback"] = sensor_metadata(supplied["semantic_image"], receipt_raw_ns=1)
        supplied["rgb"]["depth_callback"] = sensor_metadata(supplied["depth_image"], receipt_raw_ns=1)
        kwargs = {key: supplied[key] for key in ("identity", "rgb", "semantic_image", "depth_image",
            "frozen_world", "camera_matrix", "camera_inverse")}
        kwargs["policy_timestamp_s"] = 5.
        yield root, kwargs, supplied["radar_world_xyz"]


def sync_config(root):
    return {"directory": str(root / "sync"), "raw_bundle_limit": 8,
            "raw_byte_limit": 128 * 1024 * 1024}


class FramingTests(unittest.TestCase):
    def test_round_trip_binary_is_not_json_or_pickle(self):
        body = b"\x00\xff\x80untrusted-looking-byte-string"
        meta = {"command": "RUN", "identity": {"frame_id": 12}}
        self.assertEqual(W.unpack_message(W.frame_bytes(meta, body)), (meta, body))

    def test_duplicate_nonfinite_nonobject_and_truncation_refused(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, W.WorkerError)):
                W.unpack_message(struct.pack("!Q", len(raw)) + raw)
        for payload in (b"", b"1234567", struct.pack("!Q", W.MAX_JSON + 1),
                        struct.pack("!Q", 3) + b"{}", struct.pack("!Q", 0)):
            with self.subTest(payload=payload), self.assertRaises(W.WorkerError):
                W.unpack_message(payload)

    def test_outgoing_bounds_before_socket_io(self):
        with mock.patch.object(W, "MAX_MESSAGE", 64), mock.patch.object(W, "MAX_JSON", 32):
            with self.assertRaisesRegex(W.WorkerError, "bound"):
                W.frame_bytes({"s": "x" * 32})
            with self.assertRaisesRegex(W.WorkerError, "bound"):
                W.frame_bytes({}, b"x" * 64)
            sock = mock.Mock()
            with self.assertRaisesRegex(W.WorkerError, "bound"):
                W.send(sock, b"x" * 65, time.monotonic() + 1)
            sock.sendall.assert_not_called()

    def test_receive_bounds_and_partial_eof(self):
        packets = (struct.pack("!Q", W.MAX_MESSAGE + 1), struct.pack("!Q", 7),
                   b"1234", struct.pack("!Q", 20) + b"short")
        for packet in packets:
            with self.subTest(packet=packet):
                left, right = socket.socketpair()
                try:
                    right.sendall(packet)
                    right.close()
                    with self.assertRaises(W.WorkerError):
                        W.receive(left, time.monotonic() + .2)
                finally:
                    left.close(); right.close()

    def test_send_stall_and_receive_stall_are_bounded(self):
        left, right = socket.socketpair()
        try:
            left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            started = time.monotonic()
            with self.assertRaises((TimeoutError, W.WorkerError)):
                W.send(left, b"x" * (2 * 1024 * 1024), started + .03)
            self.assertLess(time.monotonic() - started, 1.)
        finally:
            left.close(); right.close()
        left, right = socket.socketpair()
        try:
            started = time.monotonic()
            with self.assertRaises((TimeoutError, W.WorkerError)):
                W.receive(left, started + .03)
            self.assertLess(time.monotonic() - started, 1.)
            with self.assertRaises(W.WorkerError):
                W.receive(left, time.monotonic() - 1)
        finally:
            left.close(); right.close()


class EngineTests(unittest.TestCase):
    def test_binding_and_frame_bounds_reject_before_output_creation(self):
        with tempfile.TemporaryDirectory(prefix="worker-guard-") as temporary:
            root = Path(temporary)
            for frames, sources in ((0, W.sources()), (1001, W.sources()), (True, W.sources()), (1, {})):
                with self.subTest(frames=frames, pins=bool(sources)), self.assertRaises(W.WorkerError):
                    W.Engine(attempt_dir=root / "must_not_exist", catalogue_dir=root,
                             max_frames=frames, expected_sources=sources)
                self.assertFalse((root / "must_not_exist").exists())

    def test_real_full_gt_sync_population_and_timing_are_preserved(self):
        with fixture() as (root, kwargs, radar):
            from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S, evaluation as E
            from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
            before = (S.LiveCompleteTargetProducer, E.prepare_target, T.write_bundle, G.write_source_archive,
                      LG.write_avo_bundle)
            engine = W.Engine(attempt_dir=root / "engine", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            try:
                with mock.patch.object(threading.Thread, "start", side_effect=AssertionError("engine started a thread")):
                    first = engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                           sent_raw_ns=W.raw(), sync_config=sync_config(root))
                self.assertEqual(first["status"], "OK", first.get("error"))
                self.assertEqual(first["processed"], 1)
                self.assertEqual(first["sync"]["accepted"], 1)
                self.assertEqual(first["sync"]["rejected"], 0)
                self.assertEqual(len(first["rows"]), 5)
                self.assertTrue(all(r["completed"] and r["process_pid"] == os.getpid() for r in first["rows"]))
                original = tree(root / "edge")
                producer = engine.owner._complete_producer
                advance(kwargs, replenish=True)
                second = engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                        sent_raw_ns=W.raw(), sync_config=sync_config(root))
                self.assertEqual(second["status"], "OK", second.get("error"))
                self.assertIs(engine.owner._complete_producer, producer)
                self.assertEqual(producer.population, {1, 2, 3})
                self.assertEqual(second["processed"], 2)
                self.assertEqual(second["sync"]["accepted"], 2)
                self.assertEqual(len(second["rows"]), 5)
                self.assertTrue(all(sha(root / "edge" / name) == digest for name, digest in original.items()))
                for result in (first, second):
                    record = result["record"]
                    verified = T.verify_full_inputs(root / "edge", expected_identity=result["identity"],
                        expected_manifest_sha256=record["complete_target"]["manifest_sha256"])
                    self.assertTrue(verified.native_inputs_verified)
                    self.assertFalse(verified.manifest["live_qualified"])
                with self.assertRaises(W.WorkerError):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=W.raw())
            finally:
                timing = engine.close()
            self.assertEqual(timing["records"], 10)
            self.assertEqual(timing["dropped"], 0)
            self.assertTrue(timing["flush_after_worker_drain"])
            self.assertEqual(engine.close(), timing)
            self.assertEqual(before, (S.LiveCompleteTargetProducer, E.prepare_target, T.write_bundle,
                                     G.write_source_archive, LG.write_avo_bundle))
            self.assertFalse(torch.cuda.is_initialized())

    def test_post_avo_failure_retains_rejection_and_disallows_further_work(self):
        with fixture() as (root, kwargs, radar):
            engine = W.Engine(attempt_dir=root / "engine", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            kwargs["frozen_world"].population_evidence["registry_complete"] = False
            try:
                result = engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                        sent_raw_ns=W.raw(), sync_config=sync_config(root))
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["processed"], 0)
                self.assertTrue(engine.failed)
                avo = result["error"]["avo_evidence_record"]
                self.assertEqual(sha(avo["path"]), avo["sha256"])
                self.assertEqual([r["completed"] for r in result["rows"]], [True, False])
                rejection = root / "engine" / root.name / "parity_rejected_targets" / T.bundle_path(root, kwargs["identity"]).name
                document = json.loads((rejection / "REJECTION.json").read_bytes())
                self.assertFalse(document["target_ready"])
                self.assertEqual(document["stage"], "complete_source_build")
                for name, digest in document["sha256"].items():
                    self.assertEqual(sha(rejection / name), digest)
                self.assertEqual(result["sync"]["accepted"], 1)
                with self.assertRaises(W.WorkerError):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=W.raw())
            finally:
                timing = engine.close()
            self.assertEqual(timing["records"], 2)

    def test_last_send_replay_clock_and_sync_owner_guards(self):
        with fixture() as (root, kwargs, radar):
            engine = W.Engine(attempt_dir=root / "engine", catalogue_dir=root / "catalogue",
                              max_frames=3, expected_sources=W.sources())
            try:
                for sent in (0, -1, True, W.raw() + 10**12):
                    with self.subTest(sent=sent), self.assertRaises(W.WorkerError):
                        engine.process(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=sent)
                with self.assertRaisesRegex(W.WorkerError, "offline timestamp"):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                   sent_raw_ns=W.raw(), replay_observed_raw_ns=123)
                result = engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                        sent_raw_ns=W.raw(), sync_config=sync_config(root))
                self.assertEqual(result["status"], "OK", result.get("error"))
                with self.assertRaisesRegex(W.WorkerError, "frame order"):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                                   sent_raw_ns=W.raw(), sync_config=sync_config(root))
                advance(kwargs)
                with self.assertRaisesRegex(W.WorkerError, "disappeared"):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=W.raw())
                with self.assertRaisesRegex(W.WorkerError, "owner changed"):
                    engine.process(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=W.raw(),
                        sync_config={**sync_config(root), "raw_bundle_limit": 9})
            finally:
                engine.close()


class ChildTests(unittest.TestCase):
    def test_real_child_matches_direct_engine_all_payloads_and_avo(self):
        with fixture() as (root, kwargs, radar):
            engine = W.Engine(attempt_dir=root / "direct", catalogue_dir=root / "catalogue",
                max_frames=2, expected_sources=W.sources(), allow_replay_clock=True,
                source_attempt_name="independent_worker_test")
            try:
                reference = engine.process(directory=root / "direct_edge", kwargs=kwargs, radar=radar,
                    sent_raw_ns=W.raw(), sync_config=sync_config(root / "direct"), replay_observed_raw_ns=123)
                self.assertEqual(reference["status"], "OK", reference.get("error"))
            finally:
                engine.close()
            client = W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                max_frames=2, expected_sources=W.sources(), allow_replay_clock=True,
                source_attempt_name="independent_worker_test")
            try:
                actual = client.run(directory=root / "child_edge", kwargs=kwargs, radar=radar,
                    sent_raw_ns=W.raw(), sync_config=sync_config(root / "child"), replay_observed_raw_ns=123)
                self.assertEqual(actual["status"], "OK", actual.get("error"))
                self.assertEqual(actual["processed"], 1)
                self.assertEqual(actual["worker"]["pid"], client.process.pid)
                self.assertNotEqual(actual["worker"]["pid"], os.getpid())
                self.assertEqual(len(actual["rows"]), 5)
                self.assertTrue(all(r["process_pid"] == client.process.pid for r in actual["rows"]))
                self.assertLess(actual["ipc"]["request_bytes"], W.MAX_MESSAGE)
                expected_bundle = T.verify_full_inputs(root / "direct_edge", expected_identity=kwargs["identity"])
                actual_bundle = T.verify_full_inputs(root / "child_edge", expected_identity=kwargs["identity"])
                self.assertEqual(actual_bundle.manifest["payloads"], expected_bundle.manifest["payloads"])
                self.assertEqual(actual_bundle.prepared.content_sha256, expected_bundle.prepared.content_sha256)
                self.assertEqual(tree(Path(actual["record"]["complete_sources"]["directory"])),
                                 tree(Path(reference["record"]["complete_sources"]["directory"])))
                first_avo = json.loads(Path(reference["record"]["path"]).read_bytes())
                child_avo = json.loads(Path(actual["record"]["path"]).read_bytes())
                for key in ("avo", "raw_pedestrian_rows", "payloads"):
                    self.assertEqual(child_avo[key], first_avo[key], key)
                self.assertEqual(actual["sync"]["accepted"], 1)
                for record in actual["sync"]["records"]:
                    prefix = Path(actual["sync"]["directory"]) / f"frame_{record['frame_id']:010d}"
                    self.assertEqual(sha(str(prefix) + ".observed.json"), record["observed_sha256"])
                    self.assertTrue(Path(str(prefix) + ".accepted.json").is_file())
                advance(kwargs, replenish=True)
                second = client.run(directory=root / "child_edge", kwargs=kwargs, radar=radar,
                    sent_raw_ns=W.raw(), sync_config=sync_config(root / "child"), replay_observed_raw_ns=124)
                self.assertEqual(second["processed"], 2)
                self.assertEqual(second["worker"]["pid"], actual["worker"]["pid"])
                self.assertEqual(second["sync"]["accepted"], 2)
            finally:
                closed = client.close()
            self.assertEqual(closed["timing"]["records"], 10)
            self.assertEqual(client.process.poll(), 0)

    def test_handled_child_failure_flushes_real_partial_timing_before_error(self):
        with fixture() as (root, kwargs, radar):
            client = W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            kwargs["frozen_world"].population_evidence["registry_complete"] = False
            with self.assertRaises(W.WorkerError) as caught:
                client.run(directory=root / "edge", kwargs=kwargs, radar=radar,
                           sent_raw_ns=W.raw(), sync_config=sync_config(root))
            self.assertTrue(client.failed and client.closed)
            self.assertIsNotNone(client.process.poll())
            self.assertTrue(hasattr(caught.exception, "avo_evidence_record"))
            record = caught.exception.avo_evidence_record
            self.assertEqual(sha(record["path"]), record["sha256"])
            timing_root = root / "child" / "engine" / "complete_gt_stage_timing"
            timing = json.loads((timing_root / "MANIFEST.json").read_bytes())
            self.assertEqual(timing["records"], 2)
            self.assertEqual(timing["dropped"], 0)
            self.assertTrue(timing["flush_after_worker_drain"])
            stages = [json.loads(row) for row in (timing_root / "stages.jsonl").read_text().splitlines()]
            self.assertEqual([row["completed"] for row in stages], [True, False])
            self.assertEqual(sha(timing_root / "stages.jsonl"), timing["sha256"]["stages.jsonl"])
            with self.assertRaises(W.WorkerError):
                client.run(directory=root / "edge", kwargs=kwargs, radar=radar, sent_raw_ns=W.raw())

    def test_stopped_child_send_timeout_kills_and_joins_without_retry(self):
        with fixture() as (root, _kwargs, _radar):
            client = W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            try:
                os.kill(client.process.pid, signal.SIGSTOP)
                client.timeout = .05
                started = time.monotonic()
                with self.assertRaises((W.WorkerError, TimeoutError, OSError)):
                    client._exchange({"command": "RUN"}, b"x" * (2 * 1024 * 1024))
                self.assertLess(time.monotonic() - started, 4.5)
                self.assertTrue(client.closed and client.failed)
                self.assertIsNotNone(client.process.poll())
                with self.assertRaises(W.WorkerError):
                    client._exchange({"command": "CLOSE"})
            finally:
                client.abort()

    def test_popen_failure_closes_both_socket_descriptors(self):
        with fixture() as (root, _kwargs, _radar):
            sockets = []
            real = socket.socketpair
            def pair(*args, **kwargs):
                value = real(*args, **kwargs)
                sockets.extend(value)
                return value
            with mock.patch.object(W.socket, "socketpair", side_effect=pair), \
                    mock.patch.object(W.subprocess, "Popen", side_effect=OSError("injected exec failure")):
                with self.assertRaisesRegex(OSError, "injected exec failure"):
                    W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                             max_frames=2, expected_sources=W.sources())
            self.assertEqual(len(sockets), 2)
            self.assertTrue(all(sock.fileno() == -1 for sock in sockets))

    def test_real_child_is_cpu_only_source_bound_and_joined(self):
        with fixture() as (root, _kwargs, _radar):
            client = W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            pid = client.process.pid
            self.assertNotEqual(pid, os.getpid())
            response = client.close()
            self.assertEqual(response["processed"], 0)
            self.assertEqual(response["timing"]["records"], 0)
            self.assertEqual(client.process.poll(), 0)
            self.assertTrue(client.closed)
            self.assertIsNone(client.close())
            self.assertFalse(torch.cuda.is_initialized())

    def test_child_source_binding_failure_leaves_no_process(self):
        with fixture() as (root, _kwargs, _radar):
            real = W.subprocess.Popen
            children = []
            def start(*args, **kwargs):
                value = real(*args, **kwargs)
                children.append(value)
                return value
            with mock.patch.object(W.subprocess, "Popen", side_effect=start):
                with self.assertRaisesRegex(W.WorkerError, "source binding"):
                    W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                             max_frames=2, expected_sources={})
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())

    def test_dead_child_fail_closed_and_abort_idempotent(self):
        with fixture() as (root, _kwargs, _radar):
            client = W.Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources())
            client.process.kill()
            client.process.wait(timeout=5)
            with self.assertRaises((W.WorkerError, OSError)):
                client._exchange({"command": "CLOSE"})
            self.assertTrue(client.closed and client.failed)
            client.abort()
            with self.assertRaises(W.WorkerError):
                client.close()


if __name__ == "__main__":
    unittest.main()
