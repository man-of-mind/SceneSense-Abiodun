"""Actual legacy AVO/priority-queue paths with CPU-only sensor fixtures."""
from collections import OrderedDict
from pathlib import Path
import json
import queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np

from . import source_ownership as S
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_object_gt_v2 as OG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_gt_priority_v2 as GP


def identity(frame):
    return dict(run_id="owned_test", cell_id="cell", stream_id="stream", frame_id=frame,
                action_id=None, profile_id=None, capture_timestamp_ns=1000000000+frame)


class Base:
    def __init__(self, **kwargs):
        self._quality_scene_lock = threading.Lock(); self._quality_scenes = {}
        self.live = NS(_run4_identity={}, _gt_identity={}, last_datagram={}, identity_for_frame=identity)
        self.edge_evidence_dir = Path(kwargs.get("attempt_dir", "."))/"unused"
        self.aligned_actor_tracker = object(); self.base_calls = []; self.fail_base = False
        self.evaluation_worker = NS(is_alive=lambda: False)  # no real thread in CPU fixture

    def _on_rgb(self, image): pass
    def _on_semantic(self, image): pass
    def _on_radar(self, image): pass

    def _ground_truth(self, **kwargs):
        self.base_calls.append(kwargs)
        if self.fail_base:
            raise RuntimeError("base worker fixture failure")
        self._quality_scenes.pop(kwargs["frame_id"], None)
        return [{"frame_id": kwargs["frame_id"], "unchanged": True}]

    def classify(self, item):
        frame = item["frame_id"]
        if frame not in self.live._run4_identity:
            raise GP.GtQueueError("foreign fixture frame")
        if not self.live._run4_identity[frame]["reward_requested"]:
            return GP.LOW
        if frame not in self.live.last_datagram:
            raise GP.GtQueueError("fixture before last datagram")
        return GP.HIGH


def image(frame, mutable=False):
    matrix = np.eye(4); matrix[0, 3] = frame/10
    raw = bytes([frame % 255])*16
    return NS(frame=frame, timestamp=frame*.05, width=2, height=2,
              raw_data=bytearray(raw) if mutable else raw,
              transform=NS(get_matrix=lambda: matrix.tolist()), _matrix=matrix)


def feed(owner, frame, reward=True, mutable=False):
    rgb, semantic, depth = [image(frame, mutable) for _ in range(3)]
    radar = image(frame, mutable)
    owner._on_rgb(rgb); owner._on_semantic(semantic); owner._on_depth(depth); owner._on_radar(radar)
    world = NS(frozen_frame_id=frame, frozen_timestamp_s=rgb.timestamp, frozen_observed_raw_ns=1)
    owner._quality_scenes[frame] = world
    owner.live._run4_identity[frame] = {"frame_id": frame, "reward_requested": reward, "unchanged_action": frame*2}
    owner.live._gt_identity[frame] = identity(frame); owner.live.last_datagram[frame] = 1
    ticket = dict(frame_id=frame, timestamp=rgb.timestamp, camera_matrix=rgb._matrix.copy(),
        camera_inverse=np.linalg.inv(rgb._matrix), radar_points={"world_xyz": np.array([[1., 2., 3.]], np.float32)},
        scene=world, camera_location=NS(x=frame/10, y=0., z=0.))
    return ticket, (rgb, semantic, depth, radar)


def make_owner(max_frames=100, max_pending=S.DEFAULT_MAX_PENDING, max_bytes=S.DEFAULT_MAX_BYTES):
    stores = []
    klass = S.owned_collector(LG.build_live_gt_collector(Base), stores=stores,
        max_frames=max_frames, max_pending=max_pending, max_bytes=max_bytes)
    owner = klass()
    queued = S.owned_queue(OG.DeferringRewardPriorityGtQueueV2)(classify=owner.classify,
        low_blocked=lambda: True, on_skip=lambda f, why: None, maxsize_high=64, maxsize_low=64)
    return owner, queued


def consume(owner, queued, write=None, ticket=None):
    ticket = queued.get_nowait() if ticket is None else ticket
    if write is None:
        write = lambda directory, **kw: {"frame_id": kw["identity"]["frame_id"], "fixture": True}
    try:
        with mock.patch.object(LG, "write_avo_bundle", side_effect=write):
            return owner._ground_truth(frame_id=ticket["frame_id"], timestamp=ticket["timestamp"],
                camera_matrix=ticket["camera_matrix"], camera_inverse=ticket["camera_inverse"],
                radar_points=ticket["radar_points"], world=ticket["scene"], camera_location=ticket["camera_location"])
    finally:
        queued.task_done()


class SourceOwnershipTests(unittest.TestCase):
    def test_actual_avo_queue_path_survives_more_than24_callbacks_and_verifies(self):
        owner, queued = make_owner(); ticket, images = feed(owner, 1)
        original_radar = ticket["radar_points"]["world_xyz"].copy()
        queued.put_nowait(ticket)
        for frame in range(2, 42):
            feed(owner, frame, reward=False)
        self.assertNotIn(1, owner._avo_rgb)
        observed = []
        def write(directory, **kw):
            observed.append(kw)
            self.assertEqual(kw["identity"], identity(1))
            self.assertEqual(kw["rgb"]["_rgb_image"].raw_data, images[0].raw_data)
            self.assertEqual(kw["depth_image"].frame, 1)
            return {"frame_id": 1}
        result = consume(owner, queued, write)
        self.assertEqual(result, [{"frame_id": 1, "unchanged": True}])
        self.assertTrue(np.array_equal(owner.base_calls[0]["radar_points"]["world_xyz"], original_radar))
        self.assertEqual(owner._source_ownership.pending, {}); self.assertEqual(queued.unfinished_tasks, 0)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"owned"; manifest = owner._source_ownership.flush(directory)
            proof = S.verify_ownership_archive(directory, expected_identities=[identity(1)],
                expected_action_identities=[owner.live._run4_identity[1]])
            self.assertTrue(manifest["complete"]); self.assertTrue(proof["passed"])
            self.assertEqual(proof["admitted"], proof["consumed"])

    def test_bytes_reused_mutable_payload_metadata_radar_and_pose_isolated(self):
        for mutable in (False, True):
            with self.subTest(mutable=mutable):
                owner, queued = make_owner(); ticket, images = feed(owner, 1, mutable=mutable)
                queued.put_nowait(ticket)
                pinned = owner._source_ownership.get(1)
                if not mutable:
                    self.assertIs(pinned.rgb.raw_data, images[0].raw_data)
                else:
                    images[0].raw_data[0] = 200; images[1].raw_data[0] = 201; images[2].raw_data[0] = 202
                images[0]._matrix[:] = 0; images[0].frame = 500
                owner._avo_rgb[1]["camera_matrix"][0][3] = 999
                ticket["camera_matrix"][:] = 0; ticket["radar_points"]["world_xyz"][:] = 999
                data = queued.get_nowait()
                self.assertEqual(pinned.rgb.frame, 1); self.assertEqual(pinned.rgb.raw_data[0], 1)
                self.assertEqual(pinned.sensors()[0]["camera_matrix"][0][3], .1)
                self.assertEqual(data["camera_matrix"][0, 0], 1)
                self.assertTrue(np.array_equal(data["radar_points"]["world_xyz"], [[1, 2, 3]]))
                for array in (data["camera_matrix"], data["radar_points"]["world_xyz"]):
                    with self.assertRaises(ValueError): array.setflags(write=True)
                owner._source_ownership.release(1, reason="TEST_RELEASE"); queued.task_done()
                event = owner._source_ownership.events[0]
                self.assertEqual(event["immutable_image_bytes_reused"], 0 if mutable else 48)
                self.assertGreaterEqual(event["admission_ms"], 0)

    def test_worker_and_target_failures_release_in_finally(self):
        for phase in ("base", "target"):
            with self.subTest(phase=phase):
                owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
                owner.fail_base = phase == "base"
                def fail(directory, **kw): raise RuntimeError("target fixture failure")
                with self.assertRaises(RuntimeError): consume(owner, queued, fail)
                self.assertEqual(owner._source_ownership.pending, {})
                self.assertEqual(owner._source_ownership.bytes, 0); self.assertEqual(queued.unfinished_tasks, 0)
                self.assertEqual(owner._source_ownership.errors[-1]["kind"], "GT_FAILED")

    def test_failure_before_ground_truth_is_released_by_actual_task_done(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
        queued.get_nowait()  # models scene_source.refresh_static raising before _ground_truth
        queued.task_done()
        self.assertFalse(owner._source_ownership.pending)
        self.assertEqual(owner._source_ownership.errors[-1]["kind"], "WORKER_SOURCE_NOT_COMPLETED")

    def test_missing_sources_admission_failure_has_evidence_and_no_queue_item(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); owner._avo_depth.clear()
        with self.assertRaisesRegex(S.SourceOwnershipError, "missing exact sensor"): queued.put_nowait(ticket)
        self.assertTrue(queued.empty()); self.assertEqual(owner._source_ownership.bytes, 0)
        self.assertEqual(owner._source_ownership.errors[-1]["kind"], "SOURCE_ADMISSION_FAILED")

    def test_last_send_and_nonreward_holds_keep_original_classifier_rules(self):
        owner, queued = make_owner(); held, _ = feed(owner, 1, reward=False); queued.put_nowait(held)
        self.assertEqual(queued.counters["low_skipped_on_enqueue"], 1)
        self.assertFalse(owner._source_ownership.seen)
        ticket, _ = feed(owner, 2); del owner.live.last_datagram[2]
        with self.assertRaises(GP.GtQueueError): queued.put_nowait(ticket)
        self.assertTrue(queued.empty()); self.assertFalse(owner._source_ownership.pending)

    def test_count_byte_bounds_and_queue_overflow_rollback(self):
        for mode in ("count", "bytes", "queue"):
            with self.subTest(mode=mode):
                owner, queued = make_owner(max_pending=1 if mode == "count" else 64,
                    max_bytes=1 if mode == "bytes" else S.DEFAULT_MAX_BYTES)
                first, _ = feed(owner, 1)
                if mode == "bytes":
                    with self.assertRaisesRegex(S.SourceOwnershipError, "byte bound"): queued.put_nowait(first)
                    self.assertFalse(owner._source_ownership.pending)
                    continue
                if mode == "queue": queued._max["HIGH"] = 1
                queued.put_nowait(first)
                second, _ = feed(owner, 2)
                with self.assertRaises((S.SourceOwnershipError, GP.GtQueueError)): queued.put_nowait(second)
                self.assertEqual(set(owner._source_ownership.pending), {1})
                consume(owner, queued)
                self.assertEqual(owner._source_ownership.bytes, 0)

    def test_duplicate_does_not_release_original_pending_source(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
        with self.assertRaises((S.SourceOwnershipError, GP.GtQueueError)): queued.put_nowait(ticket)
        self.assertEqual(set(owner._source_ownership.pending), {1}); consume(owner, queued)

    def test_exact_retained_capacity64_waiting_plus1_running(self):
        owner, queued = make_owner()
        first, _ = feed(owner, 1); queued.put_nowait(first)
        running = queued.get_nowait()
        for frame in range(2, 66):
            ticket, _ = feed(owner, frame); queued.put_nowait(ticket)
        self.assertEqual(queued.depth()["HIGH"], 64)
        self.assertEqual(len(owner._source_ownership.pending), 65)
        self.assertEqual(owner._source_ownership.max_pending, 65)
        overflow, _ = feed(owner, 66)
        with self.assertRaisesRegex(GP.GtQueueError, "HIGH queue overflow"):
            queued.put_nowait(overflow)
        self.assertEqual(queued.counters["overflow_HIGH"], 1)
        self.assertEqual(owner._source_ownership.peak_pending, 65)
        self.assertNotIn(66, owner._source_ownership.seen)
        consume(owner, queued, ticket=running)
        while not queued.empty(): consume(owner, queued)
        self.assertFalse(owner._source_ownership.pending)
        self.assertEqual(queued.unfinished_tasks, 0)

    def test_mutated_identity_fails_and_releases(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
        owner.live._run4_identity[1]["unchanged_action"] = 99
        with self.assertRaisesRegex(S.SourceOwnershipError, "scientific identity mutated"): consume(owner, queued)
        self.assertFalse(owner._source_ownership.pending)
        self.assertEqual(owner._source_ownership.events[0]["action_identity"]["unchanged_action"], 2)

    def test_final_phase6_classification_override_and_gt_gate_are_composed(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        stores = []
        base = S.owned_collector(LG.build_live_gt_collector(Base), stores=stores,
            max_frames=100, max_pending=S.DEFAULT_MAX_PENDING, max_bytes=S.DEFAULT_MAX_BYTES)
        final = U.build_run4_collector_class(base)
        owner = final.__new__(final)
        base.__init__(owner)  # omit external runtime/model/actor initialization only
        owner._run4_lock = threading.Lock()
        owner._run4_rgb_raw = OrderedDict()
        owner._run4_carla_trace = {"rgb": [], "ticks": []}
        owner.live.gt_log = OG.GtTicketLogV3()
        owner._run4_gate = OG.RewardPendingGateV2()
        queued = S.owned_queue(OG.DeferringRewardPriorityGtQueueV2)(classify=owner._run4_gt_class,
            low_blocked=owner._run4_gate.blocked, on_skip=lambda f, why: None)
        ticket, _ = feed(owner, 1); owner._run4_gate.open(1)
        queued.put_nowait(ticket)
        for frame in range(2, 42): feed(owner, frame, reward=False)
        consume(owner, queued)
        self.assertEqual(owner._source_ownership.events[0]["kind"], "SOURCE_ADMITTED")
        self.assertEqual(len([e for e in owner._source_ownership.events if e["kind"] == "SOURCE_ADMITTED"]), 1)
        self.assertFalse(owner._run4_gate.blocked())
        self.assertIsNone(owner.live.gt_log.tickets[1].get("error"))
        self.assertFalse(owner._source_ownership.pending)

    def test_archive_expected_identity_and_digest_tampering_refused(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket); consume(owner, queued)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"owned"; owner._source_ownership.flush(directory)
            with self.assertRaises(S.SourceOwnershipError):
                S.verify_ownership_archive(directory, expected_identities=[identity(2)])
            path = directory/"events.jsonl"
            with path.open("ab") as stream: stream.write(b" ")
            with self.assertRaisesRegex(S.SourceOwnershipError, "digest"):
                S.verify_ownership_archive(directory, expected_identities=[identity(1)])

    def test_shutdown_undrained_is_explicit_incomplete_and_not_qualifiable(self):
        owner, queued = make_owner(); ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"owned"; result = owner._source_ownership.flush(directory)
            self.assertFalse(result["complete"]); self.assertEqual(result["owned_bytes_at_close"], 0)
            self.assertEqual(result["errors"][0]["kind"], "UNDRAINED_SOURCE")
            with self.assertRaises(S.SourceOwnershipError): S.verify_ownership_archive(directory, expected_identities=[identity(1)])

    def test_context_installs_real_factory_and_queue_restores_and_flushes(self):
        original_factory, original_queue = LG.build_live_gt_collector, OG.DeferringRewardPriorityGtQueueV2
        with tempfile.TemporaryDirectory() as temp:
            with S.install_source_ownership(attempt_dir=temp, max_frames=100) as state:
                owner = LG.build_live_gt_collector(Base)()
                self.assertEqual(state["collectors"], [owner])
                self.assertIs(state["inspect_drain"]()["worker_drained"], True)
                queued = OG.DeferringRewardPriorityGtQueueV2(classify=owner.classify, low_blocked=lambda: True,
                    on_skip=lambda f, why: None)
                ticket, _ = feed(owner, 1); queued.put_nowait(ticket); consume(owner, queued)
            self.assertIs(LG.build_live_gt_collector, original_factory)
            self.assertIs(OG.DeferringRewardPriorityGtQueueV2, original_queue)
            self.assertTrue(state["manifest"]["complete"])
            self.assertIs(state["drained"], True)
            self.assertTrue(S.verify_ownership_archive(Path(temp)/"source_ownership", expected_identities=[identity(1)])["passed"])

    def test_alive_worker_retains_pending_and_only_incomplete_snapshot_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            with S.install_source_ownership(attempt_dir=temp, max_frames=100) as state:
                owner = LG.build_live_gt_collector(Base)()
                owner.evaluation_worker = NS(is_alive=lambda: True)
                queued = OG.DeferringRewardPriorityGtQueueV2(classify=owner.classify,
                    low_blocked=lambda: True, on_skip=lambda f, why: None)
                ticket, _ = feed(owner, 1); queued.put_nowait(ticket)
                self.assertIs(state["inspect_drain"]()["worker_drained"], False)
            manifest = state["manifest"]
            self.assertFalse(manifest["complete"]); self.assertFalse(manifest["worker_drained"])
            self.assertTrue(manifest["worker_alive"])
            self.assertEqual(manifest["pending_at_close"], 1)
            self.assertEqual(set(owner._source_ownership.pending), {1})
            self.assertIsNone(owner._source_ownership.manifest)
            consume(owner, queued)  # in-memory source remains usable after failed shutdown snapshot
            self.assertFalse(owner._source_ownership.pending)
            with self.assertRaises(S.SourceOwnershipError):
                S.verify_ownership_archive(Path(temp)/"source_ownership", expected_identities=[identity(1)])

    def test_unknown_worker_or_partial_initialization_never_claims_drain(self):
        for create_collector in (False, True):
            with self.subTest(create_collector=create_collector), tempfile.TemporaryDirectory() as temp:
                with S.install_source_ownership(attempt_dir=temp, max_frames=100) as state:
                    if create_collector:
                        owner = LG.build_live_gt_collector(Base)()
                        del owner.evaluation_worker
                    self.assertIs(state["inspect_drain"]()["worker_drained"], False)
                    self.assertIsNone(state["inspect_drain"]()["worker_alive"])
                self.assertFalse(state["manifest"]["complete"])
                self.assertFalse(state["manifest"]["worker_drained"])
                self.assertFalse(state["drained"])
                self.assertEqual(state["manifest"]["errors"][0]["kind"], "WORKER_DRAIN_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
