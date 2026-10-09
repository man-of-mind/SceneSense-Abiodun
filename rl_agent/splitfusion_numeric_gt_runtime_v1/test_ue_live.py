"""CPU-only checks of the prospective UE handoff's exact owner events."""
import threading
from collections import deque
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

from . import ue_live as U
from . import staging
from .test_source_codec import fixture as source_fixture
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C


BOOT = "12345678-1234-1234-1234-123456789abc"
DIGEST = "a" * 64


def planned(kind="POLICY_DECISION", reward=True):
    profile = NS(action_id=3, profile_id="mode3", mode_id=3, q_e4=4000,
                 execution_bundle_sha256=DIGEST)
    frame = NS(session_uuid="session", decision_seq=1, ticket_seq=1,
               reward_requested=reward)
    plan = NS(frame_identity=frame, frame_id=12, tensor_seq=7,
              capture_wall_ns=123456789, profile=profile, kind=NS(value=kind))
    return NS(plan=plan, frame_id=12, tensor_seq=7, capture_wall_ns=123456789)


def capture():
    return dict(run_id="run", cell_id="cell", stream_id="stream", frame_id=12,
                action_id=None, profile_id=None, capture_timestamp_ns=123456789)


def collector_for_source(source, stamp, rgb=None):
    frame = source["identity"]["frame_id"]
    rgb = dict(source["rgb"]) if rgb is None else rgb
    return NS(_quality_scene_lock=threading.Lock(),
        _quality_scenes={frame: source["frozen_world"]},
        _avo_condition=threading.Condition(), _avo_rgb={frame: rgb},
        _avo_semantic={frame: (source["semantic_image"], rgb["semantic_callback"])},
        _avo_depth={frame: (source["depth_image"], rgb["depth_callback"])},
        _avo_radar={frame: rgb["radar_callback"]},
        _avo_callbacks_complete={frame: {"rgb", "semantic", "depth", "radar"}},
        _numeric_radar_lock=threading.Lock(),
        _numeric_radar={stamp: {"world_xyz": source["radar_world_xyz"]}})


class FakeClient:
    def __init__(self, job):
        self.job = job
        self.calls = []
        self.identity_bytes = b"identity"
        self.semantic_sha256 = DIGEST
        self.pending = None

    def finish(self):
        self.calls.append("finish")
        return dict(job=self.job, identity_bytes=self.identity_bytes,
                    semantic_sha256=self.semantic_sha256)

    def call(self, operation, value):
        self.calls.append(operation)
        if operation == "PUBLISH_READY":
            return b"exact target"
        if operation == "EXPORT_LIVE_SOURCE":
            return dict(schema="source", identity_bytes=self.identity_bytes,
                        semantic_sha256=self.semantic_sha256)
        if operation == "PREPARE_LIVE_UNSCORED_TERMINAL":
            files = {"live_clock.json": b"clock", "unscored_terminal.json": b"terminal"}
            return dict(identity_bytes=self.identity_bytes, semantic_sha256=DIGEST,
                        files=files, terminal_sha256=C.sha256(files["unscored_terminal.json"]))
        return {"status": "OK"}


class FakePublisher:
    targets = []
    terminals = []
    unscored_files = []
    completion_sequence = []
    target_seen = threading.Event()

    @classmethod
    def publish_target(cls, path, **kwargs):
        cls.targets.append((path, kwargs))
        cls.target_seen.set()

    @classmethod
    def publish_terminal(cls, path, **kwargs):
        cls.terminals.append((path, kwargs))

    @classmethod
    def publish_unscored_files(cls, path, **kwargs):
        cls.unscored_files.append((path, kwargs))

    @classmethod
    def await_completion(cls, path, **kwargs):
        if cls.completion_sequence:
            return cls.completion_sequence.pop(0)
        return {"status": "SCORED_DURABLE", "scorer_durable_proof": {"status": "DURABLE"}}


def handoff():
    value = U.NumericUeHandoff.__new__(U.NumericUeHandoff)
    value.client = FakeClient(1)
    value._lock = threading.RLock()
    value._producer_lock = threading.RLock()
    value._producer_turn = threading.Condition(value._lock)
    value._producer_order = deque()
    value._aborted = False
    value._tickets = {}
    value.errors = []
    value.publisher = FakePublisher
    value.socket_path = Path("/tmp/test_numeric_gt.sock")
    value.pid = 101
    value.boot_id = BOOT
    value.registration = {"numeric_limits": {"process_timeout_s": 1., "retained_jobs": 2}}
    value.collector = NS(live=NS(_feedback_rows=[]))
    ticket = U._Ticket(1, 12, capture(), {},
        U.event("collector", "owned_tick_snapshot", 100, 101, BOOT),
        source_owned=U.event("ue", "source_owned", 110, 101, BOOT))
    ticket.plan_bytes = b"full plan"
    ticket.action_open = U.event("ue", "action_open", 120, 101, BOOT)
    # These tests cover terminal/retirement behavior. The independent async
    # integration suite exercises real capture/encode/stage and slow IPC.
    def prepare_sent(current):
        if current.last_send is None:
            raise AssertionError("source staging before actual final send")
        current.stage_event = U.event("producer", "source_stage_received", 140, 102, BOOT)
        return value.client.finish()
    value._prepare_sent = prepare_sent
    value._tickets[12] = ticket
    return value, ticket


class NumericUeTests(unittest.TestCase):
    def setUp(self):
        FakePublisher.targets.clear()
        FakePublisher.terminals.clear()
        FakePublisher.unscored_files.clear()
        FakePublisher.completion_sequence.clear()
        FakePublisher.target_seen.clear()

    def test_owned_live_pose_matrices_derive_strict_inverse(self):
        # Exact matrices from failed live pose archives, frames 1435 and 1671.
        # The separately rounded CARLA inverse crosses both the producer's
        # 1e-8 limit and, farther along the route, the former UE's 1e-5 gate.
        camera_1435 = np.asarray([
            [0.9980416893959045, 0.017424969002604485, 0.060075793415308, 2.1519744396209717],
            [-0.01897302083671093, 0.9994999170303345, 0.02529490739107132, 28.036731719970703],
            [-0.05960499122738838, -0.026385193690657616, 0.9978732466697693, 1.4952316284179688],
            [0., 0., 0., 1.],
        ])
        native_1435 = np.asarray([
            [0.9980416893959045, -0.01897302083671093, -0.05960499122738838, -1.5266954898834229],
            [0.017424969002604485, 0.9994999170303345, -0.026385195553302765, -28.0207576751709],
            [0.060075793415308, 0.02529490925371647, 0.9978732466697693, -2.330519676208496],
            [0., 0., 0., 1.],
        ])
        camera_1671 = np.asarray([
            [0.9971386194229126, -0.014560890384018421, 0.07417835295200348, 65.99640655517578],
            [0.013540549203753471, 0.9998068809509277, 0.014239662326872349, 27.951190948486328],
            [-0.07437137514352798, -0.013194502331316471, 0.9971432685852051, 1.5828070640563965],
            [0., 0., 0., 1.],
        ])
        native_1671 = np.asarray([
            [0.9971386194229126, 0.013540549203753471, -0.07437137514352798, -66.06832122802734],
            [-0.014560890384018421, 0.9998068809509277, -0.013194501399993896, -26.963943481445312],
            [0.07417835295200348, 0.014239661395549774, 0.9971432685852051, -6.8718061447143555],
            [0., 0., 0., 1.],
        ])
        for frame, camera, native in ((1435, camera_1435, native_1435),
                                      (1671, camera_1671, native_1671)):
            with self.subTest(frame=frame):
                original = camera.copy()
                native_residual = np.max(np.abs(camera @ native - np.eye(4)))
                self.assertGreater(native_residual, 1e-8)
                self.assertEqual(native_residual > 1e-5, frame == 1671)
                strict_inverse = U._strict_camera_inverse(camera)
                self.assertTrue(np.array_equal(camera, original))
                self.assertLess(np.max(np.abs(camera @ strict_inverse - np.eye(4))), 1e-8)
        with self.assertRaisesRegex(U.NumericUeError, "singular"):
            U._strict_camera_inverse(np.zeros((4, 4)))

    def test_full_plan_exact_action_and_context(self):
        encoded, selected, context, action = U.full_plan_bytes(planned(), lineage=DIGEST,
            capture_template=capture(), ego_pose=(0., 0., 0., 0., 0., 0.),
            stream_id="stream")
        self.assertEqual(C._parse(encoded)["capture"], selected)
        self.assertEqual(action["frame_kind"], "POLICY_DECISION")
        self.assertEqual(selected["action_id"], 3)
        self.assertEqual(context["sequence_id"], 7)
        for kind, reward in (("POLICY_HOLD", False), ("FALLBACK", False)):
            with self.assertRaises(U.NumericUeError):
                U.full_plan_bytes(planned(kind, reward), lineage=DIGEST,
                    capture_template=capture(), ego_pose=(0.,)*6, stream_id="stream")

    def test_exact_owned_source_frozen_before_stage_and_radar_consumed_once(self):
        source, stamp = source_fixture()
        frame = source["identity"]["frame_id"]
        collector = collector_for_source(source, stamp)
        actual = U.capture_owned_source(collector, frame_id=frame,
            carla_timestamp=stamp, capture_identity=source["identity"])
        self.assertEqual(actual["rgb"]["_rgb_image"].raw_data,
                         source["rgb"]["_rgb_image"].raw_data)
        chunks, extras = staging.encode_stage(actual, stamp)
        self.assertTrue(b"".join(chunks))
        self.assertEqual(extras["source_extras"]["capture_identity"], source["identity"])
        with self.assertRaises(U.NumericUeError):
            U.capture_owned_source(collector, frame_id=frame,
                carla_timestamp=stamp, capture_identity=source["identity"])

    def test_native_inverse_getter_unused_after_owned_rgb_freeze(self):
        source, stamp = source_fixture()
        rgb = dict(source["rgb"])
        original = rgb["_rgb_image"]
        def forbidden_inverse():
            raise AssertionError("raw CARLA native inverse must not be queried")
        rgb["_rgb_image"] = replace(original, transform=NS(
            get_matrix=original.transform.get_matrix,
            get_inverse_matrix=forbidden_inverse))
        collector = collector_for_source(source, stamp, rgb)
        actual = U.capture_owned_source(collector,
            frame_id=source["identity"]["frame_id"], carla_timestamp=stamp,
            capture_identity=source["identity"])
        self.assertTrue(np.array_equal(actual["camera_matrix"], np.asarray(rgb["camera_matrix"])))
        self.assertLess(np.max(np.abs(actual["camera_matrix"] @ actual["camera_inverse"]
                                      - np.eye(4))), 1e-8)
        staging.encode_stage(actual, stamp)

    def test_foreign_callback_frame_timestamp_or_pose_still_refused(self):
        for changed in ("frame_id", "timestamp", "camera_matrix"):
            with self.subTest(changed=changed):
                source, stamp = source_fixture()
                rgb = dict(source["rgb"])
                if changed == "camera_matrix":
                    rgb[changed] = np.eye(4).tolist()
                else:
                    rgb[changed] += 1
                collector = collector_for_source(source, stamp, rgb)
                with self.assertRaisesRegex(RuntimeError, "image/callback"):
                    U.capture_owned_source(collector,
                        frame_id=source["identity"]["frame_id"], carla_timestamp=stamp,
                        capture_identity=source["identity"])

    def test_process_local_factory_wraps_final_run4_radar_method(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as R
        old = (LG.build_live_gt_collector, R.build_run4_collector_class)
        with tempfile.TemporaryDirectory() as temporary:
            with U.install_numeric_ue(attempt_dir=Path(temporary), registration={},
                                      publisher=FakePublisher, client_factory=FakeClient):
                inner = LG.build_live_gt_collector(object)
                final = R.build_run4_collector_class(inner)
                self.assertEqual(final.__mro__[0].__name__, "NumericFinalCollector")
                self.assertEqual(final.__mro__[1].__name__, "Run4PassiveSplitCollectorV2")
                self.assertEqual(final.__mro__[2].__name__, "NumericCollector")
                self.assertIs(final.__dict__["_run4_timed_build"],
                              getattr(final, "_run4_timed_build"))
        self.assertEqual((LG.build_live_gt_collector, R.build_run4_collector_class), old)

    def test_final_collector_constructor_bypasses_legacy_gt_and_closes_gate(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as R
        leaf_calls = []
        gate_closes = []
        world = object()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            class Base:
                def __init__(self, **kwargs):
                    self.world = world
                    self.edge_evidence_dir = directory
                    self._quality_scene_lock = threading.Lock()
                    self._quality_scenes = {12: world}
                    self.live = NS(pipeline=NS(plan=lambda **kw: None),
                        engine=NS(on_feedback=lambda *a, **kw: None))
                def _ground_truth(self, **kwargs):
                    leaf_calls.append(kwargs)
                    return ["old GT"]
            with U.install_numeric_ue(attempt_dir=directory, registration={},
                                      publisher=FakePublisher, client_factory=FakeClient) as bridge:
                bridge.catalogue.world = world
                inner = LG.build_live_gt_collector(Base)
                final = R.build_run4_collector_class(inner)
                run4 = final.__mro__[1]
                def init_run4(self, **kwargs):
                    super(run4, self).__init__(**kwargs)
                    self._run4_gate = NS(close=lambda frame, reason:
                        gate_closes.append((frame, reason)))
                def gt_run4(self, **kwargs):
                    return super(run4, self)._ground_truth(**kwargs)
                def radar_run4(self, real_build, **kwargs):
                    return (None, {"world_xyz": np.zeros((0, 3))}, None)
                with mock.patch.object(run4, "__init__", init_run4), \
                     mock.patch.object(run4, "_ground_truth", gt_run4), \
                     mock.patch.object(run4, "_run4_timed_build", radar_run4), \
                     mock.patch.object(bridge, "prestart_from_scene") as prestart:
                    owner = final()
                    self.assertIs(bridge.collector, owner)
                    prestart.assert_called_once_with()
                    owner._run4_timed_build(None, frame_time_s=1.)
                    self.assertIn(1., owner._numeric_radar)
                    self.assertEqual(owner._ground_truth(world=world, frame_id=12), [])
                self.assertEqual(leaf_calls, [])
                self.assertEqual(gate_closes,
                                 [(12, "NUMERIC_GT_LEGACY_BYPASS")])

    def test_runtime_submit_uses_owned_pose_document_not_legacy_ego(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as R
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose
        observed = []
        class BaseRuntime:
            def submit(self, **kwargs):
                observed.append(tuple(self._numeric_current_ego_pose))
                observed.append(tuple(kwargs["ego_pose"]))
                return {"sent": True}
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(R, "build_run4_runtime_class", return_value=BaseRuntime):
                with U.install_numeric_ue(attempt_dir=Path(temporary), registration={},
                                          publisher=FakePublisher, client_factory=FakeClient) as bridge:
                    owner = object()
                    bridge.collector = owner
                    runtime = R.build_run4_runtime_class(None)()
                    runtime._numeric_handoff = bridge
                    pose._LOCAL.binding = {"owner": owner,
                        "source": NS(document=dict(frame_id=12, stream_id="stream",
                                                   timestamp=1., ego_world=[1., 2., 3., 4., 5., 6.]))}
                    try:
                        with mock.patch.object(bridge, "stage_before_action") as staged, \
                             mock.patch.object(bridge, "after_submit") as after:
                            runtime.submit(frame_bgr=None, radar_tensor=None, frame_id=12,
                                capture_timestamp_ns=123456789, ego_pose=(9.,)*6,
                                stream_id="stream", carla_timestamp=1.,
                                capture_id="stream:12")
                        staged.assert_called_once()
                        after.assert_called_once()
                    finally:
                        pose._LOCAL.binding = None
        self.assertEqual(observed, [(1., 2., 3., 4., 5., 6.), (9.,)*6])

    def test_shutdown_failure_aborts_only_owned_producer(self):
        class FailedClient:
            closed = False
            def __init__(self):
                self.aborted = False
                self.events = [dict(seq=1, operation="PREPARE_STAGED", status="ERROR")]
                self.pending = None
                self.ready = dict(status="READY", pid=102)
            def close(self):
                raise RuntimeError("owned producer close failed")
            def abort(self):
                self.aborted = True
        with tempfile.TemporaryDirectory() as temporary:
            bridge, ticket = handoff()
            ticket.fault = "frame 12: producer failed"
            bridge.errors.append(ticket.fault)
            bridge.client = FailedClient()
            bridge.attempt_dir = Path(temporary)
            bridge.directory = Path(temporary) / "numeric_gt"
            bridge.directory.mkdir()
            with self.assertRaisesRegex(RuntimeError, "owned producer close failed"):
                bridge.close()
            self.assertTrue(bridge.client.aborted)
            fault = C._parse((bridge.directory / "UE_ABORT_FAULT.json").read_bytes())
            self.assertEqual(fault["status"], "PRE_ABORT_FAILURE_EVIDENCE")
            self.assertEqual(fault["tickets"][0]["fault"], ticket.fault)
            self.assertEqual(fault["producer_events"], bridge.client.events)

    def test_target_only_after_actual_last_send_and_terminal_only_after_receipt(self):
        bridge, ticket = handoff()
        runtime = NS(last_datagram={12: 130})
        bridge.after_submit(runtime, 12, {"sent": True})
        self.assertTrue(FakePublisher.target_seen.wait(2.))
        self.assertTrue(ticket.coordinator_started.is_set())
        self.assertEqual(FakePublisher.targets[0][1]["packet"], b"exact target")
        self.assertEqual(FakePublisher.terminals, [])
        bridge.collector.live._feedback_rows.append(dict(frame_id=12,
            receipt=dict(domain="CLOCK_MONOTONIC_RAW", ns=150),
            sha256=DIGEST, kind="QUALITY", reason="OK", q_perc=70.,
            **{"class": "ACCEPTED"}))
        bridge.on_feedback(12, 150)
        ticket.thread.join(2.)
        self.assertFalse(ticket.thread.is_alive())
        self.assertEqual(bridge.errors, [])
        self.assertEqual(bridge.client.calls[:4],
                         ["finish", "CONFIRM_SEND", "PUBLISH_READY", "EXPORT_LIVE_SOURCE"])
        self.assertEqual(bridge.client.calls[-2:], ["SCORE_COMPLETED", "COMPLETE_LIVE_SCORED"])
        events = C._parse(FakePublisher.terminals[0][1]["ue_events_bytes"])["events"]
        self.assertEqual([row["name"] for row in events],
            ["owned_tick_snapshot", "source_owned", "action_open",
             "last_datagram_sent", "source_stage_received", "feedback_received"])

    def test_no_actual_send_releases_locally_and_never_publishes(self):
        bridge, ticket = handoff()
        bridge.after_submit(NS(last_datagram={}), 12, {"sent": True})
        self.assertIsNone(ticket.thread)
        self.assertIsNone(ticket.owned_source)
        self.assertTrue(ticket.retired)
        self.assertEqual(bridge.client.calls, [])
        self.assertEqual(FakePublisher.targets, [])

    def test_hold_or_stale_before_plan_never_enters_producer(self):
        bridge, ticket = handoff()
        ticket.plan_bytes = None
        bridge.after_submit(NS(last_datagram={}), 12, {"sent": False})
        self.assertEqual(bridge.client.calls, [])
        self.assertIsNone(ticket.owned_source)
        self.assertTrue(ticket.retired)
        self.assertEqual(FakePublisher.targets, [])

    def test_timeout_is_actual_controller_poll_not_inferred_deadline(self):
        bridge, ticket = handoff()
        runtime = NS(last_datagram={12: 130})
        bridge.after_submit(runtime, 12, {"sent": True})
        before = NS(resolution=None)
        after = NS(resolution=object(), resolved_by="TIMEOUT",
                   reward_frame=NS(frame_id=12))
        bridge.on_controller_poll(before, None, 200_000_121, after)
        ticket.thread.join(2.)
        self.assertFalse(ticket.thread.is_alive())
        events = C._parse(FakePublisher.terminals[0][1]["ue_events_bytes"])["events"]
        self.assertEqual(events[-1]["name"], "feedback_timeout")
        self.assertEqual(events[-1]["raw_ns"], 200_000_121)

    def test_unscored_timeout_requires_two_durable_files_before_retirement(self):
        bridge, ticket = handoff()
        FakePublisher.completion_sequence.extend([
            {"status": "UNSCORED_REQUIRED", "clock_descriptor_bytes": b"clock",
             "reason": "FEEDBACK_TIMEOUT", "fault_feedback": None},
            {"status": "UNSCORED_DURABLE", "receipt": {"status": "COMPLETE"},
             "relative_prefix": "numeric_gt/frame_0000000012/",
             "unscored_terminal_sha256": C.sha256(b"terminal")},
        ])
        bridge.after_submit(NS(last_datagram={12: 130}), 12, {"sent": True})
        bridge.on_controller_poll(NS(resolution=None), None, 200_000_121,
            NS(resolution=object(), resolved_by="TIMEOUT", reward_frame=NS(frame_id=12)))
        ticket.thread.join(2.)
        self.assertFalse(ticket.thread.is_alive())
        self.assertEqual(bridge.errors, [])
        self.assertEqual(set(FakePublisher.unscored_files[0][1]["files"]),
                         {"live_clock.json", "unscored_terminal.json"})
        self.assertEqual(bridge.client.calls[-2:],
                         ["PREPARE_LIVE_UNSCORED_TERMINAL", "COMPLETE_LIVE_UNSCORED"])


if __name__ == "__main__":
    unittest.main()
