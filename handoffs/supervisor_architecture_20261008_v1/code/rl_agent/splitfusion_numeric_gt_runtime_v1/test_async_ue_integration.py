"""CPU-only composition regressions for the owned asynchronous UE handoff.

These tests exercise the real source freezer and planner seam with a deliberately
slow producer stand-in. They never start CARLA, OAI, Docker, or a GPU model.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from . import staging, ue_live as U
from .live_clock import read_boot_id
from .test_source_codec import fixture as source_fixture


LINEAGE = "a" * 64


def source_collector(source, stamp):
    frame_id = source["identity"]["frame_id"]
    rgb = dict(source["rgb"])
    return NS(_quality_scene_lock=threading.Lock(),
        _quality_scenes={frame_id: source["frozen_world"]},
        _avo_condition=threading.Condition(), _avo_rgb={frame_id: rgb},
        _avo_semantic={frame_id: (source["semantic_image"], rgb["semantic_callback"])},
        _avo_depth={frame_id: (source["depth_image"], rgb["depth_callback"])},
        _avo_radar={frame_id: rgb["radar_callback"]},
        _avo_callbacks_complete={frame_id: {"rgb", "semantic", "depth", "radar"}},
        _numeric_radar_lock=threading.Lock(),
        _numeric_radar={stamp: {"world_xyz": source["radar_world_xyz"]}},
        live=NS(_feedback_rows=[]))


def planned(frame_id, *, kind="POLICY_DECISION", reward=True):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.run4_map_protocol_v2 import FALLBACK_SEQ
    profile = NS(action_id=3, profile_id="mode3", mode_id=3, q_e4=4000,
                 execution_bundle_sha256=LINEAGE)
    seq = FALLBACK_SEQ if kind == "FALLBACK" else 1
    frame = NS(session_uuid="session", decision_seq=seq, ticket_seq=seq,
               reward_requested=reward)
    plan = NS(frame_identity=frame, frame_id=frame_id, tensor_seq=7,
              capture_wall_ns=123125000000, profile=profile, kind=NS(value=kind))
    return NS(plan=plan, frame_id=frame_id, tensor_seq=7,
              capture_wall_ns=123125000000)


class SlowClient:
    def __init__(self, gate=None, *, fail_stage=False):
        self.gate = gate or threading.Event()
        if gate is None:
            self.gate.set()
        self.fail_stage = fail_stage
        self.stage_entered = threading.Event()
        self.calls = []
        self.stage_source = None
        self.stage_received_ns = None
        self.pending = None
        self.closed = False
        self.aborted = False
        self.ready = {"status": "READY", "pid": os.getpid()}
        self.events = []
        self.process = NS(returncode=0)

    def call(self, operation, value):
        self.calls.append(operation)
        if operation == "STAGE_SOURCE":
            self.stage_entered.set()
            if not self.gate.wait(timeout=2):
                raise RuntimeError("test stage gate was not released")
            if self.fail_stage:
                raise RuntimeError("injected owned stage failure")
            self.stage_source = staging.decode_stage(value["source_packet"], value["stage_extras"])
            self.stage_received_ns = U.raw()
            return dict(job=value["job"], capture_template=self.stage_source["identity"],
                        source_stage_received_raw_ns=self.stage_received_ns,
                        source_stage_owner_pid=os.getpid(), source_stage_boot_id=read_boot_id())
        if operation == "PREPARE_STAGED":
            return dict(job=value["job"], identity_bytes=b"owned identity",
                        semantic_sha256=LINEAGE)
        if operation == "PUBLISH_READY":
            return b"ready target"
        if operation == "EXPORT_LIVE_SOURCE":
            return dict(schema="exact producer source", identity_bytes=b"owned identity",
                        semantic_sha256=LINEAGE)
        return {"status": "OK"}

    def begin(self, operation, value):
        self.calls.append(operation)
        self.pending = (operation, value)

    def finish(self):
        self.calls.append("finish")
        if self.pending is None:
            raise AssertionError("finish without pending producer request")
        operation, value = self.pending
        self.pending = None
        if operation != "PREPARE_STAGED":
            raise AssertionError("unexpected pending operation")
        return dict(job=value["job"], identity_bytes=b"owned identity",
                    semantic_sha256=LINEAGE)

    def close(self):
        self.closed = True
        return {"valid": True}

    def abort(self):
        self.aborted = True


class Publisher:
    targets = []
    terminals = []
    target_seen = threading.Event()

    @classmethod
    def clear(cls):
        cls.targets = []
        cls.terminals = []
        cls.target_seen.clear()

    @classmethod
    def publish_target(cls, path, **kwargs):
        cls.targets.append((path, kwargs))
        cls.target_seen.set()

    @classmethod
    def publish_terminal(cls, path, **kwargs):
        cls.terminals.append((path, kwargs))

    @classmethod
    def await_completion(cls, path, **kwargs):
        return {"status": "SCORED_DURABLE", "scorer_durable_proof": {"status": "DURABLE"}}


def setup_bridge(directory, source, stamp, client):
    frame_id = source["identity"]["frame_id"]
    collector = source_collector(source, stamp)
    bridge = U.NumericUeHandoff(attempt_dir=directory,
        registration={"numeric_limits": {"process_timeout_s": 1.,
                                         "retained_bytes": 128 * 1024 * 1024,
                                         "retained_jobs": 2},
                      "numeric_worker_sources": {}}, publisher=Publisher)
    bridge.collector = collector
    bridge.socket_path = directory / "fake_numeric.sock"
    started = []
    def start(source_value):
        started.append(source_value)
        bridge.client = client
        return client
    runtime = NS(pipeline=NS(run_id=source["identity"]["run_id"],
        cell_id=source["identity"]["cell_id"], stream_id=source["identity"]["stream_id"],
        decisions=[]), engine=NS(lineage=LINEAGE),
        _numeric_current_ego_pose=(0., 0., 0., 0., 0., 0.), last_datagram={})
    return bridge, collector, runtime, started, start, frame_id


class AsyncUeIntegrationTests(unittest.TestCase):
    def setUp(self):
        Publisher.clear()

    def test_slow_stage_does_not_block_action_admission_or_restamp_owned_snapshot(self):
        source, stamp = source_fixture()
        original_rgb = bytes(source["rgb"]["_rgb_image"].raw_data)
        original_camera = np.asarray(source["camera_matrix"]).copy()
        gate = threading.Event()
        client = SlowClient(gate)
        with tempfile.TemporaryDirectory() as temporary:
            bridge, collector, runtime, started, start, frame_id = setup_bridge(
                Path(temporary), source, stamp, client)
            with mock.patch.object(bridge, "_start_client", side_effect=start):
                ticket = bridge.stage_before_action(runtime, frame_id=frame_id,
                    carla_timestamp=stamp,
                    capture_timestamp_ns=source["identity"]["capture_timestamp_ns"],
                    ego_pose=(0.,) * 6)
                self.assertEqual(started, [])  # No producer RPC before admission.
                self.assertIsNone(ticket.stage_event)
                owned_raw = ticket.collector_event["raw_ns"]
                # A later callback replacement must not alter the already owned source.
                collector._avo_rgb[frame_id] = {"foreign": "callback replacement"}
                opened = U.raw()
                runtime.pipeline.decisions.append(dict(frame_id=frame_id,
                    action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=opened)))
                bridge.on_plan(runtime, planned(frame_id))
                self.assertEqual(started, [])
                sent = U.raw()
                runtime.last_datagram[frame_id] = sent
                start_call = time.monotonic()
                bridge.after_submit(runtime, frame_id, {"sent": True})
                self.assertLess(time.monotonic() - start_call, .3,
                    "slow numeric worker must not block retained UE submit")
                try:
                    self.assertTrue(client.stage_entered.wait(1.))
                    self.assertEqual(Publisher.targets, [])
                    self.assertEqual(ticket.collector_event["raw_ns"], owned_raw)
                    self.assertEqual(ticket.action_open["raw_ns"], opened)
                finally:
                    gate.set()
                self.assertTrue(Publisher.target_seen.wait(2.))
                received = U.raw()
                collector.live._feedback_rows.append(dict(frame_id=frame_id,
                    receipt=dict(domain="CLOCK_MONOTONIC_RAW", ns=received),
                    sha256=LINEAGE, kind="QUALITY", reason="OK", q_perc=.5,
                    **{"class": "ACCEPTED"}))
                bridge.on_feedback(frame_id, received)
                ticket.thread.join(timeout=2.)
                self.assertFalse(ticket.thread.is_alive())
                self.assertEqual(bridge.errors, [])
                self.assertGreater(client.stage_received_ns, opened,
                    "actual later producer receipt must never be backdated")
                self.assertEqual(ticket.collector_event["raw_ns"], owned_raw)
                self.assertEqual(bytes(client.stage_source["rgb"]["_rgb_image"].raw_data), original_rgb)
                self.assertTrue(np.array_equal(client.stage_source["camera_matrix"], original_camera))
                self.assertEqual([item["name"] for item in C._parse(Publisher.terminals[0][1][
                    "ue_events_bytes"])["events"]],
                    ["owned_tick_snapshot", "source_owned", "action_open",
                     "last_datagram_sent", "source_stage_received", "feedback_received"])

    def test_hold_and_fallback_never_create_producer_jobs(self):
        for kind in ("POLICY_HOLD", "FALLBACK"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                source, stamp = source_fixture()
                client = SlowClient()
                bridge, _, runtime, started, start, frame_id = setup_bridge(
                    Path(temporary), source, stamp, client)
                with mock.patch.object(bridge, "_start_client", side_effect=start):
                    ticket = bridge.stage_before_action(runtime, frame_id=frame_id,
                        carla_timestamp=stamp,
                        capture_timestamp_ns=source["identity"]["capture_timestamp_ns"],
                        ego_pose=(0.,) * 6)
                    runtime.pipeline.decisions.append(dict(frame_id=frame_id,
                        action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=U.raw())))
                    bridge.on_plan(runtime, planned(frame_id, kind=kind, reward=False))
                    bridge.after_submit(runtime, frame_id, {"sent": True})
                self.assertEqual(started, [])
                self.assertEqual(client.calls, [])
                self.assertIsNone(ticket.owned_source)
                self.assertEqual(Publisher.targets, [])

    def test_native_inverse_getter_is_not_queried_by_async_capture(self):
        source, stamp = source_fixture()
        frame_id = source["identity"]["frame_id"]
        collector = source_collector(source, stamp)
        rgb = dict(collector._avo_rgb[frame_id])
        original = rgb["_rgb_image"]
        rgb["_rgb_image"] = replace(original, transform=NS(
            get_matrix=original.transform.get_matrix,
            get_inverse_matrix=lambda: (_ for _ in ()).throw(
                AssertionError("native inverse must not be used"))))
        collector._avo_rgb[frame_id] = rgb
        with tempfile.TemporaryDirectory() as temporary:
            bridge = U.NumericUeHandoff(attempt_dir=temporary,
                registration={"numeric_limits": {"retained_bytes": 128 * 1024 * 1024}},
                publisher=Publisher)
            bridge.collector = collector
            runtime = NS(pipeline=NS(run_id="codec", cell_id="cpu", stream_id="example"))
            ticket = bridge.stage_before_action(runtime, frame_id=frame_id,
                carla_timestamp=stamp,
                capture_timestamp_ns=source["identity"]["capture_timestamp_ns"],
                ego_pose=(0.,) * 6)
            self.assertLess(np.max(np.abs(ticket.owned_source["camera_matrix"] @
                                             ticket.owned_source["camera_inverse"] - np.eye(4))), 1e-8)

    def test_worker_failure_is_first_fault_and_thread_is_drained(self):
        source, stamp = source_fixture()
        client = SlowClient(fail_stage=True)
        with tempfile.TemporaryDirectory() as temporary:
            bridge, _, runtime, _, start, frame_id = setup_bridge(
                Path(temporary), source, stamp, client)
            with mock.patch.object(bridge, "_start_client", side_effect=start):
                ticket = bridge.stage_before_action(runtime, frame_id=frame_id,
                    carla_timestamp=stamp,
                    capture_timestamp_ns=source["identity"]["capture_timestamp_ns"],
                    ego_pose=(0.,) * 6)
                runtime.pipeline.decisions.append(dict(frame_id=frame_id,
                    action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=U.raw())))
                bridge.on_plan(runtime, planned(frame_id))
                runtime.last_datagram[frame_id] = U.raw()
                bridge.after_submit(runtime, frame_id, {"sent": True})
                ticket.thread.join(timeout=2.)
                self.assertFalse(ticket.thread.is_alive())
                self.assertEqual(len(bridge.errors), 1)
                self.assertIn("injected owned stage failure", bridge.errors[0])
                with self.assertRaisesRegex(Exception, "injected owned stage failure"):
                    bridge.close()
                self.assertTrue((bridge.directory / "UE_ABORT_FAULT.json").is_file())
                self.assertEqual(Publisher.targets, [])


if __name__ == "__main__":
    unittest.main()
