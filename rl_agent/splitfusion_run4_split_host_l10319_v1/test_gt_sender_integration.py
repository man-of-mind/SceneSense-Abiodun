"""Offline socketpair/loopback tests for the W10275 HIGH GT sender seam."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
import uuid

import numpy as np

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_gt_handoff_v2 as GH
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE

from . import gt_sender_integration as S
from . import gt_transport as GT


def gt_identity(*, frame: int = 1312, anchor: bool = False) -> dict:
    return {
        "run_id": "run4-live", "cell_id": "a71__favorable_stable",
        "stream_id": "ue288_a71__favorable_stable", "frame_id": frame,
        "action_id": 71 if anchor else None,
        "profile_id": "split_ae32_uint4_q9800" if anchor else None,
        "capture_timestamp_ns": 123_456_789_000 + frame,
    }


def run4_identity(*, frame: int = 1312, reward: bool = True,
                  anchor: bool = False) -> dict:
    return {
        "session_uuid": str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
        "controller_lineage_sha256": "a" * 64,
        "decision_seq": 4, "ticket_seq": 2, "frame_id": frame,
        "tensor_seq": 9, "mode_id": 11, "q_e4": 6784,
        "execution_bundle_sha256": "b" * 64,
        "anchor_action_id": 71 if anchor else None,
        "reward_requested": reward,
        "frame_kind": "POLICY_DECISION" if reward else "POLICY_HOLD",
    }


def full_identity(*, frame: int = 1312, anchor: bool = False) -> GT.GtTransportIdentityV1:
    value = S.identity_from_ue_maps(
        gt_identity=gt_identity(frame=frame, anchor=anchor),
        run4_identity=run4_identity(frame=frame, anchor=anchor))
    assert value is not None
    return value


def writers(log: Path):
    recorder = GH.GtWriteRecorderV2(log)
    return recorder.wrap(GE.write_object_ground_truth, GE.write_semantic_ground_truth)


def invoke_high(function, *args, **kwargs) -> list[BaseException]:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            function(*args, **kwargs)
        except BaseException as exc:  # recorded for the caller's assertion
            errors.append(exc)

    thread = threading.Thread(target=target, name="existing-high-gt-worker")
    thread.start()
    thread.join(2)
    if thread.is_alive():
        raise AssertionError("HIGH worker did not finish")
    return errors


class IdentityResolverTests(unittest.TestCase):
    def test_reward_and_off_anchor_identity(self) -> None:
        result = full_identity(anchor=False)
        self.assertIsNone(result.action_id)
        self.assertIsNone(result.profile_id)
        self.assertEqual(result.frame_id, 1312)

    def test_hold_returns_none_and_drift_fails(self) -> None:
        self.assertIsNone(S.identity_from_ue_maps(
            gt_identity=gt_identity(), run4_identity=run4_identity(reward=False)))
        bad = run4_identity(frame=1313)
        with self.assertRaises(S.GtSenderIntegrationError):
            S.identity_from_ue_maps(gt_identity=gt_identity(), run4_identity=bad)


class SenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.edge = self.root / "edge"
        self.source.mkdir()
        self.edge.mkdir()
        self.ident = full_identity()
        self.registry = GT.ExpectedTicketRegistryV1(
            run_id=self.ident.run_id, cell_id=self.ident.cell_id)
        self.registry.authorize(self.ident)
        self.store = GT.GtIngressStoreV1(self.edge, self.registry)
        self.log = self.root / "writes.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _resolver(identity):
        return S.identity_from_ue_maps(
            gt_identity=identity, run4_identity=run4_identity())

    @staticmethod
    def _high_thread() -> bool:
        return threading.current_thread().name == "existing-high-gt-worker"

    def _sender(self, connector, **changes) -> S.HighWorkerGtSenderV1:
        values = dict(resolve_high_identity=self._resolver,
                      is_high_worker=self._high_thread,
                      connector=connector, connect_timeout_s=0.5,
                      ack_timeout_s=0.5, component_wait_s=0.2)
        values.update(changes)
        return S.HighWorkerGtSenderV1(**values)

    def _write_semantic(self, function) -> None:
        function(self.source, identity=gt_identity(), frozen_carla_frame_id=1312,
                 mask=np.full((5, 7), 3, dtype=np.uint8))

    def _write_objects(self, function) -> None:
        function(self.source, identity=gt_identity(), frozen_carla_frame_id=1312,
                 rows=[{"label": "person", "gt_actor_id": "42"}])

    def test_socketpair_sends_after_recorder_and_only_from_high_worker(self) -> None:
        client, server = socket.socketpair()
        connector_calls = []

        def connector(address, timeout):
            connector_calls.append((address, timeout))
            return client

        sender = self._sender(connector)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        result = []

        def receive() -> None:
            result.append(GT.serve_one(server, self.store, socket_timeout_s=0.5,
                                       expectation_timeout_s=0))

        receiver = threading.Thread(target=receive)
        receiver.start()
        self._write_semantic(semantic)       # record-only; no socket send here
        self.assertTrue(receiver.is_alive())
        errors = invoke_high(self._write_objects, objects)
        receiver.join(1)
        self.assertEqual(errors, [])
        self.assertEqual(result[0].status, "STORED")
        self.assertEqual(connector_calls, [((S.REGISTERED_HOST, S.REGISTERED_PORT), 0.5)])
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([row["kind"] for row in rows], ["semantic", "semantic", "objects"])
        report = sender.close()
        self.assertTrue(report["closed"])
        self.assertFalse(report["connected"])
        self.assertEqual(report["counters"]["stored"], 1)
        server.close()

    def test_object_first_waits_for_semantic_but_send_stays_on_high_worker(self) -> None:
        client, server = socket.socketpair()
        sender = self._sender(lambda _address, _timeout: client,
                              component_wait_s=0.5)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        result, errors = [], []

        remote = threading.Thread(target=lambda: result.append(
            GT.serve_one(server, self.store, socket_timeout_s=0.8,
                         expectation_timeout_s=0)))
        remote.start()

        worker = threading.Thread(
            target=lambda: self._capture(errors, self._write_objects, objects),
            name="existing-high-gt-worker")
        worker.start()
        time.sleep(0.03)
        self.assertTrue(worker.is_alive())
        self._write_semantic(semantic)
        worker.join(1)
        remote.join(1)
        self.assertEqual(errors, [])
        self.assertEqual(result[0].status, "STORED")
        sender.close()
        server.close()

    @staticmethod
    def _capture(errors, function, *args) -> None:
        try:
            function(*args)
        except BaseException as exc:
            errors.append(exc)

    def test_unknown_ticket_failure_is_local_and_later_exact_send_succeeds(self) -> None:
        client, server = socket.socketpair()
        registry = GT.ExpectedTicketRegistryV1(
            run_id=self.ident.run_id, cell_id=self.ident.cell_id)
        store = GT.GtIngressStoreV1(self.edge, registry)
        sender = self._sender(lambda _address, _timeout: client)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(
            recorded_objects, recorded_semantic)
        self._write_semantic(semantic)

        first_result = []
        first_peer = threading.Thread(target=lambda: first_result.append(
            GT.serve_one(server, store, socket_timeout_s=0.5,
                         expectation_timeout_s=0)))
        first_peer.start()
        first_errors = invoke_high(self._write_objects, objects)
        first_peer.join(1)
        self.assertEqual(len(first_errors), 1)
        self.assertIsInstance(first_errors[0], GT.RemoteTicketRefusalError)
        self.assertEqual(first_result[0].error_code, "UNKNOWN_OR_FUTURE_TICKET")
        snapshot = sender.snapshot()
        self.assertIsNone(snapshot["fault"])
        self.assertEqual(snapshot["counters"]["unknown_future_ticket_refusals"], 1)

        registry.authorize(self.ident)
        second_result = []
        second_peer = threading.Thread(target=lambda: second_result.append(
            GT.serve_one(server, store, socket_timeout_s=0.5,
                         expectation_timeout_s=0)))
        second_peer.start()
        self.assertEqual(invoke_high(sender._send_joined, self.ident), [])
        second_peer.join(1)
        self.assertEqual(second_result[0].status, "STORED")
        self.assertIsNone(sender.snapshot()["fault"])
        self.assertEqual(sender.snapshot()["counters"]["stored"], 1)
        sender.close()
        server.close()

    def test_nonordering_remote_refusal_still_faults_sender(self) -> None:
        client, server = socket.socketpair()
        sender = self._sender(lambda _address, _timeout: client)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(
            recorded_objects, recorded_semantic)
        self._write_semantic(semantic)

        def refuse() -> None:
            bundle = GT.recv_bundle(server)
            ack = GT.GtAckV1(
                "REJECTED", bundle.identity.exact_digest(),
                bundle.bundle_sha256, None, "UNAUTHORIZED_TICKET")
            server.sendall(ack.to_wire())

        peer = threading.Thread(target=refuse)
        peer.start()
        errors = invoke_high(self._write_objects, objects)
        peer.join(1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GT.RemoteTicketRefusalError)
        snapshot = sender.snapshot()
        self.assertIn("RemoteTicketRefusalError", snapshot["fault"])
        self.assertEqual(snapshot["counters"]["unknown_future_ticket_refusals"], 0)
        sender.close()
        server.close()

    def test_wrong_thread_refuses_before_any_network_send(self) -> None:
        client, server = socket.socketpair()
        sender = self._sender(lambda _address, _timeout: client)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        self._write_semantic(semantic)
        with self.assertRaises(S.WrongWorkerError):
            self._write_objects(objects)
        server.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            server.recv(1)
        self.assertIn("WrongWorkerError", sender.snapshot()["fault"])
        sender.close()
        server.close()

    def test_low_object_writer_never_sends(self) -> None:
        client, server = socket.socketpair()
        sender = S.HighWorkerGtSenderV1(
            resolve_high_identity=lambda _identity: None,
            is_high_worker=lambda: False,
            connector=lambda _address, _timeout: client,
            connect_timeout_s=0.5, ack_timeout_s=0.5, component_wait_s=0.2)
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        self._write_semantic(semantic)
        self._write_objects(objects)
        server.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            server.recv(1)
        self.assertEqual(sender.snapshot()["counters"]["low_not_sent"], 1)
        sender.close()
        server.close()

    def test_lost_ack_reconnects_once_and_identical_retry_is_accepted(self) -> None:
        client1, server1 = socket.socketpair()
        client2, server2 = socket.socketpair()
        clients = [client1, client2]
        sender = self._sender(lambda _address, _timeout: clients.pop(0))
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        self._write_semantic(semantic)
        remote_results = []

        def lose_first_ack() -> None:
            bundle = GT.recv_bundle(server1)
            remote_results.append(self.store.accept(bundle, expectation_timeout_s=0))
            server1.close()                    # stored, ACK deliberately lost

        def accept_retry() -> None:
            remote_results.append(GT.serve_one(
                server2, self.store, socket_timeout_s=0.5,
                expectation_timeout_s=0))

        first = threading.Thread(target=lose_first_ack)
        second = threading.Thread(target=accept_retry)
        first.start()
        second.start()
        errors = invoke_high(self._write_objects, objects)
        first.join(1)
        second.join(1)
        self.assertEqual(errors, [])
        self.assertEqual([value.status for value in remote_results],
                         ["STORED", "DUPLICATE_IDENTICAL"])
        counters = sender.snapshot()["counters"]
        self.assertEqual(counters["connections"], 2)
        self.assertEqual(counters["reconnections_after_lost_ack"], 1)
        self.assertEqual(counters["duplicate_identical"], 1)
        sender.close()
        server2.close()

    def test_explicit_connect_and_close_over_loopback(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        local_address = listener.getsockname()
        requested = []

        def connector(address, timeout):
            requested.append(address)
            return socket.create_connection(local_address, timeout)

        sender = self._sender(connector)
        remote_result = []

        def remote() -> None:
            stream, _peer = listener.accept()
            try:
                remote_result.append(GT.serve_one(
                    stream, self.store, socket_timeout_s=0.8,
                    expectation_timeout_s=0))
            finally:
                stream.close()

        thread = threading.Thread(target=remote)
        thread.start()
        sender.connect()
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        self._write_semantic(semantic)
        self.assertEqual(invoke_high(self._write_objects, objects), [])
        thread.join(1)
        self.assertEqual(requested, [(S.REGISTERED_HOST, S.REGISTERED_PORT)])
        self.assertEqual(remote_result[0].status, "STORED")
        sender.close()
        with self.assertRaises(S.SenderLifecycleError):
            sender.connect()
        listener.close()

    def test_connect_is_required_and_timeout_is_bounded(self) -> None:
        client, server = socket.socketpair()
        sender = self._sender(lambda _address, _timeout: client)
        recorded_objects, recorded_semantic = writers(self.log)
        objects, semantic = sender.wrap_after_recorder(recorded_objects, recorded_semantic)
        self._write_semantic(semantic)
        errors = invoke_high(self._write_objects, objects)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], S.SenderLifecycleError)
        with self.assertRaises(S.GtSenderIntegrationError):
            self._sender(lambda _a, _t: client, ack_timeout_s=6.0)
        server.close()
        client.close()

    def test_pending_identity_registry_refuses_4097th_identity(self) -> None:
        sender = S.HighWorkerGtSenderV1(
            resolve_high_identity=lambda _identity: None,
            is_high_worker=lambda: False,
            connector=lambda _address, _timeout: None,
            connect_timeout_s=0.5, ack_timeout_s=0.5, component_wait_s=0.2)

        def semantic(directory, *, identity, **_kwargs):
            stem = str(identity["frame_id"])
            return directory / f"{stem}.semantic.npy", directory / f"{stem}.semantic.json"

        _objects, wrapped = sender.wrap_after_recorder(lambda *_a, **_k: None, semantic)
        for frame_id in range(S.MAX_PENDING_IDENTITIES):
            wrapped(self.source, identity=gt_identity(frame=frame_id))
        with self.assertRaisesRegex(S.SenderDeliveryError, "registry is full"):
            wrapped(self.source, identity=gt_identity(frame=S.MAX_PENDING_IDENTITIES))
        self.assertEqual(sender.snapshot()["pending_keys"], S.MAX_PENDING_IDENTITIES)
        sender.close()


class PurityTests(unittest.TestCase):
    def test_import_has_no_network_entrypoint_or_background_thread(self) -> None:
        source = Path(S.__file__).read_text(encoding="utf-8")
        self.assertNotIn("if __name__", source)
        self.assertNotIn(".bind(", source)
        self.assertNotIn(".listen(", source)
        self.assertNotIn("threading.Thread(", source)


if __name__ == "__main__":
    unittest.main()
