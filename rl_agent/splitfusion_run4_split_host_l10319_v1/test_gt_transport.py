"""CPU/offline tests for the persistent split-host GT byte transport."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import uuid

import numpy as np

from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE

from . import gt_transport as T


def identity(**changes) -> T.GtTransportIdentityV1:
    values = {
        "run_id": "run4-live", "cell_id": "a71__favorable_stable",
        "stream_id": "ue288_a71__favorable_stable",
        "session_uuid": str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
        "controller_lineage_sha256": "a" * 64,
        "decision_seq": 4, "ticket_seq": 2, "frame_id": 1312,
        "tensor_seq": 9, "capture_timestamp_ns": 123_456_789_000,
        "action_id": 71, "profile_id": "split_ae32_uint4_q9800",
    }
    values.update(changes)
    return T.GtTransportIdentityV1(**values)


def write_bundle(root: Path, ident: T.GtTransportIdentityV1 | None = None,
                 *, mask_value: int = 3) -> tuple[T.GtBundleV1, dict[str, Path]]:
    ident = ident or identity()
    objects = GE.write_object_ground_truth(
        root, identity=ident.gt_identity(), frozen_carla_frame_id=ident.frame_id,
        rows=[{"label": "person", "gt_actor_id": "42", "object_world_x": 1.5}],
    )
    semantic_npy, semantic_json = GE.write_semantic_ground_truth(
        root, identity=ident.gt_identity(), frozen_carla_frame_id=ident.frame_id,
        mask=np.full((5, 7), mask_value, dtype=np.uint8),
    )
    paths = {"objects.json": objects, "semantic.npy": semantic_npy,
             "semantic.json": semantic_json}
    return T.bundle_from_phase6_paths(ident, paths), paths


class IdentityTests(unittest.TestCase):
    def test_exact_gt_subset_and_round_trip(self) -> None:
        value = identity()
        self.assertEqual(T.GtTransportIdentityV1.from_mapping(value.as_dict()), value)
        self.assertEqual(value.gt_identity(), {
            "run_id": "run4-live", "cell_id": "a71__favorable_stable",
            "stream_id": "ue288_a71__favorable_stable", "frame_id": 1312,
            "action_id": 71, "profile_id": "split_ae32_uint4_q9800",
            "capture_timestamp_ns": 123_456_789_000,
        })

    def test_unsafe_and_incomplete_identity_is_refused(self) -> None:
        with self.assertRaises(T.IdentityError):
            identity(run_id="../escape")
        with self.assertRaises(T.IdentityError):
            identity(session_uuid="not-a-uuid")
        with self.assertRaises(T.IdentityError):
            identity(controller_lineage_sha256="short")
        with self.assertRaises(T.IdentityError):
            identity(action_id=None)
        raw = identity().as_dict()
        raw["filename"] = "../../payload"
        with self.assertRaises(T.IdentityError):
            T.GtTransportIdentityV1.from_mapping(raw)

    def test_phase6_constructor_checks_all_overlapping_fields(self) -> None:
        expected = identity()
        envelope = SimpleNamespace(
            session_uuid=expected.session_uuid,
            controller_lineage_sha256=expected.controller_lineage_sha256,
            decision_seq=expected.decision_seq, ticket_seq=expected.ticket_seq,
            frame_id=expected.frame_id, tensor_seq=expected.tensor_seq,
            capture_timestamp_ns=expected.capture_timestamp_ns,
            anchor_action_id=expected.action_id,
        )
        context = SimpleNamespace(stream_id=expected.stream_id,
                                  frame_id=expected.frame_id,
                                  capture_timestamp_ns=expected.capture_timestamp_ns)
        actual = T.identity_from_phase6(
            run_id=expected.run_id, cell_id=expected.cell_id, envelope=envelope,
            context=context, gt_identity=expected.gt_identity())
        self.assertEqual(actual, expected)
        envelope.frame_id += 1
        with self.assertRaises(T.IdentityError):
            T.identity_from_phase6(
                run_id=expected.run_id, cell_id=expected.cell_id, envelope=envelope,
                context=context, gt_identity=expected.gt_identity())


class WireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle, self.paths = write_bundle(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_preserves_all_exact_bytes(self) -> None:
        decoded = T.GtBundleV1.from_wire(self.bundle.to_wire())
        self.assertEqual(decoded.identity, self.bundle.identity)
        self.assertEqual(decoded.components, self.bundle.components)
        self.assertEqual(decoded.bundle_sha256, self.bundle.bundle_sha256)

    def test_truncation_tamper_and_trailing_bytes_are_refused(self) -> None:
        packet = self.bundle.to_wire()
        for bad in (packet[:-1], packet + b"x"):
            with self.assertRaises(T.ProtocolError):
                T.GtBundleV1.from_wire(bad)
        altered = bytearray(packet)
        altered[-1] ^= 1
        with self.assertRaises(T.ProtocolError):
            T.GtBundleV1.from_wire(bytes(altered))

    def test_manifest_is_strict_and_never_accepts_a_filename(self) -> None:
        packet = self.bundle.to_wire()
        _magic, size = T.HEADER.unpack(packet[:T.HEADER.size])
        manifest = json.loads(packet[T.HEADER.size:T.HEADER.size + size])
        manifest["filename"] = "../../escape"
        raw = T._canonical(manifest)
        forged = T.HEADER.pack(T.REQUEST_MAGIC, len(raw)) + raw + packet[T.HEADER.size + size:]
        with self.assertRaises(T.ProtocolError):
            T.GtBundleV1.from_wire(forged)

    def test_non_authoritative_phase6_bytes_are_refused(self) -> None:
        components = dict(self.bundle.components)
        objects = json.loads(components["objects.json"])
        components["objects.json"] = json.dumps(objects, indent=2).encode()
        with self.assertRaises(T.ProtocolError):
            T.GtBundleV1(identity=self.bundle.identity,
                         components=tuple((name, components[name])
                                          for name in T.COMPONENT_NAMES))

    def test_path_bundle_requires_exact_stem_and_no_symlink(self) -> None:
        bad = dict(self.paths)
        bad["objects.json"] = self.root / "wrong.objects.json"
        bad["objects.json"].write_bytes(self.paths["objects.json"].read_bytes())
        with self.assertRaises(T.ProtocolError):
            T.bundle_from_phase6_paths(self.bundle.identity, bad)
        link = self.root / self.paths["objects.json"].name
        original = self.paths["objects.json"]
        saved = self.root / "saved"
        original.rename(saved)
        link.symlink_to(saved)
        with self.assertRaises(T.ProtocolError):
            T.bundle_from_phase6_paths(self.bundle.identity, self.paths)


class RegistryAndStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_temp = tempfile.TemporaryDirectory()
        self.edge_temp = tempfile.TemporaryDirectory()
        self.source = Path(self.source_temp.name)
        self.edge = Path(self.edge_temp.name)
        self.bundle, self.paths = write_bundle(self.source)
        self.registry = T.ExpectedTicketRegistryV1(
            run_id=self.bundle.identity.run_id, cell_id=self.bundle.identity.cell_id)
        self.store = T.GtIngressStoreV1(self.edge, self.registry)

    def tearDown(self) -> None:
        self.source_temp.cleanup()
        self.edge_temp.cleanup()

    def test_future_ticket_waits_for_edge_authorization_then_stores(self) -> None:
        def authorize() -> None:
            time.sleep(0.02)
            self.registry.authorize(self.bundle.identity)

        thread = threading.Thread(target=authorize)
        thread.start()
        ack = self.store.accept(self.bundle, expectation_timeout_s=0.2)
        thread.join()
        self.assertEqual(ack.status, "STORED")
        stem = GE._stem(self.bundle.identity.stream_id, self.bundle.identity.frame_id)
        for name, original in self.bundle.components:
            self.assertEqual((self.edge / f"{stem}.{name}").read_bytes(), original)
        restored = GE.read_ground_truth(
            self.edge, expected_identity=self.bundle.identity.gt_identity(), timeout_s=0.01)
        self.assertEqual(restored["frozen_carla_frame_id"], self.bundle.identity.frame_id)
        receipts = list((self.edge / ".gt_transport_receipts").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(T._digest(receipts[0].read_bytes()), ack.receipt_sha256)

    def test_unknown_and_foreign_tickets_are_never_written(self) -> None:
        with self.assertRaises(T.UnauthorizedTicketError):
            self.store.accept(self.bundle, expectation_timeout_s=0)
        foreign = replace(self.bundle.identity, run_id="foreign")
        with tempfile.TemporaryDirectory() as directory:
            foreign_bundle, _ = write_bundle(Path(directory), foreign)
            with self.assertRaises(T.UnauthorizedTicketError):
                self.store.accept(foreign_bundle, expectation_timeout_s=0)
        self.assertEqual(list(self.edge.iterdir()), [])

    def test_identical_replay_is_acknowledged_without_rewrite(self) -> None:
        self.registry.authorize(self.bundle.identity)
        first = self.store.accept(self.bundle, expectation_timeout_s=0)
        stem = GE._stem(self.bundle.identity.stream_id, self.bundle.identity.frame_id)
        target = self.edge / f"{stem}.objects.json"
        stat = target.stat()
        second = self.store.accept(self.bundle, expectation_timeout_s=0)
        self.assertEqual((first.status, second.status), ("STORED", "DUPLICATE_IDENTICAL"))
        self.assertEqual(first.receipt_sha256, second.receipt_sha256)
        self.assertEqual(target.stat().st_mtime_ns, stat.st_mtime_ns)

    def test_same_logical_ticket_with_different_fields_is_conflict(self) -> None:
        self.registry.authorize(self.bundle.identity)
        conflict = replace(self.bundle.identity, stream_id="other_stream")
        with self.assertRaises(T.IdentityConflictError):
            self.registry.authorize(conflict)

    def test_same_ticket_with_different_valid_bytes_is_conflict(self) -> None:
        self.registry.authorize(self.bundle.identity)
        self.store.accept(self.bundle, expectation_timeout_s=0)
        with tempfile.TemporaryDirectory() as directory:
            time.sleep(0.001)
            changed, _ = write_bundle(Path(directory), self.bundle.identity, mask_value=4)
            with self.assertRaises(T.IdentityConflictError):
                self.store.accept(changed, expectation_timeout_s=0)

    def test_existing_conflicting_component_and_symlink_root_are_refused(self) -> None:
        self.registry.authorize(self.bundle.identity)
        stem = GE._stem(self.bundle.identity.stream_id, self.bundle.identity.frame_id)
        (self.edge / f"{stem}.objects.json").write_bytes(b"conflict")
        with self.assertRaises(T.IdentityConflictError):
            self.store.accept(self.bundle, expectation_timeout_s=0)
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real"
            real.mkdir()
            link = Path(directory) / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(T.StorageError):
                T.GtIngressStoreV1(link, self.registry)


class PersistentSocketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_temp = tempfile.TemporaryDirectory()
        self.edge_temp = tempfile.TemporaryDirectory()
        self.bundle, _ = write_bundle(Path(self.source_temp.name))
        self.registry = T.ExpectedTicketRegistryV1(
            run_id=self.bundle.identity.run_id, cell_id=self.bundle.identity.cell_id)
        self.store = T.GtIngressStoreV1(Path(self.edge_temp.name), self.registry)

    def tearDown(self) -> None:
        self.source_temp.cleanup()
        self.edge_temp.cleanup()

    def _exchange(self, authorize: bool) -> tuple[T.GtAckV1, T.GtAckV1]:
        left, right = socket.socketpair()
        try:
            if authorize:
                self.registry.authorize(self.bundle.identity)
            result = []

            def server() -> None:
                result.append(T.serve_one(right, self.store, socket_timeout_s=0.5,
                                          expectation_timeout_s=0.02))

            thread = threading.Thread(target=server)
            thread.start()
            client = T.PersistentGtSenderV1(left, timeout_s=0.5)
            if authorize:
                client_ack = client.send(self.bundle)
            else:
                with self.assertRaises(T.RemoteTicketRefusalError) as raised:
                    client.send(self.bundle)
                self.assertEqual(raised.exception.error_code,
                                 "UNKNOWN_OR_FUTURE_TICKET")
                client_ack = raised.exception.ack
            thread.join(1)
            self.assertFalse(thread.is_alive())
            return client_ack, result[0]
        finally:
            left.close()
            right.close()

    def test_persistent_socket_stored_ack_binds_identity_and_hash(self) -> None:
        client, server = self._exchange(True)
        self.assertEqual(client, server)
        self.assertEqual(client.status, "STORED")

    def test_unknown_future_ticket_gets_bounded_rejection(self) -> None:
        client, server = self._exchange(False)
        self.assertEqual(client, server)
        self.assertEqual(server.error_code, "UNKNOWN_OR_FUTURE_TICKET")

    def test_unknown_is_distinct_from_foreign_scope(self) -> None:
        with self.assertRaises(T.UnknownOrFutureTicketError):
            self.registry.await_authorized(self.bundle.identity, 0)
        foreign = replace(self.bundle.identity, cell_id="foreign-cell")
        with self.assertRaises(T.UnauthorizedTicketError) as raised:
            self.registry.await_authorized(foreign, 0)
        self.assertNotIsInstance(raised.exception, T.UnknownOrFutureTicketError)

    def test_ack_identity_or_bundle_substitution_is_refused(self) -> None:
        for field in ("identity_sha256", "bundle_sha256"):
            left, right = socket.socketpair()
            try:
                forged = T.GtAckV1(
                    "STORED", "b" * 64 if field == "identity_sha256"
                    else self.bundle.identity.exact_digest(),
                    "b" * 64 if field == "bundle_sha256" else self.bundle.bundle_sha256,
                    "c" * 64)

                def peer() -> None:
                    T.recv_bundle(right)
                    right.sendall(forged.to_wire())

                thread = threading.Thread(target=peer)
                thread.start()
                with self.assertRaises(T.ProtocolError):
                    T.PersistentGtSenderV1(left, timeout_s=0.5).send(self.bundle)
                thread.join(1)
            finally:
                left.close()
                right.close()


class PurityTests(unittest.TestCase):
    def test_no_pickle_and_import_has_no_runtime_entrypoint(self) -> None:
        source = Path(T.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import pickle", source)
        self.assertNotIn("socket.socket(", source)
        self.assertNotIn("if __name__", source)


if __name__ == "__main__":
    unittest.main()
