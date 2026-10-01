"""Focused CPU-only tests for the additive B edge process."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_process_v1 as E
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_ack_v1 as A


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def request(*, ack_host: str = E.LOCAL_HOST,
            ack_port: int = 51014,
            authorities: list[str] | None = None) -> tuple[str, dict]:
    raw = {
        "schema": E.REQUEST_SCHEMA,
        "role": E.ROLE,
        "run_id": "run4b_live_001",
        "variant": L.ActorVariant.RUN4B.value,
        "config_binding_sha256": sha("config"),
        "actor_boundary_sha256": sha("actor"),
        "feature_schema_sha256": sha("features"),
        "transmitted_budget": 300,
        "deadline_ns": E.DEADLINE_NS,
        "ack_semantics": E.ACK_SEMANTICS,
        "postrun_semantics": E.POSTRUN_SEMANTICS,
        "clock_domain": E.CLOCK_DOMAIN,
        "split_host": {
            "carla_host": E.LOCAL_HOST,
            "ue_host": E.LOCAL_HOST,
            "cn_host": E.REMOTE_HOST,
            "edge_host": E.REMOTE_HOST,
            "ext_dn_host": E.REMOTE_HOST,
            "ack_receiver_host": ack_host,
            "ack_receiver_port": ack_port,
        },
        "output_root": None,
        "evidence_root": None,
        "actor_manifest_path": None,
        "remote_attempt_root": "/tmp/run4b5b_edge_attempt",
        "required_authority_modules": (
            list(E.REQUIRED_AUTHORITIES) if authorities is None else authorities),
        "old_live_quality_runtime_permitted": False,
    }
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii"), raw


def succeeded(code: str) -> L.BranchCallbackResultV1:
    return L.BranchCallbackResultV1(
        status=B.BranchStatus.SUCCEEDED,
        detail_code=code,
        evidence_sha256=sha(code),
    )


class RequestPreflightTest(unittest.TestCase):
    def test_exact_request_and_proven_authorities_pass_without_importing_edge(self) -> None:
        encoded, raw = request()
        seen: list[str] = []

        def find(module: str):
            seen.append(module)
            return object()

        result = E.preflight_request(
            encoded, find_module=find,
            edge_symbols={
                E.PROVEN_PROCESSOR: object(), E.PROVEN_COMPUTE: object(),
                E.PROVEN_MAP_PUBLISHER: object(),
            })
        self.assertEqual(result["run_id"], raw["run_id"])
        self.assertEqual(result["ack_receiver_port"], 51014)
        self.assertTrue(result["offline_postcompute_ready"])
        self.assertFalse(result["live_start_ready"])
        self.assertEqual(seen, [*E.REQUIRED_AUTHORITIES, E.PROVEN_EDGE_MODULE])

    def test_missing_authority_or_proven_symbol_fails_closed(self) -> None:
        encoded, _ = request()
        with self.assertRaisesRegex(E.MissingRuntimeAuthority, "missing"):
            E.preflight_request(
                encoded,
                find_module=lambda module: (
                    None if module == E.REQUIRED_AUTHORITIES[0] else object()),
                edge_symbols={})
        with self.assertRaisesRegex(E.MissingRuntimeAuthority,
                                    E.PROVEN_MAP_PUBLISHER):
            E.preflight_request(
                encoded, find_module=lambda _module: object(),
                edge_symbols={E.PROVEN_PROCESSOR: object(),
                              E.PROVEN_COMPUTE: object()})

    def test_ack_endpoint_and_authority_list_are_exact(self) -> None:
        encoded, _ = request(ack_host=E.REMOTE_HOST)
        with self.assertRaises(E.MissingAckEndpoint):
            E.preflight_request(encoded, find_module=lambda _module: object())
        encoded, _ = request(ack_port=0)
        with self.assertRaises(E.MissingAckEndpoint):
            E.preflight_request(encoded, find_module=lambda _module: object())
        encoded, _ = request(authorities=[])
        with self.assertRaisesRegex(E.RequestError, "authority"):
            E.preflight_request(encoded, find_module=lambda _module: object())

    def test_noncanonical_or_foreign_request_is_refused(self) -> None:
        encoded, raw = request()
        raw["foreign"] = True
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        foreign = base64.urlsafe_b64encode(payload).decode()
        with self.assertRaisesRegex(E.RequestError, "fields"):
            E.preflight_request(foreign, find_module=lambda _module: object())
        decoded = base64.urlsafe_b64decode(encoded).decode("ascii")
        noncanonical = base64.urlsafe_b64encode(
            (decoded[:-1] + " \n}").encode("ascii")).decode("ascii")
        with self.assertRaises(E.RequestError):
            E.preflight_request(noncanonical,
                                find_module=lambda _module: object())


class PostComputeOrderingTest(unittest.TestCase):
    def _harness(self, root: Path):
        events: list[str] = []
        packets: list[bytes] = []
        registry = E.MapDocumentRegistryV1(
            lambda document: events.append("MAP") or {
                "frame_id": document["frame_id"]})
        store = B.PredictionEvidenceStoreV1.create(root / "prediction")
        retain = L.prediction_store_callback(store, clock=lambda: 99)

        def ack(packet: bytes):
            events.append("ACK")
            packets.append(bytes(packet))

        def prediction(work):
            events.append("PREDICTION")
            return retain(work)

        dispatcher = L.TailOutputDispatchV1(
            ack_sender=ack, queue_depth=4,
            map_callback=registry.callback,
            prediction_callback=prediction)
        dispatcher.start()
        seam = E.EdgePostComputeSeamV1(
            dispatcher=dispatcher, register_map_document=registry.register)
        return seam, dispatcher, events, packets, store

    def _output(self, identity: A.FrameActionIdentityV1,
                *, reward_requested: bool = True) -> E.UsableTailOutputV1:
        return E.UsableTailOutputV1(
            identity=identity, reward_requested=reward_requested,
            object_records=({"class_name": "vehicle", "world_x": 1.0,
                             "world_y": 2.0},),
            semantic_mask=np.asarray([[0, 2], [0, 0]], dtype=np.uint8),
            map_document={"frame_id": identity.frame_id},
            tail_ready_monotonic_raw_ns=44)

    def test_ack_is_first_and_prediction_is_create_only_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seam, dispatcher, events, packets, store = self._harness(Path(tmp))
            identity = E._fake_identity()
            receipt = seam.emit(self._output(identity))
            dispatcher.stop()
            self.assertIsInstance(receipt, L.DispatchReceiptV1)
            self.assertEqual(events[0], "ACK")
            self.assertCountEqual(events[1:], ["MAP", "PREDICTION"])
            ack = A.decode_ack(packets[0])
            self.assertEqual(ack.identity, identity)
            records = list(store.records.glob("*.json"))
            self.assertEqual(len(records), 1)
            record = B.PredictionEvidenceRecordV1.from_mapping(
                json.loads(records[0].read_text()))
            self.assertEqual(record.identity, identity)
            self.assertEqual(record.prediction_sha256, ack.tail_output_sha256)

    def test_exact_duplicate_and_logical_identity_conflict_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seam, dispatcher, _events, _packets, _store = self._harness(Path(tmp))
            identity = E._fake_identity()
            seam.emit(self._output(identity))
            dispatcher.drain()
            with self.assertRaisesRegex(B.BranchEvidenceError, "already"):
                seam.emit(self._output(identity))
            conflict = dataclasses.replace(identity, q_e4=3001)
            # A fresh registry avoids the map-register duplicate being the
            # first failure; the dispatcher must reject the logical conflict.
            with self.assertRaises(B.PublicationConflict):
                seam.emit(self._output(conflict))
            dispatcher.stop()

    def test_hold_has_map_only_no_ack_and_no_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seam, dispatcher, events, packets, store = self._harness(Path(tmp))
            accepted = seam.emit(self._output(
                E._fake_identity(), reward_requested=False))
            dispatcher.stop()
            self.assertTrue(accepted)
            self.assertEqual(events, ["MAP"])
            self.assertEqual(packets, [])
            self.assertEqual(list(store.records.glob("*.json")), [])


class OfflineAndCliTest(unittest.TestCase):
    def test_offline_fake_proves_order_without_live_hook(self) -> None:
        encoded, _ = request()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fake"
            result = E.offline_fake(encoded, root=root)
            self.assertEqual(result["events"][0], "ACK")
            self.assertFalse(result["contains_gt_qperc_or_reward"])
            self.assertEqual(result["prediction_record_count"], 1)
            self.assertTrue(root.is_dir())
            with self.assertRaisesRegex(E.BEdgeProcessError, "create-only"):
                E.offline_fake(encoded, root=root)

    def test_preflight_cli_reports_live_seam_not_ready(self) -> None:
        encoded, _ = request()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = E.main(["preflight", "--request-b64", encoded])
        self.assertEqual(rc, 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["live_start_ready"])
        self.assertEqual(result["required_live_hook"], E.REQUIRED_LIVE_HOOK)

    def test_live_start_refuses_instead_of_using_legacy_loop(self) -> None:
        encoded, _ = request()
        with self.assertRaisesRegex(E.RuntimeSeamRequired, "post-compute hook"):
            E.main(["start", "--request-b64", encoded,
                    "--execute", E.EXECUTE_TOKEN])


if __name__ == "__main__":
    unittest.main()
