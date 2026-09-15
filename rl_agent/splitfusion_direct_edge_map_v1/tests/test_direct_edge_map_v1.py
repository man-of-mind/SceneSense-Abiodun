"""Offline verification of the direct edge-to-map architecture.

These tests exercise the corrected contract without CARLA, OAI or CUDA: the
protocol, the publisher's address audit, the map's validation/installation
ordering, duplicate and supersession handling, agent credit, the UE ledger's
exactly-one-terminal rule, and timestamp/byte accounting reconciliation.
"""

from __future__ import annotations

import socket
import threading
import time
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from phase2_map_sharing.transport import ChunkReassembler, chunk_payload

from rl_agent.splitfusion_direct_edge_map_v1 import endpoint, protocol
from rl_agent.splitfusion_direct_edge_map_v1.edge_publisher import (
    DirectMapPublisher,
    UEControlSender,
)
from rl_agent.splitfusion_direct_edge_map_v1.map_ingest import (
    DirectMapIngestService,
    INGEST_FIELDS,
)
from rl_agent.splitfusion_direct_edge_map_v1.protocol import (
    DirectMapProtocolError,
    OUTCOME_MAP_REJECTED,
    OUTCOME_RESULT_INSTALLED,
    OUTCOME_STALE_BEFORE_EDGE,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_SUPERSEDED_PENDING,
)
from rl_agent.splitfusion_direct_edge_map_v1.ue_ledger import (
    DirectLedgerError,
    DirectTerminalLedger,
)


RUN_ID = "direct_edge_map_test_run"
CELL_ID = "a15__favorable_stable"
STREAM_ID = "ue288_a15__favorable_stable"


def make_update(
    *,
    frame_id: int = 7,
    capture_offset_s: float = 0.0,
    action_id: int = 15,
    records=None,
    run_id: str = RUN_ID,
    cell_id: str = CELL_ID,
    stream_id: str = STREAM_ID,
    now: float | None = None,
) -> dict:
    observed = time.time() if now is None else float(now)
    capture_ns = int(round((observed + capture_offset_s) * 1e9))
    if records is None:
        records = [
            {
                "id": f"{stream_id}:{frame_id}:0",
                "type": "Vehicle",
                "score": 0.91,
                "location": {"x": 12.5, "y": -3.25, "z": 0.5},
                "dimensions": {"length": 4.6, "width": 2.0, "height": 1.6},
                "model_yaw_deg": 12.0,
            }
        ]
    return protocol.build_object_map_update(
        run_id=run_id,
        cell_id=cell_id,
        stream_id=stream_id,
        frame_id=frame_id,
        sequence_id=frame_id,
        action_id=action_id,
        profile_id="split_noae_uint8_q0000",
        decoder_identity="noae_uint8",
        capture_timestamp_ns=capture_ns,
        carla_timestamp=1234.5,
        records=records,
        service_deadline_at=observed + capture_offset_s + 0.1,
        ack_timeout_at=observed + capture_offset_s + 0.5,
        edge_timing={
            "reassembly_complete_wall_s": observed,
            "admission_wall_s": observed,
            "compute_start_wall_s": observed,
            "compute_finish_wall_s": observed,
            "tail_complete_wall_s": observed,
            "publish_start_wall_s": observed,
        },
        segmentation={"available": True, "installation_status": "INSTALLED"},
    )


class RecordingInstaller:
    """Install callback that records ordering against feedback emission."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.installed: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, document, ingest_at):
        with self.lock:
            self.events.append(("INSTALL_BEGIN", time.time()))
            self.installed.append(dict(document))
            install_timestamp = time.time()
            self.events.append(("INSTALL_COMMITTED", install_timestamp))
        return {"install_timestamp": install_timestamp, "object_count": len(document["records"])}


def make_service(tmp: Path, installer, **kwargs) -> DirectMapIngestService:
    defaults = dict(
        bind_host="127.0.0.1",
        bind_port=0,
        feedback_host="127.0.0.1",
        feedback_port=9,
        install=installer,
        ingest_csv=tmp / "direct_map_ingest.csv",
        expected_run_id=RUN_ID,
        expected_cell_id=CELL_ID,
        allowed_action_ids=(15, 30, 50, 71),
        processing_horizon_s=0.5,
    )
    defaults.update(kwargs)
    return DirectMapIngestService(**defaults)


class ProtocolTests(unittest.TestCase):
    def test_object_map_update_round_trips(self) -> None:
        update = make_update()
        protocol.validate_object_map_update(update)
        payload = protocol.encode(update)
        self.assertEqual(protocol.decode(payload)["frame_id"], 7)

    def test_non_finite_records_are_refused(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            make_update(records=[{"score": float("nan"), "location": {"x": 1.0}}])
        with self.assertRaises(DirectMapProtocolError):
            make_update(records=[{"location": {"x": float("inf")}}])

    def test_update_identity_fields_are_mandatory(self) -> None:
        for field in protocol.REQUIRED_UPDATE_IDENTITY:
            update = make_update()
            update.pop(field)
            with self.assertRaises(DirectMapProtocolError, msg=field):
                protocol.validate_object_map_update(update)

    def test_protocol_version_drift_is_refused(self) -> None:
        update = make_update()
        update["protocol_version"] = 99
        with self.assertRaises(DirectMapProtocolError):
            protocol.validate_object_map_update(update)

    def test_record_count_must_match_records(self) -> None:
        update = make_update()
        update["record_count"] = 5
        with self.assertRaises(DirectMapProtocolError):
            protocol.validate_object_map_update(update)


class UEMessageContentTests(unittest.TestCase):
    """The UE must receive an ACK/control message, never a map update."""

    def test_map_feedback_carries_no_object_records(self) -> None:
        update = make_update()
        feedback = protocol.build_map_feedback(
            update=update,
            outcome=OUTCOME_RESULT_INSTALLED,
            terminal=True,
            install_timestamp=time.time(),
            map_ingest_at=time.time(),
            feedback_emit_at=time.time(),
            map_age_at_install_ms=42.0,
            direct_update_bytes=900,
            direct_update_datagrams=1,
        )
        for key in protocol.FORBIDDEN_UE_KEYS:
            self.assertNotIn(key, feedback)
        self.assertNotIn("records", protocol.encode(feedback).decode("utf-8"))
        protocol.assert_no_object_records(feedback)

    def test_edge_terminal_control_carries_no_object_records(self) -> None:
        message = protocol.build_edge_terminal_control(
            run_id=RUN_ID,
            cell_id=CELL_ID,
            stream_id=STREAM_ID,
            frame_id=11,
            action_id=15,
            profile_id="split_noae_uint8_q0000",
            capture_timestamp_ns=int(time.time() * 1e9),
            service_deadline_at=time.time() + 0.1,
            ack_timeout_at=time.time() + 0.5,
            outcome=OUTCOME_SUPERSEDED_PENDING,
            stage="EDGE_PENDING_REPLACED",
            age_ms=120.0,
            emit_at=time.time(),
            superseded_by_frame_id=12,
        )
        protocol.assert_no_object_records(message)
        self.assertEqual(message["superseded_by_frame_id"], 12)

    def test_dense_mask_and_records_are_rejected_on_the_ue_path(self) -> None:
        for forbidden in ("records", "objects", "semantic_labels_b64", "dense_mask"):
            with self.assertRaises(DirectMapProtocolError, msg=forbidden):
                protocol.assert_no_object_records({"schema": "x", forbidden: [1]})
        with self.assertRaises(DirectMapProtocolError):
            protocol.assert_no_object_records(
                {"schema": "x", "nested": {"deep": {"semantic_labels": "AAA"}}}
            )

    def test_object_map_update_itself_cannot_be_sent_to_the_ue(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            protocol.assert_no_object_records(make_update())

    def test_ue_control_sender_refuses_a_document_with_records(self) -> None:
        sender = UEControlSender(ue_host="127.0.0.1", ue_port=9)
        try:
            with self.assertRaises(DirectMapProtocolError):
                sender.send(make_update())
        finally:
            sender.close()


class AddressAuditTests(unittest.TestCase):
    """Object updates must not be able to target the UE."""

    def test_publisher_refuses_the_ue_tunnel_address(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            DirectMapPublisher(map_host="10.0.0.2", map_port=39320)

    def test_publisher_refuses_the_ue_result_port(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            DirectMapPublisher(map_host="192.168.70.129", map_port=51004)

    def test_publisher_refuses_loopback_and_unspecified(self) -> None:
        for host in ("127.0.0.1", "0.0.0.0"):
            with self.assertRaises(DirectMapProtocolError, msg=host):
                DirectMapPublisher(map_host=host, map_port=39320)

    def test_map_ingest_refuses_binding_the_ue_address(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            DirectMapIngestService(
                bind_host="10.0.0.2",
                bind_port=39320,
                feedback_host="127.0.0.1",
                feedback_port=9,
                install=lambda document, at: {"install_timestamp": time.time()},
            )

    def test_control_sender_refuses_the_superseded_result_port(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            UEControlSender(ue_host="10.0.0.2", ue_port=51004)

    def test_static_audit_reports_every_ue_address_mention(self) -> None:
        package = Path(endpoint.__file__).resolve().parent
        sources = sorted(str(path) for path in package.glob("*.py"))
        report = endpoint.static_address_audit(sources)
        self.assertGreater(report["sources_scanned"], 0)
        publisher_source = package / "edge_publisher.py"
        offenders = [
            item
            for item in report["mentions"]
            if item["path"] == str(publisher_source)
            and "must not" not in item["text"]
            and "not in" not in item["text"]
            and not item["text"].lstrip().startswith("#")
        ]
        self.assertEqual(
            offenders,
            [],
            f"edge publisher references a UE address outside a refusal: {offenders}",
        )


class MapInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.installer = RecordingInstaller()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_install_precedes_ack_emission(self) -> None:
        service = make_service(self.tmp, self.installer)
        try:
            result = service.ingest(make_update(), ingest_at=time.time(), emit=False)
        finally:
            service.close()
        self.assertEqual(result["outcome"], OUTCOME_RESULT_INSTALLED)
        committed = [at for name, at in self.installer.events if name == "INSTALL_COMMITTED"]
        self.assertEqual(len(committed), 1)
        # Feedback is constructed from the completed installation's return
        # value, so its emission instant must not precede the commit.
        self.assertGreaterEqual(result["feedback"]["feedback_emit_at"], committed[0])
        self.assertEqual(
            result["feedback"]["install_timestamp"], result["install_timestamp"]
        )

    def test_ack_installed_requires_a_real_installation(self) -> None:
        update = make_update()
        with self.assertRaises(DirectMapProtocolError):
            protocol.build_map_feedback(
                update=update,
                outcome=OUTCOME_RESULT_INSTALLED,
                terminal=True,
                install_timestamp=None,
                map_ingest_at=time.time(),
                feedback_emit_at=time.time(),
                map_age_at_install_ms=None,
            )

    def test_identity_mismatches_are_rejected(self) -> None:
        cases = {
            "run": make_update(run_id="other_run"),
            "cell": make_update(cell_id="a71__fade_recovery"),
            "action": make_update(action_id=20),
        }
        for label, update in cases.items():
            service = make_service(
                self.tmp, self.installer, ingest_csv=self.tmp / f"{label}.csv"
            )
            try:
                result = service.ingest(update, ingest_at=time.time(), emit=False)
            finally:
                service.close()
            self.assertEqual(result["outcome"], OUTCOME_MAP_REJECTED, label)
            self.assertEqual(
                result["feedback"]["agent_credit"], protocol.CREDIT_REJECTED, label
            )
        self.assertEqual(self.installer.installed, [])

    def test_duplicate_update_never_installs_twice(self) -> None:
        service = make_service(self.tmp, self.installer)
        try:
            update = make_update()
            first = service.ingest(update, ingest_at=time.time(), emit=False)
            second = service.ingest(update, ingest_at=time.time(), emit=False)
        finally:
            service.close()
        self.assertEqual(first["outcome"], OUTCOME_RESULT_INSTALLED)
        self.assertEqual(second["outcome"], OUTCOME_MAP_REJECTED)
        self.assertEqual(
            second["feedback"]["rejection_reason"], "DUPLICATE_UPDATE_IGNORED"
        )
        # Exactly one terminal for the obligation.
        self.assertTrue(first["feedback"]["terminal"])
        self.assertFalse(second["feedback"]["terminal"])
        self.assertEqual(len(self.installer.installed), 1)
        self.assertEqual(service.counters["direct_updates_duplicate"], 1)

    def test_superseded_frame_is_credited_as_replaced_not_lost(self) -> None:
        service = make_service(self.tmp, self.installer)
        try:
            now = time.time()
            fresh = make_update(frame_id=20, capture_offset_s=0.0, now=now)
            older = make_update(frame_id=19, capture_offset_s=-0.05, now=now)
            installed = service.ingest(fresh, ingest_at=now, emit=False)
            superseded = service.ingest(older, ingest_at=now, emit=False)
        finally:
            service.close()
        self.assertEqual(installed["outcome"], OUTCOME_RESULT_INSTALLED)
        self.assertEqual(superseded["outcome"], OUTCOME_SUPERSEDED_PENDING)
        feedback = superseded["feedback"]
        self.assertEqual(feedback["agent_credit"], protocol.CREDIT_SUPERSEDED_BY_FRESHER)
        self.assertTrue(feedback["terminal"])
        self.assertFalse(feedback["accepted"])
        # The selected action identity survives the supersession.
        self.assertEqual(feedback["action_id"], 15)
        self.assertEqual(feedback["profile_id"], "split_noae_uint8_q0000")
        self.assertEqual(feedback["superseded_by_frame_id"], 20)
        # A superseded frame is never credited as a network failure.
        self.assertNotEqual(feedback["agent_credit"], protocol.CREDIT_NETWORK_INCOMPLETE)
        self.assertEqual(len(self.installer.installed), 1)

    def test_stale_update_is_refused_before_installation(self) -> None:
        service = make_service(self.tmp, self.installer)
        try:
            now = time.time()
            stale = make_update(frame_id=31, capture_offset_s=-0.9, now=now)
            result = service.ingest(stale, ingest_at=now, emit=False)
        finally:
            service.close()
        self.assertEqual(result["outcome"], OUTCOME_STALE_BEFORE_MAP)
        self.assertEqual(
            result["feedback"]["agent_credit"], protocol.CREDIT_STALE_AT_MAP
        )
        self.assertEqual(self.installer.installed, [])

    def test_ingest_csv_reconciles_timestamps_and_bytes(self) -> None:
        import csv as csv_module

        service = make_service(self.tmp, self.installer)
        try:
            update = make_update()
            ingest_at = time.time()
            result = service.ingest(
                update,
                ingest_at=ingest_at,
                first_datagram_at=ingest_at - 0.001,
                update_bytes=1234,
                update_datagrams=2,
                emit=False,
            )
        finally:
            service.close()
        rows = list(csv_module.DictReader((self.tmp / "direct_map_ingest.csv").open()))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(sorted(row), sorted(INGEST_FIELDS))
        self.assertEqual(int(row["direct_update_bytes"]), 1234)
        self.assertEqual(int(row["direct_update_datagrams"]), 2)
        self.assertEqual(int(row["feedback_bytes"]), result["feedback_bytes"])
        self.assertEqual(row["outcome"], OUTCOME_RESULT_INSTALLED)
        # Monotonic ordering of the recorded stage timestamps.
        self.assertLessEqual(float(row["first_datagram_at"]), float(row["map_ingest_at"]))
        self.assertLessEqual(float(row["map_ingest_at"]), float(row["map_install_at"]))
        self.assertLessEqual(float(row["map_install_at"]), float(row["feedback_emit_at"]))
        # Map age at install is measured from capture, not from ingest.
        expected_age_ms = (
            float(row["map_install_at"]) - float(row["capture_timestamp"])
        ) * 1000.0
        self.assertAlmostEqual(
            float(row["map_age_at_install_ms"]), expected_age_ms, places=6
        )
        publish_start = float(row["edge_publish_start_wall_s"])
        self.assertAlmostEqual(
            float(row["install_latency_from_publish_ms"]),
            (float(row["map_install_at"]) - publish_start) * 1000.0,
            places=6,
        )


class PublicationTransportTests(unittest.TestCase):
    """End-to-end direct publication over a real socket pair."""

    def test_direct_publication_reassembles_and_installs(self) -> None:
        installer = RecordingInstaller()
        with TemporaryDirectory() as raw:
            tmp = Path(raw)
            # Bind the map on an ephemeral loopback port for the test only; the
            # production endpoint audit is covered by AddressAuditTests.
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receiver.bind(("127.0.0.1", 0))
            map_port = receiver.getsockname()[1]
            receiver.close()

            feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            feedback_socket.bind(("127.0.0.1", 0))
            feedback_socket.settimeout(5.0)
            feedback_port = feedback_socket.getsockname()[1]

            service = make_service(
                tmp,
                installer,
                bind_port=map_port,
                feedback_port=feedback_port,
            )
            service.start()
            try:
                publisher = DirectMapPublisher.__new__(DirectMapPublisher)
                # Bypass only the production address audit so the test can use
                # loopback; every other publication behaviour is exercised.
                publisher.remote = ("127.0.0.1", map_port)
                publisher.chunk_bytes = 12500
                from collections import Counter

                publisher.counters = Counter()
                publisher._lock = threading.Lock()
                publisher.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                update = make_update(frame_id=44)
                accounting = publisher.publish(update)
                datagram, source = feedback_socket.recvfrom(65535)
                message = protocol.decode(datagram)
            finally:
                service.close()
                feedback_socket.close()

            protocol.validate_map_feedback(message)
            protocol.assert_no_object_records(message)
            self.assertEqual(message["outcome"], OUTCOME_RESULT_INSTALLED)
            self.assertEqual(message["frame_id"], 44)
            self.assertEqual(len(installer.installed), 1)
            self.assertEqual(
                message["direct_update_datagrams"], accounting["direct_map_datagrams"]
            )
            self.assertEqual(
                message["direct_update_bytes"], accounting["direct_map_payload_bytes"]
            )
            self.assertEqual(source[0], "127.0.0.1")

    def test_publisher_chunks_with_the_production_header(self) -> None:
        big = [
            {
                "id": f"obj{index}",
                "type": "Vehicle",
                "score": 0.5,
                "location": {"x": float(index), "y": 0.0, "z": 0.0},
            }
            for index in range(900)
        ]
        update = make_update(frame_id=5, records=big)
        payload = zlib.compress(protocol.encode(update), level=1)
        chunks = chunk_payload(payload, message_id=5, chunk_bytes=2048)
        self.assertGreater(len(chunks), 1)
        reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
        complete = None
        for chunk in chunks:
            complete = reassembler.ingest("edge", chunk, received_at_s=time.monotonic())
        self.assertIsNotNone(complete)
        restored = protocol.decode(zlib.decompress(complete.payload))
        self.assertEqual(restored["record_count"], 900)


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.ledger = DirectTerminalLedger(
            output_csv=self.tmp / "map_feedback.csv",
            experiment_id=RUN_ID,
            cell_id=CELL_ID,
        )

    def tearDown(self) -> None:
        self.ledger.close()
        self._tmp.cleanup()

    def register(self, update) -> None:
        self.ledger.register_capture(
            stream_id=str(update["stream_id"]),
            capture_id=f"{update['stream_id']}:{update['frame_id']}",
            frame_id=int(update["frame_id"]),
            capture_at=float(update["capture_timestamp"]),
            action_id=str(update["action_id"]),
            profile_id=str(update["profile_id"]),
            service_deadline_at=float(update["service_deadline_at"]),
            ack_timeout_at=float(update["ack_timeout_at"]),
        )

    def test_exactly_one_terminal_per_obligation(self) -> None:
        update = make_update(frame_id=61)
        self.register(update)
        install_at = time.time()
        feedback = protocol.build_map_feedback(
            update=update,
            outcome=OUTCOME_RESULT_INSTALLED,
            terminal=True,
            install_timestamp=install_at,
            map_ingest_at=install_at,
            feedback_emit_at=install_at,
            map_age_at_install_ms=30.0,
        )
        first = self.ledger.record_message(feedback, time.time())
        second = self.ledger.record_message(feedback, time.time())
        self.assertTrue(first["terminal"])
        self.assertFalse(second["terminal"])
        summary = self.ledger.summary()
        self.assertEqual(summary["obligations_closed"], 1)
        self.assertEqual(summary["obligations_open"], 0)
        self.assertEqual(summary["late_nonterminal_messages"], 1)

    def test_unknown_capture_is_refused(self) -> None:
        update = make_update(frame_id=62)
        feedback = protocol.build_map_feedback(
            update=update,
            outcome=OUTCOME_STALE_BEFORE_MAP,
            terminal=True,
            install_timestamp=None,
            map_ingest_at=time.time(),
            feedback_emit_at=time.time(),
            map_age_at_install_ms=None,
        )
        with self.assertRaises(DirectLedgerError):
            self.ledger.record_message(feedback, time.time())

    def test_identity_mismatch_is_refused(self) -> None:
        update = make_update(frame_id=63)
        self.register(update)
        drifted = dict(
            protocol.build_map_feedback(
                update=update,
                outcome=OUTCOME_STALE_BEFORE_MAP,
                terminal=True,
                install_timestamp=None,
                map_ingest_at=time.time(),
                feedback_emit_at=time.time(),
                map_age_at_install_ms=None,
            )
        )
        drifted["action_id"] = 71
        with self.assertRaises(DirectLedgerError):
            self.ledger.record_message(drifted, time.time())

    def test_duplicate_registration_is_refused(self) -> None:
        update = make_update(frame_id=64)
        self.register(update)
        with self.assertRaises(DirectLedgerError):
            self.register(update)

    def test_feedback_timeout_closes_the_obligation(self) -> None:
        update = make_update(frame_id=65)
        self.register(update)
        closed = self.ledger.record_expired(now=float(update["ack_timeout_at"]) + 1.0)
        self.assertEqual(closed, 1)
        summary = self.ledger.summary()
        self.assertEqual(
            summary["terminal_outcomes"], {protocol.OUTCOME_FEEDBACK_TIMEOUT: 1}
        )

    def test_edge_terminal_closes_the_obligation_with_credit(self) -> None:
        update = make_update(frame_id=66)
        self.register(update)
        message = protocol.build_edge_terminal_control(
            run_id=RUN_ID,
            cell_id=CELL_ID,
            stream_id=STREAM_ID,
            frame_id=66,
            action_id=int(update["action_id"]),
            profile_id=str(update["profile_id"]),
            capture_timestamp_ns=int(update["capture_timestamp_ns"]),
            service_deadline_at=float(update["service_deadline_at"]),
            ack_timeout_at=float(update["ack_timeout_at"]),
            outcome=OUTCOME_SUPERSEDED_PENDING,
            stage="EDGE_PENDING_REPLACED",
            age_ms=95.0,
            emit_at=time.time(),
            superseded_by_frame_id=67,
        )
        row = self.ledger.record_message(message, time.time())
        self.assertTrue(row["terminal"])
        self.assertEqual(row["agent_credit"], protocol.CREDIT_SUPERSEDED_BY_FRESHER)
        self.assertEqual(row["terminal_source"], "EDGE_INFERENCE_SERVICE")
        self.assertEqual(
            self.ledger.summary()["terminal_outcomes"],
            {OUTCOME_SUPERSEDED_PENDING: 1},
        )

    def test_feedback_arrival_is_separate_from_installation(self) -> None:
        update = make_update(frame_id=68)
        self.register(update)
        install_at = time.time()
        feedback = protocol.build_map_feedback(
            update=update,
            outcome=OUTCOME_RESULT_INSTALLED,
            terminal=True,
            install_timestamp=install_at,
            map_ingest_at=install_at - 0.002,
            feedback_emit_at=install_at + 0.001,
            map_age_at_install_ms=25.0,
        )
        received_at = install_at + 0.030
        row = self.ledger.record_message(feedback, received_at)
        # Physical freshness ends at install; ACK arrival is a separate delay.
        self.assertEqual(float(row["install_timestamp"]), install_at)
        self.assertEqual(float(row["feedback_received_at"]), received_at)
        self.assertAlmostEqual(
            float(row["feedback_observation_delay_ms"]), 29.0, places=3
        )
        self.assertNotEqual(
            float(row["install_timestamp"]), float(row["feedback_received_at"])
        )


class AgentCreditTests(unittest.TestCase):
    def test_every_terminal_outcome_has_a_credit(self) -> None:
        for outcome in protocol.TERMINAL_OUTCOMES:
            if outcome == OUTCOME_RESULT_INSTALLED:
                continue
            credit = protocol.classify_agent_credit(outcome)
            self.assertIn(credit, protocol.AGENT_CREDITS, outcome)

    def test_installed_credit_splits_on_the_service_target(self) -> None:
        deadline = 1000.1
        self.assertEqual(
            protocol.classify_agent_credit(
                OUTCOME_RESULT_INSTALLED,
                install_timestamp=1000.05,
                service_deadline_at=deadline,
            ),
            protocol.CREDIT_INSTALLED_ON_TIME,
        )
        self.assertEqual(
            protocol.classify_agent_credit(
                OUTCOME_RESULT_INSTALLED,
                install_timestamp=1000.4,
                service_deadline_at=deadline,
            ),
            protocol.CREDIT_INSTALLED_LATE,
        )

    def test_supersession_is_never_a_network_failure_credit(self) -> None:
        self.assertEqual(
            protocol.classify_agent_credit(OUTCOME_SUPERSEDED_PENDING),
            protocol.CREDIT_SUPERSEDED_BY_FRESHER,
        )
        self.assertNotEqual(
            protocol.classify_agent_credit(OUTCOME_SUPERSEDED_PENDING),
            protocol.classify_agent_credit(protocol.OUTCOME_TRANSPORT_INCOMPLETE),
        )

    def test_stale_before_edge_is_distinct_from_stale_before_map(self) -> None:
        self.assertNotEqual(
            protocol.classify_agent_credit(OUTCOME_STALE_BEFORE_EDGE),
            protocol.classify_agent_credit(OUTCOME_STALE_BEFORE_MAP),
        )

    def test_unregistered_outcome_is_refused(self) -> None:
        with self.assertRaises(DirectMapProtocolError):
            protocol.classify_agent_credit("SOMETHING_ELSE")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class StageInstrumentationTests(unittest.TestCase):
    """The publication and ingest stage boundaries must be complete and ordered."""

    def test_publisher_reports_every_publication_stage_boundary(self) -> None:
        publisher = DirectMapPublisher.__new__(DirectMapPublisher)
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.bind(("127.0.0.1", 0))
        publisher.remote = sink.getsockname()
        publisher.chunk_bytes = 12500
        from collections import Counter

        publisher.counters = Counter()
        publisher._lock = threading.Lock()
        publisher.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            accounting = publisher.publish(make_update(frame_id=91))
        finally:
            sink.close()
            publisher.close()
        ordered = (
            "publish_start_wall_s",
            "serialization_start_wall_s",
            "serialization_end_wall_s",
            "first_datagram_send_wall_s",
            "last_datagram_send_wall_s",
        )
        for name in ordered:
            self.assertIn(name, accounting)
        values = [float(accounting[name]) for name in ordered]
        self.assertEqual(values, sorted(values), accounting)

    def test_ingest_rows_carry_an_ordered_map_side_decomposition(self) -> None:
        installer = RecordingInstaller()
        with TemporaryDirectory() as raw:
            tmp = Path(raw)
            service = make_service(tmp, installer)
            try:
                service.ingest(make_update(frame_id=101), ingest_at=time.time())
            finally:
                service.close()
            rows = service.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for name in (
            "last_datagram_at",
            "reassembly_complete_at",
            "map_worker_start_at",
            "map_lock_request_at",
            "map_lock_acquired_at",
            "map_lock_released_at",
            "association_start_at",
            "association_end_at",
            "ack_emit_at",
            "ack_sent_at",
        ):
            self.assertIn(name, INGEST_FIELDS, name)
            self.assertIn(name, row, name)
        ordered = (
            "map_worker_start_at",
            "association_start_at",
            "map_install_at",
            "ack_emit_at",
        )
        values = [float(row[name]) for name in ordered]
        self.assertEqual(values, sorted(values), row)

    def test_receive_owner_hands_off_without_installing(self) -> None:
        """A completed message must reach the map through the ingest owner."""

        installer = RecordingInstaller()
        with TemporaryDirectory() as raw:
            tmp = Path(raw)
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receiver.bind(("127.0.0.1", 0))
            map_port = receiver.getsockname()[1]
            receiver.close()
            feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            feedback_socket.bind(("127.0.0.1", 0))
            feedback_socket.settimeout(5.0)
            service = make_service(
                tmp,
                installer,
                bind_port=map_port,
                feedback_port=feedback_socket.getsockname()[1],
            )
            service.start()
            try:
                self.assertIsNot(service.thread, service.ingest_thread)
                sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                payload = zlib.compress(
                    protocol.encode(make_update(frame_id=57)), level=1
                )
                for chunk in chunk_payload(payload, message_id=57, chunk_bytes=12500):
                    sender.sendto(chunk, ("127.0.0.1", map_port))
                sender.close()
                message = protocol.decode(feedback_socket.recvfrom(65535)[0])
            finally:
                service.close()
                feedback_socket.close()
            rows = service.rows()
        self.assertEqual(message["outcome"], OUTCOME_RESULT_INSTALLED)
        self.assertEqual(len(installer.installed), 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # The install ran on the ingest owner, so the arrival stamp precedes
        # the worker start rather than being taken after the install.
        self.assertLessEqual(float(row["first_datagram_at"]), float(row["last_datagram_at"]))
        self.assertLessEqual(
            float(row["last_datagram_at"]), float(row["reassembly_complete_at"])
        )
        self.assertLessEqual(
            float(row["reassembly_complete_at"]), float(row["map_worker_start_at"])
        )
        self.assertGreaterEqual(float(row["ingest_queue_wait_ms"]), 0.0)
        self.assertEqual(
            service.report()["counters"].get("direct_ingest_queue_blocked", 0), 0
        )


class CpuReservationTests(unittest.TestCase):
    def test_cpu_set_parsing(self) -> None:
        from rl_agent.splitfusion_direct_edge_map_v1.cpu_reservation import parse_cpu_set

        self.assertEqual(parse_cpu_set(""), ())
        self.assertEqual(parse_cpu_set("3"), (3,))
        self.assertEqual(parse_cpu_set("2,0,1"), (0, 1, 2))
        self.assertEqual(parse_cpu_set("0-3"), (0, 1, 2, 3))
        self.assertEqual(parse_cpu_set("0-1,4"), (0, 1, 4))
        with self.assertRaises(ValueError):
            parse_cpu_set("5-2")

    def test_reservation_is_advisory_and_reported(self) -> None:
        import os

        from rl_agent.splitfusion_direct_edge_map_v1.cpu_reservation import (
            apply_thread_reservation,
            describe_reservation,
        )

        available = sorted(os.sched_getaffinity(0))
        record = apply_thread_reservation(str(available[0]), label="probe")
        try:
            self.assertTrue(record["applied"])
            self.assertEqual(record["effective_cpus"], [available[0]])
            # An unavailable CPU is reported, never silently dropped, and the
            # cell is not failed over a placement request.
            impossible = apply_thread_reservation("99999", label="impossible")
            self.assertFalse(impossible["applied"])
            self.assertEqual(impossible["unavailable_cpus"], [99999])
            self.assertTrue(impossible["error"])
            summary = describe_reservation([record, impossible])
            self.assertFalse(summary["all_requested_applied"])
            self.assertEqual(len(summary["threads"]), 2)
        finally:
            os.sched_setaffinity(0, set(available))

    def test_empty_edge_reservations_survive_shell_tokenization(self) -> None:
        import json as json_module
        import shlex

        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1

        config_path = (
            Path(__file__).resolve().parents[3]
            / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
        )
        campaign = json_module.loads(config_path.read_text(encoding="utf-8"))
        arguments = adapter_direct_v1._edge_reservation_arguments(campaign)
        self.assertEqual(
            shlex.split(" ".join(arguments)),
            ["--edge-compute-cpus=", "--edge-receive-cpus="],
        )


class MapServerArgumentSplitTests(unittest.TestCase):
    """The wrapper's options must never reach the baseline's own parser.

    A wrapper option missing from the split list is forwarded to the baseline
    parser, which rejects it and takes the map server down at launch. That
    cannot be caught by importing the module, so this drives the real argv the
    adapter builds.
    """

    def _argv(self) -> list[str]:
        import json as json_module

        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1

        config_path = (
            Path(__file__).resolve().parents[3]
            / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
        )
        campaign = json_module.loads(config_path.read_text(encoding="utf-8"))
        runtime = campaign["runtime"]
        reservation = adapter_direct_v1._reservation
        return [
            "--api-host", "127.0.0.1",
            "--api-port", "8008",
            "--default-action-id", "15",
            "--carla-host", "127.0.0.1",
            "--carla-port", "2000",
            "--output-dir", "/tmp/map",
            "--focus-follow-stream-id", "unused",
            "--installed-frame-history-size", "4096",
            "--direct-map-host", "192.168.70.129",
            "--direct-map-port", str(int(runtime["direct_map_ingest_port"])),
            "--ue-feedback-host", str(runtime["ue_bind_host"]),
            "--ue-feedback-port", str(int(runtime["ue_control_port"])),
            "--direct-run-id", str(campaign["campaign_id"]),
            "--direct-cell-id", "a15__favorable_stable",
            "--direct-allowed-action-ids", "15",
            "--direct-processing-horizon-ms", "500.0",
            "--direct-ingest-csv", "/tmp/direct_map_ingest.csv",
            "--direct-ready-file", "/tmp/direct_map_ready.json",
            "--direct-report-file", "/tmp/direct_map_report.json",
            "--direct-receive-cpus", reservation(campaign, "map_receive_cpus"),
            "--direct-ingest-cpus", reservation(campaign, "map_ingest_cpus"),
            "--direct-ingest-queue-capacity",
            str(int(reservation(campaign, "map_ingest_queue_capacity", 64))),
            "--direct-render", str(reservation(campaign, "map_render", "on") or "on"),
        ]

    def test_no_wrapper_option_reaches_the_baseline_parser(self) -> None:
        from rl_agent.splitfusion_direct_edge_map_v1 import (
            spatial_map_direct_server_v1 as server,
        )

        mine, rest = server._split_direct_arguments(self._argv())
        leaked = [token for token in rest if token.startswith("--direct-")]
        self.assertIn("--direct-render", mine)
        leaked += [token for token in rest if token.startswith("--ue-feedback-")]
        self.assertEqual(leaked, [], f"wrapper options reached the baseline parser: {leaked}")
        parsed = server._parse_direct(mine)
        self.assertEqual(parsed.direct_map_port, 39320)
        # Every wrapper option must survive the split with the value the
        # adapter sends, whatever the live config currently sets it to.
        import json as json_module

        config_path = (
            Path(__file__).resolve().parents[3]
            / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
        )
        campaign = json_module.loads(config_path.read_text(encoding="utf-8"))
        block = campaign["direct_edge_map"]["cpu_reservation"]
        self.assertEqual(
            parsed.direct_ingest_queue_capacity,
            int(block["map_ingest_queue_capacity"]),
        )
        self.assertEqual(parsed.direct_ingest_cpus, str(block["map_ingest_cpus"]))
        self.assertEqual(parsed.direct_receive_cpus, str(block["map_receive_cpus"]))
        self.assertEqual(parsed.direct_render, str(block.get("map_render", "on")))

    def test_every_wrapper_option_is_in_the_split_list(self) -> None:
        from rl_agent.splitfusion_direct_edge_map_v1 import (
            spatial_map_direct_server_v1 as server,
        )

        declared = {
            option
            for action in server._direct_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(declared, set(server.DIRECT_ARGUMENTS))


class PublicationLedgerTests(unittest.TestCase):
    """The edge-side send instants must survive the cell's own teardown."""

    def test_rows_are_readable_before_close(self) -> None:
        import csv as csv_module

        from rl_agent.splitfusion_direct_edge_map_v1.live_pilot_runtime_direct_v1 import (
            PUBLICATION_FIELDS,
            _PublicationLedger,
        )

        with TemporaryDirectory() as raw:
            directory = Path(raw)
            ledger = _PublicationLedger(directory)
            try:
                ledger.append({"stream_id": "s", "frame_id": 3, "record_count": 7})
                # The edge's mount is deleted with the cell, so a ledger that
                # only materialises at exit is never collected.
                with ledger.path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv_module.DictReader(handle))
            finally:
                ledger.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stream_id"], "s")
            self.assertEqual(rows[0]["frame_id"], "3")
            self.assertEqual(set(rows[0]), set(PUBLICATION_FIELDS))

    def test_ledger_is_in_the_edge_evidence_copy_out(self) -> None:
        import inspect

        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1

        source = inspect.getsource(
            adapter_direct_v1.stop_tail_preserving_edge_evidence
        )
        self.assertIn("direct_edge_publication.csv", source)
        self.assertIn("direct_edge_counters.json", source)
