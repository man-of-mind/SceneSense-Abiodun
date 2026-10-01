"""CPU-only tests for the GT-free operational ACK contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_ack_v1 as A


MS = 1_000_000


def identity(**changes) -> A.FrameActionIdentityV1:
    row = A.FrameActionIdentityV1(
        run_id="run4b_live_001",
        cell_id="fade_recovery",
        stream_id="ue0_route_b",
        session_uuid=str(uuid.UUID(int=4125)),
        controller_lineage_sha256=hashlib.sha256(b"lineage").hexdigest(),
        decision_seq=7,
        ticket_seq=7,
        frame_id=1312,
        tensor_seq=14,
        capture_timestamp_ns=1_790_000_000_000_000_000,
        mode_id=6,
        q_e4=6784,
        keep_count=6918,
        anchor_action_id=None,
        profile_id=None,
        execution_bundle_sha256=hashlib.sha256(b"bundle").hexdigest(),
    )
    return dataclasses.replace(row, **changes) if changes else row


def ack(row: A.FrameActionIdentityV1 | None = None,
        payload: bytes = b"tail-prediction-v1") -> A.TailOutputAckV1:
    return A.TailOutputAckV1.success(row or identity(), payload)


class IdentityTest(unittest.TestCase):
    def test_exact_roundtrip_and_postrun_join_key(self) -> None:
        row = identity()
        rebuilt = A.FrameActionIdentityV1.from_mapping(row.as_dict())
        self.assertEqual(rebuilt, row)
        key = row.postrun_join_key()
        self.assertEqual(
            (key["run_id"], key["stream_id"], key["frame_id"],
             key["capture_timestamp_ns"], key["session_uuid"],
             key["decision_seq"]),
            (row.run_id, row.stream_id, row.frame_id,
             row.capture_timestamp_ns, row.session_uuid, row.decision_seq),
        )
        self.assertEqual(key["action"]["q_e4"], 6784)
        self.assertEqual(key["action"]["execution_bundle_sha256"],
                         row.execution_bundle_sha256)

    def test_identity_refuses_foreign_fields_and_invalid_action(self) -> None:
        raw = identity().as_dict()
        raw["q_perc"] = 0.9
        with self.assertRaises(A.IdentityError):
            A.FrameActionIdentityV1.from_mapping(raw)
        with self.assertRaises(A.IdentityError):
            identity(q_e4=9801)
        with self.assertRaises(A.IdentityError):
            identity(anchor_action_id=3, profile_id=None)
        with self.assertRaises(A.IdentityError):
            identity(stream_id="../ue0")


class WireTest(unittest.TestCase):
    def test_roundtrip_and_single_bit_tamper(self) -> None:
        original = ack()
        packet = A.encode_ack(original)
        self.assertEqual(A.decode_ack(packet), original)
        corrupt = bytearray(packet)
        corrupt[len(corrupt) // 2] ^= 1
        with self.assertRaisesRegex(A.AckWireError, "digest"):
            A.decode_ack(bytes(corrupt))

    def test_ack_carries_no_quality_reward_or_gt(self) -> None:
        original = ack()
        payload = original.as_dict()

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key.lower()
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        forbidden = {"q_perc", "reward", "quality", "ground_truth", "gt"}
        self.assertTrue(forbidden.isdisjoint(set(keys(payload))))
        wire = A.encode_ack(original).lower()
        for token in (b"q_perc", b"reward", b"quality", b"ground_truth"):
            self.assertNotIn(token, wire)
        self.assertEqual(original.status, "TAIL_OUTPUT_SUCCESS")


class DeadlineTest(unittest.TestCase):
    def test_exactly_170_ms_is_success(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        result = ledger.receive(ack(row), 1_170 * MS)
        self.assertIs(result, A.AckClass.ACCEPTED)
        outcome = ledger.outcome(row)
        self.assertIs(outcome.terminal, A.OperationalTerminal.SUCCESS)
        self.assertEqual(outcome.observed_latency_ns, 170 * MS)
        self.assertEqual(outcome.state_latency_ns, 170 * MS)
        self.assertEqual(outcome.ack_receipt_monotonic_raw_ns, 1_170 * MS)
        self.assertEqual(outcome.tail_output_sha256,
                         ack(row).tail_output_sha256)

    def test_one_nanosecond_late_is_timeout_and_orphan(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        result = ledger.receive(ack(row), 1_170 * MS + 1)
        self.assertIs(result, A.AckClass.LATE_ORPHAN)
        outcome = ledger.outcome(row)
        self.assertIs(outcome.terminal, A.OperationalTerminal.TIMEOUT)
        self.assertIsNone(outcome.observed_latency_ns)
        self.assertEqual(outcome.state_latency_ns, 0)
        self.assertEqual(outcome.resolution_monotonic_raw_ns,
                         1_170 * MS + 1)
        self.assertIsNone(outcome.ack_receipt_monotonic_raw_ns)
        self.assertIsNone(outcome.tail_output_sha256)
        self.assertEqual(len(ledger.late_orphans), 1)

    def test_poll_preserves_inclusive_boundary(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        self.assertEqual(ledger.poll(1_170 * MS), ())
        self.assertIsNone(ledger.outcome(row))
        self.assertEqual(ledger.poll(1_170 * MS + 1), (row,))
        self.assertIs(ledger.outcome(row).terminal, A.OperationalTerminal.TIMEOUT)
        self.assertIs(ledger.receive(ack(row), 1_180 * MS),
                      A.AckClass.LATE_ORPHAN)
        self.assertIs(ledger.outcome(row).terminal, A.OperationalTerminal.TIMEOUT)


class DuplicateAndConflictTest(unittest.TestCase):
    def test_identical_duplicate_is_ignored_but_conflict_fails(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        original = ack(row)
        self.assertIs(ledger.receive(original, 1_100 * MS), A.AckClass.ACCEPTED)
        self.assertIs(ledger.receive(original, 1_101 * MS),
                      A.AckClass.DUPLICATE_IGNORED)
        with self.assertRaises(A.ConflictingAckError):
            ledger.receive(ack(row, b"different-tail"), 1_102 * MS)
        self.assertIs(ledger.outcome(row).terminal, A.OperationalTerminal.SUCCESS)

    def test_same_decision_with_different_frame_or_action_fails_closed(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        for changed in (identity(frame_id=row.frame_id + 1),
                        identity(q_e4=row.q_e4 + 1),
                        identity(mode_id=7)):
            with self.assertRaises(A.AckIdentityConflict):
                ledger.receive(ack(changed), 1_100 * MS)
        self.assertIsNone(ledger.outcome(row))

    def test_unknown_decision_is_orphan_and_cannot_close_ticket(self) -> None:
        row = identity()
        ledger = A.OperationalAckLedgerV1()
        ledger.open(row, 1_000 * MS)
        unknown = identity(decision_seq=8, ticket_seq=8, frame_id=1313)
        self.assertIs(ledger.receive(ack(unknown), 1_100 * MS),
                      A.AckClass.UNKNOWN_ORPHAN)
        self.assertIsNone(ledger.outcome(row))
        self.assertEqual(len(ledger.unknown_orphans), 1)


class DurableOperationalEvidenceTest(unittest.TestCase):
    def test_success_and_timeout_are_durable_and_exactly_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            store = A.OperationalEvidenceStoreV1.create(
                Path(parent) / "operational")
            ledger = A.OperationalAckLedgerV1(evidence_store=store)

            successful = identity()
            success_ack = ack(successful, b"usable-tail-success")
            ledger.open(successful, 1_000 * MS)
            self.assertIs(
                ledger.receive(success_ack, 1_170 * MS),
                A.AckClass.ACCEPTED)

            timed_out = identity(
                decision_seq=8, ticket_seq=8, frame_id=1313,
                tensor_seq=15)
            late_ack = ack(timed_out, b"usable-tail-late")
            ledger.open(timed_out, 2_000 * MS)
            self.assertIs(
                ledger.receive(late_ack, 2_170 * MS + 1),
                A.AckClass.LATE_ORPHAN)

            snapshot = store.verify_all(require_all_resolved=True)
            self.assertEqual(len(snapshot.opened_identities), 2)
            self.assertEqual(len(snapshot.outcomes), 2)
            self.assertEqual(len(snapshot.late_orphans), 1)
            self.assertEqual(len(snapshot.unknown_orphans), 0)
            by_identity = {row.identity: row for row in snapshot.outcomes}

            accepted = by_identity[successful]
            self.assertTrue(accepted.success)
            self.assertEqual(accepted.action_open_monotonic_raw_ns,
                             1_000 * MS)
            self.assertEqual(accepted.ack_receipt_monotonic_raw_ns,
                             1_170 * MS)
            self.assertEqual(accepted.observed_latency_ns, 170 * MS)
            self.assertEqual(accepted.state_latency_ns, 170 * MS)
            self.assertEqual(accepted.tail_output_sha256,
                             success_ack.tail_output_sha256)

            censored = by_identity[timed_out]
            self.assertFalse(censored.success)
            self.assertEqual(censored.action_open_monotonic_raw_ns,
                             2_000 * MS)
            self.assertEqual(censored.resolution_monotonic_raw_ns,
                             2_170 * MS + 1)
            self.assertIsNone(censored.ack_receipt_monotonic_raw_ns)
            self.assertIsNone(censored.observed_latency_ns)
            self.assertEqual(censored.state_latency_ns, 0)
            self.assertIsNone(censored.tail_output_sha256)

            orphan = snapshot.late_orphans[0]
            self.assertEqual(orphan["identity"], timed_out)
            self.assertEqual(orphan["receipt_monotonic_raw_ns"],
                             2_170 * MS + 1)
            self.assertEqual(orphan["observed_latency_ns"],
                             170 * MS + 1)
            self.assertEqual(orphan["tail_output_sha256"],
                             late_ack.tail_output_sha256)

            self.assertEqual(ledger.resolved_outcomes(), snapshot.outcomes)
            for path in store.root.rglob("*.json"):
                evidence = path.read_text(encoding="ascii").lower()
                for forbidden in ("q_perc", "reward", "ground_truth"):
                    self.assertNotIn(forbidden, evidence)

    def test_poll_timeout_and_later_ack_retain_both_clock_facts(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            store = A.OperationalEvidenceStoreV1.create(
                Path(parent) / "operational")
            ledger = A.OperationalAckLedgerV1(evidence_store=store)
            row = identity()
            ledger.open(row, 3_000 * MS)
            self.assertEqual(ledger.poll(3_170 * MS + 1), (row,))
            self.assertIs(ledger.receive(ack(row), 3_200 * MS),
                          A.AckClass.LATE_ORPHAN)

            snapshot = store.verify_all(require_all_resolved=True)
            outcome = snapshot.outcomes[0]
            orphan = snapshot.late_orphans[0]
            self.assertEqual(outcome.resolution_monotonic_raw_ns,
                             3_170 * MS + 1)
            self.assertEqual(outcome.state_latency_ns, 0)
            self.assertEqual(orphan["receipt_monotonic_raw_ns"],
                             3_200 * MS)
            self.assertEqual(orphan["observed_latency_ns"], 200 * MS)

    def test_unknown_ack_is_durable_but_is_not_a_policy_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            store = A.OperationalEvidenceStoreV1.create(
                Path(parent) / "operational")
            ledger = A.OperationalAckLedgerV1(evidence_store=store)
            row = identity()
            unknown_ack = ack(row)
            self.assertIs(ledger.receive(unknown_ack, 5_000 * MS),
                          A.AckClass.UNKNOWN_ORPHAN)
            snapshot = store.verify_all(require_all_resolved=True)
            self.assertEqual(snapshot.opened_identities, ())
            self.assertEqual(snapshot.outcomes, ())
            self.assertEqual(len(snapshot.unknown_orphans), 1)
            self.assertEqual(snapshot.unknown_orphans[0]["identity"], row)
            self.assertEqual(
                snapshot.unknown_orphans[0]["tail_output_sha256"],
                unknown_ack.tail_output_sha256)

    def test_unresolved_and_tampered_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "operational"
            store = A.OperationalEvidenceStoreV1.create(root)
            ledger = A.OperationalAckLedgerV1(evidence_store=store)
            row = identity()
            ledger.open(row, 7_000 * MS)
            self.assertEqual(len(store.verify_all().opened_identities), 1)
            with self.assertRaisesRegex(A.OperationalEvidenceError,
                                        "not every opened"):
                store.load_outcomes(require_all_resolved=True)

            ledger.receive(ack(row), 7_100 * MS)
            outcome_path = next(store.outcomes.iterdir())
            raw = json.loads(outcome_path.read_text(encoding="ascii"))
            raw["state_latency_ns"] += 1
            outcome_path.write_text(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii")
            with self.assertRaisesRegex(A.OperationalEvidenceError,
                                        "timestamps/state latency"):
                store.verify_all(require_all_resolved=True)
            with self.assertRaises(A.OperationalCreateOnlyError):
                A.OperationalEvidenceStoreV1.create(root)


if __name__ == "__main__":
    unittest.main()
