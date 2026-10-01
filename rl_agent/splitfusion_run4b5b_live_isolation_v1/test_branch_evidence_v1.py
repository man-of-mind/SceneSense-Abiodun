"""CPU-only tests for branch independence and create-only prediction evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_operational_ack_v1 import identity


class BranchIndependenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = B.TailOutputBranchCoordinatorV1()
        self.payload = b"exact-tail-prediction-bytes"
        self.publication = self.coordinator.publish(identity(), self.payload)

    def test_ack_exists_before_either_branch_and_never_changes(self) -> None:
        before = self.publication.ack.canonical_bytes()
        self.assertIsNone(self.coordinator.result(identity(), B.Branch.MAP))
        self.assertIsNone(self.coordinator.result(identity(), B.Branch.EVALUATION))

        evaluation = self.coordinator.complete(
            self.publication.evaluation_work,
            status=B.BranchStatus.FAILED,
            detail_code="GT_NOT_RETAINED",
            evidence_sha256=None,
        )
        self.assertIs(evaluation.status, B.BranchStatus.FAILED)
        self.assertIsNone(self.coordinator.result(identity(), B.Branch.MAP))
        self.assertEqual(self.coordinator.ack(identity()).canonical_bytes(), before)

        map_result = self.coordinator.complete(
            self.publication.map_work,
            status=B.BranchStatus.SUCCEEDED,
            detail_code="MAP_PUBLISHED",
            evidence_sha256=hashlib.sha256(b"map evidence").hexdigest(),
        )
        self.assertIs(map_result.status, B.BranchStatus.SUCCEEDED)
        self.assertIs(self.coordinator.result(identity(), B.Branch.EVALUATION),
                      evaluation)
        self.assertEqual(self.coordinator.ack(identity()).canonical_bytes(), before)

    def test_map_failure_does_not_complete_or_cancel_evaluation(self) -> None:
        self.coordinator.complete(
            self.publication.map_work,
            status=B.BranchStatus.FAILED,
            detail_code="MAP_ENDPOINT_DOWN",
            evidence_sha256=None,
        )
        self.assertIsNone(self.coordinator.result(identity(), B.Branch.EVALUATION))
        result = self.coordinator.complete(
            self.publication.evaluation_work,
            status=B.BranchStatus.SKIPPED,
            detail_code="NO_LOCAL_GT",
            evidence_sha256=None,
        )
        self.assertIs(result.status, B.BranchStatus.SKIPPED)

    def test_duplicate_branch_result_idempotent_conflict_refused(self) -> None:
        kwargs = dict(status=B.BranchStatus.FAILED,
                      detail_code="MAP_ENDPOINT_DOWN", evidence_sha256=None)
        first = self.coordinator.complete(self.publication.map_work, **kwargs)
        self.assertIs(self.coordinator.complete(self.publication.map_work, **kwargs),
                      first)
        with self.assertRaises(B.BranchConflict):
            self.coordinator.complete(
                self.publication.map_work,
                status=B.BranchStatus.FAILED,
                detail_code="DIFFERENT_FAILURE",
                evidence_sha256=None,
            )

    def test_publication_duplicate_and_identity_conflict_refused(self) -> None:
        with self.assertRaises(B.BranchEvidenceError):
            self.coordinator.publish(identity(), self.payload)
        with self.assertRaises(B.PublicationConflict):
            self.coordinator.publish(identity(q_e4=6785), self.payload)


class PredictionEvidenceTest(unittest.TestCase):
    def _publication(self):
        payload = b"opaque-prediction-record"
        coordinator = B.TailOutputBranchCoordinatorV1()
        return payload, coordinator.publish(identity(), payload)

    def test_create_write_verify_and_exact_join_identity(self) -> None:
        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            store = B.PredictionEvidenceStoreV1.create(root)
            record = store.write(publication, payload, 8_000_000_000)
            verified = store.verify_all()
            self.assertEqual(verified, (record,))
            self.assertEqual(record.postrun_join_key,
                             publication.ack.identity.postrun_join_key())
            self.assertEqual(
                (record.postrun_join_key["run_id"],
                 record.postrun_join_key["stream_id"],
                 record.postrun_join_key["frame_id"],
                 record.postrun_join_key["capture_timestamp_ns"],
                 record.postrun_join_key["session_uuid"],
                 record.postrun_join_key["decision_seq"]),
                (publication.ack.identity.run_id,
                 publication.ack.identity.stream_id,
                 publication.ack.identity.frame_id,
                 publication.ack.identity.capture_timestamp_ns,
                 publication.ack.identity.session_uuid,
                 publication.ack.identity.decision_seq),
            )
            self.assertEqual((root / record.artifact_relative_path).read_bytes(),
                             payload)

    def test_store_and_records_are_create_only(self) -> None:
        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            store = B.PredictionEvidenceStoreV1.create(root)
            store.write(publication, payload, 1)
            with self.assertRaises(B.CreateOnlyError):
                store.write(publication, payload, 2)
            with self.assertRaises(B.CreateOnlyError):
                B.PredictionEvidenceStoreV1.create(root)

    def test_acknowledged_and_persisted_bytes_must_match(self) -> None:
        _payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            store = B.PredictionEvidenceStoreV1.create(
                Path(directory) / "predictions")
            with self.assertRaisesRegex(B.EvidenceIntegrityError, "differ"):
                store.write(publication, b"other bytes", 1)

    def test_artifact_and_record_tampering_are_detected(self) -> None:
        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            store = B.PredictionEvidenceStoreV1.create(root)
            record = store.write(publication, payload, 1)
            artifact = root / record.artifact_relative_path
            artifact.write_bytes(payload + b"tamper")
            with self.assertRaisesRegex(B.EvidenceIntegrityError, "digest"):
                store.verify_all()

        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            store = B.PredictionEvidenceStoreV1.create(root)
            record = store.write(publication, payload, 1)
            path = root / "records" / f"{record.identity_sha256}.json"
            raw = json.loads(path.read_text(encoding="ascii"))
            raw["prediction_sha256"] = "0" * 64
            path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                            encoding="ascii")
            with self.assertRaisesRegex(B.EvidenceIntegrityError, "digest"):
                store.verify_all()

    def test_record_schema_has_no_evaluated_quality_or_gt(self) -> None:
        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            store = B.PredictionEvidenceStoreV1.create(
                Path(directory) / "predictions")
            record = store.write(publication, payload, 1)
            text = record.canonical_bytes().lower()
            for token in (b"q_perc", b"reward", b"ground_truth"):
                self.assertNotIn(token, text)
            self.assertNotIn("quality", record.as_dict())

    def test_record_refuses_foreign_identity_even_if_filename_is_unchanged(self) -> None:
        payload, publication = self._publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            store = B.PredictionEvidenceStoreV1.create(root)
            record = store.write(publication, payload, 1)
            path = root / "records" / f"{record.identity_sha256}.json"
            raw = json.loads(path.read_text(encoding="ascii"))
            raw["identity"]["frame_id"] += 1
            path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                            encoding="ascii")
            with self.assertRaises(B.EvidenceIntegrityError):
                store.verify_all()


if __name__ == "__main__":
    unittest.main()
