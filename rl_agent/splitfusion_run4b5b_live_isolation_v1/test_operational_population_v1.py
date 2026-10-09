"""Regressions for the operational-ledger-primary post-run population."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    branch_evidence_v1 as B,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    operational_trace_v1 as T,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    postrun_artifact_v1 as A,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    postrun_evaluator_v1 as E,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.postrun_operational_population_v1 import (
    evaluate_from_operational_trace,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_operational_ack_v1 import (
    identity,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_postrun_evaluator_v1 import (
    masks,
    objects,
    outcome,
    reward_spec,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class OperationalTraceStoreTest(unittest.TestCase):
    def test_create_verify_and_timeout_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = T.OperationalTraceStoreV1.create(Path(directory) / "trace")
            first = identity()
            second = dataclasses.replace(
                first, decision_seq=8, ticket_seq=8, frame_id=1313,
                tensor_seq=15, capture_timestamp_ns=first.capture_timestamp_ns + 1,
            )
            success = store.write(outcome(first, success=True), 123_000)
            timeout = store.write(outcome(second, success=False), 99_000)
            self.assertEqual(store.verify_all(), (success, timeout))
            self.assertEqual(timeout.outcome.state_latency_ns, 0)
            self.assertIsNone(timeout.outcome.observed_latency_ns)
            with self.assertRaises(T.OperationalTraceCreateOnlyError):
                store.write(outcome(first, success=True), 123_000)

    def test_tamper_and_conflicting_logical_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trace"
            store = T.OperationalTraceStoreV1.create(root)
            exact = identity()
            record = store.write(outcome(exact, success=True), 123_000)
            path = root / "records" / f"{record.identity_sha256}.json"
            raw = json.loads(path.read_text(encoding="ascii"))
            raw["payload_bytes"] = 0
            path.write_text(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaises(T.OperationalTraceIntegrityError):
                store.verify_all()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trace"
            store = T.OperationalTraceStoreV1.create(root)
            exact = identity()
            store.write(outcome(exact, success=True), 123_000)
            conflicting = dataclasses.replace(exact, q_e4=1)
            store.write(outcome(conflicting, success=True), 124_000)
            with self.assertRaisesRegex(
                    T.OperationalTraceIntegrityError, "conflicting"):
                store.verify_all()


class OperationalPopulationTest(unittest.TestCase):
    def _stores(self, root: Path):
        return (
            B.PredictionEvidenceStoreV1.create(root / "predictions"),
            A.GroundTruthEvidenceStoreV1.create(root / "gt"),
        )

    def _prediction(self, store: B.PredictionEvidenceStoreV1, exact) -> None:
        pred_mask, _truth_mask = masks()
        payload = A.encode_bundle(
            kind=A.PREDICTION_KIND, identity=exact,
            objects=objects(0.2), semantic_mask=pred_mask,
        )
        publication = B.TailOutputBranchCoordinatorV1().publish(exact, payload)
        store.write(publication, payload, 1)

    def _truth(self, store: A.GroundTruthEvidenceStoreV1, exact) -> None:
        _pred_mask, truth_mask = masks()
        store.write(
            identity=exact, eligible_objects=objects(),
            semantic_mask=truth_mask, recorded_monotonic_raw_ns=2,
        )

    def test_timeout_without_prediction_remains_primary_row_with_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            exact = identity()
            self._truth(truths, exact)
            result = E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                prediction_root=predictions.root,
                ground_truth_root=truths.root,
                outcomes=[outcome(exact, success=False)],
                payload_bytes_by_identity={exact.exact_sha256(): 123_000},
                output_root=root / "evaluation",
            )
            row = read_rows(result.frame_metrics_path)[0]
            self.assertEqual(row["operational_terminal"], "TIMEOUT")
            self.assertEqual(row["prediction_present"], "0")
            self.assertEqual(row["ground_truth_present"], "1")
            self.assertEqual(row["q_perc"], "")
            self.assertEqual(row["operational_latency_ms"], "")
            self.assertEqual(row["latency_censor_lower_bound_ms"], "170.0")
            self.assertEqual(row["evaluation_reward"], "-1.0")
            self.assertEqual(row["quality_exclusion_reason"],
                             "TAIL_OUTPUT_NOT_PRODUCED")
            summary = json.loads(result.summary_path.read_text(encoding="ascii"))
            self.assertEqual(summary["primary_population"],
                             "DURABLE_OPERATIONAL_OUTCOMES")
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["prediction_present_count"], 0)

    def test_timeout_without_prediction_or_gt_is_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            exact = identity()
            result = E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                prediction_root=predictions.root,
                ground_truth_root=truths.root,
                outcomes=[outcome(exact, success=False)],
                payload_bytes_by_identity={exact.exact_sha256(): 123_000},
                output_root=root / "evaluation",
            )
            row = read_rows(result.frame_metrics_path)[0]
            self.assertEqual(row["quality_exclusion_reason"],
                             "TAIL_OUTPUT_AND_GT_NOT_RETAINED")
            self.assertEqual(row["evaluation_reward"], "-1.0")

    def test_prediction_without_gt_is_explicitly_unscored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            exact = identity()
            self._prediction(predictions, exact)
            result = E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                prediction_root=predictions.root,
                ground_truth_root=truths.root,
                outcomes=[outcome(exact, success=False)],
                payload_bytes_by_identity={exact.exact_sha256(): 123_000},
                output_root=root / "evaluation",
            )
            row = read_rows(result.frame_metrics_path)[0]
            self.assertEqual(row["prediction_present"], "1")
            self.assertEqual(row["ground_truth_present"], "0")
            self.assertEqual(row["quality_exclusion_reason"],
                             "GROUND_TRUTH_NOT_RETAINED")
            self.assertEqual(row["q_perc"], "")
            self.assertEqual(row["evaluation_reward"], "-1.0")

    def test_success_without_prediction_and_orphan_evidence_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            exact = identity()
            with self.assertRaisesRegex(E.PostRunJoinError, "successful"):
                E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                    prediction_root=predictions.root,
                    ground_truth_root=truths.root,
                    outcomes=[outcome(exact, success=True)],
                    payload_bytes_by_identity={exact.exact_sha256(): 123_000},
                    output_root=root / "evaluation",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            orphan = identity()
            self._prediction(predictions, orphan)
            different = dataclasses.replace(
                orphan, decision_seq=10, ticket_seq=10, frame_id=1400,
                tensor_seq=20,
                capture_timestamp_ns=orphan.capture_timestamp_ns + 1,
            )
            with self.assertRaisesRegex(E.PostRunJoinError, "prediction"):
                E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                    prediction_root=predictions.root,
                    ground_truth_root=truths.root,
                    outcomes=[outcome(different, success=False)],
                    payload_bytes_by_identity={
                        different.exact_sha256(): 123_000},
                    output_root=root / "evaluation",
                )

    def test_durable_trace_seam_drives_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, truths = self._stores(root)
            exact = identity()
            trace = T.OperationalTraceStoreV1.create(root / "trace")
            trace.write(outcome(exact, success=False), 123_000)
            result = evaluate_from_operational_trace(
                E.PostRunEvaluatorV1(reward_spec=reward_spec()),
                operational_trace_root=trace.root,
                prediction_root=predictions.root,
                ground_truth_root=truths.root,
                output_root=root / "evaluation",
            )
            self.assertEqual(result.frame_count, 1)
            self.assertEqual(
                read_rows(result.frame_metrics_path)[0]["operational_terminal"],
                "TIMEOUT",
            )


if __name__ == "__main__":
    unittest.main()
