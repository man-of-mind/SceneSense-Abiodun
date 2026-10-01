"""CPU-only tests for separate GT retention and post-run evaluation."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    LocalizationCombiner,
    RewardSpecV1,
)

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_ack_v1 as O
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_artifact_v1 as A
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_evaluator_v1 as E
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_operational_ack_v1 import identity as base_identity


def reward_spec() -> RewardSpecV1:
    return RewardSpecV1(
        spec_id="splitfusion_postrun_test_qperc_v1",
        spec_version=1,
        w_loc_person=0.6,
        w_loc_vehicle=0.4,
        tau_person_m=1.2,
        tau_vehicle_m=1.0,
        localization_combiner=LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN,
        w_seg_person=0.6,
        w_seg_vehicle=0.4,
        seg_reference_person_iou=0.52789408,
        seg_reference_vehicle_iou=0.899012847,
        segmentation_modulation_beta=0.3,
        w_quality=1.0,
        w_latency=0.25,
        lambda_mode=0.0,
        lambda_q=0.0,
        r_registered_failure=-1.0,
        gamma_per_tensor=0.99,
        provenance={"purpose": "CPU_ONLY_TEST"},
    )


def objects(offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "actor_id": 100,
            "class_name": "vehicle",
            "world_x": 0.0 + offset,
            "world_y": 0.0,
            "size_x": 4.0,
            "size_y": 2.0,
            "size_z": 1.5,
            "yaw_deg": 0.0,
        },
        {
            "actor_id": 200,
            "class_name": "person",
            "world_x": 5.0 + offset,
            "world_y": 1.0,
            "size_x": 0.8,
            "size_y": 0.6,
            "size_z": 1.7,
            "yaw_deg": 5.0,
        },
    ]


def masks() -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((8, 10), dtype=np.uint8)
    truth[1:4, 1:5] = 1
    truth[4:7, 7:9] = 2
    prediction = truth.copy()
    prediction[3, 4] = 0
    prediction[4, 6] = 2
    return prediction, truth


def outcome(identity: O.FrameActionIdentityV1, *, success: bool) -> O.OperationalOutcomeV1:
    opened = 10_000_000_000 + identity.frame_id * 1_000_000
    if success:
        latency = 100_000_000
        return O.OperationalOutcomeV1(
            identity=identity,
            terminal=O.OperationalTerminal.SUCCESS,
            action_open_monotonic_raw_ns=opened,
            resolution_monotonic_raw_ns=opened + latency,
            observed_latency_ns=latency,
            state_latency_ns=latency,
            accepted_ack_sha256=hashlib.sha256(b"ack").hexdigest(),
        )
    return O.OperationalOutcomeV1(
        identity=identity,
        terminal=O.OperationalTerminal.TIMEOUT,
        action_open_monotonic_raw_ns=opened,
        resolution_monotonic_raw_ns=opened + O.FIRST_LATE_TICK_NS,
        observed_latency_ns=None,
        state_latency_ns=0,
        accepted_ack_sha256=None,
    )


class ArtifactCodecTest(unittest.TestCase):
    def test_round_trip_is_deterministic_owned_and_identity_bound(self) -> None:
        pred, _truth = masks()
        expected = base_identity()
        first = A.encode_bundle(
            kind=A.PREDICTION_KIND, identity=expected,
            objects=objects(0.2), semantic_mask=pred,
        )
        second = A.encode_bundle(
            kind=A.PREDICTION_KIND, identity=expected,
            objects=objects(0.2), semantic_mask=pred,
        )
        self.assertEqual(first, second)
        decoded = A.decode_bundle(
            first, expected_kind=A.PREDICTION_KIND,
            expected_identity=expected,
        )
        self.assertEqual(decoded.identity, expected)
        self.assertEqual(decoded.objects, tuple(objects(0.2)))
        np.testing.assert_array_equal(decoded.semantic_mask, pred)
        self.assertFalse(decoded.semantic_mask.flags.writeable)

    def test_tamper_kind_identity_and_foreign_label_fail_closed(self) -> None:
        pred, _truth = masks()
        payload = A.encode_bundle(
            kind=A.PREDICTION_KIND, identity=base_identity(),
            objects=objects(), semantic_mask=pred,
        )
        damaged = bytearray(payload)
        damaged[-1] ^= 1
        with self.assertRaisesRegex(A.PostRunArtifactError, "digest"):
            A.decode_bundle(bytes(damaged))
        with self.assertRaisesRegex(A.PostRunArtifactError, "kind"):
            A.decode_bundle(payload, expected_kind=A.GROUND_TRUTH_KIND)
        with self.assertRaisesRegex(A.PostRunArtifactError, "identity"):
            A.decode_bundle(payload, expected_identity=base_identity(q_e4=10))
        foreign = pred.copy()
        foreign[0, 0] = 3
        with self.assertRaisesRegex(A.PostRunArtifactError, "class"):
            A.encode_bundle(
                kind=A.PREDICTION_KIND, identity=base_identity(),
                objects=objects(), semantic_mask=foreign,
            )


class GroundTruthStoreTest(unittest.TestCase):
    def test_create_verify_and_tamper(self) -> None:
        _pred, truth = masks()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gt"
            store = A.GroundTruthEvidenceStoreV1.create(root)
            record = store.write(
                identity=base_identity(), eligible_objects=objects(),
                semantic_mask=truth, recorded_monotonic_raw_ns=7,
            )
            self.assertEqual(store.verify_all(), (record,))
            self.assertEqual(record.postrun_join_key,
                             base_identity().postrun_join_key())
            with self.assertRaises(A.GroundTruthCreateOnlyError):
                store.write(
                    identity=base_identity(), eligible_objects=objects(),
                    semantic_mask=truth, recorded_monotonic_raw_ns=8,
                )
            artifact = root / record.artifact_relative_path
            artifact.write_bytes(artifact.read_bytes() + b"x")
            with self.assertRaisesRegex(A.GroundTruthIntegrityError, "digest"):
                store.verify_all()


class PostRunEvaluatorTest(unittest.TestCase):
    def _build(self, root: Path, *, include_undefined: bool = False):
        prediction_root = root / "predictions"
        gt_root = root / "gt"
        predictions = B.PredictionEvidenceStoreV1.create(prediction_root)
        truths = A.GroundTruthEvidenceStoreV1.create(gt_root)
        coordinator = B.TailOutputBranchCoordinatorV1()
        pred_mask, gt_mask = masks()
        identities = [
            base_identity(),
            dataclasses.replace(
                base_identity(), frame_id=1841, tensor_seq=78,
                decision_seq=43, ticket_seq=43,
                capture_timestamp_ns=1_790_000_000_100_000_000,
            ),
        ]
        if include_undefined:
            identities.append(dataclasses.replace(
                base_identity(), frame_id=1842, tensor_seq=79,
                decision_seq=44, ticket_seq=44,
                capture_timestamp_ns=1_790_000_000_200_000_000,
            ))
        outcomes: list[O.OperationalOutcomeV1] = []
        payloads: dict[str, int] = {}
        for index, exact in enumerate(identities):
            prediction_objects = [] if index == 2 else objects(0.2)
            truth_objects = [] if index == 2 else objects()
            p_mask = np.zeros_like(pred_mask) if index == 2 else pred_mask
            t_mask = np.zeros_like(gt_mask) if index == 2 else gt_mask
            payload = A.encode_bundle(
                kind=A.PREDICTION_KIND, identity=exact,
                objects=prediction_objects, semantic_mask=p_mask,
            )
            publication = coordinator.publish(exact, payload)
            predictions.write(
                publication, payload,
                recorded_monotonic_raw_ns=8_000_000_000 + index,
            )
            truths.write(
                identity=exact, eligible_objects=truth_objects,
                semantic_mask=t_mask,
                recorded_monotonic_raw_ns=8_100_000_000 + index,
            )
            outcomes.append(outcome(exact, success=index != 1))
            payloads[exact.exact_sha256()] = 100_000 + index * 10_000
        return prediction_root, gt_root, outcomes, payloads

    def test_exact_join_outputs_quality_reward_and_censored_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt, outcomes, payloads = self._build(root, include_undefined=True)
            result = E.PostRunEvaluatorV1(
                reward_spec=reward_spec()).evaluate(
                    prediction_root=pred, ground_truth_root=gt,
                    outcomes=outcomes,
                    payload_bytes_by_identity=payloads,
                    output_root=root / "evaluation",
                )
            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.quality_defined_count, 2)
            with result.frame_metrics_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_order"]) for row in rows], [0, 1, 2])
            success, timeout, undefined = rows
            self.assertEqual(success["operational_terminal"], "SUCCESS")
            self.assertEqual(float(success["operational_latency_ms"]), 100.0)
            expected_reward = float(success["q_perc"]) - 0.25 * 100.0 / 170.0
            self.assertAlmostEqual(float(success["evaluation_reward"]), expected_reward)
            self.assertGreater(float(success["q_perc"]), 0.0)
            self.assertLessEqual(float(success["q_perc"]), 1.0)

            self.assertEqual(timeout["operational_terminal"], "TIMEOUT")
            self.assertEqual(timeout["operational_latency_ms"], "")
            self.assertEqual(float(timeout["latency_censor_lower_bound_ms"]), 170.0)
            self.assertEqual(float(timeout["evaluation_reward"]), -1.0)
            self.assertEqual(timeout["evaluation_status"],
                             "TIMEOUT_EVALUATED_QUALITY")

            self.assertEqual(undefined["quality_defined"], "0")
            self.assertEqual(undefined["q_perc"], "")
            self.assertEqual(undefined["evaluation_reward"], "")
            self.assertEqual(undefined["evaluation_status"],
                             "EXCLUDED_QUALITY_UNAVAILABLE")
            summary = json.loads(result.summary_path.read_text(encoding="ascii"))
            self.assertFalse(summary["live_policy_feedback"])
            self.assertFalse(summary["ground_truth_transmitted_to_edge"])
            self.assertEqual(summary["timeout_count"], 1)
            self.assertEqual(summary["quality_excluded_count"], 1)
            with result.object_metrics_path.open(newline="", encoding="utf-8") as handle:
                object_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["match_status"] == "TP" for row in object_rows))
            self.assertTrue(all(row["localization_error_m"]
                                for row in object_rows
                                if row["match_status"] == "TP"))

    def test_identity_set_drift_and_existing_output_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt, outcomes, payloads = self._build(root)
            evaluator = E.PostRunEvaluatorV1(reward_spec=reward_spec())
            with self.assertRaisesRegex(E.PostRunJoinError, "payload"):
                evaluator.evaluate(
                    prediction_root=pred, ground_truth_root=gt,
                    outcomes=outcomes,
                    payload_bytes_by_identity={
                        key: value for index, (key, value)
                        in enumerate(payloads.items()) if index == 0
                    },
                    output_root=root / "bad",
                )
            output = root / "evaluation"
            evaluator.evaluate(
                prediction_root=pred, ground_truth_root=gt,
                outcomes=outcomes, payload_bytes_by_identity=payloads,
                output_root=output,
            )
            with self.assertRaises(E.PostRunCreateOnlyError):
                evaluator.evaluate(
                    prediction_root=pred, ground_truth_root=gt,
                    outcomes=outcomes, payload_bytes_by_identity=payloads,
                    output_root=output,
                )

    def test_prediction_bundle_identity_is_verified_not_only_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt, outcomes, payloads = self._build(root)
            store = B.PredictionEvidenceStoreV1.open_existing(pred)
            record = store.verify_all()[0]
            artifact = pred / record.artifact_relative_path
            pred_mask, _ = masks()
            wrong = A.encode_bundle(
                kind=A.PREDICTION_KIND,
                identity=dataclasses.replace(record.identity, q_e4=1),
                objects=objects(), semantic_mask=pred_mask,
            )
            artifact.write_bytes(wrong)
            raw_path = pred / "records" / f"{record.identity_sha256}.json"
            raw = json.loads(raw_path.read_text(encoding="ascii"))
            raw["prediction_sha256"] = hashlib.sha256(wrong).hexdigest()
            raw_path.write_text(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(A.PostRunArtifactError, "identity"):
                E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
                    prediction_root=pred, ground_truth_root=gt,
                    outcomes=outcomes, payload_bytes_by_identity=payloads,
                    output_root=root / "evaluation",
                )


if __name__ == "__main__":
    unittest.main()
