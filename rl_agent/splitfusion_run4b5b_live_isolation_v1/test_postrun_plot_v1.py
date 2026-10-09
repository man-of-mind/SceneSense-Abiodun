"""CPU-only PNG regression for the aligned post-run metrics plot."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_artifact_v1 as A
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_evaluator_v1 as E
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_plot_v1 as P
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_postrun_evaluator_v1 import (
    masks,
    objects,
    outcome,
    reward_spec,
)
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_operational_ack_v1 import identity


class PostRunPlotTest(unittest.TestCase):
    def _metrics(self, root: Path) -> Path:
        prediction_root = root / "predictions"
        gt_root = root / "gt"
        prediction_store = B.PredictionEvidenceStoreV1.create(prediction_root)
        gt_store = A.GroundTruthEvidenceStoreV1.create(gt_root)
        pred_mask, gt_mask = masks()
        exact = identity()
        payload = A.encode_bundle(
            kind=A.PREDICTION_KIND, identity=exact,
            objects=objects(0.2), semantic_mask=pred_mask,
        )
        publication = B.TailOutputBranchCoordinatorV1().publish(exact, payload)
        prediction_store.write(publication, payload, 1)
        gt_store.write(
            identity=exact, eligible_objects=objects(), semantic_mask=gt_mask,
            recorded_monotonic_raw_ns=2,
        )
        result = E.PostRunEvaluatorV1(reward_spec=reward_spec()).evaluate(
            prediction_root=prediction_root, ground_truth_root=gt_root,
            outcomes=[outcome(exact, success=True)],
            payload_bytes_by_identity={exact.exact_sha256(): 123_456},
            output_root=root / "evaluation",
        )
        return result.frame_metrics_path

    def test_png_only_create_only_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = self._metrics(root)
            output = root / "offline_metrics.png"
            self.assertEqual(
                P.plot_frame_metrics_png(
                    frame_metrics_csv=metrics, output_png=output), output
            )
            data = output.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(len(data), 10_000)
            with self.assertRaises(P.PostRunPlotError):
                P.plot_frame_metrics_png(
                    frame_metrics_csv=metrics, output_png=output)
            with self.assertRaisesRegex(P.PostRunPlotError, "png"):
                P.plot_frame_metrics_png(
                    frame_metrics_csv=metrics, output_png=root / "metrics.pdf")


if __name__ == "__main__":
    unittest.main()
