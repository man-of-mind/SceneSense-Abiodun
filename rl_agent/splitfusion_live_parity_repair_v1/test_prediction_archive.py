from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .prediction_archive import PredictionArchive


def ticket():
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import Pipeline, processor
    return processor().process(Pipeline().step().wire, edge_timing={}).evaluation


class PredictionArchiveTests(unittest.TestCase):
    def test_real_processor_ticket_owned_labels_records_and_unavailable_gt(self):
        value = ticket()
        archive = PredictionArchive(max_frames=1)
        key = archive.add(value)
        scoring_ticket = archive.owned_ticket(value)
        value.predicted_mask[:] = 2
        value.records[0]["world_x"] = 999.
        np.testing.assert_array_equal(scoring_ticket.predicted_mask, 0)
        self.assertEqual(scoring_ticket.records[0]["world_x"], 1.)
        self.assertFalse(scoring_ticket.predicted_mask.flags.writeable)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "predictions"
            result = archive.flush(path)
            self.assertEqual(result["frames"], 1)
            self.assertEqual(result["evaluated_frames"], 0)
            np.testing.assert_array_equal(np.load(path / f"{key}.labels.npy"), 0)
            self.assertEqual(json.loads((path / f"{key}.records.json").read_text())[0]["world_x"], 1.)
            for name, digest in result["sha256"].items():
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(ValueError):
                archive.flush(path)

    def test_duplicate_identity_conflict_and_bounds(self):
        value = ticket()
        archive = PredictionArchive(max_frames=1)
        key = archive.add(value)
        self.assertEqual(key, archive.add(replace(value, enqueued_wall_ns=value.enqueued_wall_ns + 1)))
        value.predicted_mask[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            archive.add(value)
        with self.assertRaisesRegex(ValueError, "capacity"):
            PredictionArchive(max_frames=1, max_bytes=1).add(value)

    def test_invalid_identity_labels_scores_and_lifecycle(self):
        value = ticket()
        for invalid in (replace(value, context=replace(value.context, frame_id=1)),
                        replace(value, predicted_mask=np.full((4, 4), 255, np.uint8)),
                        replace(value, records=({"score": float("nan")},))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PredictionArchive(max_frames=1).add(invalid)
        archive = PredictionArchive(max_frames=1)
        with self.assertRaisesRegex(ValueError, "no open"):
            archive.evaluated(value, {})
        archive.add(value)
        archive.evaluated(value, {"matched": []})
        archive.evaluated(value, {"matched": []})
        with self.assertRaisesRegex(ValueError, "conflicting evaluation"):
            archive.evaluated(value, {"matched": [1]})


if __name__ == "__main__":
    unittest.main()
