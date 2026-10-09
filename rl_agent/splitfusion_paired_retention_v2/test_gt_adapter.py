"""Exact rescoring on real archived valid_b1000 frames (read-only) and GT-reference admission."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from . import fixtures as F, gt_adapter as G, observer as OB, plan as P, validate as V

HAVE_ARCHIVE = F.VALID_B1000_ARCHIVE.is_dir() and len(list(F.VALID_B1000_ARCHIVE.glob("frame_*"))) == 493
SKIP = "valid_b1000 numeric-GT archive (493 frames) not present at " + str(F.VALID_B1000_ARCHIVE)
ANCHOR_FRAME = 1992      # the one valid_b1000 decision scored through anchor provenance (q_e4 5000 == anchor 68)


@unittest.skipUnless(HAVE_ARCHIVE, SKIP)
class ArchiveRescoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordered = F.archive_frames_by_tensor()
        cls.plan = P.build_plan(plan_id="valid_b1000_replay", planned_frames=1000)

    def test_producing_worktree_scorer_sources_identical(self):
        if not F.PROFILER_WORKTREE.is_dir():
            self.skipTest("profiler worktree absent")
        main = G.scorer_source_digests()
        worktree = G.scorer_source_digests(F.PROFILER_WORKTREE)
        self.assertEqual(main, worktree)

    def test_plan_sampled_frames_rescore_exactly_field_by_field(self):
        """The pre-launch plan applied to the 493 real decisions selects 99; each must reproduce exactly."""
        frames = [self.ordered[o][1] for o in self.plan["ordinals"] if o < len(self.ordered)]
        self.assertEqual(len(frames), 99)
        frames.append(G.frame_directory(F.VALID_B1000_ARCHIVE, ANCHOR_FRAME))
        undefined = 0
        for directory in frames:
            with self.subTest(frame=directory.name):
                result = G.rescore(G.load_frame(directory))
                self.assertTrue(result["exact"], {k: v for k, v in result["checks"].items() if not v})
                for field in ("numerical_score_json_bytes", "measurement", "q_perc", "prediction_records",
                              "numerical_matches", "confusion_matrix", "vehicle_detection", "ready_identity",
                              "evaluate_prepared_measurement", "prediction_digest"):
                    self.assertIs(result["checks"][field], True, field)
                undefined += result["q_perc"] is None
        self.assertGreater(undefined, 0)       # undefined-quality frames are covered too

    def test_rescore_detects_modified_archive_evidence(self):
        frame = G.load_frame(self.ordered[2][1])
        score = json.loads(frame.raw["scorer/numerical_score.json"])
        score["prediction_records"][0]["score"] = 0.123456
        raw = dict(frame.raw, **{"scorer/numerical_score.json": json.dumps(score).encode()})
        with self.assertRaises(Exception):     # digest no longer binds labels+records: admission refuses
            G.rescore(dataclasses.replace(frame, score=score, raw=raw))
        score = json.loads(frame.raw["scorer/numerical_score.json"])
        score["measurement"]["loc_vehicle_tp"] += 1
        raw = dict(frame.raw, **{"scorer/numerical_score.json": json.dumps(score).encode()})
        result = G.rescore(dataclasses.replace(frame, score=score, raw=raw), with_evaluator=True)
        self.assertFalse(result["exact"])
        self.assertFalse(result["checks"]["measurement"])

    def test_alternative_prediction_scored_against_same_exact_gt(self):
        frame = G.load_frame(self.ordered[7][1])
        labels = np.frombuffer(frame.raw["scorer/predicted_labels.u8"], np.uint8).reshape(720, 1280)
        same = G.score_prediction(frame, labels, frame.score["prediction_records"])
        self.assertEqual(same["measurement"], frame.score["measurement"])
        self.assertEqual(same["q_perc"], frame.score["q_perc"])
        self.assertEqual(same["prediction_digest"], frame.score["prediction_identity"]["prediction_digest"])
        empty = G.score_prediction(frame, np.zeros_like(labels), [])
        self.assertTrue(empty["evaluate_prepared_measurement_equal"])     # both scorers agree off-archive too
        self.assertEqual(empty["measurement"]["seg_vehicle_pred_pixels"], 0)
        self.assertNotEqual(empty["prediction_digest"], same["prediction_digest"])

    def retained(self, directory, frame_dirs, perturb=None):
        plan = P.build_plan(plan_id="join", planned_frames=2 * len(frame_dirs), target_records=len(frame_dirs),
                            min_free_disk_bytes=0)
        cfg = P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))
        observer = OB.make_observer(cfg)
        for i, frame_dir in enumerate(frame_dirs):
            planned, prepared = F.decision_from_archive(frame_dir, perturb_ego=(perturb == i))
            token = observer.begin(planned)
            observer.observe_input(torch.full(P.INPUT_SHAPE, float(i)), "cuda:0")
            observer.end(token, prepared)
        return cfg, observer.finalize(directory, workers=F.worker_done())

    def test_retained_records_join_real_gt_by_reference_and_admit(self):
        frame_dirs = [self.ordered[2][1], self.ordered[3][1], G.frame_directory(F.VALID_B1000_ARCHIVE, ANCHOR_FRAME)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retention"
            cfg, manifest = self.retained(path, frame_dirs)
            self.assertEqual(manifest["status_counts"], {"RETAINED_COMPLETE": 3})
            refs = G.resolve_gt_references(path, manifest_sha256=manifest["manifest_sha256"],
                plan_sha256=cfg.plan_sha256, archive_root=F.VALID_B1000_ARCHIVE, output_path=Path(tmp) / "GT_REFERENCES.json")
            self.assertEqual(refs["status_counts"], {"RESOLVED": 3}, refs["rows"])
            self.assertTrue(all(r["edge_target_bundle_sha256_consulted"] is False for r in refs["rows"]))
            self.assertFalse(any(name.endswith(".u8") for name in (p.name for p in path.iterdir())))  # GT not copied
            admitted = V.admit_for_paired_replay(path, expected_manifest_sha256=manifest["manifest_sha256"],
                expected_plan_sha256=cfg.plan_sha256, references_path=Path(tmp) / "GT_REFERENCES.json",
                expected_references_sha256=refs["references_sha256"])
            self.assertEqual([a["join_key"]["frame_id"] for a in admitted],
                             [int(d.name.split("_")[1]) for d in frame_dirs])
            self.assertEqual(admitted[2]["action"]["anchor_action_id"], 68)
            # Tampered reference document is refused even if rebound.
            document = json.loads((Path(tmp) / "GT_REFERENCES.json").read_bytes())
            document["rows"][0]["status"] = "RESOLVED"
            document["rows"][0]["joins"]["gt_identity"] = False
            raw = json.dumps(document).encode()
            (Path(tmp) / "GT_REFERENCES_2.json").write_bytes(raw)
            with self.assertRaisesRegex(V.RetentionAdmissionError, "identity join"):
                V.admit_for_paired_replay(path, expected_manifest_sha256=manifest["manifest_sha256"],
                    expected_plan_sha256=cfg.plan_sha256, references_path=Path(tmp) / "GT_REFERENCES_2.json",
                    expected_references_sha256=P.sha256_bytes(raw))

    def test_frame_context_mismatch_fails_join_and_admission(self):
        frame_dirs = [self.ordered[2][1], self.ordered[3][1]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retention"
            cfg, manifest = self.retained(path, frame_dirs, perturb=1)
            refs = G.resolve_gt_references(path, manifest_sha256=manifest["manifest_sha256"],
                plan_sha256=cfg.plan_sha256, archive_root=F.VALID_B1000_ARCHIVE, output_path=Path(tmp) / "R.json")
            self.assertEqual(refs["rows"][1]["status"], "INCOMPLETE_GT_JOIN_IDENTITY_MISMATCH")
            self.assertIn("wire_frame_context_equals_target_frame_context", refs["rows"][1]["detail"])
            with self.assertRaisesRegex(V.RetentionAdmissionError, "GT INCOMPLETE"):
                V.admit_for_paired_replay(path, expected_manifest_sha256=manifest["manifest_sha256"],
                    expected_plan_sha256=cfg.plan_sha256, references_path=Path(tmp) / "R.json",
                    expected_references_sha256=refs["references_sha256"])


if __name__ == "__main__":
    unittest.main()
