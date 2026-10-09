"""CPU tests of the three-branch replay admission on REAL gc_v2 numeric-GT frames with stub model callbacks.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v \
      rl_agent.splitfusion_run5_quality_reference_v1.test_replay
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_paired_quality_replay_v1.replay import Prediction
from rl_agent.splitfusion_paired_retention_v2 import fixtures as F, gt_adapter as G, observer as OB, plan as P

from . import replay as R

NEU = Path(__file__).resolve().parents[3]
GC_V2_ARCHIVE = (NEU / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
                 "/20261006_run5_full_live_gc_v2/validation/capture/cells/a71__favorable_stable/numeric_gt/service/archive")


class StubAdapter:
    """front -> zeros; encode -> the saved inner bytes; decode -> zeros; tail -> the archived native prediction."""

    def __init__(self, archive, corrupt_frame=None):
        self.archive, self.corrupt = archive, corrupt_frame

    def replay_callbacks(self):
        return dict(front=lambda x: torch.zeros(1, 256, 112, 192), encode_selected=lambda c2, e: e.inner_payload,
                    decode_selected=lambda b, e: torch.zeros(256, 112, 192), tail=self.tail)

    def tail(self, c2, metadata):
        frame = G.load_frame(G.frame_directory(self.archive, metadata.frame_context.frame_id))
        labels = np.frombuffer(frame.raw["scorer/predicted_labels.u8"], np.uint8).reshape(720, 1280).copy()
        if metadata.frame_context.frame_id == self.corrupt:
            labels[0, 0] = 2 if labels[0, 0] != 2 else 1
        return Prediction(labels, tuple(frame.score["prediction_records"]))


def stub_baseline(adapter, dense_c2):
    return torch.zeros(256, 112, 192), b"stub-baseline", "f" * 64


@unittest.skipUnless(GC_V2_ARCHIVE.is_dir(), "gc_v2 numeric-GT archive absent")
class ReplayAdmissionTests(unittest.TestCase):
    def build(self, tmp, n=4, missing=False):
        ordered = F.archive_frames_by_tensor(GC_V2_ARCHIVE)[:n]
        plan = P.build_plan(plan_id="replay-test", planned_frames=2 * n, target_records=n, require_cuda_transfer=False)
        cfg = P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))
        observer = OB.make_observer(cfg)
        tensor = torch.zeros(1, 7, 448, 768)
        for index, (_, frame_dir) in enumerate(ordered):
            planned, prepared = F.decision_from_archive(frame_dir)
            token = observer.begin(planned)
            observer.observe_input(tensor, "cpu")
            observer.end(token, None if (missing and index == 1) else prepared)
        path = Path(tmp) / "retention"
        manifest = observer.finalize(path, workers=F.worker_done())
        refs = G.resolve_gt_references(path, manifest_sha256=manifest["manifest_sha256"], plan_sha256=cfg.plan_sha256,
                                       archive_root=GC_V2_ARCHIVE, output_path=Path(tmp) / "GT_REFERENCES.json")
        return path, manifest, cfg, refs, ordered

    def test_admitted_rows_replay_and_score_three_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest, cfg, refs, ordered = self.build(tmp)
            _, refs_doc, rows = R.admitted_rows(path, manifest_sha256=manifest["manifest_sha256"],
                plan_sha256=cfg.plan_sha256, references_path=Path(tmp) / "GT_REFERENCES.json",
                references_sha256=refs["references_sha256"])
            corrupt = int(ordered[2][1].name.split("_")[1])
            R.replay_rows(StubAdapter(GC_V2_ARCHIVE, corrupt_frame=corrupt), rows, refs_doc, path, baseline=stub_baseline)
            admissions = [r["admission"] for r in rows]
            self.assertEqual(admissions, ["ADMITTED", "ADMITTED", "MISSING_REPLAY_PARITY_OR_FAULT", "ADMITTED"])
            self.assertIn("parity failed", rows[2]["fault"])
            self.assertNotIn("full_local", rows[2])
            for row in (rows[0], rows[1], rows[3]):
                frame = G.load_frame(GC_V2_ARCHIVE / row["frame_dir"])
                for branch in R.BRANCHES:
                    self.assertEqual(R.canonical(row[branch]["measurement"]), R.canonical(frame.score["measurement"]))
                self.assertTrue(row["parity"]["archived_measurement_identical"])
                self.assertEqual(row["baseline"]["branch"], "NOAE_UINT8_Q0")
                self.assertNotEqual(row["baseline"]["execution_bundle_sha256"], row["action"]["execution_bundle_sha256"])

    def test_missing_retention_and_modified_gt_are_reported_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest, cfg, refs, ordered = self.build(tmp, n=3, missing=True)
            _, refs_doc, rows = R.admitted_rows(path, manifest_sha256=manifest["manifest_sha256"],
                plan_sha256=cfg.plan_sha256, references_path=Path(tmp) / "GT_REFERENCES.json",
                references_sha256=refs["references_sha256"])
            self.assertEqual(rows[1]["admission"], "MISSING_INCOMPLETE_PRODUCTION_EXCEPTION")
            rows[0]["reference"]["files"]["scorer/segmentation.u8"] = "0" * 64      # referenced GT changed on disk
            R.replay_rows(StubAdapter(GC_V2_ARCHIVE), rows, refs_doc, path, baseline=stub_baseline)
            self.assertEqual([r["admission"] for r in rows],
                             ["MISSING_REPLAY_PARITY_OR_FAULT", "MISSING_INCOMPLETE_PRODUCTION_EXCEPTION", "ADMITTED"])
            self.assertIn("referenced GT file modified", rows[0]["fault"])
            with self.assertRaises(Exception):
                R.admitted_rows(path, manifest_sha256=manifest["manifest_sha256"], plan_sha256=cfg.plan_sha256,
                                references_path=Path(tmp) / "GT_REFERENCES.json", references_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
