import json
from pathlib import Path
import tempfile
import unittest

from . import gt_timing, postflight_checks, runtime


class TimingTests(unittest.TestCase):
    def test_full_stage_population_and_no_io_before_flush(self):
        log = gt_timing.StageLog(1)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "timing"
            for stage in postflight_checks.STAGES:
                marker = object()
                fn = log.wrap(stage, lambda **kw: marker)
                self.assertIs(fn(identity={"frame_id": 7}), marker)
            self.assertFalse(path.exists())
            log.flush(path, worker_drained=True)
            self.assertEqual(postflight_checks.verify_gt_timing(path, expected_frames=[7])["stages"], 5)
            with self.assertRaises(ValueError):
                postflight_checks.verify_gt_timing(path, expected_frames=[8])

    def test_original_exception_and_bounded_measurements(self):
        log = gt_timing.StageLog(1)
        exc = RuntimeError("the original failure")
        def fails(**kw):
            raise exc
        with self.assertRaises(RuntimeError) as captured:
            log.wrap("complete_source_build", fails)(identity={"frame_id": 1})
        self.assertIs(captured.exception, exc)
        self.assertFalse(log.rows[0]["completed"])
        for _ in range(20):
            log.wrap("stage", lambda: None)()
        self.assertEqual(len(log.rows), 8)
        self.assertEqual(log.dropped, 13)

    def test_corruption_cannot_pass(self):
        log = gt_timing.StageLog(1)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "timing"
            log.flush(path, worker_drained=True)
            (path / "stages.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                postflight_checks.verify_gt_timing(path, expected_frames=[])

    def test_live_worker_cannot_claim_drain(self):
        log = gt_timing.StageLog(1)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "timing"
            for stage in postflight_checks.STAGES:
                log.wrap(stage, lambda **kw: None)(identity={"frame_id": 3})
            result = log.flush(path, worker_drained=False)
            self.assertFalse(result["flush_after_worker_drain"])
            with self.assertRaises(ValueError):
                postflight_checks.verify_gt_timing(path, expected_frames=[3])


class ScopeTests(unittest.TestCase):
    def test_archive_scope_restores_even_on_error(self):
        from rl_agent.splitfusion_live_parity_repair_v1 import runtime as old
        from .prediction_archive import PredictionArchive
        before = old.PredictionArchive
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with runtime.install_prediction_contract():
                self.assertIs(old.PredictionArchive, PredictionArchive)
                raise RuntimeError("stop")
        self.assertIs(old.PredictionArchive, before)

    def test_postflight_scope_preserves_old_function(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as old
        before = old.verify_cell
        with runtime.install_postflight_reader():
            self.assertIsNot(old.verify_cell, before)
        self.assertIs(old.verify_cell, before)


if __name__ == "__main__":
    unittest.main()
