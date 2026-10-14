"""Focused offline qualification of the M3.4 route-triggered schedule."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import contract as C
from .bad_to_good_controller import RouteTriggeredSequence, main as controller_main, rewrite_profile
from .route_progress import OrderedProgress, Publisher, load_geometry


ROOT = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self, value=1_000_000_000):
        self.value = value
    def __call__(self):
        return self.value


class TransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = C.channel_registration(ROOT)
        cls.schedule = cls.registration["schedule"]
        cls.route = cls.schedule["route_transition"]

    def sequence(self, directory, clock=None):
        clock = clock or Clock()
        progress = Path(directory) / "progress.json"
        trace = Path(directory) / "trace.jsonl"
        return RouteTriggeredSequence(self.schedule, progress, trace,
                                      monotonic_ns=clock, raw_ns=clock), progress, clock

    def sample(self, cumulative, *, frame=10, source_ns=1_000_000_000, status="ACTIVE"):
        return dict(schema=self.route["schema"], status=status, source=self.route["source"],
                    source_frame_id=frame, source_monotonic_ns=source_ns,
                    source_monotonic_raw_ns=source_ns, source_wall_ns=source_ns,
                    route_length_m=self.route["route_length_m"],
                    threshold_distance_m=self.route["threshold_distance_m"],
                    cumulative_distance_m=cumulative)

    def test_geometry_is_actual_planned_path_and_exact_half(self):
        geometry = load_geometry(ROOT)
        self.assertEqual(geometry["route_json_sha256"], C.ROUTE_JSON_SHA256)
        self.assertGreater(len(geometry["points"]), 500)
        self.assertTrue(math.isclose(geometry["threshold_distance_m"],
                                     geometry["route_length_m"] * 0.5, abs_tol=1e-12))

    def test_startup_without_progress_is_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence, _, _ = self.sequence(tmp)
            self.assertEqual(sequence.next_sample(), (0, 8.5))
            sequence.close()

    def test_just_below_at_and_above_half(self):
        for offset, expected in ((-1e-6, (0, 8.5)), (0.0, (2, 23.5)), (1e-6, (2, 23.5))):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as tmp:
                sequence, progress, _ = self.sequence(tmp)
                progress.write_text(json.dumps(self.sample(self.route["threshold_distance_m"] + offset)))
                self.assertEqual(sequence.next_sample(), expected)
                sequence.close()

    def test_repeats_backtracking_and_loop_closure_never_restore_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence, progress, _ = self.sequence(tmp)
            values = [self.route["threshold_distance_m"] - 1.0,
                      self.route["threshold_distance_m"],
                      self.route["threshold_distance_m"] + 20.0,
                      10.0, 0.0]
            observed = []
            for frame, value in enumerate(values, 1):
                progress.write_text(json.dumps(self.sample(value, frame=frame)))
                observed.append(sequence.next_sample()[1])
            self.assertEqual(observed, [8.5, 23.5, 23.5, 23.5, 23.5])
            self.assertEqual(sum(json.loads(line).get("threshold_crossed_now") is True
                                 for line in sequence.trace_path.read_text().splitlines()), 1)
            sequence.close()

    def test_stale_progress_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock(2_100_000_000)
            sequence, progress, _ = self.sequence(tmp, clock)
            progress.write_text(json.dumps(self.sample(1.0, source_ns=1_000_000_000)))
            with self.assertRaisesRegex(RuntimeError, "stale"):
                sequence.next_sample()
            sequence.close()

    def test_missing_after_active_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence, progress, _ = self.sequence(tmp)
            progress.write_text(json.dumps(self.sample(1.0)))
            self.assertEqual(sequence.next_sample(), (0, 8.5))
            progress.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                sequence.next_sample()
            sequence.close()

    def test_failed_or_mismatched_progress_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence, progress, _ = self.sequence(tmp)
            bad = self.sample(1.0)
            bad["route_length_m"] += 1.0
            progress.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                sequence.next_sample()
            sequence.close()

    def test_actor_contract_contains_no_route_or_phase_feature(self):
        actor = self.registration["actor_input"]
        self.assertEqual(actor["feature_index"], 22)
        self.assertFalse(actor["route_progress_to_actor"])
        self.assertFalse(actor["phase_identity_to_actor"])
        self.assertFalse(actor["future_settings_to_actor"])
        self.assertEqual(actor["values"], [8.5, 23.5])

    def test_profile_rewrite_is_exact(self):
        self.assertEqual(rewrite_profile(["--profile-id", "FAVORABLE_STABLE"]),
                         ["--profile-id", "BAD_TO_GOOD"])
        with self.assertRaises(ValueError):
            rewrite_profile(["--profile-id", "BAD_ALL"])

    def test_command_failure_propagates_and_restores_sequence_seam(self):
        from rl_agent import ue_target_snr_cell_runtime_v1 as base
        from rl_agent.splitfusion_run5_live_avo_v1 import snr_live
        original = lambda **_kwargs: (object(), [], {})
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "run5_policy"
            policy.mkdir()
            argv = ["--controller", "--snr-socket", str(Path(tmp) / "s.sock"),
                    "--snr-controller-id", "00000000-0000-0000-0000-000000000001",
                    "--snr-audit", str(policy / "CONTROLLER_ACKS.jsonl"),
                    "--profile-id", "FAVORABLE_STABLE"]

            def failed_command(_argv):
                sequence, prefix, _frozen = base.prepare_sequence(
                    campaign={"network": {"prefix_samples": 4200}}, profile_id="BAD_TO_GOOD")
                self.assertEqual(prefix, [])
                self.assertEqual(sequence.next_sample(), (0, 8.5))
                raise RuntimeError("simulated command failure")

            with mock.patch.object(base, "prepare_sequence", original), \
                    mock.patch.object(snr_live, "controller_main", failed_command):
                with self.assertRaisesRegex(RuntimeError, "simulated command failure"):
                    controller_main(argv)
                self.assertIs(base.prepare_sequence, original)


class ProgressTrackerTests(unittest.TestCase):
    def test_actual_summary_writer_matches_threshold_crossed_checker_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / "run5_policy").mkdir()
            publisher = Publisher(attempt, ROOT)
            publisher.crossing = {
                "source_frame_id": 1, "source_monotonic_ns": 1,
                "source_monotonic_raw_ns": 1, "source_wall_ns": 1,
                "cumulative_distance_m": publisher.geometry["threshold_distance_m"],
                "threshold_distance_m": publisher.geometry["threshold_distance_m"],
            }
            publisher.close("ROUTE_COMPLETED")
            summary = json.loads(publisher.summary_path.read_bytes())
            self.assertIs(summary["threshold_crossed"], True)
            self.assertIsNotNone(summary["crossing"])

    def test_backtracking_and_start_return_are_monotone(self):
        geometry = load_geometry(ROOT)
        tracker = OrderedProgress(geometry)
        points = geometry["points"]
        observed = []
        for index in list(range(0, 80)) + list(range(79, 60, -1)) + list(range(80, len(points), 8)):
            row = tracker.observe(*points[index])
            observed.append(row["cumulative_distance_m"])
        self.assertEqual(observed, sorted(observed))
        before = observed[-1]
        returned = tracker.observe(*points[0])["cumulative_distance_m"]
        self.assertGreaterEqual(returned, before)


if __name__ == "__main__":
    unittest.main()
