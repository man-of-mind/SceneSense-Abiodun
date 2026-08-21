import json
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_route_qualification import (
    MANUAL_REVIEW_SCHEMA,
    ArmedLapDetector,
    ClosedDirectRouteController,
    MonotonicTickPacer,
    OrderedRouteProgress,
    RoutePoint,
    RouteQualificationError,
    _cleanup_audit,
    atomic_write_new_text,
    closed_route_length_m,
    duration_spread_gate,
    flush_collision_tick,
    load_and_validate_config,
    project_to_closed_route,
    run_qualification,
    select_exact_blueprint,
    trial_machine_gates,
    validate_manual_review,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "ue_route_qualification_v1.json"
)


class UERouteQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.resolved = load_and_validate_config(CONFIG_PATH)

    def test_frozen_contract_resolves_exact_route(self):
        self.assertEqual(self.config["trials"]["count"], 3)
        self.assertEqual(self.config["route"]["spawn_index"], 55)
        self.assertEqual(self.config["route"]["target_speed_mps"], 6.0)
        self.assertEqual(self.config["route"]["ego_blueprint"], "vehicle.lincoln.mkz")
        self.assertEqual(self.resolved["route_point_count"], 85)
        self.assertAlmostEqual(self.resolved["open_route_length_m"], 330.775, places=3)
        self.assertAlmostEqual(self.resolved["closing_seam_m"], 7.248, places=3)
        self.assertAlmostEqual(self.resolved["closed_route_length_m"], 338.023, places=3)
        self.assertEqual(len(self.config["manual_review"]["required_checks"]), 8)
        self.assertTrue(self.config["clock"]["real_time_pacing_enabled"])
        self.assertEqual(self.config["clock"]["real_time_tick_period_s"], 0.05)
        self.assertEqual(self.config["clock"]["collision_flush_ticks"], 1)

    def test_blueprint_resolution_is_exact_and_missing_id_is_readable(self):
        class FakeBlueprint:
            def __init__(self, blueprint_id):
                self.id = blueprint_id

        class FakeLibrary:
            def __init__(self):
                self.values = [
                    FakeBlueprint("vehicle.lincoln.mkz"),
                    FakeBlueprint("vehicle.dodge.charger"),
                ]

            def filter(self, pattern):
                if pattern == "vehicle.*":
                    return list(self.values)
                return [item for item in self.values if item.id == pattern]

        library = FakeLibrary()
        selected = select_exact_blueprint(library, "vehicle.lincoln.mkz")
        self.assertEqual(selected.id, "vehicle.lincoln.mkz")
        with self.assertRaisesRegex(
            RouteQualificationError,
            "requested=vehicle.tesla.model3.*vehicle.lincoln.mkz",
        ):
            select_exact_blueprint(library, "vehicle.tesla.model3")

    def test_monotonic_pacer_sleeps_to_deadline_then_catches_up(self):
        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.sleeps = []

            def monotonic(self):
                return self.now

            def sleep(self, duration):
                self.sleeps.append(duration)
                self.now += duration

        clock = FakeClock()
        pacer = MonotonicTickPacer(
            0.05, monotonic=clock.monotonic, sleeper=clock.sleep
        )
        first = pacer.wait()
        self.assertAlmostEqual(first["sleep_s"], 0.05)
        clock.now += 0.08
        late = pacer.wait()
        self.assertEqual(late["sleep_s"], 0.0)
        self.assertAlmostEqual(late["lateness_s"], 0.03)
        caught_up = pacer.wait()
        self.assertAlmostEqual(caught_up["sleep_s"], 0.02)

    def test_collision_flush_keeps_tick_unmeasured_and_drains_delayed_mailbox(self):
        class FakeMailbox:
            def __init__(self):
                self.values = [{"frame_id": 10}]

            def rows(self):
                return [dict(value) for value in self.values]

        class FakeWorld:
            def __init__(self):
                self.tick_count = 0

            def tick(self, timeout):
                self.tick_count += 1
                self.timeout = timeout
                return 11

        mailbox = FakeMailbox()
        world = FakeWorld()
        route_trace_sentinel = [{"frame_id": 10, "lap_count": 1}]

        def delayed_callback(_duration):
            mailbox.values.append({"frame_id": 11})

        result = flush_collision_tick(
            world,
            mailbox,
            tick_timeout_s=2.0,
            settle_s=0.02,
            sleeper=delayed_callback,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["flush_tick_count"], 1)
        self.assertEqual(result["new_collision_rows"], [{"frame_id": 11}])
        self.assertEqual(world.tick_count, 1)
        self.assertEqual(route_trace_sentinel, [{"frame_id": 10, "lap_count": 1}])

    def test_projection_includes_closed_seam(self):
        points = [
            RoutePoint(0.0, 0.0),
            RoutePoint(10.0, 0.0),
            RoutePoint(10.0, 10.0),
            RoutePoint(0.0, 10.0),
        ]
        self.assertAlmostEqual(closed_route_length_m(points), 40.0)
        projection = project_to_closed_route(-1.0, 5.0, points)
        self.assertAlmostEqual(projection.cross_track_m, 1.0)
        self.assertEqual(projection.segment_index, 3)
        self.assertAlmostEqual(projection.progress_fraction, 0.875)

    def test_ordered_progress_proves_one_full_wrap(self):
        points = [
            RoutePoint(10.0, 0.0),
            RoutePoint(10.0, 10.0),
            RoutePoint(0.0, 10.0),
            RoutePoint(0.0, 0.0),
        ]
        progress = OrderedRouteProgress(points, start_index=0)
        for transition in ((0, 1), (1, 2), (2, 3)):
            result = progress.observe(transition, armed_before=False)
            self.assertTrue(result["ordered"])
        result = progress.observe((3, 0), armed_before=True)
        self.assertTrue(result["wrapped"])
        self.assertEqual(progress.wrap_count, 1)
        self.assertEqual(progress.wrap_after_arming_count, 1)
        self.assertEqual(progress.sequence_error_count, 0)
        self.assertAlmostEqual(progress.unwrapped_progress_m, 40.0)

    def test_lap_detector_rejects_startup_and_requires_95_percent_armed_wrap(self):
        detector = ArmedLapDetector(
            100.0,
            {
                "start_gate_x_m": 0.0,
                "start_gate_y_m": 0.0,
                "expected_return_heading_deg": 0.0,
                "gate_exit_radius_m": 10.0,
                "completion_radius_m": 4.0,
                "completion_heading_tolerance_deg": 15.0,
                "minimum_ordered_progress_ratio_to_arm": 0.95,
                "required_wrap_count": 1,
            },
        )
        state = detector.update(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
            unwrapped_progress_m=0.0,
            wrap_count=0,
            wrap_after_arming_count=0,
        )
        self.assertFalse(state["armed"])
        self.assertFalse(state["completed"])
        state = detector.update(
            x_m=20.0,
            y_m=0.0,
            yaw_deg=0.0,
            unwrapped_progress_m=94.9,
            wrap_count=0,
            wrap_after_arming_count=0,
        )
        self.assertFalse(state["armed"])
        state = detector.update(
            x_m=20.0,
            y_m=0.0,
            yaw_deg=0.0,
            unwrapped_progress_m=95.0,
            wrap_count=0,
            wrap_after_arming_count=0,
        )
        self.assertTrue(state["armed"])
        # The frozen route enters the 4 m start gate on its final approach
        # immediately before the discrete final-index -> zero transition.
        # That armed approach is not a false completion and must keep driving.
        state = detector.update(
            x_m=3.8,
            y_m=0.0,
            yaw_deg=2.0,
            unwrapped_progress_m=98.5,
            wrap_count=0,
            wrap_after_arming_count=0,
        )
        self.assertFalse(state["completed"])
        self.assertEqual(state["false_completion_count"], 0)
        state = detector.update(
            x_m=1.0,
            y_m=0.0,
            yaw_deg=2.0,
            unwrapped_progress_m=100.0,
            wrap_count=1,
            wrap_after_arming_count=1,
        )
        self.assertTrue(state["completed"])
        self.assertEqual(state["lap_count"], 1)

    def test_cleanup_ignores_dead_proxy_and_polls_bounded_for_live_actor(self):
        class FakeActor:
            def __init__(self, actor_id, alive):
                self.id = actor_id
                self.type_id = "vehicle.lincoln.mkz"
                self.is_alive = alive

        class FakeWorld:
            def __init__(self):
                self.actor = FakeActor(7, True)
                self.ticks = 0

            def get_actor(self, actor_id):
                return self.actor if actor_id == 7 else None

            def tick(self, timeout):
                self.ticks += 1
                self.actor.is_alive = False
                return self.ticks

        world = FakeWorld()
        result = _cleanup_audit(
            world,
            [7],
            tick_timeout_s=2.0,
            fixed_delta_s=0.05,
            maximum_sim_s=5.0,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["cleanup_ticks"], 1)
        self.assertEqual(result["remaining_owned_actors"], [])

    def test_controller_normalizes_seam_spawn_without_counting_a_wrap(self):
        points = [
            RoutePoint(10.0, 0.0),
            RoutePoint(10.0, 10.0),
            RoutePoint(0.0, 10.0),
            RoutePoint(0.0, 0.0),
        ]
        controller = ClosedDirectRouteController(
            points,
            {
                "waypoint_reach_m": 2.0,
                "lookahead_points": 0,
                "turn_minimum_speed_scale": 0.35,
                "throttle_gain": 0.30,
                "throttle_max": 0.75,
                "brake_gain": 0.35,
                "brake_max": 0.65,
                "steer_full_scale_deg": 45.0,
                "steer_max_abs": 0.70,
            },
        )
        controller.reset(0.0, 0.0)
        self.assertEqual(controller.index, 0)
        command = controller.command(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
            speed_mps=0.0,
            target_speed_mps=6.0,
        )
        self.assertEqual(command.transitions, ())
        self.assertEqual(command.route_index, 0)
        self.assertGreater(command.throttle, 0.0)

    def _passing_trial_summary(self):
        return {
            "spawn_pose_audit": {"pass": True},
            "lap_count": 1,
            "wrap_count": 1,
            "wrap_after_arming_count": 1,
            "sequence_error_count": 0,
            "unwrapped_progress_ratio": 1.0,
            "false_completion_count": 0,
            "return_position_error_m": 1.0,
            "return_heading_error_deg": 2.0,
            "cross_track_p95_m": 1.0,
            "maximum_continuous_divergence_s": 0.45,
            "maximum_cross_track_m": 2.5,
            "collision_count": 0,
            "collision_flush": {"pass": True, "flush_tick_count": 1},
            "maximum_continuous_stall_s": 5.0,
            "duration_s": 60.0,
            "observed_sim_control_rate_hz": 20.0,
            "maximum_sim_delta_error_s": 0.0,
            "nonmonotonic_frame_count": 0,
            "maximum_tick_displacement_m": 0.5,
            "cleanup": {"pass": True},
            "runtime_error": None,
        }

    def test_p95_persistence_absolute_and_frame_gates_fail_closed(self):
        summary = self._passing_trial_summary()
        self.assertTrue(all(trial_machine_gates(summary, self.config).values()))
        mutations = {
            "cross_track_p95_m": 1.51,
            "maximum_continuous_divergence_s": 0.5,
            "maximum_cross_track_m": 3.01,
            "nonmonotonic_frame_count": 1,
            "unwrapped_progress_ratio": 0.949,
            "wrap_count": 2,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                candidate = dict(summary)
                candidate[field] = value
                self.assertFalse(all(trial_machine_gates(candidate, self.config).values()))

    def test_three_trial_duration_spread_is_five_percent_of_median(self):
        self.assertTrue(duration_spread_gate([60.0, 61.0, 62.0], 0.05)["pass"])
        self.assertFalse(duration_spread_gate([60.0, 61.0, 64.0], 0.05)["pass"])
        self.assertFalse(duration_spread_gate([60.0, 61.0], 0.05)["pass"])

    def test_manual_review_requires_exactly_eight_boolean_checks(self):
        checks = {
            name: True for name in self.config["manual_review"]["required_checks"]
        }
        review = {
            "schema": MANUAL_REVIEW_SCHEMA,
            "reviewer": "Abiodun",
            "reviewed_at": "2026-08-20T12:00:00+00:00",
            "overall_verdict": "PASS",
            "trials": {
                f"trial_{index:02d}": {
                    "verdict": "PASS",
                    "checks": dict(checks),
                    "anomalies": [],
                    "notes": "",
                }
                for index in range(1, 4)
            },
        }
        validated = validate_manual_review(review, self.config)
        self.assertEqual(validated["overall_verdict"], "PASS")
        del review["trials"]["trial_01"]["checks"][next(iter(checks))]
        with self.assertRaises(RouteQualificationError):
            validate_manual_review(review, self.config)

    def test_preflight_failure_writes_failed_terminal_and_full_artifact_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_config = root / "bad.json"
            bad_config.write_text('{"schema":"wrong"}\n', encoding="utf-8")
            run_dir, status = run_qualification(bad_config, root / "outputs")
            self.assertEqual(status, "FAILED")
            expected = {
                "resolved_config.yaml",
                "manifest.json",
                "route_contract.json",
                "route_trace.csv",
                "route_events.csv",
                "ROUTE_MACHINE_REVIEW.json",
                "manual_review_template.json",
                "FAILED.json",
            }
            self.assertEqual({path.name for path in run_dir.iterdir()}, expected)
            self.assertFalse((run_dir / "REVIEW_REQUIRED.json").exists())
            self.assertFalse((run_dir / "COMPLETED.json").exists())
            machine = json.loads(
                (run_dir / "ROUTE_MACHINE_REVIEW.json").read_text(encoding="utf-8")
            )
            self.assertEqual(machine["status"], "FAILED")
            self.assertIsNotNone(machine["orchestration_error"])

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            atomic_write_new_text(path, "first\n")
            with self.assertRaises(RouteQualificationError):
                atomic_write_new_text(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")


if __name__ == "__main__":
    unittest.main()
