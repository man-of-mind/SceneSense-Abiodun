import argparse
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from data_collection import carla_fusion_policy_corpus_collector as collector
from data_collection.analyze_detection_ab_gate import (
    _longest_true_dwell,
    _paired_block_bootstrap_lift,
    _target_actor_id,
    _trajectory_comparison,
)
from data_collection.carla_fusion_policy_corpus_collector import (
    ControlledPedestrianOverlay,
    _compose_camera_world_matrix,
    _apply_direct_ego_route_control,
    _parse_overlay_args,
    _pedestrian_crowd_offsets,
    spawn_parked_ego_with_tm_overrides,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FakeTransform:
    def __init__(self, matrix):
        self._matrix = matrix

    def get_matrix(self):
        return self._matrix


class DetectionABGateTests(unittest.TestCase):
    def test_longest_true_dwell_respects_gaps(self):
        dwell = _longest_true_dwell(
            [True, True, True, True, False, True, True],
            [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3],
        )
        self.assertAlmostEqual(dwell, 0.3)

    def test_matched_trajectory_passes_small_deltas(self):
        baseline = pd.DataFrame(
            {
                "in_scope": [True, True, False],
                "distance_m": [20.0, 20.1, 26.0],
                "projected_x": [400.0, 401.0, 900.0],
                "projected_y": [200.0, 201.0, 200.0],
            }
        )
        candidate = baseline.copy()
        candidate["distance_m"] += 0.1
        candidate["projected_x"] += 1.0
        result = _trajectory_comparison(
            baseline,
            candidate,
            {
                "maximum_target_row_delta_fraction": 0.1,
                "minimum_in_scope_sequence_agreement": 0.9,
                "maximum_median_distance_delta_m": 0.75,
                "maximum_median_projection_delta_px": 10.0,
            },
        )
        self.assertTrue(result["pair_valid"])

    def test_matched_trajectory_rejects_scope_mismatch(self):
        baseline = pd.DataFrame(
            {
                "in_scope": [True] * 10,
                "distance_m": [20.0] * 10,
                "projected_x": [400.0] * 10,
                "projected_y": [200.0] * 10,
            }
        )
        candidate = baseline.copy()
        candidate["in_scope"] = [False] * 10
        result = _trajectory_comparison(
            baseline,
            candidate,
            {
                "maximum_target_row_delta_fraction": 0.1,
                "minimum_in_scope_sequence_agreement": 0.9,
                "maximum_median_distance_delta_m": 0.75,
                "maximum_median_projection_delta_px": 10.0,
            },
        )
        self.assertFalse(result["pair_valid"])

    def test_paired_bootstrap_reports_positive_lift(self):
        baseline = [False] * 80
        candidate = [False] * 40 + [True] * 40
        result = _paired_block_bootstrap_lift(
            baseline,
            candidate,
            replicates=2000,
            block_length=5,
            seed=7,
        )
        self.assertEqual(result["paired_rows"], 80)
        self.assertAlmostEqual(result["lift_pp"], 50.0)
        self.assertGreater(result["ci95_lower_pp"], 0.0)

    def test_paired_bootstrap_rejects_unpaired_arrays(self):
        with self.assertRaises(ValueError):
            _paired_block_bootstrap_lift(
                [False, True],
                [True],
                replicates=100,
                block_length=2,
                seed=7,
            )

    def test_manifest_controlled_pedestrian_wins_over_crowd_row_counts(self):
        gt = pd.DataFrame(
            {
                "actor_id": [91, 91, 91, 42, 42],
                "class_name": ["pedestrian"] * 5,
            }
        )
        actor_id = _target_actor_id(
            "pedestrian", {"controlled_target": {"actor_id": 42}}, gt
        )
        self.assertEqual(actor_id, 42)

    def test_camera_relative_transform_is_composed_into_anchor_world_pose(self):
        anchor = np.asarray(
            [
                [0.0, -1.0, 0.0, 100.0],
                [1.0, 0.0, 0.0, 50.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        relative_camera = np.asarray(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        composed = _compose_camera_world_matrix(
            FakeTransform(anchor), FakeTransform(relative_camera)
        )
        np.testing.assert_allclose(composed[:3, 3], [100.0, 52.0, 1.5])
        np.testing.assert_allclose(composed[:3, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(composed[:3, 1], [-1.0, 0.0, 0.0])

    def test_crowd_slots_are_close_in_frustum_and_numerous(self):
        overlay = ControlledPedestrianOverlay(
            crowd_count=96,
            crowd_min_spawned=81,
            crowd_depth_min_m=16.0,
            crowd_depth_max_m=22.0,
            crowd_depth_step_m=2.0,
            crowd_lateral_spacing_m=0.85,
            horizontal_fov_deg=100.0,
            headline_range_m=25.0,
        )
        offsets = _pedestrian_crowd_offsets(overlay)
        self.assertGreaterEqual(len(offsets), overlay.crowd_count)
        for forward_m, lateral_m in offsets:
            self.assertLess(math.hypot(forward_m, lateral_m), overlay.headline_range_m)
            self.assertLess(
                abs(math.degrees(math.atan2(lateral_m, forward_m))),
                0.5 * overlay.horizontal_fov_deg,
            )

    def test_overlay_parser_strips_only_track_b_flags(self):
        overlay, remaining = _parse_overlay_args(
            [
                "--fps",
                "10",
                "--controlled-pedestrian-crowd-count",
                "96",
                "--controlled-pedestrian-crowd-min-spawned",
                "81",
                "--ego-ignore-walkers-pct",
                "100",
                "--max-frames",
                "250",
            ]
        )
        self.assertEqual(overlay.crowd_count, 96)
        self.assertEqual(overlay.crowd_min_spawned, 81)
        self.assertEqual(overlay.ego_ignore_walkers_pct, 100.0)
        self.assertEqual(remaining, ["--fps", "10", "--max-frames", "250"])

    def test_ego_walker_ignore_override_uses_configured_tm(self):
        actor = mock.Mock(id=42)
        args = argparse.Namespace(host="127.0.0.1", port=2000, tm_port=8010)
        traffic_manager = mock.Mock()
        client = mock.Mock()
        client.get_trafficmanager.return_value = traffic_manager
        overlay = ControlledPedestrianOverlay(ego_ignore_walkers_pct=100.0)
        with mock.patch.object(collector, "_PEDESTRIAN_OVERLAY", overlay), \
             mock.patch.object(collector, "_SPAWN_PARKED_EGO", return_value=actor), \
             mock.patch.object(collector.base.carla, "Client", return_value=client):
            returned = spawn_parked_ego_with_tm_overrides(world=mock.Mock(), args=args)
        self.assertIs(returned, actor)
        client.get_trafficmanager.assert_called_once_with(8010)
        traffic_manager.ignore_walkers_percentage.assert_called_once_with(actor, 100.0)

    def test_ego_walker_ignore_override_destroys_actor_on_failure(self):
        actor = mock.Mock(id=42)
        args = argparse.Namespace(host="127.0.0.1", port=2000, tm_port=8010)
        client = mock.Mock()
        client.get_trafficmanager.side_effect = RuntimeError("TM unavailable")
        overlay = ControlledPedestrianOverlay(ego_ignore_walkers_pct=100.0)
        with mock.patch.object(collector, "_PEDESTRIAN_OVERLAY", overlay), \
             mock.patch.object(collector, "_SPAWN_PARKED_EGO", return_value=actor), \
             mock.patch.object(collector.base.carla, "Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "walker-ignore override"):
                spawn_parked_ego_with_tm_overrides(world=mock.Mock(), args=args)
        actor.destroy.assert_called_once_with()

    def test_direct_route_controller_disables_tm_and_commands_vehicle(self):
        actor = mock.Mock(id=77)
        actor.get_location.return_value = mock.Mock(x=0.0, y=0.0)
        actor.get_transform.return_value = mock.Mock(
            location=mock.Mock(x=0.0, y=0.0),
            rotation=mock.Mock(yaw=0.0),
        )
        actor.get_velocity.return_value = mock.Mock(x=0.0, y=0.0, z=0.0)
        args = argparse.Namespace(tm_port=8010)
        overlay = ControlledPedestrianOverlay(
            ego_route_control="direct", ego_direct_route_speed_mps=6.0
        )
        with mock.patch.object(collector, "_PEDESTRIAN_OVERLAY", overlay), \
             mock.patch.object(collector, "_DIRECT_ROUTE_STATE", {}), \
             mock.patch.object(
                 collector, "_load_direct_route", return_value=[(5.0, 0.0), (10.0, 0.0)]
             ):
            index, heading_error, yielding = _apply_direct_ego_route_control(actor, args)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(heading_error, 0.0)
        self.assertFalse(yielding)
        actor.set_autopilot.assert_called_once_with(False, 8010)
        control = actor.apply_control.call_args.args[0]
        self.assertGreater(float(control.throttle), 0.0)
        self.assertAlmostEqual(float(control.steer), 0.0)

    def test_overlay_cleanup_releases_all_owned_crowd_actors(self):
        actors = [object(), object(), object()]
        collector._OVERLAY_ACTORS = list(actors)
        with mock.patch.object(
            collector.base.pole_client, "_destroy_actors"
        ) as destroy_actors:
            collector._destroy_overlay_actors()
        destroy_actors.assert_called_once_with(actors)
        self.assertEqual(collector._OVERLAY_ACTORS, [])

    def test_v2_has_250_frames_crowd_and_unchanged_hard_gates(self):
        v1 = yaml.safe_load((CONFIG_DIR / "detection_ab_gate_v1.yaml").read_text())
        v2 = yaml.safe_load((CONFIG_DIR / "detection_ab_gate_v2.yaml").read_text())
        self.assertEqual(v2["requested_frames"], 250)
        self.assertEqual(v2["detection_ab_gate"], v1["detection_ab_gate"])
        self.assertEqual(len(v2["smoke_runs"]), 6)
        self.assertTrue(
            all(int(run["requested_frames"]) == 250 for run in v2["smoke_runs"])
        )
        max_frames_index = v2["common_args"].index("--max-frames")
        self.assertEqual(int(v2["common_args"][max_frames_index + 1]), 250)
        pedestrians = [
            run for run in v2["smoke_runs"] if run["target_class"] == "pedestrian"
        ]
        self.assertEqual(len(pedestrians), 3)
        for run in pedestrians:
            args = run["extra_args"]
            count_index = args.index("--controlled-pedestrian-crowd-count")
            minimum_index = args.index("--controlled-pedestrian-crowd-min-spawned")
            self.assertEqual(int(args[count_index + 1]), 96)
            self.assertEqual(int(args[minimum_index + 1]), 81)


if __name__ == "__main__":
    unittest.main()
