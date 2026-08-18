from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from data_collection.phase2_causal_runtime import (
    Phase2CaptureRuntime,
    Phase2RuntimeConfig,
    SourceLocalCausalTracker,
)
from data_collection.run_phase2_paired_causal_pilot import (
    _load_config as load_integration_config,
    _require_udp_ports_available,
    _wait_for_frame,
    _wait_for_tick_ready,
    build_plan,
)
from data_collection.phase2_paired_causal_collector import (
    _require_inherited_contract,
)
from data_collection.launch_phase2_paired_causal_pilot import build_launch_spec
from phase2_map_sharing.replay_paired_pilot import (
    _contribution,
    _recipient_state,
    _select_target_chain_warning,
    _validated_output_name as validated_replay_output_name,
    _warning_diagnostics,
)
from phase2_map_sharing.verify_paired_pilot import (
    _audit_record,
    _validated_output_name as validated_verification_output_name,
)
from uplink_only_spatial_map_pipeline import (
    carla_fusion_staleness_scenario_uplink_only as collector,
)


class _Settings:
    synchronous_mode = True
    fixed_delta_seconds = 0.1


class _Timestamp:
    elapsed_seconds = 12.3


class _Snapshot:
    frame = 100
    timestamp = _Timestamp()


class _World:
    def get_settings(self):
        return _Settings()

    def get_snapshot(self):
        return _Snapshot()


class _Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class _Rotation:
    yaw = 15.0


class _Transform:
    location = _Vector(1.0, 2.0, 3.0)
    rotation = _Rotation()


class _Actor:
    def get_transform(self):
        return _Transform()

    def get_velocity(self):
        return _Vector(4.0, 0.0, 0.0)


def _clock_args(**overrides):
    values = {
        "external_sync_ticker": True,
        "sync_world": False,
        "capture_pipeline": False,
        "npc_vehicles": 0,
        "npc_pedestrians": 0,
        "controlled_target": "none",
        "tracked_lead": "none",
        "experiment3_target_profile": "none",
        "fps": 10.0,
        "world_tick_hz": 10.0,
        "sensor_every_tick": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ExternalTickerContractTests(unittest.TestCase):
    def test_valid_passive_clock_contract(self) -> None:
        args = _clock_args()
        collector.validate_clock_ownership_args(args)
        collector.require_external_sync_contract(_World(), args)

    def test_passive_mode_rejects_any_internal_ticker_path(self) -> None:
        for override in (
            {"sync_world": True},
            {"capture_pipeline": True},
            {"npc_vehicles": 1},
            {"npc_pedestrians": 1},
            {"controlled_target": "vehicle"},
            {"tracked_lead": "vehicle"},
            {"experiment3_target_profile": "centered"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                collector.validate_clock_ownership_args(_clock_args(**override))

    def test_checked_in_plan_has_two_exact_spawns_and_unique_udp_stacks(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "data_collection/configs/phase2_paired_causal_pilot_integration_v1.yaml"
        )
        config, source, contract = load_integration_config(config_path)
        self.assertFalse(contract["live_run_authorized"])
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(config, source, Path(temporary))
        self.assertEqual(plan["population_mode"], "frozen_curbside_pilot_no_ambient")
        self.assertFalse(plan["inference_timing_citable"])
        self.assertTrue(
            all(not item["population_commands"] for item in plan["trajectories"])
        )
        commands = plan["trajectories"][0]["collector_commands"]
        ports = []
        for role, command in commands.items():
            self.assertEqual(
                command[1:3],
                ["-m", "data_collection.phase2_paired_causal_collector"],
            )
            self.assertIn("--external-sync-ticker", command)
            self.assertIn("--ego-spawn-require-exact", command)
            self.assertIn("--ego-freeze", command)
            self.assertNotIn("--no-ego-freeze", command)
            self.assertIn("external_orchestrator", command)
            self.assertIn("--phase2-tick-ready", command)
            self.assertEqual(command[command.index("--max-frames") + 1], "120")
            self.assertEqual(command.count("--sensor-platform"), 1)
            self.assertIn(f"scenesense_phase2_{role}", command)
            for flag in (
                "--camera-source-port",
                "--remote-port",
                "--remote-source-port",
                "--camera-result-port",
            ):
                ports.append(int(command[command.index(flag) + 1]))
        self.assertEqual(len(ports), len(set(ports)))

    def test_phase2_collector_rejects_unfrozen_or_off_contract_capture(self) -> None:
        required = [
            "--role", "loopback", "--async-world", "--external-sync-ticker",
            "--sensor-platform", "ego_vehicle", "--ego-spawn-require-exact",
            "--ego-freeze", "--npc-vehicles", "0", "--npc-pedestrians", "0",
            "--fps", "10.0", "--world-tick-hz", "10.0",
            "--camera-width", "1280", "--camera-height", "720",
            "--camera-fov", "120.0", "--radar-points-per-second", "200000",
            "--radar-hfov", "120", "--radar-rasterizer", "legacy",
            "--radar-raster-radius-px", "4",
            "--radar-temporal-window-frames", "2",
            "--object-score-threshold", "0.05", "--object-nms-radius-px", "2",
            "--topk-objects", "120", "--quantization-mode", "per_channel_uint8",
            "--entropy-coder", "zlib", "--sensor-every-tick",
            "--no-spatial-map-stream", "--disable-semantic-gt",
            "--enable-run-logging", "--headless",
        ]
        _require_inherited_contract(required)
        with self.assertRaisesRegex(ValueError, "ego-freeze"):
            _require_inherited_contract(
                [item for item in required if item != "--ego-freeze"]
            )
        drifted = list(required)
        drifted[drifted.index("--object-score-threshold") + 1] = "0.20"
        with self.assertRaisesRegex(ValueError, "object-score-threshold"):
            _require_inherited_contract(drifted)

    def test_reviewed_plan_authorizes_only_frozen_two_trajectory_pilot(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml"
        )
        config, source, contract = load_integration_config(config_path)
        self.assertTrue(contract["live_run_authorized"])
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(config, source, Path(temporary))
        self.assertTrue(plan["live_authorized"])
        self.assertEqual(len(plan["trajectories"]), 2)
        self.assertTrue(
            all(not item["population_commands"] for item in plan["trajectories"])
        )
        for trajectory in plan["trajectories"]:
            for command in trajectory["collector_commands"].values():
                self.assertEqual(
                    command[1:3],
                    ["-m", "data_collection.phase2_paired_causal_collector"],
                )
                contract_path = command[command.index("--phase2-contract-config") + 1]
                self.assertTrue(contract_path.endswith("paired_causal_pilot_reviewed_v1.yaml"))

    def test_prelaunch_rejects_an_occupied_udp_port(self) -> None:
        port = 53001
        config = {
            "capture": {
                "ports": {
                    "helper": {"one": port},
                    "recipient": {"one": port},
                }
            }
        }
        candidate = mock.MagicMock()
        candidate.bind.side_effect = OSError("address already in use")
        with mock.patch(
            "data_collection.run_phase2_paired_causal_pilot.socket.socket",
            return_value=candidate,
        ):
            with self.assertRaisesRegex(RuntimeError, str(port)):
                _require_udp_ports_available(config)
        candidate.close.assert_called()

    def test_frozen_ego_restores_exact_pose_after_disabling_physics(self) -> None:
        spawn = collector.carla.Transform(
            collector.carla.Location(x=10.0, y=20.0, z=0.6),
            collector.carla.Rotation(yaw=90.0),
        )
        world = mock.MagicMock()
        world.get_map.return_value.get_spawn_points.return_value = [spawn]
        actor = mock.MagicMock()
        world.try_spawn_actor.return_value = actor
        blueprint = mock.MagicMock()
        blueprint.id = "vehicle.lincoln.mkz"
        args = SimpleNamespace(
            ego_vehicle_blueprint="vehicle.lincoln.mkz",
            ego_spawn_index=0,
            ego_spawn_require_exact=True,
            ego_spawn_forward_offset_m=2.0,
            ego_spawn_right_offset_m=1.0,
            ego_spawn_z_offset_m=-0.2,
            ego_spawn_yaw_offset_deg=5.0,
            ego_freeze=True,
            experiment3_target_profile="none",
            experiment3_settle_ticks=0,
            ego_role_name="scenesense_phase2_helper",
        )
        with mock.patch.object(
            collector.od_demo,
            "resolve_hero_blueprint",
            return_value=(blueprint, False),
        ), mock.patch.object(
            collector.od_demo,
            "get_fresh_vehicle_blueprint",
            return_value=blueprint,
        ):
            result = collector._spawn_parked_ego_vehicle(world=world, args=args)

        self.assertIs(result, actor)
        actor.set_simulate_physics.assert_called_once_with(False)
        actor.set_transform.assert_called_once()
        restored = actor.set_transform.call_args.args[0]
        self.assertAlmostEqual(float(restored.location.x), 9.0)
        self.assertAlmostEqual(float(restored.location.y), 22.0)
        self.assertAlmostEqual(float(restored.location.z), 0.4)
        self.assertAlmostEqual(float(restored.rotation.yaw), 95.0)
        actor.set_target_velocity.assert_called_once()
        actor.set_target_angular_velocity.assert_called_once()

    def test_detached_launch_spec_is_reviewed_two_trajectory_carla_only(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config_path = (
            root
            / "data_collection/configs/phase2_paired_causal_pilot_reviewed_v1.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            spec = build_launch_spec(
                config_path,
                output_root=Path(temporary),
                timestamp="20260817_120000",
            )
        self.assertEqual(spec["status"], "validated_not_started")
        self.assertEqual(spec["trajectory_count"], 2)
        self.assertFalse(spec["inference_timing_citable"])
        self.assertEqual(
            spec["population_mode"], "frozen_curbside_pilot_no_ambient"
        )
        self.assertIn("--launch", spec["command"])
        self.assertTrue(str(spec["completion_sentinel"]).endswith("COMPLETED.json"))
        self.assertTrue(str(spec["failure_sentinel"]).endswith("FAILED.json"))


class OfflinePilotAnalysisContractTests(unittest.TestCase):
    def test_target_chain_ignores_earlier_arbitrary_warning_and_has_stable_tie_break(self):
        rows = [
            {
                "trajectory_id": "positive",
                "arm_id": "ego_only",
                "frame_id": 10,
                "warning_at_s": 1.0,
                "canonical_track_id": "vehicle-warning",
                "target_hazard_match": 0,
            },
            {
                "trajectory_id": "positive",
                "arm_id": "send_everything",
                "frame_id": 20,
                "warning_at_s": 2.0,
                "canonical_track_id": "send-target",
                "target_hazard_match": 1,
            },
            {
                "trajectory_id": "positive",
                "arm_id": "hazard_only",
                "frame_id": 20,
                "warning_at_s": 2.0,
                "canonical_track_id": "hazard-target",
                "target_hazard_match": 1,
            },
        ]
        selected = _select_target_chain_warning(rows, "positive")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["arm_id"], "hazard_only")
        self.assertEqual(selected["canonical_track_id"], "hazard-target")

    def test_warning_diagnostics_report_exposure_and_do_not_claim_adjudicated_false_alarm(self):
        metrics = [
            {
                "trajectory_id": "benign",
                "scenario_role": "matched_benign_negative",
                "arm_id": "hazard_only",
                "frame_count": 10,
            }
        ]
        warnings = [
            {
                "trajectory_id": "benign",
                "scenario_role": "matched_benign_negative",
                "arm_id": "hazard_only",
                "frame_id": 1,
                "canonical_track_id": "track-a",
                "class_name": "vehicle",
                "truth_matched": 1,
                "target_hazard_match": 0,
                "evaluation_truth_id": "1",
            },
            {
                "trajectory_id": "benign",
                "scenario_role": "matched_benign_negative",
                "arm_id": "hazard_only",
                "frame_id": 2,
                "canonical_track_id": "track-b",
                "class_name": "pedestrian",
                "truth_matched": 0,
                "target_hazard_match": 0,
                "evaluation_truth_id": None,
            },
        ]
        diagnostics, fragmentation = _warning_diagnostics(metrics, warnings)
        self.assertEqual(diagnostics[0]["warning_frame_count"], 2)
        self.assertAlmostEqual(diagnostics[0]["warning_frame_rate"], 0.2)
        self.assertEqual(diagnostics[0]["unmatched_warning_event_count"], 1)
        self.assertIn("not_hazard_adjudicated", diagnostics[0]["false_warning_adjudication_status"])
        self.assertEqual({row["truth_scope"] for row in fragmentation}, {"matched_truth_object", "unmatched"})

    def test_versioned_output_names_cannot_escape_the_batch(self):
        self.assertEqual(validated_replay_output_name("evaluation_v2", "evaluation"), "evaluation_v2")
        self.assertEqual(
            validated_verification_output_name("verification_v2", "verification"),
            "verification_v2",
        )
        for candidate in ("../evaluation_v2", "/tmp/evaluation_v2", "verification_v2"):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validated_replay_output_name(candidate, "evaluation")


class CausalCaptureRuntimeTests(unittest.TestCase):
    def test_unmatched_detection_gets_only_source_local_identity(self) -> None:
        tracker = SourceLocalCausalTracker("helper")
        tracks, associations = tracker.update(
            frame_id=7,
            timestamp_s=0.7,
            detections=[
                {
                    "class_name": "pedestrian",
                    "score": 0.12,
                    "world_x": 10.0,
                    "world_y": -2.0,
                    "world_z": 0.0,
                }
            ],
        )
        self.assertEqual(tracks[0]["source_track_id"], "helper:track:000001")
        self.assertEqual(associations[0]["association"], "birth")
        serialized = json.dumps({"tracks": tracks, "associations": associations})
        self.assertNotIn("actor_id", serialized)
        self.assertNotIn("ground_truth", serialized)

    def test_runtime_writes_causal_and_quota_bounded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = root / "capture.start.json"
            start.write_text("{}\n", encoding="utf-8")
            runtime = Phase2CaptureRuntime(
                Phase2RuntimeConfig(
                    role="recipient",
                    trajectory_id="pilot_positive_001",
                    scenario_role="controlled_positive_occlusion",
                    run_dir=root / "recipient",
                    ready_sentinel=root / "recipient.ready.json",
                    capture_start_sentinel=start,
                    tick_ready_path=root / "recipient.tick_ready.json",
                    heartbeat_path=root / "recipient.heartbeat.json",
                    contract_config_path=root / "contract.yaml",
                ),
                {
                    "maximum_window_seconds_per_trajectory": 20.0,
                    "maximum_raw_bytes_per_trajectory": 32_000_000,
                    "maximum_raw_bytes_pilot_total": 80_000_000,
                    "minimum_free_bytes_after_reservation": 1,
                },
            )
            runtime.on_pre_capture(
                world=_World(), anchor_actor=_Actor(), previous_frame_id=100
            )
            tick_ready = json.loads(
                (root / "recipient.tick_ready.json").read_text(encoding="utf-8")
            )
            self.assertEqual(tick_ready["after_frame_id"], 100)
            self.assertEqual(tick_ready["minimum_capture_frame"], 101)
            runtime.remember_radar_points(
                101,
                {
                    "world_xyz": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                    "valid_projection": np.asarray([1], dtype=np.uint8),
                },
            )
            runtime.record_inputs(
                frame_id=101,
                carla_timestamp=12.4,
                frame_bgr=np.zeros((8, 12, 3), dtype=np.uint8),
                radar_tensor=np.zeros((4, 4, 6), dtype=np.float32),
                camera_matrix=np.eye(4),
                camera_intrinsics_input=np.eye(3),
            )
            runtime.record_logits(
                101,
                {
                    "out": np.zeros((1, 3, 4, 6), dtype=np.float32),
                    "object": np.zeros((1, 12, 4, 6), dtype=np.float32),
                },
            )
            runtime.record_predictions(
                frame_id=101,
                carla_timestamp=12.4,
                objects=[
                    {
                        "class_name": "pedestrian",
                        "score": 0.08,
                        "world_x": 5.0,
                        "world_y": 1.0,
                        "world_z": 0.0,
                    }
                ],
            )
            runtime.mark_frame_complete(101, 12.4)
            runtime.close(status="complete")

            run_dir = root / "recipient"
            self.assertTrue((run_dir / "retained_inputs/frame_00000101_inputs.npz").is_file())
            self.assertTrue((run_dir / "retained_inputs/frame_00000101_logits.npz").is_file())
            audit_lines = (run_dir / "runtime/causal_decisions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(audit_lines), 2)
            placement = json.loads(audit_lines[0])
            self.assertEqual(placement["decision"]["decision_stage"], "placement")
            self.assertEqual(placement["fields"][0]["field_name"], "recipient_state")
            _audit_record(placement)
            tampered = dict(placement)
            tampered["record_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                _audit_record(tampered)
            raw_inventory = (run_dir / "runtime/raw_inference_inventory.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(",2,48,1,", raw_inventory)
            summary = json.loads(
                (run_dir / "phase2_runtime_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "complete")
            self.assertFalse(summary["retention"]["automatic_deletion_performed"])
            self.assertTrue((run_dir / "artifact_manifest.json").is_file())

    def test_runtime_rejects_a_second_pre_capture_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = root / "capture.start.json"
            start.write_text("{}\n", encoding="utf-8")
            runtime = Phase2CaptureRuntime(
                Phase2RuntimeConfig(
                    role="helper",
                    trajectory_id="pilot_positive_001",
                    scenario_role="controlled_positive_occlusion",
                    run_dir=root / "helper",
                    ready_sentinel=root / "helper.ready.json",
                    capture_start_sentinel=start,
                    tick_ready_path=root / "helper.tick_ready.json",
                    heartbeat_path=root / "helper.heartbeat.json",
                    contract_config_path=root / "contract.yaml",
                ),
                {
                    "maximum_window_seconds_per_trajectory": 20.0,
                    "maximum_raw_bytes_per_trajectory": 32_000_000,
                    "maximum_raw_bytes_pilot_total": 80_000_000,
                    "minimum_free_bytes_after_reservation": 1,
                },
            )
            runtime.on_pre_capture(
                world=_World(), anchor_actor=_Actor(), previous_frame_id=100
            )
            with self.assertRaisesRegex(RuntimeError, "before completing"):
                runtime.on_pre_capture(
                    world=_World(), anchor_actor=_Actor(), previous_frame_id=100
                )
            runtime.close(status="failed", error="expected test failure")


class TwoPhaseBarrierTests(unittest.TestCase):
    @staticmethod
    def _live_process():
        process = mock.MagicMock()
        process.poll.return_value = None
        return process

    def test_tick_wait_requires_both_collectors_at_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {role: root / f"{role}.tick_ready.json" for role in ("helper", "recipient")}
            for role, path in paths.items():
                path.write_text(
                    json.dumps(
                        {
                            "status": "armed_for_next_frame",
                            "source_role": role,
                            "after_frame_id": 100,
                            "minimum_capture_frame": 101,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            processes = {role: self._live_process() for role in paths}
            _wait_for_tick_ready(processes, paths, 100, 0.1)

    def test_frame_wait_fails_fast_when_collector_skips_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processes = {
                role: self._live_process() for role in ("helper", "recipient")
            }
            heartbeats = {
                role: root / f"{role}.heartbeat.json" for role in processes
            }
            tick_ready = {
                role: root / f"{role}.tick_ready.json" for role in processes
            }
            heartbeats["helper"].write_text(
                json.dumps({"frame_id": 101}) + "\n", encoding="utf-8"
            )
            for role, path in tick_ready.items():
                path.write_text(
                    json.dumps({"after_frame_id": 101}) + "\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(RuntimeError, "skipped CARLA frame 101"):
                _wait_for_frame(processes, heartbeats, tick_ready, 101, 0.1)

    def test_replay_contribution_uses_v2_exact_bytes_and_no_truth_identity(self) -> None:
        recipient = _recipient_state(
            pd.Series(
                {
                    "carla_timestamp": 1.0,
                    "world_x": 0.0,
                    "world_y": 0.0,
                    "velocity_x": 4.0,
                    "velocity_y": 0.0,
                }
            ),
            1.1,
        )
        tracks = pd.DataFrame(
            [
                {
                    "source_track_id": "helper:track:000001",
                    "source_role": "helper",
                    "tracker_version": "source_local_nearest_cv.v1",
                    "class_name": "pedestrian",
                    "world_x": 8.0,
                    "world_y": 0.0,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "last_observed_timestamp_s": 1.1,
                    "score": 0.8,
                }
            ]
        )
        contribution = _contribution(
            trajectory_id="pilot_positive_001",
            source_role="helper",
            sequence=1,
            captured_at_s=1.1,
            tracks=tracks,
            publication_action="PUBLISH_HAZARD_SUBSET",
            recipient=recipient,
            model_sha256="1" * 64,
            config_sha256="2" * 64,
        )
        encoded = contribution.to_json_bytes()
        self.assertEqual(contribution.application_payload_bytes, len(encoded))
        self.assertEqual(len(contribution.objects), 1)
        self.assertEqual(contribution.objects[0].source_track_id, "helper:track:000001")
        self.assertNotIn("actor_id", encoded.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
