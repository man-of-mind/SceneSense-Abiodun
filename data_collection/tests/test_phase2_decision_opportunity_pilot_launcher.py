from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_collection.launch_phase2_decision_opportunity_pilot import (
    ACCEPTANCE_SCHEMA,
    DEFAULT_CONFIG,
    EXPECTED_TRAJECTORY_IDS,
    build_launch_spec,
    launch_detached,
)
from data_collection.run_phase2_calibration_audit import _sha256


class _FakeDetachedProcess:
    pid = 424242

    def poll(self):
        return None


class Phase2DecisionOpportunityPilotLauncherTests(unittest.TestCase):
    def _config_with_missing_acceptance(self, root: Path) -> Path:
        pilot = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        pilot["visual_acceptance"]["record"] = str(root / "missing.json")
        config_path = root / "pilot_missing_acceptance.yaml"
        config_path.write_text(
            yaml.safe_dump(pilot, sort_keys=False), encoding="utf-8"
        )
        return config_path

    def _accepted_config(self, root: Path) -> Path:
        pilot = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        acceptance_path = root / "visual_acceptance.json"
        pilot["visual_acceptance"]["record"] = str(acceptance_path)
        config_path = root / "pilot.yaml"
        config_path.write_text(
            yaml.safe_dump(pilot, sort_keys=False), encoding="utf-8"
        )
        geometry_dir = root / "phase2_geometry_review_test"
        geometry_dir.mkdir()
        geometry_summary_path = geometry_dir / "geometry_review_summary.json"
        geometry_summary_path.write_text(
            json.dumps(
                {
                    "schema": "scenesense.phase2_geometry_review.v1",
                    "layout": "curbside_opposite",
                    "scenario_role": "controlled_positive_occlusion",
                    "hazard_actor_present": True,
                    "world_hz": 10.0,
                    "helper_command_speed_mps": 4.5,
                    "recipient_command_speed_mps": 5.0,
                    "pedestrian_start_delay_s": 2.0,
                    "pedestrian_speed_mps": 1.3,
                    "pedestrian_first_physical_motion_s": 2.1,
                    "pedestrian_physical_speed_gate_pass": True,
                    # Positive reviews intentionally do not use the benign-only
                    # progress gate.
                    "matched_benign_motion_gate_pass": None,
                    "collisions": [],
                    "legal_opposing_lane_contract": {
                        "pass": True,
                        "roles": {
                            "helper": {"road_id": 17, "lane_id": 1},
                            "recipient": {"road_id": 10, "lane_id": -2},
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        acceptance = {
            "schema": ACCEPTANCE_SCHEMA,
            "status": "accepted",
            "pilot_config_sha256": _sha256(config_path),
            "accepted_utc": "2026-08-19T01:30:00Z",
            "operator": "test-operator",
            "geometry_review_summary": str(geometry_summary_path),
            "geometry_review_summary_sha256": _sha256(geometry_summary_path),
            "checks": {
                name: True
                for name in pilot["visual_acceptance"]["required_checks"]
            },
        }
        acceptance_path.write_text(
            json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return config_path

    def test_validate_is_nonlaunching_and_reports_visual_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = self._config_with_missing_acceptance(Path(temporary))
            spec = build_launch_spec(
                config_path,
                timestamp="20260819_013000",
                operator_quality="Epic",
            )
        self.assertEqual(
            "validated_blocked_pending_visual_acceptance", spec["status"]
        )
        self.assertEqual(list(EXPECTED_TRAJECTORY_IDS), spec["trajectory_ids"])
        self.assertEqual(3, spec["trajectory_count"])
        self.assertEqual(2, spec["group_count"])
        self.assertFalse(spec["oai_launched"])
        self.assertFalse(spec["next_stage_chained"])
        self.assertNotIn("plan", spec)
        self.assertEqual(2.0, spec["treatment"]["pedestrian_start_delay_s"])
        self.assertEqual(
            3.0, spec["treatment"]["curbside_retention_start_offset_s"]
        )
        self.assertEqual(5_447_701_200, spec["estimated_heavy_bytes"])

    def test_checked_in_acceptance_is_current_and_launch_ready(self) -> None:
        spec = build_launch_spec(
            DEFAULT_CONFIG,
            timestamp="20260819_013008",
            operator_quality="Epic",
        )
        self.assertEqual("validated_ready_not_started", spec["status"])
        self.assertEqual("accepted", spec["visual_acceptance"]["status"])
        self.assertEqual("Abiodun", spec["visual_acceptance"]["operator"])

    def test_dry_plan_is_exact_and_contains_no_downstream_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = build_launch_spec(
                DEFAULT_CONFIG,
                output_root=Path(temporary) / "pilot-output",
                timestamp="20260819_013001",
                operator_quality="Epic",
                include_plan=True,
            )
        plan = spec["plan"]
        self.assertEqual(
            list(EXPECTED_TRAJECTORY_IDS),
            [row["trajectory_id"] for row in plan["trajectories"]],
        )
        self.assertFalse(plan["oai_launched"])
        self.assertFalse(plan["next_stage_chained"])
        self.assertEqual(
            "data_collection.run_phase2_calibration_audit", spec["command"][2]
        )
        self.assertNotIn("oai", " ".join(spec["command"]).lower())

    def test_launch_refuses_missing_visual_acceptance_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilot-output"
            config_path = self._config_with_missing_acceptance(root)
            spec = build_launch_spec(
                config_path,
                output_root=output,
                timestamp="20260819_013002",
                operator_quality="Epic",
            )
            with self.assertRaisesRegex(RuntimeError, "pending visual acceptance"):
                launch_detached(spec)
            self.assertFalse(output.exists())

    def test_acceptance_rejects_hash_valid_but_wrong_lane_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._accepted_config(root)
            pilot = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            acceptance_path = Path(pilot["visual_acceptance"]["record"])
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            summary_path = Path(acceptance["geometry_review_summary"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["legal_opposing_lane_contract"]["roles"]["recipient"][
                "lane_id"
            ] = -1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            acceptance["geometry_review_summary_sha256"] = _sha256(summary_path)
            acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "recipient road/lane"):
                build_launch_spec(
                    config_path,
                    output_root=root / "pilot-output",
                    timestamp="20260819_013006",
                    operator_quality="Epic",
                    require_visual_acceptance=True,
                )

    def test_acceptance_rejects_wrong_visual_timing_or_ego_speed(self) -> None:
        cases = (
            ("helper_command_speed_mps", 5.0, "helper_command_speed_mps"),
            (
                "pedestrian_first_physical_motion_s",
                2.4,
                "first physical motion",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path = self._accepted_config(root)
                pilot = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                acceptance_path = Path(pilot["visual_acceptance"]["record"])
                acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
                summary_path = Path(acceptance["geometry_review_summary"])
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary[field] = value
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                acceptance["geometry_review_summary_sha256"] = _sha256(summary_path)
                acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    build_launch_spec(
                        config_path,
                        output_root=root / "pilot-output",
                        timestamp="20260819_013007",
                        operator_quality="Epic",
                        require_visual_acceptance=True,
                    )

    def test_acceptance_is_hash_bound_and_all_checks_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._accepted_config(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "pilot-output",
                timestamp="20260819_013003",
                operator_quality="Epic",
                require_visual_acceptance=True,
            )
            self.assertEqual("validated_ready_not_started", spec["status"])
            self.assertEqual("accepted", spec["visual_acceptance"]["status"])

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            acceptance_path = Path(config["visual_acceptance"]["record"])
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            acceptance["pilot_config_sha256"] = "0" * 64
            acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different pilot config"):
                build_launch_spec(
                    config_path,
                    output_root=root / "second-output",
                    timestamp="20260819_013004",
                    operator_quality="Epic",
                    require_visual_acceptance=True,
                )

    def test_detached_launch_materializes_provenance_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._accepted_config(root)
            output = root / "pilot-output"
            spec = build_launch_spec(
                config_path,
                output_root=output,
                timestamp="20260819_013005",
                operator_quality="Epic",
                require_visual_acceptance=True,
            )

            def fake_popen(*_args, **_kwargs):
                batch = Path(spec["batch_root"])
                batch.mkdir(parents=True)
                (batch / "plan.json").write_text("{}\n", encoding="utf-8")
                return _FakeDetachedProcess()

            with mock.patch(
                "data_collection.launch_phase2_decision_opportunity_pilot.subprocess.Popen",
                side_effect=fake_popen,
            ):
                result = launch_detached(spec)
            self.assertEqual(
                "launched_detached_startup_acknowledged", result["status"]
            )
            self.assertTrue(Path(spec["resolved_config"]).is_file())
            self.assertTrue(Path(spec["launch_manifest"]).is_file())
            self.assertTrue(Path(spec["startup_ack"]).is_file())
            self.assertFalse(Path(spec["startup_failed"]).exists())
            self.assertEqual(
                spec["resolved_config_sha256"], _sha256(Path(spec["resolved_config"]))
            )

            with self.assertRaises(FileExistsError):
                with mock.patch(
                    "data_collection.launch_phase2_decision_opportunity_pilot.subprocess.Popen"
                ):
                    launch_detached(spec)


if __name__ == "__main__":
    unittest.main()
