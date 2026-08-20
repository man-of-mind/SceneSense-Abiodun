from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_collection.launch_phase2_factor_realization_smoke import (
    DEFAULT_CONFIG,
    EXPECTED_STATIC_ENVIRONMENT_TRUTH,
    _run_stage,
    build_corner_review_plan,
    build_launch_spec,
    launch_detached,
    record_corner_acceptance,
)
import data_collection.launch_phase2_factor_realization_smoke as factor_launcher
from data_collection.run_phase2_calibration_audit import _sha256
from data_collection.validate_phase2_factor_realization_smoke import (
    build_plan as build_factor_plan,
    load_config as load_factor_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeDetachedProcess:
    pid = 848484

    def poll(self):
        return None


class Phase2FactorRealizationLauncherTests(unittest.TestCase):
    def _ready_overlay(self, root: Path) -> Path:
        smoke = load_factor_config()
        smoke["source_design"]["manifest_sha256"] = _sha256(
            REPO_ROOT / smoke["source_design"]["manifest"]
        )
        smoke["source_design"]["design_config_sha256"] = _sha256(
            REPO_ROOT / smoke["source_design"]["design_config"]
        )
        for dependency in smoke["recipient_endpoint_runtime"][
            "dependency_sha256"
        ].values():
            dependency["sha256"] = _sha256(REPO_ROOT / dependency["path"])
        smoke["runtime_readiness"] = {
            "factor_adapter_status": "ready_verified",
            "recipient_install_event_status": "ready_verified",
            "policy_feature_projection_status": "ready_verified",
            "launch_wrapper_status": "ready_verified",
            "consequence": "exact_16_runtime_ready_separate_manual_launch_required",
        }
        smoke_path = root / "smoke.yaml"
        smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
        plan = build_factor_plan(load_factor_config(smoke_path))

        overlay = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        overlay["factor_smoke_config"] = str(smoke_path)
        overlay["factor_smoke_config_sha256"] = _sha256(smoke_path)
        contract = REPO_ROOT / "phase2_map_sharing/FACTOR_REALIZATION_SMOKE_V1.md"
        overlay["factor_smoke_contract_sha256"] = _sha256(contract)
        for field in ("base_runner", "factor_validator", "factor_postflight"):
            overlay[f"{field}_sha256"] = _sha256(REPO_ROOT / overlay[field])
        reviewer = REPO_ROOT / "data_collection/review_phase2_pair_geometry.py"
        overlay["geometry_reviewer_sha256"] = _sha256(reviewer)
        overlay["expected_factor_plan_sha256"] = plan["plan_sha256"]
        overlay["output_root"] = str(root / "experiments")
        overlay["manual_corner_review"]["output_root"] = str(root / "reviews")
        overlay["manual_corner_review"]["archive_root"] = str(root / "archive")
        overlay["manual_corner_review"]["acceptance_record"] = str(
            root / "corner_acceptance.json"
        )
        config_path = root / "detached.yaml"
        config_path.write_text(
            yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
        )
        return config_path

    def _write_review_artifacts(self, config_path: Path, root: Path) -> dict:
        plan = build_corner_review_plan(config_path, review_root=root)
        for row in plan["commands"]:
            requested = row["requested_factor_contract"]
            directory = root / str(row["trajectory_id"])
            directory.mkdir(parents=True)
            summary = {
                "schema": "scenesense.phase2_geometry_review.v1",
                "layout": (
                    "curbside_opposite"
                    if row["hazard_class"] == "pedestrian"
                    else "cross_traffic_vehicle"
                ),
                "scenario_role": "controlled_positive_occlusion",
                "hazard_actor_present": True,
                "world_hz": 10.0,
                "helper_command_speed_mps": requested["requested_helper_speed_mps"],
                "recipient_command_speed_mps": requested[
                    "requested_recipient_speed_mps"
                ],
                "collisions": [],
                "lane_contract": {"pass": True},
                "factor_runtime_contract": {
                    "trajectory_id": row["trajectory_id"],
                    "trajectory_row_sha256": row["trajectory_row_sha256"],
                    "requested_factors": requested,
                },
                "realized_factors": {
                    "realized_hazard_onset_s": requested[
                        "requested_hazard_onset_s"
                    ]
                    + 0.1,
                    "pre_intervention_radial_closing_speed_mps": requested[
                        "requested_closing_speed_target_mps"
                    ],
                    "pre_intervention_hazard_proximity_horizon_s": requested[
                        "requested_proximity_horizon_target_s"
                    ],
                    "pre_intervention_minimum_surface_clearance_m": 0.5,
                    "geometry_measurement_basis": requested[
                        "geometry_measurement_basis"
                    ],
                    "closing_speed_measurement_basis": requested[
                        "closing_speed_measurement_basis"
                    ],
                    "proximity_horizon_measurement_basis": requested[
                        "proximity_horizon_measurement_basis"
                    ],
                },
                "factor_realization_gate": {"pass": True},
            }
            if row["hazard_class"] == "pedestrian":
                summary.update(
                    {
                        "pedestrian_speed_mps": requested[
                            "requested_hazard_actor_speed_mps"
                        ],
                        "pedestrian_start_delay_s": requested[
                            "requested_hazard_onset_s"
                        ],
                        "pedestrian_first_physical_motion_s": requested[
                            "requested_hazard_onset_s"
                        ]
                        + 0.1,
                        "pedestrian_physical_speed_gate_pass": True,
                    }
                )
            else:
                summary.update(
                    {
                        "target_vehicle_command_speed_mps": requested[
                            "requested_hazard_actor_speed_mps"
                        ],
                        "target_vehicle_start_delay_s": requested[
                            "requested_hazard_onset_s"
                        ],
                        "vehicle_hazard_review_gate": {"pass": True},
                    }
                )
            (directory / "geometry_review_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (directory / "combined.png").write_bytes(b"test-png-evidence")
            (directory / "realized_pose_trace.csv").write_text(
                "frame,elapsed_s\n1,0.1\n", encoding="utf-8"
            )
        return plan

    def _accept(self, config_path: Path, review_root: Path) -> dict:
        plan = self._write_review_artifacts(config_path, review_root)
        return record_corner_acceptance(
            config_path,
            review_root=review_root,
            operator="test-operator",
            confirmed_checks=plan["required_human_checks"],
        )

    def test_review_plan_is_exact_all_eight_and_nonlaunching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            plan = build_corner_review_plan(config_path)
        self.assertEqual(8, plan["positive_corner_count"])
        self.assertFalse(plan["carla_or_oai_started_by_this_plan"])
        self.assertEqual(8, len({row["trajectory_id"] for row in plan["commands"]}))
        for row in plan["commands"]:
            command = row["command"]
            self.assertIn("--factor-trajectory-id", command)
            self.assertIn("--factor-smoke-config", command)
            self.assertNotIn("--headless", command)

    def test_ready_runtime_still_blocks_without_corner_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120000",
            )
        self.assertEqual(
            "validated_blocked_pending_manual_corner_acceptance", spec["status"]
        )
        self.assertTrue(spec["runtime_ready"])
        self.assertEqual(16, spec["trajectory_count"])
        self.assertTrue(spec["atomic_all_or_none"])
        self.assertFalse(spec["partial_admission_authorized"])

    def test_hash_bound_all_eight_acceptance_makes_launch_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            acceptance = self._accept(config_path, root / "reviews")
            self.assertTrue((root / "archive").is_dir())
            self.assertTrue(
                all(
                    Path(item["artifact_directory"]).is_relative_to(root / "archive")
                    for item in acceptance["review_artifacts"]
                )
            )
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120001",
            )
        self.assertEqual("validated_ready_not_started", spec["status"])
        self.assertEqual("accepted", spec["manual_corner_acceptance"]["status"])
        self.assertFalse(spec["oai_launched"])
        self.assertFalse(spec["old_audit_chained"])
        self.assertFalse(spec["next_stage_chained"])

    def test_acceptance_remains_bound_to_an_explicit_unique_review_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            unique_review_root = root / "fresh_final_review_root"
            acceptance = self._accept(config_path, unique_review_root)
            self.assertEqual(
                str(unique_review_root.resolve()), acceptance["source_review_root"]
            )
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120008",
            )
            self.assertEqual("validated_ready_not_started", spec["status"])

    def test_acceptance_rejects_one_out_of_band_physical_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            plan = self._write_review_artifacts(config_path, root / "reviews")
            first = root / "reviews" / plan["commands"][0]["trajectory_id"]
            summary_path = first / "geometry_review_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["realized_factors"][
                "pre_intervention_radial_closing_speed_mps"
            ] = 999.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside their exact cell"):
                record_corner_acceptance(
                    config_path,
                    review_root=root / "reviews",
                    operator="test-operator",
                    confirmed_checks=plan["required_human_checks"],
                )

    def test_retention_boundary_authored_plus_one_tick_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            acceptance = self._accept(config_path, root / "reviews")
            self.assertEqual("accepted", acceptance["status"])
            plan = build_corner_review_plan(
                config_path, review_root=root / "reviews"
            )
            bounded = next(
                row
                for row in plan["commands"]
                if row["requested_factor_contract"]["requested_hazard_onset_s"]
                >= 1.0
            )
            retention = bounded["retention_window_preflight"]
            realized_onset = (
                bounded["requested_factor_contract"]["requested_hazard_onset_s"]
                + 0.1
            )
            self.assertAlmostEqual(
                2.9,
                retention["expected_last_retained_sample_s"] - realized_onset,
            )

    def test_retention_boundary_authored_plus_two_ticks_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            plan = self._write_review_artifacts(config_path, root / "reviews")
            bounded = next(
                row
                for row in plan["commands"]
                if row["requested_factor_contract"]["requested_hazard_onset_s"]
                >= 1.0
            )
            summary_path = next(
                path
                for path in (root / "reviews").rglob(
                    "geometry_review_summary.json"
                )
                if json.loads(path.read_text(encoding="utf-8"))[
                    "factor_runtime_contract"
                ]["trajectory_id"]
                == bounded["trajectory_id"]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["realized_factors"]["realized_hazard_onset_s"] = (
                bounded["requested_factor_contract"]["requested_hazard_onset_s"]
                + 0.2
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "insufficient exact retained-sample post-window margin"
            ):
                record_corner_acceptance(
                    config_path,
                    review_root=root / "reviews",
                    operator="test-operator",
                    confirmed_checks=plan["required_human_checks"],
                )

    def test_accepted_corner_evidence_is_hash_bound_against_later_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            acceptance = self._accept(config_path, root / "reviews")
            summary_path = Path(
                acceptance["review_artifacts"][0]["geometry_review_summary"]
            )
            summary_path.write_text(
                summary_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "summary hash drifted"):
                build_launch_spec(
                    config_path,
                    output_root=root / "runs",
                    timestamp="20260819_120007",
                )

    def test_detached_launch_is_create_only_and_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120002",
                include_plan=True,
            )
            self.assertEqual(
                {
                    "schema",
                    "enabled",
                    "factor_smoke_config",
                    "factor_smoke_config_sha256",
                    "factor_smoke_plan",
                    "factor_smoke_plan_sha256",
                    "exact_trajectory_count",
                    "atomic_batch",
                },
                set(
                    spec["resolved_config_payload"][
                        "factor_realization_runtime"
                    ]
                ),
            )
            self.assertEqual(
                EXPECTED_STATIC_ENVIRONMENT_TRUTH,
                spec["resolved_config_payload"]["static_environment_truth"],
            )

            def fake_popen(*_args, **_kwargs):
                batch = Path(spec["batch_root"])
                batch.mkdir(parents=True)
                (batch / "progress.jsonl").write_text("{}\n", encoding="utf-8")
                return _FakeDetachedProcess()

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.Popen",
                side_effect=fake_popen,
            ):
                result = launch_detached(spec)
            self.assertEqual(
                "launched_detached_startup_acknowledged", result["status"]
            )
            self.assertTrue(Path(spec["resolved_config"]).is_file())
            self.assertTrue(Path(spec["factor_plan"]).is_file())
            self.assertTrue(Path(spec["launch_manifest"]).is_file())
            self.assertTrue(Path(spec["startup_ack"]).is_file())
            with self.assertRaises(FileExistsError):
                with mock.patch(
                    "data_collection.launch_phase2_factor_realization_smoke.subprocess.Popen"
                ):
                    launch_detached(spec)

    def test_immediate_stage_failure_is_not_reported_as_startup_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120009",
                include_plan=True,
            )

            def fake_popen(*_args, **_kwargs):
                batch = Path(spec["batch_root"])
                batch.mkdir(parents=True)
                (batch / "progress.jsonl").write_text("{}\n", encoding="utf-8")
                (batch / "FAILED.json").write_text(
                    json.dumps(
                        {
                            "schema": "scenesense.phase2_factor_realization_stage.v1",
                            "status": "failed_excluded_atomic_fixture",
                            "verdict": "FAIL_HOLD_EXCLUDE_ALL_16",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return _FakeDetachedProcess()

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.Popen",
                side_effect=fake_popen,
            ), self.assertRaisesRegex(
                RuntimeError, "wrote FAILED before startup acknowledgement"
            ):
                launch_detached(spec)
            self.assertTrue(Path(spec["startup_failed"]).is_file())
            self.assertFalse(Path(spec["startup_ack"]).exists())

    def test_child_preflight_failure_writes_atomic_failure_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120010",
                include_plan=True,
            )
            batch = Path(spec["batch_root"])
            returncode = factor_launcher.main(
                [
                    "--config",
                    str(config_path),
                    "--resolved-config",
                    str(spec["resolved_config"]),
                    "--factor-plan",
                    str(spec["factor_plan"]),
                    "--batch-root",
                    str(batch),
                    "--run-stage",
                ]
            )
            self.assertEqual(1, returncode)
            failure = json.loads((batch / "FAILED.json").read_text(encoding="utf-8"))
            summary = json.loads(
                (batch / "RESULTS_SUMMARY.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure, summary)
            self.assertEqual(
                "child_preflight_before_raw_capture", failure["phase"]
            )
            self.assertTrue((batch / "progress.jsonl").is_file())

    def test_launcher_source_has_no_duplicate_literal_dictionary_keys(self) -> None:
        source_path = Path(factor_launcher.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            repeated = sorted({key for key in keys if keys.count(key) > 1})
            if repeated:
                duplicates.append((node.lineno, repeated))
        self.assertEqual([], duplicates)

    def test_atomic_json_partial_temporary_write_never_publishes_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "COMPLETED.json"

            def partial_then_fail(_payload, stream, **_kwargs):
                stream.write('{"partial":')
                stream.flush()
                raise OSError("injected partial temporary write")

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.json.dump",
                side_effect=partial_then_fail,
            ), self.assertRaisesRegex(OSError, "partial temporary write"):
                factor_launcher._write_json_x(destination, {"complete": True})
            self.assertFalse(destination.exists())
            self.assertEqual([], list(root.glob(".COMPLETED.json.tmp.*")))

    def test_raw_audit_completion_without_atomic_bundle_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120003",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ):
                returncode = _run_stage(
                    config_path, resolved, plan_path, batch
                )
            self.assertEqual(1, returncode)
            self.assertTrue((batch / "FAILED.json").is_file())
            self.assertTrue((batch / "RESULTS_SUMMARY.json").is_file())
            self.assertFalse((batch / "COMPLETED.json").exists())

    def test_first_progress_write_failure_still_materializes_failure_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120011",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])
            real_append = factor_launcher._append_progress
            calls = 0

            def fail_first(path, event, **fields):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("injected first progress failure")
                return real_append(path, event, **fields)

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._append_progress",
                side_effect=fail_first,
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(1, returncode)
            self.assertTrue((batch / "FAILED.json").is_file())
            self.assertTrue((batch / "RESULTS_SUMMARY.json").is_file())
            self.assertFalse((batch / "COMPLETED.json").exists())

    def test_final_progress_failure_cannot_create_both_terminal_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120012",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])
            validation = {"verdict": "PASS_ATOMIC_EXACT_16_ADMITTED"}

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                (raw / "factor_smoke_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (raw / "factor_smoke_validation.json").write_text(
                    json.dumps(validation) + "\n", encoding="utf-8"
                )
                return mock.Mock(returncode=0)

            real_append = factor_launcher._append_progress

            def fail_final(path, event, **fields):
                if event == "stage_complete":
                    raise OSError("injected final progress failure")
                return real_append(path, event, **fields)

            stable = spec["relevant_source_tree_fingerprint"]
            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.validate_results",
                return_value=validation,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._relevant_source_tree_fingerprint",
                side_effect=[stable, stable],
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._append_progress",
                side_effect=fail_final,
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(1, returncode)
            self.assertTrue((batch / "FAILED.json").is_file())
            self.assertFalse((batch / "COMPLETED.json").exists())
            summary = json.loads(
                (batch / "RESULTS_SUMMARY.json").read_text(encoding="utf-8")
            )
            self.assertEqual("FAIL_HOLD_EXCLUDE_ALL_16", summary["verdict"])

    def test_partial_completion_publish_rolls_back_to_one_failure_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120013",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])
            validation = {"verdict": "PASS_ATOMIC_EXACT_16_ADMITTED"}

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                (raw / "factor_smoke_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (raw / "factor_smoke_validation.json").write_text(
                    json.dumps(validation) + "\n", encoding="utf-8"
                )
                return mock.Mock(returncode=0)

            real_write = factor_launcher._write_json_x

            def partial_completion(path, payload):
                if Path(path).name == "COMPLETED.json":
                    Path(path).write_text('{"partial":', encoding="utf-8")
                    raise OSError("injected partial completion publish")
                return real_write(path, payload)

            stable = spec["relevant_source_tree_fingerprint"]
            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.validate_results",
                return_value=validation,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._relevant_source_tree_fingerprint",
                side_effect=[stable, stable],
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._write_json_x",
                side_effect=partial_completion,
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(1, returncode)
            self.assertFalse((batch / "COMPLETED.json").exists())
            self.assertTrue((batch / "FAILED.json").is_file())
            summary = json.loads(
                (batch / "RESULTS_SUMMARY.json").read_text(encoding="utf-8")
            )
            self.assertEqual("FAIL_HOLD_EXCLUDE_ALL_16", summary["verdict"])

    def test_nonregistered_validator_verdict_cannot_complete_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120004",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                (raw / "factor_smoke_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (raw / "factor_smoke_validation.json").write_text(
                    json.dumps(
                        {"verdict": "PASS_BUT_NOT_THE_REGISTERED_ATOMIC_TOKEN"}
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.validate_results",
                return_value={"verdict": "PASS_BUT_NOT_THE_REGISTERED_ATOMIC_TOKEN"},
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(1, returncode)
            self.assertTrue((batch / "FAILED.json").is_file())
            self.assertFalse((batch / "COMPLETED.json").exists())
            self.assertFalse((batch / "factor_smoke_validation.json").exists())

    def test_result_defining_source_drift_fails_after_postflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120005",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                (raw / "factor_smoke_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (raw / "factor_smoke_validation.json").write_text(
                    json.dumps({"verdict": "PASS_ATOMIC_EXACT_16_ADMITTED"})
                    + "\n",
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            before = spec["relevant_source_tree_fingerprint"]
            after = {**before, "manifest_sha256": "f" * 64}
            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.validate_results",
                return_value={"verdict": "PASS_ATOMIC_EXACT_16_ADMITTED"},
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._relevant_source_tree_fingerprint",
                side_effect=[before, after],
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(1, returncode)
            self.assertTrue((batch / "FAILED.json").is_file())
            self.assertFalse((batch / "COMPLETED.json").exists())

    def test_exact_raw_bundle_and_registered_validation_complete_outer_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._ready_overlay(root)
            self._accept(config_path, root / "reviews")
            spec = build_launch_spec(
                config_path,
                output_root=root / "runs",
                timestamp="20260819_120006",
                include_plan=True,
            )
            plan_path = Path(spec["factor_plan"])
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(
                json.dumps(spec["factor_plan_payload"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            resolved = Path(spec["resolved_config"])
            resolved.write_text(
                yaml.safe_dump(spec["resolved_config_payload"], sort_keys=False),
                encoding="utf-8",
            )
            batch = Path(spec["batch_root"])
            validation = {
                "verdict": "PASS_ATOMIC_EXACT_16_ADMITTED",
                "trajectory_count": 16,
                "group_count": 8,
            }

            def fake_run(*_args, **_kwargs):
                raw = batch / "raw_capture"
                raw.mkdir(parents=True)
                (raw / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
                (raw / "factor_smoke_results.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (raw / "factor_smoke_validation.json").write_text(
                    json.dumps(validation) + "\n", encoding="utf-8"
                )
                return mock.Mock(returncode=0)

            stable = spec["relevant_source_tree_fingerprint"]
            with mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke.validate_results",
                return_value=validation,
            ), mock.patch(
                "data_collection.launch_phase2_factor_realization_smoke._relevant_source_tree_fingerprint",
                side_effect=[stable, stable],
            ):
                returncode = _run_stage(config_path, resolved, plan_path, batch)
            self.assertEqual(0, returncode)
            completed = json.loads(
                (batch / "COMPLETED.json").read_text(encoding="utf-8")
            )
            self.assertEqual("complete_atomic_exact_16_admitted", completed["status"])
            self.assertEqual(
                "PASS_ATOMIC_EXACT_16_ADMITTED", completed["verdict"]
            )
            self.assertTrue((batch / "factor_smoke_validation.json").is_file())
            self.assertFalse((batch / "FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
