from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_collection.phase2_factor_realization_runtime import (
    FactorRealizationMonitor,
    FactorRuntimeContract,
    canonical_sha256,
    instantaneous_relative_motion,
    nontreatment_plan_record,
)
from data_collection.phase2_causal_runtime import (
    Phase2CaptureRuntime,
    Phase2RuntimeConfig,
)
from data_collection.run_phase2_calibration_audit import (
    _expected_retention_bytes,
    _persist_factor_forensic_then_finalize,
    _retention_window_for_row,
)
from data_collection.review_phase2_pair_geometry import (
    _load_factor_review_contract,
)
from data_collection.validate_phase2_factor_realization_smoke import (
    build_plan as build_factor_smoke_plan,
    load_config as load_factor_smoke_config,
)


def _requested() -> dict[str, object]:
    return {
        "closing_speed_band": "low",
        "time_to_hazard_band": "long",
        "factor_realization_status": "provisional_controls_pending_bounded_factor_smoke",
        "time_to_hazard_label_status": "not_scientifically_realized_until_bounded_factor_smoke",
        "hazard_actor_role": "target_vehicle",
        "onset_driver_role": "target_vehicle",
        "geometry_measurement_basis": "typed_test_geometry",
        "closing_speed_measurement_basis": "instantaneous_radial_center_closing_at_first_realized_onset_sample",
        "proximity_horizon_measurement_basis": "instantaneous_relative_linear_motion_center_proximity_horizon_not_collision_ttc",
        "requested_helper_speed_mps": 4.5,
        "requested_recipient_speed_mps": 2.0,
        "requested_hazard_actor_speed_mps": 1.0,
        "requested_onset_driver_speed_mps": 1.0,
        "requested_hazard_onset_s": 1.0,
        "requested_closing_speed_target_mps": 3.0,
        "requested_closing_speed_band_min_mps": 2.0,
        "requested_closing_speed_band_max_mps": 4.0,
        "requested_proximity_horizon_target_s": 10.0 / 3.0,
        "requested_proximity_horizon_band_min_s": 3.0,
        "requested_proximity_horizon_band_max_s": 5.0,
        "minimum_onset_driver_speed_mps": 0.2,
    }


def _plan_row(*, positive: bool = True) -> dict[str, object]:
    return {
        "trajectory_id": "pair_pos" if positive else "pair_ben",
        "trajectory_row_sha256": "a" * 64,
        "group_id": "pair",
        "geometry_or_route_id": "occluded_cross_traffic_vehicle",
        "hazard_class": "vehicle",
        "scenario_role": (
            "controlled_positive_occlusion"
            if positive
            else "matched_benign_negative"
        ),
        "controlled_hazard_present": positive,
        "requested_factor_contract": _requested(),
    }


class _Actor:
    def __init__(self, x: float, vx: float, *, y: float = 0.0, vy: float = 0.0):
        self._location = SimpleNamespace(x=x, y=y, z=0.0)
        self._velocity = SimpleNamespace(x=vx, y=vy, z=0.0)
        self._transform = SimpleNamespace(
            location=self._location,
            rotation=SimpleNamespace(yaw=0.0),
        )
        self.bounding_box = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            extent=SimpleNamespace(x=1.0, y=0.5, z=0.5),
            rotation=SimpleNamespace(yaw=0.0),
        )

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity

    def get_transform(self):
        return self._transform


def _manifest_row() -> dict[str, object]:
    values = {
        "schema": "v2",
        "design_id": "design",
        "suite_id": "A",
        "split": "calibration",
        "group_id": "pair",
        "matched_pair_id": "pair",
        "geometry_or_route_id": "geometry",
        "geometry_or_route_status": "reviewed",
        "traffic_density": "not_applicable",
        "traffic_density_status": "not_applicable",
        "ambient_population_mode": "scenario_owned_only",
        "ambient_population_process_required": 0,
        "weather": "ClearNoon",
        "renderer_quality_level": "Epic",
        "renderer_server_launch_flag": "-quality-level=Epic",
        "carla_seed": 1,
        "traffic_seed": 2,
        "sensor_seed": 3,
        "raw_retention_tier": "inputs_plus_logits_window",
        "raw_window_duration_s": 4.0,
        "raw_window_anchor": "conflict",
        "pair_contract_id": "pair-contract",
        "route_start_anchor_id": None,
        "recipient_start_index": None,
        "helper_start_index": None,
        "recipient_route_sha256": None,
        "helper_route_sha256": None,
        "requested_helper_speed_mps": 4.5,
        "requested_recipient_speed_mps": 3.0,
        "trajectory_id": "pair_pos",
        "scenario_role": "controlled_positive_occlusion",
        "controlled_hazard_present": 1,
        "requested_hazard_actor_speed_mps": 3.6,
        "requested_hazard_onset_s": 2.0,
    }
    return values


class FactorRealizationRuntimeTests(unittest.TestCase):
    def test_exact_first_onset_sample_passes_registered_bands(self) -> None:
        contract = FactorRuntimeContract.from_plan_row(
            _plan_row(), maximum_surface_clearance_m=3.0
        )
        helper = _Actor(-5.0, 4.5)
        recipient = _Actor(0.0, 2.0)
        hazard = _Actor(10.0, -1.0)
        monitor = FactorRealizationMonitor(contract, cadence_s=0.1)

        monitor.observe(
            frame_id=10,
            elapsed_s=1.0,
            helper=helper,
            recipient=recipient,
            hazard=hazard,
            onset_driver=hazard,
            recipient_intervened=False,
        )
        result = monitor.finalize()

        realized = result["realized_factors"]
        self.assertAlmostEqual(
            realized["pre_intervention_radial_closing_speed_mps"], 3.0
        )
        self.assertAlmostEqual(
            realized["pre_intervention_hazard_proximity_horizon_s"], 10.0 / 3.0
        )
        self.assertEqual(
            "constant_velocity_fixed_orientation_obb_at_center_closest_approach",
            realized["surface_clearance_prediction_basis"],
        )
        self.assertTrue(result["factor_realization_gate"]["pass"])

    def test_radial_metric_is_not_speed_sum_or_collision_ttc(self) -> None:
        recipient = _Actor(0.0, 2.0)
        hazard = _Actor(0.0, 1.0, y=10.0, vy=-1.0)
        result = instantaneous_relative_motion(recipient, hazard)
        self.assertAlmostEqual(result["radial_closing_speed_mps"], 1.0)
        # The lateral approach is coupled with relative X motion, so this is
        # time to closest center proximity rather than distance/closing-speed.
        self.assertAlmostEqual(result["center_proximity_horizon_s"], 5.0)

    def test_authored_zero_onset_waits_for_first_physical_speed_sample(self) -> None:
        row = copy.deepcopy(_plan_row())
        row["requested_factor_contract"]["requested_hazard_onset_s"] = 0.0
        contract = FactorRuntimeContract.from_plan_row(
            row, maximum_surface_clearance_m=3.0
        )
        helper = _Actor(-5.0, 4.5)
        recipient = _Actor(0.0, 2.0)
        hazard = _Actor(10.0, 0.0)
        monitor = FactorRealizationMonitor(contract, cadence_s=0.1)
        monitor.observe(
            frame_id=1,
            elapsed_s=0.1,
            helper=helper,
            recipient=recipient,
            hazard=hazard,
            onset_driver=hazard,
            recipient_intervened=False,
        )
        self.assertIsNone(monitor.realized)
        hazard._velocity.x = -1.0
        monitor.observe(
            frame_id=2,
            elapsed_s=0.2,
            helper=helper,
            recipient=recipient,
            hazard=hazard,
            onset_driver=hazard,
            recipient_intervened=False,
        )
        self.assertEqual(
            0.2,
            monitor.finalize()["realized_factors"]["realized_hazard_onset_s"],
        )

    def test_intervened_or_out_of_band_positive_fails_closed(self) -> None:
        contract = FactorRuntimeContract.from_plan_row(
            _plan_row(), maximum_surface_clearance_m=3.0
        )
        actors = (_Actor(-5.0, 4.5), _Actor(0.0, 2.0), _Actor(10.0, -1.0))
        monitor = FactorRealizationMonitor(contract, cadence_s=0.1)
        monitor.observe(
            frame_id=10,
            elapsed_s=1.0,
            helper=actors[0],
            recipient=actors[1],
            hazard=actors[2],
            onset_driver=actors[2],
            recipient_intervened=True,
        )
        with self.assertRaisesRegex(RuntimeError, "after recipient intervention"):
            monitor.finalize()

        drift = copy.deepcopy(_plan_row())
        drift["requested_factor_contract"][
            "requested_closing_speed_band_min_mps"
        ] = 6.0
        drift["requested_factor_contract"]["requested_closing_speed_target_mps"] = 7.0
        drift["requested_factor_contract"][
            "requested_closing_speed_band_max_mps"
        ] = 8.0
        monitor = FactorRealizationMonitor(
            FactorRuntimeContract.from_plan_row(
                drift, maximum_surface_clearance_m=3.0
            ),
            cadence_s=0.1,
        )
        monitor.observe(
            frame_id=10,
            elapsed_s=1.0,
            helper=actors[0],
            recipient=actors[1],
            hazard=actors[2],
            onset_driver=actors[2],
            recipient_intervened=False,
        )
        with self.assertRaisesRegex(RuntimeError, "radial_closing_speed_out_of_band"):
            monitor.finalize()
        diagnostic = monitor.diagnostic()
        self.assertFalse(diagnostic["factor_realization_gate"]["pass"])
        self.assertIn(
            "radial_closing_speed_out_of_band",
            diagnostic["factor_realization_gate"]["failures"],
        )

    def test_degenerate_measurement_is_recorded_instead_of_escaping_observe(self) -> None:
        contract = FactorRuntimeContract.from_plan_row(
            _plan_row(), maximum_surface_clearance_m=3.0
        )
        helper = _Actor(-5.0, 4.5)
        recipient = _Actor(0.0, 2.0)
        # Equal velocities make the registered relative-motion diagnostic
        # undefined after the onset driver has crossed its physical floor.
        hazard = _Actor(10.0, 2.0)
        monitor = FactorRealizationMonitor(contract, cadence_s=0.1)

        monitor.observe(
            frame_id=10,
            elapsed_s=1.0,
            helper=helper,
            recipient=recipient,
            hazard=hazard,
            onset_driver=hazard,
            recipient_intervened=False,
        )

        diagnostic = monitor.diagnostic()
        self.assertFalse(diagnostic["factor_realization_gate"]["pass"])
        self.assertIn(
            "factor measurement failed: ValueError: relative motion is zero",
            diagnostic["factor_realization_gate"]["failures"][0],
        )
        with self.assertRaisesRegex(RuntimeError, "factor measurement failed"):
            monitor.finalize()

    def test_benign_is_typed_not_applicable_without_fabricated_metrics(self) -> None:
        contract = FactorRuntimeContract.from_plan_row(
            _plan_row(positive=False), maximum_surface_clearance_m=3.0
        )
        monitor = FactorRealizationMonitor(contract, cadence_s=0.1)
        result = monitor.finalize()
        self.assertTrue(result["registered_target_absent"])
        self.assertNotIn("realized_factors", result)
        self.assertEqual("pair_pos", result["factor_reference_trajectory_id"])

    def test_pair_hash_excludes_treatment_but_includes_live_context(self) -> None:
        positive = _manifest_row()
        benign = copy.deepcopy(positive)
        benign.update(
            trajectory_id="pair_ben",
            scenario_role="matched_benign_negative",
            controlled_hazard_present=0,
            requested_hazard_actor_speed_mps=0.0,
            requested_hazard_onset_s=99.0,
        )
        signature = [
            {
                "type_id": "vehicle.test",
                "role_name": "helper",
                "motion_mode": "frozen_scenario_contract",
                "x": 1.0,
            }
        ]
        left = nontreatment_plan_record(
            positive, scenario_owned_signature=signature
        )
        right = nontreatment_plan_record(benign, scenario_owned_signature=signature)
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        changed = nontreatment_plan_record(
            benign,
            scenario_owned_signature=[{**signature[0], "x": 1.1}],
        )
        self.assertEqual(canonical_sha256(left), canonical_sha256(changed))
        changed_membership = nontreatment_plan_record(
            benign,
            scenario_owned_signature=[{**signature[0], "type_id": "vehicle.other"}],
        )
        self.assertNotEqual(
            canonical_sha256(left), canonical_sha256(changed_membership)
        )

    def test_all_exact_manifest_twins_have_identical_nontreatment_plan_hash(self) -> None:
        path = Path(
            "phase2_map_sharing/design/phase2_suite_ab_v2/trajectory_group_manifest.csv"
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        selected = [
            row
            for row in rows
            if row["geometry_or_route_id"]
            in {
                "curbside_bus_occluded_pedestrian",
                "occluded_cross_traffic_vehicle",
            }
            and row["split"] == "calibration"
            and "_r00_" in row["trajectory_id"]
        ]
        self.assertEqual(16, len(selected))
        membership = [
            {
                "type_id": "vehicle.ego",
                "role_name": "phase2_helper",
                "motion_mode": "frozen_scenario_contract",
            },
            {
                "type_id": "vehicle.ego",
                "role_name": "phase2_recipient",
                "motion_mode": "frozen_scenario_contract",
            },
            {
                "type_id": "vehicle.occluder",
                "role_name": "phase2_test_occluder",
                "motion_mode": "frozen_scenario_contract",
            },
        ]
        by_group: dict[str, list[str]] = {}
        for row in selected:
            value = nontreatment_plan_record(
                row, scenario_owned_signature=membership
            )
            by_group.setdefault(row["group_id"], []).append(
                canonical_sha256(value)
            )
        self.assertEqual(8, len(by_group))
        self.assertTrue(
            all(len(values) == 2 and len(set(values)) == 1 for values in by_group.values())
        )

    def test_inputs_only_runtime_records_inventory_but_writes_no_logits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Phase2CaptureRuntime(
                Phase2RuntimeConfig(
                    role="helper",
                    trajectory_id="factor_inputs_only",
                    scenario_role="controlled_positive_occlusion",
                    run_dir=root / "role",
                    ready_sentinel=root / "ready.json",
                    capture_start_sentinel=root / "start.json",
                    tick_ready_path=root / "tick.json",
                    heartbeat_path=root / "heartbeat.json",
                    contract_config_path=root / "contract.yaml",
                    retention_start_offset_s=0.0,
                    retention_frame_count=1,
                    retention_tier="inputs_only_window",
                ),
                {
                    "maximum_window_seconds_per_trajectory": 4.0,
                    "maximum_raw_bytes_per_trajectory": 2_000_000,
                    "maximum_raw_bytes_pilot_total": 2_000_000,
                    "minimum_free_bytes_after_reservation": 1,
                },
            )
            runtime._capture_started = True
            runtime._capture_start_clock_s = 10.0
            runtime.record_inputs(
                frame_id=7,
                carla_timestamp=10.0,
                frame_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                radar_tensor=np.zeros((2, 2), dtype=np.float32),
                camera_matrix=np.eye(4, dtype=np.float32),
                camera_intrinsics_input=np.eye(3, dtype=np.float32),
            )
            runtime.record_logits(
                7, {"object": np.zeros((1, 12, 2, 2), dtype=np.float32)}
            )
            runtime.close(status="complete")

            raw = root / "role/retained_inputs"
            self.assertEqual(1, len(list(raw.glob("*_inputs.npz"))))
            self.assertEqual([], list(raw.glob("*_logits.npz")))
            inventory = (root / "role/runtime/raw_inference_inventory.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(",0,", inventory)
            summary = json.loads(
                (root / "role/phase2_runtime_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("inputs_only_window", summary["retention_tier"])
            self.assertEqual(0, summary["logits_files_written"])

    def test_failed_factor_gate_is_raised_only_after_forensic_is_written(self) -> None:
        contract = FactorRuntimeContract.from_plan_row(
            _plan_row(), maximum_surface_clearance_m=3.0
        )
        scenario_summary = {
            "realized_factors": {"realized_hazard_onset_s": 1.0},
            "factor_realization_gate": {
                "schema": "scenesense.phase2_factor_realization_gate.v1",
                "pass": False,
                "failures": ["radial_closing_speed_out_of_band"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            scenario_dir = Path(temporary) / "scenario"
            artifact_path = scenario_dir / "factor_realization.json"

            class _FailingRuntime:
                def factor_result(self) -> dict[str, object]:
                    if not artifact_path.is_file():
                        raise AssertionError("hard gate ran before forensic persistence")
                    raise RuntimeError("factor realization failed as expected")

            with self.assertRaisesRegex(
                RuntimeError, "factor realization failed as expected"
            ):
                _persist_factor_forensic_then_finalize(
                    scenario_dir=scenario_dir,
                    trajectory_id=contract.trajectory_id,
                    contract=contract,
                    nontreatment_plan_sha256="f" * 64,
                    scenario_summary=scenario_summary,
                    scenario_runtime=_FailingRuntime(),
                )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertFalse(artifact["factor_realization_gate"]["pass"])
            self.assertEqual("f" * 64, artifact["nontreatment_plan_sha256"])

    def test_factor_retention_window_contains_onset_and_three_seconds_after(self) -> None:
        config = {
            "factor_realization_runtime": {"enabled": True},
            "clock": {"duration_s": 12.0},
            "capture": {
                "raw_window_duration_s": 4.0,
                "raw_window_start_offset_s_by_geometry_or_route": {},
            },
        }
        for onset in (0.0, 1.0, 2.5, 3.5):
            positive = {
                "geometry_or_route_id": "geometry",
                "requested_hazard_onset_s": onset,
            }
            benign = dict(positive)
            left = _retention_window_for_row(config, positive)
            right = _retention_window_for_row(config, benign)
            self.assertEqual(left, right)
            self.assertLessEqual(left["start_offset_s"], onset)
            self.assertGreaterEqual(left["end_offset_s"], onset + 3.0)
            self.assertEqual("forbidden", left["authored_onset_policy_visibility"])

    def test_exact_factor_pairs_share_onset_aligned_windows(self) -> None:
        plan = build_factor_smoke_plan(load_factor_smoke_config())
        config = {
            "factor_realization_runtime": {"enabled": True},
            "clock": {"duration_s": 12.0},
            "capture": {
                "raw_window_duration_s": 4.0,
                "raw_window_start_offset_s_by_geometry_or_route": {},
            },
        }
        by_group: dict[str, list[dict[str, object]]] = {}
        for row in plan["rows"]:
            runtime_row = {
                "geometry_or_route_id": row["geometry_or_route_id"],
                "requested_hazard_onset_s": row["requested_factor_contract"][
                    "requested_hazard_onset_s"
                ],
            }
            by_group.setdefault(str(row["group_id"]), []).append(
                _retention_window_for_row(config, runtime_row)
            )
        self.assertEqual(8, len(by_group))
        for windows in by_group.values():
            self.assertEqual(2, len(windows))
            self.assertEqual(windows[0], windows[1])
            onset = float(windows[0]["authored_onset_s"])
            self.assertLessEqual(float(windows[0]["start_offset_s"]), onset)
            self.assertGreaterEqual(float(windows[0]["end_offset_s"]), onset + 3.0)

    def test_exact_mixed_tier_storage_estimate_does_not_charge_all_logits(self) -> None:
        input_bytes = 10
        logits_bytes = 7
        estimates, total = _expected_retention_bytes(
            storage={
                "measured_role_input_bytes_per_frame": input_bytes,
                "measured_role_logits_bytes_per_frame": logits_bytes,
            },
            retained_frames_per_role=40,
            tiers=(
                ["inputs_only_window"] * 12
                + ["inputs_plus_logits_window"] * 4
            ),
        )
        expected = 12 * 2 * 40 * input_bytes + 4 * 2 * 40 * (
            input_bytes + logits_bytes
        )
        self.assertEqual(16, len(estimates))
        self.assertEqual(expected, total)
        self.assertLess(total, 16 * 2 * 40 * (input_bytes + logits_bytes))

    def test_geometry_review_resolves_exact_positive_and_rejects_benign(self) -> None:
        smoke, row, contract = _load_factor_review_contract(
            Path("data_collection/configs/phase2_factor_realization_smoke_v1.yaml"),
            "sa_curbside_bus_occluded_pedestrian_high_long_r00_pos",
        )
        self.assertEqual(smoke["stage_id"], "phase2_factor_realization_smoke_v1")
        self.assertEqual(row["trajectory_row_sha256"], contract.trajectory_row_sha256)
        self.assertEqual(0.0, contract.requested["requested_hazard_onset_s"])
        with self.assertRaisesRegex(ValueError, "positive rows only"):
            _load_factor_review_contract(
                Path(
                    "data_collection/configs/phase2_factor_realization_smoke_v1.yaml"
                ),
                "sa_curbside_bus_occluded_pedestrian_high_long_r00_ben",
            )


if __name__ == "__main__":
    unittest.main()
