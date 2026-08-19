from __future__ import annotations

import copy
import unittest

from data_collection.validate_phase2_factor_realization_smoke import (
    ContractError,
    RESULT_SCHEMA,
    build_plan,
    load_config,
    validate_results,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _result_bundle(config, plan):
    contexts = {row["group_id"]: ("c" * 64) for row in plan["rows"]}
    trajectories = []
    for row in plan["rows"]:
        record = {
            "trajectory_id": row["trajectory_id"],
            "trajectory_row_sha256": row["trajectory_row_sha256"],
            "artifact_manifest_sha256": HASH_A,
            "group_id": row["group_id"],
            "scenario_role": row["scenario_role"],
            "nontreatment_plan_sha256": contexts[row["group_id"]],
            "requested_factors": copy.deepcopy(row["requested_factor_contract"]),
        }
        if row["controlled_hazard_present"]:
            requested = row["requested_factor_contract"]
            record["realized_factors"] = {
                "realized_hazard_onset_s": requested["requested_hazard_onset_s"],
                "realized_helper_speed_mps": requested["requested_helper_speed_mps"],
                "realized_recipient_speed_mps": requested[
                    "requested_recipient_speed_mps"
                ],
                "realized_hazard_actor_speed_mps": requested[
                    "requested_hazard_actor_speed_mps"
                ],
                "realized_onset_driver_speed_mps": requested[
                    "requested_onset_driver_speed_mps"
                ],
                "pre_intervention_radial_closing_speed_mps": requested[
                    "requested_closing_speed_target_mps"
                ],
                "pre_intervention_hazard_proximity_horizon_s": requested[
                    "requested_proximity_horizon_target_s"
                ],
                "pre_intervention_minimum_surface_clearance_m": 1.0,
                "geometry_measurement_basis": requested["geometry_measurement_basis"],
                "closing_speed_measurement_basis": requested[
                    "closing_speed_measurement_basis"
                ],
                "proximity_horizon_measurement_basis": requested[
                    "proximity_horizon_measurement_basis"
                ],
            }
            record["installed_track_endpoint"] = {
                "clock_id": "carla_simulation_time",
                "evaluation_horizon_s": 10.0,
                "endpoint_status": "numeric",
                "evidence_chain_sha256": HASH_B,
                "helper_source_confirmation": {"status": "event", "at_s": 1.0},
                "helper_track_recipient_install": {
                    "status": "event",
                    "at_s": 1.3,
                    "contribution_id": "contribution-1",
                    "source_track_id": "helper:track:1",
                    "recipient_map_track_id": "recipient:map:1",
                    "published_at_s": 1.1,
                    "installed_at_s": 1.2,
                    "available_at_s": 1.3,
                },
                "recipient_own_confirmation": {"status": "event", "at_s": 2.0},
                "recipient_available_confirmed_track_margin_s": 0.7,
            }
        else:
            record["registered_target_absent"] = True
            record["realized_factors_status"] = (
                "not_applicable_matched_benign_registered_target_absent"
            )
            record["factor_reference_trajectory_id"] = (
                row["trajectory_id"].removesuffix("_ben") + "_pos"
            )
        trajectories.append(record)
    feature = config["policy_feature_contract"]
    return {
        "schema": RESULT_SCHEMA,
        "stage_id": config["stage_id"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "policy_feature_contract_sha256": plan["policy_feature_contract_sha256"],
        "warnings_actuated": False,
        "oai_executed": False,
        "downstream_stage_chained": False,
        "policy_feature_projection": {
            "consumer_enforces_exact_projection": True,
            "consumer_code_sha256": "e" * 64,
            "placement_decision_count": 1920,
            "publication_decision_count": 1920,
            "placement_features": list(feature["placement_features"]),
            "publication_features": list(feature["publication_features"]),
        },
        "trajectories": trajectories,
    }


class FactorRealizationSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.plan = build_plan(self.config)

    def test_plan_is_exact_bounded_calibration_tranche(self):
        self.assertEqual(self.plan["trajectory_count"], 16)
        self.assertEqual(self.plan["group_count"], 8)
        self.assertEqual(self.plan["positive_trajectory_count"], 8)
        self.assertEqual(self.plan["benign_trajectory_count"], 8)
        self.assertEqual(self.plan["reuse_if_atomic_pass"], "replicate_0_calibration_tranche")
        self.assertFalse(self.plan["collection_authorized_by_this_tool"])
        self.assertEqual(
            self.plan["source_manifest_schema"],
            "scenesense.phase2_suite_design_manifest.v2",
        )
        self.assertEqual(self.plan["source_design_id"], "phase2_suite_ab_v2")
        cells = {
            (
                row["geometry_or_route_id"],
                row["closing_speed_band"],
                row["time_to_hazard_band"],
            )
            for row in self.plan["rows"]
        }
        self.assertEqual(len(cells), 8)

    def test_policy_feature_contract_rejects_scenario_clock(self):
        drift = copy.deepcopy(self.config)
        drift["policy_feature_contract"]["placement_features"].append("elapsed_s")
        with self.assertRaisesRegex(ContractError, "forbidden policy feature"):
            build_plan(drift)

    def test_policy_feature_contract_rejects_authored_onset_control(self):
        drift = copy.deepcopy(self.config)
        drift["policy_feature_contract"]["placement_features"].append(
            "requested_hazard_onset_s"
        )
        with self.assertRaisesRegex(ContractError, "forbidden policy feature"):
            build_plan(drift)

    def test_valid_typed_result_admits_exact_batch(self):
        result = _result_bundle(self.config, self.plan)
        summary = validate_results(result, self.config, self.plan)
        self.assertEqual(
            summary["verdict"],
            "PASS_ADMIT_EXACT_BATCH_AS_CALIBRATION_TRANCHE",
        )
        self.assertEqual(summary["endpoint_status_counts"], {"numeric": 8})

    def test_out_of_band_realization_fails_atomically(self):
        result = _result_bundle(self.config, self.plan)
        positive = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "controlled_positive_occlusion"
        )
        positive["realized_factors"]["pre_intervention_radial_closing_speed_mps"] = 99.0
        with self.assertRaisesRegex(ContractError, "outside its band"):
            validate_results(result, self.config, self.plan)

    def test_install_availability_not_raw_install_time_defines_margin(self):
        result = _result_bundle(self.config, self.plan)
        positive = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "controlled_positive_occlusion"
        )
        positive["installed_track_endpoint"][
            "recipient_available_confirmed_track_margin_s"
        ] = 0.8
        with self.assertRaisesRegex(ContractError, "margin is inconsistent"):
            validate_results(result, self.config, self.plan)

    def test_recipient_miss_is_typed_as_right_censored(self):
        result = _result_bundle(self.config, self.plan)
        positive = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "controlled_positive_occlusion"
        )
        endpoint = positive["installed_track_endpoint"]
        endpoint["endpoint_status"] = "ego_right_censored"
        endpoint["recipient_own_confirmation"] = {
            "status": "censored",
            "censor_at_s": 10.0,
        }
        endpoint.pop("recipient_available_confirmed_track_margin_s")
        endpoint["recipient_available_confirmed_track_margin_lower_bound_s"] = 8.7
        summary = validate_results(result, self.config, self.plan)
        self.assertEqual(summary["endpoint_status_counts"]["ego_right_censored"], 1)

    def test_benign_context_mismatch_fails(self):
        result = _result_bundle(self.config, self.plan)
        benign = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "matched_benign_negative"
        )
        benign["nontreatment_plan_sha256"] = "d" * 64
        with self.assertRaisesRegex(ContractError, "nontreatment plan differs"):
            validate_results(result, self.config, self.plan)

    def test_realized_basis_must_match_the_geometry_specific_design_basis(self):
        result = _result_bundle(self.config, self.plan)
        positive = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "controlled_positive_occlusion"
        )
        positive["realized_factors"]["geometry_measurement_basis"] = "generic_ttc"
        with self.assertRaisesRegex(ContractError, "differs from its design row"):
            validate_results(result, self.config, self.plan)

    def test_benign_cannot_fabricate_hazard_realization(self):
        result = _result_bundle(self.config, self.plan)
        benign = next(
            row
            for row in result["trajectories"]
            if row["scenario_role"] == "matched_benign_negative"
        )
        benign["realized_factors"] = {}
        with self.assertRaisesRegex(ContractError, "fabricates realized hazard factors"):
            validate_results(result, self.config, self.plan)


if __name__ == "__main__":
    unittest.main()
