from __future__ import annotations

import copy
import unittest

from data_collection.validate_phase2_factor_realization_smoke import load_config
from phase2_map_sharing.factor_smoke_runtime_contract import (
    CausalPolicyRuntimeAuditor,
    FEATURE_SOURCE_STAGE,
    FeatureComponent,
    FeatureSample,
    RecipientAvailabilityRecorder,
    analyze_installed_track_guardrails,
    build_recipient_map_target_match,
    build_recipient_available_endpoint,
    summarize_policy_audits,
)


OBS_A = "a" * 64
OBS_B = "b" * 64
FIXTURE_FIELDS = set(load_config()["policy_projection_exercise"]["fixture_backed_fields"])
ABSTRACTED_FIELDS = set(
    load_config()["policy_projection_exercise"]
    .get("local_loopback_transport_abstracted_fields", {})
    .get("fields", {})
)


def samples(names, *, available_at_s=1.0, source_stage=None):
    return {
        name: FeatureSample(
            value=0.0,
            source_stage=(FEATURE_SOURCE_STAGE[name] if source_stage is None else source_stage),
            observed_at_s=0.9,
            available_at_s=available_at_s,
            component_provenance=(
                (
                    FeatureComponent("helper_localization", 0.8, 0.8),
                    FeatureComponent(
                        "recipient_state_transport", 0.9, available_at_s
                    ),
                )
                if source_stage is None
                and FEATURE_SOURCE_STAGE[name] == "derived_relative_kinematics"
                else ()
            ),
            evidence_kind=(
                "preregistered_fixture"
                if name in FIXTURE_FIELDS
                else "local_loopback_transport_abstraction"
                if name in ABSTRACTED_FIELDS
                else "observed"
            ),
        )
        for name in names
    }


def completed_auditor():
    config = load_config()
    auditor = CausalPolicyRuntimeAuditor.from_config(
        config,
        trajectory_id="trajectory-1",
        arm_id="fixed_capture_arm",
        clock_id="carla_simulation_time",
    )
    feature = config["policy_feature_contract"]
    auditor.record_policy_state_exposure(
        sample_at_s=1.0,
        source_track_count=1,
        installed_map_track_count=1,
    )
    placement = samples(feature["placement_features"])
    publication = samples(feature["publication_features"])
    auditor.consume(
        stage="placement",
        decision_id="placement-1",
        decision_at_s=1.0,
        action="SPLIT_FEATURE",
        samples=placement,
    )
    auditor.consume(
        stage="publication",
        decision_id="publication-1",
        decision_at_s=1.0,
        action="PUBLISH_ALL",
        samples=publication,
    )
    auditor.exercise_forbidden_canary(
        stage="placement",
        decision_id="canary-1",
        decision_at_s=1.0,
        action="SPLIT_FEATURE",
        valid_samples=placement,
    )
    return auditor


def availability_record(*, recipient_confirmation_s=2.0):
    recorder = RecipientAvailabilityRecorder(
        trajectory_id="trajectory-1",
        clock_id="carla_simulation_time",
    )
    recorder.register_source_observation(
        source_role="helper",
        source_track_id="helper-track-1",
        observation_sha256=OBS_A,
        observed_at_s=0.8,
    )
    recorder.record_source_confirmation(
        source_role="helper",
        source_track_id="helper-track-1",
        confirmed_at_s=1.0,
    )
    recorder.register_source_observation(
        source_role="recipient",
        source_track_id="recipient-track-1",
        observation_sha256=OBS_B,
        observed_at_s=recipient_confirmation_s - 0.1,
    )
    recorder.record_source_confirmation(
        source_role="recipient",
        source_track_id="recipient-track-1",
        confirmed_at_s=recipient_confirmation_s,
    )
    recorder.record_recipient_local_install(
        local_install_id="local-install-1",
        source_track_id="recipient-track-1",
        source_observation_sha256=OBS_B,
        recipient_map_track_id="map-track-1",
        confirmed_at_s=recipient_confirmation_s,
        installed_at_s=recipient_confirmation_s + 0.05,
        available_at_s=recipient_confirmation_s + 0.1,
    )
    recorder.record_install_attempt(
        attempt_id="attempt-1",
        contribution_id="contribution-1",
        source_role="helper",
        source_track_id="helper-track-1",
        source_observation_sha256=OBS_A,
        published_at_s=1.1,
        attempted_at_s=1.2,
        install_status="accepted",
        recipient_map_track_id="map-track-1",
        installed_at_s=1.2,
        available_at_s=1.3,
    )
    return recorder.to_record()


def usable_target_matches(record):
    attempt = record["install_attempts"][0]
    local = record["recipient_local_installs"][0] if record["recipient_local_installs"] else None
    values = [
        build_recipient_map_target_match(
            trajectory_id=record["trajectory_id"],
            install_kind="helper_install_attempt",
            install_ref_id=attempt["attempt_id"],
            source_role="helper",
            source_track_id=attempt["source_track_id"],
            recipient_map_track_id=attempt["recipient_map_track_id"],
            available_at_s=attempt["available_at_s"],
            canonical_map_state={
                "class_name": "pedestrian",
                "x_m": 1.0,
                "y_m": 2.0,
                "snapshot_at_s": attempt["available_at_s"],
            },
            target_truth_state={
                "class_name": "pedestrian",
                "x_m": 1.0,
                "y_m": 2.0,
                "observed_at_s": attempt["available_at_s"],
            },
            center_gate_m=5.0,
        )
    ]
    if local is not None:
        values.append(
            build_recipient_map_target_match(
                trajectory_id=record["trajectory_id"],
                install_kind="recipient_local_install",
                install_ref_id=local["local_install_id"],
                source_role="recipient",
                source_track_id=local["source_track_id"],
                recipient_map_track_id=local["recipient_map_track_id"],
                available_at_s=local["available_at_s"],
                canonical_map_state={
                    "class_name": "pedestrian",
                    "x_m": 1.0,
                    "y_m": 2.0,
                    "snapshot_at_s": local["available_at_s"],
                },
                target_truth_state={
                    "class_name": "pedestrian",
                    "x_m": 1.0,
                    "y_m": 2.0,
                    "observed_at_s": local["available_at_s"],
                },
                center_gate_m=5.0,
            )
        )
    return values


class ExactPolicyLoaderTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.feature = self.config["policy_feature_contract"]

    def auditor(self):
        return CausalPolicyRuntimeAuditor.from_config(
            self.config,
            trajectory_id="trajectory-1",
            arm_id="fixed_capture_arm",
            clock_id="carla_simulation_time",
        )

    def test_exact_loader_records_both_realized_stages_and_rejected_canary(self):
        record = completed_auditor().to_record()
        self.assertEqual(record["decision_counts"], {"placement": 1, "publication": 1})
        self.assertEqual(record["forbidden_field_canary"]["rejection_count"], 1)
        summary = summarize_policy_audits([record, copy.deepcopy(record)])
        self.assertEqual(summary["trajectory_audit_count"], 2)
        self.assertEqual(summary["placement_decision_count"], 2)

    def test_wider_observation_dict_is_rejected_not_silently_projected(self):
        values = samples(self.feature["placement_features"])
        values["scenario_id"] = FeatureSample("leak", "runtime", 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "exact placement projection mismatch"):
            self.auditor().consume(
                stage="placement",
                decision_id="placement-1",
                decision_at_s=1.0,
                action="SPLIT_FEATURE",
                samples=values,
            )

    def test_nested_evaluation_key_is_rejected(self):
        values = samples(self.feature["placement_features"])
        name = self.feature["placement_features"][0]
        values[name] = FeatureSample(
            {"ground_truth_id": 99},
            FEATURE_SOURCE_STAGE[name],
            0.0,
            0.0,
            evidence_kind="preregistered_fixture",
        )
        with self.assertRaisesRegex(ValueError, "evaluation-only runtime keys"):
            self.auditor().consume(
                stage="placement",
                decision_id="placement-1",
                decision_at_s=1.0,
                action="SPLIT_FEATURE",
                samples=values,
            )

    def test_post_decision_availability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after the consuming decision"):
            self.auditor().consume(
                stage="placement",
                decision_id="placement-1",
                decision_at_s=1.0,
                action="SPLIT_FEATURE",
                samples=samples(self.feature["placement_features"], available_at_s=1.01),
            )

    def test_fixed_projection_action_cannot_drift(self):
        with self.assertRaisesRegex(ValueError, "requires fixed action"):
            self.auditor().consume(
                stage="placement",
                decision_id="placement-1",
                decision_at_s=1.0,
                action="LOCAL_INFER",
                samples=samples(self.feature["placement_features"]),
            )

    def test_evaluation_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden runtime source"):
            self.auditor().consume(
                stage="publication",
                decision_id="publication-1",
                decision_at_s=1.0,
                action="PUBLISH_ALL",
                samples=samples(
                    self.feature["publication_features"],
                    source_stage="evaluation_truth",
                ),
            )

    def test_relative_kinematics_rejects_free_recipient_pose(self):
        values = samples(self.feature["placement_features"])
        name = "helper_recipient_relative_x_m"
        values[name] = FeatureSample(
            value=1.0,
            source_stage="derived_relative_kinematics",
            observed_at_s=0.9,
            available_at_s=0.9,
            component_provenance=(
                FeatureComponent("helper_localization", 0.9, 0.9),
                FeatureComponent("recipient_localization", 0.9, 0.9),
            ),
            evidence_kind="local_loopback_transport_abstraction",
        )
        with self.assertRaisesRegex(ValueError, "recipient_state_transport"):
            self.auditor().consume(
                stage="placement",
                decision_id="placement-1",
                decision_at_s=1.0,
                action="SPLIT_FEATURE",
                samples=values,
            )

    def test_audit_cannot_finalize_without_real_both_stage_decisions(self):
        with self.assertRaisesRegex(ValueError, "placement decisions"):
            self.auditor().to_record()


class RecipientAvailabilityTest(unittest.TestCase):
    def test_margin_uses_recipient_consumer_availability_not_install(self):
        record = availability_record()
        endpoint = build_recipient_available_endpoint(
            record,
            helper_source_track_id="helper-track-1",
            recipient_source_track_id="recipient-track-1",
            recipient_map_track_id="map-track-1",
            evaluation_horizon_s=10.0,
            evaluation_recipient_map_target_matches=usable_target_matches(record),
        )
        self.assertEqual(endpoint["endpoint_status"], "numeric")
        self.assertAlmostEqual(endpoint["recipient_available_confirmed_track_margin_s"], 0.8)
        self.assertEqual(endpoint["helper_track_recipient_install"]["installed_at_s"], 1.2)
        self.assertEqual(endpoint["helper_track_recipient_install"]["available_at_s"], 1.3)
        self.assertEqual(endpoint["transport_mode"], "local_loopback")

    def test_negative_margin_is_a_valid_measured_outcome(self):
        record = availability_record(recipient_confirmation_s=1.1)
        endpoint = build_recipient_available_endpoint(
            record,
            helper_source_track_id="helper-track-1",
            recipient_source_track_id="recipient-track-1",
            recipient_map_track_id="map-track-1",
            evaluation_horizon_s=10.0,
            evaluation_recipient_map_target_matches=usable_target_matches(record),
        )
        self.assertAlmostEqual(endpoint["recipient_available_confirmed_track_margin_s"], -0.1)

    def test_wrong_canonical_map_association_is_not_credited_as_usable(self):
        record = availability_record()
        matches = usable_target_matches(record)
        helper = matches[0]
        helper = build_recipient_map_target_match(
            trajectory_id=record["trajectory_id"],
            install_kind=helper["install_kind"],
            install_ref_id=helper["install_ref_id"],
            source_role=helper["source_role"],
            source_track_id=helper["source_track_id"],
            recipient_map_track_id=helper["recipient_map_track_id"],
            available_at_s=helper["available_at_s"],
            canonical_map_state={
                "class_name": "vehicle",
                "x_m": 30.0,
                "y_m": 30.0,
                "snapshot_at_s": helper["available_at_s"],
            },
            target_truth_state={
                "class_name": "pedestrian",
                "x_m": 1.0,
                "y_m": 2.0,
                "observed_at_s": helper["available_at_s"],
            },
            center_gate_m=5.0,
        )
        endpoint = build_recipient_available_endpoint(
            record,
            helper_source_track_id="helper-track-1",
            recipient_source_track_id="recipient-track-1",
            recipient_map_track_id="map-track-1",
            evaluation_horizon_s=10.0,
            evaluation_recipient_map_target_matches=[helper, matches[1]],
        )
        self.assertEqual(endpoint["endpoint_status"], "cooperative_miss")

    def test_missing_recipient_confirmation_is_typed_right_censoring(self):
        record = availability_record()
        record["source_confirmations"] = [
            item for item in record["source_confirmations"] if item["source_role"] == "helper"
        ]
        record["recipient_local_installs"] = []
        for item in record["recipient_map_tracks"]:
            item["provenance_events"] = [
                event
                for event in item["provenance_events"]
                if event["provenance_kind"] != "recipient_local_install"
            ]
        body = {key: value for key, value in record.items() if key != "provenance_sha256"}
        from phase2_map_sharing.factor_smoke_runtime_contract import canonical_sha256

        record["provenance_sha256"] = canonical_sha256(body)
        endpoint = build_recipient_available_endpoint(
            record,
            helper_source_track_id="helper-track-1",
            recipient_source_track_id=None,
            recipient_map_track_id="map-track-1",
            evaluation_horizon_s=10.0,
            evaluation_recipient_map_target_matches=usable_target_matches(record),
        )
        self.assertEqual(endpoint["endpoint_status"], "ego_right_censored")
        self.assertAlmostEqual(
            endpoint["recipient_available_confirmed_track_margin_lower_bound_s"], 8.7
        )

    def test_confirmed_recipient_without_consumer_install_fails_closed(self):
        record = availability_record()
        record["recipient_local_installs"] = []
        for item in record["recipient_map_tracks"]:
            item["provenance_events"] = [
                event
                for event in item["provenance_events"]
                if event["provenance_kind"] != "recipient_local_install"
            ]
        body = {key: value for key, value in record.items() if key != "provenance_sha256"}
        from phase2_map_sharing.factor_smoke_runtime_contract import canonical_sha256

        record["provenance_sha256"] = canonical_sha256(body)
        with self.assertRaisesRegex(ValueError, "lacks consumer install"):
            build_recipient_available_endpoint(
                record,
                helper_source_track_id="helper-track-1",
                recipient_source_track_id="recipient-track-1",
                recipient_map_track_id="map-track-1",
                evaluation_horizon_s=10.0,
                evaluation_recipient_map_target_matches=usable_target_matches(record),
            )

    def test_non_target_install_is_not_a_protocol_false_install(self):
        record = availability_record()
        report = analyze_installed_track_guardrails(
            record,
            evaluation_truth_match_by_attempt_id={"attempt-1": False},
        )
        self.assertEqual(
            report["metrics"]["protocol_false_recipient_install_rate"]["numerator"], 0
        )
        self.assertEqual(
            report["metrics"]["truth_unmatched_recipient_install_rate"]["rate"],
            1.0,
        )
        self.assertTrue(report["structural_pass"])

    def test_missing_source_provenance_is_protocol_false_and_fails_structure(self):
        record = availability_record()
        record["install_attempts"][0]["source_observation_sha256"] = "c" * 64
        body = {key: value for key, value in record.items() if key != "provenance_sha256"}
        from phase2_map_sharing.factor_smoke_runtime_contract import canonical_sha256

        record["provenance_sha256"] = canonical_sha256(body)
        report = analyze_installed_track_guardrails(record)
        self.assertEqual(
            report["metrics"]["protocol_false_recipient_install_rate"]["numerator"], 1
        )
        self.assertFalse(report["structural_pass"])

    def test_zero_exposure_denominators_are_typed(self):
        record = RecipientAvailabilityRecorder(
            trajectory_id="benign",
            clock_id="carla_simulation_time",
        ).to_record()
        report = analyze_installed_track_guardrails(record)
        for metric in report["metrics"].values():
            self.assertEqual(metric["exposure_status"], "typed_zero_exposure")
            self.assertIsNone(metric["rate"])


if __name__ == "__main__":
    unittest.main()
