from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from phase2_map_sharing.causal_contract import (
    ArmStateToken,
    CausalAuditWriter,
    CausalDecisionAudit,
    CausalField,
    CounterfactualArmRegistry,
    DecisionRecord,
)
from phase2_map_sharing.engine_v2 import RecipientMapEngineV2, propagate_cv
from phase2_map_sharing.pilot_contract import (
    load_and_validate_pilot_config,
    validate_pilot_config,
)
from phase2_map_sharing.retention import (
    RawRetentionBudget,
    RetentionLimits,
    RetentionQuotaExceeded,
)
from phase2_map_sharing.schemas_v2 import (
    MapContributionV2,
    MapObjectObservationV2,
    RecipientStateV2,
    with_exact_payload_bytes_v2,
)


ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = ROOT / "phase2_map_sharing" / "configs" / "paired_causal_pilot_v1.yaml"
REVIEWED_PILOT_CONFIG = (
    ROOT
    / "phase2_map_sharing"
    / "configs"
    / "paired_causal_pilot_reviewed_v1.yaml"
)
HASH = "a" * 64
IDENTITY_COVARIANCE = (
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.04,
    0.0,
    0.0,
    0.0,
    0.0,
    0.04,
)
PROCESS_COVARIANCE = tuple(value * 0.1 for value in IDENTITY_COVARIANCE)


def observation(
    track_id: str = "source-track-1",
    *,
    measured_at_s: float = 0.1,
    x_m: float = 10.0,
    vx_mps: float = 0.0,
    motion_model_id: str = "CV",
    recipient_available_at_s: float | None = None,
) -> MapObjectObservationV2:
    return MapObjectObservationV2(
        source_track_id=track_id,
        tracker_id="causal_tracker",
        tracker_version="1.0",
        class_name="pedestrian",
        x_m=x_m,
        y_m=0.0,
        vx_mps=vx_mps,
        vy_mps=0.0,
        confidence=0.9,
        measured_at_s=measured_at_s,
        state_covariance=IDENTITY_COVARIANCE,
        motion_model_id=motion_model_id,
        process_noise_model_id="cv_q_v1",
        process_noise_covariance_per_s=PROCESS_COVARIANCE,
        validity_horizon_s=2.0,
        occlusion_state="occluded",
        occlusion_source="causal_geometry",
        hazard_score=0.8,
        hazard_source="causal_cv",
        recipient_state_observed_at_s=(
            None if recipient_available_at_s is None else recipient_available_at_s - 0.01
        ),
        recipient_state_available_at_s=recipient_available_at_s,
    )


def contribution(
    *,
    sequence: int = 1,
    clock_id: str = "carla_sim_clock",
    source: str = "helper",
    recipient: str = "ego",
    publication_action: str = "PUBLISH_ALL",
    obj: MapObjectObservationV2 | None = None,
) -> MapContributionV2:
    candidate = MapContributionV2(
        contribution_id=f"{source}:{recipient}:{sequence}",
        source_ue_id=source,
        recipient_ue_id=recipient,
        sequence_number=sequence,
        captured_at_s=0.1,
        placement_decision_id=f"place-{sequence}",
        placement_decision_at_s=0.0,
        inference_completed_at_s=0.2,
        publication_decision_id=f"publish-{sequence}",
        publication_decision_at_s=0.21,
        published_at_s=0.22,
        clock_id=clock_id,
        publication_decision_locus="helper",
        inference_placement="LOCAL_INFER",
        publication_action=publication_action,
        profile_id="local-compact-v1",
        target_fps=10.0,
        model_id="m-prime",
        model_sha256=HASH,
        config_sha256=HASH,
        code_revision="test-revision",
        source_sensor_ids=("rgb-helper", "radar-helper"),
        calibration_ids=("camera-cal-v1", "radar-cal-v1"),
        transport_chunk_bytes=1200,
        chunk_count=1,
        application_payload_bytes=0,
        objects=(obj or observation(),),
    )
    return with_exact_payload_bytes_v2(candidate)


def decision(stage: str = "placement") -> DecisionRecord:
    return DecisionRecord(
        trajectory_id="positive-01",
        arm_id="hazard_only",
        decision_id=f"{stage}-1",
        decision_stage=stage,
        decision_at_s=1.0,
        clock_id="carla_sim_clock",
        action="LOCAL_INFER" if stage == "placement" else "PUBLISH_HAZARD_SUBSET",
    )


def recipient_state(
    *, clock_id: str = "carla_sim_clock", motion_model_id: str = "CV"
) -> RecipientStateV2:
    return RecipientStateV2(
        recipient_ue_id="ego",
        observed_at_s=0.0,
        available_at_s=1.0,
        clock_id=clock_id,
        x_m=0.0,
        y_m=0.0,
        vx_mps=5.0,
        vy_mps=0.0,
        state_covariance=IDENTITY_COVARIANCE,
        motion_model_id=motion_model_id,
        process_noise_model_id="cv_q_v1",
        process_noise_covariance_per_s=PROCESS_COVARIANCE,
    )


def causal_field(
    record: DecisionRecord,
    name: str = "lagged_capacity_estimate_mbps",
    *,
    available_at_s: float = 0.9,
    source_stage: str = "prior_network_estimator",
    value: object = 12.0,
) -> CausalField:
    return CausalField(
        field_name=name,
        value=value,
        source_stage=source_stage,
        observed_at_s=0.8,
        available_at_s=available_at_s,
        consuming_decision_id=record.decision_id,
        consuming_decision_stage=record.decision_stage,
        clock_id=record.clock_id,
        arm_id=record.arm_id,
    )


class PilotConfigTests(unittest.TestCase):
    def test_checked_in_config_is_offline_only_and_complete(self):
        summary = load_and_validate_pilot_config(PILOT_CONFIG)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertFalse(summary["live_run_authorized"])
        self.assertTrue(summary["minimal_transition_core_required_before_rl_go_no_go"])

    def test_live_authorization_and_stale_sensor_contract_fail(self):
        config = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))
        config["authorization"]["carla"] = True
        with self.assertRaisesRegex(ValueError, "must not authorize"):
            validate_pilot_config(config)
        config["authorization"]["carla"] = False
        config["sensor_contract"]["world_hz"] = 20.0
        with self.assertRaisesRegex(ValueError, "world_hz"):
            validate_pilot_config(config)

    def test_reviewed_config_authorizes_only_the_two_trajectory_carla_pilot(self):
        summary = load_and_validate_pilot_config(REVIEWED_PILOT_CONFIG)
        self.assertTrue(summary["live_run_authorized"])
        self.assertEqual(summary["implementation_status"], "reviewed_pilot_only")
        config = yaml.safe_load(REVIEWED_PILOT_CONFIG.read_text(encoding="utf-8"))
        config["authorization"]["oai"] = True
        with self.assertRaisesRegex(ValueError, "CARLA capture only"):
            validate_pilot_config(config)
        config["authorization"]["oai"] = False
        config["review_evidence"]["host_gpu_capacity"][
            "inference_timing_citable"
        ] = True
        with self.assertRaisesRegex(ValueError, "cannot be citable"):
            validate_pilot_config(config)


class SchemaV2Tests(unittest.TestCase):
    def test_exact_round_trip_and_chunk_accounting(self):
        original = contribution()
        encoded = original.to_json_bytes()
        self.assertEqual(original.application_payload_bytes, len(encoded))
        self.assertEqual(MapContributionV2.from_json_bytes(encoded), original)
        self.assertEqual(
            original.to_dict()["resource_uri"],
            "/ss-sm-management/v2/spatial-maps/ego",
        )

    def test_unknown_fields_are_not_silently_ignored(self):
        payload = contribution().to_dict()
        payload["objects"][0]["actor_identifier_typo"] = 42
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            MapContributionV2.from_dict(payload)

    def test_recipient_state_round_trip_is_strict(self):
        original = recipient_state()
        restored = RecipientStateV2.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        payload = original.to_dict()
        payload["future_pose"] = [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            RecipientStateV2.from_dict(payload)

    def test_truth_alias_and_timestamp_inversion_fail_closed(self):
        payload = contribution().to_dict()
        payload["objects"][0]["gt_actor_id"] = 42
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "evaluation-only runtime keys"):
            MapContributionV2.from_json_bytes(encoded)
        inverted = replace(contribution(), placement_decision_at_s=0.3)
        with self.assertRaisesRegex(ValueError, "causal ordering"):
            inverted.validate()

    def test_hazard_subset_requires_causal_recipient_state(self):
        candidate = replace(contribution(), publication_action="PUBLISH_HAZARD_SUBSET")
        with self.assertRaisesRegex(ValueError, "recipient-state provenance"):
            candidate.validate()
        valid = contribution(
            publication_action="PUBLISH_HAZARD_SUBSET",
            obj=observation(recipient_available_at_s=0.2),
        )
        valid.validate()

    def test_covariance_must_be_positive_semidefinite(self):
        bad_covariance = list(IDENTITY_COVARIANCE)
        bad_covariance[0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            replace(observation(), state_covariance=tuple(bad_covariance)).validate()


class EngineV2Tests(unittest.TestCase):
    @staticmethod
    def _retimed_contribution(
        *,
        source: str,
        sequence: int,
        measured_at_s: float,
        captured_at_s: float,
        published_at_s: float,
    ) -> MapContributionV2:
        candidate = replace(
            contribution(source=source, sequence=sequence),
            objects=(
                observation(
                    track_id=f"{source}-track-1",
                    measured_at_s=measured_at_s,
                ),
            ),
            captured_at_s=captured_at_s,
            placement_decision_at_s=captured_at_s - 0.01,
            inference_completed_at_s=captured_at_s + 0.01,
            publication_decision_at_s=published_at_s - 0.01,
            published_at_s=published_at_s,
        )
        return with_exact_payload_bytes_v2(candidate)

    def test_recipient_pose_is_propagated_to_availability_time(self):
        engine = RecipientMapEngineV2("ego", track_ttl_s=2.0)
        self.assertEqual(
            engine.install(contribution(), 0.22, "carla_sim_clock"), "accepted"
        )
        with self.assertRaisesRegex(ValueError, "snapshot and map tracks"):
            engine.snapshot(0.3, "other_clock")
        warning = engine.warnings(recipient_state())[0]
        self.assertAlmostEqual(warning.time_to_closest_approach_s, 1.0, places=6)

    def test_warning_confidence_floor_does_not_filter_map_install_or_association(self):
        engine = RecipientMapEngineV2(
            "ego", warning_emission_confidence_floor=0.8, track_ttl_s=2.0
        )
        low_confidence = contribution(
            obj=replace(observation(), confidence=0.2),
        )
        self.assertEqual(
            engine.install(low_confidence, 0.22, "carla_sim_clock"), "accepted"
        )
        snapshot = engine.snapshot(0.3, "carla_sim_clock")
        self.assertEqual(len(snapshot["tracks"]), 1)
        canonical_track_id = snapshot["tracks"][0]["canonical_track_id"]
        self.assertEqual(engine.warnings(recipient_state()), [])

        high_confidence = contribution(
            sequence=2,
            obj=replace(observation(), confidence=0.9),
        )
        self.assertEqual(
            engine.install(high_confidence, 0.22, "carla_sim_clock"), "accepted"
        )
        snapshot = engine.snapshot(0.3, "carla_sim_clock")
        self.assertEqual(len(snapshot["tracks"]), 1)
        self.assertEqual(
            snapshot["tracks"][0]["canonical_track_id"], canonical_track_id
        )
        self.assertEqual(len(engine.warnings(recipient_state())), 1)

    def test_republished_missed_track_does_not_reset_map_aoi(self):
        engine = RecipientMapEngineV2("ego", track_ttl_s=2.0)
        first = self._retimed_contribution(
            source="helper",
            sequence=1,
            measured_at_s=0.1,
            captured_at_s=0.1,
            published_at_s=0.12,
        )
        republished = self._retimed_contribution(
            source="helper",
            sequence=2,
            measured_at_s=0.1,
            captured_at_s=0.8,
            published_at_s=0.82,
        )
        self.assertEqual(engine.install(first, 0.12, "carla_sim_clock"), "accepted")
        self.assertEqual(
            engine.install(republished, 0.82, "carla_sim_clock"), "accepted"
        )

        warning = engine.warnings(recipient_state())[0]
        self.assertAlmostEqual(warning.map_aoi_s, 0.9)
        self.assertAlmostEqual(warning.latest_capture_at_s, 0.8)
        self.assertAlmostEqual(warning.latest_publish_at_s, 0.82)
        snapshot = engine.snapshot(1.0, "carla_sim_clock")
        self.assertAlmostEqual(snapshot["tracks"][0]["map_aoi_s"], 0.9)

    def test_active_evidence_uses_each_sources_measurement_time(self):
        engine = RecipientMapEngineV2("ego", association_gate_m=2.0, track_ttl_s=0.5)
        helper_first = self._retimed_contribution(
            source="helper",
            sequence=1,
            measured_at_s=0.1,
            captured_at_s=0.1,
            published_at_s=0.12,
        )
        helper_republished = self._retimed_contribution(
            source="helper",
            sequence=2,
            measured_at_s=0.1,
            captured_at_s=0.45,
            published_at_s=0.47,
        )
        ego_fresh = self._retimed_contribution(
            source="ego",
            sequence=1,
            measured_at_s=0.5,
            captured_at_s=0.5,
            published_at_s=0.52,
        )
        self.assertEqual(
            engine.install(helper_first, 0.12, "carla_sim_clock"), "accepted"
        )
        self.assertEqual(
            engine.install(helper_republished, 0.47, "carla_sim_clock"), "accepted"
        )
        self.assertEqual(engine.install(ego_fresh, 0.52, "carla_sim_clock"), "accepted")

        state = replace(recipient_state(), observed_at_s=0.0, available_at_s=0.7)
        warning = engine.warnings(state)[0]
        self.assertAlmostEqual(warning.map_aoi_s, 0.2)
        self.assertEqual(warning.evidence_sources, ("ego",))
        self.assertEqual(warning.evidence_scope, "ego_only")
        snapshot = engine.snapshot(0.7, "carla_sim_clock")
        snapshot_track = snapshot["tracks"][0]
        self.assertNotIn("evidence_sources", snapshot_track)
        self.assertEqual(snapshot_track["active_evidence_sources"], ["ego"])
        self.assertEqual(
            snapshot_track["active_evidence_track_ids_by_source"],
            {"ego": "ego-track-1"},
        )
        self.assertEqual(
            snapshot_track["historical_evidence_sources"], ["ego", "helper"]
        )
        self.assertEqual(
            snapshot_track["historical_evidence_track_ids_by_source"],
            {"ego": "ego-track-1", "helper": "helper-track-1"},
        )

    def test_clock_and_motion_model_mismatches_fail_closed(self):
        engine = RecipientMapEngineV2("ego")
        self.assertEqual(
            engine.install(contribution(), 0.22, "carla_sim_clock"), "accepted"
        )
        self.assertEqual(
            engine.install(
                contribution(sequence=2, clock_id="host_clock"), 0.22, "host_clock"
            ),
            "rejected_clock_mismatch",
        )
        unsupported = contribution(
            sequence=3, obj=observation(motion_model_id="constant_acceleration")
        )
        self.assertEqual(
            engine.install(unsupported, 0.22, "carla_sim_clock"),
            "rejected_unsupported_motion_model",
        )
        with self.assertRaisesRegex(ValueError, "clock domains"):
            engine.warnings(recipient_state(clock_id="other_clock"))
        with self.assertRaisesRegex(ValueError, "only CV"):
            engine.warnings(recipient_state(motion_model_id="CA"))

    def test_cv_covariance_grows_under_process_noise(self):
        _, propagated = propagate_cv(
            (0.0, 0.0, 1.0, 0.0),
            IDENTITY_COVARIANCE,
            PROCESS_COVARIANCE,
            1.0,
        )
        self.assertGreater(propagated[0], IDENTITY_COVARIANCE[0])


class CausalBoundaryTests(unittest.TestCase):
    def test_valid_lagged_field_and_create_only_audit_log(self):
        record = decision()
        audit = CausalDecisionAudit(record, (causal_field(record),))
        audit.validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "causal.jsonl"
            with CausalAuditWriter(path) as writer:
                digest = writer.write(audit)
                self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                CausalAuditWriter(path)

    def test_late_same_frame_truth_shadow_and_stage_fields_are_rejected(self):
        record = decision()
        with self.assertRaisesRegex(ValueError, "after the consuming decision"):
            causal_field(record, available_at_s=1.01).validate_for(record)
        with self.assertRaisesRegex(ValueError, "cannot populate"):
            causal_field(record, source_stage="shadow_inference").validate_for(record)
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            causal_field(record, name="current_inference_result").validate_for(record)
        with self.assertRaisesRegex(ValueError, "cannot produce runtime field"):
            causal_field(
                record,
                name="lagged_capacity_estimate_mbps",
                source_stage="selected_inference",
            ).validate_for(record)
        with self.assertRaisesRegex(ValueError, "evaluation-only runtime keys"):
            causal_field(
                record,
                name="installed_map_summary",
                source_stage="recipient_map",
                value={"nested": {"truth_actor_id": 4}},
            ).validate_for(record)

    def test_current_inference_is_allowed_only_at_publication_after_availability(self):
        record = decision("publication")
        field = causal_field(
            record,
            name="current_inference_result",
            source_stage="selected_inference",
            value={"count": 3},
        )
        field.validate_for(record)

    def test_counterfactual_states_are_deep_copied_and_revision_guarded(self):
        registry = CounterfactualArmRegistry()
        ego = registry.initialize("positive-01", "ego_only", {"tracks": []})
        hazard = registry.initialize("positive-01", "hazard_only", {"tracks": []})
        state = registry.read(ego)
        state["tracks"].append("local-change")
        self.assertEqual(registry.read(ego)["tracks"], [])
        next_ego = registry.commit(ego, state)
        self.assertEqual(registry.read(next_ego)["tracks"], ["local-change"])
        self.assertEqual(registry.read(hazard)["tracks"], [])
        with self.assertRaisesRegex(ValueError, "stale"):
            registry.read(ego)
        with self.assertRaisesRegex(ValueError, "unknown"):
            registry.read(ArmStateToken("positive-01", "send_everything", 0))
        with self.assertRaisesRegex(ValueError, "evaluation-only runtime keys"):
            registry.commit(next_ego, {"gt_id": 9})


class RetentionTests(unittest.TestCase):
    def limits(self) -> RetentionLimits:
        return RetentionLimits(
            maximum_window_seconds_per_trajectory=10.0,
            maximum_raw_bytes_per_trajectory=100,
            maximum_raw_bytes_pilot_total=150,
            minimum_free_bytes_after_reservation=200,
        )

    def test_prewrite_permit_and_trajectory_quota_stop_raw_without_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RawRetentionBudget(
                Path(directory), self.limits(), free_bytes_provider=lambda _: 1000
            )
            budget.preflight(2)
            budget.start_window("positive-01", 0.0)
            permit = budget.authorize_write("positive-01", 90, 1.0)
            budget.record_write_complete(permit)
            with self.assertRaisesRegex(
                RetentionQuotaExceeded, "maximum_trajectory_raw_bytes_reached"
            ):
                budget.authorize_write("positive-01", 11, 2.0)
            summary = budget.summary()
            self.assertEqual(summary["windows"]["positive-01"]["status"], "quota_stopped")
            self.assertFalse(summary["automatic_deletion_performed"])

    def test_preflight_reserves_pilot_total_plus_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RawRetentionBudget(
                Path(directory), self.limits(), free_bytes_provider=lambda _: 349
            )
            with self.assertRaisesRegex(
                RetentionQuotaExceeded, "insufficient_free_space"
            ):
                budget.preflight(2)

    def test_pending_permits_cannot_overbook_total(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RawRetentionBudget(
                Path(directory), self.limits(), free_bytes_provider=lambda _: 1000
            )
            budget.preflight(2)
            budget.start_window("positive-01", 0.0)
            budget.start_window("benign-01", 0.0)
            budget.authorize_write("positive-01", 90, 1.0)
            with self.assertRaisesRegex(
                RetentionQuotaExceeded, "maximum_pilot_raw_bytes_reached"
            ):
                budget.authorize_write("benign-01", 70, 1.0)


if __name__ == "__main__":
    unittest.main()
