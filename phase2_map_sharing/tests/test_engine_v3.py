from __future__ import annotations

import unittest
from dataclasses import replace

from phase2_map_sharing.engine_v3 import (
    FUSION_RULE_ID_V3,
    RecipientMapEngineV3,
)
from phase2_map_sharing.schemas_v2 import (
    MapContributionV2,
    MapObjectObservationV2,
    RecipientStateV2,
    with_exact_payload_bytes_v2,
)


HASH = "b" * 64


def diagonal_covariance(
    position_variance: float = 0.25, velocity_variance: float = 0.04
) -> tuple[float, ...]:
    return (
        position_variance,
        0.0,
        0.0,
        0.0,
        0.0,
        position_variance,
        0.0,
        0.0,
        0.0,
        0.0,
        velocity_variance,
        0.0,
        0.0,
        0.0,
        0.0,
        velocity_variance,
    )


PROCESS_COVARIANCE = diagonal_covariance(0.025, 0.004)


def observation(
    source: str,
    *,
    measured_at_s: float = 0.1,
    x_m: float = 10.0,
    confidence: float = 0.9,
    covariance: tuple[float, ...] | None = None,
) -> MapObjectObservationV2:
    return MapObjectObservationV2(
        source_track_id=f"{source}-track-1",
        tracker_id="causal_tracker",
        tracker_version="1.0",
        class_name="pedestrian",
        x_m=x_m,
        y_m=0.0,
        vx_mps=0.0,
        vy_mps=0.0,
        confidence=confidence,
        measured_at_s=measured_at_s,
        state_covariance=covariance or diagonal_covariance(),
        motion_model_id="CV",
        process_noise_model_id="cv_q_v1",
        process_noise_covariance_per_s=PROCESS_COVARIANCE,
        validity_horizon_s=2.0,
        occlusion_state="occluded",
        occlusion_source="causal_geometry",
        hazard_score=0.8,
        hazard_source="causal_cv",
    )


def contribution(
    source: str,
    *,
    measured_at_s: float = 0.1,
    x_m: float = 10.0,
    confidence: float = 0.9,
    covariance: tuple[float, ...] | None = None,
) -> MapContributionV2:
    captured_at_s = measured_at_s
    published_at_s = measured_at_s + 0.12
    candidate = MapContributionV2(
        contribution_id=f"{source}:ego:1",
        source_ue_id=source,
        recipient_ue_id="ego",
        sequence_number=1,
        captured_at_s=captured_at_s,
        placement_decision_id=f"place-{source}-1",
        placement_decision_at_s=captured_at_s - 0.01,
        inference_completed_at_s=captured_at_s + 0.1,
        publication_decision_id=f"publish-{source}-1",
        publication_decision_at_s=published_at_s - 0.01,
        published_at_s=published_at_s,
        clock_id="carla_sim_clock",
        publication_decision_locus="helper",
        inference_placement="LOCAL_INFER",
        publication_action="PUBLISH_ALL",
        profile_id="local-compact-v1",
        target_fps=10.0,
        model_id="m-prime",
        model_sha256=HASH,
        config_sha256=HASH,
        code_revision="engine-v3-test",
        source_sensor_ids=(f"rgb-{source}", f"radar-{source}"),
        calibration_ids=("camera-cal-v1", "radar-cal-v1"),
        transport_chunk_bytes=1200,
        chunk_count=1,
        application_payload_bytes=0,
        objects=(
            observation(
                source,
                measured_at_s=measured_at_s,
                x_m=x_m,
                confidence=confidence,
                covariance=covariance,
            ),
        ),
    )
    return with_exact_payload_bytes_v2(candidate)


def recipient_state() -> RecipientStateV2:
    return RecipientStateV2(
        recipient_ue_id="ego",
        observed_at_s=0.0,
        available_at_s=1.0,
        clock_id="carla_sim_clock",
        x_m=0.0,
        y_m=0.0,
        vx_mps=5.0,
        vy_mps=0.0,
        state_covariance=diagonal_covariance(),
        motion_model_id="CV",
        process_noise_model_id="cv_q_v1",
        process_noise_covariance_per_s=PROCESS_COVARIANCE,
    )


class EngineV3Tests(unittest.TestCase):
    @staticmethod
    def _install_pair(order: tuple[str, str]) -> RecipientMapEngineV3:
        candidates = {
            "helper": contribution(
                "helper",
                x_m=9.0,
                confidence=0.9,
                covariance=diagonal_covariance(0.1, 0.04),
            ),
            "ego": contribution(
                "ego",
                x_m=11.0,
                confidence=0.5,
                covariance=diagonal_covariance(1.0, 0.04),
            ),
        }
        engine = RecipientMapEngineV3("ego", track_ttl_s=2.0)
        for source in order:
            item = candidates[source]
            assert engine.install(
                item, item.published_at_s, "carla_sim_clock"
            ) == "accepted"
        return engine

    def test_equal_time_fusion_is_source_order_invariant(self):
        helper_first = self._install_pair(("helper", "ego"))
        ego_first = self._install_pair(("ego", "helper"))

        self.assertEqual(
            helper_first.snapshot(0.3, "carla_sim_clock"),
            ego_first.snapshot(0.3, "carla_sim_clock"),
        )
        self.assertEqual(
            helper_first.warnings(recipient_state()),
            ego_first.warnings(recipient_state()),
        )
        snapshot_track = helper_first.snapshot(0.3, "carla_sim_clock")["tracks"][0]
        self.assertEqual(snapshot_track["canonical_track_id"], "map_track_v3_00001")
        self.assertEqual(snapshot_track["active_fusion_sources"], ["ego", "helper"])
        self.assertEqual(snapshot_track["fusion_rule_id"], FUSION_RULE_ID_V3)

    def test_quality_weighting_favors_precise_observation_and_keeps_disagreement(self):
        engine = self._install_pair(("helper", "ego"))
        track = engine.snapshot(0.22, "carla_sim_clock")["tracks"][0]
        self.assertGreater(track["x_m"], 9.0)
        self.assertLess(track["x_m"], 9.2)
        # The mixture moment includes the two-source state disagreement, so it
        # must not become more confident than the precise source by assumption.
        self.assertGreater(track["position_sigma_m"], 0.1**0.5)
        self.assertGreater(track["confidence"], 0.8)

    def test_newer_measurement_supersedes_older_source_in_either_arrival_order(self):
        candidates = {
            "helper": contribution("helper", measured_at_s=0.1, x_m=10.0),
            "ego": contribution("ego", measured_at_s=0.2, x_m=11.0),
        }
        snapshots = []
        for order in (("helper", "ego"), ("ego", "helper")):
            engine = RecipientMapEngineV3("ego", track_ttl_s=2.0)
            for source in order:
                item = candidates[source]
                self.assertEqual(
                    engine.install(item, item.published_at_s, "carla_sim_clock"),
                    "accepted",
                )
            snapshots.append(engine.snapshot(0.3, "carla_sim_clock"))
        self.assertEqual(snapshots[0], snapshots[1])
        track = snapshots[0]["tracks"][0]
        self.assertEqual(track["active_fusion_sources"], ["ego"])
        self.assertEqual(track["historical_evidence_sources"], ["ego", "helper"])

    def test_confirmation_hook_uses_only_available_fusion_metadata(self):
        contexts = []

        def require_two_sources(context):
            contexts.append(context)
            return len(context.active_fusion_sources) >= 2

        engine = RecipientMapEngineV3(
            "ego",
            track_ttl_s=2.0,
            warning_confirmation_policy=require_two_sources,
        )
        helper = contribution("helper", x_m=10.0)
        self.assertEqual(
            engine.install(helper, helper.published_at_s, "carla_sim_clock"),
            "accepted",
        )
        self.assertEqual(engine.warnings(recipient_state()), [])
        ego = contribution("ego", x_m=10.2)
        self.assertEqual(
            engine.install(ego, ego.published_at_s, "carla_sim_clock"),
            "accepted",
        )
        self.assertEqual(len(engine.warnings(recipient_state())), 1)
        self.assertEqual(contexts[-1].active_fusion_sources, ("ego", "helper"))
        self.assertEqual(
            contexts[-1].active_source_confidences,
            (("ego", 0.9), ("helper", 0.9)),
        )
        self.assertFalse(
            any(
                "truth" in name or "actor" in name
                for name in contexts[-1].__dataclass_fields__
            )
        )
        self.assertEqual(engine.counters["confirmation_policy_rejections"], 1)

    def test_confirmation_hook_fails_closed_on_non_boolean_result(self):
        engine = RecipientMapEngineV3(
            "ego", track_ttl_s=2.0, warning_confirmation_policy=lambda _: 1
        )
        item = contribution("helper")
        self.assertEqual(
            engine.install(item, item.published_at_s, "carla_sim_clock"),
            "accepted",
        )
        with self.assertRaisesRegex(TypeError, "must return bool"):
            engine.warnings(recipient_state())


if __name__ == "__main__":
    unittest.main()
