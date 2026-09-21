"""Offline tests for the cooperative-map normalization seam and contracts.

Pure and offline: no CARLA, no CUDA, no OAI, no Docker, no socket, no port, no
live service. Every document is a canonical fixture on disk.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

from spatial_map_coop.coop_input_v1 import (
    CLOCK_CARLA_SIM_NS,
    CLOCK_UNIX_WALL_NS,
    INCOMPATIBLE_CLOCKS,
    NOT_AVAILABLE,
    POSE_AVAILABLE,
    POSE_UNAVAILABLE_NOT_IN_PROTOCOL,
    POSE_UNAVAILABLE_NO_SIDECAR,
    TRANSPORT_DIRECT_UDP,
    TRANSPORT_HTTP_DEMO,
    ClockStamp,
    CoopIdentityConfig,
    CoopInputError,
    MetricsRegistry,
    PoseSidecarIndex,
    RollingWindow,
    association_reduction,
    attach_pose,
    build_route_contract,
    cross_clock_age_ms,
    demo_transport_labels,
    direct_timing_for,
    follow_view_contract,
    from_direct_object_map_update,
    from_live_view_update,
    load_canonical_route_b,
    parse_pose_sidecar,
    reference_pin_report,
)
from spatial_map_coop.coop_input_v1 import live_metrics_v1, route_contract_v1
from spatial_map_coop.coop_input_v1.pose_sidecar_v1 import POSE_UNAVAILABLE_LABEL


FIXTURES = Path(__file__).resolve().parent.parent / "coop_input_v1" / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def direct_identity() -> CoopIdentityConfig:
    return CoopIdentityConfig.from_documents(fixture("identity_bindings_direct_v1.json"))


def demo_identity() -> CoopIdentityConfig:
    return CoopIdentityConfig.from_documents(fixture("identity_bindings_http_demo_v1.json"))


class DirectAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.update = fixture("direct_object_map_update_a_v1.json")
        self.feedback = fixture("direct_map_feedback_a_v1.json")
        self.identity = direct_identity()

    def normalize(self, **kwargs):
        return from_direct_object_map_update(
            self.update, identity=self.identity, **kwargs
        )

    def test_production_document_normalizes_without_changing_p025_records(self):
        original = copy.deepcopy(self.update)
        batch = self.normalize()
        # Every record is byte-identical, key for key, value for value.
        self.assertEqual([dict(item) for item in batch.records], original["records"])
        self.assertEqual(batch.record_count, original["record_count"])
        self.assertEqual(batch.record_scalar_count, original["record_scalar_count"])
        # And the source document itself was not touched.
        self.assertEqual(self.update, original)

    def test_normalized_batch_retains_every_required_field(self):
        batch = self.normalize(
            received_timestamp_ns=self.update["capture_timestamp_ns"] + 200_000_000,
            map_feedback=self.feedback,
        )
        self.assertEqual(batch.transport, TRANSPORT_DIRECT_UDP)
        self.assertEqual(batch.run_id, self.update["run_id"])
        self.assertEqual(batch.cell_id, self.update["cell_id"])
        self.assertEqual(batch.stream_id, self.update["stream_id"])
        self.assertEqual(batch.frame_id, self.update["frame_id"])
        self.assertEqual(batch.sequence_id, self.update["sequence_id"])
        self.assertEqual(batch.action_id, self.update["action_id"])
        self.assertEqual(batch.profile_id, self.update["profile_id"])
        self.assertEqual(batch.decoder_identity, self.update["decoder_identity"])
        self.assertEqual(batch.ue_id, "ue-a")
        self.assertEqual(batch.session_id, "two-ue-live-action50-v1")
        self.assertEqual(batch.capture.clock_domain, CLOCK_UNIX_WALL_NS)
        self.assertEqual(batch.receipt.clock_domain, CLOCK_UNIX_WALL_NS)
        self.assertTrue(batch.edge_timing.available)
        self.assertEqual(batch.map_terminal.outcome, "RESULT_INSTALLED")
        self.assertEqual(batch.map_terminal.agent_credit, "CREDIT_INSTALLED_ON_TIME")
        self.assertEqual(len(batch.payload_sha256), 64)
        self.assertEqual(len(batch.records_sha256), 64)
        self.assertEqual(batch.identity_binding_source,
                         "EXPLICIT_CONFIGURED_IDENTITY_BINDING_V1")

    def test_production_carla_timestamp_is_never_promoted_to_a_capture_clock(self):
        # The live direct runtime passes carla_timestamp=0.0, so the document has
        # no usable simulation clock. The adapter must not adopt it.
        self.assertEqual(self.update["carla_timestamp"], 0.0)
        batch = self.normalize()
        self.assertEqual(batch.capture.clock_domain, CLOCK_UNIX_WALL_NS)
        self.assertNotEqual(batch.capture.clock_domain, CLOCK_CARLA_SIM_NS)

    def test_receipt_is_absent_rather_than_zero_when_not_supplied(self):
        batch = self.normalize()
        self.assertIsNone(batch.receipt)
        self.assertFalse(batch.as_dict()["capture_receipt_clocks_compatible"])

    def test_protocol_v1_never_yields_a_pose(self):
        batch = self.normalize()
        self.assertIsNone(batch.pose)
        self.assertEqual(batch.pose_state, POSE_UNAVAILABLE_NOT_IN_PROTOCOL)

    # ---- malformed documents must fail closed ---------------------------
    def test_schema_drift_fails(self):
        broken = copy.deepcopy(self.update)
        broken["schema"] = "splitfusion_direct_object_map_update.v2"
        with self.assertRaisesRegex(CoopInputError, "schema drift"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_protocol_version_drift_fails(self):
        broken = copy.deepcopy(self.update)
        broken["protocol_version"] = 2
        with self.assertRaisesRegex(CoopInputError, "protocol version drift"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_each_mandatory_identity_field_is_required(self):
        for name in (
            "run_id", "cell_id", "stream_id", "frame_id",
            "action_id", "profile_id", "capture_timestamp_ns",
        ):
            with self.subTest(field=name):
                broken = copy.deepcopy(self.update)
                del broken[name]
                with self.assertRaises(CoopInputError):
                    from_direct_object_map_update(broken, identity=self.identity)
                empty = copy.deepcopy(self.update)
                empty[name] = ""
                with self.assertRaises(CoopInputError):
                    from_direct_object_map_update(empty, identity=self.identity)

    def test_record_count_disagreement_fails(self):
        broken = copy.deepcopy(self.update)
        broken["record_count"] = 5
        with self.assertRaisesRegex(CoopInputError, "record_count"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_record_scalar_count_disagreement_fails(self):
        broken = copy.deepcopy(self.update)
        broken["record_scalar_count"] = 9999
        with self.assertRaisesRegex(CoopInputError, "record_scalar_count"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_non_finite_record_value_fails(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                broken = copy.deepcopy(self.update)
                broken["records"][0]["world_x"] = bad
                with self.assertRaisesRegex(CoopInputError, "non-finite"):
                    from_direct_object_map_update(broken, identity=self.identity)

    def test_record_identity_mismatch_fails(self):
        broken = copy.deepcopy(self.update)
        broken["records"][0]["frame_id"] = broken["frame_id"] + 1
        with self.assertRaisesRegex(CoopInputError, "frame identity mismatch"):
            from_direct_object_map_update(broken, identity=self.identity)
        broken = copy.deepcopy(self.update)
        broken["records"][0]["stream_id"] = "someone-else"
        with self.assertRaisesRegex(CoopInputError, "stream identity mismatch"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_bad_timestamps_fail(self):
        broken = copy.deepcopy(self.update)
        broken["capture_timestamp_ns"] = 0
        with self.assertRaises(CoopInputError):
            from_direct_object_map_update(broken, identity=self.identity)
        broken = copy.deepcopy(self.update)
        broken["capture_timestamp"] = 1.0  # disagrees with capture_timestamp_ns
        with self.assertRaisesRegex(CoopInputError, "capture_timestamp disagrees"):
            from_direct_object_map_update(broken, identity=self.identity)
        broken = copy.deepcopy(self.update)
        broken["ack_timeout_at"] = broken["service_deadline_at"] - 1.0
        with self.assertRaisesRegex(CoopInputError, "ACK timeout"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_action_outside_the_catalog_fails(self):
        broken = copy.deepcopy(self.update)
        broken["action_id"] = 72
        with self.assertRaisesRegex(CoopInputError, "outside the catalog"):
            from_direct_object_map_update(broken, identity=self.identity)

    def test_upstream_validator_runs_first_and_can_refuse(self):
        calls = []

        def validator(document):
            calls.append(document["frame_id"])
            raise CoopInputError("production validator refused this document")

        with self.assertRaisesRegex(CoopInputError, "production validator refused"):
            from_direct_object_map_update(
                self.update, identity=self.identity, upstream_validator=validator
            )
        self.assertEqual(calls, [self.update["frame_id"]])

    # ---- map feedback identity ------------------------------------------
    def test_map_feedback_identity_mismatch_is_refused(self):
        for field, value in (
            ("run_id", "other-run"),
            ("cell_id", "other-cell"),
            ("stream_id", "two-ue/ue-b"),
            ("frame_id", 999),
        ):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.feedback)
                broken[field] = value
                with self.assertRaisesRegex(CoopInputError, "identity mismatch"):
                    from_direct_object_map_update(
                        self.update, identity=self.identity, map_feedback=broken
                    )

    def test_installed_feedback_must_carry_an_install_timestamp(self):
        broken = copy.deepcopy(self.feedback)
        broken["install_timestamp"] = ""
        with self.assertRaisesRegex(CoopInputError, "install_timestamp"):
            from_direct_object_map_update(
                self.update, identity=self.identity, map_feedback=broken
            )

    def test_non_installed_feedback_must_not_carry_an_install_timestamp(self):
        broken = copy.deepcopy(self.feedback)
        broken["outcome"] = "SUPERSEDED_PENDING"
        broken["agent_credit"] = "CREDIT_SUPERSEDED_BY_FRESHER"
        with self.assertRaisesRegex(CoopInputError, "only RESULT_INSTALLED"):
            from_direct_object_map_update(
                self.update, identity=self.identity, map_feedback=broken
            )

    def test_unregistered_outcome_or_credit_is_refused(self):
        broken = copy.deepcopy(self.feedback)
        broken["outcome"] = "RESULT_PROBABLY_FINE"
        with self.assertRaisesRegex(CoopInputError, "unregistered outcome"):
            from_direct_object_map_update(
                self.update, identity=self.identity, map_feedback=broken
            )
        broken = copy.deepcopy(self.feedback)
        broken["agent_credit"] = "CREDIT_VIBES"
        with self.assertRaisesRegex(CoopInputError, "unregistered agent credit"):
            from_direct_object_map_update(
                self.update, identity=self.identity, map_feedback=broken
            )


class IdentityConfigTests(unittest.TestCase):
    def test_configured_mapping_is_mandatory(self):
        update = fixture("direct_object_map_update_a_v1.json")
        empty_ok = CoopIdentityConfig.from_documents(
            {
                "schema": "scenesense.coop_identity_binding.v1",
                "bindings": [
                    {
                        "run_id": "a-different-run", "cell_id": "a-different-cell",
                        "stream_id": "two-ue/ue-a", "ue_id": "ue-a",
                        "session_id": "s",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(CoopInputError, "no configured cooperative-map identity"):
            from_direct_object_map_update(update, identity=empty_ok)

    def test_ue_is_never_inferred_from_stream_naming(self):
        # 'two-ue/ue-a' contains 'ue-a', but that must not bind anything.
        identity = CoopIdentityConfig.from_documents(
            {
                "schema": "scenesense.coop_identity_binding.v1",
                "bindings": [
                    {"run_id": None, "cell_id": None, "stream_id": "totally-other",
                     "ue_id": "ue-a", "session_id": "s"}
                ],
            }
        )
        with self.assertRaises(CoopInputError):
            identity.resolve(run_id=None, cell_id=None, stream_id="two-ue/ue-a")

    def test_run_and_cell_are_part_of_the_key(self):
        identity = direct_identity()
        row = identity.resolve(
            run_id="coop_fixture_run_v1",
            cell_id="coop_fixture_cell_a20_favorable_stable",
            stream_id="two-ue/ue-a",
        )
        self.assertEqual(row.ue_id, "ue-a")
        with self.assertRaises(CoopInputError):
            identity.resolve(run_id=None, cell_id=None, stream_id="two-ue/ue-a")

    def test_duplicate_stream_or_ue_binding_is_refused(self):
        rows = [
            {"run_id": None, "cell_id": None, "stream_id": "s1",
             "ue_id": "ue-a", "session_id": "x"},
            {"run_id": None, "cell_id": None, "stream_id": "s1",
             "ue_id": "ue-b", "session_id": "x"},
        ]
        with self.assertRaisesRegex(CoopInputError, "duplicate identity binding"):
            CoopIdentityConfig.from_documents(rows)
        rows[1]["stream_id"] = "s2"
        rows[1]["ue_id"] = "ue-a"
        with self.assertRaisesRegex(CoopInputError, "already bound"):
            CoopIdentityConfig.from_documents(rows)

    def test_unknown_fields_and_empty_bindings_are_refused(self):
        with self.assertRaisesRegex(CoopInputError, "unknown fields"):
            CoopIdentityConfig.from_documents(
                [{"stream_id": "s", "ue_id": "u", "session_id": "x", "vehicle": "car"}]
            )
        with self.assertRaisesRegex(CoopInputError, "at least one identity binding"):
            CoopIdentityConfig.from_documents([])

    def test_cli_pairs_require_an_explicit_session(self):
        config = CoopIdentityConfig.from_cli_pairs(
            ["two-ue/ue-a=ue-a", "two-ue/ue-b=ue-b"], session_id="drive-1"
        )
        self.assertEqual(config.stream_count, 2)
        self.assertEqual(config.ue_ids, ("ue-a", "ue-b"))
        with self.assertRaises(CoopInputError):
            CoopIdentityConfig.from_cli_pairs(["bad-pair"], session_id="drive-1")
        with self.assertRaises(CoopInputError):
            CoopIdentityConfig.from_cli_pairs(["a=b"], session_id="")


class HttpDemoAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.envelope = fixture("http_live_view_update_a_v1.json")
        self.identity = demo_identity()

    def test_demo_envelope_normalizes_with_a_simulation_capture_clock(self):
        batch = from_live_view_update(self.envelope, identity=self.identity)
        self.assertEqual(batch.transport, TRANSPORT_HTTP_DEMO)
        self.assertEqual(batch.capture.clock_domain, CLOCK_CARLA_SIM_NS)
        self.assertIsNone(batch.run_id)
        self.assertIsNone(batch.cell_id)
        self.assertIsNone(batch.protocol_version)
        self.assertEqual(batch.ue_id, "ue-a")

    def test_demo_records_are_unchanged(self):
        original = copy.deepcopy(self.envelope)
        batch = from_live_view_update(self.envelope, identity=self.identity)
        self.assertEqual(
            [dict(item) for item in batch.records],
            original["payload"]["object_map_update"]["records"],
        )
        self.assertEqual(self.envelope, original)

    def test_demo_envelope_supplies_a_measured_pose_and_fov(self):
        batch = from_live_view_update(self.envelope, identity=self.identity)
        self.assertEqual(batch.pose_state, POSE_AVAILABLE)
        self.assertEqual(batch.pose.horizontal_fov_deg, 120.0)
        view = follow_view_contract(batch, radius_m=40.0)
        self.assertTrue(view["available"])
        self.assertEqual(view["radius_m"], 40.0)
        self.assertEqual(view["fov_deg"], 120.0)

    def test_demo_has_no_edge_timing_and_no_terminal(self):
        batch = from_live_view_update(self.envelope, identity=self.identity)
        self.assertFalse(batch.edge_timing.available)
        self.assertFalse(batch.map_terminal.available)
        self.assertIsNone(batch.map_terminal.outcome)

    def test_envelope_identity_must_agree_with_the_configured_binding(self):
        broken = copy.deepcopy(self.envelope)
        broken["ue_id"] = "ue-b"
        with self.assertRaisesRegex(CoopInputError, "envelope claims"):
            from_live_view_update(broken, identity=self.identity)
        broken = copy.deepcopy(self.envelope)
        broken["session_id"] = "some-other-session"
        with self.assertRaisesRegex(CoopInputError, "session identity"):
            from_live_view_update(broken, identity=self.identity)

    def test_wrong_clock_domain_is_refused(self):
        broken = copy.deepcopy(self.envelope)
        broken["clock_domain"] = "unix_wall_ns"
        with self.assertRaisesRegex(CoopInputError, "carla_simulation_ns"):
            from_live_view_update(broken, identity=self.identity)

    def test_nested_identity_mismatch_is_refused(self):
        broken = copy.deepcopy(self.envelope)
        broken["payload"]["object_map_update"]["frame_id"] += 1
        with self.assertRaisesRegex(CoopInputError, "frame identity mismatch"):
            from_live_view_update(broken, identity=self.identity)

    def test_missing_pose_or_fov_is_refused_rather_than_defaulted(self):
        broken = copy.deepcopy(self.envelope)
        del broken["ego_pose"]
        with self.assertRaisesRegex(CoopInputError, "ego pose"):
            from_live_view_update(broken, identity=self.identity)
        broken = copy.deepcopy(self.envelope)
        del broken["horizontal_fov_deg"]
        with self.assertRaisesRegex(CoopInputError, "horizontal_fov_deg"):
            from_live_view_update(broken, identity=self.identity)


class PoseSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.update = fixture("direct_object_map_update_a_v1.json")
        self.sidecar = fixture("pose_sidecar_a_v1.json")
        self.batch = from_direct_object_map_update(
            self.update, identity=direct_identity()
        )

    def test_matching_sidecar_supplies_the_pose(self):
        bound = attach_pose(self.batch, self.sidecar)
        self.assertEqual(bound.pose_state, POSE_AVAILABLE)
        self.assertEqual(bound.pose.horizontal_fov_deg, 120.0)
        self.assertEqual(bound.pose.source, "COOP_POSE_SIDECAR_V1")
        # The original batch is untouched: with_pose returns a new object.
        self.assertIsNone(self.batch.pose)
        self.assertEqual(self.batch.pose_state, POSE_UNAVAILABLE_NOT_IN_PROTOCOL)

    def test_sidecar_identity_mismatch_is_refused_on_every_key(self):
        for field, value in (
            ("run_id", "other-run"),
            ("cell_id", "other-cell"),
            ("stream_id", "two-ue/ue-b"),
            ("frame_id", 4097),
            ("capture_timestamp_ns", self.update["capture_timestamp_ns"] + 1),
        ):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.sidecar)
                broken[field] = value
                with self.assertRaisesRegex(CoopInputError, "identity mismatch"):
                    attach_pose(self.batch, broken)

    def test_absent_pose_cannot_silently_produce_an_fov(self):
        index = PoseSidecarIndex()  # deliberately empty
        resolved = index.resolve(self.batch)
        self.assertIsNone(resolved.pose)
        self.assertEqual(resolved.pose_state, POSE_UNAVAILABLE_NO_SIDECAR)
        view = follow_view_contract(resolved)
        self.assertFalse(view["available"])
        self.assertIsNone(view["fov_deg"])
        self.assertIsNone(view["bounds"])
        self.assertIsNone(view["ego_pose"])
        self.assertEqual(view["label"], POSE_UNAVAILABLE_LABEL)
        self.assertIn("pose unavailable", view["label"])

    def test_sidecar_index_resolves_only_the_exact_identity(self):
        index = PoseSidecarIndex([self.sidecar])
        self.assertEqual(len(index), 1)
        self.assertEqual(index.resolve(self.batch).pose_state, POSE_AVAILABLE)
        other = from_direct_object_map_update(
            fixture("direct_object_map_update_b_v1.json"), identity=direct_identity()
        )
        self.assertEqual(index.resolve(other).pose_state, POSE_UNAVAILABLE_NO_SIDECAR)

    def test_ground_truth_provenance_is_refused(self):
        for provenance in (
            "CARLA_ACTOR_GROUND_TRUTH", "CARLA_WORLD_SNAPSHOT", "SIMULATOR_TRUTH"
        ):
            with self.subTest(provenance=provenance):
                broken = copy.deepcopy(self.sidecar)
                broken["pose_provenance"] = provenance
                with self.assertRaisesRegex(CoopInputError, "forbidden ground-truth"):
                    parse_pose_sidecar(broken)

    def test_sidecar_schema_and_unknown_fields_are_strict(self):
        broken = copy.deepcopy(self.sidecar)
        broken["schema"] = "scenesense.coop_pose_sidecar.v2"
        with self.assertRaisesRegex(CoopInputError, "schema drift"):
            parse_pose_sidecar(broken)
        broken = copy.deepcopy(self.sidecar)
        broken["extra_field"] = 1
        with self.assertRaisesRegex(CoopInputError, "unknown fields"):
            parse_pose_sidecar(broken)
        broken = copy.deepcopy(self.sidecar)
        broken["ego_world"]["x"] = float("nan")
        with self.assertRaisesRegex(CoopInputError, "finite"):
            parse_pose_sidecar(broken)
        broken = copy.deepcopy(self.sidecar)
        broken["horizontal_fov_deg"] = 0.0
        with self.assertRaisesRegex(CoopInputError, "horizontal_fov_deg"):
            parse_pose_sidecar(broken)

    def test_conflicting_sidecars_for_one_identity_are_refused(self):
        index = PoseSidecarIndex([self.sidecar])
        conflicting = copy.deepcopy(self.sidecar)
        conflicting["ego_world"]["x"] = 99.0
        with self.assertRaisesRegex(CoopInputError, "conflicting pose sidecars"):
            index.add(conflicting)


class CrossClockAndTimingTests(unittest.TestCase):
    def test_cross_clock_latency_is_a_string_not_a_number(self):
        capture = ClockStamp(612_400_000_000, CLOCK_CARLA_SIM_NS)
        receipt = ClockStamp(1_789_000_000_400_000_000, CLOCK_UNIX_WALL_NS)
        result = cross_clock_age_ms(capture, receipt)
        self.assertIsInstance(result, str)
        self.assertEqual(result, INCOMPATIBLE_CLOCKS)
        self.assertNotIsInstance(result, float)

    def test_same_clock_latency_is_numeric(self):
        capture = ClockStamp(1_000_000_000, CLOCK_UNIX_WALL_NS)
        receipt = ClockStamp(1_100_000_000, CLOCK_UNIX_WALL_NS)
        self.assertAlmostEqual(cross_clock_age_ms(capture, receipt), 100.0)

    def test_absent_stamp_is_not_available_rather_than_zero(self):
        self.assertEqual(cross_clock_age_ms(None, None), NOT_AVAILABLE)
        self.assertEqual(
            cross_clock_age_ms(ClockStamp(1, CLOCK_UNIX_WALL_NS), None), NOT_AVAILABLE
        )

    def test_demo_transport_labels_never_claim_a_radio(self):
        labels = demo_transport_labels()
        self.assertEqual(labels["cross_clock_wall_aoi"], INCOMPATIBLE_CLOCKS)
        self.assertIn("demo-control delivery time", labels["http_post_duration"])
        self.assertEqual(labels["model_execution"], "localhost model path")
        blob = " ".join(labels.values()).lower()
        for banned in ("5g latency", "radio latency", "network latency", "uplink latency"):
            self.assertNotIn(banned, blob)

    def test_direct_timing_uses_only_compatible_stamps(self):
        batch = from_direct_object_map_update(
            fixture("direct_object_map_update_a_v1.json"),
            identity=direct_identity(),
            map_feedback=fixture("direct_map_feedback_a_v1.json"),
        )
        timing = direct_timing_for(batch)
        self.assertIsInstance(timing.tail_complete_to_install_ms, float)
        self.assertIsInstance(timing.publish_start_to_install_ms, float)
        self.assertIsInstance(timing.capture_to_install_aoi_ms, float)
        # 0.194 s install offset against a 0.181 s tail stamp -> ~13 ms.
        self.assertAlmostEqual(timing.tail_complete_to_install_ms, 13.0, places=2)
        self.assertAlmostEqual(timing.publish_start_to_install_ms, 11.0, places=2)
        self.assertAlmostEqual(timing.capture_to_install_aoi_ms, 194.0, places=1)

    def test_direct_timing_is_all_na_without_a_terminal(self):
        batch = from_direct_object_map_update(
            fixture("direct_object_map_update_a_v1.json"), identity=direct_identity()
        )
        timing = direct_timing_for(batch)
        self.assertEqual(timing.tail_complete_to_install_ms, NOT_AVAILABLE)
        self.assertEqual(timing.publish_start_to_install_ms, NOT_AVAILABLE)
        self.assertEqual(timing.capture_to_install_aoi_ms, NOT_AVAILABLE)

    def test_direct_timing_is_na_for_the_demo_transport(self):
        batch = from_live_view_update(
            fixture("http_live_view_update_a_v1.json"), identity=demo_identity()
        )
        timing = direct_timing_for(batch)
        for value in timing.as_dict().values():
            self.assertEqual(value, NOT_AVAILABLE)

    def test_boundary_inversion_is_excluded_not_clamped_to_zero(self):
        update = fixture("direct_object_map_update_a_v1.json")
        feedback = fixture("direct_map_feedback_a_v1.json")
        # An install stamp EARLIER than the tail stamp is a boundary inversion.
        feedback["install_timestamp"] = (
            update["capture_timestamp_ns"] / 1e9 + 0.100
        )
        batch = from_direct_object_map_update(
            update, identity=direct_identity(), map_feedback=feedback
        )
        timing = direct_timing_for(batch)
        self.assertEqual(timing.tail_complete_to_install_ms, NOT_AVAILABLE)
        self.assertNotEqual(timing.tail_complete_to_install_ms, 0.0)

    def test_every_latency_label_names_its_boundary(self):
        for label in (
            live_metrics_v1.LABEL_TAIL_TO_INSTALL,
            live_metrics_v1.LABEL_PUBLISH_TO_INSTALL,
            live_metrics_v1.LABEL_CAPTURE_TO_INSTALL,
        ):
            self.assertIn("->", label)
        for label in (
            live_metrics_v1.LABEL_TAIL_TO_INSTALL,
            live_metrics_v1.LABEL_PUBLISH_TO_INSTALL,
            live_metrics_v1.LABEL_CAPTURE_TO_INSTALL,
            live_metrics_v1.LABEL_DEMO_POST,
            live_metrics_v1.LABEL_LOCALHOST_MODEL,
            live_metrics_v1.LABEL_SNAPSHOT_BUILD,
        ):
            lowered = label.lower()
            for banned in live_metrics_v1.FORBIDDEN_LABELS:
                self.assertNotIn(banned, lowered)

    def test_every_forbidden_label_entry_is_lowercase(self):
        # The guard lowercases the candidate, so a mixed-case entry here would
        # be dead code that silently permits the label it was meant to ban.
        for entry in live_metrics_v1.FORBIDDEN_LABELS:
            self.assertEqual(entry, entry.lower())

    def test_a_forbidden_label_cannot_be_registered(self):
        for banned in (
            "network latency", "5G latency", "5g LATENCY",
            "Radio Latency", "total network latency (residual)",
        ):
            with self.subTest(label=banned):
                with self.assertRaisesRegex(ValueError, "prohibited term"):
                    RollingWindow(banned)


class RollingWindowTests(unittest.TestCase):
    def test_window_is_bounded_and_keeps_the_newest_samples(self):
        window = RollingWindow("edge tail-complete -> map-install", capacity=5)
        for value in range(1, 21):
            window.observe(float(value))
        self.assertEqual(len(window), 5)
        self.assertEqual(window.samples(), (16.0, 17.0, 18.0, 19.0, 20.0))

    def test_default_window_is_the_registered_120(self):
        self.assertEqual(live_metrics_v1.DEFAULT_WINDOW, 120)
        self.assertEqual(RollingWindow("x -> y").capacity, 120)

    def test_percentiles_are_deterministic_nearest_rank(self):
        window = RollingWindow("x -> y", capacity=10)
        for value in (5, 1, 9, 3, 7, 2, 8, 4, 6, 10):
            window.observe(float(value))
        first = window.summary()
        second = window.summary()
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.p50, 5.0)
        self.assertEqual(first.p95, 10.0)
        self.assertEqual(first.current, 10.0)
        # Every reported percentile is an actually observed sample.
        self.assertIn(first.p50, window.samples())
        self.assertIn(first.p95, window.samples())

    def test_empty_window_reports_na_not_zero(self):
        summary = RollingWindow("x -> y").summary()
        self.assertEqual(summary.count, 0)
        self.assertEqual(summary.current, NOT_AVAILABLE)
        self.assertEqual(summary.p50, NOT_AVAILABLE)
        self.assertEqual(summary.p95, NOT_AVAILABLE)

    def test_absent_and_non_finite_samples_are_dropped_not_zeroed(self):
        window = RollingWindow("x -> y")
        for value in (None, "N/A", float("nan"), float("inf"), True):
            self.assertFalse(window.observe(value))
        self.assertEqual(len(window), 0)
        self.assertTrue(window.observe(1.5))
        self.assertEqual(len(window), 1)

    def test_latency_windows_stay_empty_without_timing_evidence(self):
        registry = MetricsRegistry(capacity=8)
        demo = from_live_view_update(
            fixture("http_live_view_update_a_v1.json"), identity=demo_identity()
        )
        registry.observe_direct_batch(demo)
        report = registry.report(now_wall_s=1_789_000_000.5)
        for window in report["latency_windows"]:
            self.assertEqual(window["count"], 0)
            self.assertEqual(window["current"], NOT_AVAILABLE)
            self.assertEqual(window["p50"], NOT_AVAILABLE)
        self.assertEqual(report["terminal_outcome_counts"], {})

    def test_latency_windows_fill_from_real_direct_evidence(self):
        registry = MetricsRegistry(capacity=8)
        batch = from_direct_object_map_update(
            fixture("direct_object_map_update_a_v1.json"),
            identity=direct_identity(),
            map_feedback=fixture("direct_map_feedback_a_v1.json"),
        )
        registry.observe_direct_batch(batch)
        report = registry.report(now_wall_s=1_789_000_000.5)
        filled = [w for w in report["latency_windows"] if w["count"] > 0]
        self.assertEqual(len(filled), 3)
        self.assertEqual(report["terminal_outcome_counts"], {"RESULT_INSTALLED": 1})
        self.assertEqual(report["window_capacity"], 8)

    def test_stream_rate_meter_reports_na_before_two_samples(self):
        registry = MetricsRegistry()
        registry.observe_receipt("s", frame_id=1, received_wall_s=100.0)
        report = registry.report(now_wall_s=100.5)
        self.assertEqual(report["streams"]["s"]["update_rate_hz"], NOT_AVAILABLE)
        self.assertEqual(report["streams"]["s"]["last_accepted_frame_id"], 1)
        registry.observe_receipt("s", frame_id=2, received_wall_s=100.1)
        report = registry.report(now_wall_s=100.5)
        self.assertAlmostEqual(report["streams"]["s"]["update_rate_hz"], 10.0, places=6)
        self.assertAlmostEqual(report["streams"]["s"]["since_last_received_ms"], 400.0, places=3)

    def test_association_reduction_is_na_with_no_raw_observations(self):
        self.assertEqual(association_reduction(0, 0), NOT_AVAILABLE)
        self.assertAlmostEqual(association_reduction(3, 2), 1.0 - 2 / 3)


class RouteContractTests(unittest.TestCase):
    def test_canonical_route_loads_and_hash_verifies(self):
        route = load_canonical_route_b()
        self.assertEqual(route.sha256, route_contract_v1.CANONICAL_ROUTE_B_SHA256)
        self.assertEqual(route.map_name, "Carla/Maps/Town10HD_Opt")
        self.assertEqual(route.waypoint_count, 589)
        self.assertTrue(route.loop)
        self.assertGreater(route.progress_samples, 0)

    def test_route_b_mode_refuses_an_absent_canonical_route(self):
        with self.assertRaisesRegex(CoopInputError, "absent"):
            build_route_contract(
                mode=route_contract_v1.ROUTE_MODE_ROUTE_B_PASSIVE,
                stream_ids=("a", "b"), ue_ids=("ue-a", "ue-b"),
                route_path="/nonexistent/route_b.json",
            )

    def test_route_b_mode_refuses_a_hash_mismatched_route(self):
        with self.assertRaisesRegex(CoopInputError, "hash mismatch"):
            build_route_contract(
                mode=route_contract_v1.ROUTE_MODE_ROUTE_B_PASSIVE,
                stream_ids=("a", "b"), ue_ids=("ue-a", "ue-b"),
                expected_sha256="0" * 64,
            )

    def test_route_b_mode_refuses_fewer_than_two_registered_streams(self):
        for streams, ues in ((("only-one",), ("ue-a",)), (("a", "b"), ("ue-a",))):
            with self.subTest(streams=streams, ues=ues):
                with self.assertRaisesRegex(CoopInputError, "at least"):
                    build_route_contract(
                        mode=route_contract_v1.ROUTE_MODE_ROUTE_B_PASSIVE,
                        stream_ids=streams, ue_ids=ues,
                    )

    def test_stage2_mode_is_never_labelled_route_b(self):
        contract = build_route_contract(
            mode=route_contract_v1.ROUTE_MODE_STAGE2,
            stream_ids=("two-ue/ue-a", "two-ue/ue-b"),
            ue_ids=("ue-a", "ue-b"),
            stage2_spawn_indices=(80, 85, 91, 94, 99, 80),
        )
        self.assertFalse(contract.is_route_b)
        self.assertIsNone(contract.route)
        self.assertEqual(contract.qualification_status, "STAGE2_LOOP_NOT_ROUTE_B")
        self.assertFalse(
            route_contract_v1.mentions_route_b_as_a_claim(contract.presentation_label)
        )
        self.assertEqual(contract.overlay_polyline(), ())
        document = contract.as_dict()
        self.assertFalse(document["is_route_b"])
        self.assertFalse(document["route_overlay_available"])
        self.assertFalse(document["live_route_b_run_authorized"])

    def test_a_non_route_b_contract_cannot_claim_route_b_in_its_label(self):
        with self.assertRaisesRegex(CoopInputError, "must not claim Route B"):
            route_contract_v1.RouteContract(
                mode=route_contract_v1.ROUTE_MODE_STAGE2,
                presentation_label="Route B full-map loop",
                qualification_status="X",
                is_route_b=False,
                registered_stream_ids=("a", "b"),
                registered_ue_ids=("ue-a", "ue-b"),
            )
        # An explicit negation is allowed.
        self.assertFalse(
            route_contract_v1.mentions_route_b_as_a_claim("Stage-2 loop (NOT Route B)")
        )
        self.assertTrue(route_contract_v1.mentions_route_b_as_a_claim("Route-B loop"))

    def test_route_b_contract_never_authorizes_a_live_run(self):
        contract = build_route_contract(
            mode=route_contract_v1.ROUTE_MODE_ROUTE_B_PASSIVE,
            stream_ids=("two-ue/ue-a", "two-ue/ue-b"),
            ue_ids=("ue-a", "ue-b"),
        )
        self.assertTrue(contract.is_route_b)
        self.assertEqual(
            contract.qualification_status,
            "ROUTE_B_CONTRACT_BOUND_LIVE_RUN_UNAUTHORIZED",
        )
        self.assertFalse(contract.as_dict()["live_route_b_run_authorized"])
        self.assertGreater(len(contract.overlay_polyline(stride=4)), 10)

    def test_missing_progress_contract_is_refused_only_when_required(self):
        with self.assertRaisesRegex(CoopInputError, "progress contract is absent"):
            load_canonical_route_b(
                progress_path="/nonexistent/progress.csv", require_progress=True
            )
        route = load_canonical_route_b(
            progress_path="/nonexistent/progress.csv", require_progress=False
        )
        self.assertIsNone(route.progress_path)
        self.assertEqual(route.progress_samples, 0)

    def test_unregistered_route_mode_is_refused(self):
        with self.assertRaisesRegex(CoopInputError, "unregistered route mode"):
            build_route_contract(
                mode="route-c", stream_ids=("a", "b"), ue_ids=("ue-a", "ue-b")
            )


class ReferencePinTests(unittest.TestCase):
    def test_production_module_is_pinned_and_not_importable_here(self):
        report = reference_pin_report()
        self.assertFalse(report["production_module_importable_in_this_branch"])
        self.assertEqual(
            report["integration_status"], "REQUIRES_MASTER_BASED_BRIDGE_WORKTREE"
        )
        names = {item["filename"] for item in report["files"]}
        self.assertEqual(
            names, {"protocol.py", "map_ingest.py", "edge_publisher.py"}
        )
        for item in report["files"]:
            self.assertEqual(len(item["expected_sha256"]), 64)
            self.assertIn(item["status"], {"MATCH", "MISMATCH", "UNAVAILABLE"})

    def test_importing_the_production_package_is_not_attempted(self):
        import sys
        self.assertNotIn(
            "rl_agent.splitfusion_direct_edge_map_v1", sys.modules
        )


if __name__ == "__main__":
    unittest.main()
