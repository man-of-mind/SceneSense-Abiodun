import copy
import sys
import unittest
from unittest import mock

from spatial_map_coop.multi_ue_v1 import (
    INGRESS_ENVELOPE_SCHEMA,
    AssociationPolicy,
    MultiUEContractError,
    MultiUEFrameBuffer,
    MultiUESpatialMapService,
    create_flask_app,
    from_splitfusion_edge_result,
)
from spatial_map_coop.multi_ue_v1.faults import (
    FAULT_CASES,
    STALE_OFFSET_NS,
    XY_BIAS_M,
    deterministic_fault_case,
    mutate_edge_result,
)
from spatial_map_coop.multi_ue_v1 import live_two_ue_action50_v1 as live_action50


def edge_result(stream_id, frame_id, capture_ns, x_coord):
    record = {
        "stream_id": stream_id,
        "capture_timestamp_ns": capture_ns,
        "sample_id": f"{stream_id}:{frame_id}",
        "frame_id": frame_id,
        "prediction_index": 0,
        "class_name": "vehicle",
        "internal_class": 1,
        "score": 0.9,
        "world_x": x_coord,
        "world_y": 2.0,
        "world_z": 0.5,
        "local_x": 1.0,
        "local_y": 2.0,
        "local_z": 3.0,
        "size_x": 4.5,
        "size_y": 1.8,
        "size_z": 1.5,
        "yaw_sin": 0.0,
        "yaw_cos": 1.0,
        "parked_score": 0.1,
        "radar_support_score": 0.8,
        "center_x_px": 100.0,
        "center_y_px": 200.0,
        "bbox_x0": 90.0,
        "bbox_y0": 180.0,
        "bbox_x1": 110.0,
        "bbox_y1": 220.0,
        "fpn_level": "p3",
        "level_index": 1,
        "point_index": 7,
        "candidate_identity": f"{frame_id}:1:7",
        "physical_ray_x_px": 100.0,
        "physical_ray_y_px": 200.0,
        "actor_forward_depth_m": 12.0,
        "depth_bin": 1,
        "depth_residual": 0.2,
    }
    update = {
        "schema": "splitfusion_object_map_update.v1",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "action_id": 71,
        "records": [record],
    }
    return {
        "schema": "splitfusion_edge_result.v2",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "action_id": 71,
        "object_map_update": update,
    }


def policy(minimum_confidence=0.0):
    return AssociationPolicy(
        maximum_observation_age_ns=200_000_000,
        alignment_tolerance_ns=50_000_000,
        maximum_pair_time_delta_ns=50_000_000,
        maximum_xy_distance_m=2.0,
        maximum_relative_size_difference=0.4,
        track_match_distance_m=3.0,
        track_stale_after_ns=500_000_000,
        minimum_confidence=minimum_confidence,
    )


class MultiUEIngressTests(unittest.TestCase):
    def test_self_contained_live_cli_owns_actor_selection(self):
        argv = [
            "live_two_ue_action50_v1.py",
            "--spawn-two-egos",
            "--execute",
            live_action50.EXECUTE_TOKEN,
            "--output",
            "unused-create-only-output",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = live_action50.parse_args()
        self.assertTrue(args.spawn_two_egos)
        self.assertEqual((args.ue_a_actor_id, args.ue_b_actor_id), (-1, -1))
        live_action50._require_actor_selection(args)

    def test_shadow_fault_schedule_and_mutations_are_deterministic(self):
        self.assertEqual(
            [deterministic_fault_case(index) for index in range(7)],
            [*FAULT_CASES, *FAULT_CASES[:2]],
        )
        source = edge_result("stream-b", 20, 2_000_000_000, 5.0)
        stale = mutate_edge_result(source, "STALE")
        self.assertEqual(
            stale["capture_timestamp_ns"],
            source["capture_timestamp_ns"] - STALE_OFFSET_NS,
        )
        self.assertEqual(
            stale["object_map_update"]["records"][0]["capture_timestamp_ns"],
            stale["capture_timestamp_ns"],
        )
        biased = mutate_edge_result(source, "XY_BIAS")
        self.assertEqual(
            biased["object_map_update"]["records"][0]["world_x"],
            source["object_map_update"]["records"][0]["world_x"] + XY_BIAS_M,
        )
        self.assertEqual(source["object_map_update"]["records"][0]["world_x"], 5.0)

    def test_shadow_nonfinite_and_identity_conflict_fail_closed(self):
        source = edge_result("stream-b", 20, 2_000_000_000, 5.0)
        nonfinite = mutate_edge_result(source, "NONFINITE")
        with self.assertRaises(MultiUEContractError):
            from_splitfusion_edge_result(
                nonfinite,
                ue_id="ue-b",
                session_id="drive",
                received_timestamp_ns=2_010_000_000,
            )
        service = MultiUESpatialMapService(policy())
        service.ingest_splitfusion(
            source,
            ue_id="ue-b",
            session_id="drive",
            received_timestamp_ns=2_010_000_000,
        )
        with self.assertRaises(MultiUEContractError):
            service.ingest_splitfusion(
                mutate_edge_result(source, "IDENTITY_CONFLICT"),
                ue_id="ue-b",
                session_id="drive",
                received_timestamp_ns=2_010_000_000,
            )

    def test_two_sources_form_one_aligned_raw_snapshot(self):
        buffer = MultiUEFrameBuffer(max_frames_per_source=2)
        first = from_splitfusion_edge_result(
            edge_result("stream-a", 10, 1_000_000_000, 5.0),
            ue_id="ue-a",
            session_id="drive-1",
            received_timestamp_ns=1_020_000_000,
        )
        second = from_splitfusion_edge_result(
            edge_result("stream-b", 20, 1_030_000_000, 5.4),
            ue_id="ue-b",
            session_id="drive-1",
            received_timestamp_ns=1_050_000_000,
        )
        self.assertEqual(buffer.ingest(first).disposition, "ACCEPTED")
        self.assertEqual(buffer.ingest(second).disposition, "ACCEPTED")
        snapshot = buffer.snapshot(
            clock_domain="unix_wall_ns",
            snapshot_timestamp_ns=1_060_000_000,
            maximum_source_age_ns=100_000_000,
            alignment_tolerance_ns=40_000_000,
        )
        self.assertEqual([item.batch.ue_id for item in snapshot.sources], ["ue-a", "ue-b"])
        self.assertTrue(all(item.batch.received_clock_domain == "unix_wall_ns"
                            for item in snapshot.sources))
        self.assertEqual(snapshot.raw_object_count, 2)
        self.assertEqual(snapshot.capture_spread_ns, 30_000_000)
        self.assertTrue(snapshot.aligned_for_fusion)

    def test_duplicate_is_idempotent_but_conflict_fails_closed(self):
        buffer = MultiUEFrameBuffer()
        payload = edge_result("stream-a", 10, 1_000_000_000, 5.0)
        batch = from_splitfusion_edge_result(
            payload,
            ue_id="ue-a",
            session_id="drive-1",
            received_timestamp_ns=1_020_000_000,
        )
        buffer.ingest(batch)
        self.assertEqual(buffer.ingest(batch).disposition, "DUPLICATE_IDENTICAL")
        changed = copy.deepcopy(payload)
        changed["object_map_update"]["records"][0]["world_x"] = 7.0
        conflict = from_splitfusion_edge_result(
            changed,
            ue_id="ue-a",
            session_id="drive-1",
            received_timestamp_ns=1_020_000_000,
        )
        with self.assertRaises(MultiUEContractError):
            buffer.ingest(conflict)

    def test_identity_drift_is_rejected_before_ingest(self):
        payload = edge_result("stream-a", 10, 1_000_000_000, 5.0)
        payload["object_map_update"]["records"][0]["frame_id"] = 11
        with self.assertRaises(MultiUEContractError):
            from_splitfusion_edge_result(
                payload,
                ue_id="ue-a",
                session_id="drive-1",
                received_timestamp_ns=1_020_000_000,
            )
        service = MultiUESpatialMapService(policy())
        service.ingest_splitfusion(
            edge_result("stream-a", 10, 1_000_000_000, 5.0),
            ue_id="ue-a",
            session_id="drive-1",
            received_timestamp_ns=1_020_000_000,
        )
        with self.assertRaises(MultiUEContractError):
            service.ingest_splitfusion(
                edge_result("stream-b", 11, 1_010_000_000, 5.0),
                ue_id="ue-a",
                session_id="drive-1",
                received_timestamp_ns=1_020_000_000,
            )

    def test_http_two_ue_delivery_forms_one_provenance_preserving_track(self):
        service = MultiUESpatialMapService(policy())
        client = create_flask_app(service).test_client()
        for ue_id, stream_id, frame_id, capture_ns, x_coord in (
            ("ue-a", "stream-a", 10, 1_000_000_000, 5.0),
            ("ue-b", "stream-b", 20, 1_030_000_000, 5.4),
        ):
            response = client.post(
                "/api/multi_ue/v1/updates",
                json={
                    "schema": INGRESS_ENVELOPE_SCHEMA,
                    "ue_id": ue_id,
                    "session_id": "drive-1",
                    "payload": edge_result(stream_id, frame_id, capture_ns, x_coord),
                },
            )
            self.assertEqual(response.status_code, 202, response.get_json())
        response = client.get(
            "/api/multi_ue/v1/spatial_map",
            query_string={
                "clock_domain": "unix_wall_ns",
                "snapshot_timestamp_ns": 1_060_000_000,
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        document = response.get_json()
        self.assertEqual(document["input_source_count"], 2)
        self.assertEqual(document["association_count"], 1)
        self.assertEqual(len(document["tracks"]), 1)
        track = document["tracks"][0]
        self.assertEqual(track["selected_ue_id"], "ue-b")
        self.assertEqual(track["world_xyz"][0], 5.4)
        self.assertEqual(len(track["contributing_sources"]), 2)
        self.assertEqual(len(track["contributing_observation_hashes"]), 2)
        self.assertEqual(document["associations"][0]["position_combination"],
                         "NONE_SELECTED_SOURCE_ONLY")

    def test_confidence_and_alignment_are_explicit_filters(self):
        service = MultiUESpatialMapService(policy(minimum_confidence=0.5))
        old = edge_result("stream-old", 1, 900_000_000, 5.0)
        low = edge_result("stream-low", 2, 1_030_000_000, 5.2)
        low["object_map_update"]["records"][0]["score"] = 0.4
        service.ingest_splitfusion(
            old, ue_id="ue-old", session_id="drive", received_timestamp_ns=1_040_000_000
        )
        service.ingest_splitfusion(
            low, ue_id="ue-low", session_id="drive", received_timestamp_ns=1_040_000_000
        )
        document = service.snapshot(
            clock_domain="unix_wall_ns", snapshot_timestamp_ns=1_050_000_000
        ).as_dict()
        self.assertEqual(document["accepted_observation_count"], 0)
        self.assertEqual(
            {item["reason"] for item in document["rejected_observations"]},
            {"SOURCE_OUTSIDE_ALIGNMENT_WINDOW", "BELOW_REGISTERED_CONFIDENCE"},
        )
        aged = service.snapshot(
            clock_domain="unix_wall_ns", snapshot_timestamp_ns=1_300_000_000
        ).as_dict()
        self.assertIn(
            "OBSERVATION_TOO_OLD",
            {item["reason"] for item in aged["rejected_observations"]},
        )

    def test_hungarian_assignment_is_one_to_one_and_track_update_is_idempotent(self):
        service = MultiUESpatialMapService(policy())
        first = edge_result("stream-a", 10, 1_000_000_000, 0.0)
        first_second = copy.deepcopy(first["object_map_update"]["records"][0])
        first_second.update(
            {"candidate_identity": "10:1:8", "point_index": 8, "world_x": 10.0}
        )
        first["object_map_update"]["records"].append(first_second)
        second = edge_result("stream-b", 20, 1_010_000_000, 9.7)
        second_second = copy.deepcopy(second["object_map_update"]["records"][0])
        second_second.update(
            {"candidate_identity": "20:1:8", "point_index": 8, "world_x": 0.3}
        )
        second["object_map_update"]["records"].append(second_second)
        service.ingest_splitfusion(
            first, ue_id="ue-a", session_id="drive", received_timestamp_ns=1_020_000_000
        )
        service.ingest_splitfusion(
            second, ue_id="ue-b", session_id="drive", received_timestamp_ns=1_020_000_000
        )
        snapshot = service.snapshot(
            clock_domain="unix_wall_ns", snapshot_timestamp_ns=1_030_000_000
        )
        self.assertEqual(len(snapshot.association.associations), 2)
        self.assertEqual([item.source_count for item in snapshot.association.associations], [2, 2])
        self.assertEqual([track.update_count for track in snapshot.tracks], [1, 1])
        repeated = service.snapshot(
            clock_domain="unix_wall_ns", snapshot_timestamp_ns=1_040_000_000
        )
        self.assertEqual([track.update_count for track in repeated.tracks], [1, 1])
        with self.assertRaises(ValueError):
            service.snapshot(
                clock_domain="unix_wall_ns", snapshot_timestamp_ns=1_039_999_999
            )


if __name__ == "__main__":
    unittest.main()
