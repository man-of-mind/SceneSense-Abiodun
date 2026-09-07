import copy
import unittest

from spatial_map_coop.multi_ue_v1 import (
    MultiUEContractError,
    MultiUEFrameBuffer,
    from_splitfusion_edge_result,
)


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


class MultiUEIngressTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
