from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collection.route_b_publication_zbuffer_visibility_v2.core import (
    ZBufferVisibilityError,
    compute_zbuffer_visibility,
    decode_depth_bgra,
    relative_transform_matrix,
    reproduce_transform_matrix,
)
from data_collection.route_b_publication_zbuffer_visibility_v2.controlled_qualification import (
    optional_vehicle_instance_diagnostic,
    safe_occluder_center_depth,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_zbuffer_visibility_evaluation_v2.protocol import (
    FrozenProtocolError,
    load_registered_protocol,
)


class _Transform:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def get_matrix(self) -> list[list[float]]:
        return self.matrix.tolist()

    def get_inverse_matrix(self) -> list[list[float]]:
        return np.linalg.inv(self.matrix).tolist()


class PublicationZBufferVisibilitySyntheticTests(unittest.TestCase):
    def test_depth_decode_uses_exact_24bit_bgra_order(self) -> None:
        raw = np.zeros((1, 4, 4), dtype=np.uint8)
        codes = np.asarray([0, 1, 0x010203, 0xFFFFFF], dtype=np.uint32)
        raw[0, :, 2] = codes & 0xFF
        raw[0, :, 1] = (codes >> 8) & 0xFF
        raw[0, :, 0] = (codes >> 16) & 0xFF
        expected = codes.astype(np.float64) / 16_777_215 * 1000.0
        np.testing.assert_array_equal(decode_depth_bgra(raw)[0], expected)

    def test_actor_support_is_exact_registered_empty_comparison(self) -> None:
        empty = np.asarray([[10.0, 10.0, 10.0, 10.0]])
        actor = np.asarray([[9.979, 9.98, 9.0, 10.0]])
        result = compute_zbuffer_visibility(empty, actor, actor)
        np.testing.assert_array_equal(result["support"], [[True, False, True, False]])
        self.assertEqual(result["support_pixels"], 2)

    def test_visible_support_requires_exact_actor_surface_depth(self) -> None:
        empty = np.full((1, 4), 20.0)
        actor = np.full((1, 4), 10.0)
        scene = np.asarray([[10.0, 10.02, 9.0, 11.0]])
        result = compute_zbuffer_visibility(empty, actor, scene)
        np.testing.assert_array_equal(result["visible"], [[True, True, False, False]])
        self.assertEqual(result["visible_pixels"], 2)
        self.assertEqual(result["visibility"], 0.5)

    def test_visibility_is_bounded_and_empty_support_fails_closed(self) -> None:
        empty = np.full((2, 3), 100.0)
        actor = np.full((2, 3), 2.0)
        clear = compute_zbuffer_visibility(empty, actor, actor)
        full = compute_zbuffer_visibility(empty, actor, np.full((2, 3), 1.0))
        self.assertEqual(clear["visibility"], 1.0)
        self.assertEqual(full["visibility"], 0.0)
        with self.assertRaises(ZBufferVisibilityError):
            compute_zbuffer_visibility(empty, empty, empty)

    def test_camera_relative_transform_reproduction(self) -> None:
        camera = np.eye(4); camera[:3, 3] = [12.0, -7.0, 2.2]
        actor = np.eye(4); actor[:3, 3] = [18.0, 5.0, 0.4]
        reference = np.eye(4); reference[:3, 3] = [0.0, 0.0, 800.0]
        relative = relative_transform_matrix(_Transform(camera), _Transform(actor))
        reproduced = reproduce_transform_matrix(_Transform(reference), relative)
        np.testing.assert_allclose(np.linalg.inv(reference) @ reproduced, relative, atol=1e-12)

    def test_registered_protocol_loads_and_hash_drift_fails_closed(self) -> None:
        loaded = load_registered_protocol()
        self.assertTrue(loaded["registered_controls_verified"])
        self.assertFalse(loaded["vehicle_instance_diagnostic_required"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "lock.json"
            protocol = root / "protocol.json"
            amendment = root / "amendment.json"
            evidence = root / "evidence.json"
            previous_failure = root / "previous_failure.json"
            lock.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            protocol.write_text("{}", encoding="utf-8")
            amendment.write_text("{}", encoding="utf-8")
            evidence.write_text("{}", encoding="utf-8")
            previous_failure.write_text("{}", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FrozenProtocolError):
                load_registered_protocol(
                    root=root,
                    lock_path=lock,
                    protocol_path=protocol,
                    amendment_path=amendment,
                    blocked_evidence_path=evidence,
                    previous_failure_path=previous_failure,
                    expected_lock_sha256="0" * 64,
                    expected_protocol_sha256=digest(protocol),
                    expected_amendment_sha256=digest(amendment),
                    expected_blocked_evidence_sha256=digest(evidence),
                    expected_previous_failure_sha256=digest(previous_failure),
                )

    def test_missing_optional_vehicle_instance_component_is_nonblocking(self) -> None:
        raw = np.zeros((3, 4, 4), dtype=np.uint8)
        support = np.zeros((3, 4), dtype=bool)
        support[1, 1:3] = True
        diagnostic = optional_vehicle_instance_diagnostic(raw, support)
        self.assertFalse(diagnostic["instance_diagnostic_available"])
        self.assertIn("unavailable or ambiguous", diagnostic["instance_diagnostic_unavailable_reason"])
        self.assertIsNone(diagnostic["vehicle_depth_support_vs_instance_iou"])
        self.assertIsNone(diagnostic["instance_component_mask"])

    def test_safe_occluder_center_depth_preserves_camera_plane_margin(self) -> None:
        offsets = np.asarray([-5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0])
        center = safe_occluder_center_depth(offsets)
        self.assertEqual(center, 5.5)
        self.assertEqual(float(np.min(offsets + center)), 0.5)
        shallow_offsets = np.asarray([-1.0, -1.0, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0])
        self.assertEqual(safe_occluder_center_depth(shallow_offsets), 4.5)


if __name__ == "__main__":
    unittest.main()
