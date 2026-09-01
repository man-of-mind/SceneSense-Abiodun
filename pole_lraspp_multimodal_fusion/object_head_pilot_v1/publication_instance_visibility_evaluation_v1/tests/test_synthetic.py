from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collection.route_b_publication_instance_visibility_v1.core import (
    VisibilityGroundTruthError,
    decode_instance_bgra,
    measure_visibility,
    prove_actor_id_mapping,
    relative_transform_matrix,
    require_renderer_proof,
    reproduce_transform_matrix,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_instance_visibility_evaluation_v1.protocol import (
    FrozenProtocolError,
    load_registered_protocol,
)


class PublicationVisibilitySyntheticTests(unittest.TestCase):
    def test_visibility_equation_and_outside_pixels(self) -> None:
        reference = np.zeros((4, 5), dtype=bool); reference[1:3, 1:4] = True
        visible = np.zeros_like(reference); visible[1:3, 2:4] = True; visible[0, 0] = True
        record = measure_visibility(visible, reference)
        self.assertEqual(record["unoccluded_pixels"], 6)
        self.assertEqual(record["overlap_pixels"], 4)
        self.assertEqual(record["visible_outside_reference_pixels"], 1)
        self.assertAlmostEqual(record["visibility"], 4 / 6)

    def test_instance_decode_and_mapping_are_fail_closed(self) -> None:
        raw = np.zeros((2, 2, 4), dtype=np.uint8)
        raw[0, 0, :3] = [44, 1, 14]
        semantic, rendered = decode_instance_bgra(raw)
        self.assertEqual(int(semantic[0, 0]), 14)
        self.assertEqual(int(rendered[0, 0]), 300)
        proof = prove_actor_id_mapping([
            {"actor_id": 300, "rendered_instance_ids": [19158]},
            {"actor_id": 511, "rendered_instance_ids": [21897]},
        ])
        self.assertTrue(proof["bijection_proven"])
        self.assertFalse(proof["actor_id_equals_rendered_instance_id"])
        with self.assertRaises(VisibilityGroundTruthError):
            prove_actor_id_mapping([{"actor_id": 300, "rendered_instance_ids": [301, 302]}])

    def test_frozen_protocol_loads_and_hash_drift_fails(self) -> None:
        loaded = load_registered_protocol()
        self.assertTrue(loaded["bound_files_verified"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); lock = root / "lock.json"; protocol = root / "protocol.json"
            lock.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            protocol.write_text("{}", encoding="utf-8")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaises(FrozenProtocolError):
                load_registered_protocol(
                    root=root, lock_path=lock, protocol_path=protocol,
                    expected_lock_sha256=digest(lock), expected_protocol_sha256=digest(protocol),
                    verify_bound_files=False,
                )
            with self.assertRaises(FrozenProtocolError):
                load_registered_protocol(
                    root=root, lock_path=lock, protocol_path=protocol,
                    expected_lock_sha256="0" * 64, expected_protocol_sha256=digest(protocol),
                    verify_bound_files=False,
                )

    def test_camera_relative_transform_reproduction(self) -> None:
        import carla
        camera = carla.Transform(
            carla.Location(x=12.0, y=-7.0, z=2.2),
            carla.Rotation(pitch=-3.0, yaw=47.0, roll=1.0),
        )
        actor = carla.Transform(
            carla.Location(x=18.0, y=5.0, z=0.4),
            carla.Rotation(pitch=2.0, yaw=-31.0, roll=-1.0),
        )
        reference = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=800.0), carla.Rotation()
        )
        relative = relative_transform_matrix(camera, actor)
        reproduced = reproduce_transform_matrix(reference, relative)
        recovered = np.linalg.inv(np.asarray(reference.get_matrix())) @ reproduced
        self.assertTrue(np.allclose(recovered, relative, atol=1e-8))

    def test_renderer_proof_blocks_instead_of_falling_back(self) -> None:
        proof = {
            "actor_id_mapping_proven": True,
            "reference_intrinsics_equal": True,
            "reference_coordinates_equal": True,
            "external_geometry_absent": True,
            "walker_bone_pose_copy_proven": False,
        }
        with self.assertRaisesRegex(VisibilityGroundTruthError, "GROUND_TRUTH_BLOCKED"):
            require_renderer_proof(proof)


if __name__ == "__main__":
    unittest.main()
