import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import sync_probe as S


def image(frame=7, timestamp=1.25, translation=0.):
    matrix = np.eye(4)
    matrix[0, 3] = translation
    return types.SimpleNamespace(frame=frame, timestamp=timestamp, width=1280, height=720,
        transform=types.SimpleNamespace(get_matrix=lambda: matrix.tolist()))


def aligned():
    snap = {"frame_id": 7, "timestamp": 1.25, "camera_matrix": np.eye(4).tolist(),
            "warmup": False, "arm": "barrier", "arm_tick": 1}
    headers = {role: S.sensor_header(image()) for role in S.ROLES}
    return snap, headers


class SyncProbeTests(unittest.TestCase):
    def test_exact_headers_pass(self):
        self.assertTrue(S.compare_headers(*aligned())["valid"])

    def test_camera_agreement_one_tick_ahead_still_fails(self):
        snap, headers = aligned()
        for role in ("rgb", "semantic", "depth"):
            headers[role]["timestamp"] += .05
        result = S.compare_headers(snap, headers)
        self.assertFalse(result["valid"])
        self.assertIn("EXACT_TIMESTAMP_MISMATCH", result["reason"])

    def test_near_timestamp_never_relaxed(self):
        snap, headers = aligned()
        headers["depth"]["timestamp"] = np.nextafter(1.25, 2.)
        self.assertFalse(S.compare_headers(snap, headers)["valid"])

    def test_pose_disagreement_detected(self):
        snap, headers = aligned()
        headers["depth"]["matrix"][0][3] = .1
        self.assertIn("CAMERA_POSE_MISMATCH", S.compare_headers(snap, headers)["reason"])

    def test_same_camera_pose_later_than_world_detected(self):
        snap, headers = aligned()
        for role in ("rgb", "semantic", "depth"):
            headers[role]["matrix"][0][3] = .25
        result = S.compare_headers(snap, headers)
        self.assertIn("OWNED_TICK_POSE_MISMATCH", result["reason"])

    def test_missing_is_not_zero(self):
        snap, headers = aligned()
        del headers["depth"]
        result = S.compare_headers(snap, headers)
        self.assertEqual(result["missing_roles"], ["depth"])
        self.assertFalse(result["valid"])

    def test_store_wait_bounded_and_duplicates_recorded(self):
        store = S.HeaderStore()
        self.assertFalse(store.wait(7, .001))
        for role in S.ROLES:
            store.put(role, image())
        self.assertTrue(store.wait(7, .001))
        store.put("rgb", image())
        self.assertEqual(store.duplicates, [{"role": "rgb", "frame_id": 7}])

    def test_actual_spawn_and_callback_composition(self):
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        class Blueprint:
            def __init__(self, name):
                self.name, self.attributes = name, {}
            def set_attribute(self, key, value):
                self.attributes[key] = value
        class Actor:
            def __init__(self, bp, identifier):
                self.id, self.attributes, self.bp = identifier, bp.attributes, bp.name
            def listen(self, callback):
                self.callback = callback
        class World:
            def __init__(self):
                self.actors = []
            def get_blueprint_library(self):
                return types.SimpleNamespace(find=Blueprint)
            def spawn_actor(self, bp, transform, attach_to):
                result = Actor(bp, len(self.actors) + 1)
                self.actors.append(result)
                return result
        world, owned, store = World(), [], S.HeaderStore()
        with patch.object(P, "sensor_transform", lambda mount: mount):
            collector = S.make_collector(world, object(), store, owned)
        self.assertEqual(len(owned), 4)
        self.assertEqual(set(a.bp for a in owned), {"sensor.camera.rgb", "sensor.camera.semantic_segmentation", "sensor.camera.depth", "sensor.other.radar"})
        for actor in owned:
            self.assertEqual(actor.attributes["sensor_tick"], "0.0")
            actor.callback(image())
        self.assertTrue(store.wait(7, .001))
        self.assertEqual(set(store.get(7)), set(S.ROLES))
        self.assertEqual(len(collector._avo_rgb), 1)
        snapshot = types.SimpleNamespace(frame=7, timestamp=types.SimpleNamespace(elapsed_seconds=1.25),
            find=lambda identifier: types.SimpleNamespace(get_transform=lambda: image().transform))
        proof = collector.wait_for_sensor_frame(7, snapshot=snapshot, timeout_s=.01)
        self.assertEqual(proof["status"], "EXACT_ALL_SENSOR_TICK_ACCEPTED")
        self.assertEqual(set(proof["completed_roles"]), set(S.ROLES))
        self.assertEqual(len(collector._avo_rgb), 1)  # Shared barrier does not consume GT inputs.

    def test_authorization_before_import_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            args = types.SimpleNamespace(execute="wrong", output_dir=output)
            with self.assertRaises(ValueError):
                S.run(args)
            self.assertFalse(output.exists())

    def test_verifier_replays_headers_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_names = ("rl_agent/splitfusion_run5_live_avo_v1/sync_probe.py",
                "rl_agent/splitfusion_run5_live_avo_v1/live_gt.py",
                "rl_agent/splitfusion_run5_live_avo_v1/gt_source_sync.py",
                "rl_agent/ue_route_b_split_cell_adapter_v1.py")
            sources = {}
            for name in source_names:
                source = root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("pinned source\n")
                sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            row = S.compare_headers(*aligned())
            barrier = {"mode": "barrier", "rows": [row], "frames": 1, "scored_frames": 1}
            document = {"schema": S.SCHEMA, "passed": True, "ticks_per_arm": 1, "warmup_ticks_per_arm": 0,
                "source_sha256": sources,
                "arms": [barrier], "settings_restored": True, "owned_actors_destroyed": True,
                "checks": {key: True for key in ("barrier_arm_complete", "barrier_scored_exact", "no_scored_barrier_timeout",
                    "settings_restored", "owned_actors_destroyed", "no_probe_error", "no_duplicate_callbacks")},
                "cleanup": {"ok": True}}
            path = root / "SYNC_PROBE_RESULT.json"
            path.write_text(json.dumps(document))
            (root / "barrier_headers.json").write_text(json.dumps(barrier))
            manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, root / "barrier_headers.json")}
            (root / "OUTPUT_SHA256.json").write_text(json.dumps(manifest))
            self.assertTrue(S.verify_probe(path, source_root=root)["passed"])
            source.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Source hash drift"):
                S.verify_probe(path, source_root=root)


if __name__ == "__main__":
    unittest.main()
