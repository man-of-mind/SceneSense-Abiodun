"""No simulator, GPU, services, or file writes; actual local CARLA value types."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch

import carla
import numpy as np

from . import snapshot as S
from rl_agent.splitfusion_live_parity_repair_v2.source_ownership import FrozenImage, FrozenTransform


def case(frame=51):
    stamp = 123.125
    width, height = 8, 5
    transform = carla.Transform(carla.Location(x=3.125, y=-6.25, z=2.5),
                                carla.Rotation(pitch=3.5, yaw=-14.75, roll=0.125))
    matrix = np.asarray(transform.get_matrix())
    camera = FrozenTransform(tuple(tuple(row) for row in matrix))
    metadata = dict(frame_id=frame, timestamp=stamp, width=width, height=height,
        camera_matrix=matrix.tolist(), timestamp_hex=stamp.hex(), timestamp_repr=repr(stamp), callback_receipt_raw_ns=999)
    rgb = {**metadata, "semantic_callback": dict(metadata), "depth_callback": dict(metadata),
           "radar_callback": {**metadata, "width": None, "height": None}}
    images = [FrozenImage(frame, stamp, width, height,
        (np.arange(width * height * 4, dtype=np.uint8) + offset).tobytes(), camera) for offset in range(3)]
    rgb["_rgb_image"] = images[0]
    actors = []
    for ident, kind in ((32, "walker.pedestrian.0001"), (7, "vehicle.test")):
        pose = carla.Transform(carla.Location(x=123.123456789, y=-12.875, z=-0.0),
            carla.Rotation(pitch=1.3, yaw=-87.27, roll=0.01))
        box = carla.BoundingBox(carla.Location(x=0.01, y=-0.02, z=1.37), carla.Vector3D(x=0.25, y=0.4, z=0.875))
        box.rotation = carla.Rotation(pitch=4.1, yaw=8.3, roll=-12.2)
        actor = SimpleNamespace(id=ident, type_id=kind, bounding_box=box,
            get_transform=lambda p=pose: p, get_location=lambda p=pose: p.location,
            get_velocity=lambda: carla.Vector3D(x=0.1, y=-0.05, z=-0.0))
        actors.append(actor)
    population = dict(frame_id=frame, registered_population_ids=[7, 32, 99], active_population_ids=[32, 7],
        registry_complete=True, owned_snapshot_complete=True, no_unregistered_spawn=True,
        population_sha256="a" * 64, population={"registered_actors": [{"actor_id": 32, "custom": "preserve me"}]},
        owned_snapshot_actor_ids=[7, 32, 99, 102], started_raw_ns=998, completed_raw_ns=1000)
    world = SimpleNamespace(frozen_frame_id=frame, frozen_timestamp_s=stamp, frozen_observed_raw_ns=998,
        population_evidence=population, get_actors=lambda: actors)
    kwargs = dict(identity=dict(stream_id="example", frame_id=frame, action_id=4, cohort="whole"), rgb=rgb,
        semantic_image=images[1], depth_image=images[2], frozen_world=world,
        camera_matrix=matrix, camera_inverse=np.linalg.inv(matrix), policy_timestamp_s=stamp,
        sync_evidence=object())
    radar = np.array([[1.25, 2.5, -0.0], [-10.3, 32.75, 1.234]], dtype=np.float64)
    return kwargs, radar


def envelope(payload, mutate=None, raw_json=None, binary_mutate=None):
    size = struct.unpack(">Q", payload[8:16])[0]
    metadata = json.loads(payload[48:48 + size])
    if mutate:
        mutate(metadata)
    encoded = raw_json if raw_json is not None else json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    binary = payload[48 + size:]
    if binary_mutate:
        binary = binary_mutate(binary, metadata)
    body = encoded + binary
    return S.MAGIC + struct.pack(">Q", len(encoded)) + hashlib.sha256(body).digest() + body


def original_geometry():
    root = Path(__file__).resolve().parents[2]
    source_path = root / "carla_collect_parked_ego_fusion_training_data.py"
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != S.SOURCE_PINS[str(source_path.relative_to(root))]:
        raise AssertionError("original geometry source drift")
    names = {"bbox_corner_offsets", "actor_bbox_world_points", "transform_world_to_actor_local", "radar_support_count"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {**vars(typing), "np": np, "carla": carla, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.kwargs, self.radar = case()
        self.payload = S.pack_snapshot(self.kwargs, self.radar)

    def test_exact_roundtrip_metadata_and_arrays(self):
        with patch.object(carla, "Client", side_effect=AssertionError("RPC prohibited")):
            actual, radar = S.unpack_snapshot(self.payload)
        self.assertEqual(set(actual), set(self.kwargs) - {"sync_evidence"})
        self.assertEqual(actual["identity"], self.kwargs["identity"])
        self.assertEqual({k: v for k, v in actual["rgb"].items() if k != "_rgb_image"},
                         {k: v for k, v in self.kwargs["rgb"].items() if k != "_rgb_image"})
        self.assertEqual(actual["frozen_world"].population_evidence, self.kwargs["frozen_world"].population_evidence)
        self.assertEqual(actual["frozen_world"].frozen_observed_raw_ns, 998)
        self.assertEqual(actual["frozen_world"].frozen_timestamp_s.hex(), float(123.125).hex())
        for name in ("camera_matrix", "camera_inverse"):
            self.assertEqual(actual[name].tobytes(), self.kwargs[name].tobytes())
            self.assertTrue(actual[name].flags.c_contiguous)
            self.assertEqual(actual[name].ctypes.data % 16, 0)
            with self.assertRaises(ValueError):
                actual[name].setflags(write=True)
        self.assertEqual(radar.dtype, self.radar.dtype)
        self.assertEqual(radar.tobytes(), self.radar.tobytes())
        self.assertEqual(radar.ctypes.data % 16, 0)
        with self.assertRaises(ValueError):
            radar.setflags(write=True)
        for old, new in ((self.kwargs["rgb"]["_rgb_image"], actual["rgb"]["_rgb_image"]),
                         (self.kwargs["semantic_image"], actual["semantic_image"]),
                         (self.kwargs["depth_image"], actual["depth_image"])):
            self.assertEqual(old.raw_data, new.raw_data)
            self.assertEqual(old.transform.get_matrix(), new.transform.get_matrix())
            self.assertEqual(old.transform.get_inverse_matrix(), new.transform.get_inverse_matrix())
            self.assertTrue(new.raw_data.readonly)
            with self.assertRaises(TypeError):
                new.raw_data[0] = 0
        self.assertEqual(S.pack_snapshot(actual, radar), self.payload)

    def test_original_geometry_and_cpp_distance_exact(self):
        actual, radar = S.unpack_snapshot(self.payload)
        helpers = original_geometry()
        camera_location = carla.Location(x=-1.125, y=20.2345, z=7.75)
        for before, after in zip(self.kwargs["frozen_world"].get_actors(), actual["frozen_world"].get_actors()):
            self.assertIsInstance(after.get_location(), carla.Location)
            self.assertEqual(before.get_location().distance(camera_location).hex(), after.get_location().distance(camera_location).hex())
            for method in ("get_matrix", "get_inverse_matrix"):
                self.assertEqual(getattr(before.get_transform(), method)(), getattr(after.get_transform(), method)())
            for first, second in ((before.get_location(), after.get_location()), (before.get_velocity(), after.get_velocity()),
                                  (before.bounding_box.location, after.bounding_box.location), (before.bounding_box.extent, after.bounding_box.extent)):
                self.assertEqual([float(getattr(first, a)).hex() for a in "xyz"], [float(getattr(second, a)).hex() for a in "xyz"])
            for first, second in ((before.get_transform().rotation, after.get_transform().rotation),
                                  (before.bounding_box.rotation, after.bounding_box.rotation)):
                self.assertEqual([float(getattr(first, a)).hex() for a in ("pitch", "yaw", "roll")],
                                 [float(getattr(second, a)).hex() for a in ("pitch", "yaw", "roll")])
            for first, second in zip(helpers["actor_bbox_world_points"](before), helpers["actor_bbox_world_points"](after)):
                self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(helpers["transform_world_to_actor_local"](self.radar, before).tobytes(),
                             helpers["transform_world_to_actor_local"](radar, after).tobytes())
            for label in ("person", "vehicle"):
                self.assertEqual(helpers["radar_support_count"](actor=before, label=label, radar_world_xyz=self.radar, margin_m=0.2),
                                 helpers["radar_support_count"](actor=after, label=label, radar_world_xyz=radar, margin_m=0.2))

    def test_native_inverse_is_not_numpy_recomputed(self):
        actor = self.kwargs["frozen_world"].get_actors()[0]
        native = np.asarray(actor.get_transform().get_inverse_matrix())
        self.assertNotEqual(native.tobytes(), np.linalg.inv(actor.get_transform().get_matrix()).tobytes())
        actual, _ = S.unpack_snapshot(self.payload)
        self.assertEqual(native.tobytes(), np.asarray(actual["frozen_world"].get_actors()[0].get_transform().get_inverse_matrix()).tobytes())

    def test_actor_order_filters_and_empty_populations(self):
        actual, _ = S.unpack_snapshot(self.payload)
        actors = actual["frozen_world"].get_actors()
        self.assertEqual([actor.id for actor in actors], [32, 7])
        self.assertEqual([actor.id for actor in actors.filter("walker.*")], [32])
        self.assertEqual(actors.find(7).type_id, "vehicle.test")
        world = self.kwargs["frozen_world"]
        world.get_actors = lambda: []
        world.population_evidence["active_population_ids"] = []
        actual, radar = S.unpack_snapshot(S.pack_snapshot(self.kwargs, np.empty((0, 3), dtype=np.float32)))
        self.assertEqual(len(actual["frozen_world"].get_actors()), 0)
        self.assertEqual(radar.shape, (0, 3))

    def test_snapshot_is_not_a_scheduler(self):
        # A worker, not a stateless codec, rejects repeated/out-of-order jobs.
        for frame in (51, 51, 2):
            kwargs, radar = case(frame)
            actual, _ = S.unpack_snapshot(S.pack_snapshot(kwargs, radar))
            self.assertEqual(actual["identity"]["frame_id"], frame)

    def test_maximum_native_image_dimensions(self):
        width, height = 1280, 720
        rgb = self.kwargs["rgb"]
        rgb.update(width=width, height=height)
        for role in ("semantic_callback", "depth_callback"):
            rgb[role].update(width=width, height=height)
        old = rgb["_rgb_image"]
        image = FrozenImage(old.frame, old.timestamp, width, height, bytes(width * height * 4), old.transform)
        rgb["_rgb_image"] = self.kwargs["semantic_image"] = self.kwargs["depth_image"] = image
        payload = S.pack_snapshot(self.kwargs, self.radar)
        actual, _ = S.unpack_snapshot(payload)
        self.assertEqual(len(actual["depth_image"].raw_data), width * height * 4)
        self.assertEqual(actual["depth_image"].raw_data, image.raw_data)

    def test_dtype_endianness_and_reject_non_native_layout(self):
        radar = np.arange(15, dtype=">f4").reshape(5, 3)
        _, actual = S.unpack_snapshot(S.pack_snapshot(self.kwargs, radar))
        self.assertEqual(actual.dtype.str, radar.dtype.str)
        self.assertEqual(actual.tobytes(), radar.tobytes())
        for bad in (np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2], np.asfortranarray(radar)):
            with self.assertRaisesRegex(S.SnapshotError, "layout"):
                S.pack_snapshot(self.kwargs, bad)
        backing = b"x" * 8 + self.kwargs["camera_inverse"].tobytes()
        bad = np.frombuffer(backing, dtype=np.float64, offset=8).reshape(4, 4)
        self.assertTrue(bad.flags.aligned)
        self.assertEqual(bad.ctypes.data % 16, 8)
        with self.assertRaisesRegex(S.SnapshotError, "alignment"):
            S.pack_snapshot({**self.kwargs, "camera_inverse": bad}, self.radar)

    def test_saved_1470_exact_matvec_with_adversarial_packet_offsets(self):
        # Exact retained frame1470, actor51: all bytes survived the first codec,
        # but an inverse at address%16==8 changed y/z by one ULP. These numbers
        # pin that observed counterexample without loading a historical dataset.
        inverse = np.array([
            [0.9978657532361672, -0.013816135801577034, -0.0638210720695113, 0.5410251166143222],
            [0.012023242593206076, 0.9995245361812385, -0.028391654309756646, -28.03113082618735],
            [0.0641829930188348, 0.027563722864649545, 0.9975574169229017, -2.2362823828577194],
            [0., 0., 0., 1.]], dtype=np.float64)
        center = np.array([106.4671612695668, -23.122317958452957, 0.6505671853050818, 1.])
        expected = np.array([107.05890038152891, -49.880845129521624, 4.60873964235468, 1.])
        self.assertEqual((inverse @ center).tobytes(), expected.tobytes())
        for offset in range(16):
            source = memoryview(bytes(offset) + inverse.tobytes())[offset:]
            restored = S._immutable_numeric_copy(source, inverse.dtype, inverse.shape)
            self.assertEqual(restored.tobytes(), inverse.tobytes())
            self.assertEqual(restored.ctypes.data % 16, 0)
            self.assertEqual((restored @ center).tobytes(), expected.tobytes())
            with self.assertRaises(ValueError):
                restored.setflags(write=True)
        # Vary actual JSON length, exercising packet parsing rather than only
        # the isolated copy helper; every possible packet residue is covered.
        residues = set()
        for padding in range(16):
            self.kwargs["rgb"]["layout_regression_padding"] = "x" * padding
            payload = S.pack_snapshot(self.kwargs, self.radar)
            size = struct.unpack(">Q", payload[8:16])[0]
            metadata = json.loads(payload[48:48 + size])
            spec = next(s for s in metadata["buffers"] if s["name"] == "camera_inverse")
            start = 48 + size + spec["offset"]
            residues.add(np.frombuffer(payload, dtype=np.uint8).ctypes.data % 16 + start % 16)
            actual, radar = S.unpack_snapshot(payload)
            for name in ("camera_matrix", "camera_inverse"):
                self.assertEqual(actual[name].ctypes.data % 16, 0)
                self.assertEqual(actual[name].tobytes(), self.kwargs[name].tobytes())
                self.assertEqual((actual[name] @ center).tobytes(), (self.kwargs[name] @ center).tobytes())
            self.assertEqual(radar.ctypes.data % 16, 0)
        self.assertEqual(residues, set(range(16)))

    def test_source_pin_is_retained_geometry(self):
        root = Path(__file__).resolve().parents[2]
        for path, expected in S.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), expected)

    def test_import_is_cpu_neutral(self):
        code = "import sys; from rl_agent.splitfusion_live_gt_optimization_v3 import snapshot; assert 'carla' not in sys.modules; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_frame_timestamp_and_population_mismatches(self):
        mutations = [lambda m: m["identity"].update(frame_id=999), lambda m: m.update(policy_timestamp_s=2.0),
            lambda m: m["images"][1].update(frame=999), lambda m: m["images"][2].update(timestamp=0.0),
            lambda m: m["world"].update(frame_id=3), lambda m: m["world"].update(timestamp_s=0),
            lambda m: m["world"]["population_evidence"].update(frame_id=6),
            lambda m: m["world"]["population_evidence"].update(active_population_ids=[7, 32]),
            lambda m: m["world"]["population_evidence"].update(registered_population_ids=[7, 7, 32]),
            lambda m: m["actors"][1].update(id=32), lambda m: m["rgb"]["radar_callback"].update(frame_id=0),
            lambda m: m["rgb"]["semantic_callback"].update(width=9),
            lambda m: m["rgb"]["depth_callback"].update(camera_matrix=np.zeros((4, 4)).tolist())]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(S.SnapshotError):
                S.unpack_snapshot(envelope(self.payload, mutate))

    def test_rejects_malformed_envelopes_and_no_pickle(self):
        for payload in (b"", self.payload[:47], b"X" + self.payload[1:], self.payload[:-1], self.payload + b"x",
                        self.payload[:8] + struct.pack(">Q", 2**64-1) + self.payload[16:], b"\x80\x04cos\nsystem\n."):
            with self.subTest(size=len(payload)), self.assertRaises(S.SnapshotError):
                S.unpack_snapshot(payload)
        for raw_json in (b'{"schema":1,"schema":2}', b'{"x":NaN}', b'null', b'{'):
            with self.subTest(json=raw_json), self.assertRaises(S.SnapshotError):
                S.unpack_snapshot(envelope(self.payload, raw_json=raw_json))

    def test_rejects_malformed_buffers_and_unbounded_dimensions(self):
        mutations = [lambda m: m["buffers"][0].update(dtype="O"), lambda m: m["buffers"][0].update(shape=[1000000, 1280, 4]),
            lambda m: m["buffers"][0].update(offset=1), lambda m: m["buffers"][0].update(nbytes=1),
            lambda m: m["buffers"][1].update(name="image_0"), lambda m: m["buffers"][1].update(dtype="f4"),
            lambda m: m["buffers"].pop(), lambda m: m["rgb"].update(width=1281),
            lambda m: m["source_pins"].update(bad="f" * 64), lambda m: m.update(schema="other")]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(S.SnapshotError):
                S.unpack_snapshot(envelope(self.payload, mutate))
        with self.assertRaises(S.SnapshotError):
            S.unpack_snapshot(self.payload, limits=S.Limits(max_payload_bytes=100))
        with self.assertRaises(S.SnapshotError):
            S.Limits(max_width=1281)

    def test_rejects_nonfinite_and_cpp_rounding(self):
        for mutate in (lambda m: m["actors"][0]["pose"].update(location=[0.1, 0., 0.]),
                       lambda m: m["actors"][0]["bbox"].update(extent=[-1., 0., 0.])):
            with self.assertRaises(S.SnapshotError):
                S.unpack_snapshot(envelope(self.payload, mutate))
        def corrupt(binary, metadata):
            spec = next(value for value in metadata["buffers"] if value["name"] == "radar")
            start = spec["offset"]
            return binary[:start] + np.array([np.nan], dtype=np.float64).tobytes() + binary[start + 8:]
        with self.assertRaises(S.SnapshotError):
            S.unpack_snapshot(envelope(self.payload, binary_mutate=corrupt))

    def test_pack_rejects_unknown_kwargs_and_nonprimitive_or_nonfinite(self):
        for kwargs in ({**self.kwargs, "live_client": object()}, {**self.kwargs, "identity": {"frame_id": 51, "live": object()}},
                       {**self.kwargs, "policy_timestamp_s": float("nan")}):
            with self.assertRaises(S.SnapshotError):
                S.pack_snapshot(kwargs, self.radar)
        for radar in (np.array([[object(), 2, 3]], dtype=object), np.zeros((2, 4)), np.array([[np.inf, 0, 0]])):
            with self.assertRaises(S.SnapshotError):
                S.pack_snapshot(self.kwargs, radar)
        with self.assertRaises(S.SnapshotError):
            S.pack_snapshot(self.kwargs, self.radar, limits=S.Limits(max_radar_points=1))
        self.kwargs["frozen_world"].get_actors()[1].id = 32
        with self.assertRaises(S.SnapshotError):
            S.pack_snapshot(self.kwargs, self.radar)


if __name__ == "__main__":
    unittest.main()
