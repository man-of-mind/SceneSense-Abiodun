"""Three-pass CPU tests of complete source construction, without a CARLA server."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import carla
import cv2
import numpy as np
import torch

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import sensor_metadata
from . import source_geometry as S


class Actors(tuple):
    def filter(self, pattern):
        import fnmatch
        return tuple(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


def actor(actor_id, kind, x, y, *, pitch=0., roll=0.):
    transform = carla.Transform(carla.Location(x, y, 0), carla.Rotation(pitch=pitch, roll=roll, yaw=15))
    bbox = carla.BoundingBox(carla.Location(0, 0, 1), carla.Vector3D(.6, .4, 1))
    return SimpleNamespace(id=actor_id, type_id=kind, bounding_box=bbox,
        get_transform=lambda: transform, get_location=lambda: transform.location,
        get_velocity=lambda: carla.Vector3D(.1, 0, 0))


def encode_depth(value=20.):
    number = int(round(value / 1000. * (256**3 - 1)))
    bgra = np.empty((720, 1280, 4), np.uint8)
    bgra[:] = [number // 65536, (number // 256) % 256, number % 256, 255]
    return bgra


class SourceGeometryTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, pitch=0., enabled=True):
        with tempfile.TemporaryDirectory(prefix="source-geometry-test-") as temporary:
            directory = Path(temporary)
            env = []
            for i, name in enumerate(("Car", "Truck", "Bus"), 10):
                transform = carla.Transform(carla.Location(20 + i, 8, 1), carla.Rotation(yaw=15))
                bbox = carla.BoundingBox(transform.location, carla.Vector3D(2, 1, 1))
                bbox.rotation = transform.rotation
                env.append(SimpleNamespace(id=i, type=getattr(carla.CityObjectLabel, name),
                    name=f"Static{name}", transform=transform, bounding_box=bbox))
            world = SimpleNamespace(get_environment_objects=lambda label: [v for v in env if v.type == label],
                get_map=lambda: SimpleNamespace(name="Town10HD_Opt", to_opendrive=lambda: "synthetic map geometry"),
                get_snapshot=lambda: SimpleNamespace(frame=1, timestamp=SimpleNamespace(elapsed_seconds=.05)))
            S.capture_catalogue(world, directory / "catalogue", semantic_labels=[getattr(carla.CityObjectLabel, n)
                for n in ("Car", "Truck", "Bus")], required_semantic_classes=["Car", "Truck", "Bus"],
                enabled_state_by_id={v.id: enabled for v in env}, selection_contract="CPU_SYNTHETIC_FULL_CATALOGUE")
            actors = Actors((actor(1, "walker.pedestrian.0001", 20., -3., pitch=pitch, roll=8.),
                             actor(2, "vehicle.test", 20., 4.)))
            frozen = SimpleNamespace(frozen_frame_id=100, frozen_timestamp_s=5., frozen_observed_raw_ns=10,
                                     get_actors=lambda: actors)
            pose = carla.Transform()
            raw = np.zeros((720, 1280, 4), np.uint8); raw[..., 2] = 14
            # Preserve vehicle protection while giving the person native depth support.
            raw[:, :640, 2] = 0
            depth = SimpleNamespace(raw_data=encode_depth().tobytes(), width=1280, height=720,
                                    frame=100, timestamp=5., transform=pose)
            sem = SimpleNamespace(raw_data=raw.tobytes(), width=1280, height=720,
                                  frame=100, timestamp=5., transform=pose)
            rgb = sensor_metadata(sem, receipt_raw_ns=1)
            rgb["radar_callback"] = sensor_metadata(sem, receipt_raw_ns=1)
            identity = dict(run_id="run", cell_id="cell", stream_id="ue", frame_id=100,
                            action_id=None, profile_id=None, capture_timestamp_ns=5000000000)
            kwargs = dict(identity=identity, rgb=rgb, semantic_image=sem, depth_image=depth,
                frozen_world=frozen, camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                radar_world_xyz=np.array([[20., -3., 1.], [20., 4., 1.]], np.float32),
                dynamic_snapshot_evidence=dict(frame_id=100, registry_complete=True,
                    no_unregistered_spawn=True, owned_snapshot_complete=True, active_population_ids=[1, 2]))
            yield directory, kwargs

    def producer(self, directory):
        return S.LiveCompleteTargetProducer(static_catalogue_dir=directory / "catalogue",
            experiment_id="prospective_cpu_test", ego_id=999, dynamic_population_ids=[1, 2],
            population_evidence_sha256="d"*64, max_frames=1000)

    def test_pass1_source_pins_and_complete_target(self):
        self.assertEqual(len(S.verify_sources()), 9)
        with self.fixture() as (directory, kwargs):
            p = self.producer(directory); value = p.build(**kwargs)
            self.assertEqual(len(value.actor_boxes), 2)
            self.assertEqual(len(value.actor_visibility), 2)
            self.assertGreater(np.count_nonzero(value.original_v3_mask == 2), 0)
            self.assertEqual(value.provenance["capture_identity"], kwargs["identity"])
            result = A.build_complete_v010(value)
            self.assertEqual(result["segmentation"].shape, (432, 768))
            self.assertFalse(result["live_qualified"])
            self.assertFalse(torch.cuda.is_initialized())

    def test_pass1_static_literal_function_disk_oracle(self):
        """Original static audit reads real PNG/NPZ fixture files, unchanged."""
        with self.fixture() as (directory, kwargs):
            p = self.producer(directory); value = p.build(**kwargs)
            f = value.frame
            for path in (directory / f["experiment_id"] / "depth", directory / "semantic", directory / "radar"):
                path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f["experiment_id"] / "depth" / f"{f['sample_id']}.png"), value.depth_bgra)
            cv2.imwrite(str(directory / f["instance_raw_path"]), value.raw_semantic)
            np.savez(directory / f["radar_points_path"], world_xyz=kwargs["radar_world_xyz"])
            # Rebind only original module globals VIEW; reads use actual cv2/np.
            import types
            namespace = dict(p.static, VIEW=directory)
            fn = p.static["audit_visibility"]
            original = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
            expected, _, _ = original([f], {f["sample_id"]: value.actor_boxes}, p.geometry)
            self.assertEqual(len(expected), 3)
            self.assertEqual(value.static_candidates, expected)

    def test_pass1_person_literal_interval_oracle(self):
        with self.fixture(pitch=17.) as (directory, kwargs):
            p = self.producer(directory); value = p.build(**kwargs)
            mask = p.semantic["carla_semantic_tags_to_training_mask"](value.raw_semantic)
            mask[mask == 2] = 0
            reference = A.original_v3_person_mask(vehicle_mask=mask,
                depth_m=p.visibility.decode_depth_bgra(value.depth_bgra),
                person_visibility=[r for r in value.actor_visibility if r["label"] == "person"])
            np.testing.assert_array_equal(value.original_v3_mask, reference["mask"])
            actor_value = list(kwargs["frozen_world"].get_actors())[0]
            _, corners = p.objects["actor_bbox_world_points"](actor_value)
            observed = next(r for r in p.last_evidence["dynamic_corners"] if r["actor_id"] == actor_value.id)
            np.testing.assert_array_equal(observed["corners"], corners)
            # Pitch/roll affect depth intervals; it must not be a yaw-only AVO box.
            self.assertGreater(float(np.ptp(corners[:, 0])), 1.2)

    def test_pass2_wrong_or_incomplete_snapshot_refused(self):
        with self.fixture() as (directory, kwargs):
            for key in ("registry_complete", "no_unregistered_spawn", "owned_snapshot_complete"):
                p = self.producer(directory)
                bad = dict(kwargs, dynamic_snapshot_evidence=dict(kwargs["dynamic_snapshot_evidence"], **{key:False}))
                with self.assertRaises(S.SourceGeometryError): p.build(**bad)
            p = self.producer(directory)
            original = kwargs["frozen_world"].get_actors
            kwargs["frozen_world"].get_actors = lambda: Actors(original()[:1])
            with self.assertRaises(S.SourceGeometryError): p.build(**kwargs)

    def test_pass2_source_array_pose_time_fail_closed(self):
        with self.fixture() as (directory, kwargs):
            for changed in (dict(camera_inverse=np.eye(4)*2),
                            dict(radar_world_xyz=np.array([[np.nan, 0, 0]]))):
                with self.assertRaises(S.SourceGeometryError): self.producer(directory).build(**dict(kwargs, **changed))
            kwargs["depth_image"].timestamp = 5.1
            with self.assertRaises(ValueError): self.producer(directory).build(**kwargs)

    def test_pass2_disabled_static_not_silently_dropped(self):
        with self.fixture(enabled=False) as (directory, _):
            with self.assertRaises(S.SourceGeometryError): self.producer(directory)

    def test_pass2_owned_despawn_admitted_unknown_spawn_rejected(self):
        with self.fixture() as (directory, kwargs):
            actors = kwargs["frozen_world"].get_actors()
            kwargs["frozen_world"].get_actors = lambda: Actors(actors[:1])
            kwargs["dynamic_snapshot_evidence"]["active_population_ids"] = [1]
            result = self.producer(directory).build(**kwargs)
            self.assertEqual([r["gt_actor_id"] for r in result.actor_boxes], ["1"])
            kwargs["dynamic_snapshot_evidence"]["active_population_ids"] = [1, 12345]
            with self.assertRaises(S.SourceGeometryError): self.producer(directory).build(**kwargs)

    def test_pass2_complete_geometry_retains_nonprojected_actor(self):
        with self.fixture() as (directory, kwargs):
            actors = kwargs["frozen_world"].get_actors()
            kwargs["frozen_world"].get_actors = lambda: Actors((actor(1, "walker.pedestrian.0001", -20., 0.), actors[1]))
            p = self.producer(directory); value = p.build(**kwargs)
            self.assertEqual(len(value.actor_boxes), 1)
            self.assertEqual({row["actor_id"] for row in p.last_evidence["dynamic_corners"]}, {1, 2})
            self.assertEqual(np.count_nonzero(value.original_v3_mask == 2), 0)

    def test_pass2_mutated_complete_array_refused(self):
        with self.fixture() as (directory, kwargs):
            value = self.producer(directory).build(**kwargs)
            value.depth_bgra[0, 0, 0] ^= np.uint8(1)
            with self.assertRaises(A.TargetInputError): A.build_complete_v010(value)

    def test_pass2_replenishment_requires_exact_owned_registration(self):
        with self.fixture() as (directory, kwargs):
            old = kwargs["frozen_world"].get_actors()
            kwargs["frozen_world"].get_actors = lambda: Actors((*old, actor(3, "vehicle.test_new", 25., 7.)))
            kwargs["dynamic_snapshot_evidence"]["active_population_ids"] = [1, 2, 3]
            p = self.producer(directory)
            with self.assertRaises(S.SourceGeometryError): p.build(**kwargs)
            p.register_population([1, 2, 3], "a"*64, 100)
            result = p.build(**kwargs)
            self.assertEqual(len(result.actor_boxes), 3)
            self.assertEqual(p.population, {1, 2, 3})
            self.assertEqual(p.last_evidence["new_population_registration"]["evidence_sha256"], "a"*64)
            with self.assertRaises(S.SourceGeometryError): p.register_population([1, 2, 3], "a"*64, 100)

    def test_pass2_registration_wrong_frame_digest_or_retroactive_population(self):
        with self.fixture() as (directory, kwargs):
            for ids, digest, frame in (([1], "a"*64, 100), ([1, 2, 2], "a"*64, 100),
                                      ([1, 2], "z"*64, 100), ([1, 2], "a"*64, -1)):
                with self.assertRaises(S.SourceGeometryError): self.producer(directory).register_population(ids, digest, frame)
            p = self.producer(directory); p.register_population([1, 2, 3], "a"*64, 101)
            with self.assertRaises(S.SourceGeometryError): p.build(**kwargs)

    def test_pass3_causal_history_repeated_identity_and_transport(self):
        with self.fixture() as (directory, kwargs):
            p = self.producer(directory); value = p.build(**kwargs)
            with self.assertRaises(S.SourceGeometryError): p.build(**kwargs)
            from .evaluation import prepare_target
            from .target_transport import write_bundle, read_bundle
            prepared = prepare_target(value)
            write_bundle(directory / "handoff", identity=kwargs["identity"], inputs=value, prepared=prepared)
            restored = read_bundle(directory / "handoff", expected_identity=kwargs["identity"], timeout_s=.05,
                                   verify_native_inputs=True)
            self.assertEqual(restored.inputs.provenance["input_content_sha256"], value.provenance["input_content_sha256"])
            self.assertEqual(restored.prepared.content_sha256, prepared.content_sha256)

    def test_pass3_source_incremental_history_matches_every_original_prefix(self):
        with self.fixture() as (directory, kwargs):
            p = self.producer(directory); prior = []
            for index in range(5):
                fid, stamp = 100 + index, 5. + index * .05
                kwargs["identity"].update(frame_id=fid, capture_timestamp_ns=int(stamp*1e9))
                kwargs["dynamic_snapshot_evidence"]["frame_id"] = fid
                kwargs["frozen_world"].frozen_frame_id = fid
                kwargs["frozen_world"].frozen_timestamp_s = stamp
                for image in (kwargs["semantic_image"], kwargs["depth_image"]):
                    image.frame, image.timestamp = fid, stamp
                kwargs["rgb"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
                kwargs["rgb"]["radar_callback"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
                value = p.build(**kwargs)
                actual = p.last_evidence["conflicts"]
                expected = S.conflicts.derive_live_quarantine(frame=value.frame, actor_boxes=value.actor_boxes,
                    actor_visibility=value.actor_visibility, static_candidates=value.static_candidates,
                    static_catalogue=p.catalogue, prior_static_candidates=prior, provenance=actual["provenance"])
                self.assertEqual(value.quarantine, expected["quarantine"])
                self.assertEqual(actual["static_static_adjudications"], expected["evidence"]["static_static_adjudications"])
                self.assertEqual(actual["causal_static_history"]["frames"], index + 1)
                self.assertFalse(actual["causal_static_history"]["history_discarded"])
                prior.extend(value.static_candidates)


if __name__ == "__main__": unittest.main()
