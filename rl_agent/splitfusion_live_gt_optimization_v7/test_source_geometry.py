"""Adversarial CPU equivalence for request-local radar and semantic reuse.

The oracles are the sealed original numerical functions and full producer.
No live scene, inference, process worker, broker, or original file is written.
"""
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as F
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import sensor_metadata
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from . import source_geometry as C


def immutable(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def actor(matrix):
    return NS(get_transform=lambda: NS(get_inverse_matrix=lambda: matrix),
        bounding_box=NS(location=NS(x=0., y=0., z=0.), extent=NS(x=1., y=1., z=1.)))


def stats():
    return dict(semantic_calls=0, semantic_builds=0, semantic_reuses=0)


def semantic_values():
    # All 256 uint8 tags, repeated at the production 720x1280 shape.
    return np.tile(np.arange(256, dtype=np.uint8), (720, 5))


def probe(static, frame, boxes, geometry, depth, semantic, radar):
    return [np.isin(semantic, tags) for tags in static]


def mutate_probe(static, frame, boxes, geometry, depth, semantic, radar):
    result = [np.isin(semantic, static).copy()]
    semantic[0, 0] = 14
    result.append(np.isin(semantic, static).copy())
    semantic[-1, -1] = 15
    result.append(np.isin(semantic, static).copy())
    return result


def fallback_probe(static, frame, boxes, geometry, depth, semantic, radar):
    return [np.isin(semantic, np.asarray(static)), np.isin(semantic, static, invert=True),
        np.isin(semantic, static, kind="sort"), np.isin(semantic.copy(), static)]


def mask_mutation_probe(static, frame, boxes, geometry, depth, semantic, radar):
    result = np.isin(semantic, static)
    result.setflags(write=True)
    result[0, 0] = not result[0, 0]
    return np.isin(semantic, static)


class RadarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects = S._original()[0]
        cls.original = staticmethod(cls.objects["transform_world_to_actor_local"])

    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(order="C"), expected.tobytes(order="C"))

    def test_exact_float32_float64_bits_and_support_thresholds_across_actor_matrices(self):
        matrices = [np.eye(4), np.array([[.8, -.6, 0., 1e8], [.6, .8, 0., -1e8],
            [0., 0., 1., .125], [0., 0., 0., 1.]]),
            np.array([[.36, -.48, .8, -2.25], [.8, .6, 0., 7.5],
                [-.48, .64, .6, .875], [0., 0., 0., 1.]])]
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__):
                # Compute adjacent values in the input dtype, not after a
                # float64 nextafter has rounded back to the float32 boundary.
                adjacent_low = np.nextafter(dtype(1.5), dtype(0.))
                adjacent_high = np.nextafter(dtype(1.5), dtype(np.inf))
                self.assertLess(adjacent_low, dtype(1.5))
                self.assertGreater(adjacent_high, dtype(1.5))
                boundary = np.array([[x, y, z] for x in (0., -0., 1.5,
                    adjacent_low, adjacent_high, -1.5, 1e8, np.finfo(dtype).tiny)
                    for y,z in ((0., 0.), (.75, 1.), (1., -1.5))], dtype=dtype)
                random_points = np.random.default_rng(871).normal(size=(257, 3)).astype(dtype)
                points = immutable(np.vstack((boundary, random_points)))
                request = C.RadarRequest(points)
                reference_calls = mock.Mock(wraps=self.original)
                cached_count = C.clone(self.objects["radar_support_count"], dict(self.objects,
                    transform_world_to_actor_local=lambda p,a: request.transform(reference_calls, p, a)))
                for matrix in matrices:
                    observed_actor = actor(matrix)
                    self.assert_bits(request.transform(reference_calls, points, observed_actor), self.original(points, observed_actor))
                    for label, mode, margin in (("person", "radius", 1.), ("person", "box", 0.),
                            ("vehicle", "radius", -1.), ("vehicle", "box", 1.)):
                        arguments = dict(actor=observed_actor, label=label, radar_world_xyz=points,
                            margin_m=margin, person_support_mode=mode)
                        self.assertEqual(cached_count(**arguments), self.objects["radar_support_count"](**arguments))
                reference_calls.assert_not_called()
                self.assertEqual(request.builds, 1)
                self.assertGreater(request.reuses, 1)
                self.assertEqual(request.fallbacks, 0)
                request.close()
                self.assertIsNone(request.points)
                self.assertIsNone(request.homogeneous)

    def test_mutable_and_readonly_mutable_backings_fall_back_after_input_mutation(self):
        for backing in ("writeable", "readonly_owner", "bytearray"):
            with self.subTest(backing=backing):
                mutable = np.arange(18, dtype=np.float64).reshape(6, 3)
                if backing == "bytearray":
                    owner = bytearray(mutable.tobytes())
                    points = np.frombuffer(memoryview(owner).toreadonly(), np.float64).reshape(6, 3)
                    change = lambda: np.frombuffer(owner, np.float64).__setitem__(0, 99.)
                elif backing == "readonly_owner":
                    points = mutable.view()
                    points.setflags(write=False)
                    change = lambda: mutable.__setitem__((0, 0), 99.)
                else:
                    points = mutable
                    change = lambda: mutable.__setitem__((0, 0), 99.)
                self.assertFalse(C.immutable_bytes_owner(points))
                request = C.RadarRequest(points)
                original = mock.Mock(wraps=self.original)
                observed_actor = actor(np.eye(4))
                first = request.transform(original, points, observed_actor).copy()
                change()
                second = request.transform(original, points, observed_actor)
                self.assert_bits(second, self.original(points, observed_actor))
                self.assertFalse(np.array_equal(first, second))
                self.assertEqual(original.call_count, 2)
                self.assertEqual(request.builds, 0)
                self.assertIsNone(request.homogeneous)
                request.close()

    def test_foreign_object_layout_dtype_and_bound_retain_original_path(self):
        original_points = immutable(np.arange(18, dtype=np.float64).reshape(6, 3))
        class Subclass(np.ndarray):
            pass
        choices = [original_points.copy(), immutable(original_points.copy()),
            original_points[:, ::-1], original_points[::-1], original_points.view(Subclass),
            immutable(original_points.astype(np.float16)), immutable(original_points.astype(np.int64)),
            np.asfortranarray(original_points),
            immutable(np.empty((0, 3), np.float64)), immutable(np.zeros((100001, 3), np.float64))]
        for value in choices:
            with self.subTest(dtype=value.dtype, shape=value.shape, kind=type(value).__name__):
                request = C.RadarRequest(original_points if value is choices[1] else value)
                original = mock.Mock(wraps=self.original)
                observed_actor = actor(np.eye(4))
                self.assert_bits(request.transform(original, value, observed_actor), self.original(value, observed_actor))
                self.assertEqual(original.call_count, 1)
                self.assertEqual(request.builds, 0)
                request.close()

    def test_invalid_shapes_and_actor_failures_preserve_original_exception(self):
        for points in (np.ones((3,)), np.ones((3, 2)), np.ones((3, 4))):
            observed_actor = actor(np.eye(4))
            with self.assertRaises(Exception) as expected:
                self.original(points, observed_actor)
            with self.assertRaises(type(expected.exception)) as actual:
                C.RadarRequest(points).transform(self.original, points, observed_actor)
            self.assertEqual(str(actual.exception), str(expected.exception))
        points = immutable(np.ones((3, 3), np.float64))
        for observed_actor in (NS(), actor(np.ones((3, 3)))):
            with self.assertRaises(Exception) as expected:
                self.original(points, observed_actor)
            request = C.RadarRequest(points)
            with self.assertRaises(type(expected.exception)) as actual:
                request.transform(self.original, points, observed_actor)
            self.assertEqual(str(actual.exception), str(expected.exception))
            request.close()
            self.assertIsNone(request.homogeneous)

    def test_no_cached_actor_pose_or_cross_request_points(self):
        points = immutable(np.arange(18, dtype=np.float64).reshape(6, 3))
        matrix = np.eye(4)
        observed_actor = actor(matrix)
        request = C.RadarRequest(points)
        first = request.transform(self.original, points, observed_actor).copy()
        matrix[0, 3] = 123.
        second = request.transform(self.original, points, observed_actor)
        self.assert_bits(second, self.original(points, observed_actor))
        self.assertFalse(np.array_equal(first, second))
        request.close()
        changed = immutable(np.full(points.shape, -73., np.float64))
        next_request = C.RadarRequest(changed)
        self.assert_bits(next_request.transform(self.original, changed, observed_actor), self.original(changed, observed_actor))
        self.assertEqual(next_request.builds, 1)
        self.assertEqual(next_request.reuses, 0)
        next_request.close()


class StaticSemanticTests(unittest.TestCase):
    def invoke(self, function, semantic, tags):
        counts = stats()
        result = C.static_rows(function, counts, tags, None, None, None, None, semantic, None)
        return result, counts

    def test_all_256_tags_exact_and_one_result_reused_within_call(self):
        semantic = semantic_values()
        tags = [[14, 15, 16], [14, 15, 16], [14, 15, 16], [12, 13], [17, 18, 19]]
        actual, counts = self.invoke(probe, semantic, tags)
        expected = probe(tags, None, None, None, None, semantic, None)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)
            self.assertEqual(observed.dtype, np.bool_)
        self.assertIs(actual[0], actual[1])
        self.assertEqual(counts, dict(semantic_calls=5, semantic_builds=3, semantic_reuses=2))
        self.assertFalse(actual[0][0, 17])
        self.assertFalse(actual[0][0, 18])
        self.assertFalse(actual[0][0, 19])

    def test_changed_source_pixels_rebuild_instead_of_reusing_stale_membership(self):
        semantic = semantic_values()
        expected = mutate_probe([14, 15, 16], None, None, None, None, semantic.copy(), None)
        actual, counts = self.invoke(mutate_probe, semantic, [14, 15, 16])
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)
        self.assertEqual(counts, dict(semantic_calls=3, semantic_builds=3, semantic_reuses=0))

    def test_cached_result_cannot_be_mutated_or_made_writeable(self):
        values, _ = self.invoke(probe, semantic_values(), [[14, 15, 16], [14, 15, 16]])
        for value in (values[0], values[0].view()):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.setflags(write=True)
            with self.assertRaises(ValueError):
                value[0, 0] = True
        with self.assertRaises(ValueError):
            self.invoke(mask_mutation_probe, semantic_values(), [14, 15, 16])

    def test_other_numpy_signatures_fall_back_and_global_isin_unchanged(self):
        original = np.isin
        semantic = semantic_values()
        actual, counts = self.invoke(fallback_probe, semantic, [14, 15, 16])
        expected = fallback_probe([14, 15, 16], None, None, None, None, semantic, None)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)
        self.assertEqual(counts, stats())
        self.assertIs(np.isin, original)

    def test_each_call_owns_its_cache_and_failure_clears_retained_entries(self):
        semantic = semantic_values()
        first, one = self.invoke(probe, semantic, [[14, 15, 16], [14, 15, 16]])
        semantic[:] = 16
        second, two = self.invoke(probe, semantic, [[14, 15, 16], [14, 15, 16]])
        self.assertEqual(one, two)
        self.assertIsNot(first[0], second[0])
        self.assertTrue(second[0].all())
        self.assertFalse(first[0].all())
        try:
            self.invoke(mask_mutation_probe, semantic, [14, 15, 16])
        except ValueError as exc:
            trace = exc.__traceback__
            found = False
            while trace:
                if trace.tb_frame.f_code is C.static_rows.__code__:
                    self.assertEqual(trace.tb_frame.f_locals["cache"], {})
                    found = True
                trace = trace.tb_next
            self.assertTrue(found)
        else:
            self.fail("injected cached-mask mutation must refuse")


class FullProducerTests(unittest.TestCase):
    fixture = F.SourceGeometryTests.fixture

    def producer(self, cls, directory):
        return cls(static_catalogue_dir=directory / "catalogue", experiment_id="v7_exact_source_fixture",
            ego_id=999, dynamic_population_ids=[1, 2], population_evidence_sha256="d"*64, max_frames=4)

    def advance(self, kwargs, frame):
        stamp = frame * .05
        kwargs["identity"].update(frame_id=frame, capture_timestamp_ns=round(stamp * 1e9))
        kwargs["dynamic_snapshot_evidence"]["frame_id"] = frame
        kwargs["frozen_world"].frozen_frame_id, kwargs["frozen_world"].frozen_timestamp_s = frame, stamp
        for image in (kwargs["semantic_image"], kwargs["depth_image"]):
            image.frame, image.timestamp = frame, stamp
        kwargs["rgb"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)
        kwargs["rgb"]["radar_callback"] = sensor_metadata(kwargs["semantic_image"], receipt_raw_ns=1)

    def assert_exact(self, reference, candidate, expected, actual):
        self.assertEqual(A.canonical(A._json_value(reference.last_evidence)),
            A.canonical(A._json_value(candidate.last_evidence)))
        self.assertEqual(expected.provenance, actual.provenance)
        for name in ("frame", "actor_boxes", "actor_visibility", "static_candidates", "quarantine"):
            self.assertEqual(getattr(expected, name), getattr(actual, name))
        for name in ("depth_bgra", "raw_semantic", "original_v3_mask"):
            np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))
        a, b = E.prepare_target(expected), E.prepare_target(actual)
        self.assertEqual(a.content_sha256, b.content_sha256)
        for name in ("segmentation_png", "object_ignore_png"):
            self.assertEqual(a.target[name], b.target[name])
        self.assertFalse(candidate.last_array_reuse["retained_after_request"])
        self.assertIsNone(candidate._array_request)

    def test_two_changed_frames_preserve_inputs_targets_evidence_and_causal_history(self):
        before = H.sources()
        objects, _, static, _, _ = S._original()
        bindings = tuple((name, id(value)) for name,value in objects.items()), tuple((name,id(value)) for name,value in static.items())
        with self.fixture() as (directory, kwargs):
            reference = self.producer(S.LiveCompleteTargetProducer, directory)
            candidate = self.producer(C.source_producer(S.LiveCompleteTargetProducer), directory)
            for index in range(2):
                if index:
                    self.advance(kwargs, 101)
                    raw = np.frombuffer(kwargs["semantic_image"].raw_data, np.uint8).reshape(720, 1280, 4).copy()
                    raw[..., 2] = 16
                    kwargs["semantic_image"].raw_data = raw.tobytes()
                radar = np.array([[20., -3., 1.], [20., 4., 1.]], np.float64 if index == 0 else np.float32)
                if index:
                    radar += np.array([.25, -.125, .5])
                kwargs["radar_world_xyz"] = immutable(radar)
                with mock.patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns", return_value=123+index):
                    expected, actual = reference.build(**kwargs), candidate.build(**kwargs)
                self.assert_exact(reference, candidate, expected, actual)
                self.assertEqual(candidate.processed_frames, reference.processed_frames)
                self.assertEqual(candidate.last_array_reuse["radar_builds"], 1)
                self.assertGreater(candidate.last_array_reuse["radar_reuses"], 0)
                self.assertEqual(candidate.last_array_reuse["radar_fallbacks"], 0)
                self.assertEqual(candidate.last_array_reuse["semantic_builds"], 1)
                self.assertTrue(candidate.last_array_reuse["completed"])
        self.assertEqual(H.sources(), before)
        self.assertEqual(bindings, (tuple((name,id(value)) for name,value in objects.items()),
            tuple((name,id(value)) for name,value in static.items())))
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_bad_inputs_fail_before_acceleration_and_clear_request(self):
        with self.fixture() as (directory, kwargs):
            bads = [dict(radar_world_xyz=immutable(np.array([[np.nan, 0., 0.]]))),
                dict(radar_world_xyz=np.ones((1, 2))), dict(camera_inverse=np.eye(4)*2),
                dict(dynamic_snapshot_evidence=dict(kwargs["dynamic_snapshot_evidence"], registry_complete=False))]
            for changed in bads:
                with self.subTest(fields=list(changed)):
                    reference = self.producer(S.LiveCompleteTargetProducer, directory)
                    candidate = self.producer(C.source_producer(S.LiveCompleteTargetProducer), directory)
                    with self.assertRaises(Exception) as expected:
                        reference.build(**dict(kwargs, **changed))
                    with self.assertRaises(type(expected.exception)) as actual:
                        candidate.build(**dict(kwargs, **changed))
                    self.assertEqual(str(expected.exception), str(actual.exception))
                    self.assertIsNone(candidate._array_request)
                    self.assertFalse(candidate.last_array_reuse["retained_after_request"])
                    self.assertFalse(candidate.last_array_reuse["completed"])
                    self.assertEqual(candidate.last_array_reuse["radar_builds"], 0)

    def test_midbuild_exception_drops_homogeneous_cache_before_valid_retry(self):
        with self.fixture() as (directory, kwargs):
            kwargs["radar_world_xyz"] = immutable(kwargs["radar_world_xyz"].astype(np.float64))
            candidate = self.producer(C.source_producer(S.LiveCompleteTargetProducer), directory)
            reference = self.producer(S.LiveCompleteTargetProducer, directory)
            original = candidate.objects["build_object_rows"]
            failure = RuntimeError("injected after radar cache initialization")
            held = []
            def fail(**arguments):
                candidate.objects["transform_world_to_actor_local"](arguments["radar_world_xyz"],
                    list(kwargs["frozen_world"].get_actors())[0])
                held.append(candidate._array_request)
                raise failure
            candidate.objects["build_object_rows"] = fail
            with self.assertRaises(RuntimeError) as caught:
                candidate.build(**kwargs)
            self.assertIs(caught.exception, failure)
            self.assertIsNone(candidate._array_request)
            self.assertEqual(candidate.last_array_reuse["radar_builds"], 1)
            self.assertFalse(candidate.last_array_reuse["completed"])
            self.assertIsNone(held[0].points)
            self.assertIsNone(held[0].homogeneous)
            candidate.objects["build_object_rows"] = original
            with mock.patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns", return_value=123):
                expected, actual = reference.build(**kwargs), candidate.build(**kwargs)
            self.assert_exact(reference, candidate, expected, actual)
            self.assertEqual(candidate.last_array_reuse["radar_builds"], 1)


if __name__ == "__main__":
    unittest.main()
