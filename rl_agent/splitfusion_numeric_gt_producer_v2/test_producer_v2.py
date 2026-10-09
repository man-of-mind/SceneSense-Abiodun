"""Focused CPU tests for the exact frame-local reuse (no CARLA; the corpus parity is ``verify_parity``).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/python3 -m unittest -v rl_agent.splitfusion_numeric_gt_producer_v2.test_producer_v2
"""
import unittest

import numpy as np

from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O

from . import exact_frame as X


def camera():
    k = np.array([[369.5041722813606, 0, 640.], [0, 369.5041722813606, 360.], [0, 0, 1.]])
    return np.eye(4), k


def box(center, half):
    cx, cy, cz = center
    hx, hy, hz = half
    return np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.vis = O.bound().visibility
        self.mine = X.make_visibility_from_corners(self.vis)
        rng = np.random.default_rng(3)
        self.depth = (rng.uniform(2, 60, (720, 1280))).astype(np.float32)

    def compare(self, corners, distance=20.0, shape=(720, 1280), model=(768, 432), depth=None):
        inv, k = camera()
        depth = self.depth if depth is None else depth
        kw = dict(distance_m=distance, width=shape[1], height=shape[0], model_width=model[0], model_height=model[1])
        try:
            expected = self.vis.visibility_from_corners(corners, inv, k, depth, **kw)
        except ValueError as error:
            with self.assertRaises(ValueError) as caught:
                self.mine(corners, inv, k, depth, **kw)
            self.assertEqual(str(caught.exception), str(error))
            return "refused"
        got = self.mine(corners, inv, k, depth, **kw)
        self.assertEqual(got[0], expected[0])
        self.assertTrue(np.array_equal(got[1], expected[1]))
        return "equal"

    def test_random_boxes_including_borders_and_partial(self):
        rng = np.random.default_rng(11)
        outcomes = []
        for _ in range(400):
            center = (rng.uniform(-5, 60), rng.uniform(-60, 60), rng.uniform(-8, 8))
            half = tuple(rng.uniform(0.1, 6, 3))
            outcomes.append(self.compare(box(center, half), distance=float(rng.uniform(1, 50))))
        self.assertGreater(outcomes.count("equal"), 150)
        self.assertGreater(outcomes.count("refused"), 10)

    def test_full_frame_and_edge_rows(self):
        self.compare(box((2.0, 0.0, 0.0), (0.5, 50.0, 50.0)))       # covers the whole image
        self.compare(box((20.0, 34.0, 0.0), (1.0, 1.0, 1.0)))       # clipped at the right edge
        self.compare(box((20.0, 0.0, 19.4), (1.0, 1.0, 1.0)))       # clipped at the top edge

    def test_non_production_geometry_falls_back_to_original(self):
        depth = self.depth[:360, :640].copy()
        inv, k = camera()
        k = k.copy(); k[0, 2], k[1, 2] = 320., 180.
        kw = dict(distance_m=10.0, width=640, height=360, model_width=384, model_height=216)
        expected = self.vis.visibility_from_corners(box((10.0, 0.0, 0.0), (1.0, 1.0, 1.0)), inv, k, depth, **kw)
        got = self.mine(box((10.0, 0.0, 0.0), (1.0, 1.0, 1.0)), inv, k, depth, **kw)
        self.assertEqual(got[0], expected[0])


class ReuseTests(unittest.TestCase):
    def test_bbox_cache_copies_and_restores_on_failure(self):
        objects = {"actor_bbox_world_points": lambda actor: (np.array([1.0, 2.0, 3.0]), np.ones((8, 3)))}
        original = objects["actor_bbox_world_points"]
        actor = object()
        with self.assertRaises(KeyError):
            with X.frame_bbox_cache(objects) as calls:
                c1, k1 = objects["actor_bbox_world_points"](actor)
                c1[0] = 99.0                                     # caller mutation must not leak
                c2, _ = objects["actor_bbox_world_points"](actor)
                self.assertEqual(c2[0], 1.0)
                self.assertEqual(calls, [2, 1])
                raise KeyError("simulated failure")
        self.assertIs(objects["actor_bbox_world_points"], original)

    def test_worker_switch_is_reversible(self):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
        from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
        from . import worker_switch as W
        from .source_v2 import NumericSourceProducerV2
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OW
        W.install()
        try:
            self.assertIs(D.NS.NumericSourceProducer, NumericSourceProducerV2)
            self.assertIs(OW.NumericSourceProducer, NumericSourceProducerV2)
            with self.assertRaises(RuntimeError):
                W.install()
        finally:
            W.uninstall()
        self.assertIs(D.NS, NS)
        self.assertIs(OW.NumericSourceProducer, NS.NumericSourceProducer)


if __name__ == "__main__":
    unittest.main()
