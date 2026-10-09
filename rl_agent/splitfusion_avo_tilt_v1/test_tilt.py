"""Focused tests of the tilt-aware pedestrian AVO candidate (CPU only, synthetic ray-cast scenes).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest -v \\
      rl_agent.splitfusion_avo_tilt_v1.test_tilt
"""
import math
import types
import unittest

import numpy as np

import carla

from data_collection.route_b_publication_actor_volume_visibility_v1 import core
from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as QA
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import geometry as G, quality as Q, switch

W, H = 640, 360
K = LG.native_intrinsics(W, H)
CAMERA = np.eye(4)
CAMERA[:3, 3] = (0.0, 0.0, 1.6)              # sensor axes == world axes: x forward, y right, z up
INVERSE = np.linalg.inv(CAMERA)
EXTENT = (0.188, 0.188, 0.92)
FRAME, STAMP = 7, 12.5


def actor(actor_id, x, y, *, yaw=0.0, pitch=0.0, roll=0.0, z=None, extent=EXTENT):
    transform = carla.Transform(carla.Location(x=x, y=y, z=extent[2] + 0.01 if z is None else z),
                                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
    box = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.0), carla.Vector3D(*extent))
    return types.SimpleNamespace(id=actor_id, get_transform=lambda t=transform: t, bounding_box=box)


def world(actors):
    view = types.SimpleNamespace(filter=lambda pattern, values=list(actors): values)
    return types.SimpleNamespace(frozen_frame_id=FRAME, get_actors=lambda: view)


def render(actors):
    """Planar depth by ray casting every actor's CARLA world box (slab test in its frame) and the ground z=0."""
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    direction = np.stack([np.ones_like(cols), (cols - K[0, 2]) / K[0, 0], -(rows - K[1, 2]) / K[1, 1]], -1)
    origin = CAMERA[:3, 3]
    depth = np.full((H, W), 1000.0)
    ground = -origin[2] / direction[..., 2]
    depth = np.where((direction[..., 2] < 0) & (ground > 0), np.minimum(depth, ground), depth)
    for a in actors:
        m = np.asarray(a.get_transform().get_matrix())
        r, c = m[:3, :3], m[:3, 3]
        o, d = r.T @ (origin - c), direction @ r              # ray in the box frame
        half = np.asarray([a.bounding_box.extent.x, a.bounding_box.extent.y, a.bounding_box.extent.z])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        near = np.max(np.minimum(t1, t2), -1)
        far = np.min(np.maximum(t1, t2), -1)
        hit = (far >= near) & (near > 0)
        depth = np.where(hit, np.minimum(depth, near), depth)  # direction x == 1, so t is planar depth
    return depth.astype(np.float32)


def rows_and_score(actors, scorer=Q.score_live_avo_people, builder=Q.frozen_people_rows, rows=None):
    rows = builder(world(actors), frame_id=FRAME, timestamp=STAMP, camera_matrix=CAMERA, camera_inverse=INVERSE,
                   intrinsics=K, width=W, height=H) if rows is None else rows
    return rows, scorer(frame_id=FRAME, timestamp=STAMP, depth_frame_id=FRAME, depth_timestamp=STAMP,
                        depth_m=render(actors), camera_matrix=CAMERA, camera_inverse=INVERSE, intrinsics=K,
                        raw_actor_rows=rows, width=W, height=H)


def qualified(result):
    return {r["gt_actor_id"]: r for r in result["qualified"]}


class Rotation(unittest.TestCase):
    def test_upright_path_is_the_frozen_yaw_rotation_exactly(self):
        rotation, path = G.box_rotation(yaw_deg=33.0, pitch_deg=1e-6, roll_deg=-1e-6,
                                        actor_rotation_matrix=[0.0] * 9, bbox_rotation_deg=(0, 0, 0))
        self.assertEqual(path, G.PATH_UPRIGHT)
        self.assertTrue(np.array_equal(rotation, core.yaw_rotation(33.0)))

    def test_tilted_corners_match_carla_world_vertices_for_signed_tilts(self):
        for pitch, yaw, roll in ((-0.03154, 14.46, -0.01056), (0.5, -120.0, 2.0), (-3.0, 179.0, 4.0), (6.0, 45.0, -6.0)):
            a = actor(1, 10.0, 2.0, yaw=yaw, pitch=pitch, roll=roll)
            t = a.get_transform()
            rotation, path = G.box_rotation(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                actor_rotation_matrix=np.asarray(t.get_matrix())[:3, :3].reshape(9), bbox_rotation_deg=(0, 0, 0))
            self.assertEqual(path, G.PATH_TILTED)
            mine = G.oriented_box_corners(np.asarray(t.get_matrix())[:3, 3], EXTENT, rotation)
            reference = np.asarray([[v.x, v.y, v.z] for v in a.bounding_box.get_world_vertices(t)])
            nearest = [np.min(np.linalg.norm(reference - c, axis=1)) for c in mine]
            self.assertLess(max(nearest), 1e-5, (pitch, yaw, roll))
            local = G.actor_local_points(mine, np.asarray(t.get_matrix())[:3, 3], rotation)
            # CARLA's matrix is float32-derived: orthonormal to ~1e-8, so the round trip is exact to ~1e-8 m
            self.assertLess(np.max(np.abs(np.abs(local) - np.asarray(EXTENT))), 1e-6)

    def test_invalid_rotation_evidence_fails_closed(self):
        good = np.asarray(actor(1, 5, 0, yaw=10, pitch=2, roll=1).get_transform().get_matrix())[:3, :3]
        cases = {
            "nonfinite": dict(actor_rotation_matrix=np.r_[good.reshape(9)[:8], np.nan]),
            "not a proper rotation": dict(actor_rotation_matrix=(good * 1.01).reshape(9)),
            "is not a proper rotation$": dict(actor_rotation_matrix=(good @ np.diag([1, 1, -1])).reshape(9)),
            "differs from its frozen": dict(actor_rotation_matrix=np.asarray(actor(1, 5, 0, yaw=10, pitch=-2, roll=1)
                                            .get_transform().get_matrix())[:3, :3].reshape(9)),
            "bounding-box rotation": dict(actor_rotation_matrix=good.reshape(9), bbox_rotation_deg=(0, 1.0, 0)),
            "9 values": dict(actor_rotation_matrix=good.reshape(9)[:6]),
        }
        for message, kwargs in cases.items():
            with self.subTest(message), self.assertRaisesRegex(LiveGtContractError, message):
                G.box_rotation(yaw_deg=10, pitch_deg=2, roll_deg=1, **dict(dict(bbox_rotation_deg=(0, 0, 0)), **kwargs))
        with self.assertRaisesRegex(LiveGtContractError, "nonfinite actor pitch"):
            G.box_rotation(yaw_deg=10, pitch_deg=math.inf, roll_deg=1, actor_rotation_matrix=good.reshape(9),
                           bbox_rotation_deg=(0, 0, 0))

    def test_missing_or_foreign_row_evidence_fails_closed(self):
        rows, _ = rows_and_score([actor(1, 8, 0, pitch=2)])
        for key in ("object_actor_rotation_matrix", "object_pitch_deg", "geometry_version"):
            broken = [{k: v for k, v in rows[0].items() if k != key}]
            with self.subTest(key), self.assertRaises(LiveGtContractError):
                rows_and_score([actor(1, 8, 0, pitch=2)], rows=broken)
        with self.assertRaisesRegex(LiveGtContractError, "different AVO geometry version"):
            rows_and_score([actor(1, 8, 0, pitch=2)], rows=[dict(rows[0], geometry_version="other")])


class Containment(unittest.TestCase):
    def test_tilted_face_point_inside_tilted_box_outside_upright_box(self):
        a = actor(1, 10.0, 0.0, pitch=6.0)
        m = np.asarray(a.get_transform().get_matrix())
        rotation = m[:3, :3]
        # face points within the 0.05 m tolerance near the top, on both faces of the tilted box
        points = np.stack([m[:3, 3] + rotation @ np.asarray([s * (EXTENT[0] + 0.04), 0.0, 0.9 * EXTENT[2]])
                           for s in (1.0, -1.0)])
        inside, _ = core.inside_actor_volume(G.actor_local_points(points, m[:3, 3], rotation), EXTENT)
        upright, _ = core.inside_actor_volume(core.actor_local_points(points, m[:3, 3], 0.0), EXTENT)
        self.assertEqual(inside.tolist(), [True, True])
        self.assertFalse(bool(np.all(upright)))                  # the yaw-only box loses one tilted face

    def test_ground_rule_is_applied_in_the_actor_frame(self):
        a = actor(1, 10.0, 0.0, roll=5.0)
        m = np.asarray(a.get_transform().get_matrix())
        low = m[:3, 3] + m[:3, :3] @ np.asarray([0.0, 0.0, -EXTENT[2] + 0.01])
        high = m[:3, 3] + m[:3, :3] @ np.asarray([0.0, 0.0, -EXTENT[2] + 0.05])
        retained, volume = core.inside_actor_volume(G.actor_local_points(np.stack([low, high]), m[:3, 3], m[:3, :3]),
                                                    EXTENT)
        self.assertEqual(volume.tolist(), [True, True])
        self.assertEqual(retained.tolist(), [False, True])

    def test_overlapping_pedestrians_ownership_uses_each_box_frame(self):
        tilted, upright = actor(1, 10.0, 0.0, pitch=7.0), actor(2, 10.0, 0.30)
        boxes = []
        for a in (tilted, upright):
            m = np.asarray(a.get_transform().get_matrix())
            boxes.append(dict(key=str(a.id), centre=m[:3, 3], extent=EXTENT, rotation=m[:3, :3]))
        # points along the segment between the two centres, inside both tolerance volumes
        points = np.linspace(boxes[0]["centre"], boxes[1]["centre"], 9)[2:7]
        first = G.assign_competing_pedestrians(points, "1", boxes)
        second = G.assign_competing_pedestrians(points, "2", boxes)
        self.assertFalse(np.any(first["owned"] & second["owned"]))          # each point has at most one owner
        self.assertEqual(first["competing_actor_boxes"], second["competing_actor_boxes"])
        for i, p in enumerate(points):
            d = [float(np.linalg.norm(G.actor_local_points(p[None], b["centre"], b["rotation"]) / np.asarray(EXTENT)))
                 for b in boxes]
            contains = [bool(core.inside_actor_volume(G.actor_local_points(p[None], b["centre"], b["rotation"]),
                                                      EXTENT)[0][0]) for b in boxes]
            if all(contains):                                      # exact ties go to the smaller key ("1")
                self.assertEqual(bool(first["owned"][i]), d[0] <= d[1])


class EndToEnd(unittest.TestCase):
    def test_old_refuses_and_candidate_scores_positive_and_negative_tilt(self):
        for pitch, roll in ((-0.03154, -0.01056), (0.03154, 0.01056), (6.0, 0.0), (0.0, -6.0)):
            scene = [actor(117, 6.5, 0.3, yaw=14.46, pitch=pitch, roll=roll), actor(131, 5.0, -1.5, yaw=-0.45)]
            with self.subTest(pitch=pitch, roll=roll):
                with self.assertRaisesRegex(LiveGtContractError, "TILTED_SCOREABLE_TARGET"):
                    rows_and_score(scene, scorer=QA.score_live_avo_people, builder=LG.frozen_people_rows)
                _, result = rows_and_score(scene)
                q = qualified(result)
                self.assertEqual(set(q), {"117", "131"})
                _, alone = rows_and_score(scene[:1])
                self.assertGreaterEqual(qualified(alone)["117"]["avo"], 0.9)   # unoccluded rendered box is supported
                self.assertEqual(q["117"]["algorithm_version"], G.GEOMETRY_VERSION)
                self.assertEqual([t["gt_actor_id"] for t in result["avo_geometry"]["tilted_people"]], ["117"])

    def test_tilt_aware_support_exceeds_yaw_only_support_for_a_visible_tilt(self):
        scene = [actor(1, 6.5, 0.0, pitch=8.0)]
        rows, result = rows_and_score(scene)
        forced = [dict(r, object_pitch_deg=0.0, object_roll_deg=0.0) for r in rows]   # yaw-only counterfactual
        _, yaw_only = rows_and_score(scene, rows=forced)
        self.assertGreater(qualified(result)["1"]["retained_actor_point_count"],
                           qualified(yaw_only)["1"]["retained_actor_point_count"])

    def test_upright_scene_identical_to_frozen_scorer(self):
        scene = [actor(1, 12.0, 0.4, yaw=20.0), actor(2, 12.2, 0.7, yaw=-60.0), actor(3, 45.0, 0.0),
                 actor(4, 6.0, 3.0, yaw=90.0)]
        old_rows, old = rows_and_score(scene, scorer=QA.score_live_avo_people, builder=LG.frozen_people_rows)
        new_rows, new = rows_and_score(scene)
        self.assertEqual(old["structural_ignored"], new["structural_ignored"])
        self.assertEqual([{k: v for k, v in r.items() if k != "algorithm_version"} for r in old["qualified"]],
                         [{k: v for k, v in r.items() if k != "algorithm_version"} for r in new["qualified"]])
        self.assertEqual(old_rows, [{k: v for k, v in r.items() if k not in Q.ROW_FIELDS} for r in new_rows])

    def test_visibility_boundaries_use_tilt_correct_projection(self):
        # image edge: projected box centre from tilted corners; 40-m range (actor origin, unchanged); 12-px area
        edge = actor(1, 6.0, 9.6, pitch=4.0, roll=-3.0)
        far = actor(2, 40.5, 0.0, pitch=4.0)
        tiny = actor(3, 30.0, 0.0, pitch=-3.0, extent=(0.02, 0.02, 0.05), z=0.06)
        rows, result = rows_and_score([edge, far, tiny])
        by_id = {r["gt_actor_id"]: r for r in rows}
        m = np.asarray(edge.get_transform().get_matrix())
        extent = [by_id["1"][f"gt_extent_{a}_m"] for a in "xyz"]          # CARLA's float32 extent
        manual = core.projected_boxes(G.oriented_box_corners(m[:3, 3], extent, m[:3, :3]), INVERSE, K, width=W, height=H)
        self.assertAlmostEqual(by_id["1"]["gt_center_x"], manual["clipped_bbox_x"] + manual["clipped_bbox_w"] / 2, places=12)
        self.assertAlmostEqual(by_id["1"]["gt_bbox_area_px"], manual["clipped_projected_area_px"], places=12)
        structural = {r["gt_actor_id"]: r["reasons"] for r in result["structural_ignored"]}
        self.assertIn("distance_gt_40m", structural["2"])
        self.assertIn("projected_area_lt_12px", structural["3"])
        self.assertEqual(QA._structural_reasons(by_id["1"], W, H) == [], "1" in qualified(result))


class Envelope(unittest.TestCase):
    def test_out_of_envelope_target_refused(self):
        with self.assertRaisesRegex(LiveGtContractError, "TILT_OUTSIDE_GROUND_RULE_ENVELOPE_TARGET"):
            rows_and_score([actor(1, 12.0, 0.0, pitch=20.0)])

    def test_out_of_envelope_structural_far_from_targets_omitted_with_proof(self):
        _, result = rows_and_score([actor(1, 12.0, 0.0), actor(2, 60.0, 5.0, pitch=25.0)])
        omitted = result["avo_geometry"]["omitted_out_of_envelope_structural"]
        self.assertEqual([o["gt_actor_id"] for o in omitted], ["2"])
        self.assertGreater(omitted[0]["target_separation"][0]["separation_margin_m"], 0)
        self.assertIn("1", qualified(result))

    def test_out_of_envelope_structural_near_a_target_refused(self):
        scene = [actor(1, 12.0, 0.0), actor(2, 12.0, 0.3, pitch=25.0, z=0.5)]
        rows = Q.frozen_people_rows(world(scene), frame_id=FRAME, timestamp=STAMP, camera_matrix=CAMERA,
                                    camera_inverse=INVERSE, intrinsics=K, width=W, height=H)
        rows[1] = dict(rows[1], gt_distance_m=41.0)                  # structural, yet overlapping the target
        with self.assertRaisesRegex(LiveGtContractError, "POSSIBLE_COMPETITOR"):
            rows_and_score(scene, rows=rows)

    def test_envelope_constant_is_the_frozen_ground_margin(self):
        rotation = np.asarray(actor(1, 0, 0, pitch=9.0).get_transform().get_matrix())[:3, :3]
        self.assertEqual(G.within_ground_envelope(rotation, EXTENT), G.footprint_tilt_m(rotation, EXTENT) <= 0.03)


class Switch(unittest.TestCase):
    def test_install_uninstall_and_double_install(self):
        from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
        original = NA.numeric_avo
        switch.install()
        try:
            self.assertIs(NA.numeric_avo, Q.numeric_avo)
            self.assertTrue(switch.installed())
            with self.assertRaises(RuntimeError):
                switch.install()
        finally:
            switch.uninstall()
        self.assertIs(NA.numeric_avo, original)
        self.assertFalse(switch.installed())


if __name__ == "__main__":
    unittest.main()
