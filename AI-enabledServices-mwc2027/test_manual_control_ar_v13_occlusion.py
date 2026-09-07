#!/usr/bin/env python3
"""Offline regression tests for the v13 cooperative-occlusion demonstrator."""

import ast
import math
import os
import time
import unittest

import manual_control_ar_v13 as manual_v13

from cooperative_occlusion_v1 import (
    evaluate_cooperative_target,
    occlusion_shadow_polygon_xy,
    point_in_sensor_fov_xy,
    segment_blocked_before_target,
    sensor_fov_polygon_xy,
    target_visibility_points_xy,
)


HERE = os.path.dirname(os.path.abspath(__file__))
V13_PATH = os.path.join(HERE, 'manual_control_ar_v13.py')

MODALITIES = {
    'camera': {'horizontal_fov_degrees': 90.0, 'range_m': 30.0},
    'radar': {'horizontal_fov_degrees': 30.0, 'range_m': 40.0},
}
OCCLUDER = {
    'actor_id': 500,
    'polygon': ((8.0, -2.0), (12.0, -2.0), (12.0, 2.0), (8.0, 2.0)),
}
TARGET = {'actor_id': 90, 'x': 20.0, 'y': 0.0}


def marker(marker_id, parent_id, x_coord, y_coord, yaw=0.0,
           modality='camera', active=True):
    return {
        'actor_id': marker_id,
        'parent_id': parent_id,
        'x': x_coord,
        'y': y_coord,
        'yaw': yaw,
        'modality': modality,
        'active': active,
        'site_key': ('site', parent_id),
    }


class FieldOfViewTests(unittest.TestCase):

    def test_command_line_fov_resolution(self):
        resolve = manual_v13.resolve_cooperative_sensor_fovs
        self.assertEqual(resolve(None, None, None), (90.0, 35.0))
        self.assertEqual(resolve(70.0, None, None), (70.0, 70.0))
        self.assertEqual(resolve(None, 120.0, 55.0), (120.0, 55.0))
        with self.assertRaises(ValueError):
            resolve(70.0, 90.0, None)
        with self.assertRaises(ValueError):
            resolve(181.0, None, None)

    def test_cardinal_membership_range_and_boundaries(self):
        self.assertTrue(point_in_sensor_fov_xy(
            (0.0, 0.0), 0.0, (10.0, 0.0), 90.0, 10.0))
        self.assertTrue(point_in_sensor_fov_xy(
            (0.0, 0.0), 90.0, (0.0, 10.0), 90.0, 10.0))
        self.assertFalse(point_in_sensor_fov_xy(
            (0.0, 0.0), 0.0, (-1.0, 0.0), 90.0, 10.0))
        self.assertFalse(point_in_sensor_fov_xy(
            (0.0, 0.0), 0.0, (10.1, 0.0), 90.0, 10.0))
        boundary = 10.0 / math.sqrt(2.0)
        self.assertTrue(point_in_sensor_fov_xy(
            (0.0, 0.0), 0.0, (boundary, boundary), 90.0, 10.0))

    def test_fov_and_shadow_polygons_are_finite_and_bounded(self):
        polygon = sensor_fov_polygon_xy((0.0, 0.0), 0.0, 90.0, 30.0)
        self.assertGreaterEqual(len(polygon), 4)
        for point in polygon:
            self.assertTrue(all(math.isfinite(value) for value in point))
            self.assertLessEqual(math.hypot(*point), 30.0 + 1.0e-6)
        shadow = occlusion_shadow_polygon_xy(
            (0.0, 0.0), 0.0, 90.0, 30.0, OCCLUDER['polygon'])
        self.assertEqual(len(shadow), 4)
        self.assertTrue(all(
            math.isfinite(value) for point in shadow for value in point))

    def test_shadow_survives_occluder_crossing_fov_boundary(self):
        def polar(distance, degrees):
            radians = math.radians(degrees)
            return distance * math.cos(radians), distance * math.sin(radians)
        boundary_crossing = (
            polar(8.0, 44.0),
            polar(10.0, 44.0),
            polar(10.0, 60.0),
            polar(8.0, 60.0),
        )
        shadow = occlusion_shadow_polygon_xy(
            (0.0, 0.0), 0.0, 90.0, 30.0, boundary_crossing)
        self.assertEqual(len(shadow), 4)
        far_bearings = [
            math.degrees(math.atan2(point[1], point[0]))
            for point in shadow
            if math.hypot(*point) > 29.0]
        self.assertTrue(any(abs(bearing - 45.0) < 0.1
                            for bearing in far_bearings))


class RayOcclusionTests(unittest.TestCase):

    def test_rectangle_blocks_only_targets_behind_it(self):
        self.assertTrue(segment_blocked_before_target(
            (0.0, 0.0), (20.0, 0.0), (OCCLUDER,)))
        self.assertFalse(segment_blocked_before_target(
            (0.0, 0.0), (5.0, 0.0), (OCCLUDER,)))

    def test_sensor_host_and_target_are_excluded(self):
        host = {'actor_id': 1, 'polygon': OCCLUDER['polygon']}
        target_box = {
            'actor_id': 90,
            'polygon': ((19.5, -0.5), (20.5, -0.5),
                        (20.5, 0.5), (19.5, 0.5)),
        }
        self.assertFalse(segment_blocked_before_target(
            (0.0, 0.0),
            (20.0, 0.0),
            (host, target_box),
            excluded_actor_ids=(1, 90)))


class CooperativeFusionTests(unittest.TestCase):

    def setUp(self):
        self.ego = marker(1, 1, 0.0, 0.0)
        self.near_peer = marker(2, 2, 0.0, 2.0)
        self.clear_peer = marker(3, 3, 0.0, 8.0)

    def evaluate(self, markers):
        return evaluate_cooperative_target(
            markers, TARGET, (OCCLUDER,), 1, MODALITIES)

    def test_three_site_progression_requires_clear_peer(self):
        ego_only = self.evaluate((self.ego,))
        two_sites = self.evaluate((self.ego, self.near_peer))
        three_sites = self.evaluate(
            (self.ego, self.near_peer, self.clear_peer))
        self.assertTrue(ego_only['blocked_from_ego'])
        self.assertFalse(ego_only['cooperatively_detected'])
        self.assertFalse(two_sites['cooperatively_detected'])
        self.assertTrue(three_sites['cooperatively_detected'])
        self.assertEqual(three_sites['peer_seen_by'], ((3, 'camera'),))

    def test_inactive_and_target_owned_markers_cannot_reveal_target(self):
        inactive_clear = dict(self.clear_peer, active=False)
        target_owned = marker(90, 90, 0.0, 8.0)
        self.assertFalse(self.evaluate(
            (self.ego, inactive_clear))['cooperatively_detected'])
        self.assertFalse(self.evaluate(
            (self.ego, target_owned))['cooperatively_detected'])

    def test_facing_away_peer_does_not_clear_occlusion(self):
        facing_away = marker(4, 4, 0.0, 8.0, yaw=180.0)
        self.assertFalse(self.evaluate(
            (self.ego, facing_away))['cooperatively_detected'])

    def test_radar_only_evidence_uses_radar_fov(self):
        ego_radar = marker(10, 1, 0.0, 0.0, modality='radar')
        peer_radar = marker(
            11, 3, 0.0, 8.0, yaw=45.0, modality='radar')
        inactive_camera = marker(
            12, 3, 0.0, 8.0, yaw=0.0, modality='camera', active=False)
        result = self.evaluate((ego_radar, peer_radar, inactive_camera))
        self.assertFalse(result['cooperatively_detected'])
        self.assertFalse(any(
            observation['modality'] == 'camera'
            for observation in result['observations']))

    def test_peer_visibility_outside_ego_fov_is_a_cooperative_reveal(self):
        ego_facing_away = marker(20, 1, 0.0, 0.0, yaw=180.0)
        peer = marker(21, 2, 0.0, 5.0, yaw=-26.565)
        result = evaluate_cooperative_target(
            (ego_facing_away, peer), TARGET, (), 1, MODALITIES)
        self.assertFalse(result['inside_ego_fov'])
        self.assertFalse(result['visible_to_ego'])
        self.assertTrue(result['visible_to_peer'])
        self.assertTrue(result['network_detected'])
        self.assertTrue(result['cooperatively_detected'])

    def test_residual_blind_samples_shrink_monotonically(self):
        grid = [
            (float(x), float(y))
            for x in range(13, 30)
            for y in range(-8, 9)
            if point_in_sensor_fov_xy(
                (0.0, 0.0), 0.0, (x, y), 90.0, 30.0)
            and segment_blocked_before_target(
                (0.0, 0.0), (x, y), (OCCLUDER,))
        ]

        def unresolved(peers):
            result = set()
            for point in grid:
                peer_visible = any(
                    point_in_sensor_fov_xy(
                        (peer['x'], peer['y']),
                        peer['yaw'],
                        point,
                        90.0,
                        30.0)
                    and not segment_blocked_before_target(
                        (peer['x'], peer['y']), point, (OCCLUDER,))
                    for peer in peers)
                if not peer_visible:
                    result.add(point)
            return result

        residual_zero = unresolved(())
        residual_near = unresolved((self.near_peer,))
        residual_clear = unresolved((self.near_peer, self.clear_peer))
        self.assertTrue(residual_near.issubset(residual_zero))
        self.assertTrue(residual_clear.issubset(residual_near))
        self.assertLess(len(residual_clear), len(residual_near))


class TargetExtentVisibilityTests(unittest.TestCase):

    def setUp(self):
        self.ego = marker(1, 1, 0.0, 0.0)

    def evaluate(self, target, occluders=(), modalities=MODALITIES):
        return evaluate_cooperative_target(
            (self.ego,), target, occluders, 1, modalities)

    def test_footprint_probes_include_center_boundary_and_interior(self):
        target = {
            'actor_id': 90,
            'x': 10.0,
            'y': 0.0,
            'footprint': ((9.5, -1.0), (10.5, -1.0),
                          (10.5, 1.0), (9.5, 1.0)),
        }
        points = target_visibility_points_xy(target)
        self.assertEqual(points[0], (10.0, 0.0))
        self.assertIn((9.5, -1.0), points)
        self.assertIn((9.75, -0.5), points)
        self.assertGreater(len(points), 9)

    def test_clear_side_detects_target_when_center_ray_is_blocked(self):
        target = {
            'actor_id': 90,
            'x': 20.0,
            'y': 0.0,
            'footprint': ((19.5, -1.0), (20.5, -1.0),
                          (20.5, 1.0), (19.5, 1.0)),
        }
        narrow_occluder = {
            'actor_id': 500,
            'polygon': ((8.0, -0.2), (12.0, -0.2),
                        (12.0, 0.2), (8.0, 0.2)),
        }
        self.assertTrue(segment_blocked_before_target(
            (0.0, 0.0), (20.0, 0.0), (narrow_occluder,)))
        result = self.evaluate(target, (narrow_occluder,))
        self.assertTrue(result['visible_to_ego'])
        observation = result['observations'][0]
        self.assertTrue(observation['inside_fov'])
        self.assertFalse(observation['blocked'])
        self.assertNotEqual(observation['visible_point'], (20.0, 0.0))

    def test_footprint_edge_inside_fov_detects_center_outside(self):
        target = {
            'actor_id': 90,
            'x': 10.0,
            'y': 2.0,
            'footprint': ((9.5, 1.4), (10.5, 1.4),
                          (10.5, 2.6), (9.5, 2.6)),
        }
        narrow_fov = {
            'camera': {'horizontal_fov_degrees': 20.0, 'range_m': 30.0},
        }
        self.assertFalse(point_in_sensor_fov_xy(
            (0.0, 0.0), 0.0, (10.0, 2.0), 20.0, 30.0))
        result = self.evaluate(target, modalities=narrow_fov)
        self.assertTrue(result['visible_to_ego'])

    def test_near_footprint_edge_inside_range_detects_center_beyond_range(self):
        target = {
            'actor_id': 90,
            'x': 30.4,
            'y': 0.0,
            'footprint': ((29.8, -0.3), (31.0, -0.3),
                          (31.0, 0.3), (29.8, 0.3)),
        }
        result = self.evaluate(target)
        self.assertTrue(result['visible_to_ego'])

    def test_all_in_fov_footprint_probes_blocked_remains_hidden(self):
        target = {
            'actor_id': 90,
            'x': 10.0,
            'y': 0.0,
            'footprint': ((9.5, -1.0), (10.5, -1.0),
                          (10.5, 1.0), (9.5, 1.0)),
        }
        wide_occluder = {
            'actor_id': 500,
            'polygon': ((4.0, -5.0), (6.0, -5.0),
                        (6.0, 5.0), (4.0, 5.0)),
        }
        result = self.evaluate(target, (wide_occluder,))
        self.assertTrue(result['inside_ego_fov'])
        self.assertTrue(result['blocked_from_ego'])
        self.assertFalse(result['network_detected'])

    def test_missing_footprint_retains_center_point_fallback(self):
        self.assertEqual(
            target_visibility_points_xy(TARGET),
            ((20.0, 0.0),))
        self.assertTrue(self.evaluate(TARGET, (OCCLUDER,))['blocked_from_ego'])


class IntegrationSafetyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(V13_PATH, 'r') as source_file:
            cls.source = source_file.read()
        cls.tree = ast.parse(cls.source)

    def test_topdown_renderer_never_spawns_or_listens_to_sensors(self):
        topdown = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'TopDownMapRenderer')
        forbidden = []
        for node in ast.walk(topdown):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in (
                    'spawn_actor', 'try_spawn_actor', 'listen'):
                forbidden.append((function.attr, node.lineno))
        self.assertEqual(forbidden, [])

    def test_camera_warning_path_consumes_cooperative_state(self):
        self.assertIn(
            "cooperative_state == 'UNRESOLVED_OCCLUDED'", self.source)
        self.assertIn("'COOPERATIVELY_REVEALED'", self.source)
        self.assertIn('OCCLUDED PEDESTRIAN', self.source)

    def test_warning_authorization_accepts_any_active_sensor_visibility(self):
        authorize = (
            manual_v13.rogue_pedestrian_warning_is_sensor_authorized)
        self.assertFalse(authorize('UNRESOLVED_OCCLUDED'))
        self.assertTrue(authorize('EGO_VISIBLE'))
        self.assertTrue(authorize('COOPERATIVELY_REVEALED'))

    def test_scene_targets_include_actor_footprints(self):
        self.assertGreaterEqual(self.source.count("'footprint': footprint"), 4)

    def test_inventory_includes_ambient_vehicle_and_pedestrian_hosts(self):
        self.assertIn("'ambient_vehicle'", self.source)
        self.assertIn("'ambient_pedestrian'", self.source)

    def test_world_map_update_precedes_camera_composition(self):
        world_class = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'World')
        render = next(
            node for node in world_class.body
            if isinstance(node, ast.FunctionDef) and node.name == 'render')
        calls = []
        for node in ast.walk(render):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.append((node.func.attr, node.lineno))
        sync_line = min(line for name, line in calls
                        if name == '_sync_cooperative_visibility')
        camera_line = min(
            node.lineno
            for node in ast.walk(render)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'render'
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == 'camera_manager')
        self.assertLess(sync_line, camera_line)


class VirtualSiteSelectionTests(unittest.TestCase):

    @staticmethod
    def renderer(active_pairs=1, degraded_radars=1):
        renderer = object.__new__(manual_v13.TopDownMapRenderer)
        renderer._center_x = 0.0
        renderer._center_y = 0.0
        renderer._zoom_radius_m = 100.0
        renderer._active_sensor_pairs = active_pairs
        renderer._degraded_active_radars = degraded_radars
        renderer._selected_site_keys_by_policy = {
            'balanced': (), 'radar': ()}
        renderer._selected_live_sensor_actors = ()
        renderer._last_refresh_ms = 123
        renderer.last_cooperative_visibility = (
            renderer._empty_cooperative_snapshot())
        renderer.last_cooperative_sensor_summary = {}
        return renderer

    @staticmethod
    def site(parent_id, x_coord, host_category):
        markers = []
        for offset, modality in enumerate(('camera', 'radar'), 1):
            markers.append({
                'actor_id': -(parent_id * 10 + offset),
                'actor_ids': {-(parent_id * 10 + offset)},
                'parent_id': parent_id,
                'site_key': ('virtual', 1, parent_id),
                'x': x_coord,
                'y': 0.0,
                'z': 1.0,
                'yaw': 0.0,
                'modality': modality,
                'selectable': True,
                'active': False,
                'host_category': host_category,
            })
        return markers

    def test_positive_budget_pins_complete_ego_pair_first(self):
        renderer = self.renderer(active_pairs=1)
        markers = (
            self.site(1, 2.5, 'ego_vehicle')
            + self.site(2, 0.1, 'static_blocker_vehicle'))
        selected = renderer._apply_visual_sensor_policy(markers, False)
        active = [item for item in selected if item['active']]
        self.assertEqual(len(active), 2)
        self.assertEqual({item['parent_id'] for item in active}, {1})
        self.assertEqual(
            {item['modality'] for item in active}, {'camera', 'radar'})

    def test_zero_budget_and_radar_only_policy(self):
        renderer = self.renderer(active_pairs=0, degraded_radars=2)
        markers = (
            self.site(1, 2.5, 'ego_vehicle')
            + self.site(2, 5.0, 'static_blocker_vehicle')
            + self.site(3, 8.0, 'blocker_pedestrian'))
        normal = renderer._apply_visual_sensor_policy(markers, False)
        self.assertFalse(any(item['active'] for item in normal))
        degraded = renderer._apply_visual_sensor_policy(markers, True)
        active = [item for item in degraded if item['active']]
        self.assertEqual(len(active), 2)
        self.assertEqual({item['modality'] for item in active}, {'radar'})
        self.assertIn(1, {item['parent_id'] for item in active})

    def test_live_limit_change_revokes_old_visibility(self):
        renderer = self.renderer(active_pairs=1, degraded_radars=1)
        renderer.last_cooperative_visibility = {
            'sampled_at': time.perf_counter(),
            'actors': {90: {'cooperatively_detected': True}},
        }
        self.assertTrue(renderer.set_active_sensor_pair_limits(3, 3))
        self.assertEqual(renderer.active_sensor_pair_limit, 3)
        self.assertEqual(renderer.last_cooperative_visibility['actors'], {})
        self.assertIsNone(renderer._last_refresh_ms)

    def test_warning_authority_is_actor_specific_and_expires(self):
        world = object.__new__(manual_v13.World)
        world.show_topdown_map = True
        world._cooperative_visibility_snapshot = {
            'sampled_at': time.perf_counter(),
            'actors': {
                90: {
                    'visible_to_ego': False,
                    'cooperatively_detected': True,
                },
                91: {
                    'visible_to_ego': True,
                    'cooperatively_detected': False,
                },
                92: {
                    'visible_to_ego': False,
                    'visible_to_peer': True,
                    'cooperatively_detected': False,
                },
            },
        }
        self.assertEqual(
            world.cooperative_visibility_state(90),
            'COOPERATIVELY_REVEALED')
        self.assertEqual(
            world.cooperative_visibility_state(91), 'EGO_VISIBLE')
        self.assertEqual(
            world.cooperative_visibility_state(92),
            'COOPERATIVELY_REVEALED')
        self.assertEqual(
            world.cooperative_visibility_state(93),
            'UNRESOLVED_OCCLUDED')
        world._cooperative_visibility_snapshot['sampled_at'] = (
            time.perf_counter()
            - manual_v13.COOPERATIVE_VISIBILITY_MAX_AGE_SECONDS
            - 0.1)
        self.assertEqual(
            world.cooperative_visibility_state(90),
            'UNRESOLVED_OCCLUDED')


if __name__ == '__main__':
    unittest.main()
