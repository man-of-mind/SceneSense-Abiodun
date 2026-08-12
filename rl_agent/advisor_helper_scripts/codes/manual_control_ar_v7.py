#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Allows controlling a vehicle with a keyboard.

Welcome to CARLA manual control.

Version 7 adds exact ego-vehicle blueprint selection, starts with the HUD
hidden, and uses the front-mounted RGB camera as the initial view.

Use WASD keys for vehicle control and the arrow keys for the camera view.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot or route autonomy
    J            : toggle looping route-to-destination mode
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    LEFT/RIGHT   : yaw active sensor left/right
    UP/DOWN      : pitch active sensor up/down
    KP4/KP6      : yaw active sensor left/right alternative
    KP8/KP2      : pitch active sensor up/down alternative
    KP5          : reset active sensor yaw/pitch
    HOME/END     : yaw active sensor left/right fallback
    PGUP/PGDN    : pitch active sensor up/down fallback
    INSERT       : reset active sensor yaw/pitch fallback
    SHIFT        : faster yaw/pitch while held
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle (fixed type when --vehicle-blueprint is set)
    Y            : respawn ego at the configured --ego-spawn-x/--ego-spawn-y

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry
    U            : toggle actor boxes and ego-following top-down map

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import os
import sys
import weakref

CARLA_AGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'carla'))
if CARLA_AGENT_PATH not in sys.path:
    sys.path.append(CARLA_AGENT_PATH)

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    GlobalRoutePlanner = None

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_END
    from pygame.locals import K_HOME
    from pygame.locals import K_INSERT
    from pygame.locals import K_KP2
    from pygame.locals import K_KP4
    from pygame.locals import K_KP5
    from pygame.locals import K_KP6
    from pygame.locals import K_KP8
    from pygame.locals import K_LEFT
    from pygame.locals import K_PAGEDOWN
    from pygame.locals import K_PAGEUP
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_j
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import cv2
except ImportError:
    cv2 = None

OBJECT_TO_COLOR = [
    (255, 255, 255),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142,  35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0,  60, 100),
    (0,  80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (110, 190, 160),
    (170, 120, 50),
    (55, 90, 80),
    (45, 60, 150),
    (157, 234, 50),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
    (180, 165, 180),
]

DEFAULT_TOPDOWN_ZOOM_RADIUS_M = 60.0
MIN_TOPDOWN_ZOOM_RADIUS_M = 1.0
MAX_TOPDOWN_ZOOM_RADIUS_M = 10000.0
TOPDOWN_MAP_REFRESH_HZ = 10.0
TOPDOWN_WAYPOINT_SPACING_M = 3.0

# Default startup/respawn coordinates. Command-line X/Y values replace both.
DEFAULT_EGO_SPAWN_X = 73.63
DEFAULT_EGO_SPAWN_Y = 66.36
# CARLA vehicle spawn transforms conventionally sit above the road surface.
EGO_SPAWN_ROAD_HEIGHT_OFFSET_M = 0.60
MAX_EGO_SPAWN_ROAD_PROJECTION_M = 5.0
EGO_SPAWN_OCCUPANCY_RADIUS_M = 3.0
EGO_SPAWN_POSITION_TOLERANCE_M = 0.10

# Transform 1 is the centered, rigid camera mounted at the front of a vehicle.
DEFAULT_CAMERA_TRANSFORM_INDEX = 1

# Match the Physical AI scenario map while accounting for OpenCV's BGR order.
TOPDOWN_COLOR_BACKGROUND = (30, 23, 18)
TOPDOWN_COLOR_GRID = (61, 50, 43)
TOPDOWN_COLOR_BUILDING_FILL = (42, 42, 42)
TOPDOWN_COLOR_BUILDING_EDGE = (64, 64, 64)
TOPDOWN_COLOR_LANE_CENTERLINE = (85, 85, 85)
TOPDOWN_COLOR_VEHICLE = (220, 150, 72)
TOPDOWN_COLOR_PEDESTRIAN = (178, 195, 82)
TOPDOWN_COLOR_EGO = (68, 173, 255)
TOPDOWN_COLOR_ROUTE = (255, 171, 87)
TOPDOWN_COLOR_DESTINATION = (68, 173, 255)

MIN_BUILDING_HEIGHT_M = 2.0
MIN_BUILDING_AREA_M2 = 20.0
MIN_BUILDING_VOLUME_M3 = 80.0
BUILDING_ROAD_PROXIMITY_M = 20.0
BUILDING_EDGE_SAMPLE_M = 5.0

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def topdown_zoom_radius(value):
    """Argparse converter for a numerically safe top-down radius."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a number')
    if (
            not math.isfinite(parsed)
            or parsed < MIN_TOPDOWN_ZOOM_RADIUS_M
            or parsed > MAX_TOPDOWN_ZOOM_RADIUS_M):
        raise argparse.ArgumentTypeError(
            'must be between {:.1f} and {:.1f} meters'.format(
                MIN_TOPDOWN_ZOOM_RADIUS_M,
                MAX_TOPDOWN_ZOOM_RADIUS_M))
    return parsed


def finite_float(value):
    """Argparse converter that rejects NaN and infinite coordinates."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a number')
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError('must be finite')
    return parsed


def vehicle_blueprint_id(value):
    """Argparse converter for an exact CARLA vehicle blueprint identifier."""
    blueprint_id = value.strip()
    if not blueprint_id:
        raise argparse.ArgumentTypeError('must not be empty')
    if not blueprint_id.startswith('vehicle.'):
        raise argparse.ArgumentTypeError(
            'must be an exact vehicle blueprint ID beginning with "vehicle."')
    if any(character in blueprint_id for character in '*?[]'):
        raise argparse.ArgumentTypeError(
            'must be an exact blueprint ID; wildcard patterns are not supported')
    return blueprint_id


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3, 4]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def draw_geofence(world, location, radius):
    """
    Draw a cylindrical geofence approximation using debug lines.
    """
    thickness = 0.1
    color = carla.Color(255, 0, 0)
    lifetime = 0.1
    z_base = location.z
    z_top = location.z + 10.0

    num_segments = 24
    angle_step = 2 * math.pi / num_segments

    for i in range(num_segments):
        angle1 = i * angle_step
        angle2 = (i + 1) * angle_step

        x1 = location.x + radius * math.cos(angle1)
        y1 = location.y + radius * math.sin(angle1)

        x2 = location.x + radius * math.cos(angle2)
        y2 = location.y + radius * math.sin(angle2)

        p1_base = carla.Location(x=x1, y=y1, z=z_base)
        p2_base = carla.Location(x=x2, y=y2, z=z_base)
        p1_top = carla.Location(x=x1, y=y1, z=z_top)
        p2_top = carla.Location(x=x2, y=y2, z=z_top)

        world.debug.draw_line(p1_base, p2_base, thickness=thickness, color=color, life_time=lifetime)
        world.debug.draw_line(p1_top, p2_top, thickness=thickness, color=color, life_time=lifetime)
        world.debug.draw_line(p1_base, p1_top, thickness=thickness, color=color, life_time=lifetime)


def copy_transform(transform, z_offset=0.0):
    location = transform.location
    rotation = transform.rotation
    return carla.Transform(
        carla.Location(x=location.x, y=location.y, z=location.z + z_offset),
        carla.Rotation(pitch=rotation.pitch, yaw=rotation.yaw, roll=rotation.roll))


def resolve_ego_spawn_transform(carla_map, x_coord, y_coord):
    """Keep requested X/Y and derive a vehicle-safe road Z and heading."""
    try:
        x_coord = float(x_coord)
        y_coord = float(y_coord)
    except (TypeError, ValueError) as exc:
        raise ValueError('Ego spawn X/Y coordinates must be numbers') from exc
    if not math.isfinite(x_coord) or not math.isfinite(y_coord):
        raise ValueError('Ego spawn X/Y coordinates must be finite')
    requested_location = carla.Location(
        x=x_coord,
        y=y_coord,
        z=0.0)
    try:
        waypoint = carla_map.get_waypoint(
            requested_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
    except RuntimeError as exc:
        raise ValueError(
            'Unable to resolve ego spawn road transform at '
            'x={:.2f}, y={:.2f}'.format(x_coord, y_coord)) from exc
    if waypoint is None:
        raise ValueError(
            'No driving waypoint is available near ego spawn '
            'x={:.2f}, y={:.2f}'.format(x_coord, y_coord))
    road_transform = waypoint.transform
    projection_distance = math.hypot(
        road_transform.location.x - x_coord,
        road_transform.location.y - y_coord)
    if projection_distance > MAX_EGO_SPAWN_ROAD_PROJECTION_M:
        raise ValueError(
            'Ego spawn x={:.2f}, y={:.2f} is {:.2f} m from the nearest '
            'driving lane (maximum: {:.2f} m)'.format(
                x_coord,
                y_coord,
                projection_distance,
                MAX_EGO_SPAWN_ROAD_PROJECTION_M))
    return carla.Transform(
        carla.Location(
            x=x_coord,
            y=y_coord,
            z=float(
                road_transform.location.z
                + EGO_SPAWN_ROAD_HEIGHT_OFFSET_M)),
        carla.Rotation(yaw=float(road_transform.rotation.yaw)))


def copy_location(location):
    return carla.Location(x=location.x, y=location.y, z=location.z)


def _deg2rad(degrees_value):
    return degrees_value * math.pi / 180.0


def rotation_matrix_from_carla_rotation(rotation):
    roll = _deg2rad(rotation.roll)
    pitch = _deg2rad(rotation.pitch)
    yaw = _deg2rad(rotation.yaw)

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rotation_x = np.array(
        [[1, 0, 0],
         [0, cr, -sr],
         [0, sr, cr]],
        dtype=np.float32)
    rotation_y = np.array(
        [[cp, 0, sp],
         [0, 1, 0],
         [-sp, 0, cp]],
        dtype=np.float32)
    rotation_z = np.array(
        [[cy, -sy, 0],
         [sy, cy, 0],
         [0, 0, 1]],
        dtype=np.float32)

    return (rotation_z @ rotation_y @ rotation_x).astype(np.float32)


def carla_rotation_from_matrix(rotation_matrix):
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation_matrix[2, 0]))))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]))
        yaw = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation_matrix[0, 1]), float(rotation_matrix[1, 1]))
    return carla.Rotation(
        pitch=math.degrees(pitch),
        yaw=math.degrees(yaw),
        roll=math.degrees(roll))


def get_camera_K(width, height, fov_degrees):
    focal = width / (2.0 * np.tan(fov_degrees * np.pi / 360.0))
    calibration = np.identity(3, dtype=np.float32)
    calibration[0, 0] = calibration[1, 1] = focal
    calibration[0, 2] = width / 2.0
    calibration[1, 2] = height / 2.0
    return calibration


def world_to_camera(points_world, camera_transform):
    if points_world.size == 0:
        return points_world
    inverse_matrix = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    homogeneous_points = np.concatenate(
        [points_world.astype(np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1)
    points_camera = (inverse_matrix @ homogeneous_points.T).T
    return points_camera[:, :3]


def project_to_image(points_camera, calibration, width, height):
    if points_camera.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    x_values = points_camera[:, 0]
    y_values = points_camera[:, 1]
    z_values = points_camera[:, 2]
    points_in_front = x_values > 0.05
    if not np.any(points_in_front):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    x_values = x_values[points_in_front]
    y_values = y_values[points_in_front]
    z_values = z_values[points_in_front]
    u_values = calibration[0, 2] + (y_values / x_values) * calibration[0, 0]
    v_values = calibration[1, 2] - (z_values / x_values) * calibration[1, 1]
    u_values = u_values.astype(np.int32)
    v_values = v_values.astype(np.int32)
    valid_pixels = (
        (u_values >= 0) & (u_values < width) &
        (v_values >= 0) & (v_values < height))
    return u_values[valid_pixels], v_values[valid_pixels]


def project_bbox_corners_to_2d(actor_transform, bounding_box, camera_transform, calibration, width, height):
    extent_x = bounding_box.extent.x
    extent_y = bounding_box.extent.y
    extent_z = bounding_box.extent.z
    local_corners = np.array([
        [extent_x, extent_y, extent_z],
        [extent_x, extent_y, -extent_z],
        [extent_x, -extent_y, extent_z],
        [extent_x, -extent_y, -extent_z],
        [-extent_x, extent_y, extent_z],
        [-extent_x, extent_y, -extent_z],
        [-extent_x, -extent_y, extent_z],
        [-extent_x, -extent_y, -extent_z],
    ], dtype=np.float32)

    bbox_rotation = rotation_matrix_from_carla_rotation(bounding_box.rotation)
    bbox_location = np.array(
        [bounding_box.location.x, bounding_box.location.y, bounding_box.location.z],
        dtype=np.float32)
    actor_space_corners = (bbox_rotation @ local_corners.T).T + bbox_location.reshape(1, 3)

    actor_matrix = np.array(actor_transform.get_matrix(), dtype=np.float32)
    homogeneous_corners = np.concatenate(
        [actor_space_corners, np.ones((len(actor_space_corners), 1), dtype=np.float32)],
        axis=1)
    world_corners = (actor_matrix @ homogeneous_corners.T).T[:, :3]
    camera_corners = world_to_camera(world_corners, camera_transform)
    u_values, v_values = project_to_image(camera_corners, calibration, width, height)
    if u_values.size == 0 or v_values.size == 0:
        return None

    x1 = int(np.clip(u_values.min(), 0, width - 1))
    y1 = int(np.clip(v_values.min(), 0, height - 1))
    x2 = int(np.clip(u_values.max(), 0, width - 1))
    y2 = int(np.clip(v_values.max(), 0, height - 1))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return x1, y1, x2, y2


def get_actor_footprint_points(actor, actor_transform=None):
    bounding_box = actor.bounding_box
    extent_x = bounding_box.extent.x
    extent_y = bounding_box.extent.y
    local_corners = np.array([
        [extent_x, extent_y, 0.0],
        [extent_x, -extent_y, 0.0],
        [-extent_x, -extent_y, 0.0],
        [-extent_x, extent_y, 0.0],
    ], dtype=np.float32)

    bbox_rotation = rotation_matrix_from_carla_rotation(bounding_box.rotation)
    bbox_location = np.array(
        [bounding_box.location.x, bounding_box.location.y, bounding_box.location.z],
        dtype=np.float32)
    actor_space_corners = (bbox_rotation @ local_corners.T).T + bbox_location.reshape(1, 3)

    if actor_transform is None:
        actor_transform = actor.get_transform()
    actor_matrix = np.array(actor_transform.get_matrix(), dtype=np.float32)
    homogeneous_corners = np.concatenate(
        [actor_space_corners, np.ones((len(actor_space_corners), 1), dtype=np.float32)],
        axis=1)
    world_corners = (actor_matrix @ homogeneous_corners.T).T[:, :2]
    return world_corners


def draw_route_waypoints(world, route_trace, origin_transform=None, destination_transform=None):
    """
    Route guidance is drawn as a post-camera AR overlay by CameraManager.

    CARLA debug lines are rendered as bright in-world primitives and can bloom
    into a white stripe in the RGB camera, so keep them out of the camera pass.
    """
    return


class TopDownMapRenderer(object):
    """Ego-centred, world-aligned view of static geometry and live actors."""

    def __init__(
            self,
            carla_world,
            carla_map,
            zoom_radius_m,
            width=960,
            height=960,
            refresh_hz=TOPDOWN_MAP_REFRESH_HZ):
        self._world = carla_world
        self._map = carla_map
        self._zoom_radius_m = float(zoom_radius_m)
        if (
                not math.isfinite(self._zoom_radius_m)
                or self._zoom_radius_m < MIN_TOPDOWN_ZOOM_RADIUS_M
                or self._zoom_radius_m > MAX_TOPDOWN_ZOOM_RADIUS_M):
            raise ValueError(
                'top-down zoom radius must be between {:.1f} and {:.1f} meters'.format(
                    MIN_TOPDOWN_ZOOM_RADIUS_M,
                    MAX_TOPDOWN_ZOOM_RADIUS_M))

        self._width = int(width)
        self._height = int(height)
        self._window_name = 'CARLA Ego-Following Top-Down Map'
        self._window_created = False
        self._last_refresh_ms = None
        self._refresh_period_ms = max(1, int(round(1000.0 / float(refresh_hz))))
        self._header_height = 58
        self._footer_height = 34
        self._margin = 28
        available_height = self._height - self._header_height - self._footer_height
        self._plot_size = min(self._width - (2 * self._margin), available_height)
        if self._plot_size < 64:
            raise ValueError('top-down map dimensions are too small')
        self._plot_left = (self._width - self._plot_size) // 2
        self._plot_top = self._header_height + (
            available_height - self._plot_size) // 2
        self._plot_center_pixel = (self._plot_size - 1) / 2.0
        self._scale = (self._plot_size - 1) / (2.0 * self._zoom_radius_m)
        self._center_x = 0.0
        self._center_y = 0.0
        self._road_polylines = []
        self._building_footprints = []
        self.ready = cv2 is not None
        if self.ready:
            self._build_static_geometry()

    @staticmethod
    def _geometry_entry(points):
        points_array = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        bounds = (
            float(np.min(points_array[:, 0])),
            float(np.min(points_array[:, 1])),
            float(np.max(points_array[:, 0])),
            float(np.max(points_array[:, 1])),
        )
        return points_array, bounds

    @staticmethod
    def _smooth_polyline(points, passes=2):
        result = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        for _ in range(max(0, int(passes))):
            if len(result) < 3:
                break
            smoothed = [result[0]]
            for first, second in zip(result, result[1:]):
                smoothed.append((0.75 * first) + (0.25 * second))
                smoothed.append((0.25 * first) + (0.75 * second))
            smoothed.append(result[-1])
            result = np.asarray(smoothed, dtype=np.float32)
        return result

    @classmethod
    def _build_road_polylines(cls, waypoints, sample_spacing):
        lane_samples = {}
        for waypoint in waypoints:
            try:
                if waypoint.lane_type != carla.LaneType.Driving:
                    continue
                key = (
                    int(waypoint.road_id),
                    int(waypoint.section_id),
                    int(waypoint.lane_id),
                )
                location = waypoint.transform.location
                lane_samples.setdefault(key, []).append(
                    (float(waypoint.s), float(location.x), float(location.y)))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue

        polylines = []
        minimum_separation = max(0.1, float(sample_spacing) * 0.25)
        for samples in lane_samples.values():
            samples.sort(key=lambda item: item[0])
            points = []
            for _, x_coord, y_coord in samples:
                if points:
                    separation = math.hypot(
                        x_coord - points[-1][0], y_coord - points[-1][1])
                    if separation < minimum_separation:
                        continue
                points.append((x_coord, y_coord))
            if len(points) >= 2:
                smoothed = cls._smooth_polyline(points, passes=2)
                polylines.append(cls._geometry_entry(smoothed))
        return polylines

    @staticmethod
    def _building_footprint(bounding_box):
        transform = carla.Transform(
            bounding_box.location,
            bounding_box.rotation)
        extent = bounding_box.extent
        corners = []
        for x_coord, y_coord in (
                (extent.x, extent.y),
                (-extent.x, extent.y),
                (-extent.x, -extent.y),
                (extent.x, -extent.y)):
            corner = transform.transform(carla.Location(
                x=float(x_coord),
                y=float(y_coord),
                z=-float(extent.z)))
            corners.append((float(corner.x), float(corner.y)))
        return np.asarray(corners, dtype=np.float32)

    @staticmethod
    def _polygon_area(points):
        if len(points) < 3:
            return 0.0
        total = 0.0
        for current, following in zip(points, np.roll(points, -1, axis=0)):
            total += float(current[0]) * float(following[1])
            total -= float(current[1]) * float(following[0])
        return abs(total) * 0.5

    @staticmethod
    def _sample_polygon_edges(points, spacing):
        samples = []
        if len(points) < 2:
            return samples
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            length = math.hypot(
                float(end[0] - start[0]), float(end[1] - start[1]))
            steps = max(1, int(math.ceil(length / max(0.1, spacing))))
            for step in range(steps + 1):
                fraction = float(step) / float(steps)
                samples.append((
                    float(start[0] + (end[0] - start[0]) * fraction),
                    float(start[1] + (end[1] - start[1]) * fraction),
                ))
        return samples

    def _build_building_footprints(self, road_locations):
        try:
            environment_objects = self._world.get_environment_objects(
                carla.CityObjectLabel.Buildings)
        except Exception as exc:
            logging.warning('Building footprints unavailable for top-down map: %s', exc)
            return []

        cell_size = BUILDING_ROAD_PROXIMITY_M
        road_grid = {}
        for x_coord, y_coord in road_locations:
            key = (
                math.floor(float(x_coord) / cell_size),
                math.floor(float(y_coord) / cell_size),
            )
            road_grid.setdefault(key, []).append((float(x_coord), float(y_coord)))

        maximum_distance_squared = BUILDING_ROAD_PROXIMITY_M ** 2
        footprints = []
        for environment_object in environment_objects:
            try:
                bounding_box = environment_object.bounding_box
                footprint = self._building_footprint(bounding_box)
                area = self._polygon_area(footprint)
                height = float(bounding_box.extent.z) * 2.0
                if (
                        height < MIN_BUILDING_HEIGHT_M
                        or area < MIN_BUILDING_AREA_M2
                        or area * height < MIN_BUILDING_VOLUME_M3):
                    continue

                close_to_road = not road_grid
                for sample_x, sample_y in self._sample_polygon_edges(
                        footprint, BUILDING_EDGE_SAMPLE_M):
                    cell_x = math.floor(sample_x / cell_size)
                    cell_y = math.floor(sample_y / cell_size)
                    for offset_x in (-1, 0, 1):
                        for offset_y in (-1, 0, 1):
                            nearby = road_grid.get(
                                (cell_x + offset_x, cell_y + offset_y), [])
                            for road_x, road_y in nearby:
                                dx = sample_x - road_x
                                dy = sample_y - road_y
                                if dx * dx + dy * dy <= maximum_distance_squared:
                                    close_to_road = True
                                    break
                            if close_to_road:
                                break
                        if close_to_road:
                            break
                    if close_to_road:
                        break
                if close_to_road:
                    footprints.append(self._geometry_entry(footprint))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return footprints

    def _build_static_geometry(self):
        waypoints = list(self._map.generate_waypoints(
            TOPDOWN_WAYPOINT_SPACING_M))
        if not waypoints:
            raise RuntimeError('Unable to build top-down map without waypoints')

        road_locations = np.asarray([
            (
                float(waypoint.transform.location.x),
                float(waypoint.transform.location.y),
            )
            for waypoint in waypoints
        ], dtype=np.float32)
        self._road_polylines = self._build_road_polylines(
            waypoints, TOPDOWN_WAYPOINT_SPACING_M)
        self._building_footprints = self._build_building_footprints(
            road_locations)
        logging.info(
            'Top-down map geometry: %d lane polylines, %d building footprints',
            len(self._road_polylines),
            len(self._building_footprints))

    @staticmethod
    def _nice_grid_spacing(raw_spacing):
        if raw_spacing <= 0.0:
            return 10.0
        exponent = math.floor(math.log10(raw_spacing))
        fraction = raw_spacing / (10.0 ** exponent)
        if fraction <= 1.0:
            nice = 1.0
        elif fraction <= 2.0:
            nice = 2.0
        elif fraction <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10.0 ** exponent)

    @staticmethod
    def _bounds_intersect(first, second):
        return not (
            first[2] < second[0]
            or first[0] > second[2]
            or first[3] < second[1]
            or first[1] > second[3])

    def _visible_world_bounds(self):
        return (
            self._center_x - self._zoom_radius_m,
            self._center_y - self._zoom_radius_m,
            self._center_x + self._zoom_radius_m,
            self._center_y + self._zoom_radius_m,
        )

    def _world_xy_to_pixel(self, x_coord, y_coord):
        # Match the Physical AI map: CARLA +X is right and +Y is down.
        pixel_x = self._plot_center_pixel + (
            float(x_coord) - self._center_x) * self._scale
        pixel_y = self._plot_center_pixel + (
            float(y_coord) - self._center_y) * self._scale
        return int(round(pixel_x)), int(round(pixel_y))

    def _world_to_pixel(self, location):
        return self._world_xy_to_pixel(location.x, location.y)

    def _points_to_pixels(self, points):
        pixels = np.empty_like(points, dtype=np.float32)
        pixels[:, 0] = self._plot_center_pixel + (
            points[:, 0] - self._center_x) * self._scale
        pixels[:, 1] = self._plot_center_pixel + (
            points[:, 1] - self._center_y) * self._scale
        return np.rint(pixels).astype(np.int32)

    def _location_is_visible(self, location):
        x_coord = float(location.x)
        y_coord = float(location.y)
        return (
            math.isfinite(x_coord)
            and math.isfinite(y_coord)
            and abs(x_coord - self._center_x) <= self._zoom_radius_m
            and abs(y_coord - self._center_y) <= self._zoom_radius_m)

    def _vehicle_footprint_in_view(self, actor, actor_transform):
        bounding_box = actor.bounding_box
        extent = bounding_box.extent
        coarse_margin = (
            math.hypot(float(extent.x), float(extent.y))
            + math.hypot(
                float(bounding_box.location.x),
                float(bounding_box.location.y)))
        location = actor_transform.location
        if (
                abs(float(location.x) - self._center_x)
                > self._zoom_radius_m + coarse_margin
                or abs(float(location.y) - self._center_y)
                > self._zoom_radius_m + coarse_margin):
            return None
        footprint = get_actor_footprint_points(actor, actor_transform)
        footprint_bounds = self._geometry_entry(footprint)[1]
        if not self._bounds_intersect(
                footprint_bounds, self._visible_world_bounds()):
            return None
        return footprint

    def _draw_grid(self, image):
        spacing = self._nice_grid_spacing((2.0 * self._zoom_radius_m) / 8.0)
        bounds = self._visible_world_bounds()
        value = math.ceil(bounds[0] / spacing) * spacing
        while value <= bounds[2]:
            pixel_x, _ = self._world_xy_to_pixel(value, self._center_y)
            cv2.line(
                image,
                (pixel_x, 0),
                (pixel_x, self._plot_size - 1),
                TOPDOWN_COLOR_GRID,
                1,
                lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                '{:.0f}'.format(value),
                (pixel_x + 3, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (130, 135, 142),
                1,
                cv2.LINE_AA)
            value += spacing

        value = math.ceil(bounds[1] / spacing) * spacing
        while value <= bounds[3]:
            _, pixel_y = self._world_xy_to_pixel(self._center_x, value)
            cv2.line(
                image,
                (0, pixel_y),
                (self._plot_size - 1, pixel_y),
                TOPDOWN_COLOR_GRID,
                1,
                lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                '{:.0f}'.format(value),
                (3, max(13, pixel_y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (130, 135, 142),
                1,
                cv2.LINE_AA)
            value += spacing

    def _draw_static_map(self):
        image = np.full(
            (self._plot_size, self._plot_size, 3),
            TOPDOWN_COLOR_BACKGROUND,
            dtype=np.uint8)
        self._draw_grid(image)
        visible_bounds = self._visible_world_bounds()

        for points, bounds in self._building_footprints:
            if not self._bounds_intersect(bounds, visible_bounds):
                continue
            pixels = self._points_to_pixels(points)
            cv2.fillPoly(
                image, [pixels], TOPDOWN_COLOR_BUILDING_FILL, lineType=cv2.LINE_AA)
            cv2.polylines(
                image,
                [pixels],
                True,
                TOPDOWN_COLOR_BUILDING_EDGE,
                1,
                lineType=cv2.LINE_AA)

        for points, bounds in self._road_polylines:
            if not self._bounds_intersect(bounds, visible_bounds):
                continue
            cv2.polylines(
                image,
                [self._points_to_pixels(points)],
                False,
                TOPDOWN_COLOR_LANE_CENTERLINE,
                2,
                lineType=cv2.LINE_AA)
        return image

    def _draw_route(self, image, route_trace):
        if not route_trace:
            return
        route_points = []
        for index, (waypoint, _) in enumerate(route_trace):
            if index % 3 != 0 and index != len(route_trace) - 1:
                continue
            location = waypoint.transform.location
            route_points.append((float(location.x), float(location.y)))
        if len(route_points) >= 2:
            cv2.polylines(
                image,
                [self._points_to_pixels(np.asarray(
                    route_points, dtype=np.float32))],
                False,
                TOPDOWN_COLOR_ROUTE,
                3,
                lineType=cv2.LINE_AA)

    def _draw_vehicle(
            self,
            image,
            actor,
            actor_transform,
            color,
            ego=False,
            footprint=None):
        if footprint is None:
            footprint = get_actor_footprint_points(actor, actor_transform)
        footprint_pixels = self._points_to_pixels(footprint)
        if len(footprint_pixels) >= 3:
            cv2.fillPoly(
                image, [footprint_pixels], color, lineType=cv2.LINE_AA)
            cv2.polylines(
                image,
                [footprint_pixels],
                True,
                (235, 240, 247) if ego else color,
                2 if ego else 1,
                lineType=cv2.LINE_AA)

        center = self._world_to_pixel(actor_transform.location)
        cv2.circle(image, center, 3 if ego else 2, color, -1, lineType=cv2.LINE_AA)
        forward_vector = actor_transform.get_forward_vector()
        heading_length = max(1.5, float(actor.bounding_box.extent.x) * 2.0)
        heading_location = carla.Location(
            x=actor_transform.location.x + forward_vector.x * heading_length,
            y=actor_transform.location.y + forward_vector.y * heading_length,
            z=actor_transform.location.z)
        cv2.line(
            image,
            center,
            self._world_to_pixel(heading_location),
            (235, 240, 247) if ego else color,
            2 if ego else 1,
            lineType=cv2.LINE_AA)

    def _draw_pedestrian(self, image, actor_transform, color, ego=False):
        center = self._world_to_pixel(actor_transform.location)
        cv2.circle(image, center, 5 if ego else 4, (18, 23, 30), -1)
        cv2.circle(
            image,
            center,
            4 if ego else 3,
            color,
            -1,
            lineType=cv2.LINE_AA)

    def _draw_live_actors(self, image, carla_world, hero_actor, hero_transform):
        visible_vehicle_count = 0
        visible_pedestrian_count = 0
        hero_id = int(hero_actor.id)
        try:
            actors = carla_world.get_actors()
            vehicles = actors.filter('vehicle.*')
            pedestrians = actors.filter('walker.pedestrian.*')
        except RuntimeError:
            vehicles = []
            pedestrians = []

        for vehicle in vehicles:
            if int(vehicle.id) == hero_id:
                continue
            try:
                actor_transform = vehicle.get_transform()
                footprint = self._vehicle_footprint_in_view(
                    vehicle, actor_transform)
                if footprint is None:
                    continue
                self._draw_vehicle(
                    image,
                    vehicle,
                    actor_transform,
                    TOPDOWN_COLOR_VEHICLE,
                    footprint=footprint)
                visible_vehicle_count += 1
            except (AttributeError, RuntimeError):
                continue

        for pedestrian in pedestrians:
            if int(pedestrian.id) == hero_id:
                continue
            try:
                actor_transform = pedestrian.get_transform()
                if not self._location_is_visible(actor_transform.location):
                    continue
                self._draw_pedestrian(
                    image, actor_transform, TOPDOWN_COLOR_PEDESTRIAN)
                visible_pedestrian_count += 1
            except (AttributeError, RuntimeError):
                continue

        if hero_actor.type_id.startswith('walker.pedestrian.'):
            self._draw_pedestrian(
                image, hero_transform, TOPDOWN_COLOR_EGO, ego=True)
            visible_pedestrian_count += 1
        else:
            self._draw_vehicle(
                image,
                hero_actor,
                hero_transform,
                TOPDOWN_COLOR_EGO,
                ego=True)
            visible_vehicle_count += 1
        return visible_vehicle_count, visible_pedestrian_count

    def _draw_status(
            self,
            frame,
            ego_location,
            visible_vehicle_count,
            visible_pedestrian_count):
        title = 'Ego-following top-down map | radius {:.1f} m'.format(
            self._zoom_radius_m)
        coordinates = 'ego x={:.2f}  y={:.2f} | +X right, +Y down'.format(
            float(ego_location.x), float(ego_location.y))
        cv2.putText(
            frame,
            title,
            (self._plot_left, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (235, 240, 247),
            1,
            cv2.LINE_AA)
        cv2.putText(
            frame,
            coordinates,
            (self._plot_left, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 170, 180),
            1,
            cv2.LINE_AA)

        legend_y = self._height - 12
        legend_entries = (
            ('EGO', TOPDOWN_COLOR_EGO),
            (
                'ALL VEHICLES {}'.format(visible_vehicle_count),
                TOPDOWN_COLOR_VEHICLE,
            ),
            (
                'ALL PEDESTRIANS {}'.format(visible_pedestrian_count),
                TOPDOWN_COLOR_PEDESTRIAN,
            ),
        )
        x_coord = self._plot_left
        for label, color in legend_entries:
            cv2.circle(frame, (x_coord + 5, legend_y - 4), 5, color, -1)
            cv2.putText(
                frame,
                label,
                (x_coord + 15, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (215, 220, 227),
                1,
                cv2.LINE_AA)
            x_coord += 38 + (len(label) * 8)

    def render(
            self,
            carla_world,
            hero_actor,
            route_trace=None,
            destination_transform=None):
        if not self.ready or hero_actor is None:
            return

        now_ms = pygame.time.get_ticks()
        if (
                self._last_refresh_ms is not None
                and now_ms - self._last_refresh_ms < self._refresh_period_ms):
            return
        self._last_refresh_ms = now_ms

        try:
            hero_transform = hero_actor.get_transform()
        except RuntimeError:
            return
        self._center_x = float(hero_transform.location.x)
        self._center_y = float(hero_transform.location.y)

        plot_image = self._draw_static_map()
        self._draw_route(plot_image, route_trace)
        if (
                destination_transform is not None
                and self._location_is_visible(destination_transform.location)):
            cv2.circle(
                plot_image,
                self._world_to_pixel(destination_transform.location),
                8,
                TOPDOWN_COLOR_DESTINATION,
                2,
                lineType=cv2.LINE_AA)

        vehicle_count, pedestrian_count = self._draw_live_actors(
            plot_image, carla_world, hero_actor, hero_transform)
        frame = np.full(
            (self._height, self._width, 3),
            TOPDOWN_COLOR_BACKGROUND,
            dtype=np.uint8)
        plot_bottom = self._plot_top + self._plot_size
        plot_right = self._plot_left + self._plot_size
        frame[
            self._plot_top:plot_bottom,
            self._plot_left:plot_right,
        ] = plot_image
        cv2.rectangle(
            frame,
            (self._plot_left, self._plot_top),
            (plot_right - 1, plot_bottom - 1),
            (75, 82, 92),
            1)
        self._draw_status(
            frame,
            hero_transform.location,
            vehicle_count,
            pedestrian_count)

        try:
            if not self._window_created:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name, self._width, self._height)
                self._window_created = True
            cv2.imshow(self._window_name, frame)
            cv2.waitKey(1)
        except cv2.error:
            self.close()
            self.ready = False

    def close(self):
        self._last_refresh_ms = None
        if cv2 is None or not self._window_created:
            return
        try:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._window_created = False


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, traffic_manager, args):
        self.world = carla_world
        self.sync = args.sync
        self.traffic_manager = traffic_manager
        self.actor_role_name = args.rolename
        self.geofence_center = carla.Location(x=args.geofence_x, y=args.geofence_y, z=0.0)
        self.geofence_radius = args.geofence_radius
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.ego_spawn_transform = resolve_ego_spawn_transform(
            self.map,
            args.ego_spawn_x,
            args.ego_spawn_y)
        logging.info(
            'Configured ego spawn x=%.3f y=%.3f resolved z=%.3f yaw=%.2f',
            self.ego_spawn_transform.location.x,
            self.ego_spawn_transform.location.y,
            self.ego_spawn_transform.location.z,
            self.ego_spawn_transform.rotation.yaw)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._vehicle_blueprint_id = args.vehicle_blueprint
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self._route_sampling_resolution = args.route_sampling_resolution
        self._route_min_distance = args.route_min_distance
        self._route_arrival_threshold = args.route_arrival_threshold
        self._route_autonomy_refresh_interval_ms = 500
        self._next_route_refresh_at_ms = 0
        self._traffic_manager_path_spacing = max(6.0, self._route_sampling_resolution * 4.0)
        self._route_planner = None
        self.route_loop_active = False
        self.route_loop_autonomous = False
        self.route_origin_transform = None
        self.route_destination_transform = None
        self.route_trace = []
        self.route_path = []
        self.traffic_manager_route_path = []
        self.last_spawn_transform = None
        self.show_actor_bboxes = False
        self.show_topdown_map = False
        self.topdown_zoom_radius = args.topdown_zoom_radius
        self.topdown_renderer = None
        if args.destination_x is not None and args.destination_y is not None:
            destination_z = args.destination_z if args.destination_z is not None else 0.0
            self.route_destination_override = carla.Location(
                x=args.destination_x,
                y=args.destination_y,
                z=destination_z)
        else:
            self.route_destination_override = None
        if not self.respawn_at_ego_start(notify=False):
            target = self.ego_spawn_transform.location
            raise RuntimeError(
                'Configured ego spawn at x={:.2f}, y={:.2f} is occupied'.format(
                    target.x,
                    target.y))
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def _select_ego_blueprint(self):
        """Return the requested exact blueprint or a legacy random match."""
        if self._vehicle_blueprint_id is not None:
            matches = [
                blueprint
                for blueprint in self.world.get_blueprint_library().filter(
                    self._vehicle_blueprint_id)
                if blueprint.id == self._vehicle_blueprint_id
            ]
            if not matches:
                raise ValueError(
                    "Vehicle blueprint '{}' is unavailable in this CARLA "
                    "server. Use an exact vehicle.* blueprint ID.".format(
                        self._vehicle_blueprint_id))
            return matches[0]

        blueprint_list = get_actor_blueprints(
            self.world,
            self._actor_filter,
            self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        return random.choice(blueprint_list)

    def restart(self, spawn_transform=None, allow_random_fallback=True):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else DEFAULT_CAMERA_TRANSFORM_INDEX)
        requested_spawn = copy_transform(spawn_transform) if spawn_transform is not None else None
        blueprint = self._select_ego_blueprint()
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            if requested_spawn is not None:
                # Explicit configured/route transforms are vehicle-safe and
                # must remain authoritative across repeated respawns.
                spawn_point = copy_transform(requested_spawn)
            else:
                # Preserve the legacy Backspace behavior when restarting at
                # the current pose by adding temporary vertical clearance.
                spawn_point = copy_transform(
                    self.player.get_transform(),
                    z_offset=2.0)
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE5 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            if requested_spawn is not None:
                spawn_point = copy_transform(requested_spawn)
                requested_spawn = None
            elif not allow_random_fallback:
                raise RuntimeError(
                    'Required ego spawn at x={:.2f}, y={:.2f} is unavailable '
                    'or occupied'.format(
                        spawn_transform.location.x,
                        spawn_transform.location.y))
            else:
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.traffic_manager.update_vehicle_lights(self.player, True)
        self.last_spawn_transform = copy_transform(self.player.get_transform())

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _ego_spawn_blocking_actor(self):
        """Return an actor that would block the configured ego spawn."""
        player_id = getattr(self.player, 'id', None)
        try:
            actors = self.world.get_actors()
        except RuntimeError:
            return None
        target = self.ego_spawn_transform.location
        for actor in actors:
            try:
                if actor.id == player_id:
                    continue
                if not (
                        actor.type_id.startswith('vehicle.')
                        or actor.type_id.startswith('walker.')):
                    continue
                location = actor.get_location()
                if math.hypot(
                        location.x - target.x,
                        location.y - target.y) < EGO_SPAWN_OCCUPANCY_RADIUS_M:
                    return actor
            except (AttributeError, RuntimeError):
                continue
        return None

    def respawn_at_ego_start(self, notify=True):
        """Respawn the ego at its configured start without random fallback."""
        blocking_actor = self._ego_spawn_blocking_actor()
        if blocking_actor is not None:
            message = 'Configured ego spawn blocked by actor id={}'.format(
                blocking_actor.id)
            logging.warning(message)
            if notify:
                self.hud.notification(message, seconds=3.0)
            return False

        self.restart(
            spawn_transform=self.ego_spawn_transform,
            allow_random_fallback=False)
        actual_location = self.player.get_location()
        target_location = self.ego_spawn_transform.location
        position_error = math.hypot(
            actual_location.x - target_location.x,
            actual_location.y - target_location.y)
        if position_error > EGO_SPAWN_POSITION_TOLERANCE_M:
            raise RuntimeError(
                'Ego spawned {:.2f} m from required x={:.2f}, y={:.2f}'.format(
                    position_error,
                    target_location.x,
                    target_location.y))
        logging.info(
            'Ego spawned at x=%.3f y=%.3f z=%.3f yaw=%.2f',
            actual_location.x,
            actual_location.y,
            actual_location.z,
            self.player.get_transform().rotation.yaw)
        if notify:
            self.hud.notification(
                'Ego respawned at (%.2f, %.2f)' % (
                    target_location.x,
                    target_location.y),
                seconds=3.0)
        return True

    def set_vehicle_autopilot(self, enabled):
        if isinstance(self.player, carla.Vehicle):
            self.player.set_autopilot(enabled, self.traffic_manager.get_port())

    def _get_route_planner(self):
        if GlobalRoutePlanner is None:
            return None
        if self._route_planner is None:
            self._route_planner = GlobalRoutePlanner(self.map, self._route_sampling_resolution)
        return self._route_planner

    def _resolve_route_destination_transform(self, origin_transform):
        if self.route_destination_override is not None:
            destination_waypoint = self.map.get_waypoint(
                self.route_destination_override,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            return copy_transform(destination_waypoint.transform) if destination_waypoint is not None else None

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            return None

        far_spawn_points = [
            spawn_point for spawn_point in spawn_points
            if spawn_point.location.distance(origin_transform.location) >= self._route_min_distance
        ]
        candidate_points = far_spawn_points if far_spawn_points else spawn_points
        destination_transform = max(
            candidate_points,
            key=lambda spawn_point: spawn_point.location.distance(origin_transform.location))
        return copy_transform(destination_transform)

    def _build_route_from_location(self, start_location):
        if self.route_destination_transform is None:
            return False

        route_planner = self._get_route_planner()
        if route_planner is None:
            return False

        start_waypoint = self.map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        destination_waypoint = self.map.get_waypoint(
            self.route_destination_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if start_waypoint is None or destination_waypoint is None:
            return False

        route_trace = route_planner.trace_route(
            start_waypoint.transform.location,
            destination_waypoint.transform.location)
        if not route_trace:
            return False

        self.route_trace = route_trace
        self.route_path = [copy_location(waypoint.transform.location) for waypoint, _ in route_trace]
        self.traffic_manager_route_path = self._build_traffic_manager_path(start_location)
        return True

    def _build_traffic_manager_path(self, start_location):
        if not self.route_path:
            return []

        reference_location = copy_location(start_location)
        traffic_manager_path = []
        for waypoint_location in self.route_path:
            if reference_location.distance(waypoint_location) >= self._traffic_manager_path_spacing:
                traffic_manager_path.append(copy_location(waypoint_location))
                reference_location = waypoint_location

        destination_location = (
            copy_location(self.route_destination_transform.location)
            if self.route_destination_transform is not None else copy_location(self.route_path[-1]))
        if not traffic_manager_path or traffic_manager_path[-1].distance(destination_location) > 1.0:
            traffic_manager_path.append(destination_location)
        return traffic_manager_path

    def _apply_traffic_manager_route(self):
        if not isinstance(self.player, carla.Vehicle):
            return False
        if not self.traffic_manager_route_path:
            return False

        self.set_vehicle_autopilot(True)
        try:
            self.traffic_manager.auto_lane_change(self.player, False)
        except Exception:
            pass
        self.traffic_manager.set_path(self.player, list(self.traffic_manager_route_path))
        self._next_route_refresh_at_ms = pygame.time.get_ticks() + self._route_autonomy_refresh_interval_ms
        return True

    def _refresh_route_autonomy(self):
        if not self.route_loop_autonomous or self.player is None or self.route_destination_transform is None:
            return
        current_ticks = pygame.time.get_ticks()
        if current_ticks < self._next_route_refresh_at_ms:
            return
        if not self._build_route_from_location(self.player.get_location()):
            self._next_route_refresh_at_ms = current_ticks + self._route_autonomy_refresh_interval_ms
            return
        self._apply_traffic_manager_route()

    def enable_route_loop(self, autonomous_enabled=True):
        if not isinstance(self.player, carla.Vehicle):
            self.hud.notification('Route loop requires a vehicle actor')
            return False
        if self._get_route_planner() is None:
            self.hud.notification('Route planner is unavailable in this CARLA install')
            return False

        route_origin = copy_transform(self.last_spawn_transform) if self.last_spawn_transform is not None else copy_transform(self.player.get_transform())
        route_destination = self._resolve_route_destination_transform(route_origin)
        if route_destination is None:
            self.hud.notification('No valid route destination could be resolved')
            return False

        self.route_loop_active = True
        self.route_loop_autonomous = False
        self.route_origin_transform = route_origin
        self.route_destination_transform = route_destination
        self.route_trace = []
        self.route_path = []
        self.traffic_manager_route_path = []
        self.set_vehicle_autopilot(False)

        if self.player.get_location().distance(route_origin.location) > 3.0:
            self.restart(spawn_transform=route_origin)

        if not self._build_route_from_location(self.player.get_location()):
            self.disable_route_loop()
            self.hud.notification('Unable to build a route to the destination')
            return False

        if autonomous_enabled and not self.set_route_loop_autonomous(True):
            self.disable_route_loop()
            self.hud.notification('Unable to start route autonomy')
            return False

        if not autonomous_enabled:
            self.hud.notification('Route loop active: manual drive with highlighted waypoints')

        self.hud.notification(
            'Route destination: (%.1f, %.1f)' % (
                self.route_destination_transform.location.x,
                self.route_destination_transform.location.y),
            seconds=4.0)
        return True

    def disable_route_loop(self):
        self.route_loop_active = False
        self.route_loop_autonomous = False
        self.route_origin_transform = None
        self.route_destination_transform = None
        self.route_trace = []
        self.route_path = []
        self.traffic_manager_route_path = []
        self._next_route_refresh_at_ms = 0
        if isinstance(self.player, carla.Vehicle):
            try:
                self.traffic_manager.auto_lane_change(self.player, True)
            except Exception:
                pass
        self.set_vehicle_autopilot(False)

    def set_route_loop_autonomous(self, enabled):
        if not self.route_loop_active:
            self.set_vehicle_autopilot(False)
            return False

        self.route_loop_autonomous = enabled
        if not enabled:
            self._next_route_refresh_at_ms = 0
            self.set_vehicle_autopilot(False)
            return True

        if not self._build_route_from_location(self.player.get_location()):
            self.route_loop_autonomous = False
            self.set_vehicle_autopilot(False)
            return False

        if not self._apply_traffic_manager_route():
            self.route_loop_autonomous = False
            self.set_vehicle_autopilot(False)
            return False
        return True

    def sync_route_loop_after_respawn(self, autonomous_enabled):
        if not self.route_loop_active or self.route_destination_transform is None:
            return

        if autonomous_enabled:
            if not self.set_route_loop_autonomous(True):
                self.hud.notification('Unable to resume route autonomy after respawn')
        else:
            self.route_loop_autonomous = False
            self.set_vehicle_autopilot(False)
            if not self._build_route_from_location(self.player.get_location()):
                self.hud.notification('Unable to rebuild highlighted waypoints after respawn')

    def _handle_route_arrival(self):
        if self.route_origin_transform is None:
            return

        autonomous_mode = self.route_loop_autonomous
        self.restart(spawn_transform=self.route_origin_transform)
        self.sync_route_loop_after_respawn(autonomous_mode)
        self.hud.notification(
            'Destination reached. Respawned at route origin in %s mode' % (
                'autonomous' if autonomous_mode else 'manual'),
            seconds=3.0)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def toggle_actor_visualizations(self):
        self.show_actor_bboxes = not self.show_actor_bboxes
        if self.show_actor_bboxes:
            if self.topdown_renderer is not None and not self.topdown_renderer.ready:
                self.topdown_renderer.close()
                self.topdown_renderer = None
            if self.topdown_renderer is None and cv2 is not None:
                try:
                    self.topdown_renderer = TopDownMapRenderer(
                        self.world,
                        self.map,
                        self.topdown_zoom_radius)
                except Exception as exc:
                    logging.warning('Unable to initialize top-down map: %s', exc)
                    self.topdown_renderer = None
            self.show_topdown_map = self.topdown_renderer is not None and self.topdown_renderer.ready
            if self.show_topdown_map:
                self.hud.notification('Actor boxes and top-down map On')
            elif cv2 is None:
                self.hud.notification('Actor boxes On (top-down map unavailable: OpenCV missing)')
            else:
                self.hud.notification('Actor boxes On (top-down map unavailable)')
        else:
            self.show_topdown_map = False
            if self.topdown_renderer is not None:
                self.topdown_renderer.close()
            self.hud.notification('Actor boxes and top-down map Off')

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        if self.geofence_radius > 0.0:
            draw_geofence(self.world, self.geofence_center, self.geofence_radius)
        if self.route_loop_active:
            if self.player is not None and self.route_destination_transform is not None:
                destination_distance = self.player.get_location().distance(self.route_destination_transform.location)
                if destination_distance <= self._route_arrival_threshold:
                    self._handle_route_arrival()
                elif self.route_loop_autonomous:
                    self._refresh_route_autonomy()
            draw_route_waypoints(
                self.world,
                self.route_trace,
                origin_transform=self.route_origin_transform,
                destination_transform=self.route_destination_transform)
        if self.camera_manager is not None:
            self.camera_manager.sync_head_pose_to_vehicle()
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        if self.show_topdown_map and self.topdown_renderer is not None and self.player is not None:
            self.topdown_renderer.render(
                self.world,
                self.player,
                route_trace=self.route_trace if self.route_loop_active else None,
                destination_transform=self.route_destination_transform)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self, close_visualizers=False):
        if self.radar_sensor is not None:
            self.toggle_radar()
        if close_visualizers and self.topdown_renderer is not None:
            self.topdown_renderer.close()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.set_vehicle_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def _reset_control_for_new_player(self, world):
        """Clear stale input state after replacing the controlled actor."""
        self._steer_cache = 0.0
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation

    def _respawn_ego_at_start(self, world):
        """Respawn at the configured start and restore route/autopilot state."""
        autonomous_mode = self._autopilot_enabled
        if autonomous_mode:
            world.set_vehicle_autopilot(False)

        if not world.respawn_at_ego_start():
            if autonomous_mode:
                world.set_vehicle_autopilot(True)
            return

        self._reset_control_for_new_player(world)
        world.constant_velocity_enabled = False
        world.doors_are_open = False
        if world.route_loop_active:
            world.sync_route_loop_after_respawn(autonomous_mode)
        elif autonomous_mode:
            world.set_vehicle_autopilot(True)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.set_vehicle_autopilot(False)
                        world.restart()
                        if world.route_loop_active:
                            world.sync_route_loop_after_respawn(True)
                        else:
                            world.set_vehicle_autopilot(True)
                    else:
                        world.restart()
                        if world.route_loop_active:
                            world.sync_route_loop_after_respawn(False)
                elif event.key == K_y:
                    self._respawn_ego_at_start(world)
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_KP5 or event.key == K_INSERT:
                    world.camera_manager.reset_head_pose()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.set_vehicle_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_j:
                        if world.route_loop_active:
                            world.disable_route_loop()
                            self._autopilot_enabled = False
                            world.hud.notification('Route loop Off')
                        else:
                            if world.enable_route_loop(autonomous_enabled=True):
                                self._autopilot_enabled = True
                                world.hud.notification('Route loop On (autonomous)')
                    elif event.key == K_u:
                        world.toggle_actor_visualizations()
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        if world.route_loop_active:
                            if self._autopilot_enabled:
                                if world.set_route_loop_autonomous(True):
                                    world.hud.notification('Route autonomy On')
                                else:
                                    self._autopilot_enabled = False
                                    world.hud.notification('Unable to enable route autonomy')
                            else:
                                world.set_route_loop_autonomous(False)
                                world.hud.notification('Route autonomy Off')
                        else:
                            if self._autopilot_enabled and not sync_mode:
                                print("WARNING: You are currently in asynchronous mode and could "
                                      "experience some issues with the traffic simulation")
                            world.set_vehicle_autopilot(self._autopilot_enabled)
                            world.hud.notification(
                                'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        keys = pygame.key.get_pressed()
        world.camera_manager.update_head_pose_from_keys(
            keys,
            clock.get_time(),
            fast=bool(pygame.key.get_mods() & KMOD_SHIFT))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(keys, clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    world.player.set_light_state(carla.VehicleLightState(current_lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(keys, clock.get_time(), world)
                world.player.apply_control(self._control)

        self._lights = current_lights

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_s]:
            self._control.speed = 0.0
        if keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        # F1 remains available to show the full HUD when it is needed.
        self._show_info = False
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, world_wrapper):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self._world_wrapper = world_wrapper
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = DEFAULT_CAMERA_TRANSFORM_INDEX
        self.head_yaw_offset = 0.0
        self.head_pitch_offset = 0.0
        self._head_yaw_limit = 80.0
        self._head_pitch_min = -45.0
        self._head_pitch_max = 45.0
        self._head_look_rate = 55.0
        self._head_look_fast_multiplier = 3.0
        self._head_notice_interval_ms = 350
        self._next_head_notice_at_ms = 0
        self._head_pose_tracking_active = False
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.lidar.ray_cast_semantic', None, 'Semantic Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def _get_active_sensor_relative_transform(self):
        base_transform = self._camera_transforms[self.transform_index][0]
        base_location = base_transform.location
        base_rotation = base_transform.rotation
        return carla.Transform(
            carla.Location(
                x=base_location.x,
                y=base_location.y,
                z=base_location.z),
            carla.Rotation(
                pitch=base_rotation.pitch + self.head_pitch_offset,
                yaw=base_rotation.yaw + self.head_yaw_offset,
                roll=base_rotation.roll))

    def _get_active_sensor_world_transform(self):
        relative_transform = self._get_active_sensor_relative_transform()
        parent_transform = self._parent.get_transform()
        parent_matrix = np.array(parent_transform.get_matrix(), dtype=np.float32)
        relative_location = relative_transform.location
        relative_point = np.array(
            [relative_location.x, relative_location.y, relative_location.z, 1.0],
            dtype=np.float32)
        world_point = parent_matrix @ relative_point
        world_rotation_matrix = (
            rotation_matrix_from_carla_rotation(parent_transform.rotation) @
            rotation_matrix_from_carla_rotation(relative_transform.rotation))
        return carla.Transform(
            carla.Location(
                x=float(world_point[0]),
                y=float(world_point[1]),
                z=float(world_point[2])),
            carla_rotation_from_matrix(world_rotation_matrix))

    def sync_head_pose_to_vehicle(self):
        if self._head_pose_tracking_active and self.sensor is not None:
            self.sensor.set_transform(self._get_active_sensor_world_transform())

    def _notify_head_pose(self, force=False):
        current_ticks = pygame.time.get_ticks()
        if not force and current_ticks < self._next_head_notice_at_ms:
            return
        self._next_head_notice_at_ms = current_ticks + self._head_notice_interval_ms
        self.hud.notification(
            'Sensor view yaw %+0.1f pitch %+0.1f' % (
                self.head_yaw_offset,
                self.head_pitch_offset),
            seconds=0.45)

    def reset_head_pose(self):
        if (
                abs(self.head_yaw_offset) < 1e-3 and
                abs(self.head_pitch_offset) < 1e-3 and
                not self._head_pose_tracking_active):
            return
        self.head_yaw_offset = 0.0
        self.head_pitch_offset = 0.0
        self._head_pose_tracking_active = False
        self.set_sensor(self.index, notify=False, force_respawn=True)
        self._notify_head_pose(force=True)

    def update_head_pose_from_keys(self, keys, milliseconds, fast=False):
        yaw_direction = 0
        if keys[K_LEFT] or keys[K_KP4] or keys[K_HOME]:
            yaw_direction -= 1
        if keys[K_RIGHT] or keys[K_KP6] or keys[K_END]:
            yaw_direction += 1

        pitch_direction = 0
        if keys[K_UP] or keys[K_KP8] or keys[K_PAGEUP]:
            pitch_direction += 1
        if keys[K_DOWN] or keys[K_KP2] or keys[K_PAGEDOWN]:
            pitch_direction -= 1

        if yaw_direction == 0 and pitch_direction == 0:
            return

        multiplier = self._head_look_fast_multiplier if fast else 1.0
        delta_degrees = self._head_look_rate * multiplier * milliseconds * 1e-3
        old_yaw = self.head_yaw_offset
        old_pitch = self.head_pitch_offset
        self.head_yaw_offset = min(
            self._head_yaw_limit,
            max(-self._head_yaw_limit, self.head_yaw_offset + yaw_direction * delta_degrees))
        self.head_pitch_offset = min(
            self._head_pitch_max,
            max(self._head_pitch_min, self.head_pitch_offset + pitch_direction * delta_degrees))

        if abs(self.head_yaw_offset - old_yaw) > 1e-3 or abs(self.head_pitch_offset - old_pitch) > 1e-3:
            self._head_pose_tracking_active = True
            self.sync_head_pose_to_vehicle()
            self._notify_head_pose()

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._get_active_sensor_relative_transform(),
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            rendered_surface = self.surface
            should_draw_route = self._should_draw_route_overlay()
            if self._world_wrapper.show_actor_bboxes or should_draw_route:
                rendered_surface = self.surface.copy()
                if self._world_wrapper.show_actor_bboxes:
                    self._draw_actor_bboxes(rendered_surface)
                if should_draw_route:
                    self._draw_route_overlay(rendered_surface)
            display.blit(rendered_surface, (0, 0))

    def _is_camera_sensor(self):
        return (
            self.sensor is not None and
            self.index is not None and
            self.sensors[self.index][0].startswith('sensor.camera'))

    def _should_draw_route_overlay(self):
        return (
            self._is_camera_sensor() and
            self._world_wrapper.route_loop_active and
            bool(self._world_wrapper.route_trace))

    def _route_locations_ahead(self, max_distance=65.0):
        route_locations = self._world_wrapper.route_path
        if not route_locations:
            route_locations = [waypoint.transform.location for waypoint, _ in self._world_wrapper.route_trace]
        if len(route_locations) < 2:
            return []

        ego_transform = self._parent.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        front_offset = self._parent.bounding_box.extent.x + 0.75
        front_location = ego_location + carla.Location(
            x=ego_forward.x * front_offset,
            y=ego_forward.y * front_offset,
            z=0.0)

        def _forward_dot(location):
            return (
                (location.x - ego_location.x) * ego_forward.x +
                (location.y - ego_location.y) * ego_forward.y)

        def _project_to_segment(start_location, end_location, target_location):
            delta_x = end_location.x - start_location.x
            delta_y = end_location.y - start_location.y
            delta_z = end_location.z - start_location.z
            segment_length_sq = delta_x * delta_x + delta_y * delta_y
            if segment_length_sq < 1e-6:
                return copy_location(start_location)
            ratio = (
                ((target_location.x - start_location.x) * delta_x +
                 (target_location.y - start_location.y) * delta_y) /
                segment_length_sq)
            ratio = max(0.0, min(1.0, ratio))
            return carla.Location(
                x=start_location.x + delta_x * ratio,
                y=start_location.y + delta_y * ratio,
                z=start_location.z + delta_z * ratio)

        best_segment_index = 0
        best_start_location = copy_location(route_locations[0])
        best_distance = float('inf')
        saw_forward_segment = False
        for index in range(len(route_locations) - 1):
            projected_location = _project_to_segment(
                route_locations[index],
                route_locations[index + 1],
                front_location)
            is_forward = _forward_dot(projected_location) >= -4.0
            if saw_forward_segment and not is_forward:
                continue
            distance = projected_location.distance(front_location)
            if is_forward and not saw_forward_segment:
                saw_forward_segment = True
                best_distance = float('inf')
            if distance < best_distance:
                best_segment_index = index
                best_start_location = projected_location
                best_distance = distance

        selected_locations = [best_start_location]
        travelled = 0.0
        previous_location = best_start_location

        for location in route_locations[best_segment_index + 1:]:
            segment_length = previous_location.distance(location)
            if segment_length < 0.01:
                previous_location = location
                continue

            if travelled + segment_length >= max_distance:
                remaining = max(0.0, max_distance - travelled)
                ratio = remaining / segment_length
                selected_locations.append(carla.Location(
                    x=previous_location.x + (location.x - previous_location.x) * ratio,
                    y=previous_location.y + (location.y - previous_location.y) * ratio,
                    z=previous_location.z + (location.z - previous_location.z) * ratio))
                break

            selected_locations.append(copy_location(location))
            travelled += segment_length
            previous_location = location

        return selected_locations if len(selected_locations) >= 2 else []

    def _route_segments_for_camera(self, surface):
        width = surface.get_width()
        height = surface.get_height()
        field_of_view = float(self.sensor.attributes.get('fov', 90.0))
        calibration = get_camera_K(width, height, field_of_view)
        camera_transform = self.sensor.get_transform()
        locations = [
            location + carla.Location(z=0.08)
            for location in self._route_locations_ahead(max_distance=65.0)]
        if not locations:
            return []

        points_world = np.array(
            [[location.x, location.y, location.z] for location in locations],
            dtype=np.float32)
        points_camera = world_to_camera(points_world, camera_transform)
        x_values = points_camera[:, 0]
        y_values = points_camera[:, 1]
        z_values = points_camera[:, 2]

        u_values = calibration[0, 2] + (y_values / np.maximum(x_values, 1e-3)) * calibration[0, 0]
        v_values = calibration[1, 2] - (z_values / np.maximum(x_values, 1e-3)) * calibration[1, 1]
        extended_margin = 120

        segments = []
        segment = []
        for x_value, u_value, v_value in zip(x_values, u_values, v_values):
            is_visible = (
                x_value > 0.25 and
                -extended_margin <= u_value <= width + extended_margin and
                -extended_margin <= v_value <= height + extended_margin)
            if is_visible:
                segment.append((int(u_value), int(v_value)))
            elif segment:
                if len(segment) >= 2:
                    segments.append(segment)
                segment = []

        if len(segment) >= 2:
            segments.append(segment)

        return segments

    @staticmethod
    def _draw_arrow_head(surface, segment, color, length, half_width):
        if len(segment) < 2:
            return

        tip = np.array(segment[-1], dtype=np.float32)
        tail = None
        for candidate in reversed(segment[:-1]):
            candidate = np.array(candidate, dtype=np.float32)
            if np.linalg.norm(tip - candidate) >= 2.0:
                tail = candidate
                break
        if tail is None:
            return

        direction = tip - tail
        direction_length = np.linalg.norm(direction)
        if direction_length < 1.0:
            return

        unit_direction = direction / direction_length
        perpendicular = np.array([-unit_direction[1], unit_direction[0]], dtype=np.float32)
        base = tip - unit_direction * length
        points = [
            (int(tip[0]), int(tip[1])),
            (int(base[0] + perpendicular[0] * half_width), int(base[1] + perpendicular[1] * half_width)),
            (int(base[0] - perpendicular[0] * half_width), int(base[1] - perpendicular[1] * half_width)),
        ]
        pygame.draw.polygon(surface, color, points)

    def _draw_route_overlay(self, surface):
        route_segments = self._route_segments_for_camera(surface)
        if not route_segments:
            return

        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        width = surface.get_width()
        line_width = max(8, int(width / 150))
        outline_width = line_width + 5
        arrow_length = max(26, line_width * 4)
        arrow_half_width = max(15, line_width * 2)
        route_outline = (35, 0, 0, 220)
        route_red = (185, 12, 12, 238)
        route_dot = (230, 24, 24, 235)

        last_visible_segment = None
        for segment in route_segments:
            pygame.draw.lines(overlay, route_outline, False, segment, outline_width)
            pygame.draw.lines(overlay, route_red, False, segment, line_width)
            for point in segment[::10]:
                pygame.draw.circle(overlay, route_dot, point, max(3, int(line_width * 0.45)))
            last_visible_segment = segment

        if last_visible_segment is not None:
            self._draw_arrow_head(
                overlay,
                last_visible_segment,
                route_outline,
                arrow_length + 5,
                arrow_half_width + 4)
            self._draw_arrow_head(
                overlay,
                last_visible_segment,
                route_red,
                arrow_length,
                arrow_half_width)

        surface.blit(overlay, (0, 0))

    def _draw_actor_bboxes(self, surface):
        if not self._is_camera_sensor():
            return

        width = surface.get_width()
        height = surface.get_height()
        field_of_view = float(self.sensor.attributes.get('fov', 90.0))
        calibration = get_camera_K(width, height, field_of_view)
        camera_transform = self.sensor.get_transform()
        ego_location = self._parent.get_location()
        world_actors = self._parent.get_world().get_actors()
        actor_specs = (
            ('vehicle.*', (64, 160, 255)),
            ('walker.pedestrian.*', (0, 255, 0)),
        )

        for pattern, color in actor_specs:
            for actor in world_actors.filter(pattern):
                if actor.id == self._parent.id:
                    continue
                if actor.get_location().distance(ego_location) > 90.0:
                    continue
                bounding_box = project_bbox_corners_to_2d(
                    actor.get_transform(),
                    actor.bounding_box,
                    camera_transform,
                    calibration,
                    width,
                    height)
                if bounding_box is None:
                    continue
                x1, y1, x2, y2 = bounding_box
                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    2)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0] == 'sensor.lidar.ray_cast':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0] == 'sensor.lidar.ray_cast_semantic':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 6), 6))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            for i in range(len(image)):
                point = lidar_data[i]
                lidar_tag = image[i].object_tag
                lidar_img[tuple(point.T)] = OBJECT_TO_COLOR[int(lidar_tag)]
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        traffic_manager = client.get_trafficmanager()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA Manual Control AR v7')
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, traffic_manager, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy(close_visualizers=True)

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--ego-spawn-x', metavar='X', default=None, type=finite_float,
        help=(
            'ego startup/respawn X coordinate; requires --ego-spawn-y '
            'and must be near a driving lane (default: {:.2f})'.format(
                DEFAULT_EGO_SPAWN_X)))
    argparser.add_argument(
        '--ego-spawn-y', metavar='Y', default=None, type=finite_float,
        help=(
            'ego startup/respawn Y coordinate; requires --ego-spawn-x '
            '(default: {:.2f})'.format(DEFAULT_EGO_SPAWN_Y)))
    argparser.add_argument(
        '--topdown-zoom-radius', metavar='METERS',
        default=DEFAULT_TOPDOWN_ZOOM_RADIUS_M, type=topdown_zoom_radius,
        help=(
            'ego-centered top-down map half-width/half-height in meters '
            '(range: 1-10000; default: %(default)s)'))
    argparser.add_argument(
        '--vehicle-blueprint', metavar='ID', default=None,
        type=vehicle_blueprint_id,
        help=(
            'exact ego vehicle blueprint ID, for example '
            '"vehicle.tesla.model3"; overrides --filter and --generation '
            '(default: randomly select a matching vehicle)'))
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--rolename', metavar='NAME', default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma', default=1.0, type=float,
        help='Gamma correction of the camera (default: 1.0)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--geofence-x', default=0.0, type=float,
        help='X coordinate of the geofenced area center (default: 0.0)')
    argparser.add_argument(
        '--geofence-y', default=0.0, type=float,
        help='Y coordinate of the geofenced area center (default: 0.0)')
    argparser.add_argument(
        '--geofence-radius', default=20.0, type=float,
        help='Radius of the geofenced area (default: 20.0)')
    argparser.add_argument(
        '--destination-x', default=None, type=float,
        help='X coordinate of the route destination waypoint (default: auto-select a far waypoint)')
    argparser.add_argument(
        '--destination-y', default=None, type=float,
        help='Y coordinate of the route destination waypoint (default: auto-select a far waypoint)')
    argparser.add_argument(
        '--destination-z', default=None, type=float,
        help='Z coordinate of the route destination waypoint (default: snap to road)')
    argparser.add_argument(
        '--route-min-distance', default=100.0, type=float,
        help='Minimum preferred distance in meters when auto-selecting the destination waypoint (default: 100.0)')
    argparser.add_argument(
        '--route-arrival-threshold', default=6.0, type=float,
        help='Distance in meters used to consider the destination reached (default: 6.0)')
    argparser.add_argument(
        '--route-sampling-resolution', default=2.0, type=float,
        help='Waypoint spacing used by the route planner in meters (default: 2.0)')
    args = argparser.parse_args()

    if (args.destination_x is None) != (args.destination_y is None):
        argparser.error('--destination-x and --destination-y must be provided together')
    if (args.ego_spawn_x is None) != (args.ego_spawn_y is None):
        argparser.error('--ego-spawn-x and --ego-spawn-y must be provided together')
    if args.ego_spawn_x is None:
        args.ego_spawn_x = DEFAULT_EGO_SPAWN_X
        args.ego_spawn_y = DEFAULT_EGO_SPAWN_Y

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
